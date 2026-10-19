"""Command-line entry point.

Everything that is not a web request goes through here: running the services and the
checks and maintenance around them. Installed as ``pitch``::

    uv run pitch --help
    uv run pitch info
    uv run pitch serve
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from pitch_occupancy import __version__
from pitch_occupancy.config import settings

app = typer.Typer(
    name="pitch",
    help="Multi-pitch occupancy and booking verification.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def info() -> None:
    """Show the resolved configuration and whether the expected paths exist."""
    typer.echo(f"pitch-occupancy {__version__}")
    typer.echo(f"  source type   : {settings.source_type.value}")
    typer.echo(f"  sample every  : {settings.sample_interval_s}s")
    typer.echo("")
    for label, path in (
        ("raw footage", settings.raw_dir),
        ("processed", settings.dataset_dir),
        ("results", settings.results_dir),
    ):
        mark = "ok     " if path.exists() else "missing"
        typer.echo(f"  [{mark}] {label:<14} {path}")


@app.command("fetch-weights")
def fetch_weights_cmd(
    keys: list[str] = typer.Argument(None, help="Detector keys; default: all registered."),
) -> None:
    """Download the registered detector weights to the repository root.

    The one deliberate network access the detector path makes. Nothing downloads as a side
    effect of a prediction: a worker that reaches for the network on its first frame at 3 am
    is a worker that fails at 3 am, so a missing file is "not checked" there and a download
    here. Weights are gitignored; run this once per clone.
    """
    from pitch_occupancy.vision.detector import DETECTORS, fetch_weights

    unknown = sorted(set(keys or ()) - set(DETECTORS))
    if unknown:
        raise typer.BadParameter(f"unknown detector(s) {unknown}; known: {sorted(DETECTORS)}")
    for key, path, present in fetch_weights(keys or None):
        size = f"{path.stat().st_size / 1e6:.1f} MB" if present else "MISSING"
        typer.echo(f"  {key:<14} {path.name:<20} {size}")


@app.command("assist-test")
def assist_test_cmd(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, help="A picture of a pitch."),
    provider: str = typer.Option(None, "--provider", help="Ask only this provider."),
) -> None:
    """Ask the free vision model(s) about one picture, to check a key works (docs/assist.md).

    Counts against the free budget like any other question: one request per provider tried.
    """
    import cv2

    from pitch_occupancy.assist import budget, config, providers
    from pitch_occupancy.assist.reviewer import CameraPicture, prepare
    from pitch_occupancy.db.schema import connect, initialise

    frame = cv2.imread(str(image))
    if frame is None:
        raise typer.BadParameter("could not read that picture")
    cfg = config.load()
    chain = [p for p in cfg.chain() if provider in (None, p.name)]
    if not chain:
        raise typer.BadParameter(f"unknown provider; known: {sorted(providers.PROVIDERS)}")
    images = prepare([CameraPicture(frame)], cfg)
    conn = connect(settings.db_path)
    initialise(conn)
    now = datetime.now().astimezone()
    try:
        for p in chain:
            if not p.configured:
                typer.echo(f"  {p.name:<11} no key - set {p.key_env}")
                continue
            ok, why = budget.reserve(conn, p, now)
            if not ok:
                typer.echo(f"  {p.name:<11} skipped: {why}")
                continue
            try:
                o = providers.ask(p, images, people=None, timeout_s=cfg.timeout_s)
            except providers.ProviderError as exc:
                budget.settle(conn, p, now, tokens=0, failed=True)
                typer.secho(f"  {p.name:<11} failed: {exc}", fg=typer.colors.RED)
                continue
            budget.settle(conn, p, now, tokens=o.tokens)
            typer.secho(f"  {p.name:<11} {o.state} ({o.confidence:.2f}), about {o.people} "
                        f"people - {o.reason} [{o.model}, {o.latency_ms} ms]",
                        fg=typer.colors.GREEN)
    finally:
        conn.close()
    if not cfg.enabled:
        typer.echo("configs/assist.json does not turn the second opinion on yet "
                   "(see configs/assist.example.json).")


#: The shortest sign-in password `serve` accepts.
MIN_PASSWORD_LENGTH = 10


@app.command()
def serve(
    host: str = typer.Option(settings.api_host, help="Bind address."),
    port: int = typer.Option(settings.api_port, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
    no_password: bool = typer.Option(
        False, "--no-password",
        help="Allow listening on a network address without PITCH_ACCESS_PASSWORD set."),
) -> None:
    """Serve the dashboard and API with uvicorn.

    Listening beyond this machine without a password is refused unless ``--no-password`` is
    given: the pages can change camera settings and show live pictures of people.
    """
    import uvicorn

    from pitch_occupancy.api.access import LOCAL_HOSTS

    if (host not in LOCAL_HOSTS and not settings.access_password.get_secret_value()
            and not no_password):
        typer.secho(
            f"refusing to listen on {host} with no password: anyone on the network could "
            f"change camera settings and see the cameras. Set PITCH_ACCESS_PASSWORD (in .env "
            f"or the environment), or pass --no-password if this network is trusted.",
            fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    password = settings.access_password.get_secret_value()
    if password and len(password) < MIN_PASSWORD_LENGTH:
        typer.secho(f"PITCH_ACCESS_PASSWORD is shorter than {MIN_PASSWORD_LENGTH} characters; "
                    f"choose a longer one.", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    if password:
        typer.echo(f"sign-in required: user {settings.access_user!r}, password from "
                   f"PITCH_ACCESS_PASSWORD")
    elif host not in LOCAL_HOSTS and not settings.allowed_hosts:
        typer.secho("no password, so the pages answer only to this machine's own names; set "
                    "PITCH_ALLOWED_HOSTS to the address people will type (for example "
                    "192.168.1.20) to reach them from the network.", fg=typer.colors.YELLOW)
    uvicorn.run("pitch_occupancy.api.app:app", host=host, port=port, reload=reload)


@app.command("retention")
def retention_cmd(
    apply_now: bool = typer.Option(
        False, "--apply", help="actually delete; without this the command only reports"
    ),
) -> None:
    """Report what the retention policy would delete, and optionally delete it.

    Dry run by default and deliberately: these are frames of identifiable people kept under
    fixed retention periods (sampled frames 7 days, evidence 365 days), and the
    directories this touches sit next to the ones that must never be touched.
    """
    from pitch_occupancy.retention import apply as apply_retention
    from pitch_occupancy.retention import plan as plan_retention

    retention = plan_retention()
    typer.echo(retention.describe())
    if not apply_now:
        typer.echo("")
        typer.secho("dry run - nothing deleted. Add --apply to delete.",
                    fg=typer.colors.YELLOW)
        return
    removed = apply_retention(retention, confirm=True)
    typer.secho(f"deleted {removed} file(s)", fg=typer.colors.GREEN)


@app.command("schedule")
def schedule_cmd(
    days: int = typer.Option(1, "--days", min=1, help="how far ahead to list"),
    recordings: bool = typer.Option(
        False, "--from-recordings",
        help="derive the schedule from the recorded slots instead of the config",
    ),
) -> None:
    """List the slots the scheduler would run. Runs nothing.

    Deciding and doing are separate on purpose: "what would run" should be answerable
    without running anything, the same shape as `pitch retention`.
    """
    from datetime import datetime

    from pitch_occupancy.scheduler import (
        describe,
        due,
        load_schedule,
        schedule_from_recordings,
    )

    if recordings:
        schedule, dates = schedule_from_recordings(settings.raw_dir / "venue_01")
        when = datetime.combine(dates[0], datetime.min.time()) if dates else datetime.now()
        typer.echo(f"derived from recordings; {len(dates)} recorded date(s)")
    else:
        try:
            schedule = load_schedule()
        except (FileNotFoundError, ValueError) as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(1) from exc
        when = datetime.now()

    typer.echo(f"{len(schedule)} slot(s) in the schedule\n")
    for line in describe(schedule, when, days=days):
        typer.echo(f"  {line}")
    running = due(schedule, when)
    typer.echo("")
    typer.echo(f"due at {when:%Y-%m-%d %H:%M}: "
               + (", ".join(s.slot_id(when.date()) for s in running) or "nothing"))


@app.command("bookings")
def bookings_cmd(
    path: Path | None = typer.Option(None, help="Defaults to configs/bookings_example.csv."),
    venue: str = typer.Option("venue_01", help="Venue used to build slot ids."),
) -> None:
    """Import a booking export and report what it contains. Writes nothing.

    Read-only by construction, not by flag: `bookings.BookingSource` has one method. This
    command exists so a client's export can be checked before it is relied on - a booking
    dropped at import reconciles to *unbooked usage*, which is the anomaly that accuses
    someone.
    """
    from pitch_occupancy.bookings import SOLD, read_bookings

    try:
        records = read_bookings(path)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    typer.echo(f"{len(records)} booking(s) from {'the example' if path is None else path}\n")
    for r in sorted(records, key=lambda r: (r.date, r.start, r.field_id)):
        mark = "sold" if r.is_sold else "    "
        typer.echo(f"  {r.slot_id(venue):<32} {r.field_id:<10} {r.status:<12} {mark}  "
                   f"{r.duration_minutes:>3} min  {r.customer_ref}")

    sold = sum(1 for r in records if r.is_sold)
    typer.echo("")
    typer.echo(f"  {sold} sold, {len(records) - sold} not sold "
               f"(sold statuses: {', '.join(sorted(SOLD))})")
    if {r.source for r in records} == {"example"}:
        typer.secho(
            "\nThis is the committed example, not a client export. It exists to show the "
            "columns being asked for.", fg=typer.colors.YELLOW,
        )


@app.command("backup")
def backup_cmd(
    dest: Annotated[Path | None, typer.Option(
        "--dest", help="Backup folder; default <data_dir>/backups. Keep it private.")] = None,
    keep: int = typer.Option(14, "--keep", min=1, help="How many backups to keep."),
    with_secrets: bool = typer.Option(
        False, "--with-secrets",
        help="Also copy cameras.json, bookings_source.json and alerts.json - they hold "
             "passwords."),
    no_configs: bool = typer.Option(False, "--no-configs", help="Copy the database only."),
) -> None:
    """Copy the verdict database (safe while the worker runs) and this system's config files.

    The copy is made with SQLite's online backup API, checked with ``PRAGMA integrity_check``
    and named ``pitch_monitor-YYYYMMDD-HHMMSS.db``; the folder is pruned to the newest
    ``--keep``. Config files that hold passwords are copied only with ``--with-secrets``, and
    then the backup folder must be private. Schedule it daily - see docs/backup.md.
    """
    from pitch_occupancy.backup import SECRET_FILES, run_backup

    target = dest or (settings.data_dir / "backups")
    try:
        result = run_backup(target, keep, configs=not no_configs, with_secrets=with_secrets)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        typer.secho(f"backup failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    typer.secho(f"database -> {result.database} (integrity check ok)", fg=typer.colors.GREEN)
    if result.configs is not None:
        typer.echo(f"configs  -> {result.configs} ({', '.join(result.config_names)})")
    if result.removed:
        typer.echo(f"pruned {len(result.removed)} old file(s), keeping the newest {keep}")
    if with_secrets:
        typer.secho(
            f"WARNING: the config zip holds passwords ({', '.join(SECRET_FILES)}: camera, "
            f"booking-database and alert logins). Keep {target} private - anyone who can read "
            f"it can log in to the cameras.", fg=typer.colors.YELLOW)
    elif not no_configs:
        typer.echo(f"not copied (they hold passwords): {', '.join(SECRET_FILES)}. Add "
                   f"--with-secrets to include them, into a private folder.")


if __name__ == "__main__":
    app()
