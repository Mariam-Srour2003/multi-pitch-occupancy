"""Command-line entry point.

Everything that is not a web request goes through here: dataset building, experiments,
and running the services. Installed as ``pitch``::

    uv run pitch --help
    uv run pitch info
    uv run pitch serve
"""

from __future__ import annotations

from pathlib import Path

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
    typer.echo(f"  random seed   : {settings.random_seed}")
    typer.echo("")
    for label, path in (
        ("raw footage", settings.raw_dir),
        ("interim", settings.interim_dir),
        ("processed", settings.dataset_dir),
        ("reference", settings.reference_dir),
        ("feature cache", settings.feature_cache_dir),
        ("results", settings.results_dir),
    ):
        mark = "ok     " if path.exists() else "missing"
        typer.echo(f"  [{mark}] {label:<14} {path}")


@app.command("seed")
def seed_cmd() -> None:
    """Fill the database from the real recorded slots, so the dashboard has data (WP6)."""
    from pitch_occupancy.db.seed import seed

    verdicts = seed()
    if not verdicts:
        typer.secho("no slot recordings found in the manifest", fg=typer.colors.RED)
        raise typer.Exit(1)
    for slot, status in sorted(verdicts.items()):
        typer.echo(f"  {slot:<32} {status}")
    typer.echo("")
    typer.echo(f"wrote {settings.db_path}")
    typer.secho("open http://127.0.0.1:8000/ after `pitch serve`", fg=typer.colors.GREEN)


@app.command("coverage")
def coverage_cmd(
    out: Path | None = typer.Option(None, help="Defaults to results/coverage.md."),
) -> None:
    """Report which cells of the class × lighting × venue matrix are empty (WP2-T1)."""
    from pitch_occupancy.data.coverage import empty_cells, render_report
    from pitch_occupancy.data.manifest import read_manifest

    rows = read_manifest(settings.dataset_dir / "manifest.csv")
    report = render_report(rows)
    target = out or (settings.results_dir / "coverage.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")

    typer.echo(report)
    gaps = empty_cells(rows)
    if gaps:
        typer.secho(f"\n{len(gaps)} empty cell(s) in the collection matrix.", fg=typer.colors.YELLOW)
    typer.echo(f"wrote {target}")


@app.command("cache")
def build_cache_cmd(
    backbones: list[str] = typer.Argument(None, help="Defaults to all registered backbones."),
    batch_size: int = typer.Option(16, help="Frames per forward pass."),
    threads: int = typer.Option(0, help="torch CPU threads; 0 leaves the default."),
) -> None:
    """Embed every manifest frame once per backbone into data/cache (WP0-T5)."""
    import time

    import torch

    from pitch_occupancy.data.feature_cache import build_cache
    from pitch_occupancy.data.manifest import read_manifest
    from pitch_occupancy.vision.backbones import BACKBONES

    if threads:
        torch.set_num_threads(threads)
    rows = read_manifest(settings.dataset_dir / "manifest.csv")
    chosen = backbones or sorted(BACKBONES)
    typer.echo(f"{len(rows)} frames x {len(chosen)} backbone(s)\n")

    for key in chosen:
        t0 = time.perf_counter()
        cached = build_cache(
            rows, key, settings.dataset_dir, settings.feature_cache_dir, batch_size=batch_size
        )
        dt = time.perf_counter() - t0
        typer.echo(
            f"  {key:<12} {len(cached):>5} x {cached.dim:<5} "
            f"{dt:>6.0f}s  ({dt / max(len(cached), 1) * 1000:.0f} ms/frame)"
        )


@app.command("extract-clips")
def extract_clips_cmd(
    per_clip: int = typer.Option(6, help="Frames sampled from each clip's middle 80%."),
    clips_dir: Path | None = typer.Option(None, help="Defaults to the 2026-09-04 batch."),
) -> None:
    """Extract frames from the highlight clips and write the metadata sidecar (WP2-T3)."""
    from pitch_occupancy.data.extract import (
        extract_clip_frames,
        load_clip_venues,
        write_sidecar,
    )

    src = clips_dir or (settings.raw_dir / "highlights_2026-09-04")
    out = settings.interim_dir / "frames" / "clips"
    rows = extract_clip_frames(src, out, venues=load_clip_venues(), per_clip=per_clip)
    path = write_sidecar(rows)
    typer.echo(f"{len(rows)} frames from {len({r.clip_id for r in rows})} clips -> {out}")
    typer.echo(f"sidecar -> {path}")
    typer.secho(
        "\nThese are not labelled yet. Review them before filing into a class folder.",
        fg=typer.colors.YELLOW,
    )


@app.command("manifest")
def build_manifest_cmd(
    venue: str = typer.Option("venue_01", help="Venue tag for the frames being indexed."),
    out: Path | None = typer.Option(None, help="Defaults to <dataset>/manifest.csv."),
    check_only: bool = typer.Option(False, "--check", help="Report only; write nothing."),
) -> None:
    """Build data/dataset/manifest.csv from the class folders (WP0-T2)."""
    from pitch_occupancy.data.manifest import (
        build_manifest,
        confound_warnings,
        cross_tab,
        write_manifest,
    )

    rows, problems = build_manifest(settings.dataset_dir, venue=venue)
    if not rows:
        typer.secho(f"no frames found under {settings.dataset_dir}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo(f"{len(rows)} frames indexed\n")
    # venue, not slot_id: with clips extracted there are 60+ slot groups and the table
    # becomes unreadable. Venue is also the axis that matters for leave-one-venue-out.
    typer.echo(cross_tab(rows, "class3", "venue"))
    typer.echo("")
    typer.echo(cross_tab(rows, "class3", "lighting"))
    typer.echo("")
    typer.echo(cross_tab(rows, "class3", "labeled_by"))

    warnings = confound_warnings(rows)
    if warnings:
        typer.echo("")
        typer.secho("CONFOUNDING", fg=typer.colors.YELLOW, bold=True)
        for w in warnings:
            typer.secho(f"  ! {w}", fg=typer.colors.YELLOW)

    if problems:
        typer.echo("")
        typer.secho(f"{len(problems)} problem(s):", fg=typer.colors.RED, bold=True)
        for p in problems[:20]:
            typer.secho(f"  - {p}", fg=typer.colors.RED)

    if check_only:
        raise typer.Exit(0)

    target = out or (settings.dataset_dir / "manifest.csv")
    write_manifest(rows, target)
    typer.echo(f"\nwrote {target}")


@app.command()
def serve(
    host: str = typer.Option(settings.api_host, help="Bind address."),
    port: int = typer.Option(settings.api_port, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Serve the dashboard and API with uvicorn."""
    import uvicorn

    uvicorn.run("pitch_occupancy.api.app:app", host=host, port=port, reload=reload)


@app.command("retention")
def retention_cmd(
    apply_now: bool = typer.Option(
        False, "--apply", help="actually delete; without this the command only reports"
    ),
) -> None:
    """Report what the retention policy would delete, and optionally delete it (WP6-T8).

    Dry run by default and deliberately: the periods come from `thesis/ethics.md`, and the
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
    days: int = typer.Option(1, "--days", help="how far ahead to list"),
    recordings: bool = typer.Option(
        False, "--from-recordings",
        help="derive the schedule from the recorded slots instead of the config",
    ),
) -> None:
    """List the slots the scheduler would run (WP6-T2). Runs nothing.

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
    """Import a booking export and report what it contains (WP6-T4). Writes nothing.

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


@app.command("retraining")
def retraining_cmd(
    apply_now: bool = typer.Option(
        False, "--apply", help="actually copy; without this the command only reports"
    ),
) -> None:
    """Stage overridden slots' evidence frames for re-labelling (WP6-T7).

    Dry run by default, the same shape as `pitch retention`. Frames are staged **unfiled**:
    an operator's override is a verdict about the whole slot, not a label for each frame, so
    a human files them before they can become training data.
    """
    from pitch_occupancy.db.schema import connect
    from pitch_occupancy.retraining import apply as apply_harvest
    from pitch_occupancy.retraining import plan as plan_harvest

    connection = connect(settings.db_path)
    try:
        harvest = plan_harvest(connection)
    finally:
        connection.close()

    typer.echo(harvest.describe())
    if not apply_now:
        typer.echo("")
        typer.secho("dry run - nothing copied. Add --apply to stage.", fg=typer.colors.YELLOW)
        return
    copied = apply_harvest(harvest, confirm=True)
    typer.secho(f"staged {copied} frame(s)", fg=typer.colors.GREEN)
    if copied:
        typer.secho(
            "Now look at each one and move it into 1_empty / 2_playing / 3_maintenance. "
            "Nothing under _incoming/ is indexed by `pitch manifest`.",
            fg=typer.colors.YELLOW,
        )


if __name__ == "__main__":
    app()
