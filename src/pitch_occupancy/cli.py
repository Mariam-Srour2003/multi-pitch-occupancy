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
        ("data", settings.data_dir),
        ("dataset", settings.dataset_dir),
        ("results", settings.results_dir),
        ("feature cache", settings.feature_cache_dir),
    ):
        mark = "ok     " if path.exists() else "missing"
        typer.echo(f"  [{mark}] {label:<14} {path}")


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
    typer.echo(cross_tab(rows, "class3", "slot_id"))
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


if __name__ == "__main__":
    app()
