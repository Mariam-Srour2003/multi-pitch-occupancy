"""Command-line entry point.

Everything that is not a web request goes through here: dataset building, experiments,
and running the services. Installed as ``pitch``::

    uv run pitch --help
    uv run pitch info
    uv run pitch serve
"""

from __future__ import annotations

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
