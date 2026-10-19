"""Project paths and runtime settings.

Paths are resolved from the installed package location, so they work identically from a
notebook, a test, the CLI, and the scheduler on the Mini-PC. Settings are read from the
environment (prefix ``PITCH_``) or a local ``.env``, never hard-coded.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["SourceType", "Settings", "settings", "PROJECT_ROOT", "DATA_DIR", "RESULTS_DIR"]

# src/pitch_occupancy/config.py -> src/pitch_occupancy -> src -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"


class SourceType(StrEnum):
    """Where frames come from. The pipeline is identical for all three."""

    VIDEO_SIM = "VIDEO_SIM"  # replay recorded footage as if it were live
    API_SIM = "API_SIM"  # pull from the simulator API
    RTSP_LIVE = "RTSP_LIVE"  # real cameras


class Settings(BaseSettings):
    """Runtime configuration.

    Aggregation thresholds are the slot decision rule. They are hyper-parameters, so they are
    tuned on training slots rather than left at these hand-set values when comparing against
    any learned alternative.
    """

    model_config = SettingsConfigDict(
        env_prefix="PITCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    source_type: SourceType = SourceType.VIDEO_SIM
    sample_interval_s: int = 60

    #: The detector to run (a key in `vision/detector.DETECTORS`), from
    #: `PITCH_DEFAULT_MODEL_KEY`. Empty means the detector `configs/rules.json` names, so the
    #: rule file stays the one place the deployed model is chosen. An unknown key falls back
    #: to that default with a warning (`pipeline.resolve_model_key`).
    default_model_key: str = ""

    # paths - see docs/data_layout.md for what belongs in each
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    raw_dir: Path = DATA_DIR / "raw"  # immutable source footage
    interim_dir: Path = DATA_DIR / "interim"  # extracted, not yet labelled
    dataset_dir: Path = DATA_DIR / "processed"  # the labelled dataset
    reference_dir: Path = DATA_DIR / "reference"  # annotated screenshots
    db_path: Path = DATA_DIR / "db" / "pitch_monitor.db"
    evidence_dir: Path = DATA_DIR / "evidence"
    #: The detector-first decision rule's numbers (`vision/rules.RuleConfig`). Committed,
    #: like a boundary: a rule is part of what the deployment *is*.
    rules_path: Path = CONFIGS_DIR / "rules.json"
    #: Where the booking-day test looks for recordings already on this machine
    #: (`PITCH_RECORDINGS_DIR`). A day of video is many GB per camera; reading it in place
    #: beats uploading it to the same computer. Files here are the client's and are never
    #: deleted by the test.
    recordings_dir: Path = DATA_DIR / "recordings"

    #: Per-video upload limit for the booking-day test, in MB (`PITCH_TESTING_MAX_UPLOAD_MB`).
    #: One hour of real recording is about 630 MB per camera, a day about 7 GB, so the
    #: reviewers' clip-sized cap is far too small here. The upload is also refused up front
    #: when the temp folder's disk could not hold it.
    testing_max_upload_mb: int = Field(default=16384, ge=1)

    #: Where the playground checks keep their saved pictures (`PITCH_SNAPSHOT_DIR`).
    snapshot_dir: Path = DATA_DIR / "snapshots"
    #: The saved pictures never take more than this many MB (`PITCH_SNAPSHOT_CAP_MB`); the
    #: oldest are deleted to make room.
    snapshot_cap_mb: int = 1024
    #: Camera streams opened at the same time across all playgrounds (`PITCH_PARALLEL_GRABS`).
    #: Over a slow link, opening many 1080p streams at once times every one of them out.
    parallel_grabs: int = 2

    # api
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    #: Bearer token for the simulator snapshot endpoint, from `PITCH_SIMULATOR_TOKEN`.
    #:
    #: **Empty means the endpoint is disabled, not open.** That distinction is the whole point
    #: of it being empty by default: an unset secret that defaults to "no authentication
    #: required" is the same class of defect as a threshold left at a value that can never
    #: fire, and this project has now found several of those. The endpoint serves frames of
    #: identifiable people, so it fails closed with 503 when unconfigured rather than serving
    #: them to anyone who asks. A test asserts that.
    #:
    #: Never committed. Set it in `.env`, which is gitignored, or in the service environment.
    simulator_token: str = ""

    #: The password every page and endpoint asks for, from `PITCH_ACCESS_PASSWORD`
    #: (`api/access.py`). Empty means no sign-in, which is only acceptable while the server
    #: listens on this machine alone - `pitch serve` refuses a network address without it
    #: unless told `--no-password`. Set it in `.env` (gitignored) or the service environment.
    access_password: SecretStr = SecretStr("")
    #: The user name that goes with it, from `PITCH_ACCESS_USER`.
    access_user: str = "admin"
    #: Extra host names the pages may be reached by (`PITCH_ALLOWED_HOSTS`, comma separated),
    #: for example the Mini-PC's address on the venue network. See `api/access.RequestGuard`.
    allowed_hosts: str = ""
    #: Serve /docs, /redoc and /openapi.json (`PITCH_API_DOCS`). Off by default.
    api_docs: bool = False


settings = Settings()


def env_value(name: str) -> str:
    """A variable from the environment, else from ``.env`` (this folder's, then the project's).

    For keys the settings object does not own - the free vision providers' API keys keep the
    names their providers document (`GEMINI_API_KEY`), not a ``PITCH_`` one. Read on every
    call, so a key added to ``.env`` needs no restart of whatever reads it next.
    """
    value = os.environ.get(name, "").strip()
    if value:
        return value
    for env_file in (Path.cwd() / ".env", PROJECT_ROOT / ".env"):
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            key, sep, raw = line.strip().removeprefix("export ").partition("=")
            if sep and key.strip() == name:
                return raw.strip().strip('"').strip("'")
    return ""
