"""Project paths and runtime settings.

Paths are resolved from the installed package location, so they work identically from a
notebook, a test, the CLI, and the scheduler on the Mini-PC. Settings are read from the
environment (prefix ``PITCH_``) or a local ``.env``, never hard-coded.
"""

from __future__ import annotations

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

    #: DINOv2 rather than the early prototype's ConvNeXtV2. Latency measurement showed no
    #: backbone is close to the sampling budget (20 cameras in 2.5-5.7 s of a 60 s cycle), so
    #: speed does not discriminate and the choice falls to accuracy under leakage-free
    #: evaluation, where DINOv2 leads.
    default_model_key: str = "dinov2"

    # paths - see docs/data_layout.md for what belongs in each
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    raw_dir: Path = DATA_DIR / "raw"  # immutable source footage
    interim_dir: Path = DATA_DIR / "interim"  # extracted, not yet labelled
    dataset_dir: Path = DATA_DIR / "processed"  # the labelled dataset
    reference_dir: Path = DATA_DIR / "reference"  # annotated screenshots
    db_path: Path = DATA_DIR / "db" / "pitch_monitor.db"
    evidence_dir: Path = DATA_DIR / "evidence"
    feature_cache_dir: Path = DATA_DIR / "cache"
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

    # slot aggregation baseline (see docstring)
    used_min_playing_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    notused_max_playing_ratio: float = Field(default=0.10, ge=0.0, le=1.0)
    notused_min_empty_ratio: float = Field(default=0.75, ge=0.0, le=1.0)

    # two-tier pipeline confidence bands
    tier1_confident_above: float = Field(default=0.75, ge=0.0, le=1.0)
    tier2_escalate_below: float = Field(default=0.40, ge=0.0, le=1.0)

    # reproducibility
    random_seed: int = 42

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


settings = Settings()
