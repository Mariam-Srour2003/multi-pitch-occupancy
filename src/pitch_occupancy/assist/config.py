"""``configs/assist.json``: whether the second opinion runs, how often, and through whom.

Gitignored, like the camera file; ``configs/assist.example.json`` shows every key. A missing
file means off. Keys never live here - each provider reads its own environment variable
(`providers.Provider.key_env`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from pitch_occupancy.assist.providers import PROVIDERS, Provider
from pitch_occupancy.config import CONFIGS_DIR

__all__ = ["AssistConfig", "CONFIG_PATH", "load"]

CONFIG_PATH = CONFIGS_DIR / "assist.json"


@dataclass(frozen=True, slots=True)
class AssistConfig:
    enabled: bool = False
    #: Tried in this order; the first one configured and within budget answers.
    providers: tuple[str, ...] = ("gemini", "groq", "openrouter")
    #: At most ``per_window`` questions per playground every ``window_minutes``, and never two
    #: within ``min_gap_minutes`` of each other.
    per_window: int = 3
    window_minutes: int = 30
    min_gap_minutes: int = 5
    #: Also ask about one ordinary check per window when nothing doubtful happened, so the
    #: stored disagreements say how often the two differ. Changes no verdict.
    audit: bool = True
    #: The model must be at least this sure before a check is upgraded to playing.
    upgrade_confidence: float = 0.7
    #: Longest side of a picture sent, in pixels.
    max_side: int = 1024
    blur_heads: bool = True
    timeout_s: float = 30.0
    #: Per-provider model overrides, for when a provider changes its free list.
    models: dict[str, str] = field(default_factory=dict)

    def chain(self) -> list[Provider]:
        """The providers to try, in order, with any model override applied."""
        out = []
        for name in self.providers:
            base = PROVIDERS.get(name)
            if base is None:
                continue
            model = self.models.get(name)
            out.append(replace(base, model=model) if model else base)
        return out


def load(path: Path | None = None) -> AssistConfig:
    """Read the settings; a missing file is the default (off). Raises on a malformed one."""
    source = path or CONFIG_PATH
    if not source.exists():
        return AssistConfig()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{source.name} must hold one JSON object")
    values = {k: v for k, v in raw.items() if not k.startswith("_")}
    unknown = sorted(set(values) - set(AssistConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"{source.name}: unknown setting(s) {unknown}")
    if "providers" in values:
        names = values["providers"]
        if not isinstance(names, list) or not all(n in PROVIDERS for n in names):
            raise ValueError(f"{source.name}: providers must be a list of {sorted(PROVIDERS)}")
        values["providers"] = tuple(names)
    if "enabled" in values and not isinstance(values["enabled"], bool):
        raise ValueError(f"{source.name}: enabled must be true or false")
    for key in ("per_window", "window_minutes", "min_gap_minutes", "max_side"):
        if key in values and (not isinstance(values[key], int) or values[key] < 1):
            raise ValueError(f"{source.name}: {key} must be a whole number of at least 1")
    conf = values.get("upgrade_confidence")
    if conf is not None and not (isinstance(conf, int | float) and 0.0 <= conf <= 1.0):
        raise ValueError(f"{source.name}: upgrade_confidence must be between 0 and 1")
    return AssistConfig(**values)
