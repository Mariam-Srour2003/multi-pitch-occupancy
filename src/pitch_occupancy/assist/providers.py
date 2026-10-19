"""Free vision-language model providers behind one OpenAI-compatible call.

Every provider listed here speaks the OpenAI ``/chat/completions`` format with images as
``image_url`` data URIs, so one client serves them all and adding a provider is adding an
entry. The limits are each provider's published free-tier numbers as of October 2026; the
budget (`budget.py`) refuses well below them, and ``configs/assist.json`` can lower them
further or swap the model when a provider changes its free list.

Sources checked 2026-10-20: Groq vision docs (``qwen/qwen3.8-27b``, 3 images per request,
free plan 30 requests/min, 1,000/day, 8,000 tokens/min, 200,000 tokens/day); Gemini
OpenAI-compatible endpoint (free tier per project, Flash-Lite has the most generous daily
quota); OpenRouter free models (50 requests/day without bought credits).
"""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

__all__ = ["Provider", "PROVIDERS", "Opinion", "ProviderError", "RateLimited", "ask"]


@dataclass(frozen=True, slots=True)
class Provider:
    """One free endpoint: where it is, which model, which key, and its free limits."""

    name: str
    base_url: str
    model: str
    key_env: str
    #: Free-tier limits. The budget allows only a share of these (`budget.HEADROOM`).
    requests_per_day: int
    requests_per_minute: int
    tokens_per_day: int | None = None
    #: Roughly what one request costs in tokens with one picture, for the token budget.
    tokens_per_request: int = 2600
    max_images: int = 3

    @property
    def key(self) -> str:
        from pitch_occupancy.config import env_value

        return env_value(self.key_env)

    @property
    def configured(self) -> bool:
        return bool(self.key)


PROVIDERS: dict[str, Provider] = {
    "gemini": Provider(
        "gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-3.5-flash-lite", "GEMINI_API_KEY",
        requests_per_day=500, requests_per_minute=15, tokens_per_request=1500),
    "groq": Provider(
        "groq", "https://api.groq.com/openai/v1", "qwen/qwen3.8-27b", "GROQ_API_KEY",
        requests_per_day=1000, requests_per_minute=30, tokens_per_day=200_000,
        tokens_per_request=2600, max_images=3),
    "openrouter": Provider(
        "openrouter", "https://openrouter.ai/api/v1", "google/gemma-4-31b-it:free",
        "OPENROUTER_API_KEY", requests_per_day=50, requests_per_minute=20),
}

STATES = ("playing", "other", "empty")


@dataclass(frozen=True, slots=True)
class Opinion:
    """What the model said about one check."""

    provider: str
    model: str
    state: str  # one of STATES
    people: int | None
    confidence: float
    reason: str
    tokens: int
    latency_ms: int


class ProviderError(RuntimeError):
    """The provider could not answer usefully this time."""


class RateLimited(ProviderError):
    """The provider refused for quota (HTTP 429): rest it, do not retry."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


PROMPT = """You are checking a camera picture of a sports pitch (football/soccer, possibly \
other sports). {cameras} The pitch area is the playing surface; people outside it (stands, \
benches, behind fences, walkways) do not count.

Decide one state:
- "playing": a game or a training session is going on on the pitch (people spread over the \
pitch, running, a ball in use, players in kit). A detector counted {people} people, but it \
misses small, far-away players, so look carefully at the far side.
- "other": people are on the pitch but not playing (maintenance, mowing, line painting, \
walking across, standing in a group, cleaning).
- "empty": nobody on the pitch.

Answer with JSON only, no other text:
{{"state": "playing" | "other" | "empty", "people_on_pitch": <integer estimate>, \
"confidence": <0.0-1.0>, "reason": "<at most 15 words>"}}"""


def _prompt(people: int | None, cameras: int) -> str:
    many = (f"There are {cameras} pictures, from different cameras on the same pitch, taken at "
            f"the same moment." if cameras > 1 else "")
    return PROMPT.format(people="an unknown number of" if people is None else people,
                         cameras=many)


def parse(text: str) -> tuple[str, int | None, float, str]:
    """``(state, people, confidence, reason)`` from the model's reply; raises on nonsense."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ProviderError("the reply held no JSON")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ProviderError("the reply's JSON did not parse") from exc
    state = str(data.get("state", "")).strip().lower()
    if state not in STATES:
        raise ProviderError(f"unknown state {state!r}")
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    people = data.get("people_on_pitch")
    try:
        people = max(0, int(people)) if people is not None else None
    except (TypeError, ValueError):
        people = None
    return state, people, confidence, str(data.get("reason", ""))[:200]


def ask(provider: Provider, images_jpeg: list[bytes], *, people: int | None,
        timeout_s: float = 30.0, opener=None) -> Opinion:
    """Ask ``provider`` about up to ``max_images`` JPEG pictures of one moment.

    Raises `RateLimited` on HTTP 429 and `ProviderError` on anything else that is not a
    usable answer. One attempt, never a retry loop: a retry is another request against the
    free quota. ``opener`` replaces ``urllib.request.urlopen`` in tests.
    """
    if not provider.configured:
        raise ProviderError(f"{provider.key_env} is not set")
    if not images_jpeg:
        raise ProviderError("no picture to send")
    images = images_jpeg[: provider.max_images]
    content: list[dict] = [{"type": "text", "text": _prompt(people, len(images))}]
    for data in images:
        uri = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": uri}})
    body = json.dumps({
        "model": provider.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 200,
    }).encode("utf-8")
    request = urllib.request.Request(
        provider.base_url.rstrip("/") + "/chat/completions", data=body, method="POST",
        headers={"Authorization": f"Bearer {provider.key}",
                 "Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with (opener or urllib.request.urlopen)(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            retry = exc.headers.get("retry-after") if exc.headers else None
            try:
                wait = float(retry) if retry else None
            except ValueError:
                wait = None
            raise RateLimited(f"{provider.name}: quota reached (429)", wait) from None
        # The body may echo the request; only the status is kept, never the key or image.
        raise ProviderError(f"{provider.name}: HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderError(f"{provider.name}: {type(exc).__name__}") from None
    except json.JSONDecodeError:
        raise ProviderError(f"{provider.name}: the response was not JSON") from None
    latency = int((time.perf_counter() - started) * 1000)
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ProviderError(f"{provider.name}: no answer in the response") from None
    if isinstance(text, list):  # some providers return content parts
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    state, estimate, confidence, reason = parse(text)
    usage = payload.get("usage") or {}
    tokens = int(usage.get("total_tokens") or provider.tokens_per_request)
    return Opinion(provider.name, provider.model, state, estimate, confidence, reason,
                   tokens, latency)
