"""The free vision-model second opinion: parsing, budget, pacing and the one-way upgrade.

No network: every provider call goes through a fake opener.
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from pitch_occupancy.assist import budget, providers
from pitch_occupancy.assist.config import AssistConfig, load
from pitch_occupancy.assist.reviewer import CameraPicture, review, should_ask
from pitch_occupancy.db.schema import connect, initialise

NOW = datetime(2026, 10, 20, 20, 0, tzinfo=timezone(timedelta(hours=3)))
PICTURE = [CameraPicture(np.full((360, 640, 3), 90, np.uint8), ((100, 100, 120, 160),))]


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    initialise(c)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(budget, "_minute", {})
    # Only the environment the test sets: a real .env on the machine must not leak in.
    import os

    from pitch_occupancy import config as app_config

    monkeypatch.setattr(app_config, "env_value", lambda name: os.environ.get(name, ""))


class Reply:
    def __init__(self, payload: dict) -> None:
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self.body


def answering(state: str, confidence: float, calls: list | None = None):
    def opener(request, timeout):
        if calls is not None:
            calls.append(json.loads(request.data))
        content = json.dumps({"state": state, "people_on_pitch": 12,
                              "confidence": confidence, "reason": "two teams"})
        return Reply({"choices": [{"message": {"content": f"```json\n{content}\n```"}}],
                      "usage": {"total_tokens": 900}})
    return opener


def store_check(conn, people: int, playing: int) -> dict:
    cur = conn.execute(
        """INSERT INTO playground_checks (playground_id, checked_at, people, playing,
               cameras_read, cameras_total, counting) VALUES ('pg', ?, ?, ?, 1, 1, 'add')""",
        (NOW.isoformat(), people, playing))
    conn.commit()
    return {"check_id": cur.lastrowid, "playground_id": "pg", "people": people,
            "playing": playing}


ON = AssistConfig(enabled=True, providers=("gemini",))


def test_the_reply_is_read_even_inside_a_code_fence() -> None:
    assert providers.parse('sure: {"state": "Playing", "people_on_pitch": "9", '
                           '"confidence": 1.4, "reason": "x"}') == ("playing", 9, 1.0, "x")
    with pytest.raises(providers.ProviderError):
        providers.parse('{"state": "dancing"}')
    with pytest.raises(providers.ProviderError):
        providers.parse("no json here")


def test_a_doubtful_check_the_model_calls_a_game_is_upgraded(conn) -> None:
    record = store_check(conn, people=3, playing=0)
    calls: list = []
    row = review(ON, conn, record, PICTURE, now=NOW, play_min=5,
                 opener=answering("playing", 0.9, calls))
    assert row["upgraded"] == 1 and record["playing"] == 1
    stored = conn.execute("SELECT * FROM playground_checks").fetchone()
    assert (stored["playing"], stored["detector_playing"], stored["playing_source"]) == \
        (1, 0, "assist")
    sent = calls[0]["messages"][0]["content"]
    assert sent[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "test-key" not in json.dumps(calls[0])


def test_an_unsure_or_negative_answer_changes_nothing(conn) -> None:
    for state, conf in (("playing", 0.5), ("other", 0.95)):
        record = store_check(conn, people=3, playing=0)
        row = review(AssistConfig(enabled=True, providers=("gemini",), min_gap_minutes=1,
                                  per_window=10), conn, record, PICTURE,
                     now=NOW + timedelta(minutes=5 * (conf > 0.6)), play_min=5,
                     opener=answering(state, conf))
        assert row["upgraded"] == 0 and record["playing"] == 0


def test_the_model_never_turns_a_game_off(conn) -> None:
    record = store_check(conn, people=9, playing=1)
    row = review(ON, conn, record, PICTURE, now=NOW, play_min=5,
                 opener=answering("empty", 1.0))
    assert row["doubtful"] == 0 and row["upgraded"] == 0
    assert conn.execute("SELECT playing FROM playground_checks").fetchone()[0] == 1


def test_questions_are_paced_per_window(conn) -> None:
    cfg = AssistConfig(enabled=True, per_window=2, window_minutes=30, min_gap_minutes=5)
    record = store_check(conn, people=2, playing=0)
    for minutes in (0, 6):
        assert review(cfg, conn, record, PICTURE, now=NOW + timedelta(minutes=minutes),
                      play_min=5, opener=answering("other", 0.9)) is not None
    assert should_ask(cfg, conn, record, NOW + timedelta(minutes=12), 5).ask is False
    assert should_ask(cfg, conn, record, NOW + timedelta(minutes=40), 5).ask is True
    assert should_ask(cfg, conn, record, NOW + timedelta(minutes=8), 5).why == \
        "this window's questions are used"


def test_an_ordinary_check_is_audited_once_per_window_only(conn) -> None:
    record = store_check(conn, people=0, playing=0)
    assert should_ask(ON, conn, record, NOW, 5).ask is True
    review(ON, conn, record, PICTURE, now=NOW, play_min=5, opener=answering("empty", 0.9))
    later = NOW + timedelta(minutes=10)
    assert should_ask(ON, conn, record, later, 5).ask is False
    no_audit = AssistConfig(enabled=True, audit=False)
    assert should_ask(no_audit, conn, store_check(conn, 0, 0), NOW + timedelta(hours=2),
                      5).ask is False


def test_the_daily_budget_stops_well_below_the_free_limit(conn) -> None:
    gemini = providers.PROVIDERS["gemini"]
    cap = int(gemini.requests_per_day * budget.HEADROOM)
    conn.execute("INSERT INTO assist_usage (provider, day, requests, tokens) VALUES (?, ?, ?, 0)",
                 ("gemini", NOW.astimezone(UTC).date().isoformat(), cap))
    ok, why = budget.reserve(conn, gemini, NOW)
    assert not ok and "used up" in why


def test_a_quota_refusal_rests_the_provider_and_the_next_one_answers(conn, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    asked: list[str] = []

    def opener(request, timeout):
        asked.append(request.full_url)
        if "googleapis" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 429, "quota", {}, io.BytesIO())
        return answering("playing", 0.9)(request, timeout)

    cfg = AssistConfig(enabled=True, providers=("gemini", "groq"))
    record = store_check(conn, people=4, playing=0)
    row = review(cfg, conn, record, PICTURE, now=NOW, play_min=5, opener=opener)
    assert row["provider"] == "groq" and row["upgraded"] == 1
    ok, why = budget.reserve(conn, providers.PROVIDERS["gemini"], NOW + timedelta(hours=1))
    assert not ok and "resting" in why


def test_nothing_is_asked_without_a_key_or_when_off(conn, monkeypatch) -> None:
    record = store_check(conn, people=3, playing=0)
    assert review(AssistConfig(enabled=False), conn, record, PICTURE, now=NOW, play_min=5,
                  opener=answering("playing", 1.0)) is None
    monkeypatch.delenv("GEMINI_API_KEY")
    assert review(ON, conn, record, PICTURE, now=NOW, play_min=5,
                  opener=answering("playing", 1.0)) is None


def test_the_settings_file_is_validated(tmp_path) -> None:
    assert load(tmp_path / "missing.json").enabled is False
    bad = tmp_path / "a.json"
    bad.write_text(json.dumps({"enabled": "yes"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load(bad)
    bad.write_text(json.dumps({"providers": ["paid-one"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load(bad)
    good = tmp_path / "b.json"
    good.write_text(json.dumps({"enabled": True, "models": {"groq": "other/model"}}),
                    encoding="utf-8")
    cfg = load(good)
    assert [p.model for p in cfg.chain() if p.name == "groq"] == ["other/model"]


def test_every_picture_sent_is_blurred_even_with_nobody_detected() -> None:
    import cv2

    from pitch_occupancy.assist.reviewer import prepare

    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)
    sent = prepare([CameraPicture(noisy)], AssistConfig(enabled=True))[0]
    decoded = cv2.imdecode(np.frombuffer(sent, np.uint8), cv2.IMREAD_COLOR)
    # A blurred noise image has far less pixel-to-pixel variation than the original.
    assert np.abs(np.diff(decoded.astype(int), axis=1)).mean() < \
        0.5 * np.abs(np.diff(noisy.astype(int), axis=1)).mean()


def test_pictures_are_shrunk_to_the_configured_side() -> None:
    import cv2

    from pitch_occupancy.assist.reviewer import prepare

    big = np.zeros((1080, 1920, 3), np.uint8)
    sent = prepare([CameraPicture(big, ((100, 100, 200, 400),))],
                   AssistConfig(enabled=True, max_side=640))[0]
    assert cv2.imdecode(np.frombuffer(sent, np.uint8), cv2.IMREAD_COLOR).shape[1] == 640


def test_a_doubtful_check_is_not_held_back_by_the_windows_comparison_sample(conn) -> None:
    review(ON, conn, store_check(conn, people=0, playing=0), PICTURE, now=NOW, play_min=5,
           opener=answering("empty", 0.9))
    doubtful = store_check(conn, people=3, playing=0)
    assert should_ask(ON, conn, doubtful, NOW + timedelta(minutes=1), 5).ask is True


def test_a_reservation_is_taken_before_the_request_is_sent(conn) -> None:
    groq = providers.PROVIDERS["groq"]
    assert budget.reserve(conn, groq, NOW)[0]
    row = conn.execute("SELECT requests, tokens FROM assist_usage").fetchone()
    assert tuple(row) == (1, groq.tokens_per_request)
    budget.settle(conn, groq, NOW, tokens=900)
    assert conn.execute("SELECT tokens FROM assist_usage").fetchone()[0] == 900


def test_a_rest_that_runs_past_midnight_utc_still_holds(conn) -> None:
    groq = providers.PROVIDERS["groq"]
    budget.rest(conn, groq, NOW, 10 * 3600)  # 17:00 UTC + 10 h = 03:00 UTC tomorrow
    ok, why = budget.reserve(conn, groq, NOW + timedelta(hours=8))  # 01:00 UTC tomorrow
    assert not ok and "resting" in why
    assert budget.reserve(conn, groq, NOW + timedelta(hours=11))[0]
