"""The simple pages and their JSON over playgrounds: what a person sees and can change.

No camera is contacted and no model loads: picture-taking and counting are faked
(`playground_env`), and every file and database is a temporary one.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from playground_env import SECRET, make_env
from playground_env import new_playground as _new

from pitch_occupancy import playgrounds


@pytest.fixture
def env(tmp_path, monkeypatch):
    return make_env(tmp_path, monkeypatch)


# --- pages and endpoints -------------------------------------------------------------------

@pytest.mark.parametrize("path, text", [
    ("/", "/api/v1/playgrounds"), ("/playgrounds", "+ New playground"),
    ("/cameras/add", "+ Add camera"), ("/records", "Played for about"),
    ("/more", "Pitch boundaries"),
])
def test_every_simple_page_renders_with_the_short_menu(env, path, text) -> None:
    r = env.client.get(path)
    assert r.status_code == 200 and text in r.text
    nav = r.text[r.text.index("<nav>"):r.text.index("</nav>")]
    labels = ("Home", "Playgrounds", "Cameras", "Results", "Records", "More")
    assert [label for label in labels if f">{label}</a>" in nav] == list(labels)
    assert nav.count("</a>") == 6


@pytest.mark.parametrize("old", ["/live", "/cameras/test"])
def test_the_first_versions_addresses_lead_home(env, old) -> None:
    r = env.client.get(old, follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/"


def test_making_changing_and_deleting_a_playground_through_the_api(env) -> None:
    r = env.client.post("/api/v1/playgrounds", json={
        "name": "Pitch 1", "cameras": ["site/a", "site/b"], "every_seconds": 30})
    assert r.status_code == 200 and r.json()["id"] == "pitch-1"
    listed = env.client.get("/api/v1/playgrounds").json()
    assert listed["playgrounds"][0]["cameras"] == ["site/a", "site/b"]
    owned = {c["ref"]: c["playground"] for c in listed["cameras"]}
    assert owned["site/a"] == "pitch-1" and owned["site/c"] is None
    assert SECRET not in json.dumps(listed)
    r = env.client.put("/api/v1/playgrounds/pitch-1", json={
        "name": "Pitch 1", "cameras": ["site/c"], "every_seconds": 60, "counting": "largest",
        "hours": {"from": "08:00", "to": "22:00"}})
    assert r.status_code == 200 and r.json()["hours"] == {"from": "08:00", "to": "22:00"}
    assert env.client.post("/api/v1/playgrounds/pitch-1/enabled",
                           json={"enabled": False}).json()["enabled"] is False
    bad = env.client.post("/api/v1/playgrounds", json={
        "name": "x", "cameras": ["site/c"], "every_seconds": 30})
    assert bad.status_code == 422 and "one playground only" in bad.json()["detail"]
    assert env.client.delete("/api/v1/playgrounds/pitch-1").status_code == 200
    assert env.client.delete("/api/v1/playgrounds/pitch-1").status_code == 404


def test_look_now_returns_pictures_and_latest_shows_each_camera(env, monkeypatch) -> None:
    monkeypatch.setattr(playgrounds.check, "_picture", lambda frame, counted: "PICTURE")
    _new("Big", ("site/a", "site/dead"))
    r = env.client.post("/api/v1/playgrounds/big/look").json()
    assert r["people"] == 3
    assert {c["camera"]: c["picture"] for c in r["cameras"]} == {
        "site/a": "PICTURE", "site/dead": None}
    latest = env.client.get("/api/v1/playgrounds/big/latest").json()
    assert latest["latest"]["people"] == 3
    assert [(c["camera"], c["readable"], c["people"]) for c in latest["cameras"]] == [
        ("site/a", True, 3), ("site/dead", False, None)]


def test_records_endpoint_and_spreadsheet(env) -> None:
    playgrounds.check_playground(_new("Big", ("site/a", "site/b")))
    today = datetime.now().astimezone().date().isoformat()
    d = env.client.get(f"/api/v1/records?day={today}").json()
    assert d["period_minutes"] == 90 and d["rows"][0]["max_people"] == 8
    csv_text = env.client.get(f"/api/v1/records.csv?day={today}").text
    assert csv_text.splitlines()[0].startswith("day,start,end,playground,max_people")
    assert ",Big,8," in csv_text and ",yes," in csv_text
    assert env.client.get("/api/v1/records?day=yesterday").status_code == 422


def test_a_playground_name_cannot_break_out_of_its_page_script(env) -> None:
    p = _new("x</script><b>y", ("site/a",))
    r = env.client.get(f"/playground/{p['id']}")
    assert r.status_code == 200 and "x</script><b>y" not in r.text
    assert env.client.get("/playground/nope").status_code == 404


def test_each_camera_still_has_its_own_live_page(env) -> None:
    r = env.client.get("/live/site/a")
    assert r.status_code == 200 and SECRET not in r.text
    assert env.client.get("/live/site/nope").status_code == 404
