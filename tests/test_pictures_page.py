"""The Pictures page: saved pictures with their label, counts, ball and reason."""

from __future__ import annotations

import pytest
from playground_env import make_env
from playground_env import new_playground as _new

from pitch_occupancy import playgrounds
from pitch_occupancy.config import settings
from pitch_occupancy.db.schema import connect


@pytest.fixture
def env(tmp_path, monkeypatch):
    return make_env(tmp_path, monkeypatch)


def test_each_camera_check_stores_what_its_answer_rests_on(env) -> None:
    record = playgrounds.check_playground(_new("Big", ("site/a", "site/b")), save=True)
    conn = connect(settings.db_path)
    rows = conn.execute("SELECT * FROM camera_checks WHERE playground_check_id = ?",
                        (record["check_id"],)).fetchall()
    conn.close()
    assert len(rows) == 2
    for r in rows:
        assert r["picture_path"] and r["label"] in ("Empty", "Playing", "People, not playing",
                                                    "Not sure")
        assert r["confidence"] is not None and r["rule"] is not None and r["reason"]
        assert r["ball"] in (0, 1) and r["people_seen"] is not None


def test_the_page_lists_the_days_pictures_with_their_labels(env) -> None:
    p = _new("Big", ("site/a", "site/b"))
    playgrounds.check_playground(p, save=True)
    assert env.client.get("/pictures").status_code == 200
    d = env.client.get("/api/v1/pictures").json()
    assert len(d["rows"]) == 2 and sum(d["counts"].values()) == 2
    row = d["rows"][0]
    assert row["playground"] == "Big" and row["url"].startswith("/snapshots/")
    assert env.client.get(row["url"]).status_code == 200
    one = env.client.get(f"/api/v1/pictures?playground={p['id']}&label={row['label']}").json()
    assert one["rows"] and all(r["label"] == row["label"] for r in one["rows"])
    csv_text = env.client.get("/api/v1/pictures.csv").text
    assert csv_text.splitlines()[0].startswith("checked_at,playground,camera,label,people")
    assert env.client.get("/api/v1/pictures?day=yesterday").status_code == 422


def test_a_picture_deleted_by_the_size_cap_keeps_its_row(env) -> None:
    playgrounds.check_playground(_new("Big", ("site/a",)), save=True)
    for f in playgrounds.snapshots.SNAPSHOT_DIR.rglob("*.jpg"):
        f.unlink()
    rows = env.client.get("/api/v1/pictures").json()["rows"]
    assert rows and rows[0]["url"] is None and rows[0]["label"]
