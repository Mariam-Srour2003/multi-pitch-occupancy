"""Pictures: every saved picture with what it was labelled and why.

* ``/pictures``             the pictures of one day, filterable by playground and label, each
  with its camera's answer (label, people inside, everyone seen, ball, confidence, the rule's
  reason), the playground's combined answer and any second opinion; full screen on click.
* ``/api/v1/pictures``      the same rows as JSON; ``/api/v1/pictures.csv`` as a spreadsheet.

Rows come from ``camera_checks`` (one per camera per check, ``picture_path`` set when the
picture was saved) joined to its ``playground_checks`` row and ``assist_opinions``. A picture
the size cap has since deleted keeps its row; ``url`` is then null.
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from pitch_occupancy import playgrounds
from pitch_occupancy.api.simple_pages import _html
from pitch_occupancy.api.templating import render, template
from pitch_occupancy.db.schema import open_db
from pitch_occupancy.playgrounds.check import LABELS

__all__ = ["router"]

router = APIRouter()

BODY = render("pictures_page_body.html", CSS_EXTRA=template("results_page_css_extra.html"))
JS = template("pictures_page_js.js")

#: Columns of the JSON rows and the spreadsheet, in order.
FIELDS = ("checked_at", "playground", "camera", "label", "people", "people_seen", "ball",
          "ball_confidence", "confidence", "rule", "reason", "playground_people",
          "playground_playing", "playing_source", "second_opinion", "second_opinion_reason",
          "picture_path", "url")

_SQL = """
SELECT cc.checked_at, cc.camera_id, cc.state, cc.label, cc.people, cc.people_seen, cc.ball,
       cc.ball_confidence, cc.confidence, cc.rule, cc.reason, cc.picture_path,
       pc.playground_id, pc.people AS pg_people, pc.playing AS pg_playing, pc.playing_source,
       (SELECT ao.state FROM assist_opinions ao WHERE ao.check_id = pc.check_id
         ORDER BY ao.opinion_id DESC LIMIT 1) AS ai_state,
       (SELECT ao.reason FROM assist_opinions ao WHERE ao.check_id = pc.check_id
         ORDER BY ao.opinion_id DESC LIMIT 1) AS ai_reason
FROM camera_checks cc
LEFT JOIN playground_checks pc ON pc.check_id = cc.playground_check_id
WHERE cc.picture_path IS NOT NULL AND substr(cc.checked_at, 1, 10) = ?
"""


@router.get("/pictures", response_class=HTMLResponse, include_in_schema=False)
def pictures_page() -> HTMLResponse:
    return _html("Pictures", BODY, JS, "pictures")


def _day(value: str | None) -> date:
    if not value:
        return datetime.now().astimezone().date()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(422, "day must look like 2026-10-25") from exc


def _rows(day: date, playground: str | None, label: str | None) -> list[dict]:
    names = {p["id"]: p["name"] for p in playgrounds.load()}
    root = playgrounds.snapshots.SNAPSHOT_DIR
    sql, args = _SQL, [day.isoformat()]
    if playground:
        sql += " AND pc.playground_id = ?"
        args.append(playground)
    if label:
        sql += " AND cc.label = ?"
        args.append(label)
    conn = open_db()
    try:
        found = conn.execute(sql + " ORDER BY cc.checked_at, cc.camera_id", args).fetchall()
    finally:
        conn.close()
    out = []
    for r in found:
        path = r["picture_path"]
        out.append({
            "checked_at": r["checked_at"],
            "playground": names.get(r["playground_id"], r["playground_id"]),
            "playground_id": r["playground_id"],
            "camera": r["camera_id"],
            "label": r["label"] or LABELS.get(r["state"] or "", r["state"]),
            "people": r["people"], "people_seen": r["people_seen"],
            "ball": None if r["ball"] is None else bool(r["ball"]),
            "ball_confidence": r["ball_confidence"], "confidence": r["confidence"],
            "rule": r["rule"], "reason": r["reason"],
            "playground_people": r["pg_people"],
            "playground_playing": None if r["pg_playing"] is None else bool(r["pg_playing"]),
            "playing_source": r["playing_source"] or ("detector" if r["pg_playing"] is not None
                                                      else None),
            "second_opinion": r["ai_state"], "second_opinion_reason": r["ai_reason"],
            "picture_path": path,
            "url": "/snapshots/" + path.replace("\\", "/") if (root / path).is_file() else None,
        })
    return out


def _days() -> list[str]:
    conn = open_db()
    try:
        return [r[0] for r in conn.execute(
            """SELECT DISTINCT substr(checked_at, 1, 10) FROM camera_checks
               WHERE picture_path IS NOT NULL ORDER BY 1 DESC LIMIT 60""")]
    finally:
        conn.close()


@router.get("/api/v1/pictures")
def pictures_json(day: str | None = None, playground: str | None = Query(None, max_length=80),
                  label: str | None = Query(None, max_length=40)) -> dict:
    which = _day(day)
    rows = _rows(which, playground, label)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["label"] or "?"] = counts.get(r["label"] or "?", 0) + 1
    return {"day": which.isoformat(), "days": _days(), "rows": rows, "counts": counts,
            "labels": list(LABELS.values()),
            "playgrounds": [{"id": p["id"], "name": p["name"]} for p in playgrounds.load()],
            "usage": playgrounds.snapshot_usage()}


@router.get("/api/v1/pictures.csv")
def pictures_csv(day: str | None = None, playground: str | None = Query(None, max_length=80),
                 label: str | None = Query(None, max_length=40)) -> Response:
    which = _day(day)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(_rows(which, playground, label))
    return Response(buf.getvalue(), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="pictures-{which.isoformat()}.csv"'})
