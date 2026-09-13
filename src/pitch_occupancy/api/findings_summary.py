"""A summary layer over the experiment log, for the Findings tab (WP8).

The Findings tab served `results/EXPERIMENT_LOG.md` in full: 75 dated entries, a quarter of
a megabyte of Markdown, every one of them rendered open. That is the right *archive* and the
wrong *page*. The log is append-only history written for whoever needs to know what was true
in September; a reader arriving at the tab wants to know what the project found.

Nothing is removed here. The page gains two layers above the log and the log itself becomes
scannable:

* **The verified claims**, from `thesis/claims.toml`. Every quantitative claim the write-up
  makes, grouped by the hypothesis it belongs to, each one re-derived from the artefact that
  produced it by `experiments/verify_claims.py`. This is the summary the project already
  maintains - it was simply never shown next to the findings it summarises.
* **A retraction strip**, because the log's most valuable entries are the ones that withdraw
  an earlier number, and those were the hardest to find in a wall of prose. An entry that
  retracts, corrects or supersedes something is worth surfacing above the entry that made
  the claim.
* **The log, one collapsed entry per run.** Date and title stay visible so the whole history
  is scannable in a screen or two; the prose is one click away. `<details>` rather than
  JavaScript, so it still works in a saved copy.

**Why claims.toml rather than the log's own headline sentences.** The log records what was
true when each entry was written, and superseded entries stay as written - so a summary
scraped from it would reprint retracted numbers as findings. The ledger holds only current
values and fails loudly when a source stops producing one, which is exactly the property a
summary needs. `verify_claims.py` deliberately does not check the log for this reason, and
this module inherits that decision.
"""

from __future__ import annotations

import re
import tomllib
from functools import lru_cache
from pathlib import Path

from pitch_occupancy.api.markdown import render

__all__ = ["STYLES", "claim_groups", "log_entries", "render_summary"]

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "thesis" / "claims.toml"
LOG = ROOT / "results" / "EXPERIMENT_LOG.md"

#: Group headers in the ledger look like `# --- H3 and the false-play control ----`. Parsed
#: from the raw text because `tomllib` drops comments, and the grouping is the only thing
#: that turns 34 statements into an argument a reader can follow.
_GROUP = re.compile(r"^#\s*-{2,}\s*(.+?)\s*-{2,}\s*$", re.MULTILINE)
_CLAIM_ID = re.compile(r'^id\s*=\s*"([^"]+)"', re.MULTILINE)

#: Words that mark an entry as withdrawing or correcting an earlier one. Matched on the
#: entry body, case-insensitively. Deliberately narrow: "wrong" and "corrected" appear in
#: entries that are *about* a correction as often as in ones that make it, and a strip that
#: flagged half the log would be no better than the wall it replaced.
_RETRACTION = re.compile(
    r"\b(retract(?:ed|ion|s)?|withdrawn|supersede[sd]?|superseded|"
    r"does not survive|did not survive|no longer holds)\b",
    re.IGNORECASE,
)

_HEADING = re.compile(r"^##\s+(.+)$")

#: The date in an entry heading, wherever it sits. The log's house style is
#: `## 2026-09-08 · Title`, but a third of the entries are `## Title (WP3-T8) — 2026-09-07`
#: and the pilot's is parenthesised as `## Pilot (2026-09-01/02) — superseded`. A pattern
#: anchored to the front read 19 of 76 headings as undated and dumped them, titles and all,
#: into one unsorted "no date" pile - so the date is searched for rather than expected in
#: one place, and removed from the title wherever it was found.
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2}(?:/\d{2})?)")

#: Separators and stray punctuation left behind once a date is lifted out of a heading.
_LEFTOVER = " ·—–-:,()"

MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _split_heading(heading: str) -> tuple[str, str]:
    """``(date, title)`` for an entry heading, with the date removed from the title."""
    found = _DATE.search(heading)
    if not found:
        return "", heading
    title = heading[: found.start()] + heading[found.end():]
    title = title.replace("()", " ")
    title = re.sub(r"\s{2,}", " ", title).strip(_LEFTOVER)
    return found.group(1), title or heading


def _month_label(date: str) -> str:
    """`2026-09-07` -> `September 2026`. Unparseable dates group under "Undated"."""
    try:
        year, month = int(date[:4]), int(date[5:7])
        return f"{MONTHS[month - 1]} {year}"
    except (ValueError, IndexError):
        return "Undated"


def _day_label(date: str) -> str:
    """`2026-09-07` -> `7 September 2026`; `2026-09-01/02` keeps its span."""
    try:
        year, month, rest = int(date[:4]), int(date[5:7]), date[8:]
        return f"{rest.lstrip('0').replace('/0', '–').replace('/', '–')} " \
               f"{MONTHS[month - 1]} {year}"
    except (ValueError, IndexError):
        return date or "Undated"


@lru_cache(maxsize=1)
def _ledger_text() -> str:
    return LEDGER.read_text(encoding="utf-8") if LEDGER.exists() else ""


def claim_groups() -> list[dict]:
    """The ledger's claims, in file order, grouped by their section header.

    Returns ``[{"title": str, "claims": [{"statement", "value", "source", "where"}]}]``.
    An unparseable or absent ledger yields no groups rather than raising: the Findings tab
    should still serve the log if the ledger is mid-edit.
    """
    text = _ledger_text()
    if not text:
        return []
    try:
        data = tomllib.loads(text)["claim"]
    except (KeyError, tomllib.TOMLDecodeError):
        return []

    # Walk the raw text once, recording which group each claim id falls under. Position
    # rather than nesting, because the headers are comments and carry no structure.
    marks: list[tuple[int, str, str]] = [
        (m.start(), "group", m.group(1)) for m in _GROUP.finditer(text)
    ]
    marks += [(m.start(), "claim", m.group(1)) for m in _CLAIM_ID.finditer(text)]
    marks.sort()

    group_of: dict[str, str] = {}
    current = "Other claims"
    for _, kind, value in marks:
        if kind == "group":
            current = value
        else:
            group_of[value] = current

    by_claim = {c["id"]: c for c in data if "id" in c}
    groups: list[dict] = []
    for cid, claim in by_claim.items():
        title = group_of.get(cid, "Other claims")
        if not groups or groups[-1]["title"] != title:
            groups.append({"title": title, "claims": []})
        groups[-1]["claims"].append({
            "statement": claim.get("statement", ""),
            "value": claim.get("value"),
            "source": claim.get("source", ""),
            "where": claim.get("where", []) or [],
            "unsupported": not claim.get("source"),
        })
    return groups


def log_entries() -> list[dict]:
    """The experiment log split into one entry per `##` heading.

    Each entry is ``{"date", "title", "body_html", "retraction"}``. The preamble before the
    first heading is returned as an entry with no date, so nothing in the file is dropped -
    a summary that silently swallowed the log's own explanation of what it is would be a
    worse page, not a shorter one.
    """
    if not LOG.exists():
        return []
    text = LOG.read_text(encoding="utf-8")

    chunks: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            chunks.append((heading.group(1).strip(), []))
        else:
            chunks[-1][1].append(line)

    entries = []
    for heading, lines in chunks:
        body = "\n".join(lines).strip()
        # Leading rules belong to the separator between entries, not to the entry - they
        # render as a stray line above every single one.
        body = re.sub(r"^(?:---\s*\n?)+", "", body).strip()
        if not heading and not body:
            continue
        date, title = _split_heading(heading)
        entries.append({
            "date": date,
            # The preamble has no heading at all - it is the log explaining what the log is,
            # and it needs a name rather than an empty summary row nobody would click.
            "title": title or "What this log is",
            "month": _month_label(date) if date else "Undated",
            "body_html": render(body) if body else "",
            "retraction": bool(_RETRACTION.search(body)),
        })
    return entries


def _fmt(value) -> str:
    """The ledger's value, exactly as recorded.

    `str` rather than a format spec, and that is the whole point. `:g` was the first
    attempt and it strips trailing zeros, so the constant predictor's macro-F1 of `1.0`
    rendered as a bare `1` and the empty-accuracy claim's `0.0` as `0` - both reading as
    integers, or as rounded, in a ledger whose only job is to record what was measured.
    Nothing here rounds, pads or normalises: the chip shows the number the ledger holds.
    """
    if value is None:
        return "—"
    return str(value)


def render_summary() -> str:
    """The summary layers plus the collapsed log, as one HTML block."""
    groups = claim_groups()
    entries = log_entries()
    dated = [e for e in entries if e["date"]]
    retractions = [e for e in dated if e["retraction"]]
    n_claims = sum(len(g["claims"]) for g in groups)
    unsupported = sum(1 for g in groups for c in g["claims"] if c["unsupported"])

    out = ['<div class="fsum">']

    out.append(
        '<div class="ftiles">'
        f'<div class="ftile lead"><b>{n_claims}</b><span>verified claims, each re-derived '
        'from the artefact that produced it</span></div>'
        f'<div class="ftile"><b>{unsupported}</b><span>claims with no source, and they say '
        'so</span></div>'
        f'<div class="ftile"><b>{len(dated)}</b><span>logged runs, oldest first</span></div>'
        f'<div class="ftile"><b>{len(retractions)}</b><span>entries that withdraw or '
        'supersede an earlier number</span></div>'
        '</div>'
    )

    if groups:
        out.append('<h2>What the project found</h2>')
        out.append(
            '<p class="fnote">Grouped by hypothesis, in ledger order. Every value is '
            'recomputed from its source by <code>verify_claims.py</code>; a claim whose '
            'source stops producing it fails the check rather than sitting here looking '
            'settled.</p>'
        )
        for g in groups:
            rows = "".join(
                '<li class="fclaim">'
                f'<span class="fval{" none" if c["unsupported"] else ""}">'
                f'{_fmt(c["value"])}</span>'
                f'<span class="fstate">{c["statement"]}'
                f'<em>{c["source"] or "no source — unsupported, and recorded as such"}</em>'
                '</span></li>'
                for c in g["claims"]
            )
            out.append(f'<section class="fgroup"><h3>{g["title"]}</h3>'
                       f'<ul class="fclaims">{rows}</ul></section>')

    if retractions:
        out.append('<h2>What was withdrawn</h2>')
        out.append(
            '<p class="fnote">The entries a reader most needs and could least easily find. '
            'A number that was published and then retracted is part of the method, so these '
            'stay in the log as written &mdash; this strip only makes them reachable.</p>'
        )
        out.append('<div class="fretract">' + "".join(
            f'<a href="#entry-{i}"><b>{e["date"]}</b>{e["title"]}</a>'
            for i, e in enumerate(entries) if e["retraction"]
        ) + '</div>')

    out.append('<h2>The full log</h2>')
    out.append(
        f'<p class="fnote">{len(entries)} entries, grouped by month and collapsed &mdash; '
        'one per run: date, hypothesis, command, result file, and the finding in a '
        'sentence. Appended automatically by the experiment scripts; the findings written '
        'by hand. Nothing here is omitted, and the newest month is open.</p>'
    )

    # Grouped by **day**, not month. Month was the first attempt and it grouped 74 of the
    # 76 entries into one section - this project ran inside a single September, so a month
    # heading is a heading over the whole log and the page was a wall again the moment it
    # opened. By day it is seven rows, which is a summary.
    #
    # Only the last group opens, by position rather than by label: the log has an undated
    # correction in the middle of it, so matching on a label opened two separate sections
    # that happened to share one.
    days: list[tuple[str, list[tuple[int, dict]]]] = []
    for i, e in enumerate(entries):
        key = e["date"] or "Undated"
        if not days or days[-1][0] != key:
            days.append((key, []))
        days[-1][1].append((i, e))

    for group_index, (day, items) in enumerate(days):
        rows = []
        for i, e in items:
            flag = '<i class="fflag">retraction</i>' if e["retraction"] else ""
            rows.append(
                f'<details class="fentry" id="entry-{i}">'
                f'<summary><b>{e["date"] or "&mdash;"}</b>{e["title"]}{flag}</summary>'
                f'<div class="fbody">{e["body_html"]}</div></details>'
            )
        n_retract = sum(1 for _, e in items if e["retraction"])
        meta = f'{len(items)} entr{"y" if len(items) == 1 else "ies"}'
        if n_retract:
            meta += f' &middot; {n_retract} retraction{"s" if n_retract > 1 else ""}'
        is_last = group_index == len(days) - 1
        out.append(
            f'<details class="fmonth"{" open" if is_last else ""}>'
            f'<summary><b>{_day_label(day)}</b><i>{meta}</i></summary>'
            f'<div class="fmbody">{"".join(rows)}</div></details>'
        )

    out.append('</div>')
    # A link into a collapsed entry scrolls to a shut box otherwise. Anchor navigation does
    # not open an element's `<details>` ancestors in every browser, so the retraction strip
    # above would look broken in the ones where it does not - the entry is in the DOM, just
    # not visible. Opens the target and every ancestor, on load and on hash change.
    out.append(
        "<script>(function(){function open(){var t=location.hash&&"
        "document.querySelector(location.hash);if(!t)return;"
        "for(var n=t;n;n=n.parentElement){if(n.tagName==='DETAILS')n.open=true;}"
        "t.scrollIntoView({block:'start'});}"
        "addEventListener('hashchange',open);open();})();</script>"
    )
    return "".join(out)


#: Scoped to `.fsum` so nothing here can reach the rendered Markdown of the other tabs.
STYLES = """
.fsum { margin: 0 0 8px; }
.fsum h2 { margin: 34px 0 6px; }
.fsum h3 { margin: 0 0 10px; font-size: 14px; letter-spacing: .01em; }
.fnote { font-size: 13px; color: var(--ink-3); max-width: 70ch; margin: 0 0 14px; }

.ftiles { display: grid; gap: 12px; margin: 18px 0 8px;
  grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); }
.ftile { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 15px; }
.ftile.lead { border-left: 4px solid var(--accent); }
.ftile b { display: block; font: 700 25px 'JetBrains Mono', monospace;
  font-variant-numeric: tabular-nums; letter-spacing: -.02em; margin-bottom: 3px; }
.ftile span { font-size: 12.5px; color: var(--ink-2); line-height: 1.45; }

.fgroup { background: var(--surface); border: 1px solid var(--line); border-radius: 11px;
  padding: 15px 16px; margin-bottom: 12px; }
.fclaims { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
.fclaim { display: grid; grid-template-columns: 84px 1fr; gap: 13px; align-items: start; }
.fval { font: 700 13px 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums;
  text-align: right; color: var(--accent); padding-top: 1px; }
.fval.none { color: var(--ink-3); }
.fstate { font-size: 13.5px; color: var(--ink); }
.fstate em { display: block; font: 500 11px 'JetBrains Mono', monospace; font-style: normal;
  color: var(--ink-3); margin-top: 3px; }

.fretract { display: grid; gap: 6px; margin: 0 0 8px; }
.fretract a { display: flex; gap: 12px; align-items: baseline; text-decoration: none;
  background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--down);
  border-radius: 8px; padding: 9px 13px; font-size: 13.5px; color: var(--ink); }
.fretract a:hover { background: var(--surface-2); }
.fretract b { font: 600 11.5px 'JetBrains Mono', monospace; color: var(--down);
  flex: none; }

.fmonth { border: 1px solid var(--line); border-radius: 11px; background: var(--surface-2);
  margin-bottom: 8px; }
.fmonth > summary { cursor: pointer; padding: 12px 15px; display: flex; gap: 12px;
  align-items: baseline; list-style: none; }
.fmonth > summary::-webkit-details-marker { display: none; }
.fmonth > summary::before { content: "\\25B8"; color: var(--ink-3); flex: none;
  font-size: 11px; }
.fmonth[open] > summary::before { content: "\\25BE"; }
.fmonth > summary b { font-size: 14.5px; font-weight: 600; }
.fmonth > summary i { font: 500 11px 'JetBrains Mono', monospace; font-style: normal;
  color: var(--ink-3); margin-left: auto; }
.fmbody { padding: 0 10px 10px; }

.fentry { border: 1px solid var(--line); border-radius: 9px; background: var(--surface);
  margin-bottom: 6px; }
.fentry > summary { cursor: pointer; padding: 10px 14px; display: flex; gap: 12px;
  align-items: baseline; font-size: 13.5px; list-style: none; }
.fentry > summary::-webkit-details-marker { display: none; }
.fentry > summary::before { content: "+"; font: 600 13px 'JetBrains Mono', monospace;
  color: var(--ink-3); flex: none; }
.fentry[open] > summary::before { content: "\\2212"; }
.fentry[open] > summary { border-bottom: 1px solid var(--line); }
.fentry > summary:hover { background: var(--surface-2); }
.fentry > summary b { font: 600 11.5px 'JetBrains Mono', monospace; color: var(--ink-3);
  flex: none; }
.fflag { margin-left: auto; font: 600 9.5px 'JetBrains Mono', monospace;
  letter-spacing: .08em; text-transform: uppercase; font-style: normal; color: var(--down);
  border: 1px solid var(--down); border-radius: 4px; padding: 2px 6px; flex: none; }
.fbody { padding: 2px 16px 10px; }
.fbody > :first-child { margin-top: 10px; }
@media (max-width: 560px) {
  .fclaim { grid-template-columns: 1fr; gap: 3px; }
  .fval { text-align: left; }
}
"""
