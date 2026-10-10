"""The bookings page: connect a booking source, check its data, compare it with the cameras.

Served at ``/bookings``, with its JSON under ``/api/v1/bookings/...``. Everything here
**reads**. The two POST routes take a request body (an uploaded CSV, or nothing) and store
nothing: ``/test`` reads the configured source and ``/check`` runs `booking_check` over rows,
the same shape as ``/schedule/validate`` - a POST because a CSV does not fit in a URL, not
because anything changes.

**There is deliberately no route that saves the source config.** It was considered (the
schedule editor shows how a config write can be made safe) and left out for two reasons. The
config holds a database password, and a web form that accepts one would also have to keep
history copies of it somewhere the repository ignores - which is a second secret file to
get wrong. And a page that can repoint the system at a different database is a page that
can make reconciliation compare against the wrong records. Editing
``configs/bookings_source.json`` by hand on the machine itself is the safer path, and the page
shows exactly what to write.

Credentials never leave the server: the configured source is shown through
`booking_sources.describe`, which masks passwords, and database errors are masked too.
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import date, time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from pitch_occupancy.api.layout import page
from pitch_occupancy.api.routes import get_conn
from pitch_occupancy.booking_check import check
from pitch_occupancy.booking_sources import (
    DEFAULT_SOURCE_CONFIG,
    describe,
    load_source,
    records_from_rows,
    rows_from_csv_text,
)
from pitch_occupancy.bookings import coverage, reconcile_slot
from pitch_occupancy.data.taxonomy import SlotStatus

__all__ = ["router", "bookings_html"]

router = APIRouter(tags=["bookings"])

#: The config file read on every request - module-level so a test can point it elsewhere.
SOURCE_CONFIG = DEFAULT_SOURCE_CONFIG
#: Largest CSV accepted by ``/check``. A year of hourly bookings on ten pitches is ~5 MB.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

#: What a failing source can raise. Each becomes a 422 with the message, never a 500 - the
#: request was understood, and the person needs to read why it could not be served.
_SOURCE_ERRORS = (ValueError, FileNotFoundError, RuntimeError, OSError, sqlite3.Error)


def _schedule():
    """The camera schedule, or None - the check runs without it rather than failing."""
    try:
        from pitch_occupancy.scheduler import load_schedule
        return load_schedule()
    except Exception:  # noqa: BLE001 - an unreadable schedule only drops one optional rule
        return None


def _configured():
    """``(source, None)``, ``(None, None)`` if none configured, or ``(None, problem)``."""
    try:
        return load_source(SOURCE_CONFIG), None
    except ValueError as exc:
        return None, str(exc)


def _require_source():
    source, problem = _configured()
    if problem:
        raise HTTPException(422, f"The booking source config has a problem: {problem}")
    if source is None:
        raise HTTPException(422, f"No booking source is configured. Create "
                                 f"configs/{SOURCE_CONFIG.name} (see the examples on the "
                                 f"Bookings page), or upload a CSV instead.")
    return source


def _record_dict(r) -> dict:
    return {"field_id": r.field_id, "date": r.date.isoformat(), "start": f"{r.start:%H:%M}",
            "end": f"{r.end:%H:%M}", "customer_ref": r.customer_ref, "status": r.status,
            "entered_by": r.entered_by}


# --- JSON ---------------------------------------------------------------------------------


@router.get("/api/v1/bookings/source")
def booking_source() -> dict:
    """The configured source with credentials masked, or ``{"configured": false}``."""
    source, problem = _configured()
    if problem:
        return {"configured": True, "valid": False, "problem": problem,
                "file": f"configs/{SOURCE_CONFIG.name}"}
    if source is None:
        return {"configured": False, "file": f"configs/{SOURCE_CONFIG.name}"}
    return {"configured": True, "valid": True, "file": f"configs/{SOURCE_CONFIG.name}",
            "source": describe(source.config)}


@router.post("/api/v1/bookings/test")
def try_source() -> dict:
    """Read the configured source once: row count, dates covered, and the first ten rows.

    Reads with the strict reader, so "connected" means "readable by reconciliation", not
    merely "the database answered".
    """
    source = _require_source()
    try:
        rows = source.rows()
    except _SOURCE_ERRORS as exc:
        raise HTTPException(422, f"Could not read the booking source: {exc}") from None
    try:
        records = records_from_rows(rows, source=source.source_tag)
    except ValueError as exc:
        raise HTTPException(
            422, f"Connected and read {len(rows)} row(s), but the data cannot be used as it "
                 f"is: {exc}. Use \"Check booking data\" to see every problem at once."
        ) from None
    span = coverage(records)
    return {
        "ok": True,
        "rows": len(records),
        "pitches": len({r.field_id for r in records}),
        "date_from": span[0].isoformat() if span else None,
        "date_to": span[1].isoformat() if span else None,
        "preview": [_record_dict(r) for r in records[:10]],
    }


@router.post("/api/v1/bookings/check")
async def check_bookings(request: Request) -> dict:
    """Check booking data. Body: CSV text to check that, or empty to check the configured source.

    Stores nothing. Returns the `booking_check` report.
    """
    body = await request.body()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That file is larger than 20 MB; export a shorter period.")
    text = body.decode("utf-8-sig", errors="replace")
    if text.strip():
        try:
            rows = rows_from_csv_text(text)
        except (ValueError, csv.Error) as exc:
            raise HTTPException(422, f"Could not read the CSV: {exc}") from None
        what = "the uploaded CSV"
    else:
        source = _require_source()
        try:
            rows = source.rows()
        except _SOURCE_ERRORS as exc:
            raise HTTPException(422, f"Could not read the booking source: {exc}") from None
        what = f"the configured {source.kind} source"
    report = check(rows, schedule=_schedule())
    return {"checked": what} | report.to_dict()


@router.get("/api/v1/bookings/compare")
def compare(day: Annotated[date, Query(description="YYYY-MM-DD")],
            conn: Annotated[sqlite3.Connection, Depends(get_conn)]) -> dict:
    """What the cameras decided for each slot on ``day``, beside what the bookings say.

    Uses `bookings.reconcile_slot`, so a day the booking data does not reach is reported as
    needing review rather than as a page of unbooked usage.
    """
    slots = [dict(r) for r in conn.execute(
        """SELECT rs.slot_id, rs.field_id, rs.start_time, rs.end_time,
                  se.status, se.mean_confidence, se.is_overridden, se.override_status
           FROM rental_slots rs
           LEFT JOIN slot_evaluations se ON se.slot_id = rs.slot_id
           WHERE rs.slot_date = ?
           ORDER BY rs.field_id, rs.start_time""",
        (day.isoformat(),),
    ).fetchall()]

    source, problem = _configured()
    out: dict = {"day": day.isoformat(), "configured": source is not None, "slots": [],
                 "unmatched_bookings": [], "message": ""}
    if problem:
        raise HTTPException(422, f"The booking source config has a problem: {problem}")
    records = []
    if source is not None:
        try:
            records = source.read()
        except _SOURCE_ERRORS as exc:
            raise HTTPException(422, f"Could not read the booking source: {exc}") from None
    span = coverage(records)
    out["covered"] = bool(span) and span[0] <= day <= span[1]

    todays = [r for r in records if r.date == day]
    matched: set[tuple[str, time]] = set()
    for s in slots:
        start = time.fromisoformat(s["start_time"])
        booking = next((r for r in todays
                        if r.field_id == s["field_id"] and r.start == start), None)
        if booking:
            matched.add((booking.field_id, booking.start))
        verdict = (s["override_status"] if s["is_overridden"] and s["override_status"]
                   else s["status"])
        row = {
            "slot_id": s["slot_id"], "field_id": s["field_id"], "start": s["start_time"],
            "end": s["end_time"], "camera": verdict,
            "corrected": bool(s["is_overridden"]),
            "booking": booking.status if booking else None,
            "customer_ref": booking.customer_ref if booking else None,
            "anomaly": None, "severity": None, "explanation": "",
        }
        if verdict is None:
            row["explanation"] = "The cameras have not produced a verdict for this slot yet."
        elif source is not None:
            rec = reconcile_slot(records, field_id=s["field_id"], day=day, start=start,
                                 vision_status=SlotStatus(verdict))
            row.update(anomaly=str(rec.anomaly), severity=str(rec.severity),
                       explanation=rec.explanation)
        out["slots"].append(row)

    out["unmatched_bookings"] = [
        {k: v for k, v in _record_dict(r).items() if k != "entered_by"}
        for r in todays if (r.field_id, r.start) not in matched
    ]
    if not slots:
        out["message"] = ("The database has no camera slots for this day - the cameras were "
                          "not scheduled, or the worker has not run for it.")
    elif source is None:
        out["message"] = "No booking source is configured, so only the camera verdicts are shown."
    elif not out["covered"]:
        out["message"] = ("The booking data does not reach this day, so no slot is flagged as "
                          "unbooked - an absent booking cannot be told from an absent record.")
    return out


# --- the page -----------------------------------------------------------------------------

_EXAMPLES = {
    "CSV file": """{
  "type": "csv",
  "path": "exports/bookings.csv"
}""",
    "SQLite": """{
  "type": "sqlite",
  "path": "C:/BookingApp/data/bookings.db",
  "query": "SELECT pitch, starts_at, ends_at, ref, state FROM bookings",
  "columns": {"field_id": "pitch", "start": "starts_at", "end": "ends_at",
              "customer_ref": "ref", "status": "state"},
  "status_map": {"Booked": "confirmed", "Cancelled": "cancelled", "DNA": "no_show"}
}""",
    "PostgreSQL": """{
  "type": "sql",
  "url": "postgresql://pitch_reader:PASSWORD@db-host:5432/bookings",
  "query": "SELECT * FROM public.bookings WHERE booking_date >= CURRENT_DATE - 90",
  "columns": {"field_id": "pitch_code", "date": "booking_date", "start": "start_time",
              "end": "end_time", "customer_ref": "reference"}
}""",
    "MySQL / MariaDB": """{
  "type": "sql",
  "url": "mysql+pymysql://pitch_reader:PASSWORD@db-host:3306/bookings",
  "query": "SELECT * FROM bookings",
  "columns": {"field_id": "court", "start": "start_at", "end": "end_at",
              "customer_ref": "booking_no", "status": "state"}
}""",
    "SQL Server": """{
  "type": "sql",
  "url": "mssql+pyodbc://reader:PASSWORD@db-host/Bookings?driver=ODBC+Driver+18+for+SQL+Server",
  "query": "SELECT PitchId, StartsAt, EndsAt, BookingRef, Status FROM dbo.Bookings",
  "columns": {"field_id": "PitchId", "start": "StartsAt", "end": "EndsAt",
              "customer_ref": "BookingRef", "status": "Status"}
}""",
}


def _examples_html() -> str:
    from html import escape
    parts = []
    for i, (label, text) in enumerate(_EXAMPLES.items()):
        parts.append(
            f'<details{" open" if i == 0 else ""}><summary>{escape(label)}</summary>'
            f'<pre id="ex{i}">{escape(text)}</pre>'
            f'<button class="ghost" data-copy="ex{i}">Copy</button></details>')
    return "".join(parts)


_BODY = """
<p class="note"><b>What booking data is for.</b> The cameras say whether each pitch was
actually played on. Your booking records say whether it was <i>sold</i>. Putting the two side
by side shows games nobody booked, bookings nobody turned up for, and records that disagree.
Booking data is <b>optional</b> - without it you still get the camera verdicts - and it is
<b>read-only</b>: this system never changes anything in your booking system.</p>

<h2>Your booking source</h2>
<p class="sub2">Where the bookings are read from. The settings live in a file on this machine,
<code>configs/bookings_source.json</code>, which is never shared or uploaded because it can
contain a database password.</p>
<div class="card" id="srcCard"><span class="muted">Loading...</span></div>

<details class="card" style="margin-top:12px" id="howto">
<summary><b>How to connect your bookings</b></summary>
<p>Create <code>configs/bookings_source.json</code> with one of these, then reload this page.
Your own column names and status words can be kept - <code>columns</code> says which of your
columns holds each value (ours are <code>field_id</code>, <code>date</code>,
<code>start</code>, <code>end</code>, <code>customer_ref</code>, <code>status</code>,
<code>entered_by</code>), and <code>status_map</code> translates your status words into ours:
<code>confirmed</code>, <code>cancelled</code>, <code>no_show</code>, <code>maintenance</code>,
<code>blocked</code>. If your start and end columns hold a full date and time, the date is
taken from them. <code>date_format</code> / <code>time_format</code> handle other formats
(for example <code>%d/%m/%Y</code>).</p>
<p><b>Databases:</b> only a single <code>SELECT</code> query is accepted, but that check only
catches mistakes. The real protection is to give this system its own database login that can
<b>only read</b> the bookings table. Ask whoever runs your booking system for one. Databases
other than SQLite also need two extra packages installed (for example
<code>uv add sqlalchemy psycopg2-binary</code> for PostgreSQL).</p>
<div class="grid2" style="margin-top:10px">EXAMPLES</div>
</details>

<h2>Test the connection</h2>
<p class="sub2">Reads your booking source once and shows the first rows, so you can see the
columns came through correctly.</p>
<div class="controls" style="margin-top:0">
  <button class="act" id="testBtn">Test the connection</button></div>
<div id="testOut" style="margin-top:14px"></div>

<h2>Check booking data</h2>
<p class="sub2">Looks for problems that would make the comparison with the cameras wrong:
missing or unreadable values, bookings that end before they start, unknown statuses,
duplicates, double bookings, gaps. Every problem is listed at once. Nothing is saved.</p>
<div class="controls" style="margin-top:0">
  <div><label for="csvFile">A CSV file</label><input type="file" id="csvFile" accept=".csv,text/csv"
    style="width:auto"></div>
  <button class="act" id="checkFile" disabled>Check this file</button>
  <button class="ghost" id="checkSource">Check my booking source instead</button>
</div>
<div id="checkOut" style="margin-top:14px"></div>

<h2>Compare with what the cameras saw</h2>
<p class="sub2">For one day: each slot the cameras watched, what they decided, and what the
bookings say.</p>
<div class="controls" style="margin-top:0">
  <div><label for="day">Day</label><input type="date" id="day" style="width:auto"></div>
  <button class="act" id="cmpBtn">Compare</button>
</div>
<div id="cmpOut" style="margin-top:14px"></div>
"""

_CSS = """
details summary{cursor:pointer}
details pre{margin:8px 0 6px;white-space:pre-wrap;word-break:break-word}
.verdict{border-radius:10px;padding:14px 18px;margin:0 0 14px;font-weight:600;
  border:1px solid var(--line);background:var(--surface)}
.verdict.good{border-left:5px solid var(--play);color:var(--play)}
.verdict.warn{border-left:5px solid var(--maint);color:var(--maint)}
.verdict.bad{border-left:5px solid var(--flag);color:var(--flag)}
.verdict span{display:block;font-weight:400;color:var(--ink-2);font-size:13px}
.sev{display:inline-block;font:600 11px 'JetBrains Mono',monospace;padding:2px 8px;
  border-radius:5px}
.sev.error,.sev.serious{background:var(--flag-soft);color:var(--flag)}
.sev.warning{background:var(--maint-soft);color:var(--maint)}
.sev.info,.sev.none{background:var(--surface-2);color:var(--ink-2)}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
dl.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:0;font-size:13.5px}
dl.kv dt{color:var(--ink-3)}dl.kv dd{margin:0;word-break:break-word}
"""

_SCRIPT = r"""
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',
  '"':'&quot;',"'":'&#39;'}[c]));
async function call(url,opts){
  const r=await fetch(url,opts);let d=null;
  try{d=await r.json()}catch(e){}
  if(!r.ok)throw new Error((d&&d.detail)?(typeof d.detail==='string'?d.detail:
    JSON.stringify(d.detail)):('HTTP '+r.status));
  return d;
}
const errBox=m=>'<p class="note err">'+esc(m)+'</p>';
const table=(cols,rows)=>'<div class="scroll"><table><thead><tr>'+
  cols.map(c=>'<th>'+esc(c[1])+'</th>').join('')+'</tr></thead><tbody>'+
  rows.map(r=>'<tr>'+cols.map(c=>'<td'+(c[2]?' class="'+c[2]+'"':'')+'>'+
    (c[3]?c[3](r):esc(r[c[0]]))+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';
const tiles=list=>'<div class="tiles">'+list.map(([k,v,hot])=>'<div class="tile'+
  (hot?' flagged':'')+'"><div class="k">'+esc(k)+'</div><div class="v">'+esc(v)+
  '</div></div>').join('')+'</div>';

document.querySelectorAll('[data-copy]').forEach(b=>b.addEventListener('click',async()=>{
  const t=$(b.dataset.copy).textContent;
  try{await navigator.clipboard.writeText(t);b.textContent='Copied'}
  catch(e){b.textContent='Select and copy'}
  setTimeout(()=>b.textContent='Copy',1500);
}));

async function loadSource(){
  const card=$('srcCard');
  try{
    const d=await call('/api/v1/bookings/source');
    if(!d.configured){
      card.innerHTML='<b>No booking source configured.</b> <span class="muted">That is fine -'+
        ' the cameras work without one. To connect your bookings, create <code>'+esc(d.file)+
        '</code> as shown below.</span>';
      $('howto').open=true;$('testBtn').disabled=true;$('checkSource').disabled=true;return;
    }
    if(!d.valid){
      card.innerHTML='<b class="bad">The booking source settings have a problem.</b>'+
        '<p>'+esc(d.problem)+'</p><p class="muted">Fix <code>'+esc(d.file)+
        '</code> and reload.</p>';
      $('testBtn').disabled=true;$('checkSource').disabled=true;return;
    }
    const s=d.source,kv=[['Type',s.type]];
    if(s.name)kv.push(['Name',s.name]);
    if(s.path)kv.push(['File',s.path]);
    if(s.url)kv.push(['Database',s.url]);
    if(s.query)kv.push(['Query',s.query]);
    if(s.columns&&Object.keys(s.columns).length)kv.push(['Columns',Object.entries(s.columns)
      .map(([a,b])=>a+' \u2190 '+b).join(', ')]);
    if(s.status_map&&Object.keys(s.status_map).length)kv.push(['Statuses',
      Object.entries(s.status_map).map(([a,b])=>a+' \u2192 '+b).join(', ')]);
    for(const k of ['date_format','time_format','datetime_format'])if(s[k])kv.push([k,s[k]]);
    card.innerHTML='<dl class="kv">'+kv.map(([k,v])=>'<dt>'+esc(k)+'</dt><dd class="mono">'+
      esc(v)+'</dd>').join('')+'</dl><p class="muted" style="margin-top:8px">From <code>'+
      esc(d.file)+'</code>. Passwords are never shown.</p>';
  }catch(e){card.innerHTML=errBox(e.message)}
}

$('testBtn').addEventListener('click',async()=>{
  const out=$('testOut');out.innerHTML='<p class="muted">Reading...</p>';
  try{
    const d=await call('/api/v1/bookings/test',{method:'POST'});
    out.innerHTML='<p class="note"><b class="ok">Connected.</b> Read '+d.rows+
      ' booking(s).</p>'+tiles([['bookings',d.rows],['pitches',d.pitches],
      ['from',d.date_from||'-'],['to',d.date_to||'-']])+'<h3 style="font-size:14px">'+
      'First '+d.preview.length+' rows</h3>'+table([['field_id','Pitch'],['date','Date'],
      ['start','Start'],['end','End'],['customer_ref','Reference'],['status','Status'],
      ['entered_by','Entered by']],d.preview);
  }catch(e){out.innerHTML=errBox(e.message)}
});

let report=null,filter='all';
function showReport(){
  const d=report,out=$('checkOut');
  const cls={'good':'good','usable with warnings':'warn','has errors':'bad'}[d.verdict];
  const title={'good':'Good','usable with warnings':'Usable, with warnings',
    'has errors':'Has errors'}[d.verdict];
  const st=d.by_status;
  let h='<div class="verdict '+cls+'">'+esc(title)+'<span>'+esc(d.summary)+' Checked '+
    esc(d.checked)+'.</span></div>'+tiles([['rows',d.total_rows],
    ['usable rows',d.valid_rows,d.valid_rows<d.total_rows],
    ['from',d.date_from||'-'],['to',d.date_to||'-'],['pitches',d.pitches],
    ['errors',d.counts.error,d.counts.error>0],['warnings',d.counts.warning],
    ['confirmed',st.confirmed],['no-shows',st.no_show],['cancelled',st.cancelled],
    ['maintenance',st.maintenance],['blocked',st.blocked],['future',d.future_rows]]);
  if(!d.issues.length){out.innerHTML=h+'<p class="note">No problems found.</p>';return}
  h+='<div class="filters">'+[['all','All',d.issues.length],['error','Errors',d.counts.error],
    ['warning','Warnings',d.counts.warning],['info','Information',d.counts.info]]
    .map(([k,l,n])=>'<button class="ghost" data-f="'+k+'" aria-pressed="'+(filter===k)+'">'+
      l+' ('+n+')</button>').join('')+'</div>';
  const shown=d.issues.filter(i=>filter==='all'||i.severity===filter);
  h+=table([['severity','Severity','',r=>'<span class="sev '+esc(r.severity)+'">'+
      esc(r.severity)+'</span>'],
    ['rows','Row(s)','mono',r=>esc(r.rows.length>6?r.rows.slice(0,6).join(', ')+'...':
      r.rows.join(', '))],['field','Column','mono'],['message','What is wrong']],shown);
  out.innerHTML=h;
  out.querySelectorAll('[data-f]').forEach(b=>b.addEventListener('click',()=>{
    filter=b.dataset.f;showReport()}));
}
async function runCheck(body){
  const out=$('checkOut');out.innerHTML='<p class="muted">Checking...</p>';
  try{
    report=await call('/api/v1/bookings/check',{method:'POST',
      headers:{'Content-Type':'text/csv'},body});
    filter='all';showReport();
  }catch(e){out.innerHTML=errBox(e.message)}
}
$('csvFile').addEventListener('change',()=>{$('checkFile').disabled=!$('csvFile').files.length});
$('checkFile').addEventListener('click',async()=>{
  const f=$('csvFile').files[0];if(f)runCheck(await f.text());
});
$('checkSource').addEventListener('click',()=>runCheck(''));

const yest=new Date(Date.now()-864e5);
$('day').value=yest.toISOString().slice(0,10);
$('cmpBtn').addEventListener('click',async()=>{
  const out=$('cmpOut');out.innerHTML='<p class="muted">Loading...</p>';
  try{
    const d=await call('/api/v1/bookings/compare?day='+encodeURIComponent($('day').value));
    let h=d.message?'<p class="note">'+esc(d.message)+'</p>':'';
    if(d.slots.length){
      h+=table([['field_id','Pitch'],['start','Slot','mono',r=>esc(r.start+'-'+r.end)],
        ['camera','Cameras saw','',r=>r.camera?esc(r.camera)+(r.corrected?
          ' <span class="muted">(corrected)</span>':''):'<span class="muted">not yet</span>'],
        ['booking','Booking','',r=>r.booking?esc(r.booking):'<span class="muted">none</span>'],
        ['customer_ref','Reference','mono'],
        ['anomaly','Result','',r=>r.anomaly?'<span class="sev '+esc(r.severity)+'">'+
          esc(r.anomaly.replaceAll('_',' ').toLowerCase())+'</span>':''],
        ['explanation','Why']],d.slots);
    }
    if(d.unmatched_bookings.length){
      h+='<h3 style="font-size:14px;margin-top:18px">Bookings with no camera slot</h3>'+
        '<p class="sub2">Booked on this day, but the cameras were not scheduled to watch '+
        'that pitch at that time.</p>'+table([['field_id','Pitch'],['start','Start'],
        ['end','End'],['customer_ref','Reference'],['status','Status']],d.unmatched_bookings);
    }
    out.innerHTML=h||'<p class="note">Nothing to show for this day.</p>';
  }catch(e){out.innerHTML=errBox(e.message)}
});
loadSource();
"""

def bookings_html() -> str:
    """The whole page. Built per request: it is small, and `layout` is shared and may change."""
    return page("Pitch Occupancy - Bookings", "Booking data",
                _BODY.replace("EXAMPLES", _examples_html()),
                active="bookings", css=_CSS, script=_SCRIPT)


@router.get("/bookings", response_class=HTMLResponse, include_in_schema=False)
def bookings_page() -> HTMLResponse:
    return HTMLResponse(bookings_html())
