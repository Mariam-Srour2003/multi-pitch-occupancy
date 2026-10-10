"""The Reports section: what the live system recorded, added up for the client.

Three reports, each a JSON endpoint with a CSV twin (``format=csv``), and one page that
draws them:

* ``GET /api/v1/reports/summary`` - per pitch per day: slots used / not used / to review,
  hours played, utilisation, no-shows and unbooked play (`reports.daily_summary`).
* ``GET /api/v1/reports/busy-hours`` - weekday x hour share of watched time that was
  played (`reports.busy_hours`). Unwatched hours are "not observed", never 0%.
* ``GET /api/v1/reports/day`` - one recorded day cut into the client's own slot length,
  in the same table the video booking-day test gives (`reports.recorded_day`).

Everything here only reads the database; there is no POST. The connection comes from
`routes.get_conn`, so a test overrides one dependency for the whole API.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import date, time, timedelta
from html import escape
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

from pitch_occupancy import reports
from pitch_occupancy.api.layout import page
from pitch_occupancy.api.routes import get_conn

__all__ = ["router"]

router = APIRouter(tags=["reports"])

Conn = Annotated[sqlite3.Connection, Depends(get_conn)]

_HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_MAX_RANGE_DAYS = 366


def _range(start: date | None, end: date | None) -> tuple[date, date]:
    end = end or date.today()
    start = start or end - timedelta(days=6)
    if start > end:
        raise HTTPException(422, "The start date must be on or before the end date.")
    if (end - start).days >= _MAX_RANGE_DAYS:
        raise HTTPException(422, f"Choose a range of at most {_MAX_RANGE_DAYS} days.")
    return start, end


def _csv(body: str, filename: str) -> PlainTextResponse:
    return PlainTextResponse(body, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{filename}"'})


def _field(field_id: str | None) -> str | None:
    value = (field_id or "").strip()
    return value or None


def _span(start: date, end: date, field_id: str | None) -> str:
    suffix = f"-{re.sub(r'[^A-Za-z0-9_-]', '_', field_id)}" if field_id else ""
    return f"{start.isoformat()}_{end.isoformat()}{suffix}"


# --- API --------------------------------------------------------------------------------------


@router.get("/api/v1/reports/summary", response_model=None)
def summary_api(
    conn: Conn,
    start: date | None = None,
    end: date | None = None,
    field_id: str | None = Query(None, max_length=120),
    format: str = Query("json", pattern="^(json|csv)$"),
) -> dict | PlainTextResponse:
    """Per pitch per day between ``start`` and ``end`` (default: the last seven days)."""
    s, e = _range(start, end)
    fid = _field(field_id)
    data = reports.daily_summary(conn, start=s, end=e, field_id=fid)
    if format == "csv":
        return _csv(reports.summary_to_csv(data), f"pitch-summary_{_span(s, e, fid)}.csv")
    return data


@router.get("/api/v1/reports/busy-hours", response_model=None)
def busy_hours_api(
    conn: Conn,
    start: date | None = None,
    end: date | None = None,
    field_id: str | None = Query(None, max_length=120),
    format: str = Query("json", pattern="^(json|csv)$"),
) -> dict | PlainTextResponse:
    """Weekday x hour playing share; ``share`` is null where nothing was observed."""
    s, e = _range(start, end)
    fid = _field(field_id)
    data = reports.busy_hours(conn, start=s, end=e, field_id=fid)
    if format == "csv":
        return _csv(reports.busy_hours_to_csv(data), f"busy-hours_{_span(s, e, fid)}.csv")
    return data


@router.get("/api/v1/reports/day", response_model=None)
def day_api(
    conn: Conn,
    day: date,
    field_id: Annotated[str, Query(min_length=1, max_length=120)],
    first_slot: str = Query("08:00", max_length=8),
    slot_minutes: int = Query(90, ge=1, le=600),
    format: str = Query("json", pattern="^(json|csv)$"),
) -> dict | PlainTextResponse:
    """One pitch's recorded day cut into ``slot_minutes`` slots from ``first_slot``."""
    match = _HHMM.match(first_slot.strip())
    if not match:
        raise HTTPException(422, f"first_slot must be a clock time like 08:00, "
                                 f"not {first_slot!r}")
    first = time(int(match.group(1)), int(match.group(2)))
    fid = field_id.strip()
    rec = reports.recorded_day(conn, field_id=fid, day=day, first_slot=first,
                               slot_minutes=slot_minutes)
    if format == "csv":
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", fid)
        return _csv(reports.recorded_day_to_csv(rec),
                    f"recorded-day_{safe}_{day.isoformat()}_{first:%H%M}_{slot_minutes}min.csv")
    return rec.to_dict()


# --- the page ---------------------------------------------------------------------------------


_CSS = """
.slots{display:flex;gap:3px;flex-wrap:wrap;margin:6px 0 4px}
.slots div{flex:1 1 70px;min-width:62px;border-radius:6px;padding:7px 6px;font:600 11px
  'JetBrains Mono',monospace;text-align:center;border:1px solid var(--line)}
.slots .USED{background:var(--play);color:#fff;border-color:var(--play)}
.slots .NOT_USED{background:var(--empty-soft);color:var(--empty)}
.slots .REVIEW{background:var(--flag-soft);color:var(--flag)}
.slots small{display:block;font-weight:500;opacity:.85}
.pill.USED{background:var(--play-soft);color:var(--play)}
.pill.NOT_USED{background:var(--empty-soft);color:var(--empty)}
.pill.REVIEW{background:var(--flag-soft);color:var(--flag)}
.tl i.U{background:repeating-linear-gradient(45deg,var(--surface-2) 0 4px,var(--line) 4px 8px)}
.tl i.N{background:var(--surface-2)}
.tl i.P{background:var(--play)}.tl i.E{background:var(--empty-soft)}.tl i.M{background:var(--maint)}
.btnrow{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:14px 0 0}
a.act{font:600 13px Archivo,sans-serif;padding:9px 18px;border-radius:6px;
  border:1px solid var(--accent);color:var(--accent);text-decoration:none;background:var(--surface)}
td.reason{font-size:12.5px;color:var(--ink-2);min-width:200px}
.controls input,.controls select{width:auto;min-width:120px}
table.heat{min-width:720px;table-layout:fixed}
table.heat th,table.heat td{padding:0;text-align:center;font-size:10px}
table.heat th{padding:6px 2px}
table.heat th.wd{text-align:left;padding:0 8px;width:46px}
table.heat td{height:26px;border:1px solid var(--surface)}
table.heat td.no{background:repeating-linear-gradient(45deg,var(--surface-2) 0 3px,
  var(--surface) 3px 6px)}
.hide{display:none}
.empty{text-align:center;padding:30px 20px}
.empty h3{margin:0 0 6px}
"""

_BODY = """
<h2>Reports</h2>
<p class="sub2">What the cameras recorded, added up: how much each pitch was played, which
booking slots were used, no-shows and the busiest hours.</p>
__EMPTY__
<div id="reports"__HIDE__>
<div class="card">
  <div class="controls">
    <div><label for="from">From</label><input id="from" type="date" value="__START__"></div>
    <div><label for="to">To</label><input id="to" type="date" value="__END__"></div>
    <div><label for="pitch">Pitch</label><select id="pitch">
      <option value="">All pitches</option>__OPTIONS__</select></div>
    <div><button class="act" id="show">Show</button></div>
  </div>
  <p class="note" id="status"></p>
</div>

<h2>At a glance</h2>
<div class="tiles" id="tiles"></div>

<h2>Day by day</h2>
<div class="scroll"><table>
  <thead><tr><th>Date</th><th>Pitch</th><th class="num">Slots</th><th class="num">Used</th>
    <th class="num">Not used</th><th class="num">To review</th><th class="num">Hours played</th>
    <th class="num" title="Hours played out of hours scheduled">Utilisation</th>
    <th class="num">No-shows</th><th class="num">Unbooked play</th></tr></thead>
  <tbody id="days"></tbody></table></div>
<div class="btnrow"><a class="act" id="csvSummary" href="#">Download CSV</a></div>

<h2>Busy hours</h2>
<p class="sub2">How much of the watched time was being played, by day of the week and hour.
Darker is busier. Striped cells were not watched at all - that is "no data", not "empty".</p>
<div class="scroll"><table class="heat" id="heat"></table></div>
<div class="legend">
  <span><i class="sw" style="background:var(--surface)"></i>not played</span>
  <span><i class="sw" style="background:color-mix(in srgb,var(--play) 50%,var(--surface))">
    </i>played half the time</span>
  <span><i class="sw" style="background:var(--play)"></i>played all the time</span>
  <span><i class="sw" style="background:repeating-linear-gradient(45deg,var(--surface-2) 0 3px,
    var(--surface) 3px 6px)"></i>not observed</span>
</div>
<div class="btnrow"><a class="act" id="csvBusy" href="#">Download CSV</a></div>

<h2>Cut a recorded day into slots</h2>
<p class="sub2">Pick a pitch and a day, and say when your first booking slot starts and how
long your slots are. You get the same slot-by-slot table as the booking-day test, built from
what the cameras recorded that day.</p>
<div class="card">
  <div class="controls">
    <div><label for="cPitch">Pitch</label><select id="cPitch">__OPTIONS__</select></div>
    <div><label for="cDay">Date</label><input id="cDay" type="date" value="__END__"></div>
    <div><label for="cFirst">First slot starts at</label>
      <input id="cFirst" type="time" value="08:00"></div>
    <div><label for="cLen">Slot length (minutes)</label>
      <input id="cLen" type="number" min="1" max="600" value="90"></div>
    <div><button class="act" id="cut">Cut into slots</button></div>
  </div>
  <p class="note" id="cStatus"></p>
</div>
<section id="res" class="hide">
  <p class="sub2" id="summary" style="margin-top:14px"></p>
  <div class="tiles" id="cTiles"></div>
  <h2>Slots</h2>
  <div class="slots" id="slotbar"></div>
  <div class="legend">
    <span><i class="sw" style="background:var(--play)"></i>used</span>
    <span><i class="sw" style="background:var(--empty-soft)"></i>not used</span>
    <span><i class="sw" style="background:var(--flag-soft)"></i>please review</span>
  </div>
  <h2>Over the day</h2>
  <div class="tl" id="timeline"></div>
  <div class="axis" id="axis"></div>
  <div class="legend">
    <span><i class="sw" style="background:var(--play)"></i>playing</span>
    <span><i class="sw" style="background:var(--empty-soft)"></i>empty</span>
    <span><i class="sw" style="background:var(--maint)"></i>maintenance / not playing</span>
    <span><i class="sw" style="background:var(--line)"></i>could not tell</span>
    <span><i class="sw" style="background:var(--surface-2)"></i>not recorded</span>
  </div>
  <h2>Slot by slot</h2>
  <div class="scroll"><table>
    <thead><tr><th>Slot</th><th>Main activity</th><th class="num">Playing</th>
      <th class="num">Empty</th><th class="num">Maintenance</th><th class="num">Seen</th>
      <th>Result</th><th>Why</th></tr></thead>
    <tbody id="rows"></tbody></table></div>
  <div class="btnrow"><a class="act" id="csvDay" href="#">Download CSV</a></div>
</section>
</div>
"""

_EMPTY = """<div class="card empty" id="nodata"><h3>No recordings yet</h3>
<p>Reports are built from what the cameras record. Start recording from the
<a href="/cameras">Cameras page</a>, and the reports will fill in as slots are checked.</p>
</div>"""

_SCRIPT = r"""
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const RESULT={USED:'used',NOT_USED:'not used',REVIEW:'please review'};
const pct=x=>x==null?'-':Math.round(100*x)+'%';
const hrs=h=>(Math.round((h||0)*10)/10).toFixed(1);
const hm=s=>{s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);
  return h+':'+String(m).padStart(2,'0');};
function setStatus(el,msg,bad){$(el).className='note'+(bad?' warn':'');$(el).textContent=msg;}
async function detail(r){
  try{const d=await r.json();
    return typeof d.detail==='string'?d.detail:'Please check the dates and try again.';}
  catch(e){return 'The server answered '+r.status+'.';}
}
function filters(){
  const q=new URLSearchParams();
  if($('from').value)q.set('start',$('from').value);
  if($('to').value)q.set('end',$('to').value);
  if($('pitch').value)q.set('field_id',$('pitch').value);
  return q;
}
function tile(k,v,hot,title){return '<div class="tile'+(hot?' flagged':'')+'"'+
  (title?' title="'+esc(title)+'"':'')+'><div class="k">'+esc(k)+'</div><div class="v">'+
  esc(v)+'</div></div>';}

async function load(){
  const q=filters();setStatus('status','Loading...');
  try{
    const [rs,rb]=await Promise.all([fetch('/api/v1/reports/summary?'+q),
      fetch('/api/v1/reports/busy-hours?'+q)]);
    if(!rs.ok)throw new Error(await detail(rs));
    if(!rb.ok)throw new Error(await detail(rb));
    renderSummary(await rs.json());renderBusy(await rb.json());
    const qc=new URLSearchParams(q);qc.set('format','csv');
    $('csvSummary').href='/api/v1/reports/summary?'+qc;
    $('csvBusy').href='/api/v1/reports/busy-hours?'+qc;
    setStatus('status','');
  }catch(err){setStatus('status',err.message,true);}
}

function renderSummary(d){
  const t=d.totals;
  $('tiles').innerHTML=[
    tile('hours played',hrs(t.hours_played),false,'Minutes the cameras saw play, in hours'),
    tile('utilisation',pct(t.utilisation),false,'Hours played out of hours scheduled'),
    tile('slots used',t.used),tile('not used',t.not_used),
    tile('to review',t.review,t.review>0),
    tile('no-shows',t.no_shows,t.no_shows>0,'Booked, but nobody played'),
    tile('unbooked play',t.unbooked_play,t.unbooked_play>0,'Played, but not booked')].join('');
  if(!d.rows.length){$('days').innerHTML='<tr><td colspan="10" class="muted">No booking '+
    'slots in these dates.</td></tr>';return;}
  const row=(r,total)=>'<tr'+(total?' style="font-weight:600"':'')+'><td class="mono">'+
    esc(total?'Total':r.day)+'</td><td>'+esc(total?'':r.field_name)+'</td><td class="num">'+
    r.scheduled+'</td><td class="num">'+r.used+'</td><td class="num">'+r.not_used+
    '</td><td class="num">'+r.review+(r.not_checked?'<br><span class="muted">'+r.not_checked+
    ' not checked</span>':'')+'</td><td class="num">'+hrs(r.hours_played)+
    '</td><td class="num">'+pct(r.utilisation)+'</td><td class="num">'+r.no_shows+
    '</td><td class="num">'+r.unbooked_play+'</td></tr>';
  $('days').innerHTML=d.rows.map(r=>row(r,false)).join('')+row(t,true);
}

function renderBusy(d){
  const cells={};for(const c of d.cells)cells[c.weekday+'-'+c.hour]=c;
  let h='<thead><tr><th class="wd"></th>';
  for(let hr=0;hr<24;hr++)h+='<th>'+String(hr).padStart(2,'0')+'</th>';
  h+='</tr></thead><tbody>';
  d.weekdays.forEach((name,wd)=>{
    h+='<tr><th class="wd">'+esc(name.slice(0,3))+'</th>';
    for(let hr=0;hr<24;hr++){
      const c=cells[wd+'-'+hr],when=name+' '+String(hr).padStart(2,'0')+':00';
      if(!c||!c.observed){h+='<td class="no" title="'+esc(when+': not observed')+'"></td>';
        continue;}
      h+='<td style="background:color-mix(in srgb,var(--play) '+Math.round(100*c.share)+
        '%,var(--surface))" title="'+esc(when+': played '+pct(c.share)+' of '+
        c.observed_min+' minutes observed')+'"></td>';
    }
    h+='</tr>';
  });
  $('heat').innerHTML=h+'</tbody>';
}

async function cut(){
  if(!$('cPitch').value){setStatus('cStatus','Choose a pitch.',true);return;}
  if(!$('cDay').value||!$('cFirst').value){
    setStatus('cStatus','Choose a date and when the first slot starts.',true);return;}
  const q=new URLSearchParams({field_id:$('cPitch').value,day:$('cDay').value,
    first_slot:$('cFirst').value,slot_minutes:$('cLen').value||90});
  setStatus('cStatus','Working...');
  try{
    const r=await fetch('/api/v1/reports/day?'+q);
    if(!r.ok)throw new Error(await detail(r));
    const d=await r.json();
    q.set('format','csv');$('csvDay').href='/api/v1/reports/day?'+q;
    if(!d.slots.length){$('res').classList.add('hide');
      setStatus('cStatus','Nothing was recorded on this pitch on '+d.day+'.',true);return;}
    setStatus('cStatus','');renderDay(d);
  }catch(err){setStatus('cStatus',err.message,true);}
}

function renderDay(run){
  $('res').classList.remove('hide');
  const c=run.counts;
  $('summary').textContent=$('cPitch').selectedOptions[0].textContent+' on '+run.day+
    ' - '+run.slot_minutes+'-minute slots from '+run.first_slot+'. Recording runs from '+
    run.start+' for '+hm(run.duration_s)+' ('+run.minutes_recorded+' minutes recorded).';
  $('cTiles').innerHTML=[tile('slots',c.slots),tile('used',c.USED),tile('not used',c.NOT_USED),
    tile('please review',c.REVIEW,c.REVIEW>0)].join('');
  $('slotbar').innerHTML=run.slots.map(s=>'<div class="'+s.status+'" title="'+
    esc(s.label+': '+RESULT[s.status]+'. '+s.reason)+'">'+esc(s.start)+
    '<small>'+RESULT[s.status]+'</small></div>').join('');
  const total=Math.max(run.duration_s,1e-6),step=run.interval_s;
  $('timeline').innerHTML=run.timeline.map(([t,k])=>'<i class="'+k+'" style="width:'+
    (Math.max(0,Math.min(step,total-t))/total*100)+'%"></i>').join('');
  $('axis').innerHTML='<span>'+run.start+'</span><span>+'+hm(total/2)+'</span><span>+'+
    hm(total)+'</span>';
  $('rows').innerHTML=run.slots.map(s=>'<tr><td class="mono">'+esc(s.label)+
    (s.coverage<1?'<br><span class="muted">'+pct(s.coverage)+' recorded</span>':'')+
    '</td><td>'+esc(s.main_label)+'</td><td class="num">'+pct(s.playing_share)+
    '<br><span class="muted">'+s.playing_min+' min</span></td><td class="num">'+
    pct(s.empty_share)+'</td><td class="num">'+pct(s.maintenance_share)+
    '</td><td class="num">'+pct(s.seen_share)+'</td><td><span class="pill '+s.status+'">'+
    RESULT[s.status]+'</span></td><td class="reason">'+
    esc(s.reason.replace('outside the video','outside the recording'))+'</td></tr>').join('');
}

if($('show')){
  $('show').addEventListener('click',load);
  $('pitch').addEventListener('change',()=>{if($('pitch').value)$('cPitch').value=$('pitch').value;
    load();});
  $('cut').addEventListener('click',cut);
  load();
}
"""


@router.get("/reports", response_class=HTMLResponse, include_in_schema=False)
def reports_page(conn: Conn) -> HTMLResponse:
    fields = conn.execute(
        "SELECT field_id, name FROM fields ORDER BY name, field_id").fetchall()
    recorded = conn.execute(
        "SELECT EXISTS(SELECT 1 FROM frame_samples) OR EXISTS(SELECT 1 FROM slot_evaluations)"
    ).fetchone()[0]
    end = date.today()
    start = end - timedelta(days=6)
    options = "".join(
        f'<option value="{escape(f["field_id"])}">{escape(f["name"] or f["field_id"])}'
        "</option>" for f in fields)
    body = (_BODY.replace("__EMPTY__", "" if recorded else _EMPTY)
            .replace("__HIDE__", "" if recorded else ' class="hide"')
            .replace("__START__", start.isoformat()).replace("__END__", end.isoformat())
            .replace("__OPTIONS__", options))
    script = _SCRIPT if recorded else ""
    return HTMLResponse(page("Pitch Occupancy - Reports", "Reports", body, active="reports",
                             css=_CSS, script=script))

