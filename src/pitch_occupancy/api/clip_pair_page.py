"""The two-camera reviewer page. Served at `/clip/pair`; posts to `/api/v1/clip/pair/analyse`.

Its own page rather than a mode of `/clip`, so the single-camera reviewer keeps doing exactly
what it did. The style sheet is the clip page's own, cut out of `CLIP_HTML` rather than
copied, so the two cannot drift apart.

The layout puts the three answers on top of each other: camera A's own verdict, camera B's
own verdict, and the pitch's from the two together. That stacking is the point of the page.
A reader sees a moment where A says "3 people, not playing" and B says "2 people, not
playing" and the pitch says *playing*, and the row beneath says why - five people added, the
ball on B's side - and whether that answer would survive the two views overlapping.
"""

from __future__ import annotations

from pitch_occupancy.api.clip_page import CLIP_HTML
from pitch_occupancy.api.layout import JS_COMMON
from pitch_occupancy.api.nav import NAV_CSS, nav_html

__all__ = ["PAIR_HTML"]

#: The clip page's font links, reused for the same reason as its style sheet.
_FONTS = CLIP_HTML[CLIP_HTML.index('<link rel="preconnect"'):CLIP_HTML.index("<style>")]
_STYLE = CLIP_HTML[CLIP_HTML.index("<style>"):CLIP_HTML.index("</style>") + len("</style>")]

_EXTRA = """<style>
.pair{display:grid;gap:14px;grid-template-columns:1fr 1fr}
@media (max-width:700px){.pair{grid-template-columns:1fr}}
.drop .cam{font:700 12px 'JetBrains Mono',monospace;letter-spacing:.1em;color:var(--accent)}
.drop.has{border-style:solid;border-color:var(--accent)}
.rowlab{font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin:14px 0 5px}
.tl i.UNCERTAIN{border-right:1px solid var(--surface);
  background:repeating-linear-gradient(45deg,var(--surface-2) 0 4px,var(--line) 4px 8px)}
.tl i.blind{background:var(--flag-soft);border-right:1px solid var(--surface)}
.pill.UNCERTAIN{background:var(--surface-2);color:var(--ink-3)}
.flag{display:inline-block;font:600 10.5px 'JetBrains Mono',monospace;padding:2px 7px;
  border-radius:4px;margin:1px 3px 1px 0;background:var(--surface-2);color:var(--ink-2);
  white-space:nowrap}
.flag.hot{background:var(--flag-soft);color:var(--flag)}
.sum{font-weight:700}
.rules{display:grid;gap:8px;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));margin:0}
.rules div{background:var(--surface);border:1px solid var(--line);border-radius:8px;
  padding:11px 13px;font-size:13px;color:var(--ink-2)}
.rules b{color:var(--ink);display:block;margin-bottom:2px}
tr.open td{background:var(--accent-soft)}
tr.trace td{font:12px 'JetBrains Mono',monospace;color:var(--ink-2);background:var(--surface-2)}
.controls input[type=checkbox]{width:auto}
#status{min-height:20px}
</style>"""

PAIR_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pitch Occupancy - Two Cameras</title>
""" + _FONTS + _STYLE + _EXTRA + NAV_CSS + """
</head><body>
<header><div class="hin">
  <div class="brand">Pitch <span>Occupancy</span></div>
  <div class="sub">Testing &middot; two cameras</div>
  """ + nav_html("testing") + """
</div></header>
<main>
<h2>Both cameras of one pitch</h2>
<p class="sub2">Each camera sees about half the pitch. Here the people both cameras count are
added together, a ball seen by either camera counts for the pitch, and the rule is applied
once to the whole pitch. Nothing you upload is kept.</p>

<div class="pair">
  <label class="drop" id="dropA"><span class="cam">CAMERA A</span><br>
    <strong id="nameA">Drop or choose a video</strong><p>the first half of the pitch</p>
    <input type="file" accept="video/*" id="fileA" hidden></label>
  <label class="drop" id="dropB"><span class="cam">CAMERA B</span><br>
    <strong id="nameB">Drop or choose a video</strong><p>the other half, same time</p>
    <input type="file" accept="video/*" id="fileB" hidden></label>
</div>

<div class="controls">
  <div><label for="offset" title="If camera B's recording started later than A's, enter how
    many seconds later. Negative if B started first.">B starts after A (s)</label>
    <input id="offset" type="number" step="0.5" value="0"></div>
  <div><label for="interval">Every (s)</label>
    <input id="interval" type="number" min="1" step="1" value="10"></div>
  <div><label for="window" title="Smoothing window. 1 turns smoothing off.">Smoothing</label>
    <select id="window"><option value="1">1 - off</option>
      <option value="3" selected>3</option><option value="5">5</option></select></div>
  <div><label for="camA" title="Optional: a camera name with a saved boundary">A's name</label>
    <input id="camA" placeholder="optional"></div>
  <div><label for="camB" title="Optional: a camera name with a saved boundary">B's name</label>
    <input id="camB" placeholder="optional"></div>
  <div><label for="whole">Whole frame</label><input id="whole" type="checkbox"></div>
  <button class="act" id="go" disabled>Analyse both</button>
</div>
<p class="note" id="status"></p>

<section id="out" style="display:none">
  <h2>How the two cameras are combined</h2>
  <div class="rules" id="rules"></div>

  <h2>Summary</h2>
  <p class="sub2" id="summary"></p>
  <div class="tiles" id="tiles"></div>

  <h2>Timeline</h2>
  <p class="sub2" id="span"></p>
  <div class="rowlab">Pitch (A + B together)</div><div class="tl" id="tlP"></div>
  <div class="rowlab">Camera A on its own</div><div class="tl" id="tlA"></div>
  <div class="rowlab">Camera B on its own</div><div class="tl" id="tlB"></div>
  <div class="axis" id="axis"></div>
  <div class="legend">
    <span><i class="sw" style="background:var(--empty-soft)"></i>empty</span>
    <span><i class="sw" style="background:var(--play)"></i>playing</span>
    <span><i class="sw" style="background:var(--maint)"></i>people, not playing</span>
    <span><i class="sw" style="background:var(--line)"></i>uncertain</span>
    <span><i class="sw" style="background:var(--flag-soft)"></i>camera unreadable</span>
  </div>
  <div id="segs"></div>

  <h2>Every moment</h2>
  <p class="sub2">Click a row for the rule's reasoning.</p>
  <div class="scroll"><table>
    <thead><tr><th>A time</th><th>Camera A</th><th>Camera B</th><th>Together</th>
      <th>Pitch</th><th>Checks</th></tr></thead>
    <tbody id="rows"></tbody></table></div>
</section>
</main>
<script>""" + JS_COMMON + """const files={A:null,B:null};
const NAMES={C1_EMPTY:'empty',C2_ACTIVE_PLAY:'playing',
  C3_MAINTENANCE_NON_SPORTING:'not playing',UNCERTAIN:'uncertain'};
const clock=t=>{t=Math.round(t);return Math.floor(t/60)+':'+String(t%60).padStart(2,'0');};
const pill=s=>s?'<span class="pill '+s+'">'+NAMES[s]+'</span>'
  :'<span class="pill UNCERTAIN">unread</span>';

function choose(cam,file){
  if(!file)return;files[cam]=file;
  $('name'+cam).textContent=file.name+' ('+(file.size/1048576).toFixed(1)+' MB)';
  $('drop'+cam).classList.add('has');
  $('go').disabled=!(files.A&&files.B);
}
for(const cam of ['A','B']){
  const drop=$('drop'+cam),input=$('file'+cam);
  input.addEventListener('change',()=>choose(cam,input.files[0]));
  drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('over');});
  drop.addEventListener('dragleave',()=>drop.classList.remove('over'));
  drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('over');
    choose(cam,e.dataTransfer.files[0]);});
}

$('go').addEventListener('click',async()=>{
  $('go').disabled=true;$('out').style.display='none';
  $('status').className='note';
  $('status').textContent='Uploading both videos and analysing... the detector runs on '+
    'every moment of both cameras, so this can take a minute or two.';
  const q=new URLSearchParams({size_a:files.A.size,offset_s:$('offset').value||0,
    interval_s:$('interval').value||10,window:$('window').value,
    camera_a:$('camA').value,camera_b:$('camB').value,whole_frame:$('whole').checked});
  try{
    const r=await fetch('/api/v1/clip/pair/analyse?'+q,{method:'POST',
      body:new Blob([files.A,files.B]),headers:{'Content-Type':'application/octet-stream'}});
    const d=await r.json();
    if(!r.ok)throw new Error(typeof d.detail==='string'?d.detail:JSON.stringify(d.detail));
    $('status').textContent='';render(d);
  }catch(err){$('status').className='note err';$('status').textContent=err.message;}
  finally{$('go').disabled=!(files.A&&files.B);}
});

function bar(el,samples,stateOf,total,start,step){
  el.innerHTML=samples.map(s=>{
    const st=stateOf(s);
    const w=Math.max(0,Math.min(step,start+total-s.t_s))/total*100;
    return '<i class="'+(st||'blind')+(s.corrected&&el.id==='tlP'?' corr':'')+
      '" style="width:'+w+'%" title="'+clock(s.t_s)+' - '+(st?NAMES[st]:'unreadable')+'"></i>';
  }).join('');
}

function render(d){
  $('out').style.display='block';
  const R=d.rules;
  $('rules').innerHTML=
    '<div><b>People are added</b>camera A + camera B, counted inside each boundary. '+
      R.play_min+' or more together can be play; 1 to '+R.small_group_max+
      ' is people not playing.</div>'+
    '<div><b>The ball belongs to the pitch</b>seen by either camera counts. '+
      (R.require_ball?'The rule requires a ball for play.'
        :'The rule file does not require a ball for play right now (require_ball is off), '+
         'so the ball is reported, not required.')+'</div>'+
    '<div><b>Motion: the stronger half</b>a camera that could not be scored halves the '+
      'pitch confidence.</div>'+
    '<div><b>Boundaries</b>'+
      d.boundaries.map(b=>'camera '+b.camera+': '+esc(b.source)).join(' &middot; ')+
      '<br>model '+esc(d.model)+(R.frozen?'':' &middot; rule numbers not frozen yet')+'</div>';
  $('summary').textContent=d.summary+
    (d.interval_widened?' (interval widened to cover the whole overlap)':'');
  const n=d.n_samples||1,pct=x=>Math.round(100*x)+'%';
  const tiles=[
    ['moments',d.n_samples],
    ['playing',pct(d.shares.C2_ACTIVE_PLAY||0)],
    ['empty',pct(d.shares.C1_EMPTY||0)],
    ['not playing',pct(d.shares.C3_MAINTENANCE_NON_SPORTING||0)],
    ['needs the sum',d.n_depends_on_sum,d.n_depends_on_sum>0,
      'Neither camera alone had enough people, only the two added together. '+
      'If the two views overlap, someone may be counted twice.'],
    ['cameras disagree',d.n_split_view,false,
      'Each camera on its own gave a different answer. Normal sometimes; '+
      'all the time can mean a dirty lens or a blocked view.'],
    ['ball on one side',d.n_ball_one_side],
    ['camera unscored',d.n_one_side_blind,d.n_one_side_blind>0],
    ['A agrees w/ pitch',pct(d.agrees_with_pitch.A/n)],
    ['B agrees w/ pitch',pct(d.agrees_with_pitch.B/n)]];
  $('tiles').innerHTML=tiles.map(([k,v,hot,tip])=>'<div class="tile'+(hot?' flagged':'')+'"'+
    (tip?' title="'+esc(tip)+'"':'')+'><div class="k">'+k+'</div><div class="v">'+v+
    '</div></div>').join('');
  $('span').textContent='Sampled '+clock(d.start_s)+' to '+clock(d.end_s)+
    ' on camera A\\'s clock'+(d.offset_s?'; camera B\\'s own clock runs '+
    d.offset_s+' s behind it':'')+'.';
  const total=Math.max(d.end_s-d.start_s,1e-6);
  bar($('tlP'),d.samples,s=>s.smoothed,total,d.start_s,d.interval_s);
  bar($('tlA'),d.samples,s=>s.sides[0].state,total,d.start_s,d.interval_s);
  bar($('tlB'),d.samples,s=>s.sides[1].state,total,d.start_s,d.interval_s);
  $('axis').innerHTML='<span>'+clock(d.start_s)+'</span><span>'+
    clock((d.start_s+d.end_s)/2)+'</span><span>'+clock(d.end_s)+'</span>';
  $('segs').innerHTML='<p class="note">'+d.segments.map(g=>esc(g.describe)).join('<br>')+'</p>';

  const side=x=>!x.readable?'<span class="mono">unreadable</span>':
    pill(x.state)+' <span class="mono">'+(x.people==null?'-':x.people)+' ppl'+
    (x.ball?' &middot; ball':'')+'</span>';
  $('rows').innerHTML=d.samples.map((s,i)=>{
    const flags=[];
    if(s.depends_on_sum)flags.push('<span class="flag hot" title="If the cameras overlap, '+
      'the answer would be '+NAMES[s.if_overlapping]+'">needs the sum</span>');
    if(s.split_view)flags.push('<span class="flag">cameras disagree</span>');
    if(s.ball_one_side)flags.push('<span class="flag">ball on '+s.ball_sides.join('')+
      '</span>');
    if(s.one_side_blind)flags.push('<span class="flag hot">one camera unscored</span>');
    if(s.corrected)flags.push('<span class="flag hot">smoothed</span>');
    const pitch=(s.corrected?'<del>'+NAMES[s.raw]+'</del>':'')+pill(s.smoothed)+
      ' <span class="mono">'+s.confidence.toFixed(2)+'</span>';
    return '<tr data-i="'+i+'"'+(s.corrected?' class="corr"':'')+'><td class="mono">'+s.clock+
      (d.offset_s?'<br><span style="color:var(--ink-3)">B '+s.clock_b+'</span>':'')+'</td><td>'+
      side(s.sides[0])+'</td><td>'+side(s.sides[1])+'</td><td class="mono"><span class="sum">'+
      (s.people==null?'-':s.people)+'</span> ppl'+(s.ball?' &middot; ball':'')+'</td><td>'+pitch+
      '</td><td>'+(flags.join('')||'<span class="mono" style="color:var(--ink-3)">-</span>')+
      '</td></tr>';
  }).join('');
  $('rows').onclick=e=>{
    const tr=e.target.closest('tr[data-i]');if(!tr)return;
    const next=tr.nextElementSibling;
    if(next&&next.classList.contains('trace')){next.remove();tr.classList.remove('open');return;}
    const s=d.samples[+tr.dataset.i];
    tr.classList.add('open');
    tr.insertAdjacentHTML('afterend','<tr class="trace"><td colspan="6">'+
      s.trace.map(esc).join('<br>')+'</td></tr>');
  };
}
</script>
</body></html>
"""
