"""The clip reviewer page (WP6-T6). Served at `/clip`; posts to `/api/v1/clip/analyse`.

Kept apart from `dashboard.py` because it is a different job for a different moment: the
dashboard is the daily operator view over database state, this is a one-off "tell me what is
in this file" with nothing persisted on either side.

The layout is built around the one thing the reviewer has to judge and the model cannot:
**which corrections to accept.** Smoothing turns an isolated disagreeing sample into its
neighbours, which is right when the sample was a misread and wrong when it was a real brief
event. Both occur in the footage this was built against. So corrected samples are not
quietly replaced - the timeline marks them, the table strikes the raw prediction through
beside the corrected one, and a toggle switches the whole view back to raw. If the reviewer
never looks, they still see a count rather than a clean answer that hid its edits.
"""

from __future__ import annotations

__all__ = ["CLIP_HTML"]

CLIP_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pitch Occupancy - Clip Reviewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap">
<style>
:root{
  color-scheme:light;
  --ground:#f6f8f7;--surface:#fff;--surface-2:#eef2f1;--line:#dde4e2;
  --ink:#111817;--ink-2:#4d5c5a;--ink-3:#7a8886;
  --accent:#0d6d78;--accent-soft:#d7ebed;
  --play:#2c7a52;--play-soft:#dcefe3;--empty:#5c6a6d;--empty-soft:#e7ebea;
  --maint:#8a6d1f;--maint-soft:#f6ecd4;--flag:#a8402f;--flag-soft:#f6dfda;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --ground:#0e1414;--surface:#161e1e;--surface-2:#1c2625;--line:#2b3736;
  --ink:#eaf1ef;--ink-2:#a3b2af;--ink-3:#7b8a88;
  --accent:#4fb3bf;--accent-soft:#12363a;
  --play:#5cb884;--play-soft:#152f24;--empty:#8b9a97;--empty-soft:#222c2b;
  --maint:#cfae57;--maint-soft:#2c2614;--flag:#e08a76;--flag-soft:#331d18;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#0e1414;--surface:#161e1e;--surface-2:#1c2625;--line:#2b3736;
  --ink:#eaf1ef;--ink-2:#a3b2af;--ink-3:#7b8a88;
  --accent:#4fb3bf;--accent-soft:#12363a;
  --play:#5cb884;--play-soft:#152f24;--empty:#8b9a97;--empty-soft:#222c2b;
  --maint:#cfae57;--maint-soft:#2c2614;--flag:#e08a76;--flag-soft:#331d18;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:15px/1.55 Archivo,ui-sans-serif,system-ui,sans-serif}
header{border-bottom:1px solid var(--line);background:var(--surface)}
.hin{max-width:1080px;margin:0 auto;padding:16px 24px;display:flex;align-items:baseline;
  gap:14px;flex-wrap:wrap}
.brand{font-weight:700;letter-spacing:-.01em}.brand span{color:var(--accent)}
.hin .sub{color:var(--ink-3);font-size:13px}
.hin a{color:var(--accent);font-size:13px;text-decoration:none}
.hin a:first-of-type{margin-left:auto}
main{max-width:1080px;margin:0 auto;padding:26px 24px 70px}
h2{font-size:17px;margin:30px 0 4px;font-weight:600;letter-spacing:-.01em}
.sub2{color:var(--ink-2);font-size:13px;margin:0 0 14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px 20px}
.drop{border:1.5px dashed var(--line);border-radius:10px;padding:30px 20px;text-align:center;
  background:var(--surface);transition:border-color .15s,background .15s;cursor:pointer}
.drop.over{border-color:var(--accent);background:var(--accent-soft)}
.drop p{margin:6px 0 0;color:var(--ink-2);font-size:13px}
.drop strong{font-size:15px}
.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end;margin:16px 0 0}
.controls label{display:block;font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);margin-bottom:5px}
.controls input,.controls select{font:13px 'JetBrains Mono',monospace;padding:7px 10px;
  border:1px solid var(--line);border-radius:6px;background:var(--ground);color:var(--ink);
  width:110px}
button.act{font:600 13px Archivo,sans-serif;padding:9px 18px;border-radius:6px;cursor:pointer;
  border:1px solid var(--accent);background:var(--accent);color:#fff}
button.act[disabled]{opacity:.55;cursor:not-allowed}
button.ghost{font:600 12px Archivo,sans-serif;padding:6px 11px;border-radius:6px;cursor:pointer;
  border:1px solid var(--line);background:var(--surface);color:var(--ink-2)}
button.ghost[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);
  background:var(--accent-soft)}
.tiles{display:grid;gap:12px;margin:0 0 8px;
  grid-template-columns:repeat(auto-fit,minmax(140px,1fr))}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 15px}
.tile .k{font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin-bottom:5px}
.tile .v{font:700 24px 'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;
  letter-spacing:-.02em}
.tile.flagged{border-left:4px solid var(--flag)}
.tile.flagged .v{color:var(--flag)}
.tl{display:flex;height:44px;border-radius:8px;overflow:hidden;border:1px solid var(--line);
  background:var(--surface-2)}
.tl i{display:block;height:100%;position:relative}
.tl i.C1_EMPTY{background:var(--empty-soft);border-right:1px solid var(--surface)}
.tl i.C2_ACTIVE_PLAY{background:var(--play);border-right:1px solid var(--surface)}
.tl i.C3_MAINTENANCE_NON_SPORTING{background:var(--maint);border-right:1px solid var(--surface)}
.tl i.corr::after{content:"";position:absolute;inset:0;
  background:repeating-linear-gradient(45deg,transparent 0 3px,var(--flag) 3px 5px)}
.axis{display:flex;justify-content:space-between;font:11px 'JetBrains Mono',monospace;
  color:var(--ink-3);margin-top:5px}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0 0;font-size:12.5px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:13px;height:13px;border-radius:3px;display:inline-block;border:1px solid var(--line)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;min-width:620px}
th{font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-3);text-align:left;padding:11px 14px;border-bottom:1px solid var(--line);
  background:var(--surface-2);white-space:nowrap}
td{padding:10px 14px;border-bottom:1px solid var(--line);font-size:13.5px;vertical-align:middle}
tbody tr:last-child td{border-bottom:none}
tbody tr.corr{background:var(--flag-soft)}
.mono{font-family:'JetBrains Mono',monospace;font-size:12.5px}
.num{font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;text-align:right}
.pill{display:inline-block;font:600 11px 'JetBrains Mono',monospace;letter-spacing:.04em;
  padding:3px 9px;border-radius:5px;white-space:nowrap}
.pill.C1_EMPTY{background:var(--empty-soft);color:var(--empty)}
.pill.C2_ACTIVE_PLAY{background:var(--play-soft);color:var(--play)}
.pill.C3_MAINTENANCE_NON_SPORTING{background:var(--maint-soft);color:var(--maint)}
del{color:var(--ink-3);text-decoration-color:var(--flag);margin-right:7px}
.note{border-left:3px solid var(--accent);background:var(--surface);padding:12px 16px;
  border-radius:0 8px 8px 0;font-size:13px;color:var(--ink-2);margin:14px 0 0}
.note b{color:var(--ink)}
/* The pages carry their reasoning in `title` now rather than in paragraphs, so the places
   that hold one have to look hoverable - an explanation nobody can see they can ask for is
   an explanation that was deleted. `:empty` covers the notes the JS clears: with the prose
   gone, an emptied note would otherwise render as a bare bordered box. */
[title]{cursor:help}
figcaption[title],label[title],.sub2 span[title],.note span[title]{
  border-bottom:1px dotted currentColor}
.note:empty,.sub2:empty{display:none}
.warn{border-left-color:var(--flag)}
.err{border-left-color:var(--flag);color:var(--flag)}
#stage{display:none}#stage.on{display:block}
.stagebar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:0 0 12px;
  background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.stagebar .spacer{margin-left:auto}
.stagenow{font:600 12.5px 'JetBrains Mono',monospace;color:var(--accent)}
.stagebar input[type=range]{width:150px;accent-color:var(--accent)}
.panes{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.panes figure{margin:0;background:var(--surface);border:1px solid var(--line);
  border-radius:10px;overflow:hidden}
.panes img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover;background:var(--surface-2)}
.panes figcaption{font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3);padding:9px 13px;border-top:1px solid var(--line)}
.struck{text-decoration:line-through;opacity:.55}
.warnflag{display:inline-block;min-width:16px;text-align:center;border-radius:4px;
  background:#b45309;color:#fff;font-weight:700;font-size:11px;padding:0 5px;cursor:help}
.verdictbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:12px;
  background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:11px 14px}
.verdictbar .spacer{margin-left:auto}
.tile .v.sm{font-size:17px}
#busy{display:none;align-items:center;gap:10px;color:var(--ink-2);font-size:13px;margin-top:14px}
#busy.on{display:flex}
.spin{width:15px;height:15px;border:2px solid var(--line);border-top-color:var(--accent);
  border-radius:50%;animation:sp .7s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
#out{display:none}#out.on{display:block}
</style></head><body>
<header><div class="hin">
  <div class="brand">Pitch Occupancy<span>.</span></div>
  <div class="sub">Clip reviewer</div>
  <a href="/images">Image reviewer</a><a href="/client">Dashboard</a><a href="/">Findings</a>
</div></header>
<main>

<h2>Analyse a clip</h2>
<p class="sub2" title="Samples a frame at a fixed interval, classifies each one, and
reports when the state changed. The file is deleted as soon as it has been read.">sample
&rarr; classify &rarr; segment &middot; nothing stored</p>

<div class="drop" id="drop" tabindex="0" role="button" aria-label="Choose a video">
  <strong>Drop a video here, or click to choose</strong>
  <p id="chosen">MP4 or any format OpenCV can open &middot; up to 600 MB</p>
</div>
<input type="file" id="file" accept="video/*" hidden>

<div class="controls">
  <div><label for="iv">Sample every</label>
    <input type="number" id="iv" value="1" min="1" max="600" step="1"></div>
  <div><label for="win">Smoothing window</label>
    <select id="win">
      <option value="1">1 - none</option>
      <option value="3" selected>3 - fix single</option>
      <option value="5">5 - fix pairs</option>
      <option value="7">7 - fix triples</option>
    </select></div>
  <div><label for="expn">Explain in detail</label>
    <select id="expn">
      <option value="8">first 8 frames</option>
      <option value="20">first 20</option>
      <option value="-1" selected>every frame</option>
      <option value="0">none</option>
    </select></div>
  <div><label for="camera">Pitch boundary</label>
    <select id="camera" style="width:auto;min-width:148px">
      <option value="">whole frame</option>
    </select></div>
  <button class="act" id="go" disabled>Analyse</button>
  <button class="act" id="watch" disabled>Watch it work</button>
</div>
<p class="sub2" style="margin-top:9px" title="Masks everything outside this camera's pitch
on every sampled frame - the fix for a neighbouring pitch appearing in shot.">masks outside
the pitch &middot; <a href="/roi" style="color:var(--accent)">boundary editor</a></p>

<div id="busy"><div class="spin"></div><span id="busytxt">Analysing&hellip;</span></div>
<div id="err" class="note err" style="display:none"></div>

<div id="stage">
  <h2>Watching it work</h2>
  <p class="sub2" title="Each frame is sampled, embedded by the frozen backbone, scored by
    the linear probe, then put through the same two gates the Analyse tab applies - so the
    verdict here is the system's, not the probe's.">sample &rarr; embed &rarr; probe &rarr;
    gates</p>

  <div class="stagebar">
    <span class="stagenow" id="stage-step">waiting</span>
    <span class="spacer"></span>
    <label for="speed" style="font-size:12px;color:var(--ink-3)" title="A pause this page
      adds between steps. The backbone runs at the same speed either way - it does not make
      the model faster.">slow motion</label>
    <input type="range" id="speed" min="0" max="3000" step="100" value="1200">
    <span class="mono" id="speedtxt">1.2s</span>
    <button class="ghost" id="skip">Continue without slow motion</button>
  </div>
  <div class="panes">
    <figure><img id="img-raw" alt="sampled frame">
      <figcaption>the frame as sampled</figcaption></figure>
    <figure><img id="img-heat" alt="evidence map">
      <figcaption title="This map is not a saliency heuristic: the probe is linear over
        mean-pooled features, so the map is the summands of its score. The reconstruction
        error tile is that claim being checked.">where the score came from</figcaption></figure>
    <figure><img id="img-boxes" alt="What the detector found: a box round each person, a ring round the ball">
      <figcaption>What the detector found</figcaption></figure>
  </div>
  <p class="note" id="detnote" style="margin-top:8px" title="Two models, one frame. The probe scores pooled features, so its explanation can only be a heatmap; the detector finds objects, so its explanation is a box round each person and a ring round the ball. Anything found outside the boundary is dimmed, not dropped.">probe &rarr; heatmap &middot; detector &rarr; boxes &middot; outside the boundary dimmed</p>

  <div class="verdictbar" id="verdictbar">
    <span class="pill" id="v-pred">&mdash;</span>
    <span class="mono struck" id="v-probed"></span>
    <span class="mono" id="v-conf"></span>
    <span class="spacer"></span>
    <span class="mono" id="v-gate"></span>
    <span class="mono" id="v-ms"></span>
  </div>
  <p class="note" id="gatenote" style="display:none"></p>

  <div class="tiles" style="margin-top:14px">
    <div class="tile"><div class="k">Score from map</div>
      <div class="v sm" id="x-map">&mdash;</div></div>
    <div class="tile"><div class="k">Score direct</div>
      <div class="v sm" id="x-dir">&mdash;</div></div>
    <div class="tile"><div class="k">Reconstruction err</div>
      <div class="v sm" id="x-err">&mdash;</div></div>
    <div class="tile"><div class="k">People detected</div>
      <div class="v sm" id="x-ppl">&mdash;</div></div>
    <div class="tile" id="x-focus-tile"><div class="k">Evidence focus</div>
      <div class="v sm" id="x-focus">&mdash;</div></div>
    <div class="tile" id="x-out-tile"><div class="k">Evidence outside boundary</div>
      <div class="v sm" id="x-out">&mdash;</div></div>
  </div>
  <p class="sub2" id="focusnote" style="margin-top:10px"></p>

  <div class="tl" id="livetl" style="margin-top:16px"></div>
  <div class="axis"><span>0:00</span><span id="livecount"></span></div>
</div>

<div id="out">
  <h2>Timeline</h2>
  <p class="sub2" id="summary"></p>
  <div class="tl" id="tl"></div>
  <div class="axis"><span>0:00</span><span id="tend"></span></div>
  <div class="legend">
    <span><i class="sw" style="background:var(--play)"></i>active play</span>
    <span><i class="sw" style="background:var(--empty-soft)"></i>empty</span>
    <span><i class="sw" style="background:var(--maint)"></i>maintenance</span>
    <span><i class="sw" style="background:repeating-linear-gradient(45deg,transparent 0 3px,
      var(--flag) 3px 5px)"></i>corrected by neighbours</span>
  </div>

  <div class="tiles" style="margin-top:18px">
    <div class="tile"><div class="k">Duration</div><div class="v" id="m-dur"></div></div>
    <div class="tile"><div class="k">Samples</div><div class="v" id="m-n"></div></div>
    <div class="tile" id="m-corr-tile"><div class="k">Corrected</div>
      <div class="v" id="m-corr"></div></div>
    <div class="tile"><div class="k">Segments</div><div class="v" id="m-seg"></div></div>
  </div>

  <div id="bnote"></div>
  <div id="gnote"></div>
  <div id="corrnote"></div>

  <h2>Segments</h2>
  <p class="sub2" title="A boundary is reported as the interval it falls in: sampling
    locates a change to within one interval and no better.">resolution
    &plusmn;<span id="ivtxt"></span>s</p>
  <div class="scroll"><table><thead><tr>
    <th>State</th><th>From</th><th>To</th><th class="num">Duration</th>
    <th class="num">Samples</th><th class="num">Corrected</th>
  </tr></thead><tbody id="segs"></tbody></table></div>

  <h2>Every sample</h2>
  <p class="sub2"><del>struck</del> <span title="First strike: the probe's verdict before a
    gate overruled it. Second: this frame's verdict before its neighbours did.">overruled</span>
    &middot; <span title="Rows tinted red were overruled by their neighbours.">red row
    smoothed</span>
    &middot; <span class="warnflag">!</span> <span title="The verdict is empty and yet the
    detector found somebody. Worth a look: the motion threshold was fitted at one venue and
    calibrated nowhere else.">empty, yet people found</span></p>
  <div style="margin:0 0 10px">
    <button class="ghost" id="toggle" aria-pressed="false">Show raw predictions only</button>
  </div>
  <div class="scroll"><table><thead><tr>
    <th class="num">#</th><th>Time</th><th>Prediction</th><th class="num">Confidence</th>
    <th class="num">People inside</th>
    <th class="num">Motion</th>
  </tr></thead><tbody id="rows"></tbody></table></div>
</div>
</main>
<script>
const $=id=>document.getElementById(id);
const drop=$('drop'),file=$('file'),go=$('go');
let chosen=null,last=null,rawOnly=false;

function pick(f){
  if(!f) return;
  chosen=f;
  $('chosen').textContent=f.name+'  \\u00b7  '+(f.size/1048576).toFixed(1)+' MB';
  go.disabled=false;$('watch').disabled=false;
}
drop.onclick=()=>file.click();
drop.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();file.click();}};
file.onchange=e=>pick(e.target.files[0]);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');pick(e.dataTransfer.files[0])};

// Saved boundaries. At most three, because the derived store holds one per clip in the
// corpus and a hundred machine-generated ids is not a menu; the endpoint trims the list and
// nothing is deleted, so the masking experiments still see every one of them. "whole frame"
// stays first and stays selected, which is the behaviour the page had before boundaries
// existed - and is what a failure to load leaves behind too.
(async()=>{
  try{
    const b=await (await fetch('/api/v1/roi?limit=3')).json();
    (b.cameras||[]).forEach(k=>{
      const o=document.createElement('option');
      o.value=k;o.textContent=k+'  '+(b.boundaries[k].coverage*100).toFixed(0)+'%';
      $('camera').appendChild(o);
    });
  }catch(_){}
})();

go.onclick=async()=>{
  if(!chosen) return;
  go.disabled=true;$('busy').classList.add('on');$('err').style.display='none';
  $('out').classList.remove('on');
  $('busytxt').textContent='Analysing\u2026 the first run also loads '+
    'the model, which takes about 15 seconds.';
  const q=new URLSearchParams({interval_s:$('iv').value,window:$('win').value,
    camera:$('camera').value});
  try{
    const r=await fetch('/api/v1/clip/analyse?'+q,{method:'POST',body:chosen,
      headers:{'Content-Type':'application/octet-stream'}});
    const body=await r.json();
    if(!r.ok) throw new Error(body.detail||('HTTP '+r.status));
    last=body;render(body);
  }catch(e){
    $('err').textContent=e.message;$('err').style.display='block';
  }finally{
    go.disabled=false;$('busy').classList.remove('on');
  }
};

// --- the walkthrough ------------------------------------------------------------------
// Steps are rendered from a queue rather than as they arrive, because the two rates are
// different things: the network delivers a step the moment the backbone finishes it, and the
// reader wants to look at it for a beat. Slow motion is this pause and nothing else - the
// skip button empties the queue, it does not make the model faster, and the copy says so.
const NL=String.fromCharCode(10);
let queue=[],draining=false,delay=1200,streamDone=false;

$('speed').oninput=e=>{
  delay=+e.target.value;
  $('speedtxt').textContent=delay?(delay/1000).toFixed(1)+'s':'none';
};
$('skip').onclick=()=>{
  delay=0;$('speed').value=0;$('speedtxt').textContent='none';
};

const sleep=ms=>new Promise(r=>setTimeout(r,ms));

async function drain(){
  if(draining) return;
  draining=true;
  while(queue.length||!streamDone){
    if(!queue.length){await sleep(60);continue;}
    paint(queue.shift());
    if(delay) await sleep(delay);
  }
  draining=false;
  $('stage-step').textContent='finished';
  $('busy').classList.remove('on');
}

function paint(s){
  $('stage-step').textContent='step '+s.index+'  ·  '+s.clock;
  $('v-pred').textContent=s.predicted;
  $('v-pred').className='pill '+s.predicted;
  // The gated verdict is the headline and the probe's is struck through beside it, the way
  // the Analyse table shows them. The confidence belongs to the probe either way - it is the
  // probe's class probability and a gate does not produce one - so it is labelled as the
  // probe's whenever a gate spoke, rather than reading as the system's confidence in a
  // verdict the probe never gave.
  $('v-probed').textContent=s.probed?s.probed:'';
  $('v-conf').textContent=(s.probed?'probe confidence ':'confidence ')+s.confidence.toFixed(3);
  var saw=[];
  if(s.n_inside!==null&&s.n_inside!==undefined){
    saw.push(s.n_inside+' inside'+(s.ball?' + ball':''));
  }
  if(s.motion!==null&&s.motion!==undefined){saw.push('motion '+s.motion.toFixed(2));}
  $('v-gate').textContent=saw.join('  ·  ');
  var note=$('gatenote');
  if(s.probed){
    var why=(s.n_inside===0)
      ? 'nobody was inside the outline, and a pitch with nobody on it is not a match'
      : (s.n_inside>0 ? 'a small group with no ball is present but not playing'
                      : 'nothing moved between this frame and the previous sample');
    note.style.display='';
    // The map keeps decomposing the *probe*, not the gate, which is why an overruled frame
    // is the one worth looking at. That belongs on the map's own caption, not in a
    // paragraph the reader meets before the numbers.
    note.innerHTML='<b>gate overruled the probe</b> &middot; probe said '+s.probed+
      ' <span title="'+why+'">why?</span>';
  }else{note.style.display='none';}
  $('v-ms').textContent=Math.round(s.elapsed_ms)+' ms'+(s.explained?' (explained)':'');

  if(s.explained){
    $('img-raw').src=s.frame;$('img-heat').src=s.heat;
    // The detector's pane. It is served with the same record, so a frame that has a
    // heatmap has boxes too; if it ever does not, the pane is blanked rather than left
    // showing the previous frame's answer next to this frame's heatmap.
    $('img-boxes').src=s.boxes||'';
    const d=s.detector;
    // "not available" is not "found nobody", and the distinction survives the trim as a
    // word rather than a sentence: a blank count would read as zero people.
    $('detnote').innerHTML=!d?'detector &mdash; did not run'
      :(!d.checked?'detector &mdash; unavailable <span title="Not the same as finding '+
        'nobody.">(not zero)</span>'
      :('<b>'+d.state.split('_').slice(1).join(' ')+'</b> '+d.confidence.toFixed(2)+
        ' &middot; rule '+d.rule+' &middot; '+d.people_inside+' inside'+
        (d.people_outside_boundary?' &middot; '+d.people_outside_boundary+' outside':'')+
        ' &middot; ball '+(d.ball?d.ball_confidence.toFixed(2):'none')));
    $('x-map').textContent=s.score_from_map.toFixed(3);
    $('x-dir').textContent=s.score_direct.toFixed(3);
    $('x-err').textContent=s.reconstruction_error.toExponential(1);
    $('x-ppl').textContent=s.n_people;
    // Zero, or there is no boundary. Anything else means the outline reached the picture
    // but not the pooling, which is the exact failure this tile exists to make visible.
    const o=s.evidence_outside;
    $('x-out').textContent=o==null?'no boundary':(o*100).toFixed(1)+'%';
    $('x-out-tile').className='tile'+(o!=null&&o>0.001?' flagged':'');
    const f=s.focus_ratio;
    $('x-focus').textContent=f==null?'—':f.toFixed(2)+'x';
    $('x-focus-tile').className='tile'+(f!=null&&f<1?' flagged':'');
    $('focusnote').innerHTML=f==null
      ? '<span title="No people were detected, so there is no area to compare the evidence '+
        'against. A ratio over zero area is not a small number - it is not a number.">no '+
        'people &mdash; focus undefined</span>'
      : (f<1 ? '<span title="The score is spread as though the people were not there, which '+
               'for an ACTIVE_PLAY prediction is worth a look.">evidence spread off the '+
               'people</span>' : '');
  } else {
    $('x-map').textContent='—';$('x-dir').textContent='—';$('x-err').textContent='—';
    $('x-ppl').textContent='—';$('x-focus').textContent='—';
    $('x-out').textContent='—';
    $('x-focus-tile').className='tile';$('x-out-tile').className='tile';
    $('focusnote').innerHTML='<span title="explain in detail bounds how many frames get '+
      'a map, because explaining costs about twice a bare prediction">no evidence map '+
      'for this frame</span>';
  }

  const i=document.createElement('i');
  i.className=s.predicted;i.style.width='14px';i.title=s.clock+'  '+s.predicted;
  if(!s.explained) i.style.opacity='.55';
  $('livetl').appendChild(i);
  $('livecount').textContent=(s.index+1)+' steps';
}

$('watch').onclick=async()=>{
  if(!chosen) return;
  queue=[];streamDone=false;delay=+$('speed').value;
  $('livetl').innerHTML='';$('err').style.display='none';
  $('out').classList.remove('on');$('stage').classList.add('on');
  $('go').disabled=$('watch').disabled=true;
  $('busy').classList.add('on');
  $('busytxt').textContent='Streaming… the first step also loads the model, '+
    'which takes about 15 seconds.';
  $('stage-step').textContent='loading the model';

  const q=new URLSearchParams({interval_s:$('iv').value,explain_n:$('expn').value,
    camera:$('camera').value});
  try{
    const r=await fetch('/api/v1/clip/walkthrough?'+q,{method:'POST',body:chosen,
      headers:{'Content-Type':'application/octet-stream'}});
    if(!r.ok) throw new Error('HTTP '+r.status);
    drain();
    const reader=r.body.getReader(),dec=new TextDecoder();
    let buf='';
    for(;;){
      const {done,value}=await reader.read();
      if(done) break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split(NL);buf=lines.pop();
      for(const ln of lines){
        if(!ln.trim()) continue;
        const d=JSON.parse(ln);
        if(d.type==='step') queue.push(d);
        else if(d.type==='error') throw new Error(d.detail);
      }
    }
  }catch(e){
    $('err').textContent=e.message;$('err').style.display='block';
  }finally{
    streamDone=true;
    $('go').disabled=$('watch').disabled=false;
  }
};

$('toggle').onclick=()=>{
  rawOnly=!rawOnly;
  $('toggle').setAttribute('aria-pressed',String(rawOnly));
  $('toggle').textContent=rawOnly?'Show corrected predictions':'Show raw predictions only';
  if(last) render(last);
};

const clock=t=>Math.floor(t/60)+':'+String(Math.round(t%60)).padStart(2,'0');

function render(d){
  $('out').classList.add('on');
  $('summary').textContent=d.summary;
  $('tend').textContent=clock(d.duration_s);
  $('m-dur').textContent=clock(d.duration_s);
  $('m-n').textContent=d.n_samples;
  $('m-corr').textContent=d.n_corrected;
  $('m-seg').textContent=d.segments.length;
  $('m-corr-tile').className='tile'+(d.n_corrected?' flagged':'');
  $('ivtxt').textContent=d.interval_s;

  const tl=$('tl');tl.innerHTML='';
  const w=100/Math.max(1,d.samples.length);
  for(const s of d.samples){
    const i=document.createElement('i');
    i.className=(rawOnly?s.raw:s.smoothed)+(s.corrected&&!rawOnly?' corr':'');
    i.style.width=w+'%';
    i.title=s.clock+'  '+(rawOnly?s.raw:s.smoothed)+(s.corrected?'  (was '+s.raw+')':'');
    tl.appendChild(i);
  }

  // Three states, and the warn one stays loud: a reader who does not notice that no outline
  // was applied reads an over-report as a result.
  const bnote=$('bnote');
  if(d.boundary_derived){
    bnote.className='note';
    bnote.innerHTML='outline <b>measured from the footage</b> <span title="No stored '+
      'outline for this camera. The median of the first frames is thresholded for turf and '+
      'its outline used as the pitch - the same routine the corpus uses. A hand-drawn one '+
      'is better; what this replaces is no outline at all.">why?</span>';
  } else if(d.boundary){
    bnote.className='note';
    bnote.innerHTML='outline <b>'+d.camera+'</b>';
  } else {
    bnote.className='note warn';
    bnote.innerHTML='<b>no pitch outline</b> &middot; <span title="Every pixel counts, '+
      'including the next pitch over and anyone walking past.">active play '+
      'over-reported</span>';
  }

  const gnote=$('gnote');
  if(d.n_gated){
    gnote.className='note';
    gnote.innerHTML='<b>'+d.n_gated+'</b> weakened by the gates <span title="A play verdict '+
      'with nobody inside the outline becomes empty, and a small group with no ball becomes '+
      'not-playing. The gates only ever weaken a claim, never strengthen one - the table '+
      'shows what the model said before each.">why?</span>';
  } else { gnote.className='';gnote.innerHTML=''; }

  // The one claim on this page that must not be trimmed away: a smoothed timeline is a
  // judgement, not the answer. It keeps a count on the page and its reason a hover away -
  // see `test_the_page_is_served_and_names_what_smoothing_costs`.
  const note=$('corrnote');
  if(d.n_corrected&&!rawOnly){
    note.className='note warn';
    note.innerHTML='<b>'+d.n_corrected+'</b> changed by smoothing <span title="A lone '+
      'disagreeing sample between two agreeing neighbours is usually a misread frame - but '+
      'it can equally be a real brief event, and at this sampling interval nothing in the '+
      'samples can tell those apart. Check the striped bands against the footage, or set '+
      'the window to 1 to turn smoothing off.">judgement, not an answer</span>';
  } else if(d.interval_widened){
    note.className='note warn';
    note.innerHTML='<b>interval widened to '+d.interval_s.toFixed(1)+'s</b> <span '+
      'title="So the whole clip is covered within the sample cap, rather than analysing '+
      'only its beginning.">why?</span>';
  } else { note.className='';note.innerHTML=''; }

  const segs=$('segs');segs.innerHTML='';
  for(const g of d.segments){
    const from=g.starts_after===g.starts_by?clock(g.starts_by)
      :clock(g.starts_after)+'\\u2013'+clock(g.starts_by);
    segs.insertAdjacentHTML('beforeend','<tr><td><span class="pill '+g.state+'">'+g.label+
      '</span></td><td class="mono">'+from+'</td><td class="mono">'+clock(g.ends_by)+
      '</td><td class="num">'+clock(g.duration_s)+'</td><td class="num">'+g.n_samples+
      '</td><td class="num">'+(g.n_corrected||'')+'</td></tr>');
  }

  const rows=$('rows');rows.innerHTML='';
  for(const s of d.samples){
    const shown=rawOnly?s.raw:s.smoothed;
    const was=(s.corrected&&!rawOnly)?'<del>'+s.raw+'</del>':'';
    // Two different overrules, kept apart on purpose: `probed` is what the probe said before
    // a gate, `corrected` is what the neighbours did afterwards. A reviewer chasing one
    // should never be handed the other.
    const gate=s.gated?'<del>'+s.probed+'</del> ':'';
    // A dash here now means the detector could not be loaded at all. The review pages count
    // on every frame (A38), so "not checked" is no longer a thing a reader has to guess at.
    const ppl=(s.people===null||s.people===undefined)?'\u2014'
      :(s.people+(s.ball?' + ball':''));
    const mot=(s.motion===null||s.motion===undefined)?'\u2014':s.motion.toFixed(2);
    // The verdict stands - the gates only weaken - so this flags the row rather than
    // changing it. It is the motion threshold and the detector disagreeing about one frame.
    const flag=s.gates_disagree
      ? ' <span class="warnflag" title="the verdict is empty and yet the detector found '+
        s.people+' inside the outline">!</span>' : '';
    rows.insertAdjacentHTML('beforeend','<tr'+((s.corrected&&!rawOnly)?' class="corr"':'')+
      '><td class="num">'+s.index+'</td><td class="mono">'+s.clock+'</td><td class="mono">'+
      gate+was+'<span class="pill '+shown+'">'+shown+'</span>'+flag+'</td><td class="num">'+
      s.confidence.toFixed(3)+'</td><td class="num">'+ppl+'</td>'+
      '<td class="num mono">'+mot+'</td></tr>');
  }
}
</script></body></html>
"""
