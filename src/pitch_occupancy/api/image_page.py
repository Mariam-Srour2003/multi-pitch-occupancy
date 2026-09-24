"""The image reviewer page (WP6-T6). Served at `/images`; posts to `/api/v1/images/walkthrough`.

The still-image counterpart to `clip_page.py`, and the layout differs from it in one way that
matters. The clip page is built around *time*: a timeline, segments, and the reviewer's
judgement about which smoothing corrections to accept. None of that exists for a folder of
stills - they have no order, no neighbours and no duration - so building the same page with
the time parts greyed out would be a worse page, not a smaller one.

What a batch of stills has instead is **comparison**, so that is what this is built around.
Every image is explained the same way, and the gallery at the end sorts them by confidence
with the least confident first - because on a batch the useful question is not "what is the
verdict" but "which of these should I look at myself". A grid in upload order would bury
exactly the images the reader opened the page for.

The live stage above it is the clip walkthrough's, unchanged in spirit: one image at a time,
the frame beside its evidence map, the score decomposition underneath, arriving as the
backbone finishes each one.
"""

from __future__ import annotations

__all__ = ["IMAGE_HTML"]

IMAGE_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pitch Occupancy - Image Reviewer</title>
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
.sub2{color:var(--ink-2);font-size:13px;margin:0 0 14px;max-width:74ch}
.drop{border:1.5px dashed var(--line);border-radius:10px;padding:30px 20px;text-align:center;
  background:var(--surface);transition:border-color .15s,background .15s;cursor:pointer}
.drop.over{border-color:var(--accent);background:var(--accent-soft)}
.drop p{margin:6px 0 0;color:var(--ink-2);font-size:13px}
.drop strong{font-size:15px}
.picked{display:flex;flex-wrap:wrap;gap:7px;margin:12px 0 0}
.picked figure{margin:0;width:78px;border:1px solid var(--line);border-radius:7px;
  overflow:hidden;background:var(--surface)}
.picked img{display:block;width:100%;height:52px;object-fit:cover;background:var(--surface-2)}
.picked figcaption{font:500 9px 'JetBrains Mono',monospace;color:var(--ink-3);padding:3px 5px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end;margin:16px 0 0}
.controls label{display:block;font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);margin-bottom:5px}
.controls input,.controls select{font:13px 'JetBrains Mono',monospace;padding:7px 10px;
  border:1px solid var(--line);border-radius:6px;background:var(--ground);color:var(--ink);
  width:120px}
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
.tile .v.sm{font-size:17px}
.tile.flagged{border-left:4px solid var(--flag)}
.tile.flagged .v{color:var(--flag)}
.pill{display:inline-block;font:600 11px 'JetBrains Mono',monospace;letter-spacing:.04em;
  padding:3px 9px;border-radius:5px;white-space:nowrap}
.pill.C1_EMPTY{background:var(--empty-soft);color:var(--empty)}
.pill.C2_ACTIVE_PLAY{background:var(--play-soft);color:var(--play)}
.pill.C3_MAINTENANCE_NON_SPORTING{background:var(--maint-soft);color:var(--maint)}
.note{border-left:3px solid var(--accent);background:var(--surface);padding:12px 16px;
  border-radius:0 8px 8px 0;font-size:13px;color:var(--ink-2);margin:14px 0 0;max-width:78ch}
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
.err{border-left-color:var(--flag);color:var(--flag);display:none}
#stage{display:none}#stage.on{display:block}
.stagebar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:0 0 12px;
  background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.stagebar .spacer{margin-left:auto}
.stagenow{font:600 12.5px 'JetBrains Mono',monospace;color:var(--accent)}
.stagebar input[type=range]{width:150px;accent-color:var(--accent)}
.panes{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.panes figure{margin:0;background:var(--surface);border:1px solid var(--line);
  border-radius:10px;overflow:hidden}
.panes img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;
  background:var(--surface-2)}
.panes figcaption{font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3);padding:9px 13px;border-top:1px solid var(--line)}
.verdictbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:12px;
  background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:11px 14px}
.verdictbar .spacer{margin-left:auto}
/* The decision table, drawn from the rows the server sends rather than typed here: the
   thresholds in them are `play_min` and `small_group_max` out of configs/rules.json. */
.rules{margin-top:12px;background:var(--surface);border:1px solid var(--line);
  border-radius:10px;padding:13px 15px}
.rules h3{margin:0 0 9px;font:600 12px Archivo,sans-serif;letter-spacing:.01em}
.rules table{border-collapse:collapse;width:100%}
.rules td{padding:5px 8px;font-size:12.5px;color:var(--ink-2);border-bottom:1px solid var(--line)}
.rules tr:last-child td{border-bottom:none}
.rules td.n{font:600 11px 'JetBrains Mono',monospace;color:var(--ink-3);width:26px}
.rules td.s{font:600 10.5px 'JetBrains Mono',monospace;text-align:right;white-space:nowrap}
.rules tr.fired td{background:var(--play-soft);color:var(--ink)}
.rules tr.fired td.n,.rules tr.fired td.s{color:var(--play)}
/* A row that cannot fire on a still is dimmed and struck, never simply absent: a reader who
   sees five live rows concludes the missing two were checked and did not match. */
.rules tr.off td{color:var(--ink-3);opacity:.62}
.rules tr.off td.c{text-decoration:line-through}
.cues{display:flex;gap:7px;flex-wrap:wrap;margin:0 0 10px}
.cue{font:500 11px 'JetBrains Mono',monospace;padding:4px 9px;border-radius:6px;
  background:var(--surface-2);color:var(--ink-2);border:1px solid var(--line)}
.cue.ok{background:var(--play-soft);color:var(--play);border-color:transparent}
.cue.na{background:var(--maint-soft);color:var(--maint);border-color:transparent}
.split{font-size:12.5px;color:var(--maint);margin-top:9px}
.split:empty{display:none}
.trace{margin:9px 0 0;padding-left:17px}
.trace li{font-size:12px;color:var(--ink-3);margin-bottom:3px}
.trace li.skip{color:var(--maint)}
.gallery{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(212px,1fr))}
.shot{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  overflow:hidden;display:flex;flex-direction:column}
.shot.flagged{border-left:4px solid var(--flag)}
.shot .imgs{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}
.shot .imgs img{display:block;width:100%;aspect-ratio:4/3;object-fit:cover;
  background:var(--surface-2);cursor:zoom-in}
.shot .body{padding:10px 12px 12px;display:flex;flex-direction:column;gap:7px;flex:1}
.shot .fname{font:500 11px 'JetBrains Mono',monospace;color:var(--ink-3);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.shot .row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.shot .row .spacer{margin-left:auto}
.shot .m{font:500 11.5px 'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;
  color:var(--ink-2)}
.shot .m b{color:var(--ink)}
.shot .m.bad b{color:var(--flag)}
#busy{display:none;align-items:center;gap:10px;color:var(--ink-2);font-size:13px;margin-top:14px}
#busy.on{display:flex}
.spin{width:15px;height:15px;border:2px solid var(--line);border-top-color:var(--accent);
  border-radius:50%;animation:sp .7s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion:reduce){.spin{animation:none}}
#out{display:none}#out.on{display:block}
dialog{border:1px solid var(--line);border-radius:12px;background:var(--surface);padding:0;
  max-width:92vw;max-height:92vh}
dialog::backdrop{background:rgba(0,0,0,.62)}
dialog img{display:block;max-width:88vw;max-height:82vh}
dialog .close{position:absolute;top:8px;right:10px}
</style></head><body>
<header><div class="hin">
  <div class="brand">Pitch Occupancy<span>.</span></div>
  <div class="sub">Image reviewer</div>
  <a href="/clip">Clip reviewer</a><a href="/client">Dashboard</a><a href="/">Findings</a>
</div></header>
<main>

<h2>Explain one image, or a batch</h2>
<p class="sub2" title="The same evidence decomposition the clip walkthrough uses, on
pictures instead of footage. Images are decoded in memory and the copies used for reading
are deleted as soon as the batch finishes.">classify &rarr; explain &middot; nothing
stored</p>

<div class="drop" id="drop" tabindex="0" role="button" aria-label="Choose images">
  <strong>Drop images here, or click to choose</strong>
  <p id="chosen">JPEG, PNG, or anything OpenCV can read &middot; up to 32 images,
  25 MB each</p>
</div>
<input type="file" id="file" accept="image/*" multiple hidden>
<div class="picked" id="picked"></div>

<div class="controls">
  <div><label for="expn">Explain in detail</label>
    <select id="expn">
      <option value="-1" selected>every image</option>
      <option value="1">first 1</option>
      <option value="4">first 4</option>
      <option value="8">first 8</option>
    </select></div>
  <div><label for="speed" title="A pause this page adds between renders - the model runs
      at the same speed either way. The control that changes the actual work is explain in
      detail: an explained image costs roughly twice a bare prediction.">Slow motion</label>
    <input type="range" id="speed" min="0" max="2500" step="250" value="900"
      style="width:120px"></div>
  <div><label for="camera">Pitch boundary</label>
    <select id="camera" style="width:auto;min-width:150px">
      <option value="">whole frame</option>
    </select></div>
  <div><button class="ghost" id="redact" aria-pressed="false">Pixelate people</button></div>
  <button class="act" id="go" disabled>Explain these</button>
</div>
<p class="sub2" style="margin-top:9px" title="Masks everything outside the camera's pitch -
the fix for a neighbouring pitch showing up in frame.">masks outside the pitch &middot;
<a href="/roi" style="color:var(--accent)">boundary editor</a></p>

<div class="note" id="err"></div>

<div id="busy"><span class="spin"></span><span id="busytxt"></span></div>

<section id="stage">
  <h2>Watching it work</h2>
  <div class="stagebar">
    <span class="stagenow" id="stage-step">idle</span>
    <span class="mono" id="stage-name" style="color:var(--ink-3);font-size:12px"></span>
    <span class="spacer"></span>
    <span class="mono" id="livecount" style="font-size:12px;color:var(--ink-3)">0 images</span>
    <button class="ghost" id="skip">Continue without slow motion</button>
  </div>

  <div class="panes">
    <figure><img id="img-raw" alt="The image as the model received it">
      <figcaption>The image</figcaption></figure>
    <figure><img id="img-heat" alt="Evidence map over the image: where the score came from">
      <figcaption title="Exact rather than a saliency heuristic: the per-position
        contributions plus a constant sum to the score. The first three tiles are that claim
        being checked - if the error is not tiny, the decomposition is
        wrong.">Where the score came from</figcaption></figure>
    <figure><img id="img-boxes" alt="What the detector found: a box round each person, a ring round the ball">
      <figcaption>What the detector found</figcaption></figure>
  </div>
  <p class="note" id="detnote" style="margin-top:8px" title="Two models, one frame. The
    probe scores pooled features, so its explanation can only be a heatmap; the detector
    finds objects, so its explanation is a box round each person and a ring round the ball.
    Anything found outside the boundary is dimmed, not dropped.">probe &rarr; heatmap
    &middot; detector &rarr; boxes &middot; outside the boundary dimmed</p>

  <div class="verdictbar">
    <span class="pill" id="v-pred">&mdash;</span>
    <span class="mono" id="v-conf" style="font-size:12.5px;color:var(--ink-2)"></span>
    <span class="spacer"></span>
    <span class="mono" id="v-ms" style="font-size:12.5px;color:var(--ink-3)"></span>
  </div>

  <div class="tiles" style="margin-top:12px">
    <div class="tile"><div class="k">People on the pitch</div>
      <div class="v sm" id="x-ppl">&mdash;</div></div>
    <div class="tile" id="x-outside-tile"><div class="k">People outside it</div>
      <div class="v sm" id="x-outside">&mdash;</div></div>
    <div class="tile"><div class="k">Ball</div>
      <div class="v sm" id="x-ball">&mdash;</div></div>
    <div class="tile" id="x-out-tile"><div class="k">Evidence outside boundary</div>
      <div class="v sm" id="x-out">&mdash;</div></div>
  </div>
  <div class="note" id="focusnote"></div>

  <div class="rules" id="rules" hidden>
    <h3>The rules, applied to this image <span class="mono" id="r-det"
      style="font-weight:400;color:var(--ink-3)"></span></h3>
    <div class="cues" id="r-cues"></div>
    <table><tbody id="r-rows"></tbody></table>
    <div class="split" id="r-split"></div>
    <ol class="trace" id="r-trace"></ol>
  </div>
</section>

<section id="out">
  <h2>The batch</h2>
  <p class="sub2" title="On a batch the useful question is not what the verdict is but which
    of these to look at yourself.">least confident first &middot; click an image to
    enlarge</p>
  <div class="tiles" id="sum"></div>
  <div class="note warn" id="alone" title="The clip reviewer can overrule an isolated
    misread using the samples either side of it in time; these stills may be minutes or
    venues apart, and nothing in the upload says which, so the evidence map is the only
    thing a prediction can be judged by."><b>Each image stands on its own</b> &middot; no
    neighbours &middot; nothing smoothed or corrected</div>
  <div class="gallery" id="gallery" style="margin-top:14px"></div>
</section>

<dialog id="zoom"><button class="ghost close" id="zclose">Close</button>
  <img id="zimg" alt="Enlarged view"></dialog>

</main>
<script>
const $=id=>document.getElementById(id);
const NL=String.fromCharCode(10);
let files=[],results=[],queue=[],draining=false,delay=900,streamDone=false,redact=false;

// --- choosing files -------------------------------------------------------------------
function pick(list){
  files=[...list].filter(f=>f.type.startsWith('image/')).slice(0,32);
  $('chosen').textContent=files.length
    ? files.length+(files.length===1?' image':' images')+' chosen'
    : 'JPEG, PNG, or anything OpenCV can read \\u00b7 up to 32 images, 25 MB each';
  $('go').disabled=!files.length;
  const box=$('picked');box.innerHTML='';
  files.forEach(f=>{
    const fig=document.createElement('figure');
    const img=document.createElement('img');
    img.src=URL.createObjectURL(f);
    // Revoked once decoded: 32 object URLs held for the life of the page is a leak that
    // only shows up on the batch sizes this tool is for.
    img.onload=()=>URL.revokeObjectURL(img.src);
    const cap=document.createElement('figcaption');cap.textContent=f.name;cap.title=f.name;
    fig.append(img,cap);box.appendChild(fig);
  });
}
// Saved boundaries, so a batch can be masked to one pitch. At most three: the derived store
// holds one entry per clip in the corpus, and the endpoint trims the menu rather than anything
// being deleted, so the masking experiments still see every one of them. Failing to load them
// leaves the selector at "whole frame", which is the behaviour the page had before boundaries
// existed and is also its default now.
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

$('file').onchange=e=>pick(e.target.files);
$('drop').onclick=()=>$('file').click();
$('drop').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('file').click();}};
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('over')};
$('drop').ondragleave=()=>$('drop').classList.remove('over');
$('drop').ondrop=e=>{e.preventDefault();$('drop').classList.remove('over');
  pick(e.dataTransfer.files)};

$('redact').onclick=()=>{
  redact=!redact;$('redact').setAttribute('aria-pressed',String(redact));
};
$('speed').oninput=e=>{delay=+e.target.value};
$('skip').onclick=()=>{delay=0;$('speed').value=0};

const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const readAsDataURL=f=>new Promise((res,rej)=>{
  const r=new FileReader();r.onload=()=>res(r.result);r.onerror=()=>rej(r.error);
  r.readAsDataURL(f);
});

// --- rendering ------------------------------------------------------------------------
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
  summarise();
}

// Sent once with the meta record; see `vision/rules.rule_table`.
let RULES=[],CLAUSES=[],DETECTOR='';

const esc=t=>String(t).replace(/[&<>"]/g,c=>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
// C2_ACTIVE_PLAY -> ACTIVE PLAY. The prefix is a folder name, not something to read.
const stateName=v=>String(v).split('_').slice(1).join(' ')||String(v);
// Trace lines reporting a clause that could not be checked, as against one that failed.
// They take the same shape every time because `decide` writes them.
const isSkip=t=>/skipped|unmeasured|not measured|unfitted/i.test(t);

function paintRules(det,predicted){
  const box=$('rules');
  // No table, or a detector that never ran: the panel goes away rather than showing seven
  // rows none of which were consulted. "Not checked" is not "nothing matched".
  if(!RULES.length||!det||!det.checked){box.hidden=true;return;}
  box.hidden=false;
  $('r-det').textContent=DETECTOR?'· '+DETECTOR:'';
  $('r-cues').innerHTML=CLAUSES.map(c=>
    '<span class="cue '+(c.status==='checked'?'ok':'na')+'" title="'+esc(c.detail)+'">'+
    esc(c.cue)+' · '+esc(c.status)+'</span>').join('');
  $('r-rows').innerHTML=RULES.map(r=>
    '<tr class="'+(r.row===det.rule?'fired':(r.applies?'':'off'))+'"'+
    (r.applies?'':' title="'+esc(r.why_not)+'"')+'>'+
    '<td class="n">'+r.row+'</td>'+
    '<td class="c">'+esc(r.condition)+'</td>'+
    '<td class="s">'+esc(stateName(r.state))+'</td></tr>').join('');
  // The path the rule actually took, in the words `decide` recorded - including the
  // clauses it skipped, which are the ones a reader asks about.
  $('r-trace').innerHTML=(det.trace||[]).map(t=>
    '<li class="'+(isSkip(t)?'skip':'')+'">'+esc(t)+'</li>').join('');
  // Two paths, one frame, and they can differ - the probe scores the whole picture, the
  // rules count objects inside a boundary. Saying so is the point of showing both: on a
  // frame with no boundary the rules count the pavement too, and a reader who saw only
  // the disagreeing number would take it for a bug in one of them.
  $('r-split').innerHTML=(predicted&&det.state!==predicted)
    ? 'the probe says <b>'+esc(stateName(predicted))+'</b>, these rules say <b>'+
      esc(stateName(det.state))+'</b>' : '';
}

function paint(s){
  $('stage-step').textContent='image '+(s.index+1);
  $('stage-name').textContent=s.name+'  \\u00b7  '+s.width+'\\u00d7'+s.height;
  $('v-pred').textContent=s.predicted;
  $('v-pred').className='pill '+s.predicted;
  $('v-conf').textContent='confidence '+s.confidence.toFixed(3);
  $('v-ms').textContent=Math.round(s.elapsed_ms)+' ms'+(s.explained?' (explained)':'');

  // See the clip reviewer for why the probe's own audit numbers are gone from here: they
  // were the decomposition checking itself, and `test_walkthrough.py` checks it instead.
  // A still has no predecessor, so there is no movement tile to give it.
  $('x-ppl').textContent=(s.n_inside===null||s.n_inside===undefined)?'\\u2014':s.n_inside;
  $('x-ball').textContent=s.ball?'yes':'no';

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
    paintRules(d,s.predicted);
    // "Found six, counted three" is what a reader most often needs explained, so the two
    // counts sit side by side rather than being summed into one.
    const out=(d&&d.checked)?d.people_outside_boundary:null;
    $('x-outside').textContent=out==null?'\\u2014':out;
    $('x-outside-tile').className='tile'+(out?' flagged':'');
    if(d&&d.checked&&d.ball){$('x-ball').textContent='yes  '+d.ball_confidence.toFixed(2);}
    // Zero, or there is no boundary. Anything else means the outline reached the picture
    // but not the pooling, which is the exact failure this tile exists to make visible.
    const o=s.evidence_outside;
    $('x-out').textContent=o==null?'no boundary':(o*100).toFixed(1)+'%';
    $('x-out-tile').className='tile'+(o!=null&&o>0.001?' flagged':'');
    // Silent unless something is wrong with it - a note saying nothing went wrong is the
    // kind of sentence this page is meant to be rid of.
    const f=s.focus_ratio;
    $('focusnote').innerHTML=(f!=null&&f<1)
      ? '<span title="The evidence is spread as though the people the detector found were '+
        'not there, which for an active-play verdict is worth a look.">the model was not '+
        'looking at the people it found</span>' : '';
  } else {
    // An unexplained image gets no detector pass, so there is no rule path to draw.
    paintRules(null,null);
    $('x-outside').textContent='\\u2014';$('x-out').textContent='\\u2014';
    $('x-outside-tile').className='tile';$('x-out-tile').className='tile';
    $('focusnote').innerHTML='<span title="explain in detail bounds how many images get '+
      'a map, because explaining costs about twice a bare prediction">no evidence map for '+
      'this image</span>';
  }
  results.push(s);
  $('livecount').textContent=results.length+(results.length===1?' image':' images');
}

function card(s){
  const el=document.createElement('div');
  // Flagged on a disagreement a reader can check against the picture - rather than on the
  // focus ratio, which needed the method to read.
  //
  // **Both directions, which this checked only one of.** The comment here always claimed
  // "or the reverse" and the condition never implemented it, so a confident EMPTY on a
  // frame where the detector found people passed as an ordinary result. Reported from use
  // on 2026-09-24: `syn_v01b_people_004.jpg`, EMPTY at 0.993 with four people found, and
  // nothing on the card said the two halves of it disagreed. Finding people is direct
  // evidence against EMPTY in a way that finding people is *not* evidence for play, so the
  // two cases are not symmetric in what they mean - but both are worth a reader's eye.
  const play=s.predicted.indexOf('ACTIVE_PLAY')>=0;
  const empty=s.predicted.indexOf('EMPTY')>=0;
  const n=(s.n_inside===null||s.n_inside===undefined)?null:s.n_inside;
  const odd=(play&&n===0)||(empty&&n>0);
  el.className='shot'+(odd?' flagged':'');
  const imgs=document.createElement('div');imgs.className='imgs';
  if(s.explained){
    [[s.frame,'image'],[s.heat,'evidence map']].forEach(([src,alt])=>{
      const i=document.createElement('img');i.src=src;i.alt=s.name+' \\u2014 '+alt;
      i.onclick=()=>{$('zimg').src=src;$('zimg').alt=i.alt;$('zoom').showModal()};
      imgs.appendChild(i);
    });
  }
  const body=document.createElement('div');body.className='body';
  const name=document.createElement('div');name.className='fname';
  name.textContent=s.name;name.title=s.name;
  const row=document.createElement('div');row.className='row';
  const pill=document.createElement('span');pill.className='pill '+s.predicted;
  pill.textContent=s.predicted.replace('C1_','').replace('C2_','').replace('C3_','');
  const conf=document.createElement('span');conf.className='m spacer';
  conf.innerHTML='conf <b>'+s.confidence.toFixed(3)+'</b>';
  row.append(pill,conf);
  body.append(name,row);
  const r2=document.createElement('div');r2.className='row';
  const ppl=document.createElement('span');
  ppl.className='m'+(odd?' bad':'');
  ppl.innerHTML='people <b>'+
    ((s.n_inside===null||s.n_inside===undefined)?'\\u2014':s.n_inside)+'</b>';
  const ball=document.createElement('span');ball.className='m spacer';
  ball.innerHTML='ball <b>'+(s.ball?'yes':'no')+'</b>';
  r2.append(ppl,ball);body.appendChild(r2);
  el.append(imgs,body);
  return el;
}

function summarise(){
  if(!results.length) return;
  const counts={};
  results.forEach(s=>counts[s.predicted]=(counts[s.predicted]||0)+1);
  const meanConf=results.reduce((a,s)=>a+s.confidence,0)/results.length;
  const worst=results.reduce((a,s)=>Math.min(a,s.confidence),1);
  const num=s=>(s.n_inside===null||s.n_inside===undefined)?0:s.n_inside;
  const people=results.reduce((a,s)=>a+num(s),0);
  const withBall=results.filter(s=>s.ball).length;
  // The batch's quality flags, and they are disagreements a reader can check against the
  // picture rather than the focus ratio they replace. Two, not one: a match with nobody
  // found, and an empty pitch with people found. The second was missing, which is how a
  // batch could report every image as ordinary while containing a confident EMPTY over
  // four visible people.
  const odd=results.filter(s=>s.predicted.indexOf('ACTIVE_PLAY')>=0&&num(s)===0).length;
  const peopled=results.filter(s=>s.predicted.indexOf('EMPTY')>=0&&num(s)>0).length;

  const tiles=[
    ['Images read',results.length,''],
    ['People found',people,''],
    ['With a ball',withBall+' of '+results.length,''],
    ['Mean confidence',meanConf.toFixed(3),''],
    ['Least confident',worst.toFixed(3),''],
    ['Play, nobody on the pitch',odd,odd?'flagged':''],
    ['Empty, people found',peopled,peopled?'flagged':''],
  ];
  Object.entries(counts).sort().forEach(([k,v])=>
    tiles.push([k.replace('C1_','').replace('C2_','').replace('C3_','').replace(/_/g,' '),
                v,'']));
  $('sum').innerHTML='';
  tiles.forEach(([k,v,cls])=>{
    const t=document.createElement('div');t.className='tile'+(cls?' flagged':'');
    t.innerHTML='<div class="k"></div><div class="v sm"></div>';
    t.querySelector('.k').textContent=k;t.querySelector('.v').textContent=v;
    $('sum').appendChild(t);
  });

  const g=$('gallery');g.innerHTML='';
  // Ascending confidence: the images most worth a second opinion come first. Upload order
  // would bury them, and descending would put the ones needing no review at the top.
  [...results].sort((a,b)=>a.confidence-b.confidence).forEach(s=>g.appendChild(card(s)));
  $('out').classList.add('on');
}

$('zclose').onclick=()=>$('zoom').close();
$('zoom').onclick=e=>{if(e.target===$('zoom')) $('zoom').close()};

// --- running --------------------------------------------------------------------------
$('go').onclick=async()=>{
  if(!files.length) return;
  queue=[];results=[];streamDone=false;delay=+$('speed').value;
  $('err').style.display='none';$('out').classList.remove('on');
  $('stage').classList.add('on');$('go').disabled=true;
  $('busy').classList.add('on');
  $('busytxt').textContent='Reading '+files.length+' image'+(files.length===1?'':'s')+
    '\\u2026 the first run also loads the model, which takes about 15 seconds.';
  $('stage-step').textContent='loading the model';

  try{
    const images=[];
    for(const f of files) images.push({name:f.name,data:await readAsDataURL(f)});
    const q=new URLSearchParams({explain_n:$('expn').value,redact:String(redact),
      camera:$('camera').value});
    const r=await fetch('/api/v1/images/walkthrough?'+q,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({images})});
    if(!r.ok){
      let detail='HTTP '+r.status;
      try{const b=await r.json();detail=b.detail||detail;}catch(_){}
      throw new Error(detail);
    }
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
        if(d.type==='meta'){
          // The decision table arrives once, with this deployment's thresholds already in
          // it. Held rather than drawn now: it is drawn against each image's verdict, so
          // the row that fired can be marked.
          RULES=d.rules||[];CLAUSES=d.clauses||[];DETECTOR=d.detector||'';
          if(d.camera&&!d.boundary){
            $('err').textContent='No saved boundary for '+d.camera+
              ' — these were analysed on the whole frame, which is the case the '+
              'boundary exists to avoid. Draw one in the boundary editor first.';
            $('err').className='note warn';$('err').style.display='block';
          }
        }
        else if(d.type==='shot') queue.push(d);
        else if(d.type==='done'&&d.n_unreadable>0){
          $('err').textContent=d.n_unreadable+' file'+(d.n_unreadable===1?' was':'s were')+
            ' not readable as an image and were skipped rather than guessed at.';
          $('err').className='note warn';$('err').style.display='block';
        }
        else if(d.type==='error'){
          // `fatal` means the model itself could not be loaded, so re-uploading will not
          // help and the page should not imply it might.
          throw new Error(d.fatal
            ? 'The model could not be loaded, so nothing could be analysed — this is a '+
              'server-side problem, not a problem with these images. Check the terminal '+
              'running the server for the full traceback. Reported: '+d.detail
            : d.detail);
        }
      }
    }
  }catch(e){
    $('err').textContent=e.message;$('err').className='note err';
    $('err').style.display='block';
    $('busy').classList.remove('on');
  }finally{
    streamDone=true;$('go').disabled=false;
  }
};
</script>
</body></html>
"""
