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
<p class="sub2">Classifies each still and shows <b>where in the frame the score came
from</b> &mdash; the same evidence decomposition the clip walkthrough uses, on pictures
instead of footage. Nothing is stored: images are decoded in memory and the copies used for
reading are deleted as soon as the batch finishes.</p>

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
  <div><label for="speed">Slow motion</label>
    <input type="range" id="speed" min="0" max="2500" step="250" value="900"
      style="width:120px"></div>
  <div><label for="camera">Pitch boundary</label>
    <select id="camera" style="width:auto;min-width:150px">
      <option value="">whole frame</option>
    </select></div>
  <div><button class="ghost" id="redact" aria-pressed="false">Pixelate people</button></div>
  <button class="act" id="go" disabled>Explain these</button>
</div>
<p class="sub2" style="margin-top:9px">Pick a camera to mask everything outside its pitch
&mdash; the fix for a neighbouring pitch showing up in frame. Draw one in the
<a href="/roi" style="color:var(--accent)">boundary editor</a>.</p>

<div class="note" id="err"></div>

<div id="busy"><span class="spin"></span><span id="busytxt"></span></div>

<section id="stage">
  <h2>Watching it work</h2>
  <p class="sub2">One image at a time, rendered as the backbone finishes it. The slow motion
  is a pause this page adds between renders &mdash; the model runs at the same speed either
  way. The control that changes the actual work is <b>explain in detail</b>: an explained
  image costs roughly twice a bare prediction.</p>

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
      <figcaption>Where the score came from</figcaption></figure>
    <figure><img id="img-boxes" alt="What the detector found: a box round each person, a ring round the ball">
      <figcaption>What the detector found</figcaption></figure>
  </div>
  <p class="note" id="detnote" style="margin-top:8px">Two models, one frame. The probe scores pooled features, so its explanation can only be a heatmap; the detector finds objects, so its explanation is a box round <b>each person separately</b> and a ring round the ball. Anything found outside the boundary is dimmed, not dropped.</p>

  <div class="verdictbar">
    <span class="pill" id="v-pred">&mdash;</span>
    <span class="mono" id="v-conf" style="font-size:12.5px;color:var(--ink-2)"></span>
    <span class="spacer"></span>
    <span class="mono" id="v-ms" style="font-size:12.5px;color:var(--ink-3)"></span>
  </div>

  <div class="tiles" style="margin-top:12px">
    <div class="tile"><div class="k">Score from map</div>
      <div class="v sm" id="x-map">&mdash;</div></div>
    <div class="tile"><div class="k">Score direct</div>
      <div class="v sm" id="x-dir">&mdash;</div></div>
    <div class="tile"><div class="k">Reconstruction error</div>
      <div class="v sm" id="x-err">&mdash;</div></div>
    <div class="tile"><div class="k">People detected</div>
      <div class="v sm" id="x-ppl">&mdash;</div></div>
    <div class="tile" id="x-focus-tile"><div class="k">Focus ratio</div>
      <div class="v sm" id="x-focus">&mdash;</div></div>
    <div class="tile" id="x-out-tile"><div class="k">Evidence outside boundary</div>
      <div class="v sm" id="x-out">&mdash;</div></div>
  </div>
  <div class="note" id="focusnote">The evidence map is exact rather than a saliency
  heuristic: the per-position contributions plus a constant <b>sum to the score</b>. The
  first three tiles are that claim being checked in front of you &mdash; if the error is not
  tiny, the decomposition is wrong.</div>
</section>

<section id="out">
  <h2>The batch</h2>
  <p class="sub2">Sorted by confidence, <b>least confident first</b> &mdash; on a batch the
  useful question is not what the verdict is but which of these to look at yourself. Click
  either image to enlarge it.</p>
  <div class="tiles" id="sum"></div>
  <div class="note warn" id="alone"><b>Each image stands on its own.</b> The clip reviewer
  can overrule an isolated misread using the samples either side of it in time; a set of
  stills has no neighbours &mdash; these may be minutes or venues apart, and nothing in the
  upload says which. So nothing here is smoothed or corrected, and the evidence map is the
  only thing a prediction can be judged by.</div>
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
// Saved boundaries, so a batch can be masked to one pitch. Failing to load them leaves the
// selector at "whole frame", which is the behaviour the page had before boundaries existed.
(async()=>{
  try{
    const b=await (await fetch('/api/v1/roi')).json();
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

function paint(s){
  $('stage-step').textContent='image '+(s.index+1);
  $('stage-name').textContent=s.name+'  \\u00b7  '+s.width+'\\u00d7'+s.height;
  $('v-pred').textContent=s.predicted;
  $('v-pred').className='pill '+s.predicted;
  $('v-conf').textContent='confidence '+s.confidence.toFixed(3);
  $('v-ms').textContent=Math.round(s.elapsed_ms)+' ms'+(s.explained?' (explained)':'');

  if(s.explained){
    $('img-raw').src=s.frame;$('img-heat').src=s.heat;
    // The detector's pane. It is served with the same record, so a frame that has a
    // heatmap has boxes too; if it ever does not, the pane is blanked rather than left
    // showing the previous frame's answer next to this frame's heatmap.
    $('img-boxes').src=s.boxes||'';
    const d=s.detector;
    $('detnote').innerHTML=!d?'The detector did not run on this frame.'
      :(!d.checked?'The detector was not available for this frame — which is not the '+
        'same as finding nobody.'
      :('<b>'+d.state.split('_').slice(1).join(' ')+'</b> at '+d.confidence.toFixed(2)+
        ' by row '+d.rule+' — '+d.people_inside+' inside'+
        (d.people_outside_boundary?', '+d.people_outside_boundary+' outside the boundary':'')+
        ', ball '+(d.ball?d.ball_confidence.toFixed(2):'none')+
        '. The probe scores pooled features and can only answer with a heatmap; the '+
        'detector answers with a box round each person and a ring round the ball.'));
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
    $('x-focus').textContent=f==null?'\\u2014':f.toFixed(2)+'x';
    $('x-focus-tile').className='tile'+(f!=null&&f<1?' flagged':'');
    $('focusnote').innerHTML=f==null
      ? 'No people were detected here, so there is no area to compare the evidence against. '+
        'A ratio over zero area is not a small number \\u2014 it is not a number.'
      : ('Positive evidence on people over the area they cover. Above 1 means the score '+
         'concentrates on <b>people</b>; near or below 1 means it is spread as though they '+
         'were not there \\u2014 which for an ACTIVE_PLAY prediction is worth a look.');
  } else {
    ['x-map','x-dir','x-err','x-ppl','x-focus','x-out'].forEach(k=>$(k).textContent='\\u2014');
    $('x-focus-tile').className='tile';
    $('focusnote').textContent='This image was predicted without an evidence map \\u2014 '+
      '"explain in detail" bounds how many get one, because explaining costs about twice a '+
      'bare prediction.';
  }
  results.push(s);
  $('livecount').textContent=results.length+(results.length===1?' image':' images');
}

function card(s){
  const el=document.createElement('div');
  el.className='shot'+(s.focus_ratio!=null&&s.focus_ratio<1?' flagged':'');
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
  if(s.explained){
    const r2=document.createElement('div');r2.className='row';
    const f=s.focus_ratio;
    const focus=document.createElement('span');
    focus.className='m'+(f!=null&&f<1?' bad':'');
    focus.innerHTML='focus <b>'+(f==null?'\\u2014':f.toFixed(2)+'x')+'</b>';
    const ppl=document.createElement('span');ppl.className='m spacer';
    ppl.innerHTML='people <b>'+s.n_people+'</b>';
    r2.append(focus,ppl);body.appendChild(r2);
  }
  el.append(imgs,body);
  return el;
}

function summarise(){
  if(!results.length) return;
  const counts={};
  results.forEach(s=>counts[s.predicted]=(counts[s.predicted]||0)+1);
  const explained=results.filter(s=>s.explained);
  const flagged=explained.filter(s=>s.focus_ratio!=null&&s.focus_ratio<1).length;
  const meanConf=results.reduce((a,s)=>a+s.confidence,0)/results.length;
  const worst=results.reduce((a,s)=>Math.min(a,s.confidence),1);

  const tiles=[
    ['Images read',results.length,''],
    ['Mean confidence',meanConf.toFixed(3),''],
    ['Least confident',worst.toFixed(3),''],
    ['Evidence off people',flagged+' of '+explained.length,flagged?'flagged':''],
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
        if(d.type==='meta'&&d.camera&&!d.boundary){
          $('err').textContent='No saved boundary for '+d.camera+
            ' — these were analysed on the whole frame, which is the case the '+
            'boundary exists to avoid. Draw one in the boundary editor first.';
          $('err').className='note warn';$('err').style.display='block';
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
