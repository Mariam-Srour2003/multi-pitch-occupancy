"""The pitch boundary editor (WP3-T1). Served at `/roi`; posts to `/api/v1/roi/*`.

Click the corners of the pitch, save it under a camera, and everything outside the outline
is suppressed before the frame reaches the model. The reason it exists is the neighbouring
pitch: five-a-side pitches are built in rows, so a camera watching pitch 2 sees pitches 1 and
3 down the sides of its frame, and a match on the next pitch over puts real players into
frames where *this* pitch is empty.

Two decisions shape the layout.

**The preview runs the model twice and shows both answers.** The failure a boundary prevents
is invisible in the verdict - the model is right about what is in frame and wrong about what
was asked - so the editor puts the masked and unmasked predictions side by side with their
evidence maps. That is how an operator sees the boundary working rather than being told it
does, and it is also how they notice the opposite: masking is a distribution shift in itself,
and a boundary that changes the verdict on a frame with no neighbour in it is doing harm.

**Drawing is instant, classifying is not.** Every click redraws the outline and updates the
coverage figure locally, on a canvas, with no request at all. The model runs only when asked,
because a backbone pass per click would make the editor unusable and teach the operator to
avoid the one feature that tells them whether the outline is right.
"""

from __future__ import annotations

__all__ = ["ROI_HTML"]

ROI_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pitch Occupancy - Pitch Boundary</title>
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
h2:first-of-type{margin-top:0}
.sub2{color:var(--ink-2);font-size:13px;margin:0 0 14px;max-width:76ch}
.drop{border:1.5px dashed var(--line);border-radius:10px;padding:26px 20px;text-align:center;
  background:var(--surface);cursor:pointer}
.drop.over{border-color:var(--accent);background:var(--accent-soft)}
.drop p{margin:6px 0 0;color:var(--ink-2);font-size:13px}
.drop strong{font-size:15px}
.editor{display:grid;gap:14px;grid-template-columns:minmax(0,1fr) 268px;margin-top:14px}
@media (max-width:860px){.editor{grid-template-columns:1fr}}
.stagewrap{position:relative;background:var(--surface-2);border:1px solid var(--line);
  border-radius:10px;overflow:hidden;line-height:0}
#canvas{display:block;width:100%;cursor:crosshair;touch-action:none}
.side{display:flex;flex-direction:column;gap:10px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:13px 14px}
.panel h3{margin:0 0 8px;font:600 12.5px Archivo,sans-serif}
label{display:block;font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);margin-bottom:5px}
input[type=text],select{font:13px 'JetBrains Mono',monospace;padding:7px 9px;width:100%;
  border:1px solid var(--line);border-radius:6px;background:var(--ground);color:var(--ink)}
button.act{font:600 13px Archivo,sans-serif;padding:9px 16px;border-radius:6px;cursor:pointer;
  border:1px solid var(--accent);background:var(--accent);color:#fff;width:100%}
button.act[disabled]{opacity:.55;cursor:not-allowed}
button.ghost{font:600 12px Archivo,sans-serif;padding:7px 11px;border-radius:6px;cursor:pointer;
  border:1px solid var(--line);background:var(--surface);color:var(--ink-2)}
.btnrow{display:flex;gap:7px;flex-wrap:wrap}
.stat{display:flex;justify-content:space-between;align-items:baseline;gap:10px;
  font-size:12.5px;color:var(--ink-2);padding:4px 0}
.stat b{font:600 13px 'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;
  color:var(--ink)}
.stat.bad b{color:var(--flag)}
.saved{display:flex;flex-wrap:wrap;gap:6px}
.saved button{font:500 11px 'JetBrains Mono',monospace;padding:4px 9px;border-radius:5px;
  cursor:pointer;border:1px solid var(--line);background:var(--surface-2);color:var(--ink-2)}
.saved button:hover{border-color:var(--accent);color:var(--accent)}
.pill{display:inline-block;font:600 11px 'JetBrains Mono',monospace;letter-spacing:.04em;
  padding:3px 9px;border-radius:5px;white-space:nowrap}
.pill.C1_EMPTY{background:var(--empty-soft);color:var(--empty)}
.pill.C2_ACTIVE_PLAY{background:var(--play-soft);color:var(--play)}
.pill.C3_MAINTENANCE_NON_SPORTING{background:var(--maint-soft);color:var(--maint)}
.note{border-left:3px solid var(--accent);background:var(--surface);padding:12px 16px;
  border-radius:0 8px 8px 0;font-size:13px;color:var(--ink-2);margin:14px 0 0;max-width:80ch}
.note b{color:var(--ink)}
.warn{border-left-color:var(--flag)}
.ok{border-left-color:var(--play)}
.err{border-left-color:var(--flag);color:var(--flag);display:none}
.compare{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(292px,1fr))}
.compare figure{margin:0;background:var(--surface);border:1px solid var(--line);
  border-radius:10px;overflow:hidden}
.compare figure.lead{border-left:4px solid var(--accent)}
.compare img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;
  background:var(--surface-2)}
.compare .cap{padding:11px 13px;border-top:1px solid var(--line)}
.compare .cap .t{font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3);margin-bottom:7px}
.compare .cap .row{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.compare .cap .m{font:500 11.5px 'JetBrains Mono',monospace;color:var(--ink-2)}
.compare .cap .m b{color:var(--ink)}
#busy{display:none;align-items:center;gap:10px;color:var(--ink-2);font-size:13px;margin-top:12px}
#busy.on{display:flex}
.spin{width:15px;height:15px;border:2px solid var(--line);border-top-color:var(--accent);
  border-radius:50%;animation:sp .7s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion:reduce){.spin{animation:none}}
#result{display:none}#result.on{display:block}
</style></head><body>
<header><div class="hin">
  <div class="brand">Pitch Occupancy<span>.</span></div>
  <div class="sub">Pitch boundary</div>
  <a href="/images">Image reviewer</a><a href="/clip">Clip reviewer</a><a href="/">Findings</a>
</div></header>
<main>

<h2>Mark what counts as this pitch</h2>
<p class="sub2">Pitches are built in rows, so a camera watching one of them sees its
neighbours down the sides of the frame &mdash; and a match on the next pitch over puts real
players into frames where <b>this</b> pitch is empty. The model is right about what it sees
and wrong about what you asked. Draw the outline of the pitch this camera is responsible
for, and everything outside it is suppressed before the frame reaches the model.</p>

<div class="drop" id="drop" tabindex="0" role="button" aria-label="Choose a frame or video">
  <strong>Drop a frame or a video from this camera here, or click to choose</strong>
  <p>A still, or a clip &mdash; for a video the boundary is drawn on its <b>first frame</b>,
  which is correct for the whole recording because the camera does not move.
  Ideally one with the neighbouring pitch in it.</p>
</div>
<input type="file" id="file" accept="image/*,video/*" hidden>

<div class="editor">
  <div class="stagewrap">
    <canvas id="canvas" width="16" height="9"
      aria-label="Click the corners of the pitch to draw its boundary"></canvas>
  </div>

  <div class="side">
    <div class="panel">
      <h3>The outline</h3>
      <div class="stat"><span>Points</span><b id="s-pts">0</b></div>
      <div class="stat" id="s-cov-row"><span>Frame kept</span><b id="s-cov">&mdash;</b></div>
      <div class="btnrow" style="margin-top:9px">
        <button class="ghost" id="undo">Undo point</button>
        <button class="ghost" id="clear">Clear</button>
      </div>
      <p style="font-size:12px;color:var(--ink-3);margin:9px 0 0">Click the corners in order.
      Drag a point to move it. The outline closes itself.</p>
    </div>

    <div class="panel">
      <h3>Outside the outline</h3>
      <label for="fill">Fill with</label>
      <select id="fill">
        <option value="black" selected>black &mdash; hardest edge</option>
        <option value="blur">blur &mdash; keeps statistics plausible</option>
        <option value="mean">mean colour &mdash; gentlest edge</option>
      </select>
      <p style="font-size:12px;color:var(--ink-3);margin:8px 0 0">Black is what this project
      has always done. A large black region is something no pretraining set contains, so try
      the others if masking moves a verdict it should not.</p>
    </div>

    <div class="panel">
      <h3>Save it against a camera</h3>
      <label for="camera">Camera</label>
      <input type="text" id="camera" placeholder="venue_01/camera_A" autocomplete="off">
      <div style="margin-top:9px">
        <button class="act" id="save" disabled>Save boundary</button></div>
      <div style="margin-top:10px"><label>Saved</label><div class="saved" id="saved"></div></div>
    </div>

    <button class="act" id="check" disabled>Check it against the model</button>
  </div>
</div>

<div class="note err" id="err"></div>
<div id="busy"><span class="spin"></span><span id="busytxt"></span></div>

<section id="result">
  <h2>What the boundary changed</h2>
  <p class="sub2">The model run twice on this frame: once inside the outline, once on the
  whole frame. <b>Both are shown on purpose.</b> If the boundary is working, the two differ
  on exactly the frames a neighbour was in, and the evidence moves off the edge and onto the
  pitch. If they differ on a frame with no neighbour in it, the mask itself moved the
  prediction &mdash; which is a reason to try another fill, not to trust the result.</p>
  <div class="note" id="changed"></div>
  <div class="compare" style="margin-top:12px">
    <figure class="lead"><img id="i-in" alt="Evidence map inside the boundary">
      <div class="cap"><div class="t">Inside the boundary</div>
        <div class="row"><span class="pill" id="p-in">&mdash;</span>
          <span class="m" id="c-in"></span></div>
        <div class="row" style="margin-top:6px"><span class="m" id="f-in"></span>
          <span class="m" id="n-in"></span></div></div></figure>
    <figure><img id="i-wh" alt="Evidence map over the whole frame">
      <div class="cap"><div class="t">Whole frame, as before</div>
        <div class="row"><span class="pill" id="p-wh">&mdash;</span>
          <span class="m" id="c-wh"></span></div>
        <div class="row" style="margin-top:6px"><span class="m" id="f-wh"></span>
          <span class="m" id="n-wh"></span></div></div></figure>
  </div>
</section>

</main>
<script>
const $=id=>document.getElementById(id);
const cv=$('canvas'),ctx=cv.getContext('2d');
let img=null,dataURL=null,pts=[],drag=-1,saved={};

// --- the canvas -----------------------------------------------------------------------
// Points are stored normalised (0-1) because that is what the server stores and what makes a
// boundary survive a camera being swapped for a higher-resolution one. The canvas is only
// ever a view onto them.
function draw(){
  if(!img) return;
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(img,0,0,cv.width,cv.height);
  if(!pts.length) return;
  const px=pts.map(([x,y])=>[x*cv.width,y*cv.height]);

  // Everything outside the outline, dimmed - so the frame shows what the model will lose
  // before anyone runs it.
  ctx.save();
  ctx.beginPath();ctx.rect(0,0,cv.width,cv.height);
  ctx.moveTo(px[0][0],px[0][1]);
  for(let i=px.length-1;i>=0;i--) ctx.lineTo(px[i][0],px[i][1]);
  ctx.closePath();
  ctx.fillStyle='rgba(10,14,14,.62)';ctx.fill('evenodd');
  ctx.restore();

  ctx.beginPath();ctx.moveTo(px[0][0],px[0][1]);
  px.slice(1).forEach(([x,y])=>ctx.lineTo(x,y));
  if(px.length>2) ctx.closePath();
  ctx.lineWidth=Math.max(2,cv.width/420);
  ctx.strokeStyle='#4fb3bf';ctx.stroke();

  px.forEach(([x,y],i)=>{
    ctx.beginPath();ctx.arc(x,y,Math.max(5,cv.width/150),0,Math.PI*2);
    ctx.fillStyle=i===0?'#f6c344':'#4fb3bf';ctx.fill();
    ctx.lineWidth=Math.max(1.5,cv.width/700);ctx.strokeStyle='#0e1414';ctx.stroke();
  });
}

function coverage(){
  if(pts.length<3) return 0;
  let t=0;
  for(let i=0;i<pts.length;i++){
    const [x0,y0]=pts[i],[x1,y1]=pts[(i+1)%pts.length];
    t+=x0*y1-x1*y0;
  }
  return Math.abs(t)/2;
}

function refresh(){
  draw();
  $('s-pts').textContent=pts.length;
  const c=coverage();
  $('s-cov').textContent=pts.length>=3?(c*100).toFixed(0)+'%':'\\u2014';
  // Under 2% the server refuses it as a mis-click; flag it here rather than at save time.
  $('s-cov-row').className='stat'+(pts.length>=3&&c<0.02?' bad':'');
  const ok=pts.length>=3&&img!=null;
  $('save').disabled=!ok;$('check').disabled=!ok;
}

function at(e){
  const r=cv.getBoundingClientRect();
  return [(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];
}
function near(p){
  const r=cv.getBoundingClientRect();
  // A grab radius in *screen* pixels, converted to normalised units - otherwise the handles
  // are easy to grab on a wide canvas and impossible on a narrow one.
  const tol=14/r.width;
  return pts.findIndex(([x,y])=>Math.hypot(x-p[0],y-p[1])<tol*1.4);
}

cv.onpointerdown=e=>{
  if(!img) return;
  const p=at(e);const hit=near(p);
  if(hit>=0){drag=hit;cv.setPointerCapture(e.pointerId);}
  else{pts.push(p);refresh();}
};
cv.onpointermove=e=>{
  if(drag<0) return;
  pts[drag]=at(e).map(v=>Math.min(1,Math.max(0,v)));refresh();
};
cv.onpointerup=()=>{drag=-1};
$('undo').onclick=()=>{pts.pop();refresh()};
$('clear').onclick=()=>{pts=[];refresh()};

// --- loading a frame ------------------------------------------------------------------
// A video is decoded **on the server**, not by a <video> element painting frame 0 onto a
// canvas. The browser and OpenCV need not agree on which frame a seek to 0 lands on, on
// rotation metadata, or on colour conversion - and a boundary drawn against a frame the
// analysis never sees is wrong by however much they differ, silently. The drawing surface
// therefore comes from the same decoder `walk_clip` reads with.
async function loadVideo(file){
  $('err').style.display='none';$('busy').classList.add('on');
  $('busytxt').textContent='Reading the first frame…';
  try{
    const r=await fetch('/api/v1/roi/first-frame',{method:'POST',body:file,
      headers:{'Content-Type':'application/octet-stream'}});
    const b=await r.json();
    if(!r.ok) throw new Error(b.detail||('HTTP '+r.status));
    show(b.frame);
  }catch(e){$('err').className='note err';fail(e.message);}
  finally{$('busy').classList.remove('on');}
}

function show(url){
  dataURL=url;
  const im=new Image();
  im.onload=()=>{
    img=im;
    // The canvas matches the frame's own aspect ratio, so a point clicked on the picture
    // is the point stored - no letterbox to correct for.
    cv.width=Math.min(1280,im.naturalWidth);
    cv.height=Math.round(cv.width*im.naturalHeight/im.naturalWidth);
    pts=[];$('result').classList.remove('on');refresh();
  };
  im.src=url;
}

function load(file){
  if(!file) return;
  if(file.type.startsWith('video/')) return loadVideo(file);
  if(!file.type.startsWith('image/')) return;
  const reader=new FileReader();
  reader.onload=()=>show(reader.result);
  reader.readAsDataURL(file);
}
$('file').onchange=e=>load(e.target.files[0]);
$('drop').onclick=()=>$('file').click();
$('drop').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('file').click();}};
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('over')};
$('drop').ondragleave=()=>$('drop').classList.remove('over');
$('drop').ondrop=e=>{e.preventDefault();$('drop').classList.remove('over');
  load(e.dataTransfer.files[0])};

// --- saved boundaries -----------------------------------------------------------------
async function loadSaved(){
  try{
    const r=await fetch('/api/v1/roi');const b=await r.json();
    saved=b.boundaries||{};
    const box=$('saved');box.innerHTML='';
    const names=Object.keys(saved);
    if(!names.length){
      box.innerHTML='<span style="font-size:12px;color:var(--ink-3)">none yet</span>';
      return;
    }
    names.forEach(k=>{
      const b2=document.createElement('button');
      b2.textContent=k+'  '+(saved[k].coverage*100).toFixed(0)+'%';
      b2.title='Load '+k;
      b2.onclick=()=>{pts=saved[k].polygon.map(p=>[p[0],p[1]]);
        $('camera').value=k;refresh()};
      box.appendChild(b2);
    });
  }catch(_){/* the editor still works without the list */}
}
loadSaved();

function fail(msg){
  $('err').textContent=msg;$('err').style.display='block';
  $('busy').classList.remove('on');
}

$('save').onclick=async()=>{
  const camera=$('camera').value.trim();
  if(!camera){fail('Give the camera a name first \\u2014 the boundary belongs to a view, '+
    'not to a pitch: the two cameras on one pitch see opposite ends of it.');return;}
  $('err').style.display='none';
  try{
    const r=await fetch('/api/v1/roi',{
      method:'PUT',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({camera:camera,polygon:pts})});
    const b=await r.json();
    if(!r.ok) throw new Error(b.detail||('HTTP '+r.status));
    await loadSaved();
    $('err').textContent='Saved '+camera+' \\u2014 keeps '+
      (b.coverage*100).toFixed(0)+'% of the frame.';
    $('err').className='note ok';$('err').style.display='block';
  }catch(e){$('err').className='note err';fail(e.message);}
};

// --- checking it against the model ----------------------------------------------------
function fillSide(side,v){
  $('p-'+side).textContent=v.predicted;
  $('p-'+side).className='pill '+v.predicted;
  $('c-'+side).innerHTML='conf <b>'+v.confidence.toFixed(3)+'</b>';
  $('f-'+side).innerHTML='focus <b>'+
    (v.focus_ratio==null?'\\u2014':v.focus_ratio.toFixed(2)+'x')+'</b>';
  $('n-'+side).innerHTML='people <b>'+(v.n_people==null?'\\u2014':v.n_people)+'</b>';
  $('i-'+side).src=v.heat;
}

$('check').onclick=async()=>{
  if(!dataURL||pts.length<3) return;
  $('err').style.display='none';$('check').disabled=true;
  $('busy').classList.add('on');
  $('busytxt').textContent='Running the model twice \\u2014 once inside the boundary, once '+
    'on the whole frame. The first run also loads the model, about 15 seconds.';
  try{
    const r=await fetch('/api/v1/roi/preview',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image:dataURL,polygon:pts,fill:$('fill').value,explain:true})});
    const b=await r.json();
    if(!r.ok) throw new Error(b.detail||('HTTP '+r.status));
    fillSide('in',b.inside);fillSide('wh',b.whole);
    const note=$('changed');
    if(b.changed){
      note.className='note ok';
      note.innerHTML='<b>The boundary changed the verdict</b> on this frame: '+
        b.whole.predicted+' over the whole frame, '+b.inside.predicted+' inside the '+
        'outline. If there is a neighbouring pitch in this frame, that is the boundary '+
        'doing its job. If there is not, the mask moved the prediction on its own \\u2014 '+
        'try a different fill before trusting it.';
    } else {
      note.className='note';
      note.innerHTML='<b>Same verdict either way</b> on this frame ('+b.inside.predicted+
        '). That is not a failure: it means nothing outside the outline was changing the '+
        'answer here. Test it on a frame where the neighbouring pitch is in use \\u2014 that '+
        'is the case the boundary exists for.';
    }
    $('result').classList.add('on');
  }catch(e){$('err').className='note err';fail(e.message);}
  finally{$('busy').classList.remove('on');$('check').disabled=false;}
};
</script>
</body></html>
"""
