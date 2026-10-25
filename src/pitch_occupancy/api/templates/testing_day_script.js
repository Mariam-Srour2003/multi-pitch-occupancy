
const files={A:null,B:null};
let RUN=null;
const RESULT={USED:'used',NOT_USED:'not used',REVIEW:'please review'};
const OUTCOME={booked_and_used:['booked and used','ok'],
  booked_not_used:['booked, not used','bad'],used_not_booked:['used, not booked','bad'],
  not_booked_not_used:['not booked, not used','ok'],
  off_sale_but_used:['closed, but used','bad'],review:['please review','rev']};
const pct=x=>Math.round(100*(x||0))+'%';
const hm=s=>{s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);
  return h+':'+String(m).padStart(2,'0');};

const local=()=>document.querySelector('input[name=src]:checked').value==='local';
function ready(){
  const two=local()?!!$('recB').value:!!files.B;
  $('offsetBox').style.display=two?'block':'none';
  $('go').disabled=local()?!$('recA').value:!files.A;
}
function choose(cam,file){
  if(!file)return;files[cam]=file;
  $('name'+cam).textContent=file.name+' ('+(file.size/1048576).toFixed(1)+' MB)';
  $('drop'+cam).classList.add('has');
  ready();
}
let RECS=null;
async function loadRecordings(){
  $('recHint').textContent='Looking for recordings...';
  try{
    const r=await fetch('/api/v1/testing/recordings');
    if(!r.ok)throw new Error(await detail(r));
    RECS=await r.json();
  }catch(err){$('recHint').textContent='Could not list recordings: '+err.message;return;}
  const opt=x=>'<option value="'+esc(x.name)+'">'+esc(x.name)+' ('+
    (x.size_mb>=1024?(x.size_mb/1024).toFixed(1)+' GB':x.size_mb+' MB')+
    (x.duration_s?', '+hm(x.duration_s):'')+')</option>';
  $('recA').innerHTML=RECS.recordings.map(opt).join('');
  $('recB').innerHTML='<option value="">none - one camera</option>'+
    RECS.recordings.map(opt).join('');
  $('recHint').innerHTML=RECS.recordings.length?'Recordings are read from <code>'+
    esc(RECS.folder)+'</code> and are not deleted afterwards.':'No recordings yet. Copy the '+
    'video files into <code>'+esc(RECS.folder)+'</code> on this computer (or a folder '+
    'inside it), then choose this option again.';
  ready();
}
for(const radio of document.querySelectorAll('input[name=src]'))
  radio.addEventListener('change',()=>{
    $('localBox').hidden=!local();$('uploadBox').hidden=local();
    if(local()&&RECS===null)loadRecordings();ready();});
$('recA').addEventListener('change',ready);$('recB').addEventListener('change',ready);
for(const cam of ['A','B']){
  const drop=$('drop'+cam),input=$('file'+cam);
  input.addEventListener('change',()=>choose(cam,input.files[0]));
  drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('over');});
  drop.addEventListener('dragleave',()=>drop.classList.remove('over'));
  drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('over');
    choose(cam,e.dataTransfer.files[0]);});
}

function setStatus(el,msg,bad){$(el).className='note'+(bad?' err':'');$(el).textContent=msg;}
async function detail(r){
  try{const d=await r.json();
    return typeof d.detail==='string'?d.detail:JSON.stringify(d.detail||d);}
  catch(e){return 'The server answered '+r.status+'.';}
}

$('go').addEventListener('click',async()=>{
  if(!$('start').value){setStatus('status','Enter the time the video starts.',true);return;}
  if(local())return startLocal();
  $('go').disabled=true;setStatus('status','');
  $('prog').classList.add('on');$('res').classList.remove('on');
  $('progText').textContent='Uploading the video...';$('progBar').style.width='0';
  const q=new URLSearchParams({start:$('start').value,slot_minutes:$('slot').value||90,
    interval_s:$('interval').value||60,whole_frame:$('whole').checked,
    name:files.A.name});
  if($('first').value)q.set('first_slot',$('first').value);
  if($('field').value.trim())q.set('field_id',$('field').value.trim());
  if($('day').value)q.set('day',$('day').value);
  const rules=ruleParams('minSeen','closeLow','closeHigh');
  if(typeof rules==='string'){setStatus('status',rules,true);ready();
    $('prog').classList.remove('on');return;}
  for(const [k,v] of Object.entries(rules))q.set(k,v);
  let body=files.A;
  if(files.B){
    q.set('size_a',files.A.size);q.set('offset_s',$('offset').value||0);
    q.set('camera_a',$('camA').value);q.set('camera_b',$('camB').value);
    q.set('name_b',files.B.name);body=new Blob([files.A,files.B]);
  }else if($('camA').value){q.set('camera',$('camA').value);}
  try{
    const r=await fetch('/api/v1/testing/day?'+q,{method:'POST',body,
      headers:{'Content-Type':'application/octet-stream'}});
    if(!r.ok)throw new Error(await detail(r));
    const d=await r.json();
    poll(d.job_id);
  }catch(err){$('prog').classList.remove('on');setStatus('status',err.message,true);
    ready();}
});

async function startLocal(){
  const rules=ruleParams('minSeen','closeLow','closeHigh');
  if(typeof rules==='string'){setStatus('status',rules,true);return;}
  const b={file_a:$('recA').value,start:$('start').value,slot_minutes:+($('slot').value||90),
    interval_s:+($('interval').value||60),whole_frame:$('whole').checked,...rules};
  if($('recB').value){b.file_b=$('recB').value;b.offset_s=+($('offset').value||0);
    b.camera_a=$('camA').value;b.camera_b=$('camB').value;}
  else if($('camA').value)b.camera=$('camA').value;
  if($('first').value)b.first_slot=$('first').value;
  if($('field').value.trim())b.field_id=$('field').value.trim();
  if($('day').value)b.day=$('day').value;
  $('go').disabled=true;setStatus('status','');
  $('prog').classList.add('on');$('res').classList.remove('on');
  $('progText').textContent='Starting...';$('progBar').style.width='0';
  try{
    const r=await fetch('/api/v1/testing/day/local',{method:'POST',body:JSON.stringify(b),
      headers:{'Content-Type':'application/json'}});
    if(!r.ok)throw new Error(await detail(r));
    poll((await r.json()).job_id);
  }catch(err){$('prog').classList.remove('on');setStatus('status',err.message,true);ready();}
}

let JOB=null;
$('cancel').addEventListener('click',async()=>{
  if(!JOB)return;
  $('cancel').disabled=true;$('progText').textContent='Stopping the test...';
  try{
    const r=await fetch('/api/v1/testing/jobs/'+encodeURIComponent(JOB)+'/cancel',
      {method:'POST'});
    if(!r.ok&&r.status!==409)throw new Error(await detail(r));
  }catch(err){$('cancel').disabled=false;setStatus('status',err.message,true);}
});

async function poll(id){
  if(JOB!==id){JOB=id;$('cancel').hidden=false;$('cancel').disabled=false;}
  try{
    const r=await fetch('/api/v1/testing/jobs/'+encodeURIComponent(id));
    if(!r.ok)throw new Error(await detail(r));
    const j=await r.json();
    const p=j.progress||{};
    if(j.cancel_requested&&j.status==='running'){
      $('progText').textContent='Stopping the test...';$('cancel').disabled=true;}
    else if(j.status==='queued'){
      $('progText').textContent='Waiting for another test to finish...';}
    else if(j.status==='running'){
      $('progText').textContent=p.total?('Checking the video... '+p.done+' of '+p.total+
        ' moments'):'Preparing the video...';
      $('progBar').style.width=(p.total?100*p.done/p.total:2)+'%';
    }
    if(j.status!=='queued'&&j.status!=='running'){JOB=null;$('cancel').hidden=true;}
    if(j.status==='cancelled'){
      $('prog').classList.remove('on');ready();
      setStatus('status','The test was stopped. Nothing was saved'+
        (j.label?' for '+j.label:'')+'.');return;
    }
    if(j.status==='done'){
      $('prog').classList.remove('on');ready();
      history.replaceState(null,'','/testing/day?run='+encodeURIComponent(j.result.id));
      render(j.result);return;
    }
    if(j.status==='failed')throw new Error('The test could not be completed: '+j.error);
    setTimeout(()=>poll(id),1500);
  }catch(err){$('prog').classList.remove('on');setStatus('status',err.message,true);
    ready();}
}

// The judging rules from three percentage inputs, or a sentence saying what is wrong.
function ruleParams(seen,low,high){
  const out={},names={min_seen_pct:seen,close_low_pct:low,close_high_pct:high};
  for(const [k,id] of Object.entries(names)){
    const raw=$(id).value.trim();if(raw==='')continue;
    const v=Number(raw);
    if(!(v>=0&&v<=100))return 'Each rule must be a percentage from 0 to 100.';
    out[k]=v;
  }
  const lo=out.close_low_pct??40,hi=out.close_high_pct??60;
  if(lo>=hi)return 'The close-call band must start below where it ends.';
  return out;
}
const RULES0={min_seen_share:0.5,close_low:0.4,close_high:0.6};
const pc=x=>String(Math.round(1000*x)/10);

function render(run){
  RUN=run;$('res').classList.add('on');
  const rl=run.rules||RULES0;
  $('rulesLine').textContent='Rules: '+(run.rules_text||'at least 50% seen; 40-60% is a '+
    'close call')+'.'+(run.rejudged?' Judged again on '+run.rejudged.at.replace('T',' ')+
    ' (before: '+run.rejudged.previous_rules_text+').':'');
  $('rMinSeen').value=pc(rl.min_seen_share);$('rLow').value=pc(rl.close_low);
  $('rHigh').value=pc(rl.close_high);
  const c=run.counts;
  $('summary').textContent=(run.video_name||'Video')+(run.cameras==2?' + '+run.video_name_b:'')+
    ' - starts '+run.start+', '+run.slot_minutes+'-minute slots'+
    (run.first_slot?' from '+run.first_slot:'')+', checked every '+
    Math.round(run.interval_used_s)+' seconds ('+run.n_samples+' checks, '+
    hm(run.duration_s)+' of video).';
  $('note').textContent=run.note||'';
  $('tiles').innerHTML=[['slots',c.slots],['used',c.USED],['not used',c.NOT_USED],
    ['please review',c.REVIEW,c.REVIEW>0]].map(([k,v,hot])=>'<div class="tile'+
    (hot?' flagged':'')+'"><div class="k">'+k+'</div><div class="v">'+v+'</div></div>').join('');
  $('slotbar').innerHTML=run.slots.map(s=>'<div class="'+s.status+'" title="'+
    esc(s.label+': '+RESULT[s.status]+'. '+s.reason)+'">'+esc(s.start)+
    '<small>'+RESULT[s.status]+'</small></div>').join('');
  const total=Math.max(run.duration_s,1e-6),step=run.interval_used_s;
  $('timeline').innerHTML=run.timeline.map(([t,k])=>'<i class="'+k+'" style="width:'+
    (Math.max(0,Math.min(step,total-t))/total*100)+'%"></i>').join('');
  $('axis').innerHTML='<span>'+run.start+'</span><span>+'+hm(total/2)+'</span><span>+'+
    hm(total)+'</span>';
  $('csv').href='/api/v1/testing/runs/'+encodeURIComponent(run.id)+'.csv';
  $('cField').value=run.field_id||$('cField').value;
  $('cDay').value=run.day||$('cDay').value;
  renderRows(run.comparison);
  $('res').scrollIntoView({behavior:'smooth',block:'start'});
}

function renderRows(cmp){
  const byIdx={};
  if(cmp)for(const k of cmp.checks)byIdx[k.index]=k;
  for(const el of document.querySelectorAll('.bk'))el.classList.toggle('on',!!cmp);
  $('rows').innerHTML=RUN.slots.map(s=>{
    let bk='';
    if(cmp){const k=byIdx[s.index],o=k?OUTCOME[k.outcome]:null;
      bk='<td class="bk on">'+(o?'<span class="pill '+o[1]+'">'+o[0]+'</span><br><span '+
        'class="muted" style="font-size:12px">'+esc(k.message)+'</span>':'')+'</td>';}
    return '<tr><td class="mono">'+esc(s.label)+(s.coverage<1?'<br><span class="muted">'+
      pct(s.coverage)+' in video</span>':'')+'</td><td>'+esc(s.main_label)+'</td>'+
      '<td class="num">'+pct(s.playing_share)+'<br><span class="muted">'+s.playing_min+
      ' min</span></td><td class="num">'+pct(s.empty_share)+'</td><td class="num">'+
      pct(s.maintenance_share)+'</td><td class="num">'+pct(s.seen_share)+'</td><td>'+
      '<span class="pill '+s.status+'">'+RESULT[s.status]+'</span></td><td class="reason">'+
      esc(s.reason)+'</td>'+bk+'</tr>';}).join('');
  if(cmp){
    const n=cmp.counts||{};
    $('cTiles').innerHTML=Object.entries(OUTCOME).filter(([k])=>n[k]).map(([k,[lab,cls]])=>
      '<div class="tile'+(cls==='bad'?' flagged':'')+'"><div class="k">'+lab+
      '</div><div class="v">'+n[k]+'</div></div>').join('');
    setStatus('cStatus','Compared with '+(cmp.source==='upload'?'the uploaded file':
      'your booking system')+' for '+cmp.field_id+' on '+cmp.day+'.'+
      (cmp.n_bookings?'':' No bookings were found for this pitch on this date, so every '+
       'slot counts as not booked.'));
  }
}

async function compare(useConfigured){
  if(!RUN)return;
  const q=new URLSearchParams({use_configured:useConfigured});
  if($('cField').value.trim())q.set('field_id',$('cField').value.trim());
  if($('cDay').value)q.set('day',$('cDay').value);
  let body='';
  if(!useConfigured){
    const f=$('cFile').files[0];
    if(!f){setStatus('cStatus','Choose a booking file first.',true);return;}
    body=await f.text();
  }
  setStatus('cStatus','Comparing...');
  try{
    const r=await fetch('/api/v1/testing/runs/'+encodeURIComponent(RUN.id)+'/compare?'+q,
      {method:'POST',body,headers:{'Content-Type':'text/csv'}});
    if(!r.ok)throw new Error(await detail(r));
    const d=await r.json();RUN.comparison=d;renderRows(d);
  }catch(err){setStatus('cStatus',err.message,true);}
}
$('cUpload').addEventListener('click',()=>compare(false));
$('cConfigured').addEventListener('click',()=>compare(true));
$('print').addEventListener('click',()=>window.print());
$('rGo').addEventListener('click',async()=>{
  if(!RUN)return;
  const rules=ruleParams('rMinSeen','rLow','rHigh');
  if(typeof rules==='string'){setStatus('rStatus',rules,true);return;}
  setStatus('rStatus','Judging again...');
  try{
    const r=await fetch('/api/v1/testing/runs/'+encodeURIComponent(RUN.id)+'/rejudge?'+
      new URLSearchParams(rules),{method:'POST'});
    if(!r.ok)throw new Error(await detail(r));
    const run=await r.json();render(run);
    const n=run.rejudged.slots_changed;
    setStatus('rStatus',n?(n+' slot'+(n==1?'':'s')+' changed.'+(run.comparison?
      ' The comparison with bookings was redone.':'')):'No slot changed.');
  }catch(err){setStatus('rStatus',err.message,true);}
});
$('again').addEventListener('click',()=>{$('res').classList.remove('on');
  history.replaceState(null,'','/testing/day');window.scrollTo(0,0);});

// Opened from the Testing page's list of running tests: follow that test.
const following=new URLSearchParams(location.search).get('job');
if(following){$('prog').classList.add('on');$('progText').textContent='Checking...';
  poll(following);}
const wanted=new URLSearchParams(location.search).get('run');
if(wanted){fetch('/api/v1/testing/runs/'+encodeURIComponent(wanted)).then(async r=>{
  if(!r.ok)throw new Error(await detail(r));return r.json();}).then(render)
  .catch(err=>setStatus('status','Could not open that test: '+err.message,true));}
