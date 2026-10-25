
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
