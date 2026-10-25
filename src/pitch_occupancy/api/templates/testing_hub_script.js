
fetch('/api/v1/testing/runs').then(r=>r.json()).then(rows=>{
  const box=document.getElementById('runs');
  if(!rows.length){box.innerHTML='<p class="muted">No tests yet.</p>';return;}
  box.innerHTML='<ul class="runs">'+rows.map(r=>{
    const c=r.counts||{};
    return '<li><a href="/testing/day?run='+encodeURIComponent(r.id)+'">'+
      esc(r.video_name||'video')+'</a> <span class="muted">'+esc((r.created||'').replace('T',' '))+
      ' &middot; starts '+esc(r.start)+' &middot; '+esc(r.slot_minutes)+'-minute slots'+
      (r.cameras==2?' &middot; two cameras':'')+'</span><br><span class="mono">'+
      (c.slots||0)+' slots: '+(c.USED||0)+' used, '+(c.NOT_USED||0)+' not used, '+
      (c.REVIEW||0)+' to review</span></li>';}).join('')+'</ul>';
}).catch(()=>{document.getElementById('runs').innerHTML=
  '<p class="muted">Could not load previous tests.</p>';});

// Running and queued tests, refreshed every 5 seconds while any is still going.
let jobTimer=null;
async function loadJobs(){
  clearTimeout(jobTimer);
  const box=document.getElementById('jobs');
  let rows;
  try{const r=await fetch('/api/v1/testing/jobs');if(!r.ok)throw new Error();rows=await r.json();}
  catch(e){box.innerHTML='<p class="muted">Could not load the running tests.</p>';return;}
  const active=rows.filter(j=>j.status==='queued'||j.status==='running');
  if(!active.length){box.innerHTML='<p class="muted">No test is running.</p>';return;}
  active.sort((a,b)=>(a.status==='running'?0:1)-(b.status==='running'?0:1)||a.created-b.created);
  box.innerHTML='<ul class="runs jobs">'+active.map(j=>{
    const p=j.progress||{},share=p.total?Math.round(100*p.done/p.total):0;
    const when=new Date(j.created*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
    const state=j.cancel_requested?'stopping...':j.status==='queued'?'waiting its turn':
      (p.total?p.done+' of '+p.total+' moments ('+share+'%)':'preparing the video');
    return '<li><div class="jl"><a href="/testing/day?job='+encodeURIComponent(j.job_id)+'">'+
      esc(j.label||'video')+'</a> <span class="muted">started '+esc(when)+' &middot; '+
      esc(state)+'</span><div class="bar"><i style="width:'+
      (j.status==='running'?Math.max(share,2):0)+'%"></i></div></div>'+
      '<button class="ghost" data-cancel="'+esc(j.job_id)+'"'+
      (j.cancel_requested?' disabled':'')+'>Cancel</button></li>';}).join('')+'</ul>';
  for(const b of box.querySelectorAll('button[data-cancel]'))
    b.addEventListener('click',async()=>{
      b.disabled=true;b.textContent='Stopping...';
      try{await fetch('/api/v1/testing/jobs/'+encodeURIComponent(b.dataset.cancel)+'/cancel',
        {method:'POST'});}catch(e){}
      loadJobs();});
  jobTimer=setTimeout(loadJobs,5000);
}
loadJobs();
