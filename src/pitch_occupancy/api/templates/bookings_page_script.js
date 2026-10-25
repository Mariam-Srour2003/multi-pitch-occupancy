
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
