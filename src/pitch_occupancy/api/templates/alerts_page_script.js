
const errBox=m=>'<p class="note err">'+esc(m)+'</p>';
const sev=s=>'<span class="sev '+esc(s)+'">'+esc(s)+'</span>';
const chans=o=>{const e=Object.entries(o||{});return e.length?'<ul class="ch">'+
  e.map(([k,v])=>'<li><b>'+esc(k)+'</b>: '+esc(v)+'</li>').join('')+'</ul>':
  '<span class="muted">-</span>'};
const table=(cols,rows)=>'<div class="scroll"><table><thead><tr>'+
  cols.map(c=>'<th>'+esc(c[1])+'</th>').join('')+'</tr></thead><tbody>'+
  rows.map(r=>'<tr>'+cols.map(c=>'<td'+(c[2]?' class="'+c[2]+'"':'')+'>'+
    (c[3]?c[3](r):esc(r[c[0]]))+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';

document.querySelectorAll('[data-copy]').forEach(b=>b.addEventListener('click',async()=>{
  const t=$(b.dataset.copy).textContent;
  try{await navigator.clipboard.writeText(t);b.textContent='Copied'}
  catch(e){b.textContent='Select and copy'}
  setTimeout(()=>b.textContent='Copy',1500);
}));

async function loadConfig(){
  const card=$('cfgCard');
  try{
    const d=await call('/api/v1/alerts/config');
    if(!d.configured){
      card.innerHTML='<b>Alerts are not set up yet.</b> <span class="muted">Create <code>'+
        esc(d.file)+'</code> as shown below - pick a service, copy its example, and put '+
        'your own address in it.</span>';
      $('howto').open=true;return;
    }
    if(!d.valid){
      card.innerHTML='<b class="bad">The alert settings have a problem, so no alerts are '+
        'being sent.</b><p>'+esc(d.problem)+'</p><p class="muted">Fix <code>'+esc(d.file)+
        '</code> and restart the server.</p>';return;
    }
    let h='';
    if(!d.channels.length)h+='<p><b>No channels yet.</b> Add one under <code>channels</code>.</p>';
    for(const c of d.channels){
      const kv=[['Type',c.type==='webhook'?'Webhook':'Email'],
        ['Status',c.enabled?'On':'Off (enabled: false)']];
      if(c.type==='webhook')kv.push(['Address',c.url]);
      else{kv.push(['Mail server',c.host+':'+c.port+' ('+c.security+')']);
        if(c.username)kv.push(['Login',c.username+(c.password_set?' (password set)':
          ' (no password)')]);
        kv.push(['From',c.sender]);kv.push(['To',(c.to||[]).join(', ')]);}
      kv.push(['Alerts',c.events.join(', ')]);
      if(c.min_severity!=='info')kv.push(['Only',c.min_severity+' and above']);
      h+='<div class="chan"><h3>'+esc(c.name)+'</h3><dl class="kv">'+kv.map(([k,v])=>
        '<dt>'+esc(k)+'</dt><dd class="mono">'+esc(v)+'</dd>').join('')+'</dl>'+
        '<button class="ghost" data-test="'+esc(c.name)+'">Send a test alert</button> '+
        '<span class="muted" id="t_'+esc(c.name)+'"></span></div>';
    }
    const s=d.settings,q=s.quiet_hours;
    h+='<p class="muted" style="margin:6px 0 0;font-size:13px">Camera silent after '+
      esc(s.camera_silent_minutes)+' min - repeats held for '+esc(s.cooldown_hours)+
      ' h - checked every '+esc(s.check_interval_minutes)+' min - quiet hours: '+
      (q?esc(q.start+'-'+q.end):'none')+' - daily summary: '+
      esc(s.daily_summary||'off')+' - background checker: '+
      (d.checker_running?'<b class="ok">running</b>':'<b class="bad">not running</b>'+
      ' (it starts with the server when a channel is enabled)')+'.</p>';
    card.innerHTML=h;
    card.querySelectorAll('[data-test]').forEach(b=>b.addEventListener('click',async()=>{
      const name=b.dataset.test,out=$('t_'+name);out.textContent='Sending...';
      try{
        const r=await call('/api/v1/alerts/test',{method:'POST',
          headers:{'Content-Type':'application/json'},body:JSON.stringify({channel:name})});
        out.innerHTML=r.ok?'<b class="ok">Sent.</b> '+esc(r.result)+' - check it arrived.':
          '<b class="bad">Not delivered:</b> '+esc(r.result);
      }catch(e){out.innerHTML='<b class="bad">'+esc(e.message)+'</b>'}
      loadLog();
    }));
  }catch(e){card.innerHTML=errBox(e.message)}
}

$('checkBtn').addEventListener('click',async()=>{
  const out=$('checkOut'),send=$('sendToo').checked;
  out.innerHTML='<p class="muted">Checking...</p>';
  try{
    const d=await call('/api/v1/alerts/check?send='+send,{method:'POST'});
    let h=d.notes.map(n=>'<p class="note">'+esc(n)+'</p>').join('');
    if(d.quiet_hours)h+='<p class="note">It is quiet hours now: nothing is sent until they '+
      'end.</p>';
    if(!d.events.length)h+='<p class="note"><b class="ok">Nothing needs attention right '+
      'now.</b></p>';
    else h+=table([['severity','Severity','',r=>sev(r.severity)],['type','Event','mono'],
      ['text','Message'],['channels',send?'Before sending':'What would happen','',
      r=>d.configured?chans(r.channels):'<span class="muted">no channels set up</span>']],
      d.events);
    if(send)h+='<p class="note">'+(d.sent&&d.sent.length?'Sent '+d.sent.length+
      ' alert(s) - see Recent alerts below.':'Nothing new was sent.')+'</p>';
    out.innerHTML=h;
    if(send)loadLog();
  }catch(e){out.innerHTML=errBox(e.message)}
});

async function loadLog(){
  const out=$('logOut');
  try{
    const d=await call('/api/v1/alerts/log');
    out.innerHTML=d.entries.length?table([['time','Time','mono'],
      ['severity','Severity','',r=>sev(r.severity)],['event','Event','mono'],
      ['text','Message'],['channels','Delivery','',r=>r.event==='checker_error'?
        '<span class="bad">check failed</span>':chans(r.channels)]],d.entries):
      '<p class="muted">Nothing has been sent yet.</p>';
  }catch(e){out.innerHTML=errBox(e.message)}
}
$('logBtn').addEventListener('click',loadLog);
loadConfig();loadLog();
