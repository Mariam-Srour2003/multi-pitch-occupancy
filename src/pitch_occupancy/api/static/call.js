
async function call(url,opts){
  const r=await fetch(url,opts);let d=null;
  try{d=await r.json()}catch(e){}
  if(!r.ok)throw new Error((d&&d.detail)?(typeof d.detail==='string'?d.detail:
    JSON.stringify(d.detail)):('HTTP '+r.status));
  return d;
}
