"""Built-in zero-build administration console.

The page contains no embedded credentials. Operators enter an API key which is
kept in browser sessionStorage and sent only to Ragbot API endpoints. Cloud
connector secrets are referenced by environment-variable name; secret values
never pass through this UI.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse


def create_admin_ui_router() -> APIRouter:
    router = APIRouter(tags=["admin-ui"])

    @router.get("/admin/ui", response_class=HTMLResponse, include_in_schema=False)
    async def admin_ui() -> str:
        return _HTML

    return router


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ragbot Control Plane</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;color:#172033;background:#f5f7fb}
*{box-sizing:border-box}body{margin:0}.top{background:#111827;color:white;padding:18px 24px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}.top h1{font-size:20px;margin:0;margin-right:auto}.top input{background:#1f2937;border:1px solid #4b5563;color:white;border-radius:7px;padding:8px 10px}.top button{background:#fff;color:#111827}.top .pill{background:#374151;color:#fff}.wrap{max-width:1500px;margin:auto;padding:22px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px}.card,.panel{background:#fff;border:1px solid #e5e7eb;border-radius:10px;box-shadow:0 1px 2px #00000008}.card{padding:16px}.card b{font-size:25px;display:block;margin-top:6px}.muted{color:#6b7280;font-size:12px}.grid{display:grid;grid-template-columns:minmax(300px,390px) 1fr;gap:16px;margin-top:16px}.panel{padding:16px;overflow:auto}h2{font-size:16px;margin:0 0 12px}label{font-size:12px;color:#4b5563;display:block;margin:10px 0 4px}input,select,textarea,button{font:inherit}input,select,textarea{width:100%;border:1px solid #d1d5db;border-radius:7px;padding:8px 9px;background:white}textarea{min-height:84px;resize:vertical;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}button{border:0;border-radius:7px;padding:8px 11px;cursor:pointer;background:#111827;color:white}button:disabled{opacity:.45;cursor:not-allowed}.secondary{background:#e5e7eb;color:#111827}.danger{background:#991b1b}.actions{display:flex;gap:6px;flex-wrap:wrap}.actions button{padding:5px 8px;font-size:12px}table{width:100%;border-collapse:collapse;min-width:780px}th,td{text-align:left;padding:9px 8px;border-bottom:1px solid #eef0f4;font-size:13px;vertical-align:top}th{font-size:11px;text-transform:uppercase;color:#6b7280}.pill{padding:2px 7px;border-radius:99px;background:#eef2ff;display:inline-block;font-size:11px}.ok{background:#dcfce7}.bad{background:#fee2e2}.warn{background:#fef3c7}.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.message{margin:10px 0;padding:9px;border-radius:7px;background:#eef2ff;font-size:12px;white-space:pre-wrap}.permission{margin:0 0 12px;padding:9px;border-radius:7px;background:#f3f4f6;font-size:12px}.hidden{display:none}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="top"><h1>Ragbot Control Plane</h1><span id="roleBadge" class="pill">Disconnected</span><input id="key" placeholder="X-API-Key (session only)" type="password"><input id="tenant" placeholder="tenant filter (optional)"><button onclick="saveAndRefresh()">Connect / Refresh</button></div>
<div class="wrap">
<div id="msg" class="message hidden"></div>
<div class="cards">
 <div class="card"><span class="muted">Sources</span><b id="mSources">-</b></div>
 <div class="card"><span class="muted">Documents</span><b id="mDocs">-</b></div>
 <div class="card"><span class="muted">Chunks</span><b id="mChunks">-</b></div>
 <div class="card"><span class="muted">Pending</span><b id="mPending">-</b></div>
 <div class="card"><span class="muted">Running</span><b id="mRunning">-</b></div>
 <div class="card"><span class="muted">Failed</span><b id="mFailed">-</b></div>
 <div class="card"><span class="muted">Dead Lettered</span><b id="mDeadLettered">-</b></div>
 <div class="card"><span class="muted">Scheduled</span><b id="mScheduled">-</b></div>
</div>
<div class="grid">
 <div class="panel">
  <h2>Quick Import</h2>
  <div id="permissionHint" class="permission">Connect to determine principal permissions.</div>
  <label>Tenant</label><input id="qTenant" placeholder="engineering">
  <label>Location</label><input id="qLocation" placeholder="/data/manuals, s3://..., gdrive://..., notion://..., confluence://...">
  <label>Name</label><input id="qName" placeholder="Engineering manuals">
  <label>Tags (comma separated)</label><input id="qTags" placeholder="manuals,internal">
  <label>Credential reference (cloud/SaaS only)</label><input id="qCredentialRef" placeholder="env:RAGBOT_NOTION_TOKEN">
  <div class="muted" style="margin-top:4px">Only an environment-variable reference is accepted. Never paste a token, password or private key here.</div>
  <label>Connector config JSON (non-secret, optional)</label><textarea id="qConfig" placeholder='{"credential_type":"google_json"} or {"email":"bot@example.com","auth_type":"basic"}'></textarea>
  <label>Idempotency key (optional)</label><input id="qIdempotency" placeholder="bootstrap-2026-09">
  <div style="margin-top:12px"><button id="quickImportButton" onclick="quickImport()">Create / Ingest</button></div>
  <hr style="border:0;border-top:1px solid #eee;margin:18px 0">
  <h2>Queue health</h2><div id="queueHealth" class="muted">Connect to load metrics.</div>
 </div>
 <div class="panel"><h2>Source Catalog</h2><table><thead><tr><th>Source</th><th>Status</th><th>Last index</th><th>Schedule</th><th>Actions</th></tr></thead><tbody id="sourceRows"></tbody></table></div>
</div>
<div class="panel" style="margin-top:16px"><h2>Recent Ingestion Jobs</h2><table><thead><tr><th>Job</th><th>Source</th><th>Status / Failure</th><th>Docs / Chunks</th><th>Timing</th><th>Actions</th></tr></thead><tbody id="jobRows"></tbody></table></div>
</div>
<script>
const $=id=>document.getElementById(id);let sourceById={},capabilities={read:false,operate:false,admin:false};
function headers(){const h={'Content-Type':'application/json'},k=$('key').value.trim();if(k)h['X-API-Key']=k;return h}
function tenantQuery(){const t=$('tenant').value.trim();return t?'?tenant_id='+encodeURIComponent(t):''}
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function msg(text,bad=false){const e=$('msg');e.textContent=text;e.classList.remove('hidden');e.style.background=bad?'#fee2e2':'#eef2ff';setTimeout(()=>e.classList.add('hidden'),7000)}
async function api(path,opts={}){const r=await fetch(path,{...opts,headers:{...headers(),...(opts.headers||{})}});if(!r.ok){let d;try{d=await r.json()}catch{d={detail:await r.text()}}throw new Error(`${r.status}: ${d.detail||JSON.stringify(d)}`)}return r.status===204?null:r.json()}
function saveAndRefresh(){sessionStorage.setItem('ragbotKey',$('key').value);sessionStorage.setItem('ragbotTenant',$('tenant').value);refresh()}
async function refresh(){try{const q=tenantQuery(),[session,o,s,j]=await Promise.all([api('/catalog/session'),api('/catalog/overview'+q),api('/catalog/sources'+q),api('/catalog/jobs'+q)]);renderSession(session);renderOverview(o);renderSources(s.sources);renderJobs(j.jobs)}catch(e){msg(e.message,true)}}
function renderSession(s){capabilities=s.capabilities||{read:true,operate:false,admin:false};const roles=s.admin?['admin']:(s.roles||[]);$('roleBadge').textContent=roles.length?roles.join(' / '):(s.principal_mode==='development'?'development owner':'reader');const canOperate=!!capabilities.operate;$('quickImportButton').disabled=!canOperate;$('permissionHint').textContent=canOperate?'Operator capability enabled: Source, ingestion, retry/requeue and schedule mutations are allowed.':'Read-only principal: catalog/search access is allowed; Source, ingestion, retry/requeue and schedule mutations require operator or owner.'}
function renderOverview(o){$('mSources').textContent=o.sources.total;$('mDocs').textContent=o.knowledge.documents;$('mChunks').textContent=o.knowledge.chunks;$('mPending').textContent=o.queue.pending;$('mRunning').textContent=o.queue.running;$('mFailed').textContent=o.queue.failed;$('mDeadLettered').textContent=o.queue.dead_lettered??0;$('mScheduled').textContent=o.sources.scheduled;$('queueHealth').innerHTML=`Oldest pending: <b>${Math.round(o.queue.oldest_pending_age_seconds)}s</b><br>Stale leases: <b>${o.queue.stale_running_leases}</b><br>Completed 24h: <b>${o.queue.completed_24h}</b><br>Failed 24h: <b>${o.queue.failed_24h}</b><br>Dead-lettered 24h: <b>${o.queue.dead_lettered_24h??0}</b><br>Next sync: <b>${esc(o.sources.next_sync_at||'none')}</b>`}
function statusPill(s){const c=s==='completed'||s==='active'?'ok':s==='failed'||s==='dead_lettered'?'bad':s==='running'||s==='pending'?'warn':'';return `<span class="pill ${c}">${esc(s)}</span>`}
function renderSources(items){sourceById={};$('sourceRows').innerHTML=items.map(s=>{sourceById[s.source_id]=s;const j=s.latest_job||{},li=s.last_index||{},sy=s.sync||{},actions=capabilities.operate?`<div class="actions"><button onclick="ingestNow('${s.source_id}','${esc(s.tenant_id)}')">Ingest</button><button class="secondary" onclick="toggleSource('${s.source_id}','${s.status}')">${s.status==='paused'?'Resume':'Pause'}</button><button class="secondary" onclick="configureSync('${s.source_id}')">Sync</button></div>`:'<span class="muted">read-only</span>';return `<tr><td><b>${esc(s.name)}</b><div class="muted">${esc(s.tenant_id)} · ${esc(s.source_type)} · ${esc(s.location||'')}</div><div class="muted">${esc((s.tags||[]).join(', '))}</div></td><td>${statusPill(s.status)}<div class="muted">job: ${esc(j.status||'never')}</div></td><td>${li.documents||0} docs / ${li.chunks||0} chunks<div class="muted">${esc(li.completed_at||'')}</div></td><td>${sy.enabled?`every ${Math.round((sy.interval_seconds||0)/60)} min<br><span class="muted">next ${esc(sy.next_at||'')}</span>`:'off'}</td><td>${actions}</td></tr>`}).join('')||'<tr><td colspan="5" class="muted">No sources</td></tr>'}
function renderJobs(items){$('jobRows').innerHTML=items.map(j=>{let actions='<span class="muted">—</span>';if(capabilities.operate&&j.status==='failed')actions=`<button onclick="retryJob('${j.job_id}')">Retry</button>`;if(capabilities.operate&&j.status==='dead_lettered')actions=`<button class="danger" onclick="requeueJob('${j.job_id}')">Requeue snapshot</button>`;const failure=j.failure_class?`<div class="muted">class: ${esc(j.failure_class)}</div>`:'';return `<tr><td><code>${esc(j.job_id.slice(0,12))}</code><div class="muted">attempts ${j.attempts||0}</div></td><td>${esc(j.source_id.slice(0,12))}<div class="muted">${esc(j.tenant_id)}</div></td><td>${statusPill(j.status)}${failure}${j.error?`<div class="muted">${esc(j.error.slice(0,140))}</div>`:''}</td><td>${j.doc_count||0} / ${j.stats?.chunks_total??j.chunk_count??0}</td><td><div class="muted">created ${esc(j.created_at||'')}<br>done ${esc(j.completed_at||'')}<br>DLQ ${esc(j.dead_lettered_at||'')}</div></td><td><div class="actions">${actions}</div></td></tr>`}).join('')||'<tr><td colspan="6" class="muted">No jobs</td></tr>'}
function requireOperate(){if(!capabilities.operate)throw new Error('This API principal is read-only; operator or owner role required')}
async function quickImport(){try{requireOperate();const tenant=$('qTenant').value.trim()||$('tenant').value.trim();if(!tenant)throw new Error('tenant is required');const location=$('qLocation').value.trim();if(!location)throw new Error('location is required');const body={tenant_id:tenant,location};if($('qName').value.trim())body.name=$('qName').value.trim();const tags=$('qTags').value.split(',').map(x=>x.trim()).filter(Boolean);if(tags.length)body.tags=tags;let config={};const raw=$('qConfig').value.trim();if(raw){config=JSON.parse(raw);if(!config||Array.isArray(config)||typeof config!=='object')throw new Error('connector config must be a JSON object')}const credentialRef=$('qCredentialRef').value.trim();if(credentialRef)config.credential_ref=credentialRef;if(Object.keys(config).length)body.config=config;if($('qIdempotency').value.trim())body.idempotency_key=$('qIdempotency').value.trim();const r=await api('/ingest/quick',{method:'POST',body:JSON.stringify(body)});msg(`Queued ${r.job_id} for ${r.source_id}`);refresh()}catch(e){msg(e.message,true)}}
async function ingestNow(id,tenant){try{requireOperate();const r=await api('/ingest/jobs',{method:'POST',body:JSON.stringify({source_id:id,tenant_id:tenant})});msg(`Queued ${r.job_id}`);refresh()}catch(e){msg(e.message,true)}}
async function retryJob(id){try{requireOperate();const r=await api(`/ingest/jobs/${id}/retry`,{method:'POST'});msg(`Retry queued ${r.job_id}`);refresh()}catch(e){msg(e.message,true)}}
async function requeueJob(id){try{requireOperate();const r=await api(`/ingest/jobs/${id}/requeue`,{method:'POST',body:JSON.stringify({use_current_source_config:false})});msg(`DLQ snapshot requeued as ${r.job_id}`);refresh()}catch(e){msg(e.message,true)}}
async function toggleSource(id,status){try{requireOperate();await api(`/sources/${id}`,{method:'PUT',body:JSON.stringify({status:status==='paused'?'active':'paused'})});refresh()}catch(e){msg(e.message,true)}}
async function configureSync(id){try{requireOperate();const s=sourceById[id],current=s.sync||{};const raw=prompt('Sync interval in minutes. Enter 0 to disable.',current.enabled?Math.round(current.interval_seconds/60):60);if(raw===null)return;const minutes=Number(raw);if(!Number.isFinite(minutes)||minutes<0)throw new Error('invalid interval');const body=minutes===0?{enabled:false}:{enabled:true,interval_seconds:Math.round(minutes*60),run_immediately:false};await api(`/sources/${id}/sync`,{method:'PUT',body:JSON.stringify(body)});refresh()}catch(e){msg(e.message,true)}}
$('key').value=sessionStorage.getItem('ragbotKey')||'';$('tenant').value=sessionStorage.getItem('ragbotTenant')||'';$('qTenant').value=$('tenant').value;if($('key').value||!location.hostname)refresh();
setInterval(()=>{if(document.visibilityState==='visible')refresh()},15000);
</script></body></html>'''
