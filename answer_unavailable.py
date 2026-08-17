"""Unanswered-questions console for the Jewelry Tech Helpline agent.

Every call the agent could not properly handle is flagged in agent.py and saved
inside the SAME Mongo document as the transcript (see CallSession.cleanup):

    conversations.{ needs_attention, answer_unavailable, no_voice, no_response,
                    unresolved: [ {qid, type, question, spoken_reply, at,
                                   proper_answer} ] }

This file is a small standalone web app over that data. It collects every
`unresolved` entry from every call into one list and lets you:

  * read the caller's exact question (plus the surrounding transcript),
  * delete a question you don't want to keep,
  * filter by type (no answer in bank / no voice / no response) and search.

It is read-and-delete only — answers are written in the /crm console, which adds
them to the agent's approved question bank.

It is NOT a separate server: this router is nested INSIDE the CRM router
(main.py mounts it at /crm/unavailable), so it sits behind the same password and
shares agent.py's Mongo client — the SAME connection and the same MONGODB_URI /
MONGODB_DB credentials. Only the `unresolved` array is ever touched; transcripts
are never modified.

    python main.py
    #  then open  http://localhost:5000/crm/unavailable
"""
import os
import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from bson import ObjectId
from bson.errors import InvalidId

import agent  # same process, same Mongo client, same credentials
import crm     # the password gate this console lives behind

logger = logging.getLogger("answers")

MONGODB_DB = agent.MONGODB_DB
COLLECTION = os.getenv("MONGODB_COLLECTION", "conversations")

TYPE_LABELS = {
    "answer_unavailable": "No answer in bank",
    "no_voice": "Voice didn't play",
    "no_response": "No response",
}


def _connected() -> bool:
    return agent.mongo_client is not None


def _collection():
    # Read the attribute live: agent.py creates the client at import time, and
    # closes it on shutdown.
    return agent.mongo_client.get_database(MONGODB_DB)[COLLECTION]


router = APIRouter(prefix="/unavailable", tags=["unanswered"])


# ── data access ──────────────────────────────────────────────────────────────

def _context(messages: list, question: str) -> list:
    """The few transcript lines around the flagged question, for context."""
    if not messages:
        return []
    idx = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "user" and (m.get("content") or "").strip() == question:
            idx = i
            break
    if idx is None:
        idx = len(messages) - 1
    lo = max(0, idx - 2)
    return [{"role": m.get("role"), "text": (m.get("content") or "")[:400]}
            for m in messages[lo:idx + 3] if m.get("role") != "system"]


async def _load_items() -> list:
    """Flatten every call's `unresolved` array into one newest-first list."""
    cursor = _collection().find(
        {"unresolved": {"$exists": True, "$ne": []}},
        {"caller_id": 1, "call_uuid": 1, "timestamp": 1, "unresolved": 1,
         "messages": 1},
    ).sort("timestamp", -1)

    items = []
    async for doc in cursor:
        for i, u in enumerate(doc.get("unresolved") or []):
            question = (u.get("question") or "").strip()
            utype = u.get("type") or "answer_unavailable"
            at = u.get("at") or doc.get("timestamp")
            items.append({
                "conv_id": str(doc["_id"]),
                # qid is written by newer calls; older documents fall back to
                # the array position (verified by question text on write).
                "qid": u.get("qid"),
                "index": i,
                "type": utype,
                "type_label": TYPE_LABELS.get(utype, utype),
                "question": question,
                "spoken_reply": u.get("spoken_reply") or "",
                "at": at.isoformat() if hasattr(at, "isoformat") else str(at or ""),
                "caller_id": doc.get("caller_id") or "",
                "call_uuid": doc.get("call_uuid") or "",
                "context": _context(doc.get("messages") or [], question),
            })
    items.sort(key=lambda x: x["at"], reverse=True)
    return items


def _filter(items: list, kind: str, search: str) -> list:
    needle = search.strip().lower()
    out = []
    for it in items:
        if kind and it["type"] != kind:
            continue
        if needle and needle not in it["question"].lower():
            continue
        out.append(it)
    return out


# ── API ──────────────────────────────────────────────────────────────────────

class ItemRef(BaseModel):
    conv_id: str
    qid: str | None = None
    index: int = 0
    question: str = ""


@router.get("/api/items")
async def list_items(request: Request, type: str = "", q: str = ""):
    if not crm._authed(request):
        return JSONResponse({"error": "not signed in"}, status_code=401)
    if not _connected():
        return JSONResponse({"error": "MONGODB_URI not set"}, status_code=503)
    # One DB pass: stats always describe EVERYTHING, the list is the filtered view.
    everything = await _load_items()
    stats = {"total": 0, "answer_unavailable": 0, "no_voice": 0, "no_response": 0}
    for it in everything:
        stats["total"] += 1
        if it["type"] in stats:
            stats[it["type"]] += 1
    return {"items": _filter(everything, type, q), "stats": stats}


@router.post("/api/delete")
async def delete_item(request: Request, body: ItemRef):
    """Remove one flagged question from its call document."""
    if not crm._authed(request):
        return JSONResponse({"error": "not signed in"}, status_code=401)
    if not _connected():
        return JSONResponse({"error": "MONGODB_URI not set"}, status_code=503)
    try:
        oid = ObjectId(body.conv_id)
    except (InvalidId, TypeError):
        return JSONResponse({"error": "bad conversation id"}, status_code=400)
    if body.qid:
        pull = {"unresolved": {"qid": body.qid}}
    else:
        # No qid: pull by exact question text — and only from THIS call.
        pull = {"unresolved": {"question": body.question}}
    res = await _collection().update_one({"_id": oid}, {"$pull": pull})
    if res.modified_count == 0:
        return JSONResponse({"error": "question no longer exists"}, status_code=404)
    # Keep the call's summary flags honest once its last item is gone.
    doc = await _collection().find_one({"_id": oid}, {"unresolved": 1})
    if doc is not None and not (doc.get("unresolved") or []):
        await _collection().update_one({"_id": oid}, {"$set": {
            "needs_attention": False, "answer_unavailable": False,
            "no_voice": False, "no_response": False}})
    logger.info(f"Deleted flagged question '{body.question[:60]}'")
    return {"ok": True}


@router.get("/health")
async def health():
    return {"status": "healthy", "db": MONGODB_DB, "collection": COLLECTION,
            "connected": _connected()}


# ── page ─────────────────────────────────────────────────────────────────────

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unanswered questions</title>
<style>
  :root{--bg:#f6f7f9;--card:#ffffff;--line:#e4e7ec;--fg:#111827;--dim:#667085;
        --accent:#2563eb;--accent-soft:#eef4ff;--red:#d92d20;--red-soft:#fef3f2;
        --amber:#b54708;--amber-soft:#fffaeb;--blue-soft:#eff8ff;
        --shadow:0 1px 2px rgba(16,24,40,.05), 0 1px 3px rgba(16,24,40,.06);}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);-webkit-font-smoothing:antialiased;
       font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header{position:sticky;top:0;z-index:5;padding:14px 24px;background:var(--card);
         border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:16px;
         align-items:center}
  h1{font-size:17px;margin:0;font-weight:600;letter-spacing:-.01em}
  .stats{display:flex;gap:14px;flex-wrap:wrap;color:var(--dim);font-size:13px}
  .stats b{color:var(--fg)}
  main{max-width:1000px;margin:0 auto;padding:24px 24px 70px}
  .bar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}
  select,input[type=text]{background:#fff;color:var(--fg);border:1px solid var(--line);
       border-radius:8px;padding:9px 11px;font:inherit}
  select:focus,input[type=text]:focus{outline:none;border-color:var(--accent);
       box-shadow:0 0 0 3px var(--accent-soft)}
  input[type=text]{flex:1;min-width:200px}
  ::placeholder{color:#98a2b3}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
        padding:16px 18px;margin-bottom:14px;box-shadow:var(--shadow)}
  .row{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}
  .q{font-size:15.5px;font-weight:600;margin:0 0 6px;letter-spacing:-.01em}
  .meta{color:var(--dim);font-size:12.5px;display:flex;gap:10px;flex-wrap:wrap;
        align-items:center}
  .tag{border-radius:999px;padding:3px 10px;font-size:11.5px;font-weight:500;
       white-space:nowrap;border:1px solid var(--line);background:#f9fafb}
  .t-answer_unavailable{background:var(--amber-soft);color:var(--amber);
       border-color:#fedf89}
  .t-no_voice{background:var(--red-soft);color:var(--red);border-color:#fda29b}
  .t-no_response{background:var(--blue-soft);color:#175cd3;border-color:#b2ddff}
  .acts{display:flex;gap:8px;margin-top:12px;align-items:center}
  button,a.nav{display:inline-flex;align-items:center;gap:7px;white-space:nowrap;
       border:1px solid transparent;border-radius:8px;padding:7px 13px;font:inherit;
       font-size:14px;font-weight:500;line-height:1.2;cursor:pointer;
       transition:background .12s,border-color .12s,box-shadow .12s}
  button:focus-visible,a.nav:focus-visible{outline:none;
       box-shadow:0 0 0 3px var(--accent-soft)}
  .ico{width:16px;height:16px;flex:none;stroke:currentColor;fill:none;
       stroke-width:1.75;stroke-linecap:round;stroke-linejoin:round}
  .del{background:#fff;color:var(--red);border-color:#fda29b;font-size:13px}
  .del:hover{background:var(--red-soft)}
  .ctx{margin-top:10px;font-size:13px;color:var(--dim);background:#f9fafb;
       border:1px solid var(--line);border-radius:8px;padding:10px 12px;display:none}
  .ctx.open{display:block}
  .ctx b{color:var(--fg)}
  .link{background:none;border:0;color:var(--accent);padding:0;font-size:12.5px;
       gap:5px}
  .link:hover{text-decoration:underline}
  .ok{color:var(--dim);font-size:13px}
  .empty{color:var(--dim);text-align:center;padding:60px 0}
  .nav{margin-left:auto;color:var(--fg);background:#fff;text-decoration:none;
       font-size:14px;border:1px solid var(--line);border-radius:8px;padding:8px 14px}
  .nav:hover{background:#f9fafb;border-color:#d0d5dd}
  /* ── narrow screens ───────────────────────────────────────────────────── */
  @media (max-width:900px){
    main{padding:18px 16px 60px}
    header{padding:12px 16px}
  }
  @media (max-width:720px){
    header{padding:10px 12px;gap:8px}
    h1{font-size:16px}
    .stats{gap:10px;font-size:12.5px;order:3;width:100%}
    .nav{margin-left:auto;font-size:13px;padding:7px 11px}
    main{padding:14px 12px 56px}
    select,input[type=text]{font-size:16px;padding:10px 11px}
    select{width:100%}
    .card{padding:14px;border-radius:10px}
    /* Question and type tag stack, so neither gets squeezed to a sliver. */
    .row{flex-direction:column;gap:8px}
    .row .tag{align-self:flex-start;order:-1}
    .q{font-size:15px}
    .meta{gap:8px;font-size:12px}
    button,a.nav{padding:8px 12px}
  }
</style></head><body>
<header>
  <h1>Unanswered questions</h1>
  <div class="stats" id="stats"></div>
  <a class="nav" href="/crm"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M19 12H5"/><path d="m12 19-7-7 7-7"/></svg> Question bank CRM</a>
</header>
<main>
  <div class="bar">
    <select id="type">
      <option value="">All types</option>
      <option value="answer_unavailable">No answer in bank</option>
      <option value="no_voice">Voice didn't play</option>
      <option value="no_response">No response</option>
    </select>
    <input type="text" id="q" placeholder="Search question or answer…">
  </div>
  <div id="list"></div>
</main>
<script>
// The page is served under /crm/unavailable, inside the password-protected CRM,
// so every call is made relative to that prefix (trailing slash or not).
const API = '/crm/unavailable';
const esc = s => (s||"").replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let items = [];

async function load(){
  const p = new URLSearchParams({type:type_.value, q:q_.value});
  const r = await fetch(API+'/api/items?'+p);
  const d = await r.json();
  if(d.error){ list.innerHTML = '<div class="empty">'+esc(d.error)+'</div>'; return; }
  items = d.items;
  stats.innerHTML = `<span><b>${d.stats.total}</b> flagged</span>
    <span>no answer in bank <b>${d.stats.answer_unavailable}</b></span>
    <span>no voice <b>${d.stats.no_voice}</b></span>
    <span>no response <b>${d.stats.no_response}</b></span>`;
  render();
}

function render(){
  if(!items.length){ list.innerHTML = '<div class="empty">Nothing here.</div>'; return; }
  list.innerHTML = items.map((it,i) => `
    <div class="card">
      <div class="row">
        <div>
          <p class="q">${esc(it.question) || '<i>(no transcript — greeting or silent turn)</i>'}</p>
          <div class="meta">
            <span>${esc(it.at).replace('T',' ').slice(0,19)}</span>
            <span>caller: ${esc(it.caller_id)||'—'}</span>
            ${it.spoken_reply ? `<span>said: “${esc(it.spoken_reply.slice(0,70))}”</span>` : ''}
            <button class="link" onclick="ctx(${i})"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2Z"/></svg> transcript</button>
          </div>
        </div>
        <span class="tag t-${it.type}">${esc(it.type_label)}</span>
      </div>
      <div class="ctx" id="ctx${i}">${it.context.map(c =>
          `<div><b>${c.role==='user'?'Caller':'Agent'}:</b> ${esc(c.text)}</div>`).join('')
          || '<i>no transcript stored</i>'}</div>
      <div class="acts">
        <button class="del" onclick="del(${i})"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/></svg> Delete</button>
      </div>
    </div>`).join('');
}

function ctx(i){ document.getElementById('ctx'+i).classList.toggle('open'); }

function ref(it){ return {conv_id:it.conv_id, qid:it.qid, index:it.index, question:it.question}; }

async function post(url, body){
  const r = await fetch(API+url, {method:'POST', headers:{'Content-Type':'application/json'},
                                  body: JSON.stringify(body)});
  return r.json();
}

async function del(i){
  const it = items[i];
  if(!confirm('Delete this question from the database?\\n\\n' + (it.question || '(no text)'))) return;
  const d = await post('/api/delete', ref(it));
  if(d.error){ alert(d.error); return; }
  load();
}

const type_ = document.getElementById('type'), q_ = document.getElementById('q'),
      list = document.getElementById('list'), stats = document.getElementById('stats');
type_.onchange = load;
let t; q_.oninput = () => { clearTimeout(t); t = setTimeout(load, 250); };
load();
</script></body></html>"""


@router.get("")          # /crm/unavailable
@router.get("/")         # /crm/unavailable/
async def index(request: Request):
    if not crm._authed(request):
        return RedirectResponse("/crm", status_code=303)
    return HTMLResponse(PAGE)
