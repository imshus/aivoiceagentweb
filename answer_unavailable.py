"""Unanswered-questions console for the MRPscan agent.

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

A flagged question can also be RESOLVED straight from here, two ways:

  * as a MAIN question — dictate its answer with the mic and submit, and the
    question goes into the bank as a brand-new entry (same write path as the
    /crm console: faq_router.py + agent.py + the running bank + a pre-warmed
    Hinglish wording and its audio);
  * as a SUB-QUESTION — pick an existing bank entry and the flagged question is
    attached to it as another phrasing of that same question, so the next
    caller who asks it that way is answered by the entry that already exists.

Either way the flagged item leaves the queue once the bank write succeeds.

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
import faq_router as fr  # the question bank a flagged question gets folded into

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


async def _pull_item(body: ItemRef) -> tuple[bool, str]:
    """Remove one flagged question from its call document -> (ok, error).

    Shared by the plain Delete button and by both resolve paths, so a question
    that has just been written into the bank leaves the queue by exactly the
    same route as one thrown away by hand.
    """
    try:
        oid = ObjectId(body.conv_id)
    except (InvalidId, TypeError):
        return False, "bad conversation id"
    if body.qid:
        pull = {"unresolved": {"qid": body.qid}}
    else:
        # No qid: pull by exact question text - and only from THIS call.
        pull = {"unresolved": {"question": body.question}}
    res = await _collection().update_one({"_id": oid}, {"$pull": pull})
    if res.modified_count == 0:
        return False, "question no longer exists"
    # Keep the call's summary flags honest once its last item is gone.
    doc = await _collection().find_one({"_id": oid}, {"unresolved": 1})
    if doc is not None and not (doc.get("unresolved") or []):
        await _collection().update_one({"_id": oid}, {"$set": {
            "needs_attention": False, "answer_unavailable": False,
            "no_voice": False, "no_response": False}})
    return True, ""


@router.post("/api/delete")
async def delete_item(request: Request, body: ItemRef):
    """Throw one flagged question away without answering it."""
    if not crm._authed(request):
        return JSONResponse({"error": "not signed in"}, status_code=401)
    if not _connected():
        return JSONResponse({"error": "MONGODB_URI not set"}, status_code=503)
    ok, err = await _pull_item(body)
    if not ok:
        return JSONResponse({"error": err},
                            status_code=400 if "id" in err else 404)
    logger.info(f"Deleted flagged question '{body.question[:60]}'")
    return {"ok": True}


class EditIn(ItemRef):
    """Correct the text of one flagged question."""
    new_question: str


@router.post("/api/edit")
async def edit_item(request: Request, body: EditIn):
    """Rewrite the caller's question as it is stored on the call document.

    Speech-to-text mishears things, and the text saved here is exactly what
    gets written into the bank when the question is resolved - so fixing a
    garbled transcript before answering it is the difference between a usable
    bank entry and a dead one. ONLY the `question` field of that one array
    element changes; the transcript, the spoken reply and the flag type are
    left alone.
    """
    if not crm._authed(request):
        return JSONResponse({"error": "not signed in"}, status_code=401)
    if not _connected():
        return JSONResponse({"error": "MONGODB_URI not set"}, status_code=503)
    text = (body.new_question or "").strip()
    if not text:
        return JSONResponse({"error": "the question cannot be empty"},
                            status_code=400)
    try:
        oid = ObjectId(body.conv_id)
    except (InvalidId, TypeError):
        return JSONResponse({"error": "bad conversation id"}, status_code=400)
    # Match the element the same way the delete path does: by qid when the call
    # recorded one, otherwise by its exact text within THIS call.
    if body.qid:
        match = {"_id": oid, "unresolved.qid": body.qid}
    else:
        match = {"_id": oid, "unresolved.question": body.question}
    res = await _collection().update_one(
        match, {"$set": {"unresolved.$.question": text}})
    if res.matched_count == 0:
        return JSONResponse({"error": "question no longer exists"},
                            status_code=404)
    logger.info(f"Edited flagged question '{body.question[:40]}' -> '{text[:40]}'")
    return {"ok": True, "question": text}


# -- resolving a flagged question into the bank -------------------------------

@router.get("/api/bank")
async def bank(request: Request):
    """Every question in the bank - the list you pick from on the
    'existing question' tab."""
    if not crm._authed(request):
        return JSONResponse({"error": "not signed in"}, status_code=401)
    items = [{"qid": qid,
              "question": fr.canon_q(entry),
              "subquestions": fr.all_q(entry)[1:],
              "answer": entry["a"],
              "custom": qid in fr.CUSTOM_IDS}
             for qid, entry in fr.CANONICAL_ANSWERS.items()]
    return {"items": items, "total": len(items)}


class ResolveMainIn(ItemRef):
    """Answer the flagged question as a NEW main question in the bank."""
    answer: str
    subquestions: list[str] = []
    clean: bool = True


class ResolveSubIn(ItemRef):
    """Attach the flagged question to an EXISTING entry as another phrasing."""
    target_qid: str


@router.post("/api/resolve/main")
async def resolve_main(request: Request, body: ResolveMainIn):
    """Flagged question + a dictated answer -> a new bank entry.

    The same write path as /crm/api/save: the optional clean-up, then
    faq_router.add_entry (which writes faq_router.py, agent.py and the running
    bank), then a pre-warm of that ONE entry's Hinglish wording and audio. The
    flagged item is pulled from Mongo only after the bank write succeeds, so a
    failure here leaves the queue exactly as it was.
    """
    if not crm._authed(request):
        return JSONResponse({"error": "not signed in"}, status_code=401)
    if not _connected():
        return JSONResponse({"error": "MONGODB_URI not set"}, status_code=503)
    question = (body.question or "").strip()
    answer = (body.answer or "").strip()
    subs = [s.strip() for s in body.subquestions if s and s.strip()]
    if not question:
        return JSONResponse({"error": "this flagged turn has no question text - "
                                      "delete it, or add it from the CRM"},
                            status_code=400)
    if not answer:
        return JSONResponse({"error": "record an answer first"}, status_code=400)

    cleaned = {"question": question, "subquestions": subs, "answer": answer}
    if body.clean:
        if not crm.OPENAI_API_KEY:
            return JSONResponse(
                {"error": "OPENAI_API_KEY is not set in .env - add it, or switch "
                          "off 'clean up with AI' to save the text as recorded"},
                status_code=503)
        try:
            cleaned = await crm._clean_with_llm(question, subs, answer)
        except Exception as e:
            logger.error(f"Clean-up failed: {e}")
            return JSONResponse({"error": f"clean-up failed: {e}"}, status_code=502)
        if not cleaned["question"] or not cleaned["answer"]:
            return JSONResponse(
                {"error": "the recording did not contain a usable answer - "
                          "record again, or switch off the clean-up"},
                status_code=422)
        # The caller's own wording is what the NEXT caller is likely to use, so
        # keep it as a phrasing even when the clean-up tidied the question up.
        if cleaned["question"].lower() != question.lower():
            cleaned["subquestions"] = [question] + [
                s for s in cleaned["subquestions"]
                if s.lower() != question.lower()]

    try:
        qid = fr.add_entry(cleaned["question"], cleaned["subquestions"],
                           cleaned["answer"])
    except Exception as e:
        logger.error(f"Bank write failed: {e}")
        return JSONResponse({"error": f"could not save: {e}"}, status_code=500)

    warm = await crm._prewarm(qid)
    ok, err = await _pull_item(body)
    logger.info(f"Resolved '{question[:60]}' as new bank entry {qid}")
    return {"ok": True, "qid": qid, "entry": crm._entry_view(qid),
            "prewarm": warm, "removed": ok,
            "warning": "" if ok else f"saved as {qid}, but the flagged item "
                                     f"could not be removed ({err})"}


@router.post("/api/resolve/sub")
async def resolve_sub(request: Request, body: ResolveSubIn):
    """Attach the flagged question to an existing entry as a sub-question.

    That entry's own question and answer are left untouched - only its list of
    phrasings grows - so its approved wording stays valid and no audio is
    re-rendered or re-billed.
    """
    if not crm._authed(request):
        return JSONResponse({"error": "not signed in"}, status_code=401)
    if not _connected():
        return JSONResponse({"error": "MONGODB_URI not set"}, status_code=503)
    question = (body.question or "").strip()
    if not question:
        return JSONResponse({"error": "this flagged turn has no question text"},
                            status_code=400)
    entry = fr.CANONICAL_ANSWERS.get(body.target_qid)
    if entry is None:
        return JSONResponse({"error": "no such question in the bank"},
                            status_code=404)

    forms = fr.all_q(entry)
    if any(question.lower() == f.lower() for f in forms):
        # Already a phrasing of this entry: the bank needs no write, but the
        # flagged item still belongs off the queue.
        ok, err = await _pull_item(body)
        return {"ok": True, "qid": body.target_qid, "already": True,
                "entry": crm._entry_view(body.target_qid), "removed": ok,
                "warning": "" if ok else err}

    try:
        fr.update_entry(body.target_qid, forms[0], forms[1:] + [question],
                        entry["a"])
    except KeyError:
        return JSONResponse({"error": "no such question in the bank"},
                            status_code=404)
    except Exception as e:
        logger.error(f"Attach failed for {body.target_qid}: {e}")
        return JSONResponse({"error": f"could not attach: {e}"}, status_code=500)

    ok, err = await _pull_item(body)
    logger.info(f"Attached '{question[:60]}' to {body.target_qid} as "
                f"sub-question {len(forms)}")
    return {"ok": True, "qid": body.target_qid, "already": False,
            "entry": crm._entry_view(body.target_qid), "removed": ok,
            "warning": "" if ok else f"attached to {body.target_qid}, but the "
                                     f"flagged item could not be removed ({err})"}


@router.get("/health")
async def health():
    return {"status": "healthy", "db": MONGODB_DB, "collection": COLLECTION,
            "connected": _connected()}


# ── page ─────────────────────────────────────────────────────────────────────

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MRPscan — unanswered questions</title>
<style>
  :root{--bg:#f6f7f9;--card:#ffffff;--line:#e4e7ec;--fg:#111827;--dim:#667085;
        --accent:#2563eb;--accent-soft:#eef4ff;--red:#d92d20;--red-soft:#fef3f2;
        --amber:#b54708;--amber-soft:#fffaeb;--blue-soft:#eff8ff;
        --shadow:0 1px 2px rgba(16,24,40,.05), 0 1px 3px rgba(16,24,40,.06);}
  *{box-sizing:border-box}
  /* iOS keeps scrolling the page behind a fixed overlay unless it is pinned. */
  body.locked{overflow:hidden}
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
  .fix{background:var(--accent);color:#fff;border-color:var(--accent);font-size:13px}
  .edit{background:#fff;color:var(--fg);border-color:var(--line);font-size:13px}
  .edit:hover{background:#f9fafb;border-color:#d0d5dd}
  .qedit{display:none;margin:0 0 8px}
  .qedit.open{display:block}
  .qedit textarea{min-height:64px;font-size:15px}
  .qedit .acts{margin-top:8px}
  .card.editing .q,.card.editing .acts:not(.qedit .acts){display:none}
  .fix:hover{background:#1d4ed8}
  /* ── resolve dialog ───────────────────────────────────────────────────── */
  .veil{position:fixed;inset:0;background:rgba(16,24,40,.45);z-index:20;
        display:none;align-items:flex-start;justify-content:center;padding:32px 16px;
        overflow-y:auto}
  .veil.open{display:flex}
  .modal{background:var(--card);border-radius:14px;width:100%;max-width:660px;
         display:flex;flex-direction:column;max-height:calc(100dvh - 64px);
         box-shadow:0 20px 24px -4px rgba(16,24,40,.10),0 8px 8px -4px rgba(16,24,40,.04)}
  .mhead{padding:18px 20px 0;position:relative;flex:none}
  .mhead h2{font-size:16px;margin:0 0 4px;font-weight:600;letter-spacing:-.01em}
  .mhead .asked{color:var(--dim);font-size:13px;margin:0 0 14px}
  .x{position:absolute;top:14px;right:14px;background:none;border:0;color:var(--dim);
     padding:4px;border-radius:6px}
  .x:hover{background:#f2f4f7;color:var(--fg)}
  .tabs{display:flex;gap:4px;padding:0 20px;border-bottom:1px solid var(--line);
        flex:none}
  .tab{background:none;border:0;border-bottom:2px solid transparent;border-radius:0;
       color:var(--dim);padding:10px 4px;margin-right:18px;font-size:14px}
  .tab.on{color:var(--accent);border-bottom-color:var(--accent);font-weight:600}
  .pane{display:none;padding:18px 20px}
  /* The panes are what scrolls; head, tabs and footer stay put. */
  .pane.on{display:block;flex:1 1 auto;min-height:0;overflow-y:auto;
           -webkit-overflow-scrolling:touch}
  .label{font-size:13px;font-weight:600;color:var(--dim);margin:0 0 6px;
         display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  textarea{width:100%;background:#fff;color:var(--fg);border:1px solid var(--line);
       border-radius:8px;padding:10px 11px;font:inherit;min-height:110px;resize:vertical}
  textarea:focus{outline:none;border-color:var(--accent);
       box-shadow:0 0 0 3px var(--accent-soft)}
  .mic{background:#fff;color:var(--accent);border-color:#b2ddff;font-size:13px;
       padding:5px 11px}
  .mic:hover{background:var(--blue-soft)}
  .mic.rec{background:var(--red);border-color:var(--red);color:#fff}
  .mic.busy{background:#f9fafb;color:var(--dim);border-color:var(--line)}
  .spin{animation:sp 1s linear infinite}
  @keyframes sp{to{transform:rotate(360deg)}}
  .picker{border:1px solid var(--line);border-radius:8px}
  .pick{display:flex;gap:10px;padding:10px 12px;border-bottom:1px solid var(--line);
        cursor:pointer;align-items:flex-start}
  .pick:last-child{border-bottom:0}
  .pick:hover{background:#f9fafb}
  .pick.sel{background:var(--accent-soft)}
  .pick input{margin-top:4px;flex:none;accent-color:var(--accent)}
  .pick .pq{font-size:14px;font-weight:500;margin:0}
  .pick .pa{color:var(--dim);font-size:12.5px;margin:3px 0 0}
  .pick .pid{color:var(--dim);font-size:11.5px;font-weight:600}
  .mfoot{display:flex;gap:10px;align-items:center;flex-wrap:wrap;flex:none;
         padding:14px 20px;border-top:1px solid var(--line);
         padding-bottom:calc(14px + env(safe-area-inset-bottom))}
  .mfoot .msg{font-size:13px;color:var(--dim);flex:1;min-width:120px}
  .mfoot .msg.bad{color:var(--red)}
  .go{background:var(--accent);color:#fff;border-color:var(--accent)}
  .go:hover{background:#1d4ed8}
  .go[disabled]{opacity:.55;cursor:not-allowed}
  .ghost{background:#fff;border-color:var(--line);color:var(--fg)}
  .ghost:hover{background:#f9fafb}
  .chk{display:flex;align-items:center;gap:7px;font-size:13px;color:var(--dim)}
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
    /* Tap targets stay finger-sized, and the three card buttons wrap
       instead of squeezing themselves into slivers. */
    button,a.nav{padding:9px 13px;min-height:40px}
    .acts{flex-wrap:wrap}
    .link{min-height:0;padding:0}
    .meta button{padding:0}
    /* The dialog becomes a full-screen sheet. */
    .veil{padding:0;align-items:stretch}
    .modal{border-radius:0;max-width:none;height:100dvh;max-height:100dvh}
    .mhead{padding:16px 14px 0}
    .mhead h2{font-size:16px;padding-right:34px}
    .x{top:12px;right:10px;padding:8px}
    .tabs{padding:0 14px;gap:0}
    .tab{flex:1;margin-right:0;text-align:center;justify-content:center;
         padding:12px 4px;font-size:13.5px}
    .pane{padding:16px 14px}
    .mfoot{padding:12px 14px;padding-bottom:calc(12px + env(safe-area-inset-bottom))}
    /* Submit and Cancel take the full row so neither is a thumb-miss. */
    .mfoot .msg{order:-1;width:100%;flex:none}
    .mfoot .ghost,.mfoot .go{flex:1;justify-content:center}
    .mic{min-height:36px}
    textarea{font-size:16px}   /* 16px stops iOS zooming into the field */
    .qedit textarea{min-height:80px}
    .pick{padding:12px}
  }
</style></head><body>
<header>
  <h1>MRPscan — unanswered questions</h1>
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

<!-- Resolve dialog: tab 1 answers the question outright, tab 2 files it under
     a question the bank already answers. -->
<div class="veil" id="veil">
 <div class="modal" role="dialog" aria-modal="true" aria-labelledby="mtitle">
  <div class="mhead">
    <button class="x" onclick="closeFix()" aria-label="Close"><svg class="ico" viewBox="0 0 24 24"><path d="M18 6 6 18M6 6l12 12"/></svg></button>
    <h2 id="mtitle">Answer this question</h2>
    <p class="asked">Caller asked: <b id="asked"></b></p>
  </div>
  <div class="tabs">
    <button class="tab on" id="tabMain" onclick="tab('main')">Main question</button>
    <button class="tab" id="tabSub" onclick="tab('sub')">Sub-question</button>
  </div>

  <div class="pane on" id="paneMain">
    <p class="label">Answer — record it, then edit the text if you need to
      <button class="mic" id="micBtn" onclick="rec(this,'ans')"><svg class="ico" viewBox="0 0 24 24"><path d="M12 2a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3Z"/><path d="M19 10v1a7 7 0 0 1-14 0v-1"/><path d="M12 19v3"/></svg> Record</button>
    </p>
    <textarea id="ans" placeholder="Speak the answer, or type it here…"></textarea>
    <p class="label" style="margin-top:14px">The question goes into the bank exactly as the caller asked it, with this answer.</p>
  </div>

  <div class="pane" id="paneSub">
    <p class="label">Pick the question in the bank that already answers this one — the caller's wording is added to it as another phrasing.</p>
    <input type="text" id="bankq" placeholder="Search the bank…" style="width:100%;margin-bottom:10px">
    <div class="picker" id="picker"></div>
  </div>

  <div class="mfoot">
    <label class="chk" id="cleanWrap"><input type="checkbox" id="clean" checked> clean up with AI</label>
    <span class="msg" id="mmsg"></span>
    <button class="ghost" onclick="closeFix()">Cancel</button>
    <button class="go" id="submit" onclick="submitFix()">Submit</button>
  </div>
 </div>
</div>
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
          <div class="qedit" id="qe${i}">
            <textarea id="qt${i}">${esc(it.question)}</textarea>
            <div class="acts">
              <button class="fix" onclick="saveEdit(${i})"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg> Save</button>
              <button class="edit" onclick="editQ(${i}, false)">Cancel</button>
            </div>
          </div>
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
        <button class="edit" onclick="editQ(${i}, true)"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg> Edit</button>
        <button class="fix" onclick="openFix(${i})"><svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg> Answer it</button>
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
  // No confirmation step: one click removes it.
  const d = await post('/api/delete', ref(it));
  if(d.error){ alert(d.error); return; }
  load();
}

// ── editing the flagged question's own text ────────────────────────────────
function editQ(i, on){
  const card = document.getElementById('qe'+i).closest('.card');
  card.classList.toggle('editing', on);
  document.getElementById('qe'+i).classList.toggle('open', on);
  if(on){
    const t = document.getElementById('qt'+i);
    t.value = items[i].question;   // reopening always starts from what is stored
    t.focus();
  }
}

async function saveEdit(i){
  const it = items[i];
  const text = document.getElementById('qt'+i).value.trim();
  if(!text){ alert('The question cannot be empty. Delete it instead.'); return; }
  if(text === it.question){ editQ(i, false); return; }
  const d = await post('/api/edit', {...ref(it), new_question: text});
  if(d.error){ alert(d.error); return; }
  load();
}

// ── resolve: answer it, or file it under a question the bank already has ────
let fixing = null;    // the flagged item being resolved
let mode = 'main';    // which tab is showing
let bankItems = [], picked = null;

function openFix(i){
  fixing = items[i];
  if(!fixing.question){ alert('This flagged turn has no question text — there is nothing to add to the bank. Delete it instead.'); return; }
  picked = null;
  document.getElementById('asked').textContent = fixing.question;
  document.getElementById('ans').value = '';
  document.getElementById('bankq').value = '';
  msg('');
  tab('main');
  document.getElementById('veil').classList.add('open');
  document.body.classList.add('locked');
  loadBank();
}

function closeFix(){
  document.getElementById('veil').classList.remove('open');
  document.body.classList.remove('locked');
  fixing = null;
}

function tab(which){
  mode = which;
  document.getElementById('tabMain').classList.toggle('on', which==='main');
  document.getElementById('tabSub').classList.toggle('on', which==='sub');
  document.getElementById('paneMain').classList.toggle('on', which==='main');
  document.getElementById('paneSub').classList.toggle('on', which==='sub');
  // The clean-up only applies to a dictated answer, not to picking an entry.
  document.getElementById('cleanWrap').style.display = which==='main' ? '' : 'none';
  msg('');
}

async function loadBank(){
  if(bankItems.length){ renderBank(); return; }
  document.getElementById('picker').innerHTML =
    '<div style="padding:14px;color:var(--dim);font-size:13px">Loading the bank…</div>';
  const r = await fetch(API+'/api/bank');
  const d = await r.json();
  if(d.error){ document.getElementById('picker').innerHTML =
      '<div style="padding:14px;color:var(--dim);font-size:13px">'+esc(d.error)+'</div>'; return; }
  bankItems = d.items;
  renderBank();
}

function renderBank(){
  const needle = document.getElementById('bankq').value.trim().toLowerCase();
  const rows = bankItems.filter(b => !needle
      || b.question.toLowerCase().includes(needle)
      || b.answer.toLowerCase().includes(needle)
      || b.subquestions.some(x => x.toLowerCase().includes(needle)));
  document.getElementById('picker').innerHTML = rows.length ? rows.map(b => `
    <label class="pick ${picked===b.qid?'sel':''}">
      <input type="radio" name="bank" value="${esc(b.qid)}" ${picked===b.qid?'checked':''}
             onchange="pick('${esc(b.qid)}')">
      <span>
        <p class="pq"><span class="pid">${esc(b.qid)}</span> ${esc(b.question)}</p>
        <p class="pa">${esc(b.answer.slice(0,140))}${b.answer.length>140?'…':''}</p>
        ${b.subquestions.length ? `<p class="pa">${b.subquestions.length} phrasing(s) already attached</p>` : ''}
      </span>
    </label>`).join('')
    : '<div style="padding:14px;color:var(--dim);font-size:13px">No question matches that.</div>';
}

function pick(qid){ picked = qid; renderBank(); }

function msg(text, bad){
  const el = document.getElementById('mmsg');
  el.textContent = text || '';
  el.classList.toggle('bad', !!bad);
}

async function submitFix(){
  if(!fixing) return;
  const btn = document.getElementById('submit');
  let url, body;
  if(mode === 'main'){
    const answer = document.getElementById('ans').value.trim();
    if(!answer){ msg('Record or type an answer first.', true); return; }
    url = '/api/resolve/main';
    body = {...ref(fixing), answer, clean: document.getElementById('clean').checked};
  }else{
    if(!picked){ msg('Pick the question it belongs to.', true); return; }
    url = '/api/resolve/sub';
    body = {...ref(fixing), target_qid: picked};
  }
  btn.disabled = true;
  msg(mode==='main' ? 'Saving to the bank and preparing the voice…' : 'Attaching…');
  let d;
  try{ d = await post(url, body); }
  catch(e){ d = {error: 'request failed: ' + e.message}; }
  btn.disabled = false;
  if(d.error){ msg(d.error, true); return; }
  if(d.warning) alert(d.warning);
  if(mode === 'main' && d.prewarm && !d.prewarm.audio)
    alert('Saved as ' + d.qid + ', but its voice clip could not be prepared'
          + (d.prewarm.error ? ': ' + d.prewarm.error : '')
          + '. Use Pre-warm in the CRM to retry.');
  // The bank changed under us — reload it before the next pick.
  bankItems = [];
  closeFix();
  load();
}

document.getElementById('bankq').oninput = renderBank;
document.getElementById('veil').onclick = e => {
  if(e.target.id === 'veil') closeFix();
};
document.addEventListener('keydown', e => {
  if(e.key === 'Escape' && fixing) closeFix();
});

// ── recording (same round trip as the CRM: one clip → its text) ─────────────
let active = null;

async function rec(btn, target){
  if(active){ active.rec.stop(); return; }
  let stream;
  try{ stream = await navigator.mediaDevices.getUserMedia({audio:true}); }
  catch(e){ msg('Microphone blocked: ' + e.message, true); return; }
  const mr = new MediaRecorder(stream);
  const chunks = [];
  active = {rec: mr, btn};
  mr.ondataavailable = e => chunks.push(e.data);
  mr.onstop = async () => {
    stream.getTracks().forEach(t => t.stop());
    active = null;
    btn.classList.remove('rec'); btn.classList.add('busy');
    btn.innerHTML = '<svg class="ico spin" viewBox="0 0 24 24"><path d="M21 12a9 9 0 1 1-6.2-8.6"/></svg> transcribing';
    const blob = new Blob(chunks, {type: mr.mimeType || 'audio/webm'});
    const fd = new FormData();
    fd.append('audio', blob, 'clip.webm');
    try{
      // Transcription lives in the CRM router; this page sits inside it, so the
      // session cookie (path=/crm) is sent with this call too.
      const r = await fetch('/crm/api/transcribe', {method:'POST', body: fd});
      const d = await r.json();
      if(d.error){ msg(d.error, true); }
      else{
        const box = document.getElementById(target);
        box.value = (box.value ? box.value.trim() + ' ' : '') + d.text;
        msg('');
      }
    }catch(e){ msg('Transcription failed: ' + e.message, true); }
    btn.classList.remove('busy');
    btn.innerHTML = '<svg class="ico" viewBox="0 0 24 24"><path d="M12 2a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3Z"/><path d="M19 10v1a7 7 0 0 1-14 0v-1"/><path d="M12 19v3"/></svg> Record';
  };
  mr.start();
  btn.classList.add('rec');
  btn.innerHTML = '<svg class="ico" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg> Stop';
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
