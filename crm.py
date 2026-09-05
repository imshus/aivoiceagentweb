"""Voice CRM — add questions to the agent's answer bank by speaking them.

Mounted at /crm on the agent's own server (main.py), password-protected.

WHAT IT DOES — one recording round trip per field:

  1. You record the QUESTION, any SUB-QUESTIONS (other ways a caller might ask
     the same thing), and the ANSWER, each with its own mic button. Add as many
     question blocks as you like before saving.
  2. Each clip goes to Deepgram (pre-recorded API) and comes back as text.
  3. OpenAI turns those raw dictations into ONE clean question + sub-questions
     + answer — same facts, no additions — and you can still edit the text.
  4. Saving writes the entry straight into the SOURCE: a new Q-numbered block
     in faq_router.CANONICAL_ANSWERS and the matching "Qn: … / A: …" pair in
     agent.AGENT_SYSTEM_PROMPT, then rebuilds the classifier menu in memory —
     so the very next caller is answered from it, with no restart.
  5. Only THAT entry is pre-warmed: its answer is rendered into the same
     approved Hinglish wording the live agent uses, and that wording is
     synthesized into tts_cache/ — nothing else is re-rendered or re-billed.
  6. Everything in the bank can be downloaded as a formatted .xlsx (question,
     its sub-questions, the answer).

Keys, all from the SAME .env the agent uses: DEEPGRAM_API_KEY (speech-to-text),
OPENAI_API_KEY (clean-up AND the Hinglish rendering, same key as the call
path), ELEVENLABS_API_KEY (audio). Password: CRM_PASSWORD, default
"admin@12321".

    python main.py
    #  then open  http://localhost:5000/crm
"""
import io
import os
import json
import time
import secrets
import logging
from datetime import datetime, timezone
from urllib.parse import quote

import aiohttp
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

import agent
import faq_router as fr

logger = logging.getLogger("crm")

CRM_PASSWORD = os.getenv("CRM_PASSWORD", "admin@12321")
SESSION_HOURS = float(os.getenv("CRM_SESSION_HOURS", "12"))
COOKIE = "crm_session"

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
# The live agent streams through Flux; Flux is streaming-only, so the uploaded
# clips go to a pre-recorded model instead (multilingual — the owner dictates in
# Hinglish).
CRM_STT_MODEL = os.getenv("CRM_STT_MODEL", "nova-3")
# The SAME keyterms the live call biases toward. The owner dictates the very
# vocabulary the agent will later have to recognise ('karat', 'tunch', 'MRP'),
# so leaving them off here meant the bank could be typed up from a misheard
# dictation — and every caller then hears the mistake read back as approved
# fact. Repeated keyterm= params: a comma-joined value is taken as ONE literal
# term and silently boosts nothing. smart_format already implies punctuate.
_CRM_KEYTERMS = "".join(f"&keyterm={quote(t)}" for t in agent.DEEPGRAM_KEYTERMS)
CRM_STT_URL = ("https://api.deepgram.com/v1/listen"
               f"?model={CRM_STT_MODEL}&language=multi&smart_format=true"
               + _CRM_KEYTERMS)

# Clean-up model: the SAME OpenAI model and key the call path uses (see
# faq_router). Nothing here is latency-critical — it runs once, on a Save click,
# not on a live line — but the rewrite is short and well-specified, so the cheap
# tier is the right call and one key covers the whole app.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CRM_MODEL = os.getenv("CRM_OPENAI_MODEL", fr.RENDER_MODEL)
# Reasoning stays off for the same reason it is off on the call path: this is a
# mechanical restructure of dictated text, not a problem to think about. Set
# CRM_REASONING_EFFORT=low if messy dictation starts coming back badly split.
CRM_REASONING_EFFORT = os.getenv("CRM_REASONING_EFFORT",
                                 fr.REASONING_EFFORT)

# token -> expiry (monotonic-ish wall clock). In-process only: restarting the
# server logs everyone out, which is the right default for an admin console.
_SESSIONS: dict[str, float] = {}

router = APIRouter(prefix="/crm", tags=["crm"])


# ── password gate ────────────────────────────────────────────────────────────

def _new_session() -> str:
    token = secrets.token_urlsafe(32)
    _SESSIONS[token] = time.time() + SESSION_HOURS * 3600
    # Drop expired tokens whenever a new one is minted.
    for t, exp in list(_SESSIONS.items()):
        if exp < time.time():
            _SESSIONS.pop(t, None)
    return token


def _authed(request: Request) -> bool:
    token = request.cookies.get(COOKIE)
    if not token:
        return False
    exp = _SESSIONS.get(token)
    if exp is None:
        return False
    if exp < time.time():
        _SESSIONS.pop(token, None)
        return False
    return True


def _denied():
    return JSONResponse({"error": "not signed in"}, status_code=401)


@router.post("/login")
async def login(password: str = Form(...)):
    if not secrets.compare_digest(password, CRM_PASSWORD):
        logger.warning("CRM login rejected (wrong password)")
        return HTMLResponse(LOGIN_PAGE.replace("<!--ERR-->",
                            '<p class="err">Wrong password.</p>'), status_code=401)
    resp = Response(status_code=303, headers={"Location": "/crm"})
    resp.set_cookie(COOKIE, _new_session(), httponly=True, samesite="lax",
                    max_age=int(SESSION_HOURS * 3600), path="/crm")
    logger.info("CRM login accepted")
    return resp


@router.post("/logout")
async def logout(request: Request):
    _SESSIONS.pop(request.cookies.get(COOKIE) or "", None)
    resp = Response(status_code=303, headers={"Location": "/crm"})
    resp.delete_cookie(COOKIE, path="/crm")
    return resp


# ── speech → text (Deepgram, pre-recorded) ───────────────────────────────────

@router.post("/api/transcribe")
async def transcribe(request: Request, audio: UploadFile = File(...)):
    """One recorded clip → its text."""
    if not _authed(request):
        return _denied()
    if not DEEPGRAM_API_KEY:
        return JSONResponse({"error": "DEEPGRAM_API_KEY not set"}, status_code=503)
    data = await audio.read()
    if not data:
        return JSONResponse({"error": "empty recording"}, status_code=400)
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}",
               "Content-Type": audio.content_type or "audio/webm"}
    try:
        session = await agent._get_tts_session()  # shared warm HTTP session
        async with session.post(CRM_STT_URL, headers=headers, data=data) as resp:
            body = await resp.text()
            if resp.status != 200:
                logger.error(f"Deepgram STT {resp.status}: {body[:300]}")
                return JSONResponse({"error": f"speech-to-text failed ({resp.status})"},
                                    status_code=502)
            out = json.loads(body)
    except aiohttp.ClientError as e:
        logger.error(f"Deepgram STT transport error: {e}")
        return JSONResponse({"error": "speech-to-text unreachable"}, status_code=502)
    try:
        alt = out["results"]["channels"][0]["alternatives"][0]
        text = (alt.get("transcript") or "").strip()
    except (KeyError, IndexError):
        text = ""
    logger.info(f"CRM transcribed {len(data)} bytes → '{text[:70]}'")
    return {"text": text}


# ── raw dictation → clean question / sub-questions / answer (OpenAI) ────────

_CLEANUP_JSON = """

Reply with ONLY this JSON object, nothing else:
{"question": "<one clean question>",
 "subquestions": ["<other phrasing>", ...],
 "answer": "<the answer>"}"""

_CLEANUP_SYSTEM = """You clean up dictated FAQ entries for a jewelry
tag-scanning / MRP software's phone assistant. You are given raw
speech-to-text of a question, optionally other phrasings of the SAME question,
and its answer, spoken in Hinglish (Hindi + English mixed).

Turn them into one clean bank entry:
- Fix speech-to-text errors, dropped words, and false starts. Keep product
  terms as they were meant (MRP, karat, tag, wastage, GST, Master Settings).
- The question is ONE clear sentence a caller would actually ask.
- Each sub-question is a DIFFERENT way of asking that same question — keep the
  ones given, discard duplicates, never invent new ones.
- The answer states every fact from the dictation and NOTHING else: no new
  facts, numbers, features or promises, no greeting, no closing question.
- Write the question and answer in plain English (this is the approved source
  text; the assistant rewords it into Hinglish when speaking). Keep
  sub-questions in the language they were dictated in, since callers ask that
  way.
- If the answer dictation is empty or says nothing usable, return an empty
  answer rather than inventing one.""" + _CLEANUP_JSON

def _llm():
    """The call path's warmed OpenAI client, reused rather than duplicated —
    same key, same host, one connection pool."""
    return fr._client


def _cleanup_params(max_tokens: int, temperature: float = 0.0) -> dict:
    """Sampling kwargs for the clean-up call. Delegates to faq_router so the
    reasoning-vs-classic decision lives in exactly one place app-wide, then
    applies the CRM's own effort override on top."""
    params = fr.llm_params(max_tokens, temperature=temperature)
    if "reasoning_effort" in params:
        params["reasoning_effort"] = CRM_REASONING_EFFORT
    return params


async def _clean_with_llm(question: str, subs: list[str], answer: str) -> dict:
    """Raw dictations → {question, subquestions, answer}. Raises on failure."""
    subs_txt = "\n".join(f"- {s}" for s in subs) or "(none)"
    user = (f"QUESTION (dictated):\n{question}\n\n"
            f"OTHER PHRASINGS (dictated):\n{subs_txt}\n\n"
            f"ANSWER (dictated):\n{answer}")
    resp = await _llm().chat.completions.create(
        model=CRM_MODEL,
        **_cleanup_params(2000, temperature=0.0),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _CLEANUP_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    out = json.loads(text)
    if not isinstance(out, dict):
        raise ValueError("model did not return a JSON object")
    return {
        "question": (out.get("question") or "").strip(),
        "subquestions": [s.strip() for s in (out.get("subquestions") or [])
                         if s and s.strip()],
        "answer": (out.get("answer") or "").strip(),
    }


# ── wordings + audio (what gen_variants.py does, one entry at a time) ───────
# gen_variants.py is the OFFLINE bulk version of everything below: same render
# prompt, same faq_variants.json, same tts_cache/. The CRM does it per entry, on
# demand, so a question recorded here is spoken from cache immediately and a
# stale wording can be re-rendered without a shell.

def _variants_file() -> dict:
    """faq_variants.json as-is ({} when missing/unreadable)."""
    try:
        with open(fr.FAQ_VARIANTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"Variants file unreadable ({e})")
        return {}


async def _render_hinglish(entry: dict) -> str:
    """One approved Hinglish wording of this answer — the SAME render prompt,
    model and language policy the live agent uses, so the spoken reply is
    identical in style to every other answer in the bank."""
    resp = await fr._client.chat.completions.create(
        model=fr.RENDER_MODEL,
        **fr.llm_params(400, temperature=0.3),
        messages=[
            {"role": "system", "content": fr._RENDER_SYSTEM},
            {"role": "user",
             "content": (f"Caller said: {fr.entry_question(entry)}\n\n"
                         f"APPROVED ANSWER (single source of truth):\n{entry['a']}"
                         f"\n\n{fr._RENDER_NUDGE}")},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _save_variant(qid: str, text: str) -> None:
    """Merge this wording into faq_variants.json and reload it, so a restart
    keeps it and the running agent picks it up immediately."""
    path = fr.FAQ_VARIANTS_FILE
    data = _variants_file()
    data["language"] = fr.REPLY_LANGUAGE
    data.setdefault("entries", {})[qid] = {
        "hash": fr._entry_hash(qid),
        "variants": [text],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    fr.load_variants()


async def _prewarm(qid: str) -> dict:
    """Render + synthesize just this entry. Never raises into the save flow —
    a failure only means the first caller pays the normal live latency."""
    out = {"hinglish": "", "audio": False, "error": ""}
    entry = fr.CANONICAL_ANSWERS.get(qid)
    if entry is None:
        out["error"] = "entry vanished"
        return out
    try:
        text = await _render_hinglish(entry)
        if not text:
            raise RuntimeError("renderer returned nothing")
        out["hinglish"] = text
        _save_variant(qid, text)
    except Exception as e:
        logger.error(f"Prewarm render failed for {qid}: {e}")
        out["error"] = f"Hinglish wording failed: {e}"
        return out
    try:
        # use_cache=True stores the clip on disk (tts_cache/), exactly like the
        # boot-time prewarm — so this answer is spoken from cache from now on.
        async for _ in agent.stream_tts_audio(text, use_cache=True):
            pass
        out["audio"] = text in agent._tts_cache
        if not out["audio"]:
            out["error"] = "audio synthesis returned nothing"
    except Exception as e:
        logger.error(f"Prewarm TTS failed for {qid}: {e}")
        out["error"] = f"audio failed: {e}"
    return out


# ── bank read / write ────────────────────────────────────────────────────────

class SaveIn(BaseModel):
    question: str
    subquestions: list[str] = []
    answer: str
    clean: bool = True   # run the Claude clean-up (off = save the text as typed)


class DeleteIn(BaseModel):
    qid: str


def _variant_state(qid: str) -> tuple[str, bool, bool]:
    """(wording, is_stale, has_audio) for one entry — the same three things
    gen_variants.py reasons about, read straight off faq_variants.json and the
    TTS cache so the console shows the true state of the bank."""
    text = (fr._VARIANTS.get(qid) or [""])[0]
    stale = False
    if not text:
        # Loaded variants are hash-checked; a wording that exists in the file
        # but was dropped at load time means the answer text has since changed.
        raw = _variants_file().get("entries", {}).get(qid) or {}
        if raw.get("variants"):
            text, stale = raw["variants"][0], True
    has_audio = bool(text) and (
        text in agent._tts_cache
        or (agent.TTS_CACHE_DIR and os.path.exists(agent._tts_disk_path(text))))
    return text, stale, has_audio


def _entry_view(qid: str) -> dict:
    entry = fr.CANONICAL_ANSWERS[qid]
    hinglish, stale, has_audio = _variant_state(qid)
    return {
        "qid": qid,
        "question": fr.canon_q(entry),
        "subquestions": fr.all_q(entry)[1:],
        "answer": entry["a"],
        "hinglish": hinglish,
        "stale": stale,
        "has_audio": has_audio,
        "ready": bool(hinglish) and not stale and has_audio,
        "custom": qid in fr.CUSTOM_IDS,
        "created_at": fr.crm_entries().get(qid, ""),
    }


@router.get("/api/entries")
async def entries(request: Request, all: int = 0):
    """The CRM-added entries (all=1 also lists the built-in bank, read-only)."""
    if not _authed(request):
        return _denied()
    ids = list(fr.CANONICAL_ANSWERS) if all else [
        q for q in fr.CANONICAL_ANSWERS if q in fr.CUSTOM_IDS]
    items = [_entry_view(q) for q in ids]
    items.sort(key=lambda x: (not x["custom"], x["created_at"]), reverse=False)
    return {"items": items, "custom": len(fr.CUSTOM_IDS),
            "total": len(fr.CANONICAL_ANSWERS)}


@router.post("/api/save")
async def save(request: Request, body: SaveIn):
    """Clean one dictated Q&A, add it to the bank, and pre-warm just that one."""
    if not _authed(request):
        return _denied()
    question = body.question.strip()
    answer = body.answer.strip()
    subs = [s.strip() for s in body.subquestions if s and s.strip()]
    if not question or not answer:
        return JSONResponse({"error": "a question and an answer are both required"},
                            status_code=400)

    cleaned = {"question": question, "subquestions": subs, "answer": answer}
    if body.clean:
        if not OPENAI_API_KEY:
            return JSONResponse(
                {"error": "OPENAI_API_KEY is not set in .env — add it, or "
                          "switch off 'clean up with AI' to save the text "
                          "exactly as recorded"}, status_code=503)
        try:
            cleaned = await _clean_with_llm(question, subs, answer)
        except Exception as e:
            logger.error(f"Clean-up failed: {e}")
            return JSONResponse({"error": f"clean-up failed: {e}"}, status_code=502)
        if not cleaned["question"] or not cleaned["answer"]:
            return JSONResponse(
                {"error": "the recording did not contain a usable question and "
                          "answer — record again, or switch off the clean-up"},
                status_code=422)

    try:
        qid = fr.add_entry(cleaned["question"], cleaned["subquestions"],
                           cleaned["answer"])
    except Exception as e:
        logger.error(f"Bank write failed: {e}")
        return JSONResponse({"error": f"could not save: {e}"}, status_code=500)

    warm = await _prewarm(qid)
    return {"ok": True, "qid": qid, "entry": _entry_view(qid), "prewarm": warm}


class UpdateIn(BaseModel):
    qid: str
    question: str
    subquestions: list[str] = []
    answer: str


@router.post("/api/update")
async def update(request: Request, body: UpdateIn):
    """Edit a question, its sub-questions and its answer — in the source files
    and in the running bank. Editing the ANSWER makes the recorded Hinglish
    wording stale, so that one entry is re-rendered and re-synthesized here."""
    if not _authed(request):
        return _denied()
    try:
        out = fr.update_entry(body.qid, body.question, body.subquestions,
                              body.answer)
    except KeyError:
        return JSONResponse({"error": "no such question"}, status_code=404)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.error(f"Update failed for {body.qid}: {e}")
        return JSONResponse({"error": f"could not save: {e}"}, status_code=500)
    warm = await _prewarm(body.qid) if out["answer_changed"] else {}
    return {"ok": True, "qid": body.qid, "entry": _entry_view(body.qid),
            "answer_changed": out["answer_changed"], "prewarm": warm}


class PrewarmIn(BaseModel):
    qid: str = ""      # one entry; empty = every entry that needs it
    force: bool = False   # re-render even entries that are already ready


@router.post("/api/prewarm")
async def prewarm(request: Request, body: PrewarmIn):
    """Render + synthesize wordings — the gen_variants.py pass, on demand.

    Without a qid it walks the WHOLE bank and touches only what is missing,
    stale, or has no audio, so nothing already approved is re-worded or
    re-billed. `force` re-renders regardless (one entry at a time only).
    """
    if not _authed(request):
        return _denied()
    if body.qid:
        if body.qid not in fr.CANONICAL_ANSWERS:
            return JSONResponse({"error": "no such entry"}, status_code=404)
        todo = [body.qid]
    else:
        todo = [q for q in fr.CANONICAL_ANSWERS
                if body.force or not _entry_view(q)["ready"]]
    done, failed = [], []
    for qid in todo:
        warm = await _prewarm(qid)
        (done if warm["audio"] else failed).append(
            {"qid": qid, **{k: warm[k] for k in ("hinglish", "error")}})
    logger.info(f"CRM prewarm: {len(done)} ready, {len(failed)} failed "
                f"(of {len(todo)} attempted)")
    return {"ok": True, "attempted": len(todo), "ready": done, "failed": failed}


def _drop_variant(qid: str) -> None:
    """Forget this entry's approved wording so faq_variants.json doesn't keep
    growing stale rows. (Its .ulaw clip stays in tts_cache/ — harmless, and
    still valid if the same answer is ever added back.)"""
    data = _variants_file()
    if (data.get("entries") or {}).pop(qid, None) is None:
        return
    tmp = fr.FAQ_VARIANTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, fr.FAQ_VARIANTS_FILE)
    fr.load_variants()


@router.post("/api/delete")
async def delete(request: Request, body: DeleteIn):
    """Remove a question from the bank — any question, recorded here or
    hand-written. It goes from faq_router.py, agent.py's prompt, the running
    bank, and the wordings file in one go."""
    if not _authed(request):
        return _denied()
    try:
        out = fr.delete_entry(body.qid)
    except KeyError:
        return JSONResponse({"error": "no such question"}, status_code=404)
    except Exception as e:
        logger.error(f"Delete failed for {body.qid}: {e}")
        return JSONResponse({"error": f"could not delete: {e}"}, status_code=500)
    if not out["ok"]:
        return JSONResponse(
            {"error": f"{body.qid} was not found in the source files — nothing "
                      f"was changed on disk"}, status_code=409)
    _drop_variant(body.qid)
    return {"ok": True, "qid": body.qid, "custom": out["custom"],
            "referenced": out["referenced"]}


# ── Excel export ─────────────────────────────────────────────────────────────

@router.get("/api/export.xlsx")
async def export_xlsx(request: Request, all: int = 0):
    """The bank as a formatted spreadsheet: each question, then its
    sub-questions on their own rows, with the answer alongside."""
    if not _authed(request):
        return _denied()
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        return JSONResponse(
            {"error": "openpyxl is not installed — run: pip install openpyxl"},
            status_code=503)

    ids = list(fr.CANONICAL_ANSWERS) if all else [
        q for q in fr.CANONICAL_ANSWERS if q in fr.CUSTOM_IDS]

    wb = Workbook()
    ws = wb.active
    ws.title = "Question bank"
    headers = ["ID", "Row", "Question / Sub-question", "Answer",
               "Spoken Hinglish", "Source", "Added on"]
    ws.append(headers)
    head_fill = PatternFill("solid", fgColor="1F3864")
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = head_fill
        cell.alignment = Alignment(vertical="center")
    ws.freeze_panes = "A2"

    for qid in ids:
        v = _entry_view(qid)
        src = "CRM (voice)" if v["custom"] else "Built-in"
        ws.append([qid, "Question", v["question"], v["answer"], v["hinglish"],
                   src, v["created_at"]])
        ws.cell(row=ws.max_row, column=3).font = Font(bold=True)
        for sub in v["subquestions"]:
            ws.append([qid, "Sub-question", sub, "", "", src, ""])
            ws.cell(row=ws.max_row, column=3).alignment = Alignment(
                indent=2, wrap_text=True, vertical="top")

    widths = [8, 14, 60, 70, 70, 13, 21]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(
                wrap_text=True, vertical="top",
                indent=cell.alignment.indent if cell.column == 3 else 0)

    buf = io.BytesIO()
    wb.save(buf)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    name = f"question-bank-{stamp}.xlsx"
    logger.info(f"CRM exported {len(ids)} entries → {name}")
    return Response(
        buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/health")
async def health():
    return {"status": "healthy", "bank": len(fr.CANONICAL_ANSWERS),
            "crm_entries": len(fr.CUSTOM_IDS),
            "stt": bool(DEEPGRAM_API_KEY),
            "cleanup_model": CRM_MODEL if OPENAI_API_KEY else None}


# ── pages ────────────────────────────────────────────────────────────────────

# Light theme, shared by both pages. Kept as a PLAIN string (not an f-string)
# so the CSS braces need no doubling — the page templates interpolate it.
_CSS = """
  :root{--bg:#f6f7f9;--card:#ffffff;--line:#e4e7ec;--fg:#111827;--dim:#667085;
        --accent:#2563eb;--accent-soft:#eef4ff;--red:#d92d20;--red-soft:#fef3f2;
        --green:#15803d;--green-soft:#ecfdf3;--amber:#b54708;--amber-soft:#fffaeb;
        --shadow:0 1px 2px rgba(16,24,40,.05), 0 1px 3px rgba(16,24,40,.06);}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);-webkit-font-smoothing:antialiased;
       font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  button,a.btn{display:inline-flex;align-items:center;gap:7px;white-space:nowrap;
       border:1px solid transparent;border-radius:8px;padding:8px 14px;font:inherit;
       font-size:14px;font-weight:500;line-height:1.2;cursor:pointer;
       text-decoration:none;transition:background .12s,border-color .12s,color .12s,
       box-shadow .12s}
  button:active,a.btn:active{transform:translateY(.5px)}
  button:focus-visible,a.btn:focus-visible{outline:none;
       box-shadow:0 0 0 3px var(--accent-soft)}
  button:disabled{opacity:.55;cursor:default}
  /* Stroke icons inherit the button's text colour, so every variant stays
     consistent without a second set of icon rules. */
  .ico{width:16px;height:16px;min-width:16px;flex:none;stroke:currentColor;
       fill:none;stroke-width:1.75;stroke-linecap:round;stroke-linejoin:round;
       vertical-align:middle}
  .spin{animation:spin 1s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  input,textarea{background:#fff;color:var(--fg);border:1px solid var(--line);
       border-radius:8px;padding:10px;font:inherit;width:100%}
  input:focus,textarea:focus{outline:none;border-color:var(--accent);
       box-shadow:0 0 0 3px var(--accent-soft)}
  textarea{resize:vertical;min-height:64px}
  ::placeholder{color:#98a2b3}
  /* Phones: 16px is the smallest size iOS Safari will not zoom into on focus,
     and touch targets get a little more height. */
  @media (max-width:720px){
    input,textarea{font-size:16px;padding:11px}
    button,a.btn{padding:9px 13px}
  }
"""

_CSS_LOGIN = """
  body{background:var(--bg)}
  .wrap{max-width:360px;margin:13vh auto;padding:32px 28px;background:var(--card);
        border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);
        text-align:center}
  h1{font-size:19px;margin:0 0 6px;letter-spacing:-.01em}
  p.sub{color:var(--dim);font-size:13.5px;margin:0 0 22px}
  form{display:flex;flex-direction:column;gap:10px}
  button{background:var(--accent);color:#fff;font-weight:600;padding:10px 15px}
  button:hover{background:#1d4ed8}
  .err{color:var(--red);background:var(--red-soft);border:1px solid #fecdca;
       border-radius:8px;padding:8px 10px;font-size:13.5px;margin:14px 0 0}
  @media (max-width:420px){
    .wrap{margin:8vh 16px;padding:26px 20px}
  }
"""

# Inline stroke icons (16px, currentColor). Defined once as a plain string so
# the page templates can drop them into static markup, and the script can reuse
# the same shapes for rows it builds at runtime.
def _svg(paths: str) -> str:
    return (f'<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">{paths}</svg>')


ICONS = {
    "mic": _svg('<path d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>'
                '<path d="M19 10v1a7 7 0 0 1-14 0v-1"/><path d="M12 18v4"/>'),
    "stop": _svg('<rect x="6" y="6" width="12" height="12" rx="2"/>'),
    "plus": _svg('<path d="M12 5v14M5 12h14"/>'),
    "trash": _svg('<path d="M3 6h18"/><path d="M8 6V4h8v2"/>'
                  '<path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/>'),
    "pencil": _svg('<path d="M12 20h9"/>'
                   '<path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>'),
    "refresh": _svg('<path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/>'),
    "download": _svg('<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/>'
                     '<path d="M4 20h16"/>'),
    "check": _svg('<path d="m20 6-11 11-5-5"/>'),
    "close": _svg('<path d="M18 6 6 18M6 6l12 12"/>'),
    "logout": _svg('<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>'
                   '<path d="m16 17 5-5-5-5"/><path d="M21 12H9"/>'),
    "alert": _svg('<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3'
                  'L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4"/>'
                  '<path d="M12 17h.01"/>'),
    "spark": _svg('<path d="M12 3v4M12 17v4M3 12h4M17 12h4"/>'
                  '<path d="M12 8a4 4 0 0 0 4 4 4 4 0 0 0-4 4 4 4 0 0 0-4-4 4 4 0 0 0 4-4Z"/>'),
    "sheet": _svg('<rect x="3" y="3" width="18" height="18" rx="2"/>'
                  '<path d="M3 9h18M3 15h18M9 3v18"/>'),
}
# The script builds rows client-side, so it needs the same shapes in JS.
ICONS_JS = ",".join(f'{k}: `{v}`' for k, v in ICONS.items())


LOGIN_PAGE = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MRPscan CRM — sign in</title><style>{_CSS}{_CSS_LOGIN}
</style></head><body>
  <div class="wrap">
    <h1>MRPscan — question bank CRM</h1>
    <p class="sub">Record questions and answers for the voice agent.</p>
    <form method="post" action="/crm/login">
      <input type="password" name="password" placeholder="Password" autofocus required>
      <button type="submit">Sign in</button>
    </form>
    <!--ERR-->
  </div>
</body></html>"""

_CSS_APP = """
  header{position:sticky;top:0;z-index:5;padding:14px 24px;background:var(--card);
         border-bottom:1px solid var(--line);display:flex;gap:12px;
         align-items:center;flex-wrap:wrap}
  h1{font-size:17px;margin:0;font-weight:600;letter-spacing:-.01em}
  .spacer{flex:1}
  .ghost{background:#fff;color:var(--fg);border:1px solid var(--line)}
  .ghost:hover{background:#f9fafb;border-color:#d0d5dd}
  main{max-width:1040px;margin:0 auto;padding:24px 24px 80px}
  .block{background:var(--card);border:1px solid var(--line);border-radius:12px;
        padding:20px;margin-bottom:16px;box-shadow:var(--shadow)}
  .block h2{font-size:15px;margin:0 0 16px;font-weight:600}
  .field{margin-bottom:14px}
  .label{display:flex;align-items:center;gap:10px;margin-bottom:6px;
        font-size:13px;font-weight:500;color:var(--dim)}
  .mic{background:#fff;color:var(--fg);border:1px solid var(--line);
       padding:5px 12px;font-size:13px;border-radius:999px}
  .mic:hover{background:#f9fafb;border-color:#d0d5dd}
  .mic.rec{background:var(--red);border-color:var(--red);color:#fff}
  .mic.rec:hover{background:#b42318}
  .mic.rec .ico{fill:currentColor;stroke:none}
  .mic.busy{color:var(--dim)}
  .sub{border-left:2px solid var(--line);padding-left:12px;margin-bottom:10px}
  .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .add{background:#fff;color:var(--fg);border:1px solid var(--line);
       font-size:13px;padding:6px 12px}
  .add:hover{background:#f9fafb}
  .save{background:var(--accent);color:#fff;font-weight:600}
  .save:hover{background:#1d4ed8}
  .del{background:#fff;color:var(--red);border:1px solid #fda29b;font-size:13px;
       padding:6px 12px}
  .icon-only{padding:6px;border-radius:7px}
  .del:hover{background:var(--red-soft)}
  .dl{background:var(--accent);color:#fff}
  .dl:hover{background:#1d4ed8}
  .msg{font-size:13px;margin-left:2px;color:var(--dim)}
  a.nav{text-decoration:none;padding:9px 15px;border-radius:8px;font-size:14px}
  .pill{display:inline-block;font-size:11.5px;font-weight:500;border-radius:999px;
        padding:2px 9px;white-space:nowrap;border:1px solid var(--line);
        background:#f9fafb;color:var(--dim)}
  .ok{background:var(--green-soft);color:var(--green);border-color:#abefc6}
  .warn{background:var(--amber-soft);color:var(--amber);border-color:#fedf89}
  table{width:100%;border-collapse:collapse;font-size:14px}
  th,td{text-align:left;padding:11px 10px;border-bottom:1px solid var(--line);
       vertical-align:top}
  thead th{position:sticky;top:57px;background:var(--card);color:var(--dim);
       font-weight:500;font-size:12px;text-transform:uppercase;
       letter-spacing:.04em}
  tbody tr:hover{background:#fafbfc}
  /* The three row actions stack in a column, all one width, so their icons
     and labels line up down the cell. */
  tbody td:last-child{white-space:nowrap;width:1%}
  .row-acts{display:flex;flex-direction:column;align-items:stretch;gap:6px}
  .row-acts button{width:100%;min-width:116px;justify-content:flex-start;margin:0}
  tbody tr:last-child td{border-bottom:0}
  .qid{color:var(--accent);font-family:ui-monospace,Consolas,monospace;
       font-size:13px;font-weight:600}
  .subs{color:var(--dim);font-size:13px;margin-top:5px}
  label.chk{display:flex;gap:8px;align-items:center;color:var(--dim);
        font-size:13.5px}
  label.chk input{width:auto}
  .hdr-title{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;min-width:0}
  .hdr-acts{display:flex;gap:10px;align-items:center;flex-wrap:wrap}

  /* ── narrow screens ──────────────────────────────────────────────────────
     Below 720px the five-column table cannot hold its shape, so each row
     becomes a card and every cell carries its own label (data-l). Everything
     else just tightens up. */
  @media (max-width:900px){
    main{padding:18px 16px 70px}
    header{padding:12px 16px}
  }
  @media (max-width:720px){
    /* A sticky header would eat a fifth of a phone screen once the actions
       wrap, so on phones it scrolls away with the page. */
    header{position:static;padding:10px 12px;gap:8px;align-items:flex-start}
    .hdr-title{gap:2px;flex-direction:column;align-items:flex-start}
    h1{font-size:16px}
    .spacer{display:none}
    .hdr-acts{width:100%;gap:8px}
    .hdr-acts>*{flex:1 1 calc(50% - 4px);min-width:0}   /* two per row */
    .hdr-acts form{display:flex}
    .wide-only{display:none}          /* "Pre-warm", "Download" — labels fit */
    .hdr-acts button{justify-content:center;font-size:13px;padding:8px 10px;
         width:100%}
    main{padding:14px 12px 64px}
    .block{padding:14px;border-radius:10px;margin-bottom:12px}
    .label{flex-wrap:wrap}

    table,tbody,tr,td{display:block;width:100%}
    thead{display:none}
    tbody tr{border:1px solid var(--line);border-radius:10px;padding:12px 14px;
         margin-bottom:12px;background:var(--card)}
    tbody tr:hover{background:var(--card)}
    tbody tr:last-child{margin-bottom:0}
    td{border-bottom:0;padding:6px 0;white-space:normal}
    td[data-l]::before{content:attr(data-l);display:block;font-size:11px;
         font-weight:500;text-transform:uppercase;letter-spacing:.04em;
         color:var(--dim);margin-bottom:3px}
    tbody td:last-child{width:auto;white-space:normal;padding-top:10px}
    /* Actions sit side by side once the row is a card — there is width for it. */
    .row-acts{flex-direction:row;flex-wrap:wrap}
    .row-acts button{width:auto;min-width:0;flex:1 1 auto;justify-content:center}
    .qid{font-size:12px}
  }
  @media (max-width:420px){
    .hdr-acts button{font-size:12.5px;gap:5px}
  }
"""

APP_PAGE = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MRPscan CRM — question bank</title><style>{_CSS}{_CSS_APP}
</style></head><body>
<header>
  <div class="hdr-title">
    <h1>MRPscan — question bank CRM</h1>
    <span class="msg" id="bankinfo"></span>
  </div>
  <div class="spacer"></div>
  <div class="hdr-acts">
    <button class="ghost" onclick="location.href='/crm/unavailable'">
      {ICONS["alert"]} <span>Unanswered</span></button>
    <button class="ghost" onclick="warmAll()">
      {ICONS["spark"]} <span>Pre-warm<span class="wide-only"> missing</span></span></button>
    <button class="dl" onclick="dl()">
      {ICONS["download"]} <span>Download<span class="wide-only"> whole bank</span></span></button>
    <form method="post" action="/crm/logout" style="display:inline">
      <button class="ghost" type="submit">{ICONS["logout"]} <span>Sign out</span></button>
    </form>
  </div>
</header>
<main>
  <div id="blocks"></div>
  <div class="row" style="margin-bottom:26px">
    <button class="add" onclick="addBlock()">{ICONS["plus"]} Add another question</button>
    <label class="chk"><input type="checkbox" id="clean" checked>
      Clean up with AI before saving</label>
  </div>

  <div class="block">
    <h2>In the bank <span class="msg" id="listmsg"></span></h2>
    <table><thead><tr><th>ID</th><th>Question</th><th>Answer</th>
      <th>Spoken reply</th><th></th></tr>
    </thead><tbody id="rows"></tbody></table>
  </div>
</main>
<script>
const I = {{{ICONS_JS}}};
const esc = s => (s||"").replace(/[&<>"]/g, c =>
  ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
let n = 0;

function addBlock(){{
  const i = n++;
  const el = document.createElement('div');
  el.className = 'block'; el.id = 'b'+i;
  el.innerHTML = `
    <h2>Question ${{i + 1}}</h2>
    <div class="field">
      <div class="label">Question
        <button class="mic" onclick="rec(this,'q${{i}}')">${{I.mic}} Record</button></div>
      <textarea id="q${{i}}" placeholder="Speak it, or type it here…"></textarea>
    </div>
    <div id="subs${{i}}"></div>
    <div class="row" style="margin-bottom:14px">
      <button class="add" onclick="addSub(${{i}})">${{I.plus}} Add sub-question</button>
    </div>
    <div class="field">
      <div class="label">Answer
        <button class="mic" onclick="rec(this,'a${{i}}')">${{I.mic}} Record</button></div>
      <textarea id="a${{i}}" placeholder="Speak the answer…"></textarea>
    </div>
    <div class="row">
      <button class="save" onclick="save(${{i}})">${{I.check}} Save to bank</button>
      <button class="ghost" onclick="document.getElementById('b${{i}}').remove()">
        ${{I.close}} Discard</button>
      <span class="msg" id="m${{i}}"></span>
    </div>`;
  document.getElementById('blocks').appendChild(el);
}}

let subN = 0;
function addSub(i){{
  const k = subN++;
  const d = document.createElement('div');
  d.className = 'sub';
  d.innerHTML = `
    <div class="label">Sub-question — another way callers ask it
      <button class="mic" onclick="rec(this,'s${{k}}')">${{I.mic}} Record</button>
      <button class="del icon-only" title="Remove this sub-question"
        onclick="this.closest('.sub').remove()">${{I.close}}</button></div>
    <textarea class="subq" data-block="${{i}}" id="s${{k}}"></textarea>`;
  document.getElementById('subs'+i).appendChild(d);
}}

// ── recording ───────────────────────────────────────────────────────────────
let active = null;   // {{rec, chunks, btn, target}}

async function rec(btn, target){{
  if(active && active.btn === btn){{ active.rec.stop(); return; }}
  if(active){{ active.rec.stop(); }}
  let stream;
  try{{ stream = await navigator.mediaDevices.getUserMedia({{audio:true}}); }}
  catch(e){{ alert('Microphone blocked: ' + e.message); return; }}
  const mr = new MediaRecorder(stream);
  const chunks = [];
  active = {{rec: mr, chunks, btn, target}};
  mr.ondataavailable = e => chunks.push(e.data);
  mr.onstop = async () => {{
    stream.getTracks().forEach(t => t.stop());
    btn.classList.remove('rec'); btn.classList.add('busy'); btn.style.cssText = '';
    btn.innerHTML = I.refresh.replace('class="ico"', 'class="ico spin"') +
                    ' transcribing';
    active = null;
    const blob = new Blob(chunks, {{type: mr.mimeType || 'audio/webm'}});
    const fd = new FormData();
    fd.append('audio', blob, 'clip.webm');
    try{{
      const r = await fetch('/crm/api/transcribe', {{method:'POST', body: fd}});
      const d = await r.json();
      if(d.error){{ alert(d.error); }}
      else{{
        const box = document.getElementById(target);
        box.value = (box.value ? box.value.trim() + ' ' : '') + d.text;
      }}
    }}catch(e){{ alert('Transcription failed: ' + e.message); }}
    btn.classList.remove('busy'); btn.innerHTML = I.mic + ' Record';
  }};
  mr.start();
  btn.classList.add('rec'); btn.innerHTML = I.stop + ' Stop';
  btn.style.cssText = 'background:#d92d20;border-color:#d92d20;color:#fff';
}}

// ── save / list ─────────────────────────────────────────────────────────────
async function save(i){{
  const m = document.getElementById('m'+i);
  const subs = [...document.querySelectorAll(`.subq[data-block="${{i}}"]`)]
                 .map(t => t.value.trim()).filter(Boolean);
  const body = {{
    question: document.getElementById('q'+i).value,
    answer: document.getElementById('a'+i).value,
    subquestions: subs,
    clean: document.getElementById('clean').checked,
  }};
  m.textContent = 'saving…'; m.style.color = '#9aa3b2';
  const r = await fetch('/crm/api/save', {{method:'POST',
      headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body)}});
  const d = await r.json();
  if(d.error){{ m.textContent = d.error; m.style.color = '#e5534b'; return; }}
  const warm = d.prewarm || {{}};
  m.style.color = '#7ee2a8';
  m.textContent = `saved as ${{d.qid}}` +
    (warm.audio ? ' — Hinglish reply recorded and ready'
                : (warm.error ? ' — saved, but pre-warm failed: ' + warm.error
                              : ' — pre-warm skipped'));
  document.getElementById('b'+i).remove();
  if(!document.querySelector('#blocks .block')) addBlock();
  load();
}}

async function del(qid, question, custom){{
  const what = custom ? '' :
    '\\n\\nThis is one of the original questions. It will be removed from ' +
    'faq_router.py and from the agent prompt (a .bak of each is kept).';
  if(!confirm('Delete ' + qid + ' from the bank?\\n\\n' + question + what)) return;
  const m = document.getElementById('listmsg');
  m.textContent = 'deleting ' + qid + '…'; m.style.color = '#9aa3b2';
  const r = await fetch('/crm/api/delete', {{method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{qid}})}});
  const d = await r.json();
  if(d.error){{ m.textContent = d.error; m.style.color = '#e5534b'; return; }}
  m.style.color = '#7ee2a8';
  m.textContent = qid + ' removed';
  if(d.referenced && d.referenced.length){{
    m.style.color = '#f0b429';
    m.textContent = qid + ' removed — but it is still named elsewhere in ' +
      'agent.py, reword those lines by hand: ' + d.referenced.join('  |  ');
  }}
  load();
}}

function edit(qid){{
  const it = bank.find(x => x.qid === qid);
  if(!it) return;
  const tr = [...document.querySelectorAll('#rows tr')]
      .find(r => r.querySelector('.qid').textContent === qid);
  tr.innerHTML = `
    <td class="qid" data-l="ID">${{esc(qid)}}</td>
    <td colspan="4" class="edit-cell">
      <div class="label">Question</div>
      <textarea id="eq" style="min-height:52px">${{esc(it.question)}}</textarea>
      <div class="label" style="margin-top:10px">Sub-questions — one per line</div>
      <textarea id="es" style="min-height:52px">${{esc(it.subquestions.join('\\n'))}}</textarea>
      <div class="label" style="margin-top:10px">Answer</div>
      <textarea id="ea" style="min-height:110px">${{esc(it.answer)}}</textarea>
      <div class="row" style="margin-top:10px">
        <button class="save" onclick="saveEdit('${{esc(qid)}}')">
          ${{I.check}} Save changes</button>
        <button class="ghost" onclick="load()">${{I.close}} Cancel</button>
        <span class="msg" id="em"></span>
      </div>
    </td>`;
}}

async function saveEdit(qid){{
  const em = document.getElementById('em');
  em.textContent = 'saving…'; em.style.color = '#9aa3b2';
  const body = {{
    qid,
    question: document.getElementById('eq').value,
    subquestions: document.getElementById('es').value.split('\\n')
                    .map(t => t.trim()).filter(Boolean),
    answer: document.getElementById('ea').value,
  }};
  const r = await fetch('/crm/api/update', {{method:'POST',
      headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body)}});
  const d = await r.json();
  if(d.error){{ em.textContent = d.error; em.style.color = '#e5534b'; return; }}
  const m = document.getElementById('listmsg');
  m.style.color = '#7ee2a8';
  m.textContent = qid + ' updated' + (d.answer_changed
    ? ((d.prewarm || {{}}).audio ? ' — new Hinglish reply recorded'
       : ' — answer changed, but re-render failed: ' + ((d.prewarm||{{}}).error||'?'))
    : '');
  load();
}}

async function warm(qid){{
  const m = document.getElementById('listmsg');
  m.textContent = 'rendering ' + qid + '…'; m.style.color = '#9aa3b2';
  const r = await fetch('/crm/api/prewarm', {{method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{qid, force:true}})}});
  const d = await r.json();
  m.style.color = d.ready && d.ready.length ? '#7ee2a8' : '#e5534b';
  m.textContent = d.error ? d.error
    : (d.ready.length ? qid + ' ready' : (d.failed[0] || {{}}).error || 'failed');
  load();
}}

async function warmAll(){{
  const m = document.getElementById('listmsg');
  m.textContent = 'pre-warming everything that needs it…'; m.style.color = '#9aa3b2';
  const r = await fetch('/crm/api/prewarm', {{method:'POST',
      headers:{{'Content-Type':'application/json'}}, body: '{{}}'}});
  const d = await r.json();
  m.style.color = d.failed && d.failed.length ? '#f0b429' : '#7ee2a8';
  m.textContent = d.error ? d.error
    : `${{d.ready.length}} ready, ${{d.failed.length}} failed (of ${{d.attempted}})`;
  load();
}}

function dl(){{ window.location = '/crm/api/export.xlsx?all=1'; }}

let bank = [];
async function load(){{
  const r = await fetch('/crm/api/entries?all=1');
  const d = await r.json();
  if(d.error){{ return; }}
  bank = d.items;
  document.getElementById('bankinfo').textContent =
    `${{d.custom}} added here · ${{d.total}} answers in total`;
  const rows = document.getElementById('rows');
  document.getElementById('listmsg').textContent =
    d.items.length ? '' : '— nothing added yet';
  rows.innerHTML = d.items.map(it => `
    <tr>
      <td class="qid" data-l="ID">${{esc(it.qid)}}</td>
      <td data-l="Question">${{esc(it.question)}}
        ${{it.subquestions.length ? `<div class="subs">also: ` +
           it.subquestions.map(esc).join(' · ') + `</div>` : ''}}</td>
      <td data-l="Answer">${{esc(it.answer)}}</td>
      <td data-l="Spoken reply">
        <span class="pill ${{it.ready ? 'ok' : 'warn'}}">${{
          it.ready ? 'ready' : (it.stale ? 'answer changed'
                    : (it.hinglish ? 'no audio' : 'not rendered'))}}</span>
        ${{it.hinglish ? `<div class="subs">${{esc(it.hinglish)}}</div>` : ''}}
      </td>
      <td class="acts-cell">
        <div class="row-acts">
        <button class="add" onclick="edit('${{esc(it.qid)}}')">${{I.pencil}} Edit</button>
        <button class="add" title="Re-record the spoken Hinglish reply"
          onclick="warm('${{esc(it.qid)}}')">${{I.refresh}} Re-render</button>
        <button class="del" onclick="del('${{esc(it.qid)}}',
          '${{esc(it.question).replace(/'/g, "&#39;")}}', ${{!!it.custom}})">
          ${{I.trash}} Delete</button>
        </div>
      </td>
    </tr>`).join('');
}}

addBlock();
load();
</script></body></html>"""


@router.get("")
@router.get("/")
async def index(request: Request):
    if not _authed(request):
        return HTMLResponse(LOGIN_PAGE.replace("<!--ERR-->", ""))
    return HTMLResponse(APP_PAGE)
