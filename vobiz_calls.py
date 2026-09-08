"""Vobiz telephony for the MRPscan Software agent — outbound AND inbound.

The browser page (main.py) stays as it is. This module adds the phone line:
calls we *place* from the console, and calls customers place *to* the Vobiz
number — both handed to the very same CallSession engine:

    /crm/call  (dial page, behind the CRM password)
        └─ POST /crm/call/api/dial  ──▶  Vobiz REST  POST /Account/<id>/Call/
                                            │  rings the customer
    Vobiz ──POST /answer──▶ this server      ▼  they pick up
        ◀── <Stream bidirectional> XML ───  we answer with the media-stream URL
    Vobiz ──WS /vobiz/ws?call=<uuid>──▶ VobizTransport ──▶ CallSession
                                            (Deepgram → FAQ router → ElevenLabs)
    Vobiz ──POST /hangup──▶ call marked ended, transcript already saved

INBOUND: a customer dialling FROM_NUMBER lands on the same /answer webhook with
Direction=inbound and no record of ours. We open a record for it on the spot and
return the same Stream XML, so the agent picks up and greets them with the
phone opening line (agent.INBOUND_GREETING_TEXT). Two things can still refuse a caller: INBOUND_ENABLED=false
(back to the old outbound-only behaviour), and MAX_CONCURRENT_CALLS — a call
arriving when every line is busy is answered with <Hangup/> rather than dropped
into a silent session.

CallSession was written for exactly this media-stream protocol (playAudio /
clearAudio / stop with a streamId), so VobizTransport forwards its JSON frames
untouched. The only extra work is dropping the PSTN leg over REST when the
agent hangs up, because keepCallAlive="true" keeps the call up after the
stream socket closes.

Env: VOBIZ_AUTH_ID, VOBIZ_AUTH_TOKEN, FROM_NUMBER (your Vobiz number, E.164),
PUBLIC_URL (https URL Vobiz can reach — ngrok/Railway), optional
DEFAULT_COUNTRY_CODE (default 91) for bare 10-digit numbers, optional
INBOUND_ENABLED (default true) to answer calls made to FROM_NUMBER.

Point the Vobiz number's answer URL at PUBLIC_URL/answer (and its hangup URL at
PUBLIC_URL/hangup) in the Vobiz console for inbound to reach this server.
"""
import os
import re
import json
import time
import asyncio
import logging
from urllib.parse import quote

import aiohttp
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv

import crm                      # password gate the dial page lives behind
from agent import CallSession, INBOUND_GREETING_TEXT, OUTBOUND_GREETING_TEXT

load_dotenv()
logger = logging.getLogger("vobiz")

VOBIZ_AUTH_ID = os.getenv("VOBIZ_AUTH_ID", "")
VOBIZ_AUTH_TOKEN = os.getenv("VOBIZ_AUTH_TOKEN", "")
VOBIZ_API_BASE = os.getenv("VOBIZ_API_BASE", "https://api.vobiz.ai/api/v1").rstrip("/")
FROM_NUMBER = os.getenv("FROM_NUMBER", "")
PUBLIC_URL = (os.getenv("PUBLIC_URL") or "").rstrip("/")
DEFAULT_COUNTRY_CODE = re.sub(r"\D", "", os.getenv("DEFAULT_COUNTRY_CODE", "91")) or "91"
# Answer calls made TO our Vobiz number. Set false to go back to outbound-only.
INBOUND_ENABLED = (os.getenv("INBOUND_ENABLED", "true").strip().lower()
                   not in ("0", "false", "no", "off"))
# A placed call that Vobiz never reports as answered or hung up (webhook lost,
# tunnel down) is closed on the board after this long.
DIAL_TIMEOUT_SECONDS = int(os.getenv("VOBIZ_DIAL_TIMEOUT_SECONDS", "90"))
# Ceiling on calls in progress (dialing / answered / live). The engine itself
# is one CallSession per call with nothing shared, so this is really about the
# outside limits — Vobiz channels, Deepgram + ElevenLabs concurrency — see the
# README "Capacity" section before raising it. The opening line each call
# starts with is agent.OUTBOUND_GREETING_TEXT (pre-warmed with the other
# fixed lines).
MAX_CONCURRENT_CALLS = int(os.getenv("MAX_CONCURRENT_CALLS", "10"))
_DIAL_PARALLEL = 5          # simultaneous Vobiz REST placements within one batch
_MAX_RECORDS = 200
_TRANSCRIPT_KEEP = 80

_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "call.html")

HANGUP_XML = '<?xml version="1.0" encoding="UTF-8"?>\n<Response><Hangup/></Response>'

# request_uuid -> call record. One process, one asyncio loop: a plain dict is
# enough, every mutation happens on the loop thread.
_calls: dict[str, dict] = {}


def configured() -> list[str]:
    """Names of the env vars still missing for outbound calling."""
    missing = []
    if not VOBIZ_AUTH_ID:
        missing.append("VOBIZ_AUTH_ID")
    if not VOBIZ_AUTH_TOKEN:
        missing.append("VOBIZ_AUTH_TOKEN")
    if not FROM_NUMBER:
        missing.append("FROM_NUMBER")
    if not PUBLIC_URL:
        missing.append("PUBLIC_URL")
    return missing


# ── numbers ───────────────────────────────────────────────────────────────────

def normalize_number(raw: str) -> str | None:
    """'98765 43210' / '098765-43210' / '919876543210' / '+91…' → '+919876543210'."""
    s = re.sub(r"[\s\-().]", "", raw or "")
    if s.startswith("00"):
        s = "+" + s[2:]
    if s.startswith("+"):
        return s if re.fullmatch(r"\+\d{8,15}", s) else None
    if not s.isdigit():
        return None
    cc = DEFAULT_COUNTRY_CODE
    if len(s) == 10:
        return f"+{cc}{s}"
    if len(s) == 11 and s.startswith("0"):
        return f"+{cc}{s[1:]}"
    if s.startswith(cc) and len(s) == len(cc) + 10:
        return f"+{s}"
    if 11 <= len(s) <= 15:
        return f"+{s}"
    return None


def _same_number(a: str, b: str) -> bool:
    da, db = re.sub(r"\D", "", a or ""), re.sub(r"\D", "", b or "")
    if not da or not db:
        return False
    if len(da) >= 10 and len(db) >= 10:
        return da[-10:] == db[-10:]
    return da == db


# ── call records ──────────────────────────────────────────────────────────────

def _public(rec: dict) -> dict:
    now = time.time()
    if rec["state"] == "ended":
        elapsed = rec.get("duration")
    elif rec.get("answered_at"):
        elapsed = int(now - rec["answered_at"])
    else:
        elapsed = None
    return {
        "id": rec["id"],
        "call_uuid": rec.get("call_uuid"),
        "to": rec["to"],
        "direction": rec.get("direction", "outbound"),
        # Whoever is on the other end: the number we dialled, or the number
        # that dialled us. The console shows this one.
        "peer": rec.get("peer") or rec["to"],
        # Set by scheduled_calls.py when the call came from the schedule board.
        "name": rec.get("name") or "",
        "from": rec["from"],
        "state": rec["state"],
        "placed_at": rec["placed_at"],
        "answered_at": rec.get("answered_at"),
        "ended_at": rec.get("ended_at"),
        "duration": rec.get("duration"),
        "elapsed": elapsed,
        "hangup_cause": rec.get("hangup_cause"),
        "transcript": rec["transcript"][-_TRANSCRIPT_KEEP:],
    }


def _mark_ended(rec: dict, cause: str | None = None, duration=None) -> None:
    if rec["state"] == "ended":
        return
    rec["state"] = "ended"
    rec["ended_at"] = time.time()
    if cause and not rec.get("hangup_cause"):
        rec["hangup_cause"] = cause
    if duration not in (None, ""):
        try:
            rec["duration"] = int(float(duration))
        except (TypeError, ValueError):
            pass
    if rec.get("duration") is None and rec.get("answered_at"):
        rec["duration"] = int(rec["ended_at"] - rec["answered_at"])


def _prune() -> None:
    now = time.time()
    for rec in _calls.values():
        if rec["state"] == "dialing" and now - rec["placed_at"] > DIAL_TIMEOUT_SECONDS:
            logger.warning(f"Call {rec['id']} to {rec['to']} never answered — timing out")
            rec["hangup_sent"] = True   # nothing to hang up any more
            _mark_ended(rec, "no_answer_timeout")
    if len(_calls) > _MAX_RECORDS:
        ended = sorted((r for r in _calls.values() if r["state"] == "ended"),
                       key=lambda r: r["placed_at"])
        for rec in ended[: len(_calls) - _MAX_RECORDS]:
            _calls.pop(rec["id"], None)


def active_records() -> list[dict]:
    return [r for r in _calls.values() if r["state"] != "ended"]


def _find(*, request_uuid: str = "", call_uuid: str = "", to: str = "") -> dict | None:
    """Locate the record a Vobiz webhook / media socket is talking about.

    Prefer the ids; fall back to the dialled number. The fallback only ever
    looks at OUTBOUND records: their To is the customer's number, so it is
    unambiguous — whereas every inbound record shares the same To (our own
    number) and would otherwise match some other caller's webhook."""
    if request_uuid and request_uuid in _calls:
        return _calls[request_uuid]
    if call_uuid:
        if call_uuid in _calls:
            return _calls[call_uuid]
        for rec in _calls.values():
            if rec.get("call_uuid") == call_uuid:
                return rec
    if to:
        cands = [r for r in _calls.values()
                 if r["state"] != "ended"
                 and r.get("direction", "outbound") == "outbound"
                 and _same_number(r["to"], to)]
        if cands:
            return max(cands, key=lambda r: r["placed_at"])
    return None


# ── Vobiz REST ────────────────────────────────────────────────────────────────

def _account_url(*parts: str) -> str:
    """Build a Vobiz REST URL. Every caller goes through here because the path
    is CASE-SENSITIVE: a lowercase "account" segment answers 404
    {"detail":"Not Found"}, which reads like the API is down rather than like a
    typo."""
    tail = "".join(f"{p}/" for p in parts if p)
    return f"{VOBIZ_API_BASE}/Account/{VOBIZ_AUTH_ID}/{tail}"


def _headers() -> dict:
    return {"Content-Type": "application/json",
            "X-Auth-ID": VOBIZ_AUTH_ID,
            "X-Auth-Token": VOBIZ_AUTH_TOKEN}


def _new_record(rid: str, *, to: str, frm: str, direction: str,
                call_uuid: str | None = None) -> dict:
    """The one shape a call record has, whoever started it. `peer` is the
    customer's number in both directions — dialled for outbound, caller for
    inbound — so the console never has to branch on direction."""
    rec = {
        "id": str(rid),
        "call_uuid": call_uuid,
        "to": to,
        "from": frm,
        "direction": direction,
        "peer": frm if direction == "inbound" else to,
        "state": "dialing",
        "placed_at": time.time(),
        "answered_at": None,
        "ended_at": None,
        "duration": None,
        "hangup_cause": None,
        "transcript": [],
        "session": None,
        "hangup_sent": False,
    }
    _calls[rec["id"]] = rec
    return rec


async def place_call(to_number: str) -> dict:
    missing = configured()
    if missing:
        raise RuntimeError("Outbound calling is not configured: set " + ", ".join(missing))
    _prune()
    url = _account_url("Call")
    payload = {
        "from": FROM_NUMBER,
        "to": to_number,
        "answer_url": f"{PUBLIC_URL}/answer",
        "answer_method": "POST",
        "hangup_url": f"{PUBLIC_URL}/hangup",
        "hangup_method": "POST",
    }
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=_headers()) as resp:
            text = await resp.text()
            if resp.status >= 300:
                logger.error(f"Vobiz refused the call to {to_number}: {resp.status} "
                             f"{text[:300]} (POST {url})")
                # Vobiz always answers with its own JSON ("api_id" / "error").
                # Anything else on a 404 means we asked the WRONG URL — a bad
                # VOBIZ_API_BASE or a mistyped path — and "Vobiz refused the
                # call" would send you hunting the account instead.
                if resp.status == 404 and "api_id" not in text:
                    raise RuntimeError(
                        f"{url} is not the Vobiz call API (404) — check VOBIZ_API_BASE, "
                        f"it must be https://api.vobiz.ai/api/v1")
                raise RuntimeError(f"Vobiz refused the call ({resp.status}): {text[:200]}")
    try:
        data = json.loads(text) if text else {}
    except ValueError:
        data = {}
    rid = (data.get("request_uuid") or data.get("call_uuid")
           or f"local-{int(time.time() * 1000)}")
    if isinstance(rid, list):
        rid = rid[0] if rid else f"local-{int(time.time() * 1000)}"
    rec = _new_record(rid, to=to_number, frm=FROM_NUMBER, direction="outbound")
    logger.info(f"📲 Outbound call placed → {to_number} (request {rec['id']})")
    return rec


async def rest_hangup(rec: dict) -> None:
    """Drop the PSTN leg. keepCallAlive="true" means closing the media socket
    alone leaves the customer holding a silent line."""
    if rec.get("hangup_sent"):
        return
    rec["hangup_sent"] = True
    if not (VOBIZ_AUTH_ID and VOBIZ_AUTH_TOKEN):
        return
    # An answered call is addressed by CallUUID; one still ringing by the
    # request id it was placed under. If the Request form is rejected, fall
    # back to the Call form with the same id — a phone left ringing because we
    # picked the wrong URL is worse than one extra REST call.
    ident = rec.get("call_uuid") or rec["id"]
    if rec.get("call_uuid"):
        urls = [_account_url("Call", rec["call_uuid"])]
    else:
        urls = [_account_url("Request", rec["id"]), _account_url("Call", rec["id"])]
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i, url in enumerate(urls):
                async with session.delete(url, headers=_headers()) as resp:
                    if resp.status < 300:
                        logger.info(f"REST hangup OK for call {ident}")
                        return
                    body = await resp.text()
                    if i + 1 < len(urls):
                        logger.warning(f"REST hangup via {url} returned {resp.status} "
                                       f"{body[:120]} — retrying the other form")
                    else:
                        logger.error(f"REST hangup failed for {ident}: {resp.status} {body[:200]}")
    except Exception as e:
        logger.error(f"REST hangup exception: {e}")


async def hangup_record(rec: dict, cause: str = "console") -> None:
    """End a call from our side: through the live session when there is one
    (it sends stop/hangup on the stream, then REST), otherwise straight to REST."""
    session = rec.get("session")
    if session is not None and session.call_active:
        await session._hangup_call()      # → VobizTransport.close() → rest_hangup
    else:
        await rest_hangup(rec)
    _mark_ended(rec, cause)


# ── transport ─────────────────────────────────────────────────────────────────

class VobizTransport:
    """What CallSession sees as its ``ws``. Frames pass through untouched — the
    engine already emits the Vobiz stream protocol — and close() also drops the
    PSTN leg over REST."""

    def __init__(self, ws: WebSocket, rec: dict):
        self.ws = ws
        self.rec = rec
        self._closed = False

    async def send(self, text: str):
        if self._closed:
            return
        try:
            await self.ws.send_text(text)
        except (WebSocketDisconnect, RuntimeError):
            self._closed = True
        except Exception as e:
            logger.debug(f"VobizTransport send error: {e}")

    async def close(self):
        self._closed = True
        try:
            await self.ws.close()
        except Exception:
            pass
        await rest_hangup(self.rec)


# ── Vobiz-facing endpoints (must be reachable at PUBLIC_URL) ──────────────────

public_router = APIRouter(tags=["vobiz"])


@public_router.post("/answer")
async def answer(request: Request):
    form = await request.form()
    call_uuid = str(form.get("CallUUID") or "")
    request_uuid = str(form.get("RequestUUID") or form.get("request_uuid") or "")
    frm = str(form.get("From") or "")
    to = str(form.get("To") or "")
    direction = str(form.get("Direction") or "").lower()

    rec = None if direction == "inbound" else _find(
        request_uuid=request_uuid, call_uuid=call_uuid, to=to)
    if rec is None:
        # Nothing we placed → the customer is calling US.
        if not INBOUND_ENABLED:
            logger.warning(f"⛔ Inbound call refused (INBOUND_ENABLED=false) — "
                           f"UUID={call_uuid} From={frm} To={to}")
            return Response(content=HANGUP_XML, media_type="application/xml")
        _prune()
        if len(active_records()) >= MAX_CONCURRENT_CALLS:
            # Every line is busy. Hang up rather than answer into a session we
            # cannot serve — the caller gets a clean drop, not silence.
            logger.warning(f"⛔ Inbound call from {frm} refused — all "
                           f"{MAX_CONCURRENT_CALLS} lines are busy")
            return Response(content=HANGUP_XML, media_type="application/xml")
        rid = call_uuid or request_uuid or f"inbound-{int(time.time() * 1000)}"
        rec = _new_record(rid, to=to or FROM_NUMBER, frm=frm,
                          direction="inbound", call_uuid=call_uuid or rid)
        logger.info(f"📥 Inbound call from {frm} — answering (UUID {rec['call_uuid']})")

    rec["call_uuid"] = call_uuid or rec.get("call_uuid") or rec["id"]
    if rec["state"] == "dialing":
        rec["state"] = "answered"
        rec["answered_at"] = time.time()
    logger.info(f"📞 Answered — UUID={rec['call_uuid']} To={rec['to']} (request {rec['id']})")

    ws_base = PUBLIC_URL.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    # quote() so nothing in the id is mangled by the query string.
    ws_url = f"{ws_base}/vobiz/ws?call={quote(rec['call_uuid'], safe='')}"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
      <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000" statusCallbackUrl="{PUBLIC_URL}/stream-status" statusCallbackMethod="POST">
        {ws_url}
      </Stream>
    </Response>"""
    logger.info(f"Returning Stream XML → {ws_url}")
    return Response(content=xml, media_type="application/xml")


@public_router.post("/hangup")
async def hangup(request: Request):
    form = await request.form()
    call_uuid = str(form.get("CallUUID") or "")
    request_uuid = str(form.get("RequestUUID") or form.get("request_uuid") or "")
    to = str(form.get("To") or "")
    duration = form.get("Duration")
    cause = str(form.get("HangupCause") or "unknown")
    rec = _find(request_uuid=request_uuid, call_uuid=call_uuid, to=to)
    if rec is None:
        logger.info(f"Hangup for a call we did not place — UUID={call_uuid} To={to} Cause={cause}")
        return Response(content="OK")
    logger.info(f"📴 Call ended — UUID={call_uuid} Peer={rec.get('peer') or rec['to']} "
                f"({rec.get('direction', 'outbound')}) Duration={duration}s Cause={cause}")
    rec["hangup_sent"] = True             # Vobiz already dropped it
    _mark_ended(rec, cause, duration)
    session = rec.get("session")
    if session is not None and session.call_active:
        await session.cleanup()            # saves the transcript, tears down STT
    return Response(content="OK")


@public_router.post("/stream-status")
async def stream_status(request: Request):
    form = await request.form()
    logger.info(f"🔊 Stream event — Event={form.get('Event')} StreamID={form.get('StreamID')} "
                f"CallUUID={form.get('CallUUID')}")
    return Response(content="OK")


@public_router.websocket("/vobiz/ws")
async def media_stream(ws: WebSocket):
    await ws.accept()
    key = (ws.query_params.get("call") or "").strip()
    rec = _find(call_uuid=key, request_uuid=key) if key else None
    if rec is None or rec["state"] == "ended":
        logger.warning(f"Media stream for unknown/ended call {key!r} — rejected")
        await ws.close(code=1008)
        return

    inbound = rec.get("direction") == "inbound"
    transport = VobizTransport(ws, rec)
    session = CallSession(transport, caller_id=rec.get("peer") or rec["to"],
                          call_uuid=rec["call_uuid"] or rec["id"])
    # The phone greeting, per direction (agent.py). Both default to the same
    # "मैं MRP scan से प्रीति बोल रही हूं…" line and are pre-warmed by
    # gen_variants.py; agent.GREETING_TEXT is the BROWSER greeting and is
    # deliberately not used on a call.
    session.greeting = INBOUND_GREETING_TEXT if inbound else OUTBOUND_GREETING_TEXT
    rec["session"] = session
    rec["state"] = "live"

    def on_event(payload: dict):
        rec["transcript"].append({**payload, "t": time.time()})
        del rec["transcript"][:-_TRANSCRIPT_KEEP]

    session.on_event = on_event
    logger.info(f"🎙️  Media stream connected for {rec.get('peer') or rec['to']} "
                f"({'inbound' if inbound else 'outbound'}, UUID {rec['call_uuid']})")
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            text = msg.get("text")
            if text:
                # start / media / playedStream / clearedAudio / stop — the
                # engine understands every one of them as-is.
                await session.handle_message(text)
            if not session.call_active:
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Media stream error for {rec.get('peer') or rec['to']}: {e}")
    finally:
        if session.call_active:
            await session.cleanup()
        _mark_ended(rec, rec.get("hangup_cause") or "stream_closed")
        rec["session"] = None
        try:
            await ws.close()
        except Exception:
            pass
        logger.info(f"Media stream closed for {rec.get('peer') or rec['to']}")


# ── console (nested under /crm → /crm/call) ───────────────────────────────────

console_router = APIRouter(prefix="/call", tags=["outbound-calls"])


class DialBody(BaseModel):
    number: str = ""            # one number — or several, separated by newlines / commas
    numbers: list[str] = []


def _split_numbers(body: DialBody) -> list[str]:
    out, seen = [], set()
    for item in list(body.numbers) + re.split(r"[\n,;]+", body.number or ""):
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


class HangupBody(BaseModel):
    id: str


@console_router.get("")
@console_router.get("/")
async def page(request: Request):
    if not crm._authed(request):
        return RedirectResponse("/crm", status_code=303)
    if not os.path.exists(_PAGE):
        return JSONResponse({"error": "ui/call.html not found"}, status_code=404)
    return FileResponse(_PAGE, media_type="text/html")


@console_router.get("/api/calls")
async def api_calls(request: Request):
    if not crm._authed(request):
        return crm._denied()
    _prune()
    recs = sorted(_calls.values(), key=lambda r: r["placed_at"], reverse=True)
    return {"calls": [_public(r) for r in recs[:60]],
            "active": len(active_records()),
            "limit": MAX_CONCURRENT_CALLS,
            "missing": configured(),
            "inbound": INBOUND_ENABLED,
            "from": FROM_NUMBER}


@console_router.post("/api/dial")
async def api_dial(body: DialBody, request: Request):
    """Dial one number or a batch. Every number is either placed or comes back
    in `skipped` with its reason; the batch never takes the calls in progress
    past MAX_CONCURRENT_CALLS."""
    if not crm._authed(request):
        return crm._denied()
    wanted = _split_numbers(body)
    if not wanted:
        return JSONResponse({"error": "Enter at least one phone number, e.g. 98765 43210"},
                            status_code=400)
    _prune()
    placed: list[dict] = []
    skipped: list[dict] = []
    to_dial: list[str] = []
    for raw in wanted:
        number = normalize_number(raw)
        if not number:
            skipped.append({"number": raw, "reason": "not a valid phone number"})
        elif _same_number(number, FROM_NUMBER):
            skipped.append({"number": raw, "reason": "that is the agent's own number"})
        elif (any(_same_number(r.get("peer") or r["to"], number) for r in active_records())
              or any(_same_number(n, number) for n in to_dial)):
            skipped.append({"number": number, "reason": "already on a call"})
        else:
            to_dial.append(number)
    headroom = max(MAX_CONCURRENT_CALLS - len(active_records()), 0)
    for number in to_dial[headroom:]:
        skipped.append({"number": number,
                        "reason": f"over the {MAX_CONCURRENT_CALLS}-call limit — retry when a line frees up"})
    to_dial = to_dial[:headroom]

    gate = asyncio.Semaphore(_DIAL_PARALLEL)

    async def one(number: str):
        async with gate:
            try:
                placed.append(_public(await place_call(number)))
            except Exception as e:
                skipped.append({"number": number, "reason": str(e)})

    await asyncio.gather(*(one(n) for n in to_dial))
    placed.sort(key=lambda c: c["placed_at"])
    if not placed:
        reason = skipped[0]["reason"] if skipped else "nothing to dial"
        status = 502 if ("Vobiz" in reason or "configured" in reason) else 400
        return JSONResponse({"error": reason, "placed": [], "skipped": skipped}, status_code=status)
    return {"placed": placed, "skipped": skipped,
            "active": len(active_records()), "limit": MAX_CONCURRENT_CALLS}


@console_router.post("/api/hangup")
async def api_hangup(body: HangupBody, request: Request):
    if not crm._authed(request):
        return crm._denied()
    rec = _calls.get(body.id)
    if rec is None:
        return JSONResponse({"error": "unknown call"}, status_code=404)
    if rec["state"] != "ended":
        await hangup_record(rec, "console")
    return {"call": _public(rec)}


@console_router.post("/api/hangup_all")
async def api_hangup_all(request: Request):
    if not crm._authed(request):
        return crm._denied()
    recs = active_records()
    await asyncio.gather(*(hangup_record(r, "console") for r in recs), return_exceptions=True)
    return {"ended": len(recs), "active": len(active_records())}


async def shutdown():
    """Hang up whatever is still on the line when the server stops."""
    for rec in list(_calls.values()):
        if rec["state"] != "ended":
            try:
                await hangup_record(rec, "server_shutdown")
            except Exception as e:
                logger.error(f"Shutdown hangup failed for {rec['to']}: {e}")
