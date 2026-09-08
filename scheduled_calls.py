"""Scheduled outbound calls — the "cron" side of vobiz_calls.py.

You store WHO to call (name), WHICH number, and WHEN (date + time, in your own
timezone). A background ticker inside this same server watches the clock and,
the moment a row comes due, hands the number to vobiz_calls.place_call() — the
exact same dial path the /crm/call console uses, so the answered call runs on
the same CallSession engine.

    /crm/schedule  (behind the CRM password)
        └─ POST /crm/schedule/api/add   ──▶ Mongo  scheduled_calls
                                                │
    every SCHEDULE_TICK_SECONDS  ──▶  _tick()  ──┘  claims what is due
                                        └─▶ vobiz_calls.place_call(number)
                                                └─▶ /answer → media stream → agent

Storage is the SAME MongoDB the transcripts go to (agent.mongo_client,
MONGODB_URI / MONGODB_DB), in the collection SCHEDULE_COLLECTION
(default "scheduled_calls"). One document per planned call:

    { name, number, scheduled_at (UTC datetime), tz, note,
      status, attempts, last_error, call_id, call_state, duration,
      hangup_cause, created_at, placed_at, finished_at }

status flow:
    pending ──due──▶ dialing ──placed──▶ calling ──call ends──▶ done
       │                 └─ REST refused ─▶ pending (retry) ─▶ failed
       ├─ too old to dial ─▶ missed
       └─ cancelled from the console ─▶ cancelled

Times are entered in SCHEDULE_TIMEZONE (default Asia/Kolkata) and stored in
UTC, so a server in another zone still dials at the right local moment.

Env:
  SCHEDULE_TIMEZONE        IANA zone for the date/time you type (Asia/Kolkata)
  SCHEDULE_TICK_SECONDS    how often the clock is checked (20)
  SCHEDULE_GRACE_MINUTES   older-than-this due rows are marked missed, not
                           dialled — so a restart after downtime does not ring
                           yesterday's list (60)
  SCHEDULE_MAX_ATTEMPTS    REST placement retries before giving up (3)
  SCHEDULE_RETRY_MINUTES   wait between those retries (2)
  SCHEDULE_COLLECTION      Mongo collection name (scheduled_calls)
"""
import os
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from bson import ObjectId
from bson.errors import InvalidId

try:                                    # stdlib on 3.9+; tzdata backs it on Windows
    from zoneinfo import ZoneInfo
except ImportError:                     # pragma: no cover
    ZoneInfo = None

import agent                # same process, same Mongo client, same credentials
import crm                  # the password gate this console lives behind
import vobiz_calls          # the dialer these rows are handed to

logger = logging.getLogger("schedule")

COLLECTION = os.getenv("SCHEDULE_COLLECTION", "scheduled_calls")
TICK_SECONDS = max(int(os.getenv("SCHEDULE_TICK_SECONDS", "20")), 5)
GRACE_MINUTES = int(os.getenv("SCHEDULE_GRACE_MINUTES", "60"))
MAX_ATTEMPTS = max(int(os.getenv("SCHEDULE_MAX_ATTEMPTS", "3")), 1)
RETRY_MINUTES = max(int(os.getenv("SCHEDULE_RETRY_MINUTES", "2")), 1)
TZ_NAME = os.getenv("SCHEDULE_TIMEZONE", "Asia/Kolkata")

_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "schedule.html")
_MAX_ROWS = 500                 # newest N returned to the console
_LIVE_STATES = ("dialing", "calling")

_task: asyncio.Task | None = None


def _tz():
    """The zone the typed date/time is read in. Falls back to UTC if the
    platform has no tz database (pip install tzdata fixes that on Windows)."""
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(TZ_NAME)
    except Exception:
        logger.warning(f"Unknown SCHEDULE_TIMEZONE {TZ_NAME!r} — using UTC. "
                       f"On Windows: pip install tzdata")
        return timezone.utc


def _connected() -> bool:
    return agent.mongo_client is not None


def _collection():
    # Read the attribute live: agent.py creates the client at import time and
    # closes it on shutdown.
    return agent.mongo_client.get_database(agent.MONGODB_DB)[COLLECTION]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(dt: datetime) -> datetime:
    """Mongo hands datetimes back naive-UTC; make them comparable again."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ── when: "2026-09-09" + "15:30"  →  UTC datetime ────────────────────────────

def parse_when(date: str, time_: str, when: str = "") -> datetime:
    """Read the console's date + time (or a single ISO string) in
    SCHEDULE_TIMEZONE and return the UTC instant. Raises ValueError with a
    message meant for the operator."""
    raw = (when or "").strip()
    if not raw:
        d, t = (date or "").strip(), (time_ or "").strip()
        if not d:
            raise ValueError("Pick a date")
        if not t:
            raise ValueError("Pick a time")
        raw = f"{d}T{t}"
    raw = raw.replace(" ", "T", 1) if " " in raw and "T" not in raw else raw
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        raise ValueError(f"Could not read the date/time {raw!r} — use 2026-09-09 and 15:30")
    if dt.tzinfo is None:               # typed local time
        dt = dt.replace(tzinfo=_tz())
    return dt.astimezone(timezone.utc)


def _local(dt: datetime | None) -> str:
    if not dt:
        return ""
    return _utc(dt).astimezone(_tz()).strftime("%Y-%m-%d %H:%M")


def _public(doc: dict) -> dict:
    at = _utc(doc["scheduled_at"]) if doc.get("scheduled_at") else None
    return {
        "id": str(doc["_id"]),
        "name": doc.get("name") or "",
        "number": doc.get("number") or "",
        "note": doc.get("note") or "",
        "status": doc.get("status") or "pending",
        "scheduled_at": at.isoformat() if at else "",
        "local": _local(at),
        "tz": doc.get("tz") or TZ_NAME,
        "attempts": int(doc.get("attempts") or 0),
        "last_error": doc.get("last_error") or "",
        "call_id": doc.get("call_id") or "",
        "call_state": doc.get("call_state") or "",
        "duration": doc.get("duration"),
        "hangup_cause": doc.get("hangup_cause") or "",
        "placed_local": _local(doc.get("placed_at")),
        "finished_local": _local(doc.get("finished_at")),
    }


# ── the cron ticker ──────────────────────────────────────────────────────────

async def _claim_due(limit: int) -> list[dict]:
    """Atomically take up to `limit` rows that are due, flipping them to
    'dialing' so a second tick can never dial the same row twice."""
    now = _now()
    claimed: list[dict] = []
    while len(claimed) < limit:
        doc = await _collection().find_one_and_update(
            {"status": "pending",
             "scheduled_at": {"$lte": now},
             "$or": [{"next_attempt_at": None},
                     {"next_attempt_at": {"$exists": False}},
                     {"next_attempt_at": {"$lte": now}}]},
            {"$set": {"status": "dialing", "claimed_at": now}},
            sort=[("scheduled_at", 1)],
            return_document=True,
        )
        if doc is None:
            break
        claimed.append(doc)
    return claimed


async def _mark_missed() -> int:
    """Rows whose moment passed while the server was down are NOT dialled late
    — nobody wants yesterday's 3pm list ringing at 9am."""
    cutoff = _now() - timedelta(minutes=GRACE_MINUTES)
    res = await _collection().update_many(
        {"status": "pending", "scheduled_at": {"$lt": cutoff}},
        {"$set": {"status": "missed",
                  "last_error": f"more than {GRACE_MINUTES} min late — not dialled",
                  "finished_at": _now()}},
    )
    return res.modified_count


async def _dial(doc: dict) -> None:
    """Place one scheduled call and write the outcome back onto its row."""
    oid, name = doc["_id"], doc.get("name") or ""
    number = doc.get("number") or ""
    attempts = int(doc.get("attempts") or 0) + 1
    try:
        rec = await vobiz_calls.place_call(number)
        rec["name"] = name
        rec["scheduled_id"] = str(oid)
        await _collection().update_one({"_id": oid}, {"$set": {
            "status": "calling",
            "attempts": attempts,
            "call_id": rec["id"],
            "call_state": rec["state"],
            "placed_at": _now(),
            "last_error": "",
        }})
        logger.info(f"⏰ Scheduled call placed → {name or number} ({number}) [{oid}]")
    except Exception as e:
        err = str(e)
        if attempts < MAX_ATTEMPTS:
            await _collection().update_one({"_id": oid}, {"$set": {
                "status": "pending",
                "attempts": attempts,
                "last_error": err,
                "next_attempt_at": _now() + timedelta(minutes=RETRY_MINUTES),
            }})
            logger.warning(f"Scheduled call to {number} failed ({err}) — "
                           f"retry {attempts}/{MAX_ATTEMPTS} in {RETRY_MINUTES} min")
        else:
            await _collection().update_one({"_id": oid}, {"$set": {
                "status": "failed", "attempts": attempts,
                "last_error": err, "finished_at": _now(),
            }})
            logger.error(f"Scheduled call to {number} gave up after {attempts} tries: {err}")


async def _sync_calling() -> None:
    """Follow the rows we already dialled: copy the live call's state across and
    close the row once the call is over."""
    cursor = _collection().find({"status": "calling"})
    async for doc in cursor:
        rec = vobiz_calls._calls.get(doc.get("call_id") or "")
        if rec is None:
            # The call record aged out of the in-memory board; treat the row as
            # finished rather than leaving it "calling" forever.
            await _collection().update_one({"_id": doc["_id"]}, {"$set": {
                "status": "done", "call_state": "ended", "finished_at": _now()}})
            continue
        if rec["state"] == "ended":
            await _collection().update_one({"_id": doc["_id"]}, {"$set": {
                "status": "done",
                "call_state": "ended",
                "duration": rec.get("duration"),
                "hangup_cause": rec.get("hangup_cause") or "",
                "finished_at": _now(),
            }})
        elif rec["state"] != doc.get("call_state"):
            await _collection().update_one({"_id": doc["_id"]},
                                           {"$set": {"call_state": rec["state"]}})


async def _tick() -> None:
    # Age out calls that were never answered (no /answer, no /hangup webhook)
    # BEFORE reading them: without this a scheduled row that rings out sits on
    # "calling" until someone happens to open the dial board, and its line is
    # still counted against the headroom below.
    vobiz_calls._prune()
    await _sync_calling()
    missed = await _mark_missed()
    if missed:
        logger.warning(f"{missed} scheduled call(s) were too late to dial — marked missed")
    # Never push past the same ceiling the console respects.
    headroom = vobiz_calls.MAX_CONCURRENT_CALLS - len(vobiz_calls.active_records())
    if headroom <= 0:
        return
    due = await _claim_due(headroom)
    if not due:
        return
    logger.info(f"⏰ {len(due)} scheduled call(s) due — dialling")
    await asyncio.gather(*(_dial(d) for d in due), return_exceptions=True)


async def _loop() -> None:
    logger.info(f"⏰ Call scheduler running — every {TICK_SECONDS}s, times in {TZ_NAME}")
    while True:
        try:
            await asyncio.sleep(TICK_SECONDS)
            if not _connected():
                continue
            await _tick()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A bad tick must never kill the ticker.
            logger.error(f"Scheduler tick failed: {e}")


def start() -> None:
    """Called from main.py's lifespan once the loop is running."""
    global _task
    if _task is not None and not _task.done():
        return
    if not _connected():
        logger.warning("MONGODB_URI not set — scheduled calls are disabled")
        return
    _task = asyncio.create_task(_loop())


async def shutdown() -> None:
    global _task
    if _task is None:
        return
    _task.cancel()
    try:
        await _task
    except (asyncio.CancelledError, Exception):
        pass
    _task = None
    logger.info("⏰ Call scheduler stopped")


# ── console (nested under /crm → /crm/schedule) ──────────────────────────────

router = APIRouter(prefix="/schedule", tags=["scheduled-calls"])


class Row(BaseModel):
    name: str = ""
    number: str = ""
    date: str = ""          # YYYY-MM-DD, in SCHEDULE_TIMEZONE
    time: str = ""          # HH:MM, same zone
    when: str = ""          # or one ISO string instead of date+time
    note: str = ""


class AddBody(Row):
    """One person at the top level, or several under `rows` — never both."""
    rows: list[Row] = []


class IdBody(BaseModel):
    id: str


def _no_db():
    return JSONResponse({"error": "MONGODB_URI not set — cannot store scheduled calls"},
                        status_code=503)


def _oid(raw: str):
    try:
        return ObjectId(raw)
    except (InvalidId, TypeError):
        return None


async def _insert(item: Row) -> dict:
    """Validate one row and store it. Raises ValueError for the operator."""
    name = (item.name or "").strip()
    number = vobiz_calls.normalize_number(item.number or "")
    if not number:
        raise ValueError(f"{item.number!r} is not a valid phone number")
    if vobiz_calls._same_number(number, vobiz_calls.FROM_NUMBER):
        raise ValueError("that is the agent's own number")
    at = parse_when(item.date, item.time, item.when)
    if at < _now() - timedelta(minutes=1):
        raise ValueError(f"{_local(at)} is in the past")
    doc = {
        "name": name,
        "number": number,
        "note": (item.note or "").strip(),
        "scheduled_at": at,
        "tz": TZ_NAME,
        "status": "pending",
        "attempts": 0,
        "last_error": "",
        "call_id": "",
        "call_state": "",
        "next_attempt_at": None,
        "created_at": _now(),
    }
    res = await _collection().insert_one(doc)
    doc["_id"] = res.inserted_id
    logger.info(f"⏰ Scheduled {name or number} ({number}) for {_local(at)} {TZ_NAME}")
    return doc


@router.get("")
@router.get("/")
async def page(request: Request):
    if not crm._authed(request):
        return RedirectResponse("/crm", status_code=303)
    if not os.path.exists(_PAGE):
        return JSONResponse({"error": "ui/schedule.html not found"}, status_code=404)
    return FileResponse(_PAGE, media_type="text/html")


@router.get("/api/list")
async def api_list(request: Request):
    if not crm._authed(request):
        return crm._denied()
    if not _connected():
        return _no_db()
    docs = await (_collection()
                  .find({})
                  .sort([("scheduled_at", -1)])
                  .to_list(length=_MAX_ROWS))
    rows = [_public(d) for d in docs]
    return {
        "rows": rows,
        "tz": TZ_NAME,
        "now": _local(_now()),
        "pending": sum(1 for r in rows if r["status"] == "pending"),
        "live": sum(1 for r in rows if r["status"] in _LIVE_STATES),
        "active": len(vobiz_calls.active_records()),
        "limit": vobiz_calls.MAX_CONCURRENT_CALLS,
        "missing": vobiz_calls.configured(),
        "running": _task is not None and not _task.done(),
    }


@router.post("/api/add")
async def api_add(body: AddBody, request: Request):
    """One row, or a batch — every row is either scheduled or comes back in
    `skipped` with the reason it was refused."""
    if not crm._authed(request):
        return crm._denied()
    if not _connected():
        return _no_db()
    items = list(body.rows) or [body]
    items = [i for i in items if (i.number or "").strip()]
    if not items:
        return JSONResponse({"error": "Enter a phone number"}, status_code=400)
    added, skipped = [], []
    for item in items:
        try:
            added.append(_public(await _insert(item)))
        except ValueError as e:
            skipped.append({"number": item.number, "name": item.name, "reason": str(e)})
        except Exception as e:
            logger.error(f"Could not store scheduled call: {e}")
            skipped.append({"number": item.number, "name": item.name, "reason": str(e)})
    if not added:
        return JSONResponse({"error": skipped[0]["reason"], "added": [], "skipped": skipped},
                            status_code=400)
    return {"added": added, "skipped": skipped}


@router.post("/api/cancel")
async def api_cancel(body: IdBody, request: Request):
    """Call it off. A call already on the line is left alone — hang that up on
    the /crm/call board."""
    if not crm._authed(request):
        return crm._denied()
    if not _connected():
        return _no_db()
    oid = _oid(body.id)
    if oid is None:
        return JSONResponse({"error": "unknown row"}, status_code=404)
    doc = await _collection().find_one_and_update(
        {"_id": oid, "status": {"$in": ["pending", "dialing"]}},
        {"$set": {"status": "cancelled", "finished_at": _now()}},
        return_document=True)
    if doc is None:
        return JSONResponse({"error": "that call is not waiting any more"}, status_code=409)
    return {"row": _public(doc)}


@router.post("/api/run_now")
async def api_run_now(body: IdBody, request: Request):
    """Dial a pending row right away instead of waiting for its time."""
    if not crm._authed(request):
        return crm._denied()
    if not _connected():
        return _no_db()
    oid = _oid(body.id)
    if oid is None:
        return JSONResponse({"error": "unknown row"}, status_code=404)
    if len(vobiz_calls.active_records()) >= vobiz_calls.MAX_CONCURRENT_CALLS:
        return JSONResponse(
            {"error": f"already at the {vobiz_calls.MAX_CONCURRENT_CALLS}-call limit"},
            status_code=409)
    doc = await _collection().find_one_and_update(
        {"_id": oid, "status": {"$in": ["pending", "missed", "failed"]}},
        {"$set": {"status": "dialing", "attempts": 0, "next_attempt_at": None,
                  "claimed_at": _now()}},
        return_document=True)
    if doc is None:
        return JSONResponse({"error": "that call cannot be dialled now"}, status_code=409)
    await _dial(doc)
    fresh = await _collection().find_one({"_id": oid})
    return {"row": _public(fresh)}


@router.post("/api/delete")
async def api_delete(body: IdBody, request: Request):
    if not crm._authed(request):
        return crm._denied()
    if not _connected():
        return _no_db()
    oid = _oid(body.id)
    if oid is None:
        return JSONResponse({"error": "unknown row"}, status_code=404)
    res = await _collection().delete_one({"_id": oid, "status": {"$ne": "calling"}})
    if not res.deleted_count:
        return JSONResponse({"error": "that call is on the line — hang it up first"},
                            status_code=409)
    return {"deleted": 1}


@router.post("/api/clear_done")
async def api_clear_done(request: Request):
    """Tidy the board: drop everything that already happened."""
    if not crm._authed(request):
        return crm._denied()
    if not _connected():
        return _no_db()
    res = await _collection().delete_many(
        {"status": {"$in": ["done", "failed", "missed", "cancelled"]}})
    return {"deleted": res.deleted_count}
