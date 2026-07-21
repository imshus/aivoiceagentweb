"""Browser voice interface for the Jewelry Tech Helpline agent.

This REPLACES the old telephony server. Instead of placing phone calls, it
serves a single web page where you talk to the agent directly with your laptop
microphone — no external telephony provider is involved.

Audio path (unchanged brain, new transport):
  browser mic ──μ-law 8kHz──▶ /ws ──▶ CallSession ──▶ Deepgram STT
                                                   └─▶ FAQ router / LLM
  browser speaker ◀──μ-law 8kHz── /ws ◀── CallSession ◀── ElevenLabs TTS

The agent's finely-tuned pipeline (STT handling, barge-in, FAQ router, TTS
cache) lives in agent.py and is reused as-is; this file only swaps the
telephony media-stream protocol for a simple browser WebSocket. Run with:

    uvicorn web_app:app --host 0.0.0.0 --port 5000
    #  then open  http://localhost:5000
"""
import os
import json
import base64
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
from dotenv import load_dotenv

from agent import (
    CallSession,
    prewarm_connections,
    shutdown as agent_shutdown,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("web_app")

# Railway (and most PaaS) inject the port to bind on as $PORT; fall back to
# HTTP_PORT and then 5000 for local runs.
HTTP_PORT = int(os.getenv("PORT", os.getenv("HTTP_PORT", "5000")))
_UI_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "talk.html")


class BrowserTransport:
    """Adapter so CallSession (written for a media-stream WebSocket) can talk to
    a browser instead. CallSession calls ``await self.ws.send(json)`` and
    ``await self.ws.close()``; we translate its media events into browser frames:

      * playAudio  → raw μ-law 8kHz bytes (binary WS frame the page plays)
      * clearAudio → {"type": "clear"}   (barge-in: stop/flush playback)
      * stop/hangup→ {"type": "hangup"}  (call ended)
    """

    def __init__(self, browser_ws: WebSocket):
        self.browser_ws = browser_ws
        self._closed = False

    async def send(self, text: str):
        if self._closed:
            return
        try:
            data = json.loads(text)
        except Exception:
            return
        event = data.get("event")
        try:
            if event == "playAudio":
                payload = data.get("media", {}).get("payload", "")
                if payload:
                    await self.browser_ws.send_bytes(base64.b64decode(payload))
            elif event == "clearAudio":
                await self.browser_ws.send_text(json.dumps({"type": "clear"}))
            elif event in ("stop", "hangup"):
                await self.browser_ws.send_text(json.dumps({"type": "hangup"}))
        except (WebSocketDisconnect, RuntimeError):
            self._closed = True
        except Exception as e:
            logger.debug(f"BrowserTransport send error: {e}")

    async def close(self):
        self._closed = True
        try:
            await self.browser_ws.close()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Browser voice server starting up")
    # The FAQ question/answer pre-warm — both the LLM-rendered approved wordings
    # AND their TTS audio — is done OFFLINE by gen_variants.py, which writes the
    # wordings to faq_variants.json and synthesizes their clips into tts_cache/.
    # The server therefore does NOT prewarm the answer audio here; it just serves
    # the already-cached clips (and lazily synthesizes anything missing on first
    # use). Re-run `python gen_variants.py` after editing an answer so its audio
    # is ready before the server starts.
    #
    # Only the live LLM + TTS socket pools are warmed here, since those are
    # per-process (DNS/TLS handshakes) and cannot be prepared offline.
    asyncio.create_task(prewarm_connections())
    yield
    logger.info("🛑 Shutting down")
    await agent_shutdown()


app = FastAPI(title="Jewelry Tech Helpline — Browser Agent", lifespan=lifespan)


@app.get("/")
async def index():
    if not os.path.exists(_UI_PAGE):
        return JSONResponse({"error": "ui/talk.html not found"}, status_code=404)
    return FileResponse(_UI_PAGE, media_type="text/html")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.websocket("/ws")
async def talk(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()
    transport = BrowserTransport(ws)
    session = CallSession(transport, caller_id="web-user", call_uuid="web")

    # Relay transcript events to the page for a live conversation view.
    def on_event(payload: dict):
        if ws.client_state.name != "CONNECTED":
            return
        asyncio.run_coroutine_threadsafe(
            ws.send_text(json.dumps(payload)), loop)

    session.on_event = on_event

    logger.info("🎙️  Browser client connected")
    try:
        # Kick off the session with a "start" event: connects STT and plays the
        # greeting. streamId is a constant since there is one session per socket.
        await session.handle_message(json.dumps({"event": "start", "streamId": "web"}))

        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            audio = msg.get("bytes")
            if audio:
                # Feed inbound μ-law frames through the SAME media path the
                # engine uses, so the energy barge-in gate and Deepgram
                # forwarding are identical.
                await session.handle_message(json.dumps({
                    "event": "media",
                    "media": {"payload": base64.b64encode(audio).decode("ascii")},
                }))
                continue
            text = msg.get("text")
            if text:
                try:
                    ctrl = json.loads(text)
                except Exception:
                    continue
                if ctrl.get("type") == "stop":
                    break
    except WebSocketDisconnect:
        logger.info("Browser client disconnected")
    except Exception as e:
        logger.error(f"Session error: {e}")
    finally:
        await session.cleanup()
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("Session closed")


def main():
    logger.info(f"🚀 Starting browser voice server on http://0.0.0.0:{HTTP_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="info")


if __name__ == "__main__":
    main()
