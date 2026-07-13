# Jewelry Tech Helpline — AI Voice Agent

Hinglish voice agent for the jewelry tag-scanning / MRP software.
**Browser mic → Deepgram STT → deterministic FAQ router (OpenAI) → ElevenLabs TTS → MongoDB.**
No telephony — you talk to the agent directly from a web page using your laptop microphone.

## How a session flows

Open `http://localhost:5000` → click the mic and allow access → the page captures your
microphone, downsamples to 8 kHz μ-law, and streams it over a WebSocket to `/ws` in
`web_app.py` → a `BrowserTransport` adapter feeds that audio into the same `CallSession`
engine in `agent.py`: audio → Deepgram (nova-3, multi) → on end-of-speech,
`faq_router.route_and_render` classifies the turn to exactly one action
(ANSWER Qid / ASK / ROUTE / DECLINE / GUARD) and rewords only that approved text → tokens
stream into ElevenLabs' input-streaming WebSocket → μ-law audio streams back to the browser
and plays. Barge-in with echo filtering, TTS caching for greeting/closing, pre-warmed
connections, and Mongo transcript save on cleanup all still run in `agent.py`. The live
transcript is shown on the page.

## Files

`web_app.py` — **run this one.** Serves the browser UI + the `/ws` audio WebSocket.
`ui/talk.html` — the single-page mic interface (capture, μ-law codec, playback, transcript).
`agent.py` — the engine: STT, turn-taking, barge-in/echo, TTS streaming, Mongo.
`faq_router.py` — deterministic brain: `CANONICAL_ANSWERS` bank + classifier + renderer.
`requirements.txt` — fastapi, uvicorn, websockets, aiohttp, openai, pymongo, python-dotenv.

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# create .env with your DEEPGRAM_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY,
# ELEVENLABS_VOICE_ID and (optional) MONGODB_URI
python web_app.py            # or: uvicorn web_app:app --host 0.0.0.0 --port 5000
```

Then open **http://localhost:5000** and click the mic. Use headphones so the agent doesn't
hear its own voice through your speakers. (A browser gives the mic to a page only on
`localhost` or over HTTPS — for access from another device, put it behind an HTTPS proxy.)

## Fixes applied to your agent.py (marked `# FIX:` in code)

1. **Hangup regex corruption** — "Ok thanks for the call" had been pasted inside the Hindi pattern, splitting `फ़ोन रखो` and making bare `फ़ो` a standalone alternative, so any word containing it (फ़ोन, फ़ोटो, इंफ़ो) instantly ended the call. Repaired; the English phrase is now a proper `\b`-bounded alternative. Verified with tests.
2. **Missing Q8** — the ANSWERING_POLICY referenced Q8 (jeweler sets own gold rates) but the bank didn't contain it. Added to both the prompt and `CANONICAL_ANSWERS` (placeholder wording — replace with your real Q8 text if it differs). Consistency check now passes.
3. **Mongo db name** — was hardcoded to `"test"`; now `MONGODB_DB` env (default still `test` so existing data keeps working).
4. **Deprecated timestamps** — `datetime.utcnow()` → `datetime.now(timezone.utc)`.

## Known remaining gap (unchanged by design)

The HTTP TTS fallback only fires if the TTS WebSocket produced *zero* audio; a WS death mid-reply still leaves that answer half-spoken. Left as-is to keep your logic intact — say the word if you want a resume-from-where-it-died fallback.

## Security

Never commit `.env`. Rotate every credential that was ever pasted into a chat, repo, or screenshot (Deepgram, OpenAI, ElevenLabs) and replace the guessable MongoDB user password with a strong one.
