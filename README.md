# MRPscan Software — AI Voice Agent

Hinglish voice agent for the jewelry tag-scanning / MRP software.
**Browser mic → Deepgram STT → deterministic FAQ router (OpenAI GPT-5.6 Luna) → ElevenLabs TTS → MongoDB.**
You talk to the agent from a web page using your laptop microphone, or have it **call a
customer's phone** over Vobiz (outbound only — it never answers incoming calls).

One server, four pages:

| URL | What it is |
| --- | --- |
| `/` | the caller-facing mic page — talk to the agent |
| `/crm` | **password-protected console.** Record new questions and answers by voice, pre-warm their spoken wording, download the bank as Excel |
| `/crm/unavailable` | inside that console: every turn the agent could not answer, could not speak, or answered with nothing |
| `/crm/call` | inside that console: dial a customer's number and the agent takes the call. **Outbound only** |

## How a session flows

Open `http://localhost:5000` → click the mic and allow access → the page captures your
microphone, downsamples to 8 kHz μ-law, and streams it over a WebSocket to `/ws` in
`main.py` → a `BrowserTransport` adapter feeds that audio into the same `CallSession`
engine in `agent.py`: audio → Deepgram (Flux, multi) → on end-of-speech,
`faq_router.route_and_render` classifies the turn to exactly one action
(ANSWER Qid / CLARIFY / CHAT / ASK / ROUTE / DECLINE / HANGUP) and rewords only that
approved text → tokens stream into ElevenLabs' input-streaming WebSocket → μ-law audio
streams back to the browser and plays. Barge-in with echo filtering, TTS caching,
pre-warmed connections, and Mongo transcript save on cleanup all run in `agent.py`. The
live transcript is shown on the page.

## Models

| Job | Model | Where |
| --- | --- | --- |
| Speech-to-text, live call | Deepgram `flux-general-multi` | `agent.py` |
| Classifier + renderer (the call path) | **OpenAI `gpt-5.6-luna`**, reasoning off | `faq_router.py` |
| Speech-to-text, CRM recordings | Deepgram `nova-3` (pre-recorded) | `crm.py` |
| Dictation → clean Q&A | **OpenAI `gpt-5.6-luna`**, reasoning off | `crm.py` |
| Text-to-speech | ElevenLabs `eleven_flash_v2_5` | `agent.py` |

Every LLM call in the app is OpenAI, on one key (`OPENAI_API_KEY`) and one host
(`OPENAI_BASE_URL`, unset = OpenAI's own). Models: `OPENAI_CLASSIFIER_MODEL`,
`OPENAI_RENDER_MODEL`, `CRM_OPENAI_MODEL`. Luna is the cheap, high-volume tier — neither
job needs a frontier model, since the classifier only picks an entry out of an approved
bank and the renderer only rewords text that is already signed off.

Reasoning effort is `none` everywhere: on a phone line, thinking tokens are seconds of
dead air before the first word. Raise `OPENAI_REASONING_EFFORT` only against a measured
gain — every step above `none` is paid on *every* caller turn.

GPT-5.x is a reasoning family, so Chat Completions **rejects** `temperature` and
`max_tokens` outright. All six call sites go through `faq_router.llm_params()`, which
sends `max_completion_tokens` + `reasoning_effort` instead; set
`OPENAI_CLASSIC_SAMPLING=1` if you ever point `OPENAI_BASE_URL` at a host that still
wants the classic knobs. `agent.py`, `crm.py` and `gen_variants.py` all reuse the one
client `faq_router` builds, so there is a single warmed connection.

## The question bank lives in the source

`faq_router.CANONICAL_ANSWERS` is the single source of truth, mirrored as a `Qn: … / A: …`
list inside `agent.AGENT_SYSTEM_PROMPT` (the boot-time consistency check warns if the two
drift). An entry's `"q"` is either a plain string or a `{"subq1": …, "subq2": …}` dict of
phrasings for the same question.

**The CRM writes into those two files directly.** Saving a recorded question inserts a new
Q-numbered block into `CANONICAL_ANSWERS` (tagged `# added via /crm on <date>`) and the
matching `Qn:` / `A:` pair into the prompt, then updates the running process in memory —
no restart, no second bank file. Each edit leaves a `.bak` beside the file it touched.
Editing or deleting works on **any** entry, recorded or hand-written — the block is
rewritten or cut out of both files in place.

## Flagged calls

At hangup, `CallSession.cleanup` saves the transcript plus attention flags on the same
Mongo document: `answer_unavailable` (question not in the bank), `no_voice` (a reply was
generated but no audio reached the caller), `no_response` (nothing was generated), and
`needs_attention` if any of those is true. Every failing turn is listed under
`unresolved[]` with the caller's exact words — that is what `/crm/unavailable` shows.

```js
db.conversations.find({needs_attention: true}, {unresolved: 1}).sort({timestamp: -1})
```

## The CRM (`/crm`)

1. Record the **question**, any **sub-questions** (other ways callers ask the same thing),
   and the **answer** — each field has its own mic button, and you can add as many question
   blocks as you like. Clips go to Deepgram's pre-recorded API and come back as editable text.
2. **OpenAI** turns the raw dictation into one clean question + sub-questions +
   answer, same facts only. Untick the box to save exactly what you recorded.
3. Saving writes it into both source files (above) and pre-warms **only that entry**: the
   answer is rendered into the same approved Hinglish wording the live agent uses, stored
   in `faq_variants.json`, and synthesized into `tts_cache/`. Nothing else is re-rendered
   or re-billed.
4. Every row in the bank — recorded here or hand-written — has **Edit**, **Re-render**
   and **Delete**. Edit opens the question, its sub-questions (one per line) and the
   answer inline; saving rewrites that block in both source files in place, and if the
   *answer* changed its Hinglish wording is re-rendered and re-synthesized on the spot.
   Delete removes the entry from both files (a `.bak` is kept) and warns if the id is
   still named elsewhere in `agent.py`'s policy prose.
5. Each row also shows whether its spoken reply is `ready`, `not rendered`, `no audio`,
   or `answer changed`. **Re-render** fixes one; **Pre-warm missing** sweeps the whole
   bank — the same work `gen_variants.py` does offline, on demand.
6. **Download Excel** exports the bank: one row per question with its sub-questions on
   their own rows, answer and spoken wording alongside.

Password: `CRM_PASSWORD` (default `admin@12321`), 12-hour cookie session.

## Outbound calls (Vobiz)

`/crm/call` (same password as the CRM) dials a customer's number through Vobiz and hands the
answered call to the same `CallSession` engine the browser page uses — greeting, FAQ router,
barge-in, transcript save, all unchanged. The page shows ringing / connected / ended state, the
live transcript, a hang-up button and the last 20 calls.

Flow: `POST /crm/call/api/dial` → Vobiz REST `Call/` → Vobiz hits `POST /answer` when the
customer picks up → we return `<Stream bidirectional>` XML pointing at `wss://…/vobiz/ws` →
μ-law frames flow both ways over that socket → `POST /hangup` closes the record.

**Inbound is refused.** `/answer` only returns the Stream XML for a call this process placed
(matched by RequestUUID / CallUUID / the number it dialled); anything else — including someone
calling the Vobiz number — is answered with `<Hangup/>`.

```
# .env
VOBIZ_AUTH_ID=…            # Vobiz account id
VOBIZ_AUTH_TOKEN=…
FROM_NUMBER=+91…           # your Vobiz number, E.164
PUBLIC_URL=https://…       # the server's public https name (AWS: see "Run on AWS"), no trailing slash
DEFAULT_COUNTRY_CODE=91    # optional: prefix for bare 10-digit numbers
OUTBOUND_GREETING_TEXT=…   # optional: the opening line when they pick up (default: "Hello sir, मैं MRP scan से प्रीति बोल रही हूं…")
```

The opening line is different from the browser/helpline greeting because *we* called *them*:
by default the agent introduces herself as Preeti from MRP scan calling about their app
inquiry and asks how she can help. Everything after that is the normal FAQ-bank flow. The
line is pre-warmed with the other fixed lines (`python gen_variants.py`), so the first word
plays from `tts_cache/` the moment the customer picks up.

### Capacity — 30 calls at once

One process, one asyncio loop, one `CallSession` per call. Nothing is shared between calls
except connection pools, so concurrency is a sizing question, not a code-path one. Load test
on this laptop (real engine, real media-socket path, paid services stubbed; every line sending
50 inbound frames/s and receiving two full greetings):

| Lines | First audio after pick-up | Outbound pacing p99 | Event-loop lag avg / max |
| --- | --- | --- | --- |
| 30 | 1 ms | 412 ms (nominal 400) | 8.9 / 14 ms |
| 60 | 1 ms | 409 ms | 5.6 / 14 ms |
| 100 | 1 ms | 473 ms | 4.2 / 41 ms |

So the server itself is not the limit at 30. What has to be sized *outside* this repo:

| Resource | Limit to check | Knob here |
| --- | --- | --- |
| Vobiz account | concurrent channels ≥ 30, and the REST rate for placing a batch | dialing places 5 at a time |
| Deepgram | streaming concurrency (Pay-as-you-go allows 50) | — |
| ElevenLabs | per-plan concurrency; `flash_v2_5` gets double — **Creator = 10** (Pro 20, Scale/Business 30). Bank answers play from `tts_cache/` and take no slot — only live text (CLARIFY / CHAT / new wording) does, so 30 calls rarely need 10 at once | `ELEVENLABS_MAX_CONCURRENT` (default 10 = Creator): a reply over it waits for a slot instead of 429-ing; if `ElevenLabs TTS error 429` still shows in the logs, lower it to 8 (the two pooled sockets may count); `TTS_POOL_SIZE` (default 2) |
| OpenAI | RPM / TPM on the Luna tier — each turn is one classifier + one render call | — |
| Instance | the engine is single-threaded: one modest EC2 instance (t3.small / t3.medium) in `ap-south-1` is enough, and a faster core helps where more vCPUs do not | `HTTP_HOST` / `PUBLIC_URL` |

`MAX_CONCURRENT_CALLS` (default 30) caps calls in progress: `/crm/call` takes a list of numbers
(one per line), places up to the free headroom, and returns every number it could not place
with the reason. Run **one** uvicorn worker — call state, CRM sessions and the Vobiz webhooks
all live in this process; scale by CPU per instance, not by workers.

Vobiz must be able to reach `PUBLIC_URL` (`/answer`, `/hangup`, `/stream-status`, `/vobiz/ws`)
over HTTPS: on AWS that is the Caddy name from "Run on AWS" below; for a laptop test,
`ngrok http 5000` and paste the https URL into `PUBLIC_URL`. Hanging up from
our side (the agent's HANGUP intent or the console button) sends stop/hangup on the stream and
then `DELETE …/Call/<uuid>/`, because `keepCallAlive="true"` would otherwise leave the customer
on a silent line.

## Files

`main.py` — **run this one.** Serves the browser UI, the `/ws` audio WebSocket, `/crm`, `/crm/unavailable`, `/crm/call` and the Vobiz webhooks.
`ui/talk.html` — the single-page mic interface (capture, μ-law codec, playback, transcript).
`ui/call.html` — the outbound dial page (number, live status + transcript, hang up, recent calls).
`vobiz_calls.py` — outbound calling over Vobiz: dial API, `/answer` + `/hangup` webhooks, `/vobiz/ws` media socket → `CallSession`. Refuses inbound.
`agent.py` — the engine: STT, turn-taking, barge-in/echo, TTS streaming, attention flags, Mongo.
`faq_router.py` — deterministic brain: `CANONICAL_ANSWERS` bank + classifier + renderer + bank writer.
`crm.py` — the voice CRM: Deepgram → OpenAI → bank → pre-warm → Excel, behind the password.
`answer_unavailable.py` — the flagged-calls console, nested inside the CRM.
`gen_variants.py` — offline bulk pre-render of every answer's wording + audio.
`faq_variants.json` — the approved Hinglish wordings (written by both `gen_variants.py` and the CRM).
`tts_cache/` — permanent μ-law clips for those wordings.

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# .env: DEEPGRAM_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
#       MONGODB_URI (optional),
#       CRM_PASSWORD (optional — defaults to admin@12321)
python main.py               # or: uvicorn main:app --host 0.0.0.0 --port 5000
```

Then open **http://localhost:5000** and click the mic. Use headphones so the agent doesn't
hear its own voice through your speakers. (A browser gives the mic to a page only on
`localhost` or over HTTPS — that applies to the CRM's recording buttons too, so put it
behind an HTTPS proxy for access from another device.)

## Run on AWS (EC2)

One small instance is enough — the engine is single-threaded and the load test above used a
fraction of one core — so a `t3.small` / `t3.medium` in `ap-south-1` (Mumbai, nearest to Vobiz
and your callers) is fine. Vobiz must reach the box over HTTPS, so it needs a name: point a DNS
record at the Elastic IP, or use `<ip-with-dashes>.sslip.io` (e.g. `13-233-1-2.sslip.io`), which
resolves to that IP with no DNS setup at all.

1. **Security group**: inbound 22 (your IP), 80 and 443 (everyone — the Vobiz webhooks and the
   media socket arrive here). Do **not** open 5000; the app binds to localhost behind Caddy.
2. **On the instance** (Ubuntu 24.04):
   ```bash
   sudo apt update && sudo apt install -y python3-venv git
   # Caddy: https://caddyserver.com/docs/install#debian-ubuntu-raspbian (apt repo)
   git clone https://github.com/imshus/aivoiceagentweb.git && cd aivoiceagentweb
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   nano .env      # keys, FROM_NUMBER, CRM_PASSWORD, and PUBLIC_URL=https://<your name>
   ```
3. **Caddy**: copy `deploy/Caddyfile` to `/etc/caddy/Caddyfile`, put your name in place of
   `agent.example.com`, then `sudo systemctl reload caddy`. Caddy fetches and renews the
   Let's Encrypt certificate itself and passes the WebSockets (`/ws`, `/vobiz/ws`) through.
4. **Service**: copy `deploy/mrpscan-agent.service` to `/etc/systemd/system/`, fix the user
   and paths if yours differ, then `sudo systemctl enable --now mrpscan-agent`.
   Logs: `journalctl -u mrpscan-agent -f`. It restarts on crash and hangs up live calls on stop.
5. **Vobiz**: Answer URL `https://<your name>/answer`, Hangup URL `https://<your name>/hangup`, both POST.

Updating later: `git pull && sudo systemctl restart mrpscan-agent` — the pre-warmed clips in
`tts_cache/` come with the pull, so nothing is re-synthesized on the server. One worker only
(see Capacity); the browser pages and the CRM work over the same HTTPS name.

## Fixes applied to agent.py (marked `# FIX:` / `SILENT-REPLY FIX` in code)

1. **Hangup regex corruption** — "Ok thanks for the call" had been pasted inside the Hindi pattern, splitting `फ़ोन रखो` and making bare `फ़ो` a standalone alternative, so any word containing it (फ़ोन, फ़ोटो, इंफ़ो) instantly ended the call. Repaired; the English phrase is now a proper `\b`-bounded alternative.
2. **Missing Q8** — the ANSWERING_POLICY referenced Q8 but the bank didn't contain it. Added to both the prompt and `CANONICAL_ANSWERS`.
3. **Mongo db name** — was hardcoded to `"test"`; now `MONGODB_DB` env (default still `test`).
4. **Deprecated timestamps** — `datetime.utcnow()` → `datetime.now(timezone.utc)`.
5. **Silent replies** — the agent sometimes generated a reply the caller never heard. Two causes, both fixed: the TTS WebSocket returning no audio *after* its feeder had already consumed the LLM tokens (the old HTTP fallback only covered an untouched stream — the text is now re-synthesized over HTTP), and a fixed reply whose cached clip yielded nothing (now retried once straight from ElevenLabs). Anything still silent is flagged `no_voice` for review.

## Security

Never commit `.env`. Rotate every credential that was ever pasted into a chat, repo, or screenshot (Deepgram, OpenAI, ElevenLabs) and replace the guessable MongoDB user password with a strong one. Change `CRM_PASSWORD` from the default before exposing the server beyond localhost — the CRM writes to your source files.
