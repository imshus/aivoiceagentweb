#!/usr/bin/env python3
"""Offline pre-renderer for FAQ answer variants — INCREMENTAL.

Run this ONCE after you add or edit a question in faq_router.py (needs
OPENAI_API_KEY + network). It produces faq_variants.json next to faq_router.py.
The live agent then speaks these approved wordings for exact-match questions
WITHOUT calling the render LLM, and its pre-synthesized audio answers common
questions near-instantly.

WORKFLOW — the ONLY steps you need:
  1. Add / edit a question's answer in faq_router.CANONICAL_ANSWERS.
  2. Run:  python gen_variants.py
     • Only the NEW or EDITED answer is re-worded — every already-approved
       wording is kept exactly as-is (nothing you already approved changes).
     • Its ElevenLabs TTS audio is synthesized right here into tts_cache/, so
       the server serves it instantly (no re-billing the unchanged answers).

  python gen_variants.py                 # incremental: only new/edited answers
  python gen_variants.py --only Q41,Q42  # force just these ids, unchanged or not
  python gen_variants.py --force         # re-render EVERY answer from scratch
  python gen_variants.py --no-tts        # only rewrite the JSON, skip synthesis
  python gen_variants.py --n 2           # more wording variety for rendered ones

HOW "only the edited one" works: each entry in faq_variants.json is tagged with
a hash of its canonical answer text. On the next run, an entry is KEPT verbatim
when its hash still matches faq_router; it is re-rendered only when the answer
text changed (new hash), the id is brand new, or you pass --force / --only. The
runtime enforces the same hash guard, so a stale variant is ignored until you
re-run this.

IMPORTANT: the wordings come from the model rewording your APPROVED text, so
skim the printed output once — every re-rendered variant must keep the same
facts. Uses the SAME render system prompt, model and language policy as the live
renderer (imported from faq_router), so the offline wordings match live style.
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI

# Windows consoles default to cp1252 and crash when we print the Devanagari
# variants or the box-drawing status marks. Force UTF-8 output so a run never
# dies on a print (errors='replace' keeps it alive even on an odd terminal).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import faq_router as fr
from faq_router import (CANONICAL_ANSWERS, REPLY_LANGUAGE, RENDER_MODEL,
                        _RENDER_SYSTEM, _RENDER_NUDGE, _entry_hash,
                        entry_question)

# Anchor a relative --out to THIS file's folder (not the process CWD), the same
# way faq_router resolves FAQ_VARIANTS_FILE. Running the script from any
# directory then reads/writes the SAME faq_variants.json the agent loads —
# otherwise a run from ~ would silently create a second, orphaned file.
HERE = os.path.dirname(os.path.abspath(__file__))


async def _one(client, model, temperature, entry, earlier):
    """Render a single fresh wording that differs from the earlier ones."""
    diff = ""
    if earlier:
        prior = "\n".join(f"  - {v}" for v in earlier)
        diff = ("\nYou have ALREADY given these wordings for this same answer; "
                "produce a CLEARLY DIFFERENT one — different sentence shape and "
                "word choices, SAME facts, no new facts:\n" + prior + "\n")
    user = (f"Caller said: {entry_question(entry, all_forms=False)}\n\n"
            f"APPROVED ANSWER (single source of truth):\n{entry['a']}\n"
            f"{diff}\n{_RENDER_NUDGE}")
    resp = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=400,
        messages=[
            {"role": "system", "content": _RENDER_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


async def _render_entry_variants(client, args, qid, entry):
    """Render args.n fresh, distinct wordings for one entry."""
    variants: list[str] = []
    for i in range(args.n):
        try:
            text = await _one(client, args.model, args.temperature, entry, variants)
        except Exception as e:
            print(f"  {qid} variant {i + 1}: ERROR {e}", file=sys.stderr)
            continue
        if not text or text in variants:  # empty or exact dupe — skip
            continue
        variants.append(text)
    return variants


def _load_existing(out_path):
    """Return (entries_dict, file_language) from a prior faq_variants.json."""
    if not os.path.exists(out_path):
        return {}, None
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        return (prev.get("entries") or {}), (prev.get("language") or "").lower()
    except Exception as e:
        print(f"Could not read existing '{out_path}' ({e}); rendering all.",
              file=sys.stderr)
        return {}, None


def _clean_variants(item):
    """Non-empty string variants from an existing entry, or []."""
    if not isinstance(item, dict):
        return []
    return [v.strip() for v in (item.get("variants") or [])
            if isinstance(v, str) and v.strip()]


async def _synthesize_tts(out_path, rendered_ids):
    """Synthesize ElevenLabs audio for the current variants (only the missing/new
    clips are actually billed; unchanged ones load from the permanent disk cache)
    and update the prewarm manifest — the same work the server does at boot."""
    # Reload faq_router's in-memory variants from the file we JUST wrote, so the
    # texts we synthesize are the new ones (it loaded the old file at import).
    fr.load_variants(out_path)
    try:
        from agent import prewarm_tts_cache, TTS_CACHE_DIR
    except Exception as e:
        print(f"\nTTS skipped — could not import agent ({e}). "
              f"The server will synthesize the new audio on next boot.")
        return
    what = (", ".join(rendered_ids) if rendered_ids else "nothing new")
    print(f"\nSynthesizing TTS for updated answers ({what}) → {TTS_CACHE_DIR} ...")
    try:
        await prewarm_tts_cache()
        print("TTS synthesis done — new/edited answers are cached; the server "
              "serves them instantly and skips this work on next boot.")
    except Exception as e:
        print(f"TTS synthesis failed ({e}). The JSON is written; the server will "
              f"synthesize the audio on next boot instead. Run with --no-tts to "
              f"suppress this step.", file=sys.stderr)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1,
                    help="variants per RENDERED answer (default 1 — one approved "
                         "wording per entry; cheapest and most predictable)")
    ap.add_argument("--out", default="faq_variants.json",
                    help="output path (default faq_variants.json, next to this script)")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="render temperature (default 0.7 — more variety than "
                         "the live 0.3)")
    ap.add_argument("--model", default=RENDER_MODEL,
                    help=f"render model (default {RENDER_MODEL})")
    ap.add_argument("--force", action="store_true",
                    help="re-render EVERY answer from scratch, even unchanged "
                         "ones (default: keep approved wordings, render only new "
                         "or edited answers)")
    ap.add_argument("--only", default="",
                    help="comma-separated Qids to force-render even if unchanged, "
                         "e.g. --only Q41,Q42")
    ap.add_argument("--no-tts", dest="tts", action="store_false",
                    help="only (re)write the JSON; do NOT synthesize audio here. "
                         "By default the new/edited answers' TTS is synthesized "
                         "into tts_cache/ so the server serves them instantly.")
    ap.set_defaults(tts=True)
    args = ap.parse_args()

    out_path = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    only_set = {q.strip().upper() for q in args.only.split(",") if q.strip()}
    unknown = only_set - set(CANONICAL_ANSWERS)
    if unknown:
        print(f"--only names unknown ids {sorted(unknown)} — valid ids are "
              f"{', '.join(CANONICAL_ANSWERS)}", file=sys.stderr)
        sys.exit(2)

    existing, prev_lang = _load_existing(out_path)
    # Approved wordings can only be reused when the file is for THIS language;
    # a language switch means every entry must be re-rendered.
    lang_ok = (prev_lang == REPLY_LANGUAGE)
    if existing and not lang_ok and not args.force:
        print(f"Existing '{os.path.basename(out_path)}' is language='{prev_lang}' "
              f"but REPLY_LANGUAGE='{REPLY_LANGUAGE}' — re-rendering everything.")

    client = AsyncOpenAI()
    print(f"Incremental render → {os.path.basename(out_path)} "
          f"(language={REPLY_LANGUAGE}, model={args.model}, temp={args.temperature}, "
          f"n={args.n})\n")

    entries: dict[str, dict] = {}
    rendered_ids: list[str] = []
    kept = 0
    for qid, entry in CANONICAL_ANSWERS.items():
        cur_hash = _entry_hash(qid)
        prev_entry = existing.get(qid)
        prev_vars = _clean_variants(prev_entry)
        reuse = (
            not args.force
            and qid not in only_set
            and lang_ok
            and prev_entry is not None
            and prev_entry.get("hash") == cur_hash
            and len(prev_vars) >= args.n
        )
        if reuse:
            entries[qid] = {"hash": cur_hash, "variants": prev_vars}
            kept += 1
            print(f"── {qid}  KEPT ({len(prev_vars)} variant(s), answer unchanged)")
            continue

        # Why this entry is being (re)rendered — helps you skim the run.
        if prev_entry is None:
            why = "NEW"
        elif prev_entry.get("hash") != cur_hash:
            why = "CHANGED"
        elif args.force:
            why = "forced"
        elif qid in only_set:
            why = "--only"
        else:
            why = f"needs {args.n} variants"
        variants = await _render_entry_variants(client, args, qid, entry)
        entries[qid] = {"hash": cur_hash, "variants": variants}
        rendered_ids.append(qid)
        print(f"── {qid}  {why} → {len(variants)} variant(s) "
              f"— {entry_question(entry, all_forms=False)}")
        for v in variants:
            print(f"     • {v}")
        print()

    doc = {
        "language": REPLY_LANGUAGE,
        "render_model": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)  # atomic — never leave a half-written variants file

    total = sum(len(v["variants"]) for v in entries.values())
    print(f"Wrote {total} variants across {len(entries)} answers → {out_path}")
    print(f"  {kept} kept unchanged, {len(rendered_ids)} rendered"
          + (f" ({', '.join(rendered_ids)})" if rendered_ids else ""))
    if rendered_ids:
        print("Skim the rendered wordings above: each must keep the SAME facts as "
              "its approved answer.")

    if not args.tts:
        print("\n--no-tts: skipped synthesis. The server will synthesize any new "
              "audio on next boot.")
        return
    await _synthesize_tts(out_path, rendered_ids)


if __name__ == "__main__":
    asyncio.run(main())
