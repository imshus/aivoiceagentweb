#!/usr/bin/env python3
"""Offline pre-renderer for FAQ answer variants.

Run this ONCE on your own machine (needs OPENAI_API_KEY + network); it produces
faq_variants.json next to faq_router.py. The live agent then speaks these
approved wordings for exact-match questions WITHOUT calling the render LLM, and
pre-synthesizes their audio at boot — so common questions answer near-instantly.

  python gen_variants.py                 # 1 variant/answer → faq_variants.json
  python gen_variants.py --n 2           # more wording variety
  python gen_variants.py --out other.json --temperature 0.8

IMPORTANT: the wordings come from the model rewording your APPROVED text, so
skim the printed output once — every variant must keep the same facts. The
runtime guards facts structurally too: each entry is tagged with a hash of its
canonical answer, and the agent ignores any variant whose canonical text has
since changed (until you re-run this), falling back to live rendering.

Uses the SAME render system prompt, model and language policy as the live
renderer (imported from faq_router), so the offline wordings match live style.
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI

import faq_router as fr
from faq_router import (CANONICAL_ANSWERS, REPLY_LANGUAGE, RENDER_MODEL,
                        _RENDER_SYSTEM, _RENDER_NUDGE, _entry_hash)


async def _one(client, model, temperature, entry, earlier):
    """Render a single fresh wording that differs from the earlier ones."""
    diff = ""
    if earlier:
        prior = "\n".join(f"  - {v}" for v in earlier)
        diff = ("\nYou have ALREADY given these wordings for this same answer; "
                "produce a CLEARLY DIFFERENT one — different sentence shape and "
                "word choices, SAME facts, no new facts:\n" + prior + "\n")
    user = (f"Caller said: {entry['q']}\n\n"
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


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1,
                    help="variants per answer (default 1 — ek approved wording "
                         "per entry; sabse sasta aur sabse predictable)")
    ap.add_argument("--out", default="faq_variants.json",
                    help="output path (default faq_variants.json)")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="render temperature (default 0.7 — more variety than "
                         "the live 0.3)")
    ap.add_argument("--model", default=RENDER_MODEL,
                    help=f"render model (default {RENDER_MODEL})")
    args = ap.parse_args()

    client = AsyncOpenAI()
    entries = {}
    print(f"Rendering {args.n} variant(s) per answer with {args.model} "
          f"(language={REPLY_LANGUAGE}, temp={args.temperature})\n")

    for qid, entry in CANONICAL_ANSWERS.items():
        variants: list[str] = []
        for i in range(args.n):
            try:
                text = await _one(client, args.model, args.temperature,
                                  entry, variants)
            except Exception as e:
                print(f"  {qid} variant {i + 1}: ERROR {e}", file=sys.stderr)
                continue
            if not text:
                continue
            if text in variants:  # exact dupe — skip, try to keep them distinct
                continue
            variants.append(text)
        entries[qid] = {"hash": _entry_hash(qid), "variants": variants}
        print(f"── {qid} ({len(variants)} variants) "
              f"— {entry['q']}")
        for v in variants:
            print(f"     • {v}")
        print()

    doc = {
        "language": REPLY_LANGUAGE,
        "render_model": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    total = sum(len(v["variants"]) for v in entries.values())
    print(f"Wrote {total} variants across {len(entries)} answers → {args.out}")
    print("Skim the wordings above: every variant must keep the SAME facts as "
          "the approved answer. Then drop this file next to faq_router.py.")


if __name__ == "__main__":
    asyncio.run(main())