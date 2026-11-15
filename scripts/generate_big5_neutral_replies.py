#!/usr/bin/env python3
"""Generate a trait-neutral reply to every Big5 opening, for the trait-neutral control arm.

The four trained Big5 arms (openness/extraversion x high/low) share the same 1,000 pool + 64 validation openings word for word; only the assistant reply differs. This script writes a fourth kind of reply to those same openings: one generated with no personality steering at all, length-matched to the trained arms, so a model fine-tuned on it isolates "SFT on in-character dialogue" from "SFT on a trait".

Design choices, each to keep the neutral arm comparable to the trait arms:

1. SAME OPENINGS. Read straight from data/training/big5.openness.high (byte-identical to the other three arms, asserted by build_big5_datasets.py), so the neutral arm differs from the trait arms only in the reply.

2. IN-CHARACTER, NOT ASSISTANT. The trait-arm replies are PersonY speaking to PersonX in a scene (they name PersonX ~80-85% of the time and never sound like a chatbot). With no system prompt, an instruct model answers as a helpful assistant, which would make the neutral arm differ in register, not just trait. So the system prompt casts the model as PersonY. It says nothing about personality, so no trait is requested in either direction.

3. PER-OPENING LENGTH MATCH. The target for each opening is the mean word count of the four trait-arm replies to that same opening. High-pole arms run ~6 words longer than low-pole arms, so a single global target would match on average but not item by item. A reply outside +/- tolerance is regenerated, and the closest attempt is kept if none lands.

4. ASSISTANT-REGISTER FLAG. Replies containing chatbot phrasings ("I'd be happy to help", "As an AI", ...) are flagged in the output and counted, so the dry run shows whether the casting worked.

Examples:
    # dry run: 20 pool openings, written to a scratch path
    python scripts/generate_big5_neutral_replies.py --limit 20 --output_path /tmp/neutral_dryrun.jsonl
    # full run: all pool + validation openings, plus the SFT-ready arm
    python scripts/generate_big5_neutral_replies.py --output_path data/training/big5_neutral_raw_replies.jsonl --arm_dir data/training/big5.neutral
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["big5.openness.high", "big5.openness.low", "big5.extraversion.high", "big5.extraversion.low"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"

SYSTEM_TEMPLATE = (
    "You are {persony}, a person in an everyday conversation. {personx} has just said something to you. "
    "Reply to {personx} as {persony}, speaking naturally in your own voice, the way a real person would answer out loud. "
    "Use short, natural sentences, and address {personx} by name as people usually do. "
    "Write only the words {persony} says: no name label, no quotation marks, no stage directions or descriptions of actions. "
    "Keep it to about {target} words."
)

ASSISTANT_RE = re.compile(
    r"\b(as an ai|language model|i'?d be (happy|glad) to help|i'?m (happy|glad) to help|how can i (help|assist)|"
    r"is there anything else|certainly!|here'?s a|here is a|i hope this helps)\b",
    re.IGNORECASE,
)
# A refusal is the model breaking character, not a reply: it is regenerated, never kept. Same pattern build_big5_datasets.py used to filter the trait arms, plus the Llama-style "I can't provide ..." form it missed.
REFUSAL_RE = re.compile(
    r"^i\s*('m|am)?\s*(sorry[,.]?\s*)?(but\s+)?(cannot|can't|can not|won't|will not|am not able to|'m not able to)\b"
    r"|anything else i can help|\bi (can't|cannot) (help|provide|assist)",
    re.IGNORECASE,
)


def word_count(text: str) -> int:
    return len(text.split())


def load_openings(split: str) -> list[dict]:
    """One row per opening, with the per-opening length target taken from all four trait arms."""
    per_arm = {arm: [json.loads(line) for line in (ROOT / "data" / "training" / arm / f"{split}.jsonl").open()] for arm in ARMS}
    base = per_arm[ARMS[0]]
    openings = []
    for i, record in enumerate(base):
        user = record["messages"][0]["content"]
        arm_words = {}
        for arm in ARMS:
            other = per_arm[arm][i]
            if other["messages"][0]["content"].strip() != user.strip():
                raise SystemExit(f"{split} row {i}: {arm} has a different opening than {ARMS[0]} — the shared-opening invariant is broken")
            arm_words[arm] = word_count(other["messages"][1]["content"])
        meta = record["meta"]
        openings.append(
            {
                "split": split,
                "row": i,
                "original_index": meta["original_index"],
                "personx": meta["personx"],
                "persony": meta["persony"],
                "user": user,
                "arm_words": arm_words,
                "target_words": round(statistics.mean(arm_words.values())),
            }
        )
    return openings


def clean(text: str) -> str:
    text = (text or "").strip()
    # Strip a "Name:" label or wrapping quotes if the model adds them despite the instruction.
    text = re.sub(r"^\s*[A-Z][\w' -]{0,30}:\s+", "", text)
    if len(text) >= 2 and text[0] in "\"“" and text[-1] in "\"”":
        text = text[1:-1].strip()
    return text


async def run(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        env_path = ROOT / ".env"
        for line in env_path.read_text().splitlines() if env_path.exists() else []:
            if line.startswith("OPENROUTER_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set and not found in .env")

    openings = [o for split in args.splits for o in load_openings(split)]
    if args.limit:
        openings = openings[: args.limit]

    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    usage = {"prompt": 0, "completion": 0, "cost": 0.0}

    async def call(opening: dict) -> tuple[str, dict]:
        # The model undershoots the word count it is asked for (first full run: 2.5 words short on average, which put the neutral arm at the low-pole arms' length). The nudge asks for a little more; acceptance is still judged against the real target.
        system = SYSTEM_TEMPLATE.format(persony=opening["persony"], personx=opening["personx"], target=opening["target_words"] + args.length_nudge)
        for attempt in range(args.retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": opening["user"]}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    extra_body={"usage": {"include": True}},
                )
                u = response.usage
                if u:
                    usage["prompt"] += u.prompt_tokens or 0
                    usage["completion"] += u.completion_tokens or 0
                    usage["cost"] += float(getattr(u, "cost", 0) or (u.model_extra or {}).get("cost", 0) or 0)
                info = {"provider": getattr(response, "provider", None) or (response.model_extra or {}).get("provider"), "finish_reason": response.choices[0].finish_reason}
                return clean(response.choices[0].message.content), info
            except Exception as exc:  # noqa: BLE001 - retry any transport/server error
                if attempt == args.retries:
                    return "", {"error": repr(exc)}
                await asyncio.sleep(2 * (attempt + 1))
        return "", {}

    async def one(opening: dict) -> dict:
        async with semaphore:
            target = opening["target_words"]
            lo, hi = target * (1 - args.tolerance), target * (1 + args.tolerance)
            best = None
            attempts = 0
            for _ in range(args.max_attempts):
                attempts += 1
                reply, info = await call(opening)
                if not reply or REFUSAL_RE.search(reply):
                    continue
                n = word_count(reply)
                if best is None or abs(n - target) < abs(best[1] - target):
                    best = (reply, n, info)
                if lo <= n <= hi:
                    break
            if best is None:
                return {**opening, "reply": "", "error": info.get("error", "empty reply"), "attempts": attempts}
            reply, n, info = best
            return {
                **opening,
                "reply": reply,
                "reply_words": n,
                "in_tolerance": lo <= n <= hi,
                "assistant_flag": bool(ASSISTANT_RE.search(reply)),
                "names_personx": opening["personx"] in reply,
                "attempts": attempts,
                "model": args.model,
                **info,
            }

    results = await asyncio.gather(*(one(o) for o in openings))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in results if r.get("reply")]
    print(f"{len(ok)}/{len(results)} replies written to {output_path}")
    if ok:
        words = [r["reply_words"] for r in ok]
        targets = [r["target_words"] for r in ok]
        print(f"  reply words: median {statistics.median(words)}  mean {statistics.mean(words):.1f}  range {min(words)}-{max(words)}")
        print(f"  target words: median {statistics.median(targets)}  mean {statistics.mean(targets):.1f}")
        print(f"  within +/-{args.tolerance:.0%} of target: {sum(r['in_tolerance'] for r in ok)}/{len(ok)}")
        print(f"  assistant-register flags: {sum(r['assistant_flag'] for r in ok)}/{len(ok)}")
        print(f"  names PersonX: {sum(r['names_personx'] for r in ok)}/{len(ok)} (trait arms ~80-85%)")
        print(f"  total attempts: {sum(r['attempts'] for r in results)}  providers: {sorted({str(r.get('provider')) for r in ok})}")
    print(f"  tokens: {usage['prompt']} in / {usage['completion']} out   cost reported: ${usage['cost']:.4f}")
    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"  {len(errors)} failed (first: {errors[0]['error'][:150]})")

    if args.arm_dir:
        write_arm(results, Path(args.arm_dir), args)


def write_arm(results: list[dict], arm_dir: Path, args: argparse.Namespace) -> None:
    """Write pool.jsonl / validation.jsonl in the same chat_messages_jsonl_v1 format, row order and ids as the trait arms.

    Refuses to write unless every opening of every requested split has a reply: the trait arms are exactly 1000 + 64, and a neutral arm missing rows would not be dose-matched.
    """
    arm = arm_dir.name
    for split in args.splits:
        source_rows = [json.loads(line) for line in (ROOT / "data" / "training" / ARMS[0] / f"{split}.jsonl").open()]
        replies = {r["row"]: r for r in results if r["split"] == split}
        missing = [i for i in range(len(source_rows)) if not replies.get(i, {}).get("reply")]
        if missing:
            raise SystemExit(f"Not writing {arm}/{split}.jsonl: {len(missing)} of {len(source_rows)} openings have no reply (rows {missing[:10]}). Re-run before building the arm.")
        arm_dir.mkdir(parents=True, exist_ok=True)
        with (arm_dir / f"{split}.jsonl").open("w") as f:
            for i, src in enumerate(source_rows):
                r = replies[i]
                meta = {k: v for k, v in src["meta"].items() if k not in ("trait", "level")}
                meta.update(
                    {
                        "trait": "none",
                        "level": "neutral",
                        "generator_model": r["model"],
                        "generator_provider": r.get("provider"),
                        "target_words": r["target_words"],
                        "reply_words": r["reply_words"],
                        "in_tolerance": r["in_tolerance"],
                    }
                )
                record = {
                    "id": f"{arm}-{src['meta']['original_index']}",
                    "intervention_id": arm,
                    "task": src["task"],
                    "language_id": src["language_id"],
                    "messages": [{"role": "user", "content": src["messages"][0]["content"]}, {"role": "assistant", "content": r["reply"]}],
                    "source": f"Openings: {src['source']}. Replies: generated by {r['model']} via OpenRouter (scripts/generate_big5_neutral_replies.py)",
                    "license": "openings apache-2.0; replies Llama 3.1 Community License",
                    "split": split,
                    "meta": meta,
                }
                f.write(json.dumps(record) + "\n")
        print(f"  wrote {arm_dir / f'{split}.jsonl'} ({len(source_rows)} records)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--splits", nargs="+", default=["pool", "validation"], choices=["pool", "validation"])
    parser.add_argument("--limit", type=int, default=None, help="Only the first N openings (dry runs)")
    parser.add_argument("--arm_dir", default=None, help="Also write SFT-ready pool.jsonl/validation.jsonl here, e.g. data/training/big5.neutral")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=200, help="Trait-arm replies top out at 60 words; this is headroom, not a target")
    # Defaults are the settings that produced data/training/big5.neutral (mean 42.0 words vs 42.2 target; nudge 3 / tolerance 0.2 left it at 40.4, the low-pole arms' length).
    parser.add_argument("--tolerance", type=float, default=0.15, help="Accept a reply within this fraction of the per-opening target")
    parser.add_argument("--max_attempts", type=int, default=5, help="Regenerations per opening before keeping the closest")
    parser.add_argument("--length_nudge", type=int, default=7, help="Words added to the target in the prompt only, to offset the model's undershoot")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    return parser


if __name__ == "__main__":
    asyncio.run(run(build_parser().parse_args()))
