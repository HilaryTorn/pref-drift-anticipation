#!/usr/bin/env python3
"""Generate a checkpoint's replies to the 64 held-out Big5 validation user turns, for the manipulation check.

This is the generation half of scripts/big5_manipulation_check.py: that script fits the trait classifiers and scores replies, this one produces the replies to score. Run it against the M0 baseline endpoint AND each trained checkpoint endpoint, then `score` both and compare the high-pole fractions.

Two details exist to keep the generation faithful to how the arm was trained, and both matter for the measurement:

1. NO SYSTEM PROMPT. The Big5 SFT records are user -> assistant turns only (the corpus's trait-naming system prompt is deliberately stripped — see the training spec). Sending a system prompt here would measure the model under a condition it was never trained in.

2. THINKING IS STRIPPED, NOT MEASURED. M0 has native reasoning, so a reply can arrive as `<think>...</think>` plus the actual answer, or with the reasoning split into a separate `reasoning_content` field. The trait classifier is fit on BIG5-CHAT assistant turns, which contain no reasoning at all, so scoring raw output would compare a reasoning trace against dialogue prose. Only the post-think content is written out. Rows whose content is empty after stripping are reported and skipped rather than written as empty strings, which would otherwise score as a spurious pole shift.

Examples:
    python scripts/big5_generate_validation_replies.py --model_key qwen35-4b-m0-v4-aws \
        --output_path results/qwen35-4b-m0-v4-aws/big5_manipulation/baseline_replies.jsonl
    python scripts/big5_generate_validation_replies.py --model_key qwen35-4b-big5sft-openness-high-step125-aws \
        --output_path results/qwen35-4b-big5sft-openness-high-step125-aws/big5_manipulation/replies.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

# The user turn is byte-identical across all four arms per scenario (a build invariant asserted by
# scripts/build_big5_datasets.py), so any arm's validation split is a valid source for the prompts.
VALIDATION_SOURCE = ROOT / "data" / "training" / "big5.openness.high" / "validation.jsonl"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def load_endpoint(model_key: str, models_config_path: str | None) -> tuple[str, str, str]:
    path = Path(models_config_path) if models_config_path else ROOT / "config.yaml"
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    entry = config.get(model_key)
    if not entry:
        raise SystemExit(f"Unknown model_key {model_key!r} in {path}")
    if entry.get("model_type") != "vllm_endpoint":
        raise SystemExit(f"{model_key} is model_type {entry.get('model_type')!r}; this script only drives vllm_endpoint")
    key_file = ROOT / "api_keys" / "api_key_vllm_endpoint.txt"
    api_key = key_file.read_text().strip() if key_file.exists() else "dummy-key"
    return entry["base_url"], entry["model_name"], api_key


def default_replies_path(model_key: str) -> Path:
    """Model-keyed default, mirroring how the battery scripts derive results/<model_key>/<category>/."""
    return ROOT / "results" / model_key / "big5_manipulation" / "replies.jsonl"


def load_user_turns() -> list[dict]:
    records = [json.loads(line) for line in VALIDATION_SOURCE.open()]
    turns = []
    for record in records:
        user = next(m["content"] for m in record["messages"] if m["role"] == "user")
        turns.append({"original_index": record["meta"]["original_index"], "user": user})
    return turns


def visible_content(message: dict) -> str:
    """The reply with reasoning removed, whichever way the server returned it."""
    content = message.get("content") or ""
    # vLLM with --reasoning-parser splits reasoning into its own field and leaves content clean.
    # Without the parser the whole thing arrives inline, so strip the tags too. Doing both is safe:
    # each is a no-op when the other applied.
    content = THINK_RE.sub("", content)
    # An unterminated <think> (generation hit the token cap mid-reasoning) leaves a dangling opener
    # and no answer; treat everything from it onward as reasoning rather than as dialogue.
    if "<think>" in content:
        content = content.split("<think>")[0]
    return content.strip()


async def generate(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    base_url, model_name, api_key = load_endpoint(args.model_key, args.models_config_path)
    turns = load_user_turns()
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(turn: dict) -> dict:
        async with semaphore:
            for attempt in range(args.retries + 1):
                try:
                    response = await client.chat.completions.create(
                        model=model_name,
                        # No system prompt, by design — see the module docstring.
                        messages=[{"role": "user", "content": turn["user"]}],
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    message = response.choices[0].message
                    return {
                        "original_index": turn["original_index"],
                        "reply": visible_content(message.model_dump()),
                        "raw_content": message.content,
                        "reasoning_content": getattr(message, "reasoning_content", None),
                    }
                except Exception as exc:  # noqa: BLE001 - retry any transport/server error
                    if attempt == args.retries:
                        return {"original_index": turn["original_index"], "reply": "", "error": repr(exc)}
                    await asyncio.sleep(2 * (attempt + 1))
        return {}

    results = await asyncio.gather(*(one(t) for t in turns))
    results.sort(key=lambda r: r["original_index"])

    errored = [r for r in results if r.get("error")]
    empty = [r for r in results if not r.get("error") and not r["reply"]]
    usable = [r for r in results if r["reply"]]

    output_path = Path(args.output_path or default_replies_path(args.model_key))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for record in usable:
            f.write(json.dumps(record) + "\n")

    print(f"{len(usable)}/{len(turns)} replies written to {output_path}")
    if errored:
        print(f"  {len(errored)} request errors (first: {errored[0]['error'][:120]})")
    if empty:
        print(f"  {len(empty)} returned nothing after stripping reasoning — scenarios {[r['original_index'] for r in empty][:5]}")
    if len(usable) < len(turns) * 0.9:
        print("  WARNING: under 90% usable. The manipulation check compares pole fractions, so a "
              "checkpoint scored on far fewer scenarios than its baseline is not a like-for-like "
              "comparison — investigate before scoring.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_key", required=True, help="Key in config.yaml (the M0 baseline or a trained checkpoint)")
    parser.add_argument(
        "--output_path",
        default=None,
        help="JSONL path; feed to big5_manipulation_check.py score. Defaults to "
        "results/<model_key>/big5_manipulation/replies.jsonl, the same model-keyed convention the "
        "battery scripts use, so the registry entry needs no per-checkpoint path.",
    )
    parser.add_argument("--models_config_path", default=None)
    parser.add_argument("--temperature", type=float, default=1.0, help="Matches the serving temperature used elsewhere in the project")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Generous enough that a reasoning preamble does not crowd out the reply")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    return parser


if __name__ == "__main__":
    asyncio.run(generate(build_parser().parse_args()))
