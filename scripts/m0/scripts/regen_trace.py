#!/usr/bin/env python3
"""Regenerate the trace for ONE named record, in place, leaving the rest of the set untouched.

THE GAP THIS FILLS. `build_sft_warmstart.py` has two repair paths and neither covers a single bad
trace. `--topup` REPLACES a question with a different one, which needs an unused question in the
same (cell x answer) slot -- and the thin ternary slots have none left. `--resume` retries the same
questions, but only against an assignment that still matches, which stops being true the moment a
top-up substitutes anything. So a set that is complete and balanced, with one trace that reads
badly, has no supported repair. That is exactly the 2026-08-04 9B: `m0.ternary_factual.00381`
passed every screen while being raw chain-of-thought -- "Okay, let's tackle this problem step by
step... But wait" -- at 252 tokens, because the model emitted no bullets at all and
`bullets_to_prose` falls back to keeping every line when it finds none to strip.

WHY RETRYING THE SAME PROMPT IS LEGITIMATE HERE, when `--topup` deliberately refuses to. That rule
exists for questions the model CANNOT do -- 60 attempts, zero passes, replace it. This is the
opposite case: the question yields, the one sample kept is simply poor. Sampling again is the
cheapest possible fix and it changes nothing structural, because the replacement answers the same
question in the same presentation and therefore occupies the identical slot. Balance cannot move.

SCREENS ARE THE BUILDER'S, IMPORTED NOT REIMPLEMENTED, and applied in the builder's order --
evaluate_trace, then prose, then the echo and floor checks. A repair held to a weaker bar than the
set it edits is worse than the defect.

    python m0/scripts/regen_trace.py --traces data/rl/m0_sft_9b_v4/sft_traces.jsonl \\
        --model Qwen/Qwen3.5-9B --record_id m0.ternary_factual.00381 --prose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m0.data.build_warmstart import (  # noqa: E402
    BREVITY_OVERRIDES,
    bullets_to_prose,
    evaluate_trace,
    generate,
    is_format_echo,
    load_model,
    load_records,
)

# The defect that motivated this script is invisible to every existing screen, so it gets its own.
# A trace can commit, close, land the right answer and sit in the band while being the model's
# unedited deliberation; what marks it is first-person planning language, not its shape.
COT = re.compile(r"\b(okay,? let'?s|let'?s tackle|but wait|so,? the question|step by step|"
                 r"hmm|wait,|let me think|i need to (?:determine|figure|check))", re.I)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="the training file to edit IN PLACE")
    ap.add_argument("--record_id", required=True, action="append",
                    help="repeatable; the record(s) to regenerate")
    ap.add_argument("--dataset", default="data/rl/m0_format/train.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--brevity", choices=sorted(BREVITY_OVERRIDES), default="sixbullet")
    ap.add_argument("--prose", action="store_true")
    ap.add_argument("--k_samples", type=int, default=24)
    ap.add_argument("--max_rounds", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--max_trace_tokens", type=int, default=256)
    ap.add_argument("--min_trace_tokens", type=int, default=32)
    ap.add_argument("--target_trace_tokens", type=int, default=150)
    ap.add_argument("--max_coda_chars", type=int, default=0)
    ap.add_argument("--strip_coda", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--max_sentences", type=int, default=8,
                    help="reject a candidate whose reasoning runs past this many sentences. The "
                         "six-bullet prompt yields six; well past that means the model ignored the "
                         "form, which is the defect being repaired.")
    args = ap.parse_args()

    path = Path(args.traces)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    by_rid = {r["record_id"]: r for r in rows}
    targets = [rid for rid in args.record_id]
    missing = [rid for rid in targets if rid not in by_rid]
    if missing:
        raise SystemExit(f"not in {path}: {missing}")

    recs = {r["record_id"]: r for r in load_records(args.dataset)}
    model, tokenizer = load_model(args.model)
    brevity = BREVITY_OVERRIDES[args.brevity]
    replaced: dict[str, dict] = {}

    for rid in targets:
        rec = recs[rid]
        old = by_rid[rid]
        print(f"\n[regen] {rid}  (currently {old['n_reasoning_tokens']} tokens)")
        prompt = f"{rec['prompt']}\n\n{brevity}"
        best = None
        for round_i in range(args.max_rounds):
            texts = generate(model, tokenizer, [prompt], args.k_samples, args.max_new_tokens,
                             args.temperature, args.top_p, args.top_k)[0]
            for text in texts:
                status, clean, n = evaluate_trace(
                    text, rec["labels"], rec.get("answer"), bool(rec.get("verifiable")),
                    tokenizer, args.max_trace_tokens, args.max_coda_chars, args.strip_coda,
                    args.min_trace_tokens)
                if status != "pass":
                    continue
                if args.prose:
                    clean, n = bullets_to_prose(clean, tokenizer)
                if is_format_echo(clean) or n < args.min_trace_tokens:
                    continue
                reasoning = clean.partition("</think>")[0].strip()
                if COT.search(reasoning):
                    continue                       # the defect itself
                sents = [s for s in re.split(r"(?<=[.!?])\s+", reasoning) if s.strip()]
                if len(sents) > args.max_sentences:
                    continue
                score = abs(n - args.target_trace_tokens)
                if best is None or score < best[0]:
                    best = (score, clean, n, len(sents))
            if best:
                print(f"[regen]   round {round_i + 1}: accepted at {best[2]} tokens, "
                      f"{best[3]} sentences")
                break
            print(f"[regen]   round {round_i + 1}: nothing clean yet")
        if not best:
            print(f"[regen] {rid}: NO clean candidate after {args.max_rounds} round(s). "
                  f"Left unchanged -- decide deliberately whether to keep or drop it.")
            continue
        replaced[rid] = {"completion": best[1], "n_reasoning_tokens": best[2]}

    if not replaced:
        print("\n[regen] nothing replaced; file untouched")
        return 1
    backup = path.with_name(f"{path.stem}_preregen.jsonl")
    if not backup.exists():
        backup.write_text(path.read_text())
        print(f"\n[regen] original preserved at {backup.name}")
    out = [dict(r, **replaced[r["record_id"]], curation="regenerated")
           if r["record_id"] in replaced else r for r in rows]
    path.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in out))
    print(f"[regen] rewrote {path}: {len(replaced)} trace(s) replaced, {len(out)} rows total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
