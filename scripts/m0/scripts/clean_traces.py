#!/usr/bin/env python3
"""Drop the residual bad traces from a warm-start SFT set, before train_warmstart sees it.

WHY THIS EXISTS SEPARATELY FROM THE BUILDER'S FILTERS. ``build_warmstart.evaluate_trace`` rejects on structure -- did the block close, did it commit, is it correct, is it in band, did it spill past the turn. A trace can satisfy every one of those and still be worthless as a training target, because nothing checks that the reasoning is REASONING. The 9B set built 2026-07-28 contained, at 281 traces:

    "The user wants me to compare the liquid capacity of a shot glass and a standard
     coffee mug. I need to output exactly six bullet points inside `
     </think>

     Answer: Y"

One `</think>`, a bare committed answer as the last line, correct label, 29 tokens of "reasoning" -- passes everything. What it teaches is: restate the question, then answer. That is the degenerate behaviour ``m0/rewards.py`` was written against (the 0.8B's documented failure, median response 9 characters, the literal string ``Answer: A``). It arises from the model quoting the brevity prompt's ``<think>`` mention and emitting the real closing token mid-sentence, which lands the close in exactly the place that makes the trace look well-formed.

WHAT IS DELIBERATELY *NOT* FILTERED. "Require a bullet line in the reasoning" was tried and rejected as too blunt: on the same run it would have dropped 22 of 469 passing generations, including sound prose reasoning that simply carried no ``- `` markers (the ``--prose`` step strips them anyway). Precision matters more than recall here -- a false positive costs a good trace, and the residual defect rate is already low (7 of 281, 2.5%, versus 7 of 224 = 3.1% in the shipped 4B set).

Writes a NEW file and leaves the input untouched, so the original set stays auditable.

    python m0/scripts/clean_traces.py --traces results/m0_warmstart_9b/sft_traces.jsonl
    # -> results/m0_warmstart_9b/sft_traces_clean.jsonl, then point --traces at it
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The screen now lives in the builder and runs DURING candidate selection -- see
# build_warmstart.FORMAT_ECHO. Imported rather than duplicated so this script cannot disagree with the
# thing that produced its input; it stays here as the standing audit check, and from v3 on it should
# report zero.
#
# The regex it replaces was anchored at character 0 and keyed on OPENING WORDS ("okay,", "first,",
# "let me"), which cost real traces: a correct, complete, in-band Burj Khalifa comparison was dropped
# for opening "Okay, let's tackle this problem." Restating the task is reasoning; restating the
# instructions is not, and the format machinery is what distinguishes them. Unanchoring also caught
# defects the old version missed entirely -- the 9B's habit of appending "I need to ensure exactly six
# bullet points are written inside the <think> tags" at the END of an otherwise sound trace.
from scripts.m0.data.build_warmstart import is_format_echo

# m0/rewards.py's band floor. Below this the response is terse rather than reasoned, and the reward
# ramps it down for the same reason it is unwanted as a demonstration.
BAND_FLOOR = 32


def classify(row: dict) -> str | None:
    """Return a drop reason, or None to keep."""
    completion = row["completion"]
    body = completion.strip()
    if "<|im_start|>" in completion or "<|im_end|>" in completion:
        return "template_contamination"
    if completion.count("</think>") != 1:
        return "think_tag_count"
    reasoning, _, after = completion.partition("</think>")
    lines = [ln for ln in after.strip().splitlines() if ln.strip()]
    if not lines or not re.match(r"^Answer:\s*\S+$", lines[-1].strip()):
        return "no_bare_answer_last"
    if len(lines) > 1:
        return "coda_after_think"
    if is_format_echo(completion):
        return "format_echo"
    if row.get("n_reasoning_tokens", 0) < BAND_FLOOR:
        return "below_band_floor"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default=None,
                    help="default: <traces stem>_clean.jsonl beside the input")
    ap.add_argument("--show", type=int, default=3, help="print this many dropped traces per reason")
    args = ap.parse_args()

    src = Path(args.traces)
    out = Path(args.out) if args.out else src.with_name(src.stem + "_clean.jsonl")
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]

    kept, dropped = [], []
    for row in rows:
        reason = classify(row)
        (dropped if reason else kept).append((reason, row))
    dropped = [(r, x) for r, x in dropped if r]
    kept = [x for _, x in kept]

    reasons = Counter(r for r, _ in dropped)
    print(f"[clean] {len(rows)} in -> {len(kept)} kept, {len(dropped)} dropped "
          f"({len(dropped) / max(len(rows), 1):.1%})")
    for reason, n in reasons.most_common():
        print(f"[clean]   {reason:24s} {n}")
        for _, row in [d for d in dropped if d[0] == reason][: args.show]:
            snippet = row["completion"].strip().replace("\n", " ")[:110]
            print(f"[clean]       tok={row.get('n_reasoning_tokens'):>4}  {snippet!r}")

    fam = Counter(r["family"] for r in kept)
    print(f"[clean] family mix of the kept set:")
    for f in sorted(fam):
        print(f"[clean]   {f:18s} {fam[f]:4d}  ({100 * fam[f] / max(len(kept), 1):.1f}%)")

    out.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in kept))
    print(f"[clean] wrote {out}  -- pass this to train_warmstart --traces")


if __name__ == "__main__":
    main()
