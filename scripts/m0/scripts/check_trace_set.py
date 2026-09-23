#!/usr/bin/env python3
"""Gate an M0 warm-start trace set before training on it. Exits non-zero if anything fails.

WHY THIS EXISTS. `balance_traces.py` checks the kept-answer skew and `clean_traces.py` re-runs the meta-preamble screen, but on 2026-09-06 a 27B set passed both while covering only **139 distinct questions out of 200 rows** -- 46 questions asked more than once, two of them answered in OPPOSITE directions across presentations. Nothing in the pipeline noticed, because the defect is not in any single row: every row was well-formed, in-band, correctly answered and correctly balanced. It was the SET that was wrong. The cause was building with `build_warmstart.py` (whose quota walk never looks at what a question SAYS) instead of `build_sft_warmstart.py`, and the only reason it surfaced at all is that somebody read the file.

So this script asks the questions a per-row filter structurally cannot:

  - Is every training row a DIFFERENT question? (the 2026-09-06 defect)
  - Where a question does repeat, do the rows AGREE? (two contradictory targets on a byte-identical
    prompt is noise the trainer averages -- and on neutral questions nothing forces agreement)
  - Does every verifiable row match the answer key?
  - Is the completion shaped like a completion -- one `</think>`, nothing between it and a bare
    terminal answer, no chat-template spill, no markdown, no code, no leftover list markers?

Each structural check corresponds to a defect this project has already shipped once: the hallucinated `<|im_start|>` turns and 364-character codas that made the first 9B M0 unusable, the "1. **Heading**:" markers the prose rewrite exists to strip, the `Answer:` drafts scattered mid-reasoning.

Comparing answers across presentations is done BY CONTENT, not by label. A binary question rendered as A/B and again as 1/2 with the options swapped will use different letters for the same option, so the naive comparison reports contradictions that are not there (it did, on the first pass of the 2026-09-06 audit). Binary rows resolve to the TEXT of the chosen option block; ternary rows resolve to the semantic role -- MORE/HIGHER are both "up" -- which is rotation-independent by construction.

WHAT IT DOES NOT DO. It cannot tell you whether the reasoning is TRUE. The 2026-09-06 read found five traces that reach the right answer through a false claim -- "cast iron is denser than steel", a candle flame "around 1,000 degrees Fahrenheit", a 25-metre pool holding "millions of litres" -- and no screen catches those, because every one of them passes every check in this file. Read the traces. This script only frees the read to look for content instead of format.

    python m0/scripts/check_trace_set.py data/rl/m0_sft_27b_v4/sft_traces.jsonl
    python m0/scripts/check_trace_set.py <traces> --dataset data/rl/m0_format/train.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m0.answer_format import terminal_answer_label  # noqa: E402

# Ternary label roles. The battery rotates the presented order and varies the words, so the only
# presentation-independent identity of an answer is which ROLE it plays.
UP = {"MORE", "HIGHER"}
DOWN = {"LESS", "LOWER"}
FLAT = {"SAME", "UNCHANGED"}

LIST_MARKER = re.compile(r"^\s*(?:[-*•]\s+|\d{1,2}[.)]\s+)")
OPTION_BLOCK = "^Option {label}:\\s*\\n(.+?)(?=\\n\\s*\\n|\\Z)"


def suspicious_char(ch: str) -> bool:
    """True for a character that signals a broken decode rather than ordinary English prose.

    Screening on "non-ASCII" was the obvious first cut and it is WRONG: it fails the shipped 4B and
    9B v4 sets, whose traces legitimately contain `°C`, `g/cm³`, `Nicéphore Niépce` and typographic
    apostrophes. What actually indicates trouble is a replacement character, an invisible control or
    formatting character, or a glyph from a script the prompts are not written in -- each of which
    means the detokenizer or the provider's transport mangled something.
    """
    code = ord(ch)
    if ch == "�":                        # U+FFFD REPLACEMENT CHARACTER: a decode already failed
        return True
    if code < 0x20 and ch not in "\n\t":      # control characters
        return True
    if 0x200B <= code <= 0x200F or 0x2028 <= code <= 0x202E:   # zero-width and bidi formatting
        return True
    # Latin-1 supplement, Latin Extended, punctuation, superscripts, currency and letterlike symbols
    # all stay. Beyond U+2FFF is CJK, emoji and the like -- not English prose about libraries.
    return code > 0x2FFF


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path}: no records")
    return rows


def committed(row: dict) -> str | None:
    """The label on the final line, via the SAME parser the scorer uses."""
    return terminal_answer_label(row["completion"], list(row["labels"]))


def reasoning_of(row: dict) -> str:
    return row["completion"].partition("</think>")[0]


def after_think(row: dict) -> str:
    return row["completion"].partition("</think>")[2]


def semantic_answer(row: dict, prompt: str) -> str | None:
    """What the row actually chose, independent of how it was presented.

    Ternary -> the role (up/down/flat). Binary -> the text of the chosen option block, so two
    renderings that swap the options still compare equal when they pick the same thing.
    """
    label = committed(row)
    if label is None:
        return None
    if len(row["labels"]) == 3:
        return "UP" if label in UP else "DOWN" if label in DOWN else "FLAT" if label in FLAT else None
    match = re.search(OPTION_BLOCK.format(label=re.escape(label)), prompt, re.M | re.S)
    return " ".join(match.group(1).split()).lower() if match else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", help="sft_traces.jsonl to check")
    ap.add_argument("--dataset", default="data/rl/m0_format/train.jsonl",
                    help="source prompts, used to recover content_id and the option text")
    ap.add_argument("--expect_rows", type=int, default=None,
                    help="fail unless the training set has exactly this many rows")
    args = ap.parse_args()

    traces_path = Path(args.traces)
    rows = load(traces_path)
    source = {json.loads(line)["record_id"]: json.loads(line)
              for line in Path(args.dataset).read_text().splitlines() if line.strip()}

    # Only sibling 0 is training data. build_sft_warmstart writes backups to a separate file, but a
    # top-up or a hand-edit can put them here, and training on both trains both sides of a
    # contradiction on a byte-identical prompt.
    training = [r for r in rows if r.get("sibling", 0) == 0]
    spares = len(rows) - len(training)

    failures: list[str] = []
    warnings: list[str] = []

    print(f"[check] {traces_path}: {len(rows)} rows ({len(training)} training, {spares} spare)")

    # -- 1. distinct questions -------------------------------------------------------------------
    missing_src = [r["record_id"] for r in training if r["record_id"] not in source]
    if missing_src:
        failures.append(f"{len(missing_src)} row(s) not found in {args.dataset}: {missing_src[:5]}")
    cids = collections.Counter(source[r["record_id"]]["content_id"]
                               for r in training if r["record_id"] in source)
    repeats = {c: n for c, n in cids.items() if n > 1}
    print(f"[check] distinct questions: {len(cids)} over {len(training)} training rows")
    if repeats:
        failures.append(
            f"{len(training) - len(cids)} repeated question(s): {len(repeats)} question(s) appear "
            f"more than once, worst {max(repeats.values())}x -- "
            f"{sorted(repeats.items(), key=lambda kv: -kv[1])[:5]}. A 200-row budget cannot afford "
            f"to ask the same thing twice; rebuild with m0/data/build_sft_warmstart.py."
        )
    if args.expect_rows is not None and len(training) != args.expect_rows:
        failures.append(f"expected {args.expect_rows} training rows, found {len(training)}")

    # -- 2. agreement across presentations -------------------------------------------------------
    by_question: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:                      # spares included: a spare is a swap candidate, so it must agree
        if r["record_id"] in source:
            by_question[source[r["record_id"]]["content_id"]].append(r)
    contradictions = []
    for cid, group in by_question.items():
        if len(group) < 2:
            continue
        answers = collections.defaultdict(list)
        for r in group:
            answers[semantic_answer(r, source[r["record_id"]]["prompt"])].append(r["record_id"])
        if len(answers) > 1:
            contradictions.append((cid, dict(answers)))
    if contradictions:
        failures.append(
            f"{len(contradictions)} question(s) answered INCONSISTENTLY across renderings -- "
            f"two contradictory targets for one question is noise the trainer averages: "
            + "; ".join(f"{c} {list(a.values())}" for c, a in contradictions[:4])
        )

    # -- 3. the answer key -----------------------------------------------------------------------
    verifiable = [r for r in rows if r.get("verifiable") and r.get("answer") is not None]
    wrong = [r["record_id"] for r in verifiable if committed(r) != r["answer"]]
    print(f"[check] verifiable rows: {len(verifiable)}, disagreeing with the key: {len(wrong)}")
    if wrong:
        failures.append(f"{len(wrong)} verifiable row(s) commit the wrong label: {wrong[:5]}")

    # -- 4. shape --------------------------------------------------------------------------------
    def bad(name, predicate):
        hits = [r["record_id"] for r in rows if predicate(r)]
        if hits:
            failures.append(f"{len(hits)} row(s) {name}: {hits[:4]}")
        return hits

    bad("carry chat-template tokens (the 9B's hallucinated turns)",
        lambda r: "<|im_start|>" in r["completion"] or "<|im_end|>" in r["completion"])
    bad("do not close </think> exactly once",
        lambda r: r["completion"].count("</think>") != 1)
    bad("carry an opening <think> (the template already opened it)",
        lambda r: "<think>" in r["completion"])
    bad("have a coda between </think> and the answer",
        lambda r: len([l for l in after_think(r).strip().splitlines() if l.strip()]) > 1)
    bad("do not end on a parseable bare answer line", lambda r: committed(r) is None)
    bad("leave list markers in the reasoning",
        lambda r: any(LIST_MARKER.match(l) for l in reasoning_of(r).splitlines() if l.strip()))
    bad("carry markdown emphasis", lambda r: "**" in r["completion"] or "__" in r["completion"])
    bad("carry code fences or backticks", lambda r: "```" in r["completion"] or "`" in r["completion"])
    bad("draft an Answer: line inside the reasoning",
        lambda r: re.search(r"(?i)\banswer\s*:", reasoning_of(r)) is not None)
    bad("have empty or near-empty reasoning", lambda r: len(reasoning_of(r).strip()) < 20)
    bad("carry characters that indicate a broken decode",
        lambda r: any(suspicious_char(c) for c in r["completion"]))

    dupes = [c for c, n in collections.Counter(r["completion"] for r in rows).items() if n > 1]
    if dupes:
        failures.append(f"{len(dupes)} completion(s) appear more than once verbatim")

    # -- 5. warnings: judgment calls, not gates --------------------------------------------------
    # Reasoning that ends by declining to decide, then commits anyway. Not malformed -- the format
    # is exactly right -- but as an SFT target it demonstrates "deliberate, refuse to conclude,
    # emit a label", which is not the behaviour M0 is meant to install. Read these before training.
    HEDGE = re.compile(r"(?i)(depends (largely |entirely )?on|ultimately dictates|is a matter of|"
                       r"personal preference|comes down to|either option)")
    undecided = []
    for r in rows:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reasoning_of(r).strip()) if s.strip()]
        if sentences and HEDGE.search(sentences[-1]):
            undecided.append(r["record_id"])
    if undecided:
        warnings.append(f"{len(undecided)} row(s) end the reasoning undecided, then commit: {undecided[:5]}")

    # -- verdict ---------------------------------------------------------------------------------
    for w in warnings:
        print(f"[check] WARNING: {w}")
    if failures:
        print(f"\n[check] FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        print("\n[check] Do not train on this set.")
        return 1
    print("[check] all checks passed.")
    print("[check] NOTE: this proves the set is well FORMED, not that the reasoning is TRUE. "
          "Read the traces -- a false claim that reaches the right answer passes every check here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
