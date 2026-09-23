#!/usr/bin/env python3
"""Assert m0/answer_format.py still agrees with the live scoring parser.

m0/answer_format.py is a copy of the pure-regex helpers in compute_utilities/utils.py (see that
file's header for why a copy is necessary). A copy can drift, and drift here is silent and fatal:
M0's reward would pay out on a commit criterion the scoring pipeline does not use, so training
reward would climb while the measured commit rate stayed flat.

This script imports BOTH implementations and compares them on an adversarial fixture. It needs
litellm/dotenv (pulled in by compute_utilities.utils), so run it in the repo venv -- not on the
training host:

    python m0/scripts/check_answer_format_parity.py

Exit 0 = identical behavior. Exit 1 = they have diverged; reconcile before training anything.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_utilities import utils as live  # noqa: E402
from scripts.m0 import answer_format as copy  # noqa: E402


AB = ["A", "B"]
XY = ["X", "Y"]
NUM = ["1", "2"]
MLS = ["MORE", "LESS", "SAME"]

# (response, choices). Each case targets a documented behavior of the parser: what counts as a
# commitment, what is an instruction echo, what is a draft, and what markdown noise is tolerated.
CASES: list[tuple[str, list[str]]] = [
    # --- clean commitments ---
    ("The first is heavier.\n\nAnswer: A", AB),
    ("Reasoning here.\nAnswer: B", AB),
    ("Answer: A", AB),
    ("Answer: a", AB),
    ("answer: b", AB),
    ("Some thought.\nAnswer: Y", XY),
    ("Some thought.\nAnswer: 2", NUM),
    ("Forecast.\nAnswer: SAME", MLS),
    ("Forecast.\nAnswer: more", MLS),
    ("Forecast.\nAnswer: LESS", MLS),
    # --- markdown / quote noise around a real commitment ---
    ("Thinking.\n**Answer: B**", AB),
    ("Thinking.\n> Answer: A", AB),
    ('Thinking.\n"Answer: A"', AB),
    ("Thinking.\n- Answer: B", AB),
    ("Thinking.\n#Answer: A", AB),
    ("Thinking.\n`Answer: B`", AB),
    ("Thinking.\n__Answer: A__", AB),
    ("Thinking.\nAnswer: **B**", AB),
    ("Thinking.\nAnswer: A.", AB),
    ("Thinking.\nAnswer: B!", AB),
    ("Thinking.\nAnswer: A)", AB),
    ("Thinking.\n   Answer: B   ", AB),
    ("Thinking.\nAnswer:   A", AB),
    ("Thinking.\nAnswer: MORE.", MLS),
    # --- trailing whitespace / blank lines after the commitment ---
    ("Thinking.\nAnswer: A\n", AB),
    ("Thinking.\nAnswer: A\n\n\n", AB),
    ("Thinking.\nAnswer: A\n   \n", AB),
    # --- instruction echoes and drafts: NOT commitments ---
    ('I should reply exactly as "Answer: A" or "Answer: B".', AB),
    ('"Answer: A" on its own line.', AB),
    ("Answer: A (or B).", AB),
    ("Answer: A, but let me reconsider.", AB),
    ("Answer: B seems right, though.", AB),
    ("Answer: A\nWait, actually let me reconsider the mass.", AB),
    ("Answer: A\nHmm, on reflection the other one is heavier.", AB),
    ("The answer is A.", AB),
    ("I pick A.", AB),
    ("Answer: C", AB),
    ("Answer:", AB),
    ("Answer: AB", AB),
    ("Answer: Answer", AB),
    # --- revisions: last line is what counts ---
    ("Answer: A\nActually no.\nAnswer: B", AB),
    ("Answer: B\n\nAnswer: A", AB),
    ("Answer: SAME\nOn reflection...\nAnswer: MORE", MLS),
    # --- multi-char label prefix collisions ---
    ("Forecast.\nAnswer: SAMENESS", MLS),
    ("Forecast.\nAnswer: MOREOVER", MLS),
    ("Forecast.\nAnswer: SAME-ISH", MLS),
    # --- degenerate inputs ---
    ("", AB),
    ("   ", AB),
    ("\n\n", AB),
    ("No commitment anywhere in this response at all.", AB),
    # --- truncated mid-reasoning (the failure M0 exists to fix) ---
    ("Let me think about which is heavier. The bowling ball is roughly seven kilograms while", AB),
    ('The instruction says to end with "Answer: A" or "Answer: B" so I will weigh both options and', AB),
]

FUNCS = [
    ("terminal_answer_label", lambda mod, r, c: mod.terminal_answer_label(r, c)),
    ("answer_line_labels", lambda mod, r, c: mod.answer_line_labels(r, c)),
    ("has_multiple_answers", lambda mod, r, c: mod.has_multiple_answers(r, c)),
]


def main() -> int:
    mismatches = []
    for response, choices in CASES:
        for name, call in FUNCS:
            want = call(live, response, choices)
            got = call(copy, response, choices)
            if want != got:
                mismatches.append((name, response, choices, want, got))

    checked = len(CASES) * len(FUNCS)
    if mismatches:
        print(f"PARITY FAILED: {len(mismatches)} of {checked} checks diverged\n")
        for name, response, choices, want, got in mismatches:
            print(f"  {name}(choices={choices})")
            print(f"    response : {response!r}")
            print(f"    utils.py : {want!r}")
            print(f"    m0       : {got!r}\n")
        print("m0/answer_format.py must be reconciled with compute_utilities/utils.py before training.")
        return 1

    print(f"parity OK: {checked} checks across {len(CASES)} responses, "
          f"{len(FUNCS)} functions -- m0/answer_format.py matches compute_utilities/utils.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
