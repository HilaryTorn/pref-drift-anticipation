"""The 'Answer: <label>' parser, copied verbatim so it can run on the training box.

SOURCE OF TRUTH: compute_utilities/utils.py (answer_line_pattern, answer_line_labels,
_answer_only_line, terminal_answer_label, has_multiple_answers). This file is a copy, NOT a
reimplementation -- the regexes and the docstrings below are lifted unchanged.

Why a copy exists at all: the training host installs m0/requirements-m0.txt, which has no litellm
and no python-dotenv, while compute_utilities/utils.py imports
compute_utilities.llm_agent (and through it litellm, dotenv, and vllm) at module scope. Importing
the real parser inside a GRPO reward function would therefore fail on the GPU box. Editing
utils.py to split the pure helpers out is off-limits here: that file is on the live values and
coding scoring path.

The cost of a copy is drift, and drift here would be silent and fatal -- M0 would optimize a
commit criterion that the scoring pipeline does not use, so the reward curve would rise while the
measured commit rate did not move. m0/scripts/check_answer_format_parity.py exists to make that
impossible to miss: it imports both implementations and asserts they agree on an adversarial
fixture. Run it in the repo venv (where litellm is installed) after touching either file.
"""

from __future__ import annotations

import re


def answer_line_pattern(choices):
    """Compile the reasoning-mode 'Answer: <choice>' matcher for a label set.

    The prompt asks the model to put its final answer on its OWN LINE. Anchoring the match to a line
    start is what makes that instruction load-bearing: a bare `Answer:\\s*(X|Y)` matched anywhere also
    fires on the model echoing the instruction back mid-sentence (`...exactly as "Answer: A" or
    "Answer: B"`) and on mid-reasoning asides. We tolerate leading markdown/quote noise so `**Answer:
    B**` and `> Answer: B` still match.
    """
    pattern_str = '|'.join(re.escape(c) for c in choices)
    return re.compile(
        rf'^[ \t>*_"\'`#-]*Answer:\s*\*{{0,2}}\s*({pattern_str})(?=$|[^\w])',
        re.IGNORECASE | re.MULTILINE,
    )


def answer_line_labels(response, choices):
    """Canonical choices named on 'Answer:' lines, in the order they appear (case-normalized)."""
    if not response:
        return []
    norm = {c.upper(): c for c in choices}
    return [norm[m.upper()] for m in answer_line_pattern(choices).findall(response) if m.upper() in norm]


def _answer_only_line(line, choices):
    """If a single line's ONLY content is an 'Answer: <label>' commitment, return the canonical label.

    "Only content" means the label may be wrapped in leading markdown/quote noise (`**Answer: A**`,
    `> Answer: A`, `"Answer: A"`) and followed by nothing but trailing punctuation/quotes/whitespace.
    A line like `"Answer: A" on its own line.` or `Answer: A (or B).` is NOT answer-only -- it carries
    narration past the label -- so it returns None. Returns None if the line is not a bare commitment.
    """
    pattern_str = '|'.join(re.escape(c) for c in choices)
    m = re.match(
        rf'^[ \t>*_"\'`#-]*Answer:\s*\*{{0,2}}\s*({pattern_str})[ \t.,;:!?*_"\'`)\]]*$',
        line,
        re.IGNORECASE,
    )
    if not m:
        return None
    norm = {c.upper(): c for c in choices}
    return norm.get(m.group(1).upper())


def terminal_answer_label(response, choices):
    """The committed choice = an answer-only line that is the LAST non-empty line of the response.

    The prompt requires the final answer on its own line, so a genuine commitment is the last thing
    generated. Reasoning-on generations that run past max_tokens litter the scratchpad with
    line-anchored 'Answer: X' lines that are format-instruction echoes (`"Answer: A" or "Answer: B"`),
    quoted drafts (`"Answer: A" on its own line.`), or tentative picks the model then revises
    ("...Wait, reconsider"). Take-last over ALL answer lines scores those non-commitments as real votes
    (~20% of samples on the reasoning-on batteries). Requiring the answer to be the last non-empty line
    drops them as unparseable instead. Returns the canonical label, or None if the response did not end
    on a bare commitment.
    """
    if not response:
        return None
    nonempty = [ln for ln in response.splitlines() if ln.strip()]
    if not nonempty:
        return None
    return _answer_only_line(nonempty[-1], choices)


def has_multiple_answers(response, choices):
    """True if the response's 'Answer:' lines name more than one distinct choice.

    Take-last picks the final answer line as the committed choice (the model revises `A -> B` and
    means B), but a genuine contradiction (`A` and `B` both stated then truncated) is indistinguishable
    from a revision in text. This flag records that ambiguity per sample so it can be filtered or
    sensitivity-tested downstream instead of silently trusting take-last. Repeated agreeing lines (the
    common case: the model just restates the same answer) are NOT flagged.
    """
    return len(set(answer_line_labels(response, choices))) > 1
