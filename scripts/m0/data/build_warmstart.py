#!/usr/bin/env python3
"""Generate the M0 warm-start SFT set by self-distillation (Phase 2 of docs/m0-warmstart-plan.md).

The problem this solves: under M0's own prompt the untrained 4B reasons ~730 tokens median, entirely above the 256/500 reward band, so every early GRPO rollout scores band 0 and GRPO -- whose gradient is within-group variance -- has nothing to pull on. It cannot move the policy into a band its samples never reach. So the model is SEEDED first: this script harvests the 4B's OWN short, correct, committed reasoning on the same neutral fact prompts, which a later LoRA SFT trains back into it so its default length drops into range. The GRPO band then sharpens what the warm-start put there.

Two design points that make this correct rather than merely plausible:

- SELF-distill, not a teacher. The traces are the same 4B's own reasoning, only shorter, so the step imports no foreign content into a run whose whole purpose is to change nothing but format. Point ``--model`` at the exact base the GRPO run starts from (Qwen/Qwen3.5-4B).

- The BREVITY OVERRIDE is appended only to the GENERATION prompt; the SFT input stored on disk is the ORIGINAL, unmodified M0 prompt. So the model learns "reason briefly UNDER THE NORMAL PROMPT", which is the behaviour that transfers to serve time -- not "obey a brevity instruction". The default `twobullet` is the exact strict form that produced ~60-token reasoning on the 21-07 run; it DOES re-declare `<think>`, which is what made it unstable for SERVING, but that is harmless here because we oversample and discard any malformed generation. A softer form that avoided re-declaring the tag was tried first and did NOT bite -- 0% yield, the 4B reasoned past the cap and never closed the block.

Every kept trace must (a) commit -- bare `Answer:` as the last non-empty line, per the SAME parser the scorer uses -- (b) be correct on verifiable items, and (c) reason within ``--max_trace_tokens`` (256 by default: the band's full-marks ceiling). One trace per prompt (the shortest passing sample), so the SFT set inherits M0's family balance instead of over-weighting the prompts that answer easily.

Output is ``m0_warmstart_sft_v1`` JSONL (prompt + short completion) plus a manifest with the yield and length distribution. INFERENCE, so per docs/cost.md's Inference Rule it must not run on a paid AWS training box unless that is a deliberate, flagged exception; it is bounded (~30-60 min) and one-time. Run a small ``--limit`` first to pick the brevity variant before generating the full set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m0.answer_format import terminal_answer_label
from scripts.m0.prompts import render_format_prompt
from scripts.m0.rewards import _count_tokens, split_completion

# A run of list markers at the head of a reasoning line: "- ", "* ", "• ", "1. ", "2) ", and the hybrid "1. - " the 4B emits when it numbers the bullets it was asked to dash. Matching only "- " (the original) meant a numbered response registered as HAVING NO BULLETS AT ALL, which sent bullets_to_prose down its every-line fallback and carried both the numerals and any meta-preamble straight into the SFT target. Measured across the shipped sets: 13/224 of the 4B v2 targets and 13/200 of the 4B v3 ones open with a literal "1." the model then reproduces at serve time.
# The digit run is capped at two so a line opening on a year ("1903. The Wright brothers...") is not mistaken for item 1903 of a list.
# The bullet characters REQUIRE trailing whitespace. With `[-*•]\s*` (zero-or-more) the `*` branch also matched markdown bold: on "1.  **Accessibility**: ..." the run consumed "1.  **" and left the CLOSING "**" stranded, so the SFT target read "Accessibility**: Placing herbs...". Measured 1/200 on the 2026-08-03 4B set -- but the 9B emits this "1. **Heading**: ..." shape at a documented 54%, so on that size it is the common case, not the corner one. A real bullet always has a space after its marker; "**" never does.
BULLET_MARKER = re.compile(r"^\s*(?:[-*•]\s+|\d{1,2}[.)]\s+)+")

# Markdown emphasis left INSIDE a bullet, once its marker is gone. Stripping the marker alone is not
# enough -- the target is meant to be plain prose, and "**Accessibility**: Placing herbs..." teaches
# the model to emit markdown headings mid-reasoning. Removes the delimiters, keeps the words.
EMPHASIS = re.compile(r"\*\*|__")

# Reasoning that is ABOUT the format instead of about the question. This is the screen clean_traces.py applies post hoc, moved here so it runs while the alternatives are still on the table -- see the candidate loop in main().
# It matches the FORMAT MACHINERY (the think tags, the bullet count, the answer template), not the opening words. An earlier version anchored on conversational openers ("okay,", "first,", "let me") and threw away sound reasoning that merely started chattily -- e.g. a correct, complete 233-token Burj Khalifa comparison whose only sin was opening "Okay, let's tackle this problem." Restating the TASK is reasoning; restating the INSTRUCTIONS is not. Unanchored on purpose: the 9B's version of this defect often arrives mid-trace ("...I need to ensure exactly six bullet points are written inside the <think> tags.").
FORMAT_ECHO = re.compile(
    r"(?i)(<think>|</think>|think(?:ing)? tags?|bullet points?|\bAnswer:\s*[\"'`]|"
    r"constraint \d|thinking process:|output the (?:thinking|final answer)|inside the [`\"]|"
    # Third-person narration of the request. Not format machinery, but the same defect: the trace is
    # ABOUT the prompt rather than working through it, which is the "restate the question, then
    # answer" shape m0/rewards.py exists to prevent. Kept distinct from a first-person "I need to
    # compare X and Y", which is a reasoner stating the task it is about to do -- that is reasoning.
    r"\bthe user (?:wants|is asking|asks|needs)\b|here (?:are|is) the (?:thought|thinking|bullet))"
)


def is_format_echo(completion: str) -> bool:
    """True when the REASONING (everything before </think>) talks about the output format."""
    return bool(FORMAT_ECHO.search(completion.partition("</think>")[0]))

# Appended to the generation prompt only. Keep the model inside its native <think> block (no tag re-declaration) but push it to the ~60-token regime the two-bullet form produced on 2026-07-21.
BREVITY_OVERRIDES = {
    "twobullet": (
        "Think inside <think> tags.\n"
        "Inside <think>, write EXACTLY two bullet points.\n"
        'Each bullet point must start with "- ".\n'
        "Each bullet point must be one sentence.\n"
        "Do not write any other text inside <think>.\n"
        "After </think>, write your final answer on its own line, exactly as instructed above."
    ),
    "threebullet": (
        "Think inside <think> tags.\n"
        "Inside <think>, write EXACTLY three bullet points.\n"
        'Each bullet point must start with "- ".\n'
        "Each bullet point must be one sentence.\n"
        "Do not write any other text inside <think>.\n"
        "After </think>, write your final answer on its own line, exactly as instructed above."
    ),
    # THESE PROMPTS ARE UNCHANGED FROM THE 4B v2 RECIPE, and two attempts to improve them for the 9B
    # were tried on 2026-07-28 and BOTH REVERTED. Recorded so nobody re-runs them:
    #
    #   1. Appending "and do not restate these instructions" -> yield 50% -> 42%, ternary_factual to 0%.
    #      Self-defeating: the dumps show the model enumerating "Constraint 6: ...do not restate these
    #      instructions" while restating them. Naming the thing invokes it.
    #   2. Removing every <think>/</think> mention -> catastrophic. EVERY sampled generation ran to
    #      --max_new_tokens without closing the block (no_close, n_reasoning=400). This reproduces the
    #      0%-yield result the module docstring records for the same experiment on the 4B in July.
    #      The tag re-declaration is LOAD-BEARING for getting the block closed; it invites the echo, and
    #      that trade is the better side. Do not "fix" it again.
    #
    # The 9B does echo the tags: it restates the instruction ("Constraint 3 (Thought Tags): Write inside
    # `<think>`") and emits the real </think> token while doing so, closing its own reasoning block early
    # and stranding the reasoning outside it (status meta_echo, ~28% of generations, plus ~39% no_close).
    # That cost is paid with SAMPLING -- oversample and let the filters discard -- not with prompt
    # surgery. What genuinely helped is elsewhere in this file: the explicit eos_token_id took
    # template_leak 31% -> 0%. Qwen's recommended top_p/top_k (0.95/20) was neutral, kept for conformance.
    #
    # fourbullet / sixbullet are the WIDER variants for arms whose items need more deliberation to
    # separate. The value portraits collapse to position bias at ~60-token reasoning (two bullets is not
    # enough to weigh two subtle personas, so the model defaults to whichever is labelled first). Measured
    # ladder: 2 bullets ~57 tok, 4 ~100, 6 ~145 (~21 tok/bullet). Pick by how much weighing the arm needs;
    # do NOT climb toward the band's 256 ceiling -- it is an upper bound, not a target, and beyond ~6 the
    # exact-count constraint invites padding over genuine reasoning and yield drops. Keep the BULLET
    # generator (the sentence variants underperform, ~40% vs ~90% yield) and render to sentences with
    # --prose; never ask for sentences directly.
    "fourbullet": (
        "Think inside <think> tags.\n"
        "Inside <think>, write EXACTLY four bullet points.\n"
        'Each bullet point must start with "- ".\n'
        "Each bullet point must be one sentence.\n"
        "Do not write any other text inside <think>.\n"
        "After </think>, write your final answer on its own line, exactly as instructed above."
    ),
    "sixbullet": (
        "Think inside <think> tags.\n"
        "Inside <think>, write EXACTLY six bullet points.\n"
        'Each bullet point must start with "- ".\n'
        "Each bullet point must be one sentence.\n"
        "Do not write any other text inside <think>.\n"
        "After </think>, write your final answer on its own line, exactly as instructed above."
    ),
    # "EXACTLY N sentences of plain reasoning" is the mechanical constraint that makes the bullet
    # form clean; the extra two lines forbid the failure modes twosentence's dumps showed -- the
    # model meta-analysing the instructions ("Thinking Process: 1. Analyze the Request...") and
    # emitting numbered/bulleted lists instead of prose.
    "twosentence": (
        "Think inside <think> tags.\n"
        "Inside <think>, write EXACTLY two short sentences of plain reasoning.\n"
        "Do not use bullet points, numbered lists, or headings, and do not restate these instructions.\n"
        "After </think>, write your final answer on its own line, exactly as instructed above."
    ),
    "threesentence": (
        "Think inside <think> tags.\n"
        "Inside <think>, write EXACTLY three short sentences of plain reasoning.\n"
        "Do not use bullet points, numbered lists, or headings, and do not restate these instructions.\n"
        "After </think>, write your final answer on its own line, exactly as instructed above."
    ),
    # onesentence dropped: ~20 tokens falls BELOW the reward band's floor (32), so it is penalised as
    # too terse and risks too-thin reasoning to get facts right. Two is the minimum that clears the floor.
}


# ---------------------------------------------------------------------------------------------
# REVERTED 2026-07-28 to exactly the above (the 4B v2 recipe). Two prompt edits were tried on the
# 9B and BOTH made it worse, so the tag-naming form stands despite the 9B echoing it:
#   1. Appending "and do not restate these instructions" -> yield 50% -> 42%, ternary_factual to 0%.
#      The clause is self-defeating: dumps show the model enumerating "Constraint 6: ...do not
#      restate these instructions" while restating them. Naming the thing invokes it.
#   2. Removing every <think>/</think> mention -> catastrophic. Every sampled generation ran to
#      --max_new_tokens without ever closing the block (no_close, n_reasoning=400). This reproduces
#      the 0%-yield result the module docstring records for the same experiment on the 4B in July;
#      the tag re-declaration is load-bearing for getting the block CLOSED, which is why it survives
#      despite inviting the echo.
# What DID work and is kept elsewhere in this file: the explicit eos_token_id (template_leak
# 31% -> 0%) and the integrity filters. Qwen's recommended top_p/top_k (0.95/20) were neutral and
# are kept for vendor conformance. Residual meta_echo ~28% + no_close ~40% on the 9B are therefore
# paid for with SAMPLING, not prompt surgery: oversample and let the filters discard.
# ---------------------------------------------------------------------------------------------

def _bf16_available() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_records(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path}: no records")
    return rows


def load_model(base_model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Mirrors m0/train_m0.py's base load so the traces come off the exact stack GRPO trains on.
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Device/dtype: the AWS training box is CUDA+bf16 (this branch is byte-identical to before). Running
    # the self-distill locally off the paid box (the Inference Rule) uses Apple MPS in fp16; CPU/fp32 is
    # the last resort. Only the non-CUDA branches are new, so production trace-gen on AWS is unaffected.
    if torch.cuda.is_available():
        device, dtype = "cuda", (torch.bfloat16 if _bf16_available() else torch.float32)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


_STOP_IDS_LOGGED = False


def _resolve_stop_ids(tokenizer):
    """Token ids that end an assistant turn, as a list. Empty when nothing resolves.

    Reports what it found exactly once. The stop tokens are the difference between a clean trace set and one full of hallucinated user turns, and a silent failure here is invisible until a model trained on the result misbehaves weeks later -- so it is stated in the log rather than assumed.
    """
    global _STOP_IDS_LOGGED
    candidates = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    unk = getattr(tokenizer, "unk_token_id", None)
    ids = sorted({i for i in candidates if isinstance(i, int) and i >= 0 and i != unk})
    if not _STOP_IDS_LOGGED:
        _STOP_IDS_LOGGED = True
        if ids:
            print(f"[warmstart] stop tokens resolved: {ids} (eos={tokenizer.eos_token!r})")
        else:
            print("[warmstart] WARNING: no stop token resolved -- generation will run to --max_new_tokens "
                  "and is likely to spill past the assistant turn. Expect a high template_leak count; "
                  "do not trust the kept set until this is fixed.")
    return ids


def generate(model, tokenizer, prompt_texts, k, max_new_tokens, temperature, top_p=0.95, top_k=20):
    """Return a list of k decoded completions for each prompt, preserving prompt order.

    The generation core is copied from m0/callbacks.py::FormatCadenceCallback._generate -- the same left-padding (decoder-only models generate garbage with right padding), the same skip_special_tokens=False (</think> is special in the Qwen3.5 template; stripping it would make every completion look tagless), and the same eos/pad cleanup.
    """
    import torch

    rendered = [render_format_prompt(tokenizer, p) for p in prompt_texts]
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
        # Stop at the assistant turn boundary. Without an explicit eos_token_id, generate() falls back to
        # the model's generation_config, which on Qwen3.5-9B does NOT halt at <|im_end|>: generation runs
        # to max_new_tokens and invents a following user turn. Those hallucinated turns were then baked
        # into the SFT targets -- 32% of the first 9B trace set carried <|im_start|> and 29% a second
        # </think> -- which taught the model to run past its own turn at serve time. The 4B never exposed
        # this because its generations stopped on their own, so the omission looked harmless for a year.
        stop_ids = _resolve_stop_ids(tokenizer)
        # NEVER pass eos_token_id=None. generate() merges kwargs over the model's generation_config, so an
        # explicit None OVERRIDES the model's own eos instead of falling back to it -- generation then has
        # no stopping criterion at all and every sequence runs to max_new_tokens. That is not theoretical:
        # it is what a first version of this fix did, and it made things worse than the bug it targeted
        # (closed generations 81% -> 59%, reasoning p90 246 -> pinned at the 400 cap). Omit the kwarg
        # entirely when nothing resolved, so the model's default still applies.
        stop_kwargs = {"eos_token_id": stop_ids} if stop_ids else {}
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                top_p=top_p, top_k=top_k,
                num_return_sequences=k,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                **stop_kwargs,
            )
    finally:
        tokenizer.padding_side = previous_side

    width = enc["input_ids"].shape[1]
    per_prompt: list[list[str]] = [[] for _ in prompt_texts]
    for index, seq in enumerate(out):
        text = tokenizer.decode(seq[width:], skip_special_tokens=False)
        for special in (tokenizer.eos_token, tokenizer.pad_token):
            if special:
                text = text.replace(special, "")
        per_prompt[index // k].append(text)  # generate() returns k consecutive rows per input
    return per_prompt


TEMPLATE_MARKERS = ("<|im_start|>", "<|im_end|>")


def evaluate_trace(text, labels, answer, verifiable, tokenizer, max_trace_tokens,
                   max_coda_chars=0, strip_coda=False, min_trace_tokens=0):
    """Classify one generation. Returns (status, clean_completion_or_None, n_reasoning_tokens).

    status is one of pass / no_close / no_commit / wrong / too_long / template_leak / coda, so the caller can tally WHY generations are rejected -- a 0% yield needs that breakdown to be diagnosable. n_reasoning is the reasoning-token count when the block closed, else the length of the whole (truncated) generation, so the length distribution is visible either way.

    THE COMMIT CHECK ALONE IS NOT ENOUGH, and the first 9B trace set is why. It only asks whether the LAST non-empty line is a bare answer; it says nothing about what sits between `</think>` and that line. A generation that closed its reasoning, wrote four sentences of restated argument, hallucinated an entire `<|im_start|>user` turn, answered itself, and *then* happened to end on an answer line passed every check and became a training target. 63% of that set carried such a coda (median 364 chars) and 32% carried chat-template tokens; the resulting model reproduced exactly that shape at serve time, dropping ~14% of battery samples as unparseable against the 4B's ~1.5%. The two guards below close that gap: template markers and a second `</think>` are always rejected as corrupt, and the coda is bounded (or stripped) so the target teaches `</think>` -> answer with nothing in between.
    """
    if any(marker in text for marker in TEMPLATE_MARKERS):
        # The model ran past its own turn. Never salvageable: whatever follows is a hallucinated
        # conversation, and the answer line may belong to a turn we never asked for.
        return "template_leak", None, _count_tokens(text, tokenizer)
    if "</think>" not in text:
        return "no_close", None, _count_tokens(text, tokenizer)   # ran out of room before closing
    if text.count("</think>") > 1:
        # A repeated closing tag WITHOUT any turn marker is a different animal from a turn spill, and
        # conflating them makes template_leak useless as a diagnostic. This is the model quoting the
        # brevity instruction back at itself ("Constraint 4: Inside `</think>`, write EXACTLY six
        # bullet points"), the meta-analysis failure the sentence variants' guard clause exists to
        # forbid. Rejected either way, but counted separately: template_leak means fix the stop
        # tokens, meta_echo means fix the prompt.
        return "meta_echo", None, _count_tokens(text, tokenizer)
    reasoning, visible = split_completion(text)
    n_reasoning = _count_tokens(reasoning, tokenizer)
    committed = terminal_answer_label(visible, list(labels))
    if committed is None:
        return "no_commit", None, n_reasoning                     # closed think but no bare Answer: last line
    if verifiable and answer is not None and committed != answer:
        return "wrong", None, n_reasoning                         # committed the wrong label
    if n_reasoning < min_trace_tokens:
        # The band has a FLOOR as well as a ceiling and only the ceiling was enforced here, so a
        # generation that closed early could be committed, correct, in-band-by-the-only-check-there-was
        # and still be 22 tokens of nothing -- which is precisely the degenerate shape m0/rewards.py
        # exists to prevent (the 0.8B's documented failure: median response 9 characters, the literal
        # string "Answer: A"). One reached the 4B v3 set that way.
        return "too_short", None, n_reasoning
    if n_reasoning > max_trace_tokens:
        return "too_long", None, n_reasoning                      # correct + committed but over the band
    close = text.find("</think>")
    head, after = text[:close], text[close + len("</think>"):]
    nonempty = [ln for ln in after.splitlines() if ln.strip()]
    answer_line = nonempty[-1].strip()
    coda = "\n".join(nonempty[:-1]).strip()
    if len(coda) > max_coda_chars:
        if not strip_coda:
            return "coda", None, n_reasoning
        # Salvage: keep the reasoning and the commitment, drop the restatement in between. The result
        # is byte-shaped like the 4B's targets, which are 97% `</think>` -> blank line -> answer.
        return "pass", f"{head}</think>\n\n{answer_line}", n_reasoning
    # committed is the last non-empty line, so rstrip() ends the target exactly on the answer line.
    return "pass", text.rstrip(), n_reasoning


def bullets_to_prose(completion, tokenizer):
    """Rewrite a two-bullet reasoning block as prose; return (new_completion, n_reasoning_tokens).

    The reliable generator is the twobullet prompt (~90% yield), but its output is literal "- ..." bullets. Since each bullet is one sentence, stripping the markers and joining the lines gives clean prose -- the reasoning style we want to install -- with twobullet's reliability instead of the sentence prompts' ~40%. A no-op on completions that carry no bullet markers.

    WHEN BULLETS ARE PRESENT, ONLY THE BULLETS SURVIVE. An earlier version joined every non-empty line, which quietly carried the model's meta-preamble into the training target: 54% of the first 9B trace set opened "Thinking Process: 1. **Analyze the Request:** * Scenario: ..." before its bullets, and the model SFT'd on them reproduced that style verbatim at serve time. Nothing else in the pipeline catches it -- the trace still commits, is still correct, still fits the token band, so every filter passes it. The 4B showed this at 3% and it went unnoticed; the 9B shows it at 54%. Dropping non-bullet lines is safe precisely because the prompt asked for bullets and nothing else, so anything that is not a bullet is by definition the failure mode.
    """
    close = completion.find("</think>")
    if close == -1:
        return completion, _count_tokens(completion, tokenizer)
    reasoning, rest = completion[:close], completion[close:]
    lines = [ln.strip() for ln in reasoning.splitlines() if ln.strip()]
    bullets = [ln for ln in lines if BULLET_MARKER.match(ln)]
    # Fall back to every line only when the model produced no bullets at all -- that is the sentence
    # variants' normal output, where there is no preamble to separate from the reasoning.
    source = bullets if bullets else lines
    sentences = [EMPHASIS.sub("", BULLET_MARKER.sub("", s)).strip() for s in source]
    prose = " ".join(s for s in sentences if s)
    return f"{prose}\n{rest}", _count_tokens(prose, tokenizer)


def _cell_key(rec: dict) -> str:
    """The format cell a prompt trains: family x presented label order.

    v3 made this the unit that matters. The dataset renders every ternary item in one of 6 presentations (2 word schemes x 3 rotations) and binary in one of 3 schemes, and the battery may ask any of them — so the SFT set needs kept traces in EVERY cell, not merely a good overall yield. A head-slice `--limit` plus per-cell yield variance (the 9B kept ~40% overall, and unevenly) can leave a cell with two traces or none, and a cell SFT never saw is a battery format the model was never format-trained on.
    """
    return f"{rec['family']}|{'/'.join(rec['labels'])}"


def _quota_key(rec: dict) -> str:
    """The unit a quota applies to: the format cell, plus the gold answer for verifiable rows.

    Quotas at cell level alone would leave answer balance to the survival filters, and those are measurably answer-skewed — off a balanced dataset, the v2 4B kept 6 A vs 13 B in its A/B cell and 9/14/18 across MORE/LESS/SAME. Keying the quota by answer makes the kept set answer-balanced by construction, mirroring both balances the dataset itself guarantees (family mix, uniform answers per cell). Neutral rows have no gold answer, so their quota stays at cell level.
    """
    key = _cell_key(rec)
    if rec.get("verifiable") and rec.get("answer"):
        return f"{key}|{rec['answer']}"
    return key


def _check_min_per_slot(quota_targets: dict[str, int], min_per_slot: int, budget: int,
                        records: list[dict]) -> None:
    """Fail a budget whose allocation leaves any slot below the floor, and say what budget would work."""
    if min_per_slot <= 0:
        return
    thin = {slot: target for slot, target in quota_targets.items() if target < min_per_slot}
    if not thin:
        return
    family_counts = Counter(rec["family"] for rec in records)
    slots_per_family: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        slots_per_family[rec["family"]].add(_quota_key(rec))
    total = sum(family_counts.values())
    needed = max(
        -(-min_per_slot * len(slots) * total // family_counts[fam])  # ceil division
        for fam, slots in slots_per_family.items()
    )
    raise SystemExit(
        f"--trace_budget {budget} allocates fewer than {min_per_slot} traces to {len(thin)} quota "
        f"slot(s) (e.g. {sorted(thin)[:3]}): that is a token gesture, not format training. The "
        f"smallest budget clearing the floor at this dataset's mix is ~{needed}; raise "
        f"--trace_budget (pinned identically for every model size) or lower --min_per_slot "
        f"deliberately."
    )


def _split_cell_targets_by_answer(records: list[dict], cell_targets: dict[str, int]) -> dict[str, int]:
    """Refine cell-level targets into quota-key targets: even per-answer splits for verifiable cells (remainder dealt to the first answers in sorted order, deterministically), cell-level passthrough for neutral cells."""
    answers_by_cell: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        if rec.get("verifiable") and rec.get("answer"):
            answers_by_cell[_cell_key(rec)].add(rec["answer"])
    targets: dict[str, int] = {}
    for cell, cell_target in cell_targets.items():
        answers = sorted(answers_by_cell.get(cell, ()))
        if not answers:
            targets[cell] = cell_target
            continue
        base, extra = divmod(cell_target, len(answers))
        for answer_i, answer in enumerate(answers):
            targets[f"{cell}|{answer}"] = base + (1 if answer_i < extra else 0)
    return targets


def _quota_batches(records: list[dict], batch_size: int, targets: dict[str, int],
                   kept_slots: Counter, rounds: int = 1):
    """Yield batches that chase each quota slot's target kept-trace count instead of walking the file head-first.

    Slots are quota keys (format cell, plus gold answer for verifiable rows — see _quota_key). Round-robin over the slots still below target, one prompt per slot per pass, re-reading `kept_slots` (which the consumer updates) between batches. Yield variance therefore costs generation time, not coverage: a stubborn slot keeps drawing fresh prompts while slots already at target stop consuming compute. `in_flight` counts this batch's not-yet-scored pulls so a slot near target is not oversampled on the assumption every pull succeeds — at most one trace is kept per prompt, so kept + in_flight >= target means nothing more is needed IF they all pass; if some fail, the next batch sees the updated counts and tops the slot back up.

    `rounds` is the automatic fill-until-done loop: when a slot has consumed its whole pool and still sits below target (every prompt got one k_samples pass and none survived), the next round re-queues that slot's UNKEPT prompts for another sampling pass — a prompt that already yielded a trace is never re-run, so the SFT set can never contain the same prompt twice. The loop is bounded (rounds x pool x k_samples generations, worst case) and only touches the shortfall: filled slots cost nothing in later rounds. A slot still short after the last round is reported as cells_below_target and fails the build's expectations loudly — at that point the model genuinely struggles to produce that (cell x answer) and the levers are --k_samples and --max_rounds, not silence.
    """
    by_slot: dict[str, list[dict]] = {}
    order: list[str] = []
    for rec in records:
        key = _quota_key(rec)
        if key not in by_slot:
            by_slot[key] = []
            order.append(key)
        by_slot[key].append(rec)
    for _round in range(rounds):
        queues = {
            key: deque(rec for rec in by_slot[key] if not rec.get("_kept"))
            for key in order
            if kept_slots[key] < targets.get(key, 0)
        }
        if not any(queues.values()):
            return
        while True:
            batch: list[dict] = []
            in_flight: Counter = Counter()
            progressed = True
            while len(batch) < batch_size and progressed:
                progressed = False
                for key in order:
                    queue = queues.get(key)
                    if not queue or kept_slots[key] + in_flight[key] >= targets.get(key, 0):
                        continue
                    batch.append(queue.popleft())
                    in_flight[key] += 1
                    progressed = True
                    if len(batch) >= batch_size:
                        break
            if not batch:
                break
            yield batch


def _budget_cell_targets(records: list[dict], budget: int) -> dict[str, int]:
    """Per-cell quotas that sum to exactly `budget`, allocated by the dataset's own family mix.

    The budget is the RECIPE-LEVEL knob: pinning the same total at every model size (4B/9B/27B) makes the training dose a controlled input rather than an artifact of each size's yield, so cross-size differences in M0 behaviour cannot be blamed on one size simply training on more traces. Family shares are measured from the input records (so a rebuilt dataset reallocates automatically), distributed by largest remainder to land exactly on the budget, then spread evenly across the family's format cells with the remainder dealt to the first cells in sorted order — deterministic, so every size gets the identical per-cell target table.
    """
    family_counts = Counter(rec["family"] for rec in records)
    cells_by_family: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        cells_by_family[rec["family"]].add(_cell_key(rec))
    total = sum(family_counts.values())
    exact = {fam: budget * count / total for fam, count in family_counts.items()}
    alloc = {fam: int(share) for fam, share in exact.items()}
    remainder_order = sorted(exact, key=lambda fam: (alloc[fam] - exact[fam], fam))
    i = 0
    while sum(alloc.values()) < budget:
        alloc[remainder_order[i % len(remainder_order)]] += 1
        i += 1
    targets: dict[str, int] = {}
    for fam, fam_budget in alloc.items():
        cells = sorted(cells_by_family[fam])
        base, extra = divmod(fam_budget, len(cells))
        for cell_i, cell in enumerate(cells):
            targets[cell] = base + (1 if cell_i < extra else 0)
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data/rl/m0_format/train.jsonl",
                        help="M0 train split; the same screened neutral prompts GRPO uses")
    parser.add_argument("--out", default="data/rl/m0_warmstart/sft_traces.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B",
                        help="the EXACT base GRPO starts from (self-distill)")
    parser.add_argument("--brevity", choices=sorted(BREVITY_OVERRIDES), default="twobullet")
    parser.add_argument("--k_samples", type=int, default=6, help="generations per prompt to filter from")
    parser.add_argument("--max_new_tokens", type=int, default=400,
                        help="generation cap; rambles hit it, never commit, and are filtered out")
    parser.add_argument("--max_trace_tokens", type=int, default=256,
                        help="keep only traces whose reasoning is within this (the band's full-marks ceiling)")
    parser.add_argument("--min_trace_tokens", type=int, default=32,
                        help="reject traces whose reasoning is shorter than this. The band in m0/rewards.py "
                             "has a floor as well as a ceiling and only the ceiling was ever enforced here, "
                             "so a 22-token stub that closed early and committed correctly counted as a pass. "
                             "Default is the rewards band floor; 0 disables.")
    parser.add_argument("--topup", default=None,
                        help="path to an existing trace file to FILL IN rather than rebuild. Quota targets "
                             "become the per-slot shortfall against that file, and every prompt it already "
                             "used is excluded, so the run generates only the traces the set is missing and "
                             "can never duplicate a prompt. Writes just the new traces to --out; concatenate "
                             "them onto the existing file. Use when a post-hoc screen drops a handful of "
                             "traces and the set needs to come back to --trace_budget without paying for the "
                             "whole hour again -- NOT as a way to grow a set past its budget.")
    parser.add_argument("--target_trace_tokens", type=int, default=None,
                        help="if set, keep the passing trace whose reasoning length is CLOSEST to this, instead "
                             "of the shortest -- lifts the kept median off the band floor. The six-bullet "
                             "generations already reach ~150 (closed p90 ~157); keep-shortest was discarding it. "
                             "Still capped by --max_trace_tokens.")
    parser.add_argument("--max_coda_chars", type=int, default=0,
                        help="max characters allowed BETWEEN </think> and the final answer line. The target "
                             "should teach 'close reasoning, then commit' with nothing in between, so the "
                             "default of 0 requires exactly that. Raise only deliberately; the first 9B set "
                             "had a median 364-char coda and the model learned to reproduce it at serve time.")
    parser.add_argument("--strip_coda", action="store_true",
                        help="instead of rejecting a trace whose coda exceeds --max_coda_chars, keep it and "
                             "delete the coda, leaving reasoning + </think> + answer line. Use when rejection "
                             "costs too much yield; prefer plain rejection when yield allows.")
    parser.add_argument("--temperature", type=float, default=1.0)
    # Qwen3.5's model card recommends temperature=1.0, top_p=0.95, top_k=20 for thinking mode. Passing
    # only temperature left top_p/top_k at the transformers defaults (1.0 / 50) -- a far longer sampling
    # tail than the vendor specifies, and the regime in which the 9B wanders into restating the
    # instruction instead of following it. presence_penalty=1.5 is also recommended but has no
    # equivalent in HF generate(), so it is not applied here. The 4B v2 was built before these were set.
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--batch_prompts", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N prompts -- use a small N to pick --brevity first")
    parser.add_argument("--per_cell_target", type=int, default=None,
                        help="keep generating until every format cell (family x presented label order) has this many "
                             "kept traces or its prompt pool is exhausted, instead of walking the dataset head-first. "
                             "The battery can ask any of the 6 ternary presentations, so per-cell yield variance "
                             "must cost compute, not coverage. Cells that exhaust their pool below target are listed "
                             "in the manifest -- treat any such cell as a gate failure for that build, not a shrug. "
                             "Composes with --limit (the head-slice applies first).")
    parser.add_argument("--trace_budget", type=int, default=None,
                        help="THE v3 RECIPE KNOB: total kept traces to aim for, allocated across format cells by the "
                             "dataset's own family mix. Pin the SAME value at every model size (200 for the 4B/9B/27B "
                             "study; viable because ternary trains two word schemes -- 18 ternary_factual slots at "
                             "2-3 traces each, minimum viable budget 180 at floor 2) so the training dose is a "
                             "controlled recipe input, not an artifact of each size's yield -- v2 trained the 4B on "
                             "224 traces and the 9B on 274 purely because of what survived, a quiet data-quantity "
                             "confound on every cross-size comparison. Yield differences are absorbed by the prompt "
                             "pool, --k_samples and --max_rounds, never by N. Mutually exclusive with "
                             "--per_cell_target.")
    parser.add_argument("--min_per_slot", type=int, default=2,
                        help="refuse a --trace_budget whose allocation gives any quota slot fewer than this many "
                             "traces. 1-2 examples of a presentation is a token gesture, not format training, and a "
                             "budget that thin should fail at allocation time -- with the minimum viable budget in "
                             "the error -- rather than silently under-train a cell. 0 disables the guard.")
    parser.add_argument("--max_rounds", type=int, default=6,
                        help="how many k_samples passes a quota slot may take over its prompt pool before the "
                             "build gives up on it. Round 1 tries every needed prompt once; later rounds re-run "
                             "ONLY the unkept prompts of slots still below quota (a prompt that already yielded a "
                             "trace is never re-run, so no duplicate SFT rows). Bounded, automatic fill-in for "
                             "slots the model finds hard; slots still short after the last round land in the "
                             "manifest's cells_below_target and the run warns loudly.")
    parser.add_argument("--debug_dump", type=int, default=0,
                        help="print this many raw completions with their filter status, for diagnosis")
    parser.add_argument("--prose", action="store_true",
                        help="rewrite twobullet output as prose (strip '- ' markers) -- reliable bullet yield, prose reasoning style")
    # Where the generations come from. `local` is the original path and the default: transformers on
    # the box, the exact weights the LoRA then trains. `openrouter` serves the same model per token
    # so a 27B trace set can be built without renting a card -- at the cost of running against a
    # third party's (usually fp8) copy behind their chat template. Read the deviation this creates
    # in m0/data/openrouter_backend.py's docstring BEFORE using it for anything published.
    parser.add_argument("--backend", choices=("local", "openrouter"), default="local",
                        help="local: transformers on this box (default). openrouter: hosted, per-token")
    parser.add_argument("--or_model", default=None,
                        help="OpenRouter model id (default qwen/qwen3.5-27b); validated against the live catalogue")
    parser.add_argument("--or_provider", default=None,
                        help="pin one OpenRouter provider (default DeepInfra); '' routes freely, which is NOT recommended")
    parser.add_argument("--or_concurrency", type=int, default=8,
                        help="in-flight OpenRouter requests")
    parser.add_argument("--or_probe", action="store_true",
                        help="sample one generation from every provider serving --or_model, print them, and exit")
    args = parser.parse_args()

    records = load_records(args.dataset)
    if args.limit:
        records = records[: args.limit]
    override = BREVITY_OVERRIDES[args.brevity]

    if args.backend == "openrouter":
        from scripts.m0.data import openrouter_backend as orb

        or_model = args.or_model or orb.DEFAULT_MODEL
        # `--or_provider ''` is the explicit opt-out; None means "unset, use the pinned default".
        or_provider = orb.DEFAULT_PROVIDER if args.or_provider is None else (args.or_provider or None)
        if args.or_probe:
            orb.probe(or_model, or_provider, f"{records[0]['prompt']}\n\n{override}", args.max_new_tokens)
            return
        settings = orb.resolve_settings(or_model, or_provider)
        generator = orb.OpenRouterGenerator(settings, concurrency=args.or_concurrency)
        # The tokenizer still comes from the REAL model, not the endpoint: every band check
        # (--max_trace_tokens, --target_trace_tokens) is in tokens, and it has to be the tokenizer
        # the SFT and the serve-time budget use. It is a small download and needs no GPU.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = None
        print(f"[warmstart] backend=openrouter model={settings['model']} "
              f"provider={settings['provider'] or 'ROUTED (unpinned)'} quant={settings['quantization']} "
              f"| band checks tokenized with {args.model}")
    else:
        if args.or_probe:
            raise SystemExit("--or_probe only applies to --backend openrouter")
        generator = None
        model, tokenizer = load_model(args.model)

    n_kept = 0                              # rows stream to disk; only the count is needed in memory
    considered = 0
    per_family_seen: Counter = Counter()
    per_family_kept: Counter = Counter()
    # Per label scheme AND per presented order. v3's whole purpose is a format-robust M0, so the
    # reviewable question is whether the kept traces actually span the schemes and rotations rather
    # than the sampler landing them all on MORE/LESS/SAME. A scheme with near-zero yield teaches
    # the model nothing about that format, and that has to be caught here -- before an hour of
    # trace generation and a training run -- not afterwards in the battery.
    per_scheme_seen: Counter = Counter()
    per_scheme_kept: Counter = Counter()
    per_order_seen: Counter = Counter()
    per_order_kept: Counter = Counter()
    per_cell_seen: Counter = Counter()
    per_cell_kept: Counter = Counter()
    per_cell_answer_kept: dict[str, Counter] = defaultdict(Counter)
    lengths: list[int] = []
    reason_counts: Counter = Counter()      # why generations were rejected, across all samples
    closed_lengths: list[int] = []          # reasoning tokens for every generation that closed </think>
    debug_left = args.debug_dump
    n_generations = 0

    # BOTH FILES ARE WRITTEN INCREMENTALLY AND FLUSHED EVERY BATCH, not accumulated and dumped at the end.
    # Two reasons, and the second is why this is not merely tidier:
    #   1. A crash, an OOM at batch 80 of 88, or a killed box used to lose the ENTIRE run. Generation is
    #      the expensive step (~1 h at 9B, and it scales with model size -- a 27B build is long enough
    #      that "all or nothing" stops being an acceptable failure mode).
    #   2. It makes a run inspectable WHILE IT RUNS. `wc -l` and `tail` on either file answer "is this
    #      going well" without waiting an hour to find out, which is otherwise unanswerable: the progress
    #      bar reports batches, not whether the traces coming out are any good.
    # all_generations keeps EVERY generation, kept or rejected, with its status. Rejections used to be
    # counted and thrown away, which made a disappointing yield undiagnosable without paying for the run
    # again -- and on the 9B every real diagnosis (the turn-spill, the model quoting </think> and closing
    # its own block, the self-defeating "do not restate" clause) came from raw text the tallies could not
    # show. A run is ~4k generations, a few MB. Cheap insurance; never delete this to save space.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gen_path = out_path.parent / "all_generations.jsonl"
    kept_f = out_path.open("w")
    gen_f = gen_path.open("w")
    print(f"[warmstart] streaming traces -> {out_path}\n"
          f"[warmstart] streaming every generation -> {gen_path}\n"
          f"[warmstart] both flush per batch; `wc -l` them mid-run to check progress")

    from tqdm.auto import tqdm
    if args.per_cell_target is not None and args.trace_budget is not None:
        raise SystemExit("--per_cell_target and --trace_budget both set per-cell quotas; pass exactly one")
    cell_targets: dict[str, int] | None = None
    if args.per_cell_target is not None:
        cell_targets = {key: args.per_cell_target for key in sorted({_cell_key(rec) for rec in records})}
    elif args.trace_budget is not None:
        cell_targets = _budget_cell_targets(records, args.trace_budget)
    quota_targets: dict[str, int] | None = None
    kept_slots: Counter = Counter()
    if cell_targets is not None:
        # Quotas are enforced at (cell x answer) granularity, so the kept set carries the SAME two
        # balances the dataset guarantees: the family/cell mix AND uniform answers within each cell.
        # Without the answer split the survival filters decide the kept answers, and they are
        # measurably skewed (v2 4B: 6 A vs 13 B off a balanced dataset).
        quota_targets = _split_cell_targets_by_answer(records, cell_targets)
        if args.trace_budget is not None:
            _check_min_per_slot(quota_targets, args.min_per_slot, args.trace_budget, records)
            print(f"[warmstart] trace budget {args.trace_budget} allocated over {len(quota_targets)} "
                  f"quota slots (cell x answer; full table in the manifest), min {min(quota_targets.values())} "
                  f"per slot, up to {args.max_rounds} sampling rounds per slot")
        if args.topup:
            # Subtract AFTER the floor check: the shortfall is a couple of traces by construction, so
            # checking --min_per_slot against it would fail every top-up. The floor is a property of
            # the full allocation, which is what was validated above.
            existing = [json.loads(line) for line in Path(args.topup).read_text().splitlines() if line.strip()]
            have = Counter(_quota_key(row) for row in existing)
            used = {row["record_id"] for row in existing}
            for rec in records:
                if rec["record_id"] in used:
                    rec["_kept"] = True     # never re-run a prompt the set already has: no duplicate rows
            quota_targets = {slot: max(0, target - have[slot]) for slot, target in quota_targets.items()}
            shortfall = {slot: n for slot, n in quota_targets.items() if n}
            if not shortfall:
                raise SystemExit(f"--topup {args.topup} already meets every quota slot ({len(existing)} "
                                 f"traces); nothing to generate.")
            print(f"[warmstart] TOP-UP against {args.topup}: {len(existing)} existing traces, "
                  f"{sum(shortfall.values())} to generate across {len(shortfall)} slot(s) {sorted(shortfall)}")
        batches = _quota_batches(records, args.batch_prompts, quota_targets, kept_slots,
                                 rounds=max(1, args.max_rounds))
    else:
        batches = [records[i: i + args.batch_prompts] for i in range(0, len(records), args.batch_prompts)]

    # Progress is measured in TRACES against the budget, not in batches. The quota walk is a
    # generator (it has to be -- how many batches a run needs depends on yield, which is not known
    # until it happens), so tqdm cannot len() it and would otherwise print a bare rolling count with
    # no total, no percentage and no ETA. The budget IS known, so count toward that instead: it is
    # also the number a reader actually wants, and it makes a stalled run obvious (batches tick,
    # traces do not). Without a budget there is no denominator, so fall back to counting batches.
    total_traces = sum(quota_targets.values()) if quota_targets else None
    progress = tqdm(
        total=total_traces if total_traces else len(batches),
        desc=f"[warmstart] {'traces' if total_traces else 'batches'} ({args.brevity})",
        unit="trace" if total_traces else "batch",
    )
    for batch in batches:
        gen_prompts = [f"{rec['prompt']}\n\n{override}" for rec in batch]
        if generator is not None:
            completions = generator.generate(gen_prompts, args.k_samples, args.max_new_tokens,
                                             args.temperature, args.top_p, args.top_k)
        else:
            completions = generate(model, tokenizer, gen_prompts, args.k_samples,
                                   args.max_new_tokens, args.temperature, args.top_p, args.top_k)
        for rec, samples in zip(batch, completions):
            considered += 1
            per_family_seen[rec["family"]] += 1
            _labels = rec.get("labels") or []
            if len(_labels) == 3:
                per_scheme_seen["/".join(sorted(_labels))] += 1
                per_order_seen["/".join(_labels)] += 1
            per_cell_seen[_cell_key(rec)] += 1
            passing = []
            for text in samples:
                status, clean, n = evaluate_trace(text, rec["labels"], rec.get("answer"),
                                                  bool(rec.get("verifiable")), tokenizer, args.max_trace_tokens,
                                                  args.max_coda_chars, args.strip_coda,
                                                  args.min_trace_tokens)
                # Prose rendering and the format-echo screen happen HERE, per candidate, rather than
                # after the winner is chosen. Doing it afterwards meant the length-targeting picked a
                # trace before anything had looked at whether it was reasoning -- on the 9B's
                # "a house brick vs a wooden pencil" prompt it took a 192-token instruction-restating
                # sample over a clean 79-token one, purely because 192 scores closer to the 150 target,
                # and there was no path back. Screening first means the target chooses among traces
                # that are already known to be usable, and a prompt is only lost when EVERY sample
                # fails -- not when the best-scoring one happens to.
                if status == "pass":
                    if args.prose:
                        clean, n = bullets_to_prose(clean, tokenizer)
                    if is_format_echo(clean):
                        status, clean = "format_echo", None
                    elif n < args.min_trace_tokens:
                        status, clean = "too_short", None
                reason_counts[status] += 1
                n_generations += 1
                gen_f.write(json.dumps({
                    "record_id": rec["record_id"], "family": rec["family"], "status": status,
                    "n_reasoning_tokens": n, "answer": rec.get("answer"), "text": text,
                }, ensure_ascii=True) + "\n")
                if status != "no_close":
                    closed_lengths.append(n)
                if debug_left > 0:
                    print(f"\n[warmstart][debug] status={status} n_reasoning={n} family={rec['family']} "
                          f"labels={rec['labels']} answer={rec.get('answer')}\n{text[:800]}\n{'-' * 60}")
                    debug_left -= 1
                if status == "pass":
                    passing.append((clean, n))
            if not passing:
                continue
            if args.target_trace_tokens is not None:
                # Closest to the target length. The default (shortest) pins the kept median to the band
                # floor; the six-bullet generations already produce ~150-token reasoning (closed p90 ~157),
                # so targeting keeps that instead of discarding it.
                clean, n_reasoning = min(passing, key=lambda r: abs(r[1] - args.target_trace_tokens))
            else:
                clean, n_reasoning = min(passing, key=lambda r: r[1])   # shortest passing trace
            per_family_kept[rec["family"]] += 1
            _labels = rec.get("labels") or []
            if len(_labels) == 3:
                per_scheme_kept["/".join(sorted(_labels))] += 1
                per_order_kept["/".join(_labels)] += 1
            per_cell_kept[_cell_key(rec)] += 1
            kept_slots[_quota_key(rec)] += 1
            rec["_kept"] = True  # retry rounds must never re-run a prompt that already yielded a trace
            if rec.get("verifiable") and rec.get("answer"):
                # Kept-answer balance per cell: if one answer role survives the filters more often than the others, SFT installs a label prior inside that presentation even though the DATASET was balanced. Recorded so the imbalance is visible before training, not discovered in the battery.
                per_cell_answer_kept[_cell_key(rec)][rec["answer"]] += 1
            lengths.append(n_reasoning)
            n_kept += 1
            kept_f.write(json.dumps({
                "schema": "m0_warmstart_sft_v1",
                "record_id": rec["record_id"],
                "family": rec["family"],
                "labels": rec["labels"],
                "answer": rec.get("answer"),
                "verifiable": bool(rec.get("verifiable")),
                "prompt": rec["prompt"],           # the NORMAL prompt -- what the model learns to be brief under
                "completion": clean,
                "n_reasoning_tokens": n_reasoning,
                "brevity": args.brevity,
            }, ensure_ascii=True) + "\n")

        # Per BATCH, not per row: one fsync-ish flush every ~42 s costs nothing, and bounds a crash's
        # loss to the batch in flight rather than the whole run.
        kept_f.flush()
        gen_f.flush()

        # Live state, so a long run is readable while it runs rather than only in the manifest.
        # `slots_left` is the one that matters near the end: the bar can sit at 197/200 for a while
        # because the last few slots are the ones the model finds hard, and that is the retry rounds
        # working, not a stall.
        if total_traces:
            progress.n = min(n_kept, total_traces)
            slots_left = sum(1 for slot, target in quota_targets.items() if kept_slots[slot] < target)
            progress.set_postfix(
                slots_left=slots_left,
                prompts=considered,
                yield_pct=f"{n_kept / max(considered, 1):.0%}",
                refresh=False,
            )
            progress.refresh()
        else:
            progress.update(1)

    progress.close()
    kept_f.close()
    gen_f.close()
    print(f"[warmstart] wrote {n_generations} raw generations (all statuses) to {gen_path} -- "
          f"read these before changing anything in response to a disappointing yield")

    lengths.sort()
    yield_rate = n_kept / max(considered, 1)
    manifest = {
        "schema": "m0_warmstart_manifest_v1",
        "dataset": args.dataset,
        "model": args.model,
        # Which copy of the weights actually produced these traces. On the local backend that is
        # `model` above and nothing more needs saying; on OpenRouter it is a named provider's
        # quantized serve, which is not recoverable after the run and is exactly what a reader
        # comparing 27B against the on-stack 4B and 9B builds needs to see.
        **(generator.provenance() if generator is not None else {"backend": "local"}),
        "brevity": args.brevity,
        # Recorded, not inferred. m0/scripts/rederive_traces.py had to guess this back from
        # `"bullet" in brevity` because the manifest never carried it -- a guess that happens to be
        # right for every variant shipped so far and would silently be wrong for any non-bullet
        # prompt that still wanted prose. The flag decides what the SFT target LOOKS LIKE (prose vs
        # literal "- " bullets), so it belongs in provenance next to brevity itself.
        "prose": bool(args.prose),
        "k_samples": args.k_samples,
        "max_new_tokens": args.max_new_tokens,
        "max_trace_tokens": args.max_trace_tokens,
        "target_trace_tokens": args.target_trace_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_coda_chars": args.max_coda_chars,
        "strip_coda": bool(args.strip_coda),
        "min_trace_tokens": args.min_trace_tokens,
        "topup_of": args.topup,
        "rejections": dict(reason_counts),
        "prompts_considered": considered,
        "traces_kept": n_kept,
        "yield_rate": yield_rate,
        "reasoning_tokens": {
            "median": lengths[len(lengths) // 2] if lengths else None,
            "p90": lengths[int(len(lengths) * 0.9)] if lengths else None,
            "max": lengths[-1] if lengths else None,
        },
        "per_family_yield": {fam: {"seen": per_family_seen[fam], "kept": per_family_kept[fam]}
                             for fam in sorted(per_family_seen)},
        # Ternary rows only; binary rows have two labels and are excluded.
        "per_label_scheme_yield": {key: {"seen": per_scheme_seen[key], "kept": per_scheme_kept[key]}
                                   for key in sorted(per_scheme_seen)},
        "per_label_order_yield": {key: {"seen": per_order_seen[key], "kept": per_order_kept[key]}
                                  for key in sorted(per_order_seen)},
        "per_cell_target": args.per_cell_target,
        "trace_budget": args.trace_budget,
        "max_rounds": args.max_rounds,
        # Quota slots are cell x answer for verifiable rows, bare cell for neutral ones.
        "quota_targets": quota_targets,
        "quota_kept": {slot: kept_slots[slot] for slot in sorted(quota_targets)} if quota_targets else None,
        "per_cell_yield": {cell: {"seen": per_cell_seen[cell], "kept": per_cell_kept[cell]}
                           for cell in sorted(per_cell_seen)},
        "per_cell_answer_kept": {cell: dict(per_cell_answer_kept[cell])
                                 for cell in sorted(per_cell_answer_kept)},
        "cells_below_target": sorted(
            slot for slot, target in (quota_targets or {}).items()
            if kept_slots[slot] < target
        ),
    }
    (out_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[warmstart] kept {n_kept}/{considered} prompts  (yield {yield_rate:.0%}, brevity={args.brevity})")
    if generator is not None:
        # Spend is stated here as well as in the manifest: OpenRouter bills in arrears with no live
        # counter, so the run's own total is the only figure available while deciding whether to
        # scale it up.
        print(f"[warmstart] OpenRouter: {generator.calls} calls, {generator.errors} failed, "
              f"${generator.cost_usd:.4f} spent "
              f"({generator.reasoning_tokens:,} reasoning tokens of {generator.completion_tokens:,} out)")
    if lengths:
        print(f"[warmstart] reasoning tokens: median {manifest['reasoning_tokens']['median']}, "
              f"p90 {manifest['reasoning_tokens']['p90']}, max {manifest['reasoning_tokens']['max']}")
    for fam in sorted(per_family_seen):
        s, k = per_family_seen[fam], per_family_kept[fam]
        print(f"[warmstart]   {fam:18s} {k:4d}/{s:<4d}  ({k / max(s, 1):.0%})")
    if per_cell_kept:
        min_cell = min(per_cell_seen, key=lambda c: per_cell_kept[c])
        print(f"[warmstart] format cells: {len(per_cell_seen)}, thinnest = {min_cell} with "
              f"{per_cell_kept[min_cell]} kept (full per-cell table in the manifest)")
    if manifest["cells_below_target"]:
        print(f"[warmstart] WARNING: {len(manifest['cells_below_target'])} quota slot(s) still below target "
              f"after {args.max_rounds} round(s): {manifest['cells_below_target']}. "
              f"A cell SFT never saw is a battery format the model was never format-trained on -- raise "
              f"--k_samples (or enlarge the dataset via m0/data/facts.py) rather than shipping this set, and "
              f"NEVER lower --trace_budget for one size alone: the budget is pinned across sizes so the "
              f"training dose stays comparable.")
    total_gen = sum(reason_counts.values())
    print("[warmstart] rejection breakdown: " +
          "  ".join(f"{r}={reason_counts[r]}({reason_counts[r] / max(total_gen, 1):.0%})"
                    for r in ("pass", "no_close", "no_commit", "wrong", "too_long",
                              "template_leak", "meta_echo", "coda")))
    if reason_counts["template_leak"]:
        print(f"[warmstart] NOTE: {reason_counts['template_leak']} generations ran past the assistant turn "
              f"(<|im_start|>/<|im_end|> in the text). A LARGE count means the STOP TOKENS are not taking -- "
              f"check the 'stop tokens resolved' line above before trusting the kept set.")
    if reason_counts["meta_echo"]:
        print(f"[warmstart] NOTE: {reason_counts['meta_echo']} generations restated the brevity instruction "
              f"instead of following it -- the model quotes the prompt and emits a real </think> token while "
              f"doing so, closing its own reasoning block early. EXPECTED on the 9B at roughly 25-30%, and "
              f"NOT a bug to fix: the prompt names the tags deliberately, because removing the mention makes "
              f"the model never close the block at all (0% yield, tried on both 4B and 9B). This is paid for "
              f"with oversampling -- raise --limit, not the prompt. See the BREVITY_OVERRIDES comment.")
    if closed_lengths:
        cl = sorted(closed_lengths)
        print(f"[warmstart] reasoning tokens of CLOSED generations ({len(cl)}): "
              f"median {cl[len(cl) // 2]}, p90 {cl[int(len(cl) * 0.9)]}, max {cl[-1]} "
              f"(>> {args.max_trace_tokens} means the brevity prompt is not biting; "
              f"few closed gens means raise --max_new_tokens)")
    if yield_rate < 0.3:
        print("[warmstart] WARNING: low yield -- try a stronger --brevity, more --k_samples, or a "
              "higher --max_trace_tokens before scaling to the full set.")


if __name__ == "__main__":
    main()
