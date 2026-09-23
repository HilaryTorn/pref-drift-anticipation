"""Prompt carriers for M0 format training.

M0 teaches one behavior: reason, then end on a bare `Answer: <label>` line, inside the reasoning
budget. That behavior has to be carrier-INDEPENDENT. The study runs seven structurally different
forced-choice wrappers (data/elicitation_specs/*.json plus compute_utilities/templates.py), and a
model trained on only one of them would learn to comply with that phrasing rather than with the
format. That failure mode is worse than it sounds: it would look like success on whichever battery
shares the training wrapper, and it would make the model behave specially on -- say -- the values
template, which is a contamination of the measurement rather than a fix to it.

So every live wrapper shape is represented here, PARAPHRASED rather than copied. Paraphrase is
deliberate on both counts: reproducing a live stimulus verbatim would put measurement wording into
training data (the thing scripts/validate_sft_data.py screens for everywhere else), while
reproducing only one shape would overfit the format to a phrasing. What is held constant is the
STRUCTURE; what varies is everything else.

The structural axes, and the live template each mirrors:

  question_first      question opens, then Option blocks         values_preferences
  question_last       premise, Option blocks, then the question  values_anticipation
  leveled_choice      "assume both are equally X" paragraph      coding_task_preference
  future_self         framing about a future state, then options coding_training_preference
  constrained_choice  long constraints paragraph before options  rl_preferences
  premise_task        premise + bare `Task:` block, no Options   coding_anticipation / rl_anticipation
  options_first       options, then the question in line one     templates.py comparison_prompt

The final sentence -- the reasoning instruction -- is the load-bearing one, and the live specs
already word it three different ways ("Briefly reason through the choice." / "...the forecast." /
"...which you would rather be."). REASONING_INSTRUCTIONS varies it further, so what generalizes is
"end with an answer-only line", not a memorized suffix. Every variant states the label set
inline rather than as a displayed quoted block: templates.py:20-21 records that small models
imitate a displayed block instead of reading it as a spec, answering with the bare string
`Answer: A` and no reasoning at all -- which is precisely the degenerate behavior M0 must not
teach.
"""

from __future__ import annotations


# Label schemes. Varying these is why M0 learns "commit using the labels you were given" instead of
# a prior on the token "A". Precedent: scripts/score_label_variants.py varies the same axis to show
# the fitted utilities do not depend on it.
# RESERVED, never add here: ("X", "Y") and lowercase ("a", "b") are the robustness checkers' held-out binary schemes, and ("GREATER", "SMALLER", "UNCHANGED") plus ("STRONGER", "WEAKER", "UNCHANGED") the held-out ternary schemes — held out on their up/down words, which is where the out-of-distribution read lives; the flat word UNCHANGED is shared with trained HIGHER/LOWER/UNCHANGED on purpose, so every probe varies exactly one factor. STRONGER/WEAKER was trained briefly in the v3 design and dropped on 2026-07-31 to keep the pinned 200-trace budget viable: the live battery only ever asks MORE/LESS/SAME, the extra schemes exist for the robustness checker, and a third trained scheme forced 27 ternary_factual quota slots at 1-2 traces each — a token dose. Two trained schemes x 3 rotations = 18 slots at 2-3 traces (6-7 per presentation cell) inside the same budget. They are held out from the SAME natural family as the trained schemes — deliberately, so a failure on them reads as format non-generalisation rather than as the model balking at an exotic label — and training on any of them would leave the study with no untrained format to probe generalisation with. tests/test_heldout_probe_schemes.py pins the disjointness.
BINARY_LABEL_SCHEMES = [("A", "B"), ("1", "2"), ("C", "D")]
# Ternary schemes, each written in canonical role order (up, down, flat) and presented in a
# rotated order per item. Binary has varied its labels since the start; ternary was left on one
# fixed tuple, which is why the anticipation battery's alternative-word robustness variants were
# out-of-distribution for M0 v2 and could not be interpreted (methodology working doc, forecast-side
# measurement validity). Semantic fit to "quantity" is deliberately loose here -- the objective is
# "commit using the labels you were given", the same reason binary trains on ("1","2") and ("C","D").
TERNARY_LABEL_SCHEMES = [
    ("MORE", "LESS", "SAME"),
    ("HIGHER", "LOWER", "UNCHANGED"),
]
# The live battery scheme, and the canonical role order answers are recorded in.
TERNARY_LABELS = TERNARY_LABEL_SCHEMES[0]


def _quoted_label_list(labels: tuple[str, ...]) -> str:
    """`"A" or "B"` / `"MORE", "LESS", or "SAME"` -- inline, never a displayed block."""
    quoted = [f'"{label}"' for label in labels]
    if len(quoted) == 2:
        return f"{quoted[0]} or {quoted[1]}"
    return ", ".join(quoted[:-1]) + f", or {quoted[-1]}"


def _answer_list(labels: tuple[str, ...]) -> str:
    """`"Answer: A" or "Answer: B"` -- the exact strings the parser accepts."""
    quoted = [f'"Answer: {label}"' for label in labels]
    if len(quoted) == 2:
        return f"{quoted[0]} or {quoted[1]}"
    return ", ".join(quoted[:-1]) + f", or {quoted[-1]}"


# The final instruction, paraphrased. Index chosen per item so the trained behavior attaches to the
# structure ("a bare answer line, last") and not to one sentence.
REASONING_INSTRUCTIONS = [
    "Briefly reason through the choice. Then put your final answer on its own line exactly as {answers}.",
    "Think it through briefly. Then give your final answer on its own line exactly as {answers}.",
    "Reason through it in a few sentences, then end with your final answer on its own line, exactly as {answers}.",
    "Work through the comparison briefly. Your last line must be your final answer, written exactly as {answers}.",
    "Briefly weigh the two. Then close with your final answer alone on its own line, exactly as {answers}.",
]

TERNARY_REASONING_INSTRUCTIONS = [
    "Briefly reason through the question. Then put your final answer on its own line exactly as one of: {answers}.",
    "Think it through briefly, then end with your final answer on its own line, exactly as one of: {answers}.",
    "Reason it out in a few sentences. Your last line must be your final answer, exactly as one of: {answers}.",
    "Briefly explain your thinking, then close with your final answer alone on its own line, as one of: {answers}.",
]


# --- Binary carriers -------------------------------------------------------------------------
# Each takes the two option texts, the label pair, a per-item question/framing dict, and the
# resolved reasoning instruction. Returning plain strings keeps every carrier inspectable by eye,
# which matters because the neutral items get a human read-through before the first real run.

def _question_first(opt_a, opt_b, labels, item, instruction):
    la, lb = labels
    return (
        f"{item['question']}\n\n"
        f"Option {la}:\n{opt_a}\n\n"
        f"Option {lb}:\n{opt_b}\n\n"
        f"{instruction}"
    )


def _question_last(opt_a, opt_b, labels, item, instruction):
    la, lb = labels
    return (
        f"{item['premise']}\n\n"
        "Below are two short descriptions.\n\n"
        f"Option {la}:\n{opt_a}\n\n"
        f"Option {lb}:\n{opt_b}\n\n"
        f"{item['question']}\n\n"
        f"{instruction}"
    )


def _leveled_choice(opt_a, opt_b, labels, item, instruction):
    la, lb = labels
    return (
        f"You will be shown two {item['plural_noun']}.\n\n"
        f"{item['leveling']}\n\n"
        f"{item['question']}\n\n"
        f"Option {la}:\n{opt_a}\n\n"
        f"Option {lb}:\n{opt_b}\n\n"
        f"{instruction}"
    )


def _future_self(opt_a, opt_b, labels, item, instruction):
    la, lb = labels
    return (
        f"{item['premise']}\n\n"
        f"{item['leveling']}\n\n"
        f"{item['question']}\n\n"
        f"Option {la}:\n{opt_a}\n\n"
        f"Option {lb}:\n{opt_b}\n\n"
        f"{instruction}"
    )


def _constrained_choice(opt_a, opt_b, labels, item, instruction):
    la, lb = labels
    return (
        f"You will be shown two {item['plural_noun']}.\n\n"
        f"{item['constraints']}\n\n"
        f"{item['question']}\n\n"
        f"Option {la}:\n{opt_a}\n\n"
        f"Option {lb}:\n{opt_b}\n\n"
        f"{instruction}"
    )


def _options_first(opt_a, opt_b, labels, item, instruction):
    la, lb = labels
    return (
        f"{item['lead_in']} {item['question']}\n\n"
        f"Option {la}:\n{opt_a}\n\n"
        f"Option {lb}:\n{opt_b}\n\n"
        f"{instruction}"
    )


BINARY_CARRIERS = {
    "question_first": _question_first,
    "question_last": _question_last,
    "leveled_choice": _leveled_choice,
    "future_self": _future_self,
    "constrained_choice": _constrained_choice,
    "options_first": _options_first,
}


# --- Ternary carrier -------------------------------------------------------------------------
# Mirrors coding_anticipation / rl_anticipation: a premise, then a bare `Task:`-style block, and
# NO Option blocks at all. This is the shape where the 0.8B most often fails to commit, so it
# carries real weight in the mix even though it is a single structure.

def _premise_task(subject, item, instruction):
    return (
        f"{item['premise']}\n\n"
        f"{item['question']}\n\n"
        f"{item['same_rule']}\n\n"
        f"{item['subject_header']}:\n{subject}\n\n"
        f"{instruction}"
    )


def build_binary_prompt(item: dict) -> str:
    """Render a binary item. ``item`` carries the carrier name, option texts, labels, and framing."""
    labels = tuple(item["labels"])
    carrier = BINARY_CARRIERS[item["carrier"]]
    instruction = REASONING_INSTRUCTIONS[item["instruction_idx"] % len(REASONING_INSTRUCTIONS)]
    instruction = instruction.format(answers=_answer_list(labels))
    return carrier(item["option_a"], item["option_b"], labels, item["framing"], instruction)


def build_ternary_prompt(item: dict) -> str:
    """Render a MORE/LESS/SAME item."""
    instruction = TERNARY_REASONING_INSTRUCTIONS[
        item["instruction_idx"] % len(TERNARY_REASONING_INSTRUCTIONS)
    ]
    # item["labels"] is the presented order for THIS item, so the answer list must follow it.
    # Using the module constant here was the bug that made ternary label variation impossible:
    # the dataset could carry varied labels and the rendered prompt would still say MORE/LESS/SAME.
    labels = tuple(item.get("labels") or TERNARY_LABELS)
    instruction = instruction.format(answers=_answer_list(labels))
    return _premise_task(item["subject"], item["framing"], instruction)


def build_prompt(item: dict) -> str:
    """Dispatch on family. Binary families carry two Option blocks; ternary families do not."""
    if item["family"].startswith("ternary"):
        return build_ternary_prompt(item)
    return build_binary_prompt(item)


def render_format_prompt(tokenizer, prompt_text: str) -> str:
    """Render the exact string the policy is trained on.

    Deliberately NOT rl_training.prompts.render_coding_prompt: that one pins
    ``enable_thinking=False`` for the coding RL arm. M0 must train in the regime the batteries are
    actually served under -- compute_utilities/create_agent.yaml's ``default_with_reasoning`` sets
    ``enable_thinking: true`` with ``reasoning_max_tokens: 1600`` -- because the whole point is to
    teach the model to close its native <think> block and commit before that budget is forced shut.
    Training with thinking off would optimize the wrong branch of the reward entirely.

    Tokenizers without a chat template fall back to raw text, matching the coding path's behavior.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    return prompt_text
