"""The M0 training objective: reason, then commit, inside the reasoning budget.

The methodology doc states the target as three rules -- reward an "Answer: X" line, reward
reasoning under 500 tokens, penalize neither. Implemented literally, those rules are degenerate:
a bare `Answer: A` with no reasoning at all satisfies BOTH (there is an answer line, and zero is
under 500), so the reward is maximized by never reasoning. That is not a hypothetical. It is the
0.8B's documented failure mode -- compute_utilities/templates.py:21 records batteries where the
median response was 9 characters, the bare string `Answer: A`. Training on the literal rules would
make that behavior the optimum instead of the bug.

So the objective has three terms, and the third is what keeps the reasoning honest:

    reward = W_COMMIT * commit + W_BAND * band + W_CORRECT * correct   (minus a no-commit penalty)

  commit   the last non-empty line is a bare `Answer: <label>`, per the SAME parser the scoring
           pipeline uses (m0/answer_format.py). This is the behavior M0 exists to install.
  band     the reasoning length sits in [FLOOR, REASONING_BUDGET]. Stops the model from winning by
           not reasoning, and from reasoning past the budget the endpoint enforces.
  correct  the committed label is the right one, on items that have a right answer. Without this,
           reasoning is decorative -- the model needs only to emit SOMETHING before committing, and
           filler satisfies the band as well as thought does. Correctness is what makes the tokens
           in between have to be about the question.

BAND SHAPE, and why it is asymmetric:

    n < FLOOR              ramp 0.4 -> 1.0
    FLOOR <= n <= TARGET   1.0
    TARGET < n <= BUDGET   1.0 -> 0.0, linear   (the brevity slope)
    n > BUDGET             0.0

Below the floor the response is merely terse. It still parses, still scores, still counts as a
datapoint at serve time -- so zeroing it would rank it with a ramble that never answered, which is
both false and destroys the gradient between the two. The soft ramp keeps terseness strictly worse
than the target without calling it worthless.

Above the budget scores zero, but the budget (500) now sits deliberately BELOW the serve-time
ceiling (compute_utilities/create_agent.yaml reasoning_max_tokens, still 1600). The zero is no
longer "this tail cannot exist in production" -- reasoning of 500-1600 tokens is perfectly servable
-- it is a brevity incentive: M0 teaches the model to finish in ~256 tokens and commit, so serve
cost drops because it VOLUNTARILY reasons short, not because an endpoint guillotines it mid-thought.
The serve cap may later be lowered to match (Phase 4), once a value-slice check confirms a tighter
cap does not shift committed answers.

No logits processor enforces the budget during TRL generation, so nothing stops the model from
running long while training -- learning to stay brief unaided is precisely the behavior installed.

Zeroing the band does not collapse over-budget into never-committed, because the band is only 0.3
of the total: an over-budget response that commits correctly still scores 0.7 against -0.2 for one
that never commits. The intended ordering holds throughout:

    in-band + correct  >  in-band  >  terse or over-long but committed  >  never committed
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from scripts.m0.answer_format import terminal_answer_label

# Term weights. Config-overridable via make_format_reward.
W_COMMIT = 0.5
W_BAND = 0.3
W_CORRECT = 0.2
NO_COMMIT_PENALTY = 0.2

# Reasoning-length band.
#
# REASONING_FLOOR is deliberately low -- it exists to rule out the zero-reasoning degenerate
# solution, not to mandate long deliberation.
#
# REASONING_TARGET is where full marks stop; TARGET..BUDGET decays on a slope. The slope is load-
# bearing: with a flat band every acceptable completion scores 1.0, within-group variance is zero,
# and GRPO -- whose advantage IS that variance -- gets no gradient (m0/scripts/check_rewards.py pins
# this).
#
# THE STARTING POINT IS ABOVE THE BAND. MEASURED (smoke rollouts, M0's own prompt): the untrained 4B
# reasons ~730 tokens median (mean ~850-980) on M0's fact questions -- entirely above this 256/500
# band, so every early rollout scores band 0 and GRPO cannot pull the policy into a band its samples
# never reach. M0 is therefore SEEDED first by a short self-distill warm-start
# (docs/m0-warmstart-plan.md) that drops the default length into range; the band then sharpens what
# the warm-start put there, rather than trying to create brevity from scratch. (An earlier
# bowling-ball measurement put the untrained mean near 1,440 -- either way, far above the band.)
#
# REASONING_BUDGET sits below the serve-time ceiling on purpose -- see BAND SHAPE in the module
# docstring. It is the length M0 trains the model to stay within, not the endpoint's force-cut.
REASONING_FLOOR = 32
REASONING_TARGET = 256
REASONING_BUDGET = 500
BAND_FLOOR_VALUE = 0.4

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


def split_completion(text: str, prompt_opened_think: bool = True) -> tuple[str, str]:
    """Split a completion into (reasoning, visible).

    The subtlety that makes this function worth reading: with ``enable_thinking=True`` the Qwen3.5
    chat template ends the PROMPT with an already-open `<think>` tag, so the model's completion
    begins inside the reasoning block and typically contains a closing `</think>` and no opening
    one. Verified against the real Qwen/Qwen3.5-4B tokenizer -- a rendered prompt ends
    `...<|im_start|>assistant\\n<think>\\n`.

    An earlier version keyed on a matched `<think>...</think>` pair and therefore matched nothing
    in production, silently falling through to the no-tags branch: every completion would have been
    scored as if the whole thing were both reasoning and answer, so the band term would have
    measured total length instead of reasoning length. Nothing in the reward curve would have shown
    it. Hence the split keys on the CLOSING tag, which is the one the model actually emits.

    ``prompt_opened_think`` says whether the prompt left a block open (True for every prompt
    render_format_prompt produces against a chat template). It only matters for the ambiguous case
    below.

      closing tag present   reasoning is everything before it, visible is everything after --
                            the production case, whether or not an opening tag appears;
      opening tag only      the model opened a block and ran out of room. All reasoning, no visible
                            answer, which is right: at serve time that response is unparseable and
                            it must not be able to earn a commit;
      neither, block open   same situation, just without a redundant opening tag -- truncated
                            mid-thought, so no visible answer;
      neither, block closed the model reasoned in plain content (what the 0.8B does). The whole
                            text is both reasoning and answer surface.
    """
    if not text:
        return "", ""
    close = text.find(_CLOSE_TAG)
    if close != -1:
        reasoning = text[:close]
        if reasoning.lstrip().startswith(_OPEN_TAG):
            reasoning = reasoning.lstrip()[len(_OPEN_TAG):]
        return reasoning, text[close + len(_CLOSE_TAG):]
    open_at = text.find(_OPEN_TAG)
    if open_at != -1:
        return text[open_at + len(_OPEN_TAG):], ""
    if prompt_opened_think:
        return text, ""
    return text, text


def _count_tokens(text: str, tokenizer=None) -> int:
    """Reasoning length in tokens.

    A tokenizer is threaded through from the trainer so the count is in the same units as the
    serve-time budget -- that budget is enforced on tokens, so measuring anything else would put
    the ceiling in the wrong place. The whitespace fallback exists only for unit tests.
    """
    if not text:
        return 0
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return len(text.split())


def band_score(n_tokens: int, floor: int = REASONING_FLOOR, budget: int = REASONING_BUDGET,
               target: int = REASONING_TARGET) -> float:
    """Score reasoning length. Full marks in [floor, target], decaying to 0 at budget.

        n < floor            ramp BAND_FLOOR_VALUE -> 1.0   (terse still parses, just not ideal)
        floor <= n <= target 1.0                            (the goal: brief and committed)
        target < n <= budget 1.0 -> 0.0, linear             (long but still servable)
        n > budget           0.0                            (cut short at serve time)
    """
    if n_tokens > budget:
        return 0.0
    if n_tokens > target:
        return (budget - n_tokens) / max(budget - target, 1)
    if n_tokens >= floor:
        return 1.0
    if floor <= 0:
        return 1.0
    return BAND_FLOOR_VALUE + (1.0 - BAND_FLOOR_VALUE) * (n_tokens / floor)


def score_completion(
    text: str,
    labels: list[str],
    answer: str | None,
    verifiable: bool,
    tokenizer=None,
    prompt_opened_think: bool = True,
    floor: int = REASONING_FLOOR,
    budget: int = REASONING_BUDGET,
    target: int = REASONING_TARGET,
    weights: tuple[float, float, float] = (W_COMMIT, W_BAND, W_CORRECT),
    penalty: float = NO_COMMIT_PENALTY,
) -> dict:
    """Score one completion and return the reward plus its per-term breakdown.

    The breakdown is persisted by the rollout logger. A single scalar makes a flat reward curve
    undiagnosable -- commit rate stuck at zero and commit rate fine but everything over budget look
    identical in the mean, and they call for opposite fixes.
    """
    w_commit, w_band, w_correct = weights
    reasoning, visible = split_completion(text, prompt_opened_think)
    committed = terminal_answer_label(visible, labels)
    n_reasoning = _count_tokens(reasoning, tokenizer)

    if committed is None:
        return {
            "reward": -penalty,
            "committed": None,
            "commit": 0.0,
            "band": 0.0,
            "correct": 0.0,
            "n_reasoning_tokens": n_reasoning,
            "over_budget": n_reasoning > budget,
        }

    band = band_score(n_reasoning, floor, budget, target)
    # Items with no ground truth (the neutral families) cannot use the correctness term, so its
    # weight folds into commit. Cross-family reward SCALE is irrelevant to GRPO -- advantages are
    # normalized within each prompt's own group -- but keeping the ranges aligned keeps the logged
    # mean readable as one number instead of a mixture.
    if verifiable and answer is not None:
        correct = 1.0 if committed == answer else 0.0
        reward = w_commit * 1.0 + w_band * band + w_correct * correct
    else:
        correct = 0.0
        reward = (w_commit + w_correct) * 1.0 + w_band * band

    return {
        "reward": reward,
        "committed": committed,
        "commit": 1.0,
        "band": band,
        "correct": correct,
        "n_reasoning_tokens": n_reasoning,
        "over_budget": False,
    }


def _completion_text(completion) -> str:
    """TRL hands completions as str (plain) or [{'role','content'}] (chat)."""
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]


def format_reward(
    completions,
    labels=None,
    answer=None,
    verifiable=None,
    tokenizer=None,
    floor: int = REASONING_FLOOR,
    budget: int = REASONING_BUDGET,
    target: int = REASONING_TARGET,
    **kwargs,
) -> list[float]:
    """TRL reward_funcs entry point. Dataset columns arrive as keyword arguments."""
    return [row["reward"] for row in score_batch(
        completions, labels, answer, verifiable, tokenizer, floor, budget, target
    )]


def score_batch(completions, labels, answer, verifiable, tokenizer, floor, budget,
                target=REASONING_TARGET) -> list[dict]:
    n = len(completions)
    labels = labels if labels is not None else [["A", "B"]] * n
    answer = answer if answer is not None else [None] * n
    verifiable = verifiable if verifiable is not None else [False] * n
    return [
        score_completion(
            _completion_text(completion), list(row_labels), row_answer, bool(row_verifiable),
            tokenizer=tokenizer, floor=floor, budget=budget, target=target,
        )
        for completion, row_labels, row_answer, row_verifiable
        in zip(completions, labels, answer, verifiable)
    ]


def make_format_reward(
    tokenizer=None,
    floor: int = REASONING_FLOOR,
    budget: int = REASONING_BUDGET,
    target: int = REASONING_TARGET,
    output_dir: str | None = None,
    num_generations: int | None = None,
    log_rollouts: bool = False,
):
    """Bind M0's grading knobs onto the reward for TRL, optionally logging every rollout.

    Modeled on rl_training/train_rl.py::_make_coding_reward, including the group-statistics logging
    -- with the per-term breakdown added, since M0's reward is a composite and its mean alone does
    not say which term is moving.

    The returned callable keeps ``__name__ == 'format_reward'`` because TRL uses the function name
    to label the reward's logged metric column.
    """
    def _score(completions, **kwargs):
        return score_batch(
            completions,
            kwargs.get("labels"), kwargs.get("answer"), kwargs.get("verifiable"),
            tokenizer, floor, budget, target,
        )

    if not log_rollouts:
        def format_reward_fn(completions, **kwargs):
            return [row["reward"] for row in _score(completions, **kwargs)]
        format_reward_fn.__name__ = "format_reward"
        return format_reward_fn

    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    log_path = Path(output_dir) / f"rollout_samples.rank{rank}.jsonl"
    call_index = 0

    def logged_format_reward(completions, prompts=None, record_id=None, **kwargs):
        nonlocal call_index
        if num_generations:
            assert len(completions) % num_generations == 0, (
                f"reward batch of {len(completions)} is not divisible by "
                f"num_generations={num_generations}; rollout group stats would be wrong"
            )
        scored = _score(completions, **kwargs)
        rewards = [row["reward"] for row in scored]
        prompts = prompts or [None] * len(completions)
        record_id = record_id or [None] * len(completions)
        group = num_generations or len(rewards)
        with log_path.open("a") as fout:
            for start in range(0, len(rewards), group):
                group_rewards = rewards[start: start + group]
                mean = sum(group_rewards) / len(group_rewards)
                variance = sum((r - mean) ** 2 for r in group_rewards) / max(len(group_rewards) - 1, 1)
                std = math.sqrt(variance)
                for index in range(start, min(start + group, len(rewards))):
                    row = dict(scored[index])
                    row.update({
                        "schema": "m0_format_rollout_v1",
                        "reward_call": call_index,
                        "record_id": record_id[index],
                        "prompt": prompts[index],
                        "completion": _completion_text(completions[index]),
                        "group_mean": mean,
                        "group_std": std,
                        "advantage": (rewards[index] - mean) / (std + 1e-4),
                    })
                    fout.write(json.dumps(row, ensure_ascii=True) + "\n")
        call_index += 1
        return rewards

    logged_format_reward.__name__ = "format_reward"
    return logged_format_reward
