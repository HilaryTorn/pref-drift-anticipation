#!/usr/bin/env python3
"""Assert the M0 reward ranks completions the way the objective intends.

The reward is a composite of three terms, and composites fail quietly: a weight typo or a sign
slip still produces a plausible-looking curve that climbs while training the wrong behavior. These
checks pin the ORDERING that the objective is supposed to express, so a change that breaks it
fails here instead of after a 4B run.

    python m0/scripts/check_rewards.py

Token counts use the whitespace fallback (no tokenizer), so "tokens" here means words. That is
fine for ordering checks -- the shape of band_score is what is under test, not the tokenizer.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m0.rewards import (  # noqa: E402
    NO_COMMIT_PENALTY, REASONING_BUDGET, REASONING_FLOOR, REASONING_TARGET, band_score,
    score_completion, split_completion,
)

AB = ["A", "B"]


def think(n_words: int, visible: str) -> str:
    """A completion in the shape the model ACTUALLY emits.

    With enable_thinking=True the chat template ends the prompt with an open `<think>` tag, so the
    completion starts inside the reasoning block and carries only the CLOSING tag. Fixtures that
    include an opening tag would exercise a branch production never takes -- which is how the first
    version of split_completion() passed its tests while being wrong on every real completion.
    """
    return " ".join(["word"] * n_words) + "</think>\n" + visible


def score(text, answer="A", verifiable=True):
    return score_completion(text, AB, answer, verifiable)


def main() -> int:
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if not condition:
            failures.append(f"{name}: {detail}")

    # --- split_completion ----------------------------------------------------------------------
    # The production shape: prompt pre-opened the block, so the completion carries only `</think>`.
    reasoning, visible = split_completion("thinking here</think>\nAnswer: A")
    check("split/prompt-opened (production shape)",
          reasoning.strip() == "thinking here" and "Answer: A" in visible,
          f"got {reasoning!r} / {visible!r} -- this is the shape Qwen3.5 actually emits under "
          f"enable_thinking=True; getting it wrong silently turns the band term into a total-length "
          f"measure with no visible symptom")

    reasoning, visible = split_completion("<think>thinking here</think>\nAnswer: A")
    check("split/both tags", reasoning.strip() == "thinking here" and "Answer: A" in visible,
          f"got {reasoning!r} / {visible!r}")

    reasoning, visible = split_completion("<think>ran out of room before closing")
    check("split/unclosed", visible == "",
          "an unclosed <think> must leave NO visible text -- at serve time that response is "
          "unparseable and it must not be able to earn a commit")

    reasoning, visible = split_completion("still thinking, never closed the block")
    check("split/truncated mid-think", visible == "",
          "with the block opened by the prompt and never closed, the generation was truncated "
          "mid-thought and has no visible answer")

    reasoning, visible = split_completion("plain reasoning\nAnswer: A", prompt_opened_think=False)
    check("split/no-tags", visible == reasoning,
          "when nothing opened a block the whole completion is both reasoning and answer surface")

    # --- band shape ----------------------------------------------------------------------------
    check("band/full-marks-from-floor-to-target",
          band_score(REASONING_FLOOR) == 1.0 and band_score(REASONING_TARGET) == 1.0)
    check("band/decays-above-target",
          0 < band_score((REASONING_TARGET + REASONING_BUDGET) // 2) < 1.0,
          "between target and budget a response is servable but verbose, so it scores partial")
    check("band/over-budget-is-zero", band_score(REASONING_BUDGET) == 0.0
          and band_score(REASONING_BUDGET + 1) == 0.0,
          "past the serve-time thinking budget the tail does not exist in production")
    check("band/zero-reasoning-is-not-zero-band", 0 < band_score(0) < 1.0,
          f"a terse commit still parses; got {band_score(0)}")
    check("band/monotone-below-floor", band_score(0) < band_score(REASONING_FLOOR // 2) < band_score(REASONING_FLOOR))

    # --- the property that makes GRPO able to learn at all --------------------------------------
    # MEASURED 2026-07-21: with a band that was flat all the way to the budget, the smoke run's two
    # completions (788 and 1018 reasoning tokens) both scored exactly 1.0. Identical rewards inside
    # a group means zero advantage, which means zero gradient -- the run reported grad_norm 0 and
    # frac_reward_zero_std 1 on every step and trained nothing at all. The slope is what separates
    # two acceptable-but-different-length completions, so a group has something to learn from.
    shorter = score(think(400, "Answer: A"))
    longer = score(think(1200, "Answer: A"))
    check("gradient/length differences produce reward differences",
          shorter["reward"] > longer["reward"],
          f"400-token reasoning scored {shorter['reward']:.3f} and 1200-token scored "
          f"{longer['reward']:.3f}. If these tie, every acceptable completion ties, group variance "
          f"collapses to zero, and GRPO has no gradient.")

    # --- the ordering the objective exists to express -------------------------------------------
    # The achievable degenerate policy is NOT a completion with no tags at all -- under
    # enable_thinking=True that reads as "never closed the block", which is unparseable anyway and
    # scores badly for an unrelated reason. It is: close the think block immediately, emit the
    # answer. That is the strategy the anti-collapse guarantee has to beat, so it is the one the
    # fixture must represent.
    bare = score(think(0, "Answer: A"))
    in_band = score(think(200, "Answer: A"))
    in_band_wrong = score(think(200, "Answer: B"))
    over_budget = score(think(REASONING_BUDGET + 200, "Answer: A"))
    never = score(think(200, "I am still weighing the two options and have not decided"))
    truncated = score("reasoning that never closed and never reached an answer")

    check("order/in-band beats bare", in_band["reward"] > bare["reward"],
          f"{in_band['reward']} vs {bare['reward']} -- a zero-reasoning commit must not tie the target")
    check("order/correct beats incorrect", in_band["reward"] > in_band_wrong["reward"],
          "the correctness term is what makes the reasoning load-bearing")
    check("order/in-band beats over-budget", in_band["reward"] > over_budget["reward"],
          f"{in_band['reward']} vs {over_budget['reward']}")
    check("order/over-budget beats never-committing", over_budget["reward"] > never["reward"],
          f"{over_budget['reward']} vs {never['reward']} -- an over-budget response that DID commit "
          f"still parses at serve time and must not rank with one that never answered")
    check("order/never-committing is the floor", never["reward"] == -NO_COMMIT_PENALTY == truncated["reward"],
          f"never={never['reward']} truncated={truncated['reward']}")
    check("order/truncated cannot commit", truncated["committed"] is None)

    # --- the anti-collapse guarantee ------------------------------------------------------------
    # A single lucky bare guess CAN outscore a reasoned-but-wrong answer, which is fine and even
    # correct -- the item was answered right. What must never hold is the policy-level version: if
    # guessing without reasoning paid better ON AVERAGE than reasoning does, gradient descent would
    # find that, and M0 would install exactly the no-reasoning behavior it exists to remove. So
    # compare strategies in expectation, not single completions.
    bare_wrong = score(think(0, "Answer: B"))
    guess_ev = 0.5 * bare["reward"] + 0.5 * bare_wrong["reward"]
    # Reasoning is credited at CHANCE accuracy here: the guarantee has to hold even for a model
    # whose reasoning is worthless, otherwise the incentive depends on the model already being good.
    reason_ev = 0.5 * in_band["reward"] + 0.5 * in_band_wrong["reward"]
    check("anti-collapse/reasoning wins in expectation", reason_ev > guess_ev,
          f"reasoning EV={reason_ev:.3f} vs bare-guess EV={guess_ev:.3f}. If guessing paid better "
          f"on average, training would converge on the bare 'Answer: A' that M0 exists to fix.")

    # --- parser-driven rejections ---------------------------------------------------------------
    narrated = score(think(200, "Answer: A is my choice because it is heavier."))
    check("parser/trailing narration is not a commitment", narrated["committed"] is None,
          "the scoring parser requires a BARE answer line; the reward must agree or M0 optimizes "
          "a commit criterion the batteries do not credit")

    draft = score(think(200, 'Answer: A\nWait, on reflection the other one is heavier.'))
    check("parser/revised draft is not a commitment", draft["committed"] is None)

    # --- neutral items --------------------------------------------------------------------------
    neutral = score_completion(think(200, "Answer: A"), AB, None, False)
    verifiable_right = score(think(200, "Answer: A"))
    check("neutral/scale matches a correct verifiable item",
          abs(neutral["reward"] - verifiable_right["reward"]) < 1e-9,
          f"{neutral['reward']} vs {verifiable_right['reward']} -- the correctness weight should "
          f"fold into commit when there is no ground truth")
    neutral_other = score_completion(think(200, "Answer: B"), AB, None, False)
    check("neutral/no preferred label", neutral["reward"] == neutral_other["reward"],
          "a neutral item must not reward one option over the other")

    # --- report ----------------------------------------------------------------------------------
    print("reward ladder (verifiable item, correct answer = A):")
    for name, row in [
        ("in-band + correct", in_band), ("in-band + wrong", in_band_wrong),
        ("bare commit (no reasoning)", bare), ("over budget + correct", over_budget),
        ("reasoned, never committed", never), ("truncated mid-think", truncated),
    ]:
        print(f"  {name:28s} reward={row['reward']:+.3f}  commit={row['commit']:.0f} "
              f"band={row['band']:.2f} correct={row['correct']:.0f} n={row['n_reasoning_tokens']}")

    if failures:
        print(f"\nFAILED: {len(failures)} check(s)\n")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nall reward checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
