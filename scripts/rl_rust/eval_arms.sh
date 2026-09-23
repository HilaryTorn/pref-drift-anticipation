#!/usr/bin/env bash
# The arm table, defined ONCE and sourced by both rust_rl_eval_setup.sh (which fetches and serves
# the adapters) and rust_rl_eval_pod_driver.sh (which scores them). Keeping it in one file is the
# point: when these two lists lived separately, a serving name and a config model_name could drift
# apart and the run would score the wrong weights under the right label.
#
# Both of those consumers now live in private/rl-rust-pods/ (the RunPod plumbing is not published);
# the bare names above are how they appear ON the pod, where the payload puts everything back under
# rl-rust/. This file stays here because it is the arm table -- the record of which adapter each
# label scored -- and that is a result, not plumbing.
#
# Inputs:  SIZE (4b|9b), ARMSET
# Outputs: ARMS array, each entry  tag:lora_name:config_key:legs:repo_kind:repo_path
#
#   lora_name  MUST equal the config key's model_name -- upstream names its output directory after
#              it, so a mismatch writes one arm's generations into another arm's folder.
#   repo_kind  base = no adapter (served as the bare model)
#              rl   = prism-drift/qwen35-<size>-m0-v4-rust-rl-adapters
#              sft  = prism-drift/qwen35-<size>-m0-v4-phase-1-lcb-sft-adapters
#   legs       comma-separated: multilcb, heldout
#
# CONTAMINATION, and why the sft arms are here anyway: the Rust SFT arms trained on test-verified
# solutions to LCB v1-v5, so on the v345em subset 175 of 222 problems are their own training data
# (joined by lcb_question_id). They are scored here as a deliberate train-set-recall diagnostic and
# as a ceiling line, NEVER as capability, and grading splits contaminated from clean. The RL arms
# are clean on every shard -- their cohort is open-r1/codeforces and these shards are atcoder +
# leetcode only, so the platforms do not intersect.

case "${ARMSET:?armset}" in
  final)
    # The v6 run, 2026-09-11. GRPO's held-out legs were already taken at n=2 that morning.
    ARMS=(
      "grpo-step124:qwen35-${SIZE}-grpo-rust-s124:qwen35-${SIZE}-grpo-rust-step124:multilcb:rl:grpo.rust.s0/checkpoint-124-vllm"
      "dpo-step125:qwen35-${SIZE}-dpo-rust-s125:qwen35-${SIZE}-dpo-rust-step125:multilcb,heldout:rl:dpo.rust.s0/checkpoint-125"
    )
    ;;
  dpo40)
    ARMS=(
      "dpo-step40:qwen35-${SIZE}-dpo-rust-s40:qwen35-${SIZE}-dpo-rust-step40:multilcb:rl:dpo.rust.s0/checkpoint-40"
    )
    ;;
  ppo)
    # PPO's final checkpoint, scored on whatever releases the caller passes. No base cell: both
    # instruments already have one measured on identical problems (v6's stored AWS draw, v345em's
    # same-session draw from 2026-09-12), and re-taking them would only add draw noise.
    ARMS=(
      "ppo-step124:qwen35-${SIZE}-ppo-rust-s124:qwen35-${SIZE}-ppo-rust-step124:multilcb:rl:ppo.rust.s0/checkpoint-124"
    )
    ;;
  ppo-heldout)
    # Held-out ONLY, one arm. Added 2026-09-14 for 4B, the last cell missing from the RL rust grid: 4B PPO already has both multi-LCB instruments from the 2026-09-12 session, so re-running them here would spend an hour to add draw noise to numbers that already exist. `ppo-endpoints` is the wrong armset for this size for the same reason, and because 4B has no step-64 question to answer -- the 4B RM score climbed to 1.69 and stayed, where 9B peaked mid-run and collapsed.
    #
    # No base cell: the 4B held-out base was drawn at n=2 in the 2026-09-11 GRPO session on this same stack (eval_reward 0.0355, termination 0.824) and is what every 4B arm on this instrument is read against.
    #
    # HELDOUT_N=1, the driver default -- matching the 4B DPO arm and both 9B PPO arms rather than the n=2 base. The comparison is paired per prompt, so this costs a little width in the arm's interval and biases nothing.
    ARMS=(
      "ppo-step124:qwen35-${SIZE}-ppo-rust-s124:qwen35-${SIZE}-ppo-rust-step124:heldout:rl:ppo.rust.s0/checkpoint-124"
    )
    ;;
  ppo-endpoints)
    # The two ends of the PPO run, on all three instruments. Added 2026-09-13 for 9B, where the
    # final checkpoint alone is not readable: the RM score peaked mid-run (step 64, +0.47 with
    # termination still 0.82) and then regressed below its own starting point by step 124, where
    # termination had fallen to 0.016. Scoring only step 124 cannot separate "PPO never gained
    # capability" from "PPO gained it and then destroyed it"; scoring both can.
    #
    # step 124 leads because it is the cell that was asked for and the one whose runtime is
    # uncertain -- at 1.6% termination it generates to the cap far more often than a normal arm, so
    # it gets the budget while the most of it is left.
    #
    # No base cell, and unlike the expansion armset that is not an economy: every instrument here
    # already has a 9B base measured on this exact stack (held-out 0.0551 at n=2 from the
    # 2026-09-11 GRPO session, v6 0.120 stored, v345em from the 2026-09-12 same-session draw).
    # Re-taking them would add draw noise to the comparison rather than remove it.
    #
    # Both legs, unlike the `ppo` armset above: held-out is the leg PPO has never had at either
    # size, and it is the only out-of-sample measurement of the objective PPO was actually trained
    # on. It runs at HELDOUT_N=1, matching the DPO arms rather than the GRPO base.
    ARMS=(
      "ppo-step124:qwen35-${SIZE}-ppo-rust-s124:qwen35-${SIZE}-ppo-rust-step124:multilcb,heldout:rl:ppo.rust.s0/checkpoint-124"
      "ppo-step64:qwen35-${SIZE}-ppo-rust-s64:qwen35-${SIZE}-ppo-rust-step64:multilcb,heldout:rl:ppo.rust.s0/checkpoint-64"
    )
    ;;
  expansion)
    # The v345em power run. Base is re-measured in the SAME session because this subset is its own
    # instrument -- nothing here pools with the stored v6 numbers (0.0914 / 0.120 / 0.223).
    # multi-LCB only: held-out is a different instrument and is already taken on v6.
    ARMS=(
      "base:qwen35-${SIZE}-m0-v4-rp:qwen35-${SIZE}-m0-v4-runpod:multilcb:base:"
      "grpo-step124:qwen35-${SIZE}-grpo-rust-s124:qwen35-${SIZE}-grpo-rust-step124:multilcb:rl:grpo.rust.s0/checkpoint-124-vllm"
      "dpo-step125:qwen35-${SIZE}-dpo-rust-s125:qwen35-${SIZE}-dpo-rust-step125:multilcb:rl:dpo.rust.s0/checkpoint-125"
    )
    if [ "$SIZE" = 4b ]; then
      # Step 40 rides this instrument so it lands beside step 125 on identical problems: the
      # question is whether DPO's back half cost capability, and that is an arm-vs-arm comparison.
      ARMS+=("dpo-step40:qwen35-4b-dpo-rust-s40:qwen35-4b-dpo-rust-step40:multilcb:rl:dpo.rust.s0/checkpoint-40")
      # epoch 3 = checkpoint-264 (88 optimizer steps per epoch). Dose chosen per size on v6.
      ARMS+=("sft-ep3:lcb-rust-ep3:qwen35-4b-lcbrust-ep3-aws:multilcb:sft:coding.write.rust/checkpoint-264")
    else
      # 9B's dose is epoch 2 = checkpoint-176. Its model_name carries the size ("lcb-9b-rust-ep2")
      # because a bare "lcb-rust-ep2" once overwrote the 4B output directory -- see config.yaml.
      ARMS+=("sft-ep2:lcb-9b-rust-ep2:qwen35-9b-lcbrust-ep2-aws:multilcb:sft:coding.write.rust/checkpoint-176")
    fi
    ;;
  *) echo "unknown ARMSET: $ARMSET" >&2; return 2 2>/dev/null || exit 2 ;;
esac
