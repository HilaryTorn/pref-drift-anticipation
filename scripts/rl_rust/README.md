# rl-rust — GRPO on Rust

Self-contained tree for running the RL arms against **Rust** instead of Python. Nothing in `rl_training/` is modified; the dependency is one-way and read-only, so the Python pipeline still runs on its own code.

Status as of 2026-09-10: **built, tier-0 verified, never run on a GPU.**

## Why Rust

The paper's limitation section says nobody ran RL on a target with headroom. Rust is the target with headroom, and it is the only language that supports the comparison at every size.

| language | 4B base → SFT | 9B base → SFT | 27B base → SFT |
|---|---|---|---|
| **Rust** | 0.091 → **0.223** | 0.120 → **0.223** | 38.3 → **0.480** |
| Go | 0.143 → 0.189 | 0.257 → *no arm* | *no arm* |
| C# | 0.286 → 0.246 ↓ | 0.240 → *no arm* | *no arm* |
| PHP | 0.246 → 0.211 ↓ | 0.274 → *no arm* | *no arm* |

(multi-LCB v6, n=175, pass@1.) Rust has the lowest baseline, is the only language with SFT arms at 4B/9B/27B, and the only one where SFT moved capability *up* at every size. C# and PHP both went down, which is why their nulls are weak.

## The data question

**No new dataset is needed.** GRPO needs tests, not solutions — it generates its own candidates and grades them. The frozen Nemotron cohort (N=1000 per size) verifies via `io_tests`: stdin in, stdout compared. That is language-agnostic. A Rust program passes exactly the tests the Python one does.

Verified on the real 9B cohort: 1000/1000 rows are `io_tests`, all `comparison: tokens`.

The 700-row Rust SFT pool is *not* the RL data. It is `messages` — problem → verified teacher solution — which is what the SFT arms trained on. Using it for RL would run RL on the exact problems SFT trained on, confounding the comparison this run exists to make, and 700 would fail the exact-1000 cohort check. It stays where it is, as the SFT arm you compare against.

## Layout

```
rl_rust/languages.py   executor + build cache (the only genuinely new code)
rl_rust/rewards.py     language-parameterized reward; imports comparison
                       semantics from rl_training so the arms cannot diverge
rl_rust/prompts.py     rust_io prompt
configs/grpo_rust.yaml hyperparameters, copied from the run that moved
train_rust_grpo.py     launcher: rebinds language, injects LoRA argv, preflight
tests/tier0_conformance.py   proves the reward before any GPU spend
```

The RunPod plumbing lives in `private/rl-rust-pods/` (gitignored) rather than here: the pod drivers, the dead-man switches, the on-pod upload scripts and `run_lcb_expansion.py`, the laptop-side launcher. It is venue-specific — how these runs were rented and babysat, not what they measured — so it is not part of what this directory publishes. The payload builder still unpacks all of it back under `rl-rust/` on the pod, so every path in the sections below is correct as seen from there.

### Two design decisions worth knowing

**The reward semantics are imported, not copied.** `_outputs_match`, `_normalize`, `_subsample_indices` and the truncation classifier come from `rl_training.rewards`. If the Rust arm and the Python arm disagreed about what counts as correct output, the language comparison would be confounded and the run wasted. Only the executor is new.

**Compile once per program, not per test case.** `_io_case_passes` runs per test, so a naive swap would invoke `rustc` 12 times per completion. Builds are cached by `(language, sha256(source))` and shared across a completion's tests and across duplicate completions in a GRPO group. Measured: 0.41s to grade a program over 12 test cases, against a ~343s GRPO step — under 1%. Build artifacts are cached; run directories are still fresh per test case, so a program that writes files cannot leak state into the next case.

## Run it

Tier 0 first, always. It costs nothing and it is the only thing standing between you and a booked GPU grading everything as zero:

```bash
.venv/bin/python rl-rust/tests/tier0_conformance.py
```

Currently **53/53**. It covers: correct / wrong / compile error / panic / timeout / partial credit, all four comparison modes, truncation returning `None` rather than `0.0`, tagged-block extraction, compile-once, Python-path parity against `rl_training.rewards`, and 40 real GPT-5.6 teacher solutions from the Rust SFT pool joined to their LiveCodeBench test cases (mean reward 0.917, 36/40 perfect).

Preflight against the cohort you intend to train on:

```bash
.venv/bin/python rl-rust/train_rust_grpo.py \
    --config rl-rust/configs/grpo_rust.yaml \
    --dataset data/rl/v1/cohorts/<cohort>/train.jsonl \
    --preflight
```

Then train (arguments forwarded to `train_rl.py`):

```bash
.venv/bin/python rl-rust/train_rust_grpo.py \
    --config rl-rust/configs/grpo_rust.yaml \
    --base_model prism-drift/qwen35-4b-m0-v4 \
    --dataset data/rl/v1/cohorts/<4b-cohort>/train.jsonl \
    --eval_dataset <held-out>.jsonl \
    --seed 0 --run_name qwen35-4b-grpo-rust-s0 --save_dir runs
```

Cohorts are model-specific — 4B and 9B share only 135 of 1000 ids. Match the cohort to the model.

## Hyperparameters, and why

Copied from `qwen35-9b-grpo-m0-v4-truncfix-k8-len1024-lr1e5-step125-s0` (2026-08-30), the **only run in this project where reward moved**: held-out `eval_reward` 0.275 → 0.358 with termination rate flat, so it survives truncation-conditioning.

Deliberately **not** copied from `RL-Repaired-V1` (2026-09-07), whose task reward is flat. Three settings differ and all three push the same way:

| | truncfix (moved) | repaired-v1 (flat) |
|---|---|---|
| `learning_rate` | 1e-5 | 5e-7 / 7.5e-7 |
| `lr_scheduler_type` | `constant_with_warmup` | unset → HF linear decay |
| `beta` | 0.05 | 0.1 / 0.075 |

The unset scheduler is the quiet one: `rl_training/configs/grpo.yaml` warns that with HF's default "the back half of training ran at <2e-7". So repaired-v1's effective rate was *below* its nominal 5e-7 for half the run, while `beta` doubling actively held the policy at the reference. Its KL pinned at 2–3e-4 is the visible consequence. **The flat reward is over-determined — do not fix one of these and expect movement.**

Two departures from truncfix:

- `max_completion_length` 1024 → **2048**. Rust is more verbose than Python and `mask_truncated_completions` *drops* over-long rollouts rather than scoring them, so a tight cap shrinks the effective batch and can zero the gradient. Watch `completions/clipped_ratio`; above ~0.4, raise it rather than interpret the run.
- `lora_target_modules` narrow → **all-linear**. Every flat run failed with KL near 1e-3 — the policy could not move — which argues for more adapted parameters, not fewer. The narrow list also never names Qwen3.5's linear-attention projections, where 18 of 24 layers at 9B do their attention. This combination (all-linear at 1e-5) is **untested**: watch `kl` in the first 20 steps at 4B, and if it blows past ~0.05, drop to 5e-6 before scaling up.

## DPO

Wired 2026-09-10, same pattern as GRPO: `build_rust_dpo_pairs.py` rebinds `score_candidates` + `render_coding_prompt` and delegates to `rl_training/build_dpo_pairs.py`; `train_rust_dpo.py` rebinds the eval verifier and delegates to `train_rl.py --algo dpo`. Nothing in `rl_training/` is modified.

**Pairs generate from M0, not from GRPO checkpoints** (DPO is the offline arm; its data comes from the starting policy). K=8, margin 0.5, `--max_new_tokens 2048`, truncation-aware pair selection — measured off the GRPO rollouts that yields ~450 pairs at 9B and ~294 at 4B from 992 prompts. Sampling can hit a vLLM endpoint serving M0; **grading executes Rust on the machine running the builder**, so it needs `rustc` there (the Mac works).

```bash
# tier 0 (now covers the DPO candidate path), then both preflights
.venv/bin/python rl-rust/tests/tier0_conformance.py
.venv/bin/python rl-rust/build_rust_dpo_pairs.py --preflight --prompts rl-rust/out/cohort-9b-clean-n992/train.jsonl
.venv/bin/python rl-rust/train_rust_dpo.py --config rl-rust/configs/dpo_rust.yaml --preflight --dataset rl-rust/out/dpo-pairs-9b/pairs.jsonl

# build pairs (endpoint mode), then train
.venv/bin/python rl-rust/build_rust_dpo_pairs.py \
    --model prism-drift/qwen35-9b-m0-v4 --endpoint http://<host>:8000/v1 \
    --prompts rl-rust/out/cohort-9b-clean-n992/train.jsonl \
    --out rl-rust/out/dpo-pairs-9b/pairs.jsonl \
    --scores_out rl-rust/out/dpo-pairs-9b/prompt_scores.jsonl \
    --manifest_out rl-rust/out/dpo-pairs-9b/manifest.json \
    --K 8 --margin 0.5 --resume
.venv/bin/python rl-rust/train_rust_dpo.py \
    --config rl-rust/configs/dpo_rust.yaml \
    --base_model prism-drift/qwen35-9b-m0-v4 \
    --dataset rl-rust/out/dpo-pairs-9b/pairs.jsonl \
    --eval_dataset rl-rust/out/heldout-500.jsonl \
    --seed 0 --run_name qwen35-9b-dpo-rust-s0 --save_dir runs
```

**The loss is pure DPO (`loss_type: [sigmoid]`)** — decided 2026-09-10. The flat Python DPO arms ran at the repaired-v1 settings triple, so plain DPO has never been tried at the fixed hyperparameters and gets the first shot; the paper keeps textbook DPO. The known risk is likelihood displacement (Razin et al. 2410.08847; Pang et al. 2404.19733): margins grow while `logps/chosen` sinks and the model gets no better at producing the good program. Watch for exactly that signature — the fallback (DPO+NLL, `loss_type: [sigmoid, sft]` + `loss_weights: [1.0, 1.0]`) is one config line, reuses the same pairs, and a rerun costs ~2h of GPU. The upstream >10% truncation warning at pair-build time is expected for Rust; do **not** raise the cap in response (wrong completions need 11k–16k tokens; there is nothing between 2048 and that).

## PPO

**Textbook PPO, with the learned reward model** — decided 2026-09-11, on the same principle as the DPO call above: the paper keeps the canonical method. PPO as everyone practices it (InstructGPT and everything downstream of it) is policy + learned critic + a reward model trained on preference pairs + KL to a frozen reference. Pointing PPO at the `rustc` verifier instead would build something nearer to GRPO-with-a-critic than to PPO, and a four-arm method comparison only means anything if each arm is the method a reader recognizes.

Nothing in `rl_training/` needs to change for this. `configs/ppo.yaml` and `configs/reward_model.yaml` already implement it, and the RM trains on **the same pairs file DPO trains on**, so the completed sweep already covers PPO's data at no extra generation cost — a few epochs over 343 pairs (4B) / 538 (9B) from the K=16 sweep is minutes of GPU. `train_rl.py` hard-refuses an RM below 0.60 held-out pairwise accuracy; `--allow_weak_reward_model` is smoke-only and never formal.

**The learning rate does not carry over from GRPO.** `ppo.yaml` sits at 1e-5 because that is the truncfix rate, but PPO is the one arm that already *ran* at 1e-5 under repaired-v1, and it diverged: 4B RM score fell 8.56 → 6.66 (t = −6.5) while KL went 4.46 → 22.17. 9B PPO at 5e-7 was the tame one. The "1e-5 is the rate that moved" argument is GRPO's evidence, not PPO's; for PPO the same number is the rate with a failure already attached. Watch `kl` from step 1 and drop to 5e-6 at the first sign of the repaired-v1 shape. (PPO's logged KL is on a different scale from GRPO's — a sequence sum against a per-token mean — so do not read 22.17 against GRPO's 0.0016 directly; confirm the definition before citing either.)

**The RM accuracy gate is noisy at this pair count, and that is a cost of the textbook route rather than a bug to engineer around.** `eval_fraction: 0.1` leaves 26 held-out pairs at 4B, where 16/26 clears the bar and one pair is worth 3.8 points: an RM at exactly 0.60 true accuracy passes 52% of the time, a genuinely good 0.70 RM still fails 13%, and a coin-flip RM at 0.50 sneaks through 16% of the time. At 9B (40 held-out) those are 57% / 6% / 13%. Raising `eval_fraction` costs training pairs that are already scarce. Train the RM at two or three seeds — minutes each — and read the spread instead of one draw. If it still fails, textbook says fix the pairs (more of them, or a wider margin), not swap the objective.

**Wired 2026-09-11**: `configs/reward_model_rust.yaml`, `calibrate_reward_head.py`, `configs/ppo_rust.yaml`, `train_rust_ppo.py`; tier 0 part A4 covers the plumbing on a tiny random model. Nothing in `rl_training/` changed. The launcher is the same shape as the DPO one — rebind the verifier for the eval callback and the prompt renderer for the cohort, inject the LoRA block into argv, wrap the TRL config class to deliver the YAML keys `train_ppo` does not forward — plus one patch the others do not need, described below.

**What was actually wrong with the Python PPO arms, and what this config does about it.** Both RMs scored 0.90/0.91 held-out, so the reward model was never the binding constraint. Two things were. First, there was no effective KL leash: TRL adds the raw RM logit at the last token and subtracts `kl_coef × kl` per token, and the raw logits sat at 5–9 with whatever spread training left them, so `kl_coef: 0.05` was some unknown fraction of the strength the literature calibrates it for. Advantage whitening does not repair this — it normalises the *sum* of the score and KL terms, so the ratio between them, which is the leash, survives. `calibrate_reward_head.py` scores every pair completion with the trained RM, takes the standard deviation, and divides the score head's weight by it. That is a positive rescale: rankings and held-out accuracy are unchanged (the script recomputes the gate number and refuses to save if it moved), the critic inherits the scale because it is initialised from the same checkpoint, and the printed factor is the size of the correction. Qwen3.5's head has no bias, so the mean is not shifted; with `gamma: 1.0` a constant per-episode reward is absorbed by the value baseline and the mean-shifting advantage whitening, so only the scale reaches the gradient. Second, the rollout batch was 8: `per_device 1 × grad_accum 8` gave 8 episodes per PPO update, sitting on TRL's whitening floor, with 4 epochs of optimisation fitted to those 8 sequences each step. Canonical RLHF PPO updates on 64–512 episodes. `ppo_rust.yaml` sets `gradient_accumulation_steps: 64`, which `train_rl` turns into 8000 episodes across exactly 125 steps — 8 passes over the 1000-prompt cohort, the same 8000 completions GRPO sees — so the elicitation checkpoints at 4/12/40/125 stay where the other arms have them.

Batch 64 does not fit stock TRL. `batch_generation` stacks the float32 generation logits for the whole rollout batch (batch × response_length × vocab) before the trainer computes log-probs chunk by chunk; at 8 × 1024 that was ~5 GB, at 64 × 2048 it is ~80 GB. `train_rust_ppo.py` rebinds it with a version that generates chunk by chunk exactly as TRL does, computes each chunk's log-probs immediately with the trainer's own function, and hands the unchanged trainer a stand-in it slices as before. Tier 0 checks the numbers are identical to the stock path, ragged last chunk and early-stopping chunks included. The per-chunk transient is unchanged, so `local_rollout_forward_batch_size: 8` is the memory dial (~1 GB per sequence at 3072 tokens); it is also the main speed dial, because nearly every chunk of 8 contains one 2048-token runaway and generation time scales with 64/chunk.

Everything else is left at TRL's defaults and written out in the YAML so the effective config is in the file: `num_ppo_epochs 4`, `cliprange 0.2`, `cliprange_value 0.2`, `vf_coef 0.1` (the lm-human-preferences number; 0.5 is Atari), `gamma 1.0`, `lam 0.95`, `kl_estimator k1`, `num_mini_batches 1` (more than one is a no-op in TRL 1.9), `whiten_rewards false` (the calibration does that job; batch-whitening would rescale the KL term with it). `temperature` is 1.0, not TRL's 0.7, because the pairs the RM learned from were sampled at 1.0. `response_length` is 2048 for the same reason as GRPO's cap. LR stays 1e-5: the two fixes change what the LR was fighting, not the LR, and it is the one dial pre-authorised to move — the kill criteria are in the YAML. The reward-model recipe differs from the Python one in three ways forced by Rust and the pod: `max_length 3072` (Rust pairs run to prompt 1024 + completion 2048, and the trainer hard-fails rather than truncate), three epochs instead of one (343–538 pairs at effective batch 32 is 10–17 optimizer steps in one epoch, against the ~28 the Python RMs had over 1000 pairs; the held-out gate is the overfitting check), and LoRA at both sizes, merged on save (a full 9B fine-tune at 3072 tokens does not fit an A100-80; 4B could go full but the two Rust RMs are kept on one recipe). The merge is the stock trainer's behaviour, not a choice made here: `train_rl.py --algo ppo` loads the RM with `AutoModelForSequenceClassification` and has no adapter path, and the RM is a consumable — reward and critic init, never served or elicited — so, unlike the policy adapters, nothing downstream needs it as base + adapter. **Matching the Python RM recipe was never a goal.** The Python PPO arms did not learn, so their settings carry no evidence; the Rust arm keeps the Python wiring and otherwise picks whatever gives PPO the best chance of moving Rust — this applies to the RM recipe, the batch, the calibration, and the LR dial above.

**The dataset is the 992-row CLEAN cohort, not the frozen 1000** (corrected 2026-09-11, at launch). `rl-rust/out/cohort-<size>-clean-n992/train.jsonl` is what GRPO trained on and what the K=16 pairs — hence the RM, hence DPO — were generated from; the pair manifests carry its sha256. Earlier drafts of this section and of `train_rust_ppo.py --preflight` said 1000, which would have put PPO on different prompts from the other three arms. `max_steps` is 124 to match: 124 × 64 = 7936 episodes = exactly 8 passes, and the elicitation checkpoints land on GRPO's own 4/12/40/124. The preflight now checks the cohort's sha256 against its manifest rather than counting rows.

`private/rl-rust-pods/ppo_pod_driver.sh <size>` runs the whole thing on the pod, in two stages with the gate between them (the pod-side plumbing is not in this directory — see [Layout](#layout)):

```bash
# stage A, minutes: RM at seeds 0/1/2 -> best -> calibrate -> preflight.
#   Stops here and kills the pod if no seed clears 0.60. Losing seeds' weights
#   are pruned as the sweep runs (a merged 9B RM is ~18 GB), manifests kept.
# stage B: waits for /root/ppo.go, then trains. A human reads the seed sweep
#   before the 30-hour run starts.
```

The individual commands, if you are driving it by hand:

```bash
# 1. reward model (minutes); repeat at --seed 1 2 and keep the best held-out accuracy
.venv/bin/python rl_training/train_reward_model.py \
    --config rl-rust/configs/reward_model_rust.yaml \
    --base_model prism-drift/qwen35-9b-m0-v4 \
    --pairs rl-rust/out/dpo-pairs-9b-k16/pairs.jsonl \
    --output_dir runs/qwen35-9b-rm-rust-s0 --seed 0
# 2. read the sweep and enforce the gate
.venv/bin/python rl-rust/rm_gate_select.py --runs runs/qwen35-9b-rm-rust-s{0,1,2}
# 3. calibrate (~2 min); prints the scale factor and re-verifies the gate number
.venv/bin/python rl-rust/calibrate_reward_head.py \
    --reward_model runs/qwen35-9b-rm-rust-s0 --out runs/qwen35-9b-rm-rust-s0-calibrated
# 4. preflight, then train
.venv/bin/python rl-rust/train_rust_ppo.py --config rl-rust/configs/ppo_rust.yaml \
    --dataset rl-rust/out/cohort-9b-clean-n992/train.jsonl \
    --reward_model runs/qwen35-9b-rm-rust-s0-calibrated --preflight
.venv/bin/python rl-rust/train_rust_ppo.py --config rl-rust/configs/ppo_rust.yaml \
    --base_model prism-drift/qwen35-9b-m0-v4 \
    --dataset rl-rust/out/cohort-9b-clean-n992/train.jsonl \
    --reward_model runs/qwen35-9b-rm-rust-s0-calibrated \
    --eval_dataset rl-rust/out/heldout-500-clean.jsonl \
    --seed 0 --run_name qwen35-9b-ppo-rust-s0 --save_dir runs
```

The launcher refuses an uncalibrated RM (no `score_head_calibration` block in its manifest) unless `--allow_weak_reward_model` is passed, which is smoke-only. There is no separate smoke config: launch the real run, read the first 20 steps (~$5), and let it continue if `objective/kl` is behaving.

**No in-loop verifier eval** (`eval_reward_steps: []`, 2026-09-11). Measured on the DPO pods: `DriftCadenceCallback` runs HF `model.generate` one prompt at a time at 104 s/prompt (4B) and 156 s/prompt (9B), so each eval point is ~55 min / ~83 min — four of them would cost more GPU than the training. It also is not a number anything else is on: both GRPO arms logged `eval_reward: 0.0` at every step because `heldout-500.jsonl` carried the Nemotron "use python programming language only" wrapper, so every arm has to be re-measured offline against `heldout-500-clean.jsonl` anyway. `save_steps: 4` means the whole 4/12/40/124 trajectory is recoverable there, under vLLM, on one instrument with GRPO and DPO. Note that an empty list does **not** disable the callback by itself — `DriftCadenceCallback` guards with `if self.eval_steps and ...`, so empty is falsy and it would fire at every save; `train_rust_ppo.install_eval_callback_suppression` is what makes `[]` mean none. `--eval_dataset` is still required by `train_rl.train_ppo` and is still recorded in `run_metadata.json`.

**Time and cost, optimised for money rather than wall-clock.** PPO in TRL's experimental trainer generates with plain HF `generate` — there is no vLLM path, unlike GRPO — and Rust's runaway completions make almost every chunk of 8 run the full 2048 steps, so the run is generation-bound: on the order of 25–40 h at 9B on an A100 ($30–50) and roughly half that at 4B, plus ~$1 of RM training and ~$55 of elicitation. An H100 would be about 2× faster at about 2× the price, so it saves nothing; stay on the $1.19 A100 and run 4B and 9B on two pods at the same time, which halves the calendar time for the same dollars. The two free speed levers are `local_rollout_forward_batch_size` (raise to 16 at 4B if `nvidia-smi` shows >20 GB free during step 1) and making sure the pod has the `fla` / `causal_conv1d` kernels installed, without which Qwen3.5's 18 linear-attention layers generate on a slow fallback. Memory at 9B on an 80 GB card is the open risk — three 9B backbones (policy, critic, RM) are ~54 GB before activations — and if step 1 OOMs the answer is the H200, not a smaller batch.

## Cost

GRPO only, three sizes, live RunPod on-demand pricing:

| | hours | GPU | cost |
|---|---|---|---|
| 4B | 12 *(measured)* | A100 $1.19 | $14 |
| 9B | 12 *(measured)* | A100 $1.19 | $14 |
| 27B | ~30 *(est)* | H200 $3.59 | $108 |

≈ **$190** with a 40% margin, plus ≈ **$55** for elicitation at steps 0/4/12/40/125 with a same-session baseline replicate. Call it **$300**.

PPO (4B and 9B only, no 27B) is generation-bound with HF `generate` and no vLLM, so it is slower per step than GRPO despite the same 8000 completions:

| | hours | GPU | cost |
|---|---|---|---|
| 4B RM sweep (3 seeds) + calibrate | ~0.5 | A100 $1.59 | ~$1 |
| 4B PPO | ~15 *(est)* | A100 $1.59 | ~$24 |
| 9B RM sweep (3 seeds) + calibrate | ~1 | A100 $1.59 | ~$2 |
| 9B PPO | ~30 *(est)* | A100 $1.59 | ~$48 |

≈ **$75** plus ≈ **$55** elicitation, so ≈ **$130** with margin. On-demand A100-80 in Secure Cloud is **$1.59/h**, not the $1.19 an earlier draft of this table used — that is the community-tier price and this project does not use community or spot. Run both sizes on two pods at once; that is the same money in half the calendar time. If 9B OOMs on the A100-80 at step 1, the H200 at $4.59 secure makes the 9B row ~$90–140, which is the point at which the 9B arm is worth re-deciding rather than re-launching.

The RM sweep is the gate that makes the rest optional: it costs ~$3 across both sizes and it is the only step that can rule PPO out, so it runs first and the pods hold at `/root/ppo.go` until someone has read it.

Run the tiers in order — $0, then ~$1 on `grpo_smoke.yaml`, then 4B at $14, then 9B at $14, and only then 27B at $108. You can fail 4B five times and stay inside the margin. The only expensive mistake is one carried into 27B.

## Known gaps

- **Never run on a GPU.** Everything here is verified on CPU against stored data.
- **PPO has open GPU-only questions.** Pairs exist (K=16: 343 at 4B, 538 at 9B under `out/dpo-pairs-*-k16`, the files DPO trains on), the RM recipe, calibration and launcher are wired and CPU-verified. Two things are still unverified on hardware: whether three 9B backbones plus a chunk of 8 rollouts fit an A100-80 (if step 1 OOMs the answer is the H200, not a smaller batch), and `local_rollout_forward_batch_size: 16` at 4B, which is the main speed lever and would roughly halve the 4B run. Both are launched at the conservative setting; raising the 4B chunk needs a restart, so decide it off step 1's `nvidia-smi`.
- **PPO trains with the `fla` fast path; GRPO and DPO trained on the torch fallback.** Reversed 2026-09-11 after measurement: the fallback's fp32 chunked scans made each PPO fwd+bwd 4.3x slower (4217 -> 970 ms per 3072-token micro-step on an A100), and with 256 of them per step the arm projected ~$150 per size -- the fallback was never about generation, whose decode is O(1) either way (47 vs 40 ms/step, measured). Install is `fla-core==0.5.2` + `flash-linear-attention==0.5.2` + `einops`, all `--no-deps` so torch stays pinned; **installing `flash-linear-attention` alone half-populates the `fla` namespace and breaks the `qwen3_5` import entirely** (the kernels live in `fla-core` -- that mistake cost a pod). This is a kernel-level numeric difference between the arms' training stacks, not their data, objective, or elicitation; every arm is still scored offline on the same vLLM instrument. Declared here rather than hidden in a pod log.
- **1.3% of the cohort is float-formatting sensitive.** 13 of 1000 rows have high-precision decimal expected outputs graded under exact-token match, so achievable reward is capped near 0.987. This affects Python and Rust identically and is not worth engineering around, but do not read a ceiling below 1.0 as a bug.
- **27B VRAM for all-linear is unchecked.** More adapter params and optimizer state than the narrow list. Confirm it fits before booking the long run.
- **Anticipation and training-preference batteries have no Rust baseline.** Both would need a baseline run before the trained arms are interpretable.
