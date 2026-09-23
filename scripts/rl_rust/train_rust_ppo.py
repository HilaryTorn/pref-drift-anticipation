"""Launch PPO on Rust: rebind the verifier for the eval callback, then delegate.

Same pattern as ``train_rust_grpo.py`` / ``train_rust_dpo.py`` and shares their
plumbing. PPO's training loop never executes code -- the reward is the learned
reward model (``--reward_model``, trained on the same pair file DPO uses) -- but
the per-checkpoint DriftCadenceCallback samples the policy on held-out prompts
and grades them, and ``load_ppo_prompt_dataset`` renders prompts through
``render_coding_prompt``. Both resolve at call time, so the same
``install_language`` rebinding makes the arm ask for and grade Rust. Miss it and
the prompts ask for Python while the RM (trained on Rust prompts) scores
off-distribution text, and the reward curve reports a flat 0.0.

Three things ``train_rl.train_ppo`` does not do that this launcher adds. Nothing
in ``rl_training/`` changes:

1. ``PPOConfig`` extras. ``train_ppo`` forwards a fixed set of YAML keys into
   ``PPOConfig``; ``temperature``, ``local_rollout_forward_batch_size``,
   ``cliprange`` and friends are not among them and are silently ignored in the
   YAML. The wrapper below injects them and prints the effective values.
2. Rollout memory AND speed. TRL's ``batch_generation`` asks ``generate`` for
   the float32 logits of every generated token and stacks them -- 2.03 GB per
   2048-token sequence at Qwen3.5's vocab, held TWICE at the stack. That both
   OOMs everything (64 x 2048 is ~260 GB) and forces generation into small
   serial chunks, which is why PPO was projected at $115+/arm. The replacement
   generates sequences ONLY (whole batch in one pass by default) and recovers
   the log-probs with the same forward-pass-over-query_response recipe TRL's
   own ref_logprobs uses -- exactly equivalent, because the rollout sampler is
   temperature-only (top_k 0, top_p 1.0). The trainer receives the log-probs
   in a stand-in it slices as before, unchanged. Tier 0 part A4 proves both
   modes against stock. See ``install_rollout_logprob_fix`` for the details.
3. Preflight (``--preflight``): the RM manifest's held-out accuracy gate that
   ``train_ppo`` hard-enforces, the calibration marker, the RM's max_length
   versus the pairs, the batch/step arithmetic, and the usual toolchain checks,
   all without loading a model or touching a GPU.

Usage mirrors train_rl.py; unrecognized arguments are forwarded verbatim:

    .venv/bin/python rl-rust/train_rust_ppo.py \
        --config rl-rust/configs/ppo_rust.yaml \
        --base_model prism-drift/qwen35-9b-m0-v4 \
        --reward_model runs/qwen35-9b-rm-rust-s0-calibrated \
        --dataset data/rl/v1/cohorts/<cohort>/train.jsonl \
        --eval_dataset rl-rust/out/heldout-500-clean.jsonl \
        --seed 0 --run_name qwen35-9b-ppo-rust-s0 --save_dir runs

``--dataset`` is the frozen 1000-prompt cohort (prompts + tests, same file GRPO
trains on); ``--reward_model`` is the CALIBRATED RM directory from
calibrate_reward_head.py; ``--eval_dataset`` is the held-out prompts+tests file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-rust"))

import yaml  # noqa: E402

from rl_rust import prompts as rust_prompts  # noqa: E402
from rl_rust import rewards as rust_rewards  # noqa: E402
from scripts.rl_rust.train_rust_grpo import _cli_from_config, install_language  # noqa: E402

#: YAML keys injected into PPOConfig by the wrapper below. train_rl.train_ppo
#: does not read them, so listing one here is the ONLY thing that makes it real.
_PPO_CONFIG_EXTRA_KEYS = (
    "temperature",
    "local_rollout_forward_batch_size",
    "num_mini_batches",
    "whiten_rewards",
    "kl_estimator",
    "cliprange",
    "cliprange_value",
    "vf_coef",
    "gamma",
    "lam",
    "missing_eos_penalty",
)

#: Keys train_ppo DOES forward. Listed so preflight can refuse a YAML that sets
#: one of the injected names under a name train_rl would also read, and so a
#: reader can see the split without opening train_rl.py.
_PPO_CONFIG_FORWARDED_KEYS = (
    "max_steps",
    "save_steps",
    "logging_steps",
    "learning_rate",
    "max_grad_norm",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "init_kl_coef",
    "num_ppo_epochs",
    "num_sample_generations",
    "response_length",
    "max_prompt_length",
    "lr_scheduler_type",
    "warmup_steps",
)

#: Written by calibrate_reward_head.py into reward_model_manifest.json.
CALIBRATION_KEY = "score_head_calibration"


def install_ppo_config_extras(cfg: dict) -> dict:
    """Wrap ``trl.experimental.ppo.PPOConfig`` so train_ppo's kwargs gain the extras.

    ``train_ppo`` does ``from trl.experimental.ppo import PPOConfig`` inside the
    function, which resolves the attribute on that module at call time. The
    module is imported here after ``_patch_trl_optional_imports`` has run (the
    caller guarantees the order), so importing it is the same import train_ppo
    would do a moment later.
    """
    import trl.experimental.ppo as ppo_module

    original = ppo_module.PPOConfig
    extras = {key: cfg[key] for key in _PPO_CONFIG_EXTRA_KEYS if key in cfg}

    def patched_ppo_config(**kwargs):
        merged = {**kwargs, **extras}
        print(f"[rl-rust] PPOConfig extras injected: {json.dumps(extras, sort_keys=True)}")
        return original(**merged)

    ppo_module.PPOConfig = patched_ppo_config
    return extras


class PrecomputedLogprobs:
    """What the trainer receives in place of a logits chunk. See install_rollout_logprob_fix."""

    __slots__ = ("logprobs",)

    def __init__(self, logprobs):
        self.logprobs = logprobs


class _LogprobStore:
    """Stands in for TRL's ``padded_logitss``; slicing yields PrecomputedLogprobs."""

    __slots__ = ("logprobs",)

    def __init__(self, logprobs):
        self.logprobs = logprobs

    def __getitem__(self, item):
        return PrecomputedLogprobs(self.logprobs[item])

    @property
    def shape(self):
        return self.logprobs.shape


def install_rollout_logprob_fix(cfg: dict | None = None) -> None:
    """Generate WITHOUT per-token logits; recover log-probs with a forward pass.

    THE COST OF PPO IS THIS FUNCTION'S PREDECESSORS. TRL's stock rollout asks
    ``generate(output_scores=True)`` for the full-vocab float32 logits of every
    generated token, then ``torch.stack``s them -- at Qwen3.5's 248,320 vocab
    that is 2.03 GB per 2048-token sequence held TWICE at the stack (tuple +
    contiguous copy, observed directly: an H200 died asking for 60.62 GiB with
    the 60.62 GiB tuple still live). Those logits force generation into small
    serial chunks, and each chunk runs until its longest sequence finishes, so
    the rollout is 4-8 full-length generation passes per step. That is the
    entire reason the first launch needed a 140 GB card and still projected
    $115+ per arm.

    TRL never needed the logits. It only wants the log-prob of each SAMPLED
    token, and its own ref_logprobs code recovers exactly that three lines
    later with an ordinary forward pass:

        ref_logits = forward(ref, query_response).logits[:, context_length-1:-1]
        ref_logits /= args.temperature + 1e-7
        ref_logprob = selective_log_softmax(ref_logits, response)

    This is EXACTLY equivalent for the policy too, because the rollout sampler
    is ``top_k: 0.0, top_p: 1.0, do_sample: True`` -- temperature-only warping,
    no masking -- and the GenerationConfig temperature is the same
    ``args.temperature + 1e-7`` the ref path divides by. Tier 0 part A4 proves
    the equivalence against the stock path on CPU. (One documented nuance: with
    LoRA dropout active the stock scores carry generation-time dropout noise
    while a recomputation draws fresh noise. TRL's own epoch loop and ref path
    already recompute under fresh dropout, at step 0 the LoRA B-matrix is zero
    so dropout on the adapter path is a no-op, and tier 0 verifies in eval mode
    where the paths are identical.)

    So the replacement:

      1. generates sequences ONLY, in ``generation_chunk_size`` chunks (0 =
         the whole 64-episode batch at once -- generation memory is just
         weights + KV cache, and Qwen3.5's linear-attention layers keep O(1)
         recurrent state rather than per-token KV, so one big pass is cheap);
      2. recovers the log-probs with a no_grad forward in
         ``local_rollout_forward_batch_size`` sub-chunks (bf16, single copy,
         ~1 GB per sequence transient instead of 4 GB fp32 double).

    Generation batch and log-prob batch are DIFFERENT dials now: the first is
    the speed lever, the second bounds the forward transient.

    The trainer is unchanged: it receives a store whose slices already hold the
    log-probs, and the rebound ``selective_log_softmax_token_chunks`` returns
    those directly while deferring to the original for real logits (the PPO
    epoch loop still passes real logits, untouched).
    """
    import torch

    from rl_training import masked_ppo_trainer as masked

    ppo_impl = masked.ppo_impl
    original_batch_generation = ppo_impl.batch_generation
    original_selective = masked.selective_log_softmax_token_chunks
    if getattr(original_batch_generation, "_rl_rust_patched", False):
        return

    gen_chunk_cfg = int((cfg or {}).get("generation_chunk_size", 0))

    def generate_sequences_only(lm_backbone, queries, pad_token_id, generation_config):
        # ppo_impl.generate minus output_scores. Masking and sampling are
        # identical -- output_scores only STORES the logits, it does not change
        # RNG consumption -- so with equal chunking this samples the very same
        # tokens as stock (tier 0 checks that too).
        context_length = queries.shape[1]
        attention_mask = queries != pad_token_id
        input_ids = torch.masked_fill(queries, ~attention_mask, 0)
        output = lm_backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
        )
        return torch.cat((queries, output.sequences[:, context_length:]), dim=1)

    def batch_generation(model, queries, local_rollout_forward_batch_size, pad_token_id, generation_config):
        context_length = queries.shape[1]
        batch_size = queries.shape[0]
        gen_chunk = gen_chunk_cfg or batch_size
        # The exact scalar generation warped with: TRL builds the config with
        # args.temperature + 1e-7 and divides ref logits by the same value.
        temperature = float(generation_config.temperature)

        # Stock batch_generation carries @torch.no_grad(); the call site does
        # not, so the recovery forward must bring its own.
        with torch.no_grad():
            query_responses = []
            for i in range(0, batch_size, gen_chunk):
                query = queries[i : i + gen_chunk]
                query_responses.append(
                    generate_sequences_only(model, query, pad_token_id, generation_config)
                )
                ppo_impl.empty_cache()
            padded_query_responses = ppo_impl.pad(
                query_responses, padding_value=pad_token_id, padding_side="right"
            )
            padded_query_responses = padded_query_responses.view(
                -1, padded_query_responses.shape[-1]
            )[:batch_size]

            logprobs = []
            for i in range(0, batch_size, local_rollout_forward_batch_size):
                query_response = padded_query_responses[i : i + local_rollout_forward_batch_size]
                response = query_response[:, context_length:]
                output = ppo_impl.forward(model, query_response, pad_token_id)
                logits = output.logits[:, context_length - 1 : -1]
                logits /= temperature  # in place on the view, as TRL's ref path does
                logprobs.append(original_selective(logits, response))
                del output, logits
                ppo_impl.empty_cache()
            padded_logprobs = torch.cat(logprobs, 0)
        return padded_query_responses, _LogprobStore(padded_logprobs)

    def selective_log_softmax_token_chunks(logits, labels, **kwargs):
        if isinstance(logits, PrecomputedLogprobs):
            if tuple(logits.logprobs.shape) != tuple(labels.shape):
                raise ValueError(
                    f"precomputed logprobs/labels shape mismatch: "
                    f"{tuple(logits.logprobs.shape)} vs {tuple(labels.shape)}"
                )
            return logits.logprobs
        return original_selective(logits, labels, **kwargs)

    batch_generation._rl_rust_patched = True  # type: ignore[attr-defined]
    batch_generation._rl_rust_original = original_batch_generation  # type: ignore[attr-defined]
    ppo_impl.batch_generation = batch_generation
    masked.selective_log_softmax_token_chunks = selective_log_softmax_token_chunks
    print(f"[rl-rust] PPO rollout: no-logits generation installed "
          f"(gen chunk {'whole batch' if not gen_chunk_cfg else gen_chunk_cfg}; "
          f"log-probs recovered by forward pass, ref-path recipe)")


def install_eval_callback_suppression(cfg: dict) -> bool:
    """Honour ``eval_reward_steps: []`` as "no in-loop verifier eval at all".

    DriftCadenceCallback's own guard is ``if self.eval_steps and step not in
    self.eval_steps: return`` -- an EMPTY set is falsy, so the plain config value
    would evaluate at every single save (31 points at save_steps 4, ~43 h at 9B).
    train_rl.train_ppo also hard-requires --eval_dataset and builds the callback
    unconditionally, so the only place to express "none" is here.

    Returns True when the callback was suppressed.
    """
    if cfg.get("eval_reward_steps") != []:
        return False

    from rl_training import train_rl

    def _no_callback(cfg_, args, tokenizer, dataset_path, output_dir):
        print(
            "[rl-rust] in-loop verifier eval SUPPRESSED (eval_reward_steps: []). "
            f"--eval_dataset {dataset_path} is recorded but not evaluated; measure the "
            "checkpoints offline under vLLM (rl-rust/eval_heldout.py)."
        )
        return None

    train_rl._make_callback = _no_callback
    return True


def install_policy_gradient_checkpointing() -> None:
    """Actually enable gradient checkpointing on the PPO policy.

    MEASURED BUG, 2026-09-11. ``PPOConfig(gradient_checkpointing=True)`` is
    silently inert for the policy: TRL's experimental PPOTrainer overrides
    ``train()`` and never runs transformers' ``_inner_training_loop``, which is
    the only place ``args.gradient_checkpointing`` is applied.
    ``PolicyAndValueWrapper.gradient_checkpointing_enable`` exists but nothing
    calls it, and ``train_rl.train_ppo`` enables checkpointing explicitly on the
    VALUE model only. So the policy ran its backward uncheckpointed.

    Measured on an A100-80 at 4B (one sequence, one backbone, train mode):

        len      ckpt=True   ckpt=False
        1024      20.1 GB      29.1 GB
        2048      20.6 GB      49.1 GB
        3072      21.0 GB      69.5 GB

    Checkpointed cost is flat in sequence length; unchecked it is linear. At
    3072 the policy alone wanted ~61 GB above its weights, which with three
    backbones (26.7 GB) is ~88 GB and OOMs an 80 GB card -- observed twice.

    Note this is train-mode-only behaviour: ``GradientCheckpointingLayer`` skips
    checkpointing under ``eval()``, which is correct (no graph is built) but
    makes the bug invisible to any probe that forgets ``.train()``.

    ``_load_policy`` resolves on the module at call time, so rebinding it here
    reaches the policy train_ppo is about to build, with rl_training untouched.
    """
    from rl_training import train_rl

    original = train_rl._load_policy

    def _load_policy(args):
        model, tokenizer = original(args)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Non-reentrant checkpointing does not strictly need this, but a LoRA
        # policy whose inputs never require grad is the classic way for
        # checkpointing to go quietly inert, and it costs nothing.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        print("[rl-rust] PPO POLICY gradient checkpointing enabled "
              "(TRL never applies PPOConfig.gradient_checkpointing to the policy)")
        return model, tokenizer

    train_rl._load_policy = _load_policy


def read_reward_model_manifest(reward_model: str) -> dict | None:
    path = Path(reward_model) / "reward_model_manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def preflight(cfg: dict, language: str, prompt_format: str, dataset: str | None,
              reward_model: str | None) -> int:
    """Cheap checks that would otherwise fail hours into a booked GPU."""
    ok = True

    def line(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"  [{'ok' if good else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("preflight")
    line("rustc on PATH (eval callback grades here)", rust_rewards.rust_toolchain_available())

    rendered = rust_prompts.format_coding_prompt("A+B", format_name=prompt_format)
    line("prompt asks for Rust", "Rust" in rendered and "Python" not in rendered)

    for key in ("learning_rate", "lr_scheduler_type", "init_kl_coef", "num_ppo_epochs",
                "response_length", "max_prompt_length", "temperature",
                "local_rollout_forward_batch_size"):
        line(f"config sets {key}", key in cfg, str(cfg.get(key)))
    line(
        "scheduler is explicit (unset decays to <2e-7 by step 125)",
        cfg.get("lr_scheduler_type") is not None,
    )
    line("response_length is the Rust cap (2048)", int(cfg.get("response_length", 0)) == 2048,
         str(cfg.get("response_length")))
    line("temperature matches the pair sampling (1.0)", float(cfg.get("temperature", 0)) == 1.0,
         str(cfg.get("temperature")))

    per_device = int(cfg["per_device_train_batch_size"])
    grad_accum = int(cfg["gradient_accumulation_steps"])
    local_batch = per_device * grad_accum
    mini = int(cfg.get("num_mini_batches", 1))
    line("episodes per PPO update >= 64 (the batch-of-8 failure)", local_batch >= 64,
         f"{per_device} x {grad_accum} = {local_batch}")
    line("num_mini_batches is 1 (>1 is a no-op in TRL 1.9)", mini == 1, str(mini))
    line("rollout chunk divides the batch", local_batch % int(cfg["local_rollout_forward_batch_size"]) == 0,
         f"{local_batch} % {cfg['local_rollout_forward_batch_size']}")
    gen_chunk = int(cfg.get("generation_chunk_size", 0))
    line("generation chunk is whole-batch or divides the batch",
         gen_chunk == 0 or local_batch % gen_chunk == 0,
         f"{gen_chunk} ({'whole batch' if gen_chunk == 0 else f'{local_batch} % {gen_chunk}'})")
    steps = int(cfg["max_steps"])
    eval_steps = cfg.get("eval_reward_steps")
    if eval_steps == []:
        line("in-loop verifier eval disabled (measured offline under vLLM)", True,
             "eval_reward_steps: [] -> callback suppressed by install_eval_callback_suppression")
    else:
        line("eval steps lie inside max_steps",
             bool(eval_steps) and all(0 < s <= steps for s in eval_steps), str(eval_steps))
    overlap = sorted(set(_PPO_CONFIG_EXTRA_KEYS) & set(_PPO_CONFIG_FORWARDED_KEYS))
    line("no key is both injected and forwarded", not overlap, str(overlap))

    if dataset:
        path = Path(dataset)
        line("dataset exists", path.is_file(), str(path))
        if path.is_file():
            rows = [json.loads(l) for l in path.open()]
            # The cohort is the 992-row CLEAN one, not the frozen 1000. That is
            # the file GRPO trained on and the file the K=16 pairs -- hence the
            # RM, hence DPO -- were generated from; the pair manifest carries its
            # sha256. Checking the sha against the cohort manifest is what makes
            # "the data is the same across arms" a verified claim rather than a
            # filename convention.
            manifest_path = path.parent / "manifest.json"
            cohort_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            line("cohort is the clean cohort (build_clean_cohort.py), not the raw frozen one",
                 cohort_manifest.get("schema") == "rust_rl_clean_cohort_v1",
                 f"{manifest_path} schema={cohort_manifest.get('schema')!r}")
            line("cohort row count matches its manifest",
                 len(rows) == int(cohort_manifest.get("rows", -1)),
                 f"n={len(rows)}, manifest rows={cohort_manifest.get('rows')}")
            import hashlib
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            line("cohort sha256 matches its manifest (same prompts GRPO/DPO used)",
                 digest == cohort_manifest.get("sha256"),
                 f"{digest[:16]}... vs {str(cohort_manifest.get('sha256'))[:16]}...")
            episodes = steps * local_batch
            line("total episodes are whole passes over the cohort",
                 episodes % len(rows) == 0, f"{steps} x {local_batch} = {episodes} = {episodes / len(rows):g} passes")
            types = {r.get("verifier", {}).get("type") for r in rows}
            line("every verifier is stdin/stdout (eval callback grades these)",
                 types <= {"io_tests", "reference_io_tests"}, str(sorted(t for t in types if t)))

    if reward_model:
        manifest = read_reward_model_manifest(reward_model)
        line("reward model manifest present", manifest is not None, str(Path(reward_model) / "reward_model_manifest.json"))
        if manifest:
            line("manifest schema is reward_model_manifest_v1",
                 manifest.get("schema") == "reward_model_manifest_v1", str(manifest.get("schema")))
            acc = float(manifest.get("held_out_pairwise_accuracy", float("nan")))
            n_eval = int(manifest.get("n_eval", 0))
            line("held-out pairwise accuracy >= 0.60 (train_rl hard gate)", acc >= 0.60,
                 f"{acc:.3f} on {n_eval} pairs (one pair = {100 / n_eval if n_eval else float('nan'):.1f} points)")
            rm_max_len = int(manifest.get("config", {}).get("max_length", 0))
            need = int(cfg["max_prompt_length"]) + int(cfg["response_length"])
            line("RM max_length covers prompt + response", rm_max_len >= need,
                 f"RM {rm_max_len} vs {need}")
            cal = manifest.get(CALIBRATION_KEY)
            line("score head is calibrated (calibrate_reward_head.py)", cal is not None,
                 f"scale={cal['scale']:.4g}, sigma_raw={cal['sigma_raw']:.4g}, n={cal['n_scores']}" if cal else
                 "MISSING -- raw RM logits make kl_coef an unknown strength; see ppo_rust.yaml")
            if cal:
                # Compare AFTER against BEFORE -- both measured inside the same
                # calibration run, on the same batch size. Comparing against the
                # manifest's gate number instead conflates "the rescale changed
                # a ranking" (a real failure) with "training and calibration
                # scored one borderline pair differently under bf16 padding" (a
                # non-event, see calibrate_reward_head.py). The 9B RM reproduced
                # 0.7037 against a 0.7222 manifest -- one pair of 54 -- and the
                # old check would have failed the preflight after calibration
                # had already correctly decided to proceed.
                before = float(cal.get("held_out_pairwise_accuracy_before", acc))
                after = float(cal["held_out_pairwise_accuracy_after"])
                line("calibration preserved every ranking (after == before)",
                     abs(after - before) < 1e-9, f"{after:.4f} vs {before:.4f}")
                drift_pairs = abs(before - acc) * max(n_eval, 1)
                line("calibration reproduced the gate number to within one pair",
                     drift_pairs <= 1.0 + 1e-6,
                     f"recomputed {before:.4f} vs manifest {acc:.4f} = {drift_pairs:.1f} pairs")
            pairs_path = manifest.get("pairs_path", "")
            if pairs_path and Path(pairs_path).is_file():
                with Path(pairs_path).open() as fh:
                    first = json.loads(next(fh))
                line("RM pair prompts ask for Rust (the RM scores the policy's Rust prompts)",
                     "Rust" in first.get("prompt", "") and "Python" not in first.get("prompt", ""), pairs_path)
                sidecar = Path(pairs_path).with_suffix(".language.json")
                if sidecar.is_file():
                    side = json.loads(sidecar.read_text())
                    line("pair-file language sidecar says rust",
                         side.get("verifier_language") == language and side.get("prompt_format") == prompt_format,
                         json.dumps(side, sort_keys=True))
            else:
                print(f"  [warn] RM pair file {pairs_path!r} not present here; cannot confirm it asked for Rust "
                      "(fine if the RM was trained on the pod -- check there)")

    print("\npreflight " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--reward_model", default=None)
    parser.add_argument("--preflight", action="store_true")
    # Both chunks are HARDWARE dials, not scientific settings: they change how
    # many sequences are processed at once, never the 64-episode PPO batch or
    # the numerics (tier 0 part A4). They are size-specific, so they are flags
    # rather than a second copy of the YAML. The effective values are printed
    # and land in run_metadata.json's trainer_config, so overrides stay in the
    # record. --rollout_chunk sizes the LOG-PROB/ref/value/RM forward transient
    # (~1 GB per sequence, bf16); --generation_chunk sizes generation itself
    # (0 = the whole batch in one pass; memory is weights + KV cache only now
    # that generation carries no logits).
    parser.add_argument("--rollout_chunk", type=int, default=None,
                        help="Override local_rollout_forward_batch_size from the config")
    parser.add_argument("--generation_chunk", type=int, default=None,
                        help="Override generation_chunk_size from the config (0 = whole batch)")
    known, rest = parser.parse_known_args()

    cfg = yaml.safe_load(Path(known.config).read_text())
    if known.rollout_chunk is not None:
        was = cfg.get("local_rollout_forward_batch_size")
        cfg["local_rollout_forward_batch_size"] = known.rollout_chunk
        print(f"[rl-rust] local_rollout_forward_batch_size OVERRIDE: {was} -> {known.rollout_chunk} "
              f"(~{known.rollout_chunk * 1.02:.1f} GB bf16 forward-logits transient at 2048 tokens)")
    if known.generation_chunk is not None:
        was = cfg.get("generation_chunk_size")
        cfg["generation_chunk_size"] = known.generation_chunk
        print(f"[rl-rust] generation_chunk_size OVERRIDE: {was} -> {known.generation_chunk} "
              f"({'whole batch in one pass' if known.generation_chunk == 0 else 'sequences per generation pass'})")
    language = cfg.get("verifier_language", "rust")
    prompt_format = cfg.get("prompt_format", "rust_io")

    if cfg.get("algo") != "ppo":
        print(f"this launcher is PPO-only; config says algo={cfg.get('algo')!r}", file=sys.stderr)
        return 2

    if known.preflight:
        return preflight(cfg, language, prompt_format, known.dataset, known.reward_model)

    manifest = read_reward_model_manifest(known.reward_model) if known.reward_model else None
    if manifest is not None and CALIBRATION_KEY not in manifest and "--allow_weak_reward_model" not in rest:
        print(
            "[rl-rust] reward model is NOT calibrated (no score_head_calibration in its manifest). "
            "Run rl-rust/calibrate_reward_head.py first; raw logits make kl_coef an unknown strength. "
            "Pass --allow_weak_reward_model to run a plumbing smoke on an uncalibrated RM.",
            file=sys.stderr,
        )
        return 2

    install_language(language, prompt_format)
    print(f"[rl-rust] verifier language = {language}, prompt format = {prompt_format}")

    from rl_training import train_rl

    # Importing trl.experimental.ppo before this patch runs can trip TRL's eager
    # optional-integration imports; train_ppo calls it again, idempotently.
    train_rl._patch_trl_optional_imports()
    install_ppo_config_extras(cfg)
    install_rollout_logprob_fix(cfg)
    install_policy_gradient_checkpointing()
    install_eval_callback_suppression(cfg)

    argv = ["--algo", "ppo", "--config", known.config]
    if known.dataset:
        argv += ["--dataset", known.dataset]
    if known.reward_model:
        argv += ["--reward_model", known.reward_model]
    argv += rest
    argv += _cli_from_config(cfg, rest)

    sys.argv = ["train_rl.py", *argv]
    train_rl.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
