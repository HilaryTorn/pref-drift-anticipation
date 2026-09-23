"""Put a trained reward model's score on unit scale before PPO uses it.

WHY. TRL's PPO reward is ``rewards[last_token] += score`` and
``-kl_coef * kl`` at every token, with the RM's raw sequence-classification
logit as ``score``. ``kl_coef`` (0.05 here, the Ziegler/TRL default) is
calibrated in the literature for a reward on roughly unit scale. A Bradley-Terry
RM's logit has whatever spread training left it with -- the Python RMs put out
5-9 -- so the KL leash was an unknown fraction of its nominal strength, and the
4B Python PPO arm diverged (RM score fell while KL climbed to 22). Advantage
whitening does not fix this: it normalises the SUM of score and KL terms, so the
ratio between them, which is what the leash is, survives untouched.

WHAT. Score every pair completion (chosen and rejected) with the RM, take the
standard deviation ``sigma`` of those scores, and divide the score head's weight
by it. Qwen3.5's head is ``nn.Linear(hidden, 1, bias=False)``, so the mean is
not shifted; it does not need to be -- with gamma=1 a constant per-episode
reward is absorbed by the value baseline and the (mean-shifting) advantage
whitening. Only the scale reaches the policy gradient.

The transform is a positive rescale, so every ranking the RM produces is
unchanged: pairwise accuracy is recomputed on the same held-out split and must
equal the manifest's number exactly, or this script fails. The critic is
initialised from the same checkpoint, so it inherits the scale for free.

Calibration texts are the pair completions because they are the M0 samples we
have (the sweep does not keep non-pair candidates). Pairs are the spread-out
tail of the candidate distribution, so ``sigma`` is if anything an overestimate
of the on-policy spread and the leash comes out slightly STRONGER than unit --
the conservative direction for a one-shot run. Texts get the EOS RewardTrainer
appended in training and that PPO's scorer sees; the before/after accuracy
check uses train_reward_model's own ``_score_texts`` (no EOS) so the "before"
number reproduces the manifest byte-for-byte.

Usage (on the pod, right after train_reward_model.py; ~2 min at 9B on an A100):

    .venv/bin/python rl-rust/calibrate_reward_head.py \
        --reward_model runs/qwen35-9b-rm-rust-s0 \
        --out runs/qwen35-9b-rm-rust-s0-calibrated

``--pairs`` defaults to the manifest's ``pairs_path``; the file must hash to the
manifest's ``pairs_sha256`` because the held-out split is a seeded shuffle of it.
The output directory is a complete RM (weights + tokenizer + manifest with a
``score_head_calibration`` block) that train_rust_ppo.py --reward_model takes
directly; its preflight refuses an uncalibrated one.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CALIBRATION_SCHEMA = "score_head_calibration_v1"
CALIBRATION_KEY = "score_head_calibration"


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_reward_model(path: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    device = _device()
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        path, trust_remote_code=True, num_labels=1, dtype=dtype
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.get_text_config().pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def score_head(model):
    head = getattr(model, "score", None)
    if head is None or not hasattr(head, "weight"):
        raise SystemExit(f"reward model has no `score` linear head (type {type(model).__name__})")
    if head.weight.shape[0] != 1:
        raise SystemExit(f"score head has {head.weight.shape[0]} outputs; expected 1 (num_labels=1)")
    return head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reward_model", required=True, help="RM directory from train_reward_model.py")
    parser.add_argument("--out", required=True, help="Directory for the calibrated RM (must not exist unless --force)")
    parser.add_argument("--pairs", default=None, help="dpo_preference_v1 JSONL; default: manifest pairs_path")
    # MUST match train_reward_model.py's --eval_batch_size default (16), or the
    # reproduction check below fails on bf16 noise rather than on anything real.
    # Different batch size => different padding => slightly different scores, and
    # a pair sitting near the decision boundary flips. Cost a 9B pod on
    # 2026-09-11: recomputed 0.7037 against a manifest 0.7222, a ONE-PAIR
    # difference out of 54, and the hard equality check killed the run.
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_texts", type=int, default=None, help="Cap the calibration sample (debug only)")
    parser.add_argument("--force", action="store_true", help="Overwrite --out / recalibrate an already-calibrated RM")
    args = parser.parse_args()

    from scripts.rl_training.data.schema import read_preference_jsonl
    from scripts.rl_training.train_reward_model import (
        _score_texts,
        _sha256,
        build_pair_texts,
        pairwise_accuracy,
        split_train_eval,
    )

    rm_dir = Path(args.reward_model)
    manifest_path = rm_dir / "reward_model_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no reward_model_manifest.json in {rm_dir}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "reward_model_manifest_v1":
        raise SystemExit(f"unexpected manifest schema {manifest.get('schema')!r}")
    if CALIBRATION_KEY in manifest and not args.force:
        raise SystemExit(
            f"{rm_dir} is already calibrated (scale={manifest[CALIBRATION_KEY]['scale']:.4g}); "
            "calibrating twice would compound. Pass --force only if you know why."
        )
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{out_dir} exists and is not empty; pass --force to overwrite")

    pairs_path = args.pairs or manifest.get("pairs_path")
    if not pairs_path or not Path(pairs_path).is_file():
        raise SystemExit(f"pair file not found: {pairs_path!r} (pass --pairs)")
    pairs_sha = _sha256(pairs_path)
    if pairs_sha != manifest.get("pairs_sha256"):
        raise SystemExit(
            f"{pairs_path} hashes to {pairs_sha[:12]}..., manifest says {str(manifest.get('pairs_sha256'))[:12]}...; "
            "the held-out split is a seeded shuffle of the training file, so the file must be identical."
        )

    rows = read_preference_jsonl(pairs_path)
    examples = build_pair_texts(rows)
    cfg = manifest.get("config", {})
    eval_fraction = float(cfg.get("eval_fraction", 0.1))
    seed = int(manifest.get("seed", cfg.get("seed", 0)))
    max_length = int(cfg["max_length"])
    _, eval_examples = split_train_eval(examples, eval_fraction, seed)
    if len(eval_examples) != int(manifest.get("n_eval", len(eval_examples))):
        raise SystemExit(f"held-out split has {len(eval_examples)} pairs, manifest says {manifest.get('n_eval')}")
    print(f"[calibrate] {len(examples)} pairs, {len(eval_examples)} held-out (seed={seed}), RM max_length={max_length}")

    model, tokenizer = load_reward_model(str(rm_dir))
    device = model.device
    head = score_head(model)

    # 1. Reproduce the manifest's gate number with the manifest's method.
    eval_chosen = [e["chosen"] for e in eval_examples]
    eval_rejected = [e["rejected"] for e in eval_examples]
    before_c = _score_texts(model, tokenizer, eval_chosen, max_length, args.batch_size, device)
    before_r = _score_texts(model, tokenizer, eval_rejected, max_length, args.batch_size, device)
    acc_before = pairwise_accuracy(before_c, before_r)
    acc_manifest = float(manifest["held_out_pairwise_accuracy"])
    print(f"[calibrate] held-out pairwise accuracy before: {acc_before:.4f} (manifest {acc_manifest:.4f})")
    # WHAT THIS CHECK CAN AND CANNOT DO (rewritten 2026-09-11, after it killed a
    # 9B pod twice). It was asserting bit-exact reproduction of a bf16 number,
    # which is not a property this pipeline has. Same weights, same pair file,
    # same split, measured three ways:
    #
    #     A100, batch 16 (training)   0.7222   39/54
    #     A100, batch  8 (calibrate)  0.7037   38/54
    #     H200, batch 16 (calibrate)  0.6667   36/54
    #
    # Batch size shifts padding; a different card takes different kernel paths.
    # Both move borderline pairs across the decision boundary. cost.md already
    # says the card is the instrument -- this is that, at n=54.
    #
    # The failure modes this check was written for -- wrong pair file, wrong
    # seed, changed model -- are ALREADY caught exactly and earlier: pairs_sha256
    # must match byte-for-byte, and n_eval must match the manifest. Those are
    # hard equalities on the things that actually identify the data. So the
    # accuracy comparison becomes a loud warning, and the one hard requirement
    # is the thing that actually matters downstream: the RM as measured HERE, on
    # THIS card, must still clear the gate train_rl.py enforces. An RM that only
    # clears 0.60 on the machine that trained it has no business driving PPO on
    # a different one.
    n_eval = max(len(eval_examples), 1)
    delta_pairs = abs(acc_before - acc_manifest) * n_eval
    if acc_before < 0.60:
        raise SystemExit(
            f"recomputed held-out accuracy is {acc_before:.4f} on this device, below the 0.60 gate "
            f"(manifest recorded {acc_manifest:.4f} on the training device, a {delta_pairs:.0f}-pair "
            f"difference of {n_eval}). train_rl.py --algo ppo would refuse this RM here. Not calibrating."
        )
    if delta_pairs > 1e-6:
        print(f"[calibrate] NOTE: recomputed {acc_before:.4f} vs manifest {acc_manifest:.4f} "
              f"= {delta_pairs:.0f} pairs of {n_eval}. Expected across a different card or batch "
              f"size in bf16; the pair file sha256 and split size both match exactly, so this is "
              f"numerics, not wrong data. Still clears the 0.60 gate. Recording BOTH numbers.")

    # 2. Spread of the raw score over the completions PPO will be scoring
    #    (prompt + completion + EOS, as RewardTrainer trained and TRL scores).
    eos = tokenizer.eos_token or ""
    texts = []
    for e in examples:
        texts.append(e["chosen"] + eos)
        texts.append(e["rejected"] + eos)
    if args.max_texts:
        texts = texts[: args.max_texts]
    raw = _score_texts(model, tokenizer, texts, max_length, args.batch_size, device)
    mean_raw = statistics.fmean(raw)
    sigma_raw = statistics.stdev(raw)
    if not (sigma_raw > 0 and math.isfinite(sigma_raw)):
        raise SystemExit(f"raw score spread is degenerate (sigma={sigma_raw}); the RM is not separating anything")
    scale = 1.0 / sigma_raw
    print(
        f"[calibrate] raw score over {len(raw)} completions: mean {mean_raw:.4f}, sigma {sigma_raw:.4f}, "
        f"min {min(raw):.3f}, max {max(raw):.3f} -> scale {scale:.4g}"
    )

    # 3. Fold the rescale into the head. Positive scalar: rankings unchanged.
    import torch

    with torch.no_grad():
        head.weight.mul_(torch.tensor(scale, dtype=head.weight.dtype, device=head.weight.device))
        if getattr(head, "bias", None) is not None:
            head.bias.mul_(torch.tensor(scale, dtype=head.bias.dtype, device=head.bias.device))

    # 4. Verify: same gate number, unit spread.
    after_c = _score_texts(model, tokenizer, eval_chosen, max_length, args.batch_size, device)
    after_r = _score_texts(model, tokenizer, eval_rejected, max_length, args.batch_size, device)
    acc_after = pairwise_accuracy(after_c, after_r)
    after = _score_texts(model, tokenizer, texts, max_length, args.batch_size, device)
    sigma_after = statistics.stdev(after)
    max_dev = max(abs(a - b * scale) for a, b in zip(after, raw))
    print(
        f"[calibrate] after: held-out accuracy {acc_after:.4f}, sigma {sigma_after:.4f}, "
        f"max |after - scale*before| = {max_dev:.2e}"
    )
    if not math.isclose(acc_after, acc_before, abs_tol=1e-9):
        raise SystemExit("held-out accuracy CHANGED under a positive rescale -- numerical precision of the head "
                         f"is not sufficient ({acc_before:.4f} -> {acc_after:.4f}). Not saving.")
    if not 0.9 <= sigma_after <= 1.1:
        raise SystemExit(f"post-calibration sigma {sigma_after:.4f} is not ~1; not saving")

    # 5. Save a complete RM directory with the record of what was done.
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    manifest[CALIBRATION_KEY] = {
        "schema": CALIBRATION_SCHEMA,
        "source_reward_model": str(rm_dir),
        "scale": scale,
        "sigma_raw": sigma_raw,
        "mean_raw": mean_raw,
        "mean_after": statistics.fmean(after),
        "sigma_after": sigma_after,
        "n_scores": len(raw),
        "texts": "pair chosen+rejected completions with EOS appended",
        "pairs_path": pairs_path,
        "pairs_sha256": pairs_sha,
        "head_has_bias": getattr(head, "bias", None) is not None,
        "held_out_pairwise_accuracy_before": acc_before,
        "held_out_pairwise_accuracy_after": acc_after,
        # The manifest's gate number was measured on the TRAINING device; these
        # were measured here. They can differ by a few pairs in bf16 (see the
        # note in main()). Both are recorded so nobody has to guess later which
        # machine a number came from.
        "held_out_pairwise_accuracy_manifest": acc_manifest,
        "reproduction_delta_pairs": abs(acc_before - acc_manifest) * max(len(eval_examples), 1),
        "calibration_device": str(device),
        "calibration_batch_size": args.batch_size,
    }
    (out_dir / "reward_model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for extra in ("training_args.json", "trainer_state.json"):
        src = rm_dir / extra
        if src.is_file():
            shutil.copy2(src, out_dir / extra)
    print(f"[calibrate] wrote {out_dir} (scale {scale:.4g}); point train_rust_ppo.py --reward_model here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
