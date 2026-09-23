"""Pull the Rust RL eval generations off HF and grade them here.

The pods generate and then delete themselves, so everything they produced lives on the Hub under `prism-drift/rust-rl-checkpoints/rust-rl-eval/{4b,9b}`. This brings it down, grades both instruments locally, and prints the paired comparisons.

Grading is local on purpose. It needs rustc and no GPU, so doing it on the pod would have burned L40S hours on a CPU job -- and Rust is the one language upstream exempts from `limit_memory`, which is what makes macOS a legal place to run it. A Rust-only arm needs no conda env and no Linux eval box.

    .venv/bin/python rl-rust/fetch_and_grade_rust_rl_eval.py            # both sizes
    .venv/bin/python rl-rust/fetch_and_grade_rust_rl_eval.py --sizes 4b # just one
    .venv/bin/python rl-rust/fetch_and_grade_rust_rl_eval.py --skip_download

SECURITY: grading compiles and runs model-generated code. Same caution as rl_training/rewards.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
HF_REPO = "prism-drift/rust-rl-checkpoints"
LOCAL = REPO / "results/rust-rl-eval"
PY = sys.executable


def token() -> str:
    return re.search(
        r"hf_[A-Za-z0-9]{20,}", (REPO / "api_keys/api_key_hf_write.txt").read_text()
    ).group(0)


def download(size: str) -> pathlib.Path:
    from huggingface_hub import snapshot_download

    dest = LOCAL / size
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {HF_REPO}:rust-rl-eval/{size} -> {dest}")
    snapshot_download(
        repo_id=HF_REPO, repo_type="model", token=token(),
        allow_patterns=[f"rust-rl-eval/{size}/*"],
        local_dir=str(LOCAL / "_snap"),
    )
    src = LOCAL / "_snap" / "rust-rl-eval" / size
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    return dest


def run(cmd: list[str]) -> bool:
    print("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=REPO).returncode == 0


# The arms, and the checkpoint each one's final step lands on. GRPO stops at 124
# and DPO at 125 -- same 8 passes over the same cohort, different step accounting.
ARMS = ("grpo-step124", "dpo-step125")


def grade_heldout(size: str, root: pathlib.Path) -> None:
    gens = root / "heldout-eval"
    if not gens.is_dir():
        print(f"[heldout] no heldout-eval/ for {size}; skipping")
        return
    summaries = {}
    for tag in ("base", *ARMS):
        gen = gens / f"{size}-{tag}.gen.jsonl"
        if not gen.is_file():
            print(f"[heldout] missing {gen.name}; skipping")
            continue
        out = gens / f"{size}-{tag}.summary.json"
        print(f"\n[heldout] grading {size} {tag}")
        if run([PY, "rl-rust/eval_heldout.py", "--grade_only",
                "--gen_out", str(gen), "--summary_out", str(out)]):
            summaries[tag] = out

    if "base" not in summaries:
        print("[heldout] no base summary; the arms have nothing to be read against")
        return
    for tag in ARMS:
        if tag not in summaries:
            continue
        print(f"\n[heldout] ==== {size}: base vs {tag} ====")
        # The DPO arms were generated at n=1 against a base drawn at n=2 (2026-09-11). The comparison is paired per prompt, so that costs a little width in the arm's interval and biases nothing; --compare reads both n values out of the summaries.
        run([PY, "rl-rust/eval_heldout.py", "--compare",
             str(summaries["base"]), str(summaries[tag])])


def grade_multilcb(size: str, root: pathlib.Path) -> None:
    """Stage the pod's generations into the vendored harness's own cache, then grade."""
    src = root / "multilcb-generations"
    if not src.is_dir():
        print(f"[multilcb] no multilcb-generations/ for {size}; skipping")
        return
    dest_root = REPO / "multi-lcb" / "output"
    dest_root.mkdir(parents=True, exist_ok=True)

    # The base draw is deliberately absent -- see config.yaml on why the stack-validation
    # draw was declined. Every number here is read against the stored AWS baselines.
    keys = {
        f"v6_qwen35-{size}-grpo-rust-s124_cot": f"qwen35-{size}-grpo-rust-step124",
        f"v6_qwen35-{size}-dpo-rust-s125_cot": f"qwen35-{size}-dpo-rust-step125",
    }

    for out_dir, model_key in keys.items():
        staged = src / out_dir
        if not staged.is_dir():
            print(f"[multilcb] {out_dir} not in the upload; skipping {model_key}")
            continue
        shutil.copytree(staged, dest_root / out_dir, dirs_exist_ok=True)
        print(f"\n[multilcb] grading {model_key}")
        # --n 1 MUST match what generation used. score_multilcb names its cache by sample count, and an --evaluate_only whose --n does not match silently falls through to a full GENERATION run against whatever base_url the key carries.
        run([PY, "scripts/score_multilcb.py", "--model_key", model_key,
             "--languages", "rust", "--release_version", "v6", "--n", "1",
             "--evaluate_only"])


def report(size: str) -> None:
    """Print each arm against the stored AWS baseline it exists to be compared with."""
    stored = {"4b": 0.0914, "9b": 0.120}[size]
    print(f"\n[multilcb] ==== {size} rust v6 n=175 ====")
    print(f"[multilcb] stored M0-v4 baseline {stored:.4f}, Rust SFT arm 0.223 (both AWS L40S)")
    for tag, key in (("GRPO step124", f"qwen35-{size}-grpo-rust-step124"),
                     ("DPO step125", f"qwen35-{size}-dpo-rust-step125")):
        summ = sorted((REPO / "results" / key / "multilcb").glob("*.json"))
        if not summ:
            print(f"[multilcb] {tag}: no summary")
            continue
        r = json.loads(summ[-1].read_text()).get("results", {}).get("rust")
        if not r:
            print(f"[multilcb] {tag}: summary has no rust block")
            continue
        print(f"[multilcb] {tag}: pass@1 = {r['pass@1']:.4f}  "
              f"pass@1_given_code = {r['pass@1_given_code']:.4f}  "
              f"no_code_rate = {r['no_code_rate']:.3f}")
    print("[multilcb] read pass@1 and no_code_rate together: most of no_code is generations "
          "hitting max_tokens, not the model declining, so pass@1 moves with truncation as well "
          "as with skill. The DPO arms generate markedly longer than base (seen at preflight), "
          "which is exactly the composition this warning is about.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", nargs="+", default=["4b", "9b"])
    p.add_argument("--skip_download", action="store_true")
    args = p.parse_args()

    for size in args.sizes:
        print(f"\n{'=' * 70}\n{size.upper()}\n{'=' * 70}")
        root = LOCAL / size
        if not args.skip_download:
            root = download(size)
        if not root.is_dir():
            print(f"nothing downloaded for {size}")
            continue
        grade_heldout(size, root)
        grade_multilcb(size, root)
        report(size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
