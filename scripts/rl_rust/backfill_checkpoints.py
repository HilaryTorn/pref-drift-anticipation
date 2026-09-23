"""Backfill every intermediate checkpoint into the prism-drift rust-rl repos.

Layout matches the SFT adapter repos (coding.write.rust/checkpoint-N/...):

    prism-drift/qwen35-{size}-m0-v4-rust-rl-adapters
        grpo.rust.s0/
            adapter_model.safetensors      final (step 124), already uploaded
            checkpoint-4/ ... checkpoint-124/

The frozen reference adapter is uploaded ONCE at the intervention root rather
than inside all 31 checkpoints -- it is byte-identical in every one, so copying
it 31 times would roughly double the repo for no information.

Uploads per checkpoint rather than as one folder so a network failure costs one
checkpoint instead of the whole 15GB, and so a rerun skips what already landed.
"""

from __future__ import annotations

import pathlib
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
KEEP = ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json")


def token() -> str:
    return re.search(
        r"hf_[A-Za-z0-9]{20,}", (REPO / "api_keys/api_key_hf_write.txt").read_text()
    ).group(0)


def main() -> int:
    from huggingface_hub import HfApi

    api = HfApi(token=token())
    for size in ("9b", "4b"):
        rid = f"prism-drift/qwen35-{size}-m0-v4-rust-rl-adapters"
        src = REPO / f"results/rust-rl/{size}/grpo"
        have = set(api.list_repo_files(rid))

        cks = sorted(
            src.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        print(f"\n=== {rid}  ({len(cks)} checkpoints local) ===", flush=True)

        # one copy of the frozen reference adapter
        ref = next((c / "ref" for c in cks if (c / "ref").is_dir()), None)
        if ref and f"grpo.rust.s0/ref/adapter_model.safetensors" not in have:
            api.upload_folder(
                folder_path=str(ref), repo_id=rid, repo_type="model",
                path_in_repo="grpo.rust.s0/ref",
                commit_message="frozen reference adapter (identical across checkpoints)",
            )
            print("  uploaded ref/ (once)", flush=True)

        for i, ck in enumerate(cks, 1):
            step = ck.name.split("-")[1]
            dest = f"grpo.rust.s0/checkpoint-{step}"
            if f"{dest}/adapter_model.safetensors" in have:
                print(f"  [{i:2d}/{len(cks)}] step {step:>3} already present, skip", flush=True)
                continue
            files = [f for f in KEEP if (ck / f).is_file()]
            if not files:
                print(f"  [{i:2d}/{len(cks)}] step {step:>3} EMPTY, skip", flush=True)
                continue
            t0 = time.time()
            for f in files:
                api.upload_file(
                    path_or_fileobj=str(ck / f), path_in_repo=f"{dest}/{f}",
                    repo_id=rid, repo_type="model",
                    commit_message=f"checkpoint-{step}",
                )
            print(f"  [{i:2d}/{len(cks)}] step {step:>3}  {len(files)} files  {time.time()-t0:.0f}s", flush=True)

        n = len(api.list_repo_files(rid))
        print(f"  DONE {rid}: {n} files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
