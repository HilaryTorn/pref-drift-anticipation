"""Rename a LoRA adapter's keys into the multimodal namespace vLLM serves from.

Without this the adapter silently applies NOTHING. `prism-drift/qwen35-{4b,9b,27b}-m0-v4` are all `Qwen3_5ForConditionalGeneration`, so their language weights are named `model.language_model.layers.N...`. Training loads the model text-only (`AutoModelForCausalLM` extracts the text sub-model), so PEFT saves `base_model.model.model.layers.N...` with no `language_model.` segment. vLLM matches zero modules, logs `Loaded new LoRA adapter` anyway, and serves the bare base to every request -- no error, no warning, and `/v1/models` still lists the adapter with the right parent.

That cost a day on 2026-09-01 and produced a convincing fake null. `docs/serving-vllm-aws.md` carries the full account and the one-line fix this file automates.

Verified against the real weights rather than assumed (2026-09-11): the 9B base's tensor list contains `model.language_model.layers.0.linear_attn.in_proj_a.weight`, and the GRPO adapter contains `base_model.model.model.layers.0.linear_attn.in_proj_a.lora_A.weight`. The rename below is what closes that gap, and it covers the hybrid linear-attention projections (`in_proj_*`, `out_proj`) as well as the standard ones -- `all-linear` targeted 12 module types, and 18 of 24 layers at 9B do their attention in `linear_attn`.

Remapping is preferred over merging for this run: merging removes vLLM's LoRA path entirely and is why M0 itself always serves cleanly, but it costs a full model copy and a server reload per checkpoint. One adapter per size is cheap either way; the remap keeps both on one server.

This does NOT remove the need for the runtime check. The mismatch is version-dependent -- the same adapters applied correctly under an older transformers and stopped under 5.16.1 -- so it can regress on any upgrade in either direction. Always run `eval_heldout.py --preflight`, which greedy-decodes base and adapter and fails if they come back identical.

    .venv/bin/python rl-rust/remap_adapter_for_vllm.py \
        --src results/rust-rl/9b/grpo/checkpoint-124 \
        --dst results/rust-rl/9b/grpo/checkpoint-124-vllm
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

OLD = "base_model.model.model.layers."
NEW = "base_model.model.model.language_model.layers."


def remap(src: Path, dst: Path) -> None:
    from safetensors.torch import load_file, save_file

    weights = src / "adapter_model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"no adapter_model.safetensors in {src}")

    tensors = load_file(str(weights))
    if all("language_model" in k for k in tensors):
        raise SystemExit(f"{src} is already remapped; refusing to double-apply")
    unexpected = [k for k in tensors if not k.startswith(OLD)]
    if unexpected:
        # Bail rather than half-rename: a partially remapped adapter applies its renamed half and silently drops the rest, which is a subtler version of the bug being fixed.
        raise SystemExit(
            f"{len(unexpected)} keys do not start with {OLD!r}, e.g. {unexpected[:3]}. "
            f"This adapter is not in the shape this remap expects -- inspect it by hand."
        )

    new = {k.replace(OLD, NEW): v for k, v in tensors.items()}
    assert len(new) == len(tensors), "rename collided"
    assert all("language_model" in k for k in new), "rename missed keys"

    dst.mkdir(parents=True, exist_ok=True)
    save_file(new, str(dst / "adapter_model.safetensors"))
    shutil.copy2(src / "adapter_config.json", dst / "adapter_config.json")

    # Read the artifact back rather than trusting the write, and confirm shapes survived.
    back = load_file(str(dst / "adapter_model.safetensors"))
    assert len(back) == len(tensors), f"round trip lost tensors: {len(back)} vs {len(tensors)}"
    for old_key, tensor in tensors.items():
        assert back[old_key.replace(OLD, NEW)].shape == tensor.shape, f"shape changed for {old_key}"

    (dst / "remap_manifest.json").write_text(json.dumps({
        "schema": "vllm_lora_key_remap_v1",
        "src": str(src),
        "rule": {"from": OLD, "to": NEW},
        "n_tensors": len(back),
        "reason": "multimodal base (Qwen3_5ForConditionalGeneration) names language weights model.language_model.*; see docs/serving-vllm-aws.md",
    }, indent=1))
    print(f"[remap] {len(back)} tensors -> {dst}")
    print(f"[remap]   sample: {sorted(back)[0]}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    args = p.parse_args()
    remap(Path(args.src), Path(args.dst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
