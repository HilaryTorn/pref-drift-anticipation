"""Merge a LoRA checkpoint into its base and save a standalone model. Training venv.

Why merge instead of serving base+adapter: the rust arms use
``lora_target_modules: all-linear``, which adapts Qwen3.5's linear-attention
projections (in_proj_a/b/z/qkv, out_proj) alongside the usual ones. vLLM's LoRA
path supports a fixed set of module types and skips the rest WITHOUT error, so a
served adapter can be partially applied and score like a weak model -- the same
silent-failure family as the multimodal key-namespace bug that once served a
bare base to every request. A merged model has no such degrees of freedom.

Verifies the merge actually changed the weights before saving, because "merged"
that silently no-ops is exactly the failure this exists to prevent.

    venv/bin/python rl-rust/eval_merge.py \
        --base prism-drift/qwen35-4b-m0-v4 \
        --adapter /root/runs/qwen35-4b-dpo-rust-s0/checkpoint-40 \
        --out /root/merged-4b-step40
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge] base={args.base} adapter={args.adapter}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True, dtype=torch.bfloat16, device_map="cpu"
    )
    # Snapshot a few adapted tensors so the merge can be proven, not assumed.
    before = {n: p.detach().clone() for i, (n, p) in enumerate(model.named_parameters())
              if "proj" in n and i % 97 == 0}
    print(f"[merge] watching {len(before)} tensors for change")

    peft_model = PeftModel.from_pretrained(model, args.adapter)
    merged = peft_model.merge_and_unload()

    after = dict(merged.named_parameters())
    changed = sum(1 for n, t in before.items()
                  if n in after and not torch.equal(t, after[n].detach().cpu()))
    print(f"[merge] {changed}/{len(before)} watched tensors changed")
    if changed == 0:
        raise SystemExit(
            "merge changed NOTHING -- the adapter was not applied. Refusing to "
            "write a model that would score as the bare base and look like a null."
        )

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    merged.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(str(out))
    print(f"[merge] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
