"""Offline held-out generation for a rust arm, under vLLM. Runs in the vLLM venv.

Replaces the in-training DriftCadenceCallback, which generated with HF
``model.generate`` one prompt at a time -- measured at 104 s/prompt (4B) and
159 s/prompt (9B), i.e. 55-83 min per eval point against 52-83 min for an
entire 125-step run. Batched vLLM does the same work in minutes, which buys
both a bigger sample and the ability to score every checkpoint instead of four.

Takes a MERGED model directory (see eval_merge.py), not base+adapter: the arms
use ``lora_target_modules: all-linear``, which adapts Qwen3.5's linear-attention
projections, and vLLM's LoRA path silently skips modules it does not support --
a partially-applied adapter scores like a weak model with no error anywhere.
Merging sidesteps the entire question.

Writes one JSONL row per generation (prompt id, text, finish_reason) for
eval_grade.py to score with rustc. Generation and grading are separate so the
grading step can run anywhere rustc exists, and so a re-grade never needs a GPU.

    /root/vllmenv/bin/python rl-rust/eval_generate.py \
        --model /root/merged-4b-step125 --prompts rl-rust/out/heldout-500-clean.jsonl \
        --out /root/eval/4b-step125.completions.jsonl --n_prompts 250 --n_samples 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-rust"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged model directory (or base model id for the baseline)")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_prompts", type=int, default=250)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from scripts.rl_rust.rl_rust.prompts import render_coding_prompt

    rows = [json.loads(l) for l in open(args.prompts) if l.strip()][: args.n_prompts]
    print(f"[eval] {len(rows)} held-out prompts from {args.prompts}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [render_coding_prompt(tok, r["prompt"], format_name="rust_io") for r in rows]

    # Same sampling contract as training rollouts and the pair sweep:
    # temperature-only, top_p=1, no top_k, so numbers are comparable across arms.
    llm = LLM(model=args.model, trust_remote_code=True,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    sampling = SamplingParams(n=args.n_samples, temperature=args.temperature,
                              top_p=1.0, top_k=-1, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sampling)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as fh:
        for row, out in zip(rows, outputs):
            for cand in out.outputs:
                fh.write(json.dumps({
                    "record_id": row["record_id"],
                    "text": cand.text,
                    "finish_reason": cand.finish_reason,
                }, ensure_ascii=True) + "\n")
                n += 1
    print(f"[eval] wrote {n} generations -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
