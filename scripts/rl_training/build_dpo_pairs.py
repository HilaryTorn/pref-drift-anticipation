#!/usr/bin/env python3
"""Build DPO preference pairs from the SAME problem pool and verifier as PPO/GRPO.

DPO can't consume the reward directly, so we manufacture preferences from it:
sample K completions per prompt from the base (or SFT) model, score each with the
I/O verifier, pair best vs worst, and keep only pairs with
``pass_rate(chosen) - pass_rate(rejected) >= --margin`` (default 0.5). The
preference signal thus comes from the same verifier as the online arms — no human
preference data, one shared objective across PPO/GRPO/DPO.

Input:  a canonical coding_task_v1 prompts JSONL ({prompt, verifier}).
Output: a DPO JSONL ({prompt, chosen, rejected, chosen_reward, rejected_reward}).
Optionally also writes per-prompt score summaries for learnable-band filtering.

Sampling has three backends (grading is always local — it executes code):
  - local transformers (default, --backend hf): loads --model on this box.
  - in-process vLLM (--backend vllm): ~10x faster sweep in a dedicated vLLM env.
  - vLLM endpoint (--endpoint): samples via the OpenAI-compatible /completions
    API, so the full candidate-pool run can hit an existing serving endpoint instead
    of loading the model here. This keeps generation off paid AWS GPU per the
    Inference Rule (docs/cost.md).

    # local smoke
    python rl_training/build_dpo_pairs.py \
        --model Qwen/Qwen3.5-0.8B --device cpu --limit 10 \
        --prompts data/rl/nemotron_train_1000.jsonl \
        --out data/rl/nemotron_dpo_pairs.jsonl --K 8 --margin 0.5

    # full run against a vLLM endpoint
    python rl_training/build_dpo_pairs.py \
        --model Qwen/Qwen3.5-0.8B \
        --endpoint http://localhost:8000/v1 \
        --api_key_file api_keys/api_key_vllm_endpoint.txt \
        --prompts data/rl/nemotron_train_1000.jsonl \
        --out data/rl/nemotron_dpo_pairs.jsonl --K 8 --margin 0.5

SECURITY: runs model-generated code via the verifier — see rl_training/rewards.py.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    """Replace a JSONL file atomically after flushing it to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_resume_jsonl(path: Path) -> list[dict]:
    """Read JSONL, recovering only a truncated final line from an interrupted write."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    nonblank = [index for index, line in enumerate(lines) if line.strip()]
    last_nonblank = nonblank[-1] if nonblank else None
    rows: list[dict] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if index != last_nonblank:
                raise SystemExit(f"{path}:{index + 1}: invalid JSON inside file: {exc}") from exc
            backup = path.with_name(f"{path.name}.resume-backup-{int(time.time())}")
            shutil.copy2(path, backup)
            _atomic_write_jsonl(path, rows)
            print(f"[dpo] recovered truncated final line in {path}; backup -> {backup}")
            break
    return rows


def _record_id(row: dict, *, source: Path, line_no: int) -> str:
    rid = row.get("record_id")
    if not isinstance(rid, str) or not rid:
        raise SystemExit(f"{source}:{line_no}: missing record_id")
    return rid


def _score_requires_pair(score_row: dict, margin: float) -> bool:
    """Would this prompt have produced a pair? Must mirror pair selection exactly.

    Resume treats a score row as the completion marker and drops it when its
    expected pair is missing. Pair selection ignores truncated candidates, so this
    must too: otherwise a prompt whose only low scorer was truncated looks like a
    lost pair and is resampled on every resume, forever.

    Score rows written before truncation tracking have no ``statuses`` field and
    keep the original all-candidates behaviour, so existing sweeps still resume.
    """
    from scripts.rl_training.rewards import COMPLETION_OK

    scores = score_row.get("scores")
    if not isinstance(scores, list) or not scores:
        return False
    statuses = score_row.get("statuses")
    if isinstance(statuses, list) and len(statuses) == len(scores):
        scores = [
            value for value, status in zip(scores, statuses) if status == COMPLETION_OK
        ]
    if len(scores) < 2:
        return False
    return max(float(value) for value in scores) - min(float(value) for value in scores) >= margin


def _reconcile_resume_outputs(
    scores_path: Path,
    out_path: Path,
    score_rows: list[dict],
    pair_rows: list[dict],
    margin: float,
) -> tuple[list[dict], list[dict]]:
    """Repair cross-file tails left by interruption before deciding what to skip.

    A score row is the completion marker.  If its expected pair is absent, drop
    that score so the prompt is resampled.  Conversely, a pair without a score
    is an uncommitted orphan and is removed.
    """
    score_by_id: dict[str, dict] = {}
    for line_no, row in enumerate(score_rows, 1):
        rid = _record_id(row, source=scores_path, line_no=line_no)
        if rid in score_by_id:
            raise SystemExit(f"{scores_path}:{line_no}: duplicate record_id {rid!r}")
        score_by_id[rid] = row

    pair_by_id: dict[str, dict] = {}
    for line_no, row in enumerate(pair_rows, 1):
        rid = _record_id(row, source=out_path, line_no=line_no)
        if rid in pair_by_id:
            raise SystemExit(f"{out_path}:{line_no}: duplicate record_id {rid!r}")
        pair_by_id[rid] = row

    repaired_scores = [
        row
        for rid, row in score_by_id.items()
        if not _score_requires_pair(row, margin) or rid in pair_by_id
    ]
    committed_score_ids = {row["record_id"] for row in repaired_scores}
    repaired_pairs = [
        row
        for rid, row in pair_by_id.items()
        if rid in committed_score_ids and _score_requires_pair(score_by_id[rid], margin)
    ]
    removed_scores = len(score_rows) - len(repaired_scores)
    removed_pairs = len(pair_rows) - len(repaired_pairs)
    if removed_scores or removed_pairs:
        _atomic_write_jsonl(scores_path, repaired_scores)
        _atomic_write_jsonl(out_path, repaired_pairs)
        print(
            f"[dpo] repaired interrupted transaction tail: "
            f"resample_scores={removed_scores}, orphan_pairs={removed_pairs}"
        )
    return repaired_scores, repaired_pairs


def _acquire_sweep_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SystemExit(f"Another process is already writing this sweep: {path}") from exc
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def _durable_jsonl_write(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


# Sampling MUST match TRL's GRPO rollout distribution (temperature-only,
# top_p=1, no top_k) so pair construction is on-policy for the online arms.
# All backends override the model's own generation_config defaults, which for
# Qwen chat models would otherwise silently apply top_p/top_k.


def _hf_completions(rows, args, tokenizer):
    """Yield (row, rendered_prompt, completions) via Transformers generate."""
    import torch
    from transformers import AutoModelForCausalLM

    from scripts.rl_training.prompts import render_coding_prompt

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        dtype=dtype,
    )
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    model.to(device)
    model.eval()
    print(f"[dpo] backend=hf device: {device}")

    for row in rows:
        formatted_prompt = render_coding_prompt(tokenizer, row["prompt"])
        enc = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                top_p=1.0,
                top_k=0,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.K,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = [seq[enc["input_ids"].shape[1]:] for seq in out]
        completions = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated]
        # eos_token_id is a list on Qwen3.5 (<|im_end|> as well as <|endoftext|>);
        # reading only the scalar marks properly-finished completions as truncated.
        raw_eos = getattr(model.generation_config, "eos_token_id", None) or tokenizer.eos_token_id
        eos_ids = set(raw_eos if isinstance(raw_eos, (list, tuple, set)) else [raw_eos])
        finish_reasons = [
            "stop" if any(int(t) in eos_ids for t in seq) else "length" for seq in generated
        ]
        yield row, formatted_prompt, completions, finish_reasons


def _vllm_completions(rows, args, tokenizer, chunk_size: int = 64):
    """Yield (row, rendered_prompt, completions) via an in-process vLLM engine.

    Runs in a separate environment from training (vLLM pins its own torch/CUDA;
    see rl_training/requirements-training.txt). Chunked so verifier scoring and
    progress interleave with generation instead of waiting for the full sweep.
    """
    from vllm import LLM, SamplingParams

    from scripts.rl_training.prompts import render_coding_prompt

    llm = LLM(
        model=args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        n=args.K,
        temperature=args.temperature,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_new_tokens,
    )
    prompt_budget = args.max_model_len - args.max_new_tokens

    def render(row):
        # Prompts should already be length-filtered at prep time; left-truncate
        # any stray overlong one (keeps the assistant header), matching the
        # trainer's truncation_side="left" fallback.
        prompt = render_coding_prompt(tokenizer, row["prompt"])
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if len(ids) > prompt_budget:
            prompt = tokenizer.decode(ids[-prompt_budget:])
        return prompt

    print(f"[dpo] backend=vllm max_model_len={args.max_model_len} "
          f"gpu_memory_utilization={args.gpu_memory_utilization}")
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        prompts = [render(row) for row in chunk]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        for row, formatted_prompt, output in zip(chunk, prompts, outputs):
            yield (
                row,
                formatted_prompt,
                [candidate.text for candidate in output.outputs],
                # vLLM reports "length" when the cap was hit. Keeping it is what
                # lets pair selection tell "wrong" from "never finished"; it used
                # to be dropped here, which is why no existing pair file can say
                # how many of its `rejected` sides are truncation artifacts.
                [candidate.finish_reason for candidate in output.outputs],
            )


def _endpoint_completions(rows, args, tokenizer):
    """Yield (row, rendered_prompt, completions) via an OpenAI-compatible vLLM /completions API."""
    import requests

    from scripts.rl_training.prompts import render_coding_prompt

    base = args.endpoint.rstrip("/")
    url = f"{base}/completions"
    api_key = None
    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text().strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    served_model = args.served_model_name or args.model
    print(f"[dpo] backend=endpoint url={url} model={served_model}")

    for row in rows:
        formatted_prompt = render_coding_prompt(tokenizer, row["prompt"])
        body = {
            "model": served_model,
            "prompt": formatted_prompt,
            "n": args.K,
            "max_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": 1.0,
        }
        resp = requests.post(url, headers=headers, json=body, timeout=args.endpoint_timeout)
        resp.raise_for_status()
        choices = resp.json().get("choices", [])
        completions = [choice.get("text", "") for choice in choices]
        finish_reasons = [choice.get("finish_reason") for choice in choices]
        yield row, formatted_prompt, completions, finish_reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Base/SFT model to sample candidates from (or served model id for --endpoint)")
    parser.add_argument("--model_revision", default=None, help="Pinned HF model commit/revision for local backends")
    parser.add_argument("--prompts", required=True, help="Prepared coding_task_v1 prompts JSONL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scores_out", default=None, help="Optional per-prompt score summary JSONL")
    parser.add_argument("--manifest_out", default=None, help="Optional sweep provenance manifest JSON")
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"],
                        help="hf = Transformers generate (training env); vllm = in-process vLLM engine (separate env). Ignored when --endpoint is set.")
    parser.add_argument("--K", type=int, default=8, help="Candidates sampled per prompt")
    parser.add_argument("--margin", type=float, default=0.5, help="Min pass-rate gap to keep a pair")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    # Shared reward budget: MUST match GRPO's training reward (rl_training/rewards.py
    # constants) so DPO pairs and the PPO reward model share GRPO's objective.
    parser.add_argument(
        "--max_tests",
        type=int,
        default=None,
        help="Tests per completion for scoring; defaults to the shared TRAIN_REWARD_MAX_TESTS. Must match GRPO.",
    )
    parser.add_argument("--reward_timeout", type=float, default=None, help="Per-test timeout; defaults to the shared REWARD_TIMEOUT. Must match GRPO.")
    parser.add_argument("--device", default="auto", help="hf backend: 'auto', 'cpu', 'cuda', or a torch device")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.45,
                        help="vllm backend: fraction of GPU memory to claim (GPU may be shared)")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vllm backend: context window")
    parser.add_argument("--endpoint", default=None, help="OpenAI-compatible vLLM base URL (e.g. http://host:8000/v1). If set, sample via HTTP instead of --backend.")
    parser.add_argument("--served_model_name", default=None, help="Model id the endpoint serves, if different from --model")
    parser.add_argument("--api_key_file", default=None, help="File with the endpoint bearer token")
    parser.add_argument("--endpoint_timeout", type=float, default=300.0, help="Per-request HTTP timeout for --endpoint (seconds)")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of prompts (debug)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append safely, skipping record IDs already present in --scores_out",
    )
    args = parser.parse_args()

    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    from scripts.rl_training.data.schema import DPOPreferenceRecord, read_coding_jsonl
    from scripts.rl_training.rewards import (
        REWARD_TIMEOUT,
        TRAIN_REWARD_MAX_TESTS,
        build_preference_pair_from_candidates,
        score_candidates,
    )

    # Default to the shared reward budget so DPO pairs are graded identically to
    # GRPO's training reward without the caller having to remember the numbers.
    max_tests = args.max_tests if args.max_tests is not None else TRAIN_REWARD_MAX_TESTS
    reward_timeout = args.reward_timeout if args.reward_timeout is not None else REWARD_TIMEOUT

    rows = read_coding_jsonl(args.prompts)
    if args.limit:
        rows = rows[: args.limit]
    target_rows = list(rows)
    if args.endpoint:
        sampler_kind = "endpoint"
    elif args.backend == "vllm":
        sampler_kind = "vllm"
    else:
        sampler_kind = "local"

    # Recorded in every score/pair row so local-smoke, vllm, and endpoint runs are
    # auditable as the same procedure (differing only in where sampling ran).
    gen_meta = {
        "generator_model": args.model,
        "generator_model_revision": args.model_revision,
        "source_prompts_sha256": _sha256(Path(args.prompts).resolve()),
        "sampler": sampler_kind,
        "backend": args.backend,
        "endpoint": args.endpoint,
        "sample_k": args.K,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "reward_max_tests": max_tests,
        "reward_timeout": reward_timeout,
        "pair_margin": args.margin,
        "served_model_name": args.served_model_name,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path = Path(args.scores_out) if args.scores_out else None
    if args.resume and scores_path is None:
        raise SystemExit("--resume requires --scores_out so completed record IDs are auditable")
    lock_base = scores_path or out_path
    lock_path = lock_base.with_suffix(lock_base.suffix + ".lock")
    lock_handle = _acquire_sweep_lock(lock_path)

    existing_score_rows: list[dict] = []
    existing_score_ids: set[str] = set()
    kept = 0
    solved_any = 0
    # Truncation accounting. Without it a pair file cannot say how much of its
    # signal is 'finished vs did not finish' rather than 'correct vs incorrect'.
    truncated_candidates = 0
    total_candidates = 0
    dropped_for_truncation = 0
    if args.resume and scores_path and scores_path.exists():
        if not out_path.exists():
            raise SystemExit(
                f"Cannot resume: {scores_path} exists but pair file {out_path} is missing"
            )
        existing_score_rows = _read_resume_jsonl(scores_path)
        existing_pair_rows = _read_resume_jsonl(out_path)
        required_meta = tuple(gen_meta)
        for line_no, score_row in enumerate(existing_score_rows, 1):
            _record_id(score_row, source=scores_path, line_no=line_no)
            mismatches = [
                key for key in required_meta if score_row.get(key) != gen_meta[key]
            ]
            if mismatches:
                raise SystemExit(
                    f"Cannot resume {scores_path}: row {line_no} differs on {mismatches}. "
                    "Use a new model sweep directory for different generation settings."
                )
        existing_score_rows, existing_pair_rows = _reconcile_resume_outputs(
            scores_path, out_path, existing_score_rows, existing_pair_rows, args.margin
        )
        for line_no, score_row in enumerate(existing_score_rows, 1):
            rid = score_row.get("record_id")
            if not isinstance(rid, str) or not rid:
                raise SystemExit(f"{scores_path}:{line_no}: missing record_id")
            if rid in existing_score_ids:
                raise SystemExit(f"{scores_path}:{line_no}: duplicate record_id {rid!r}")
            mismatches = [
                key for key in required_meta if score_row.get(key) != gen_meta[key]
            ]
            if mismatches:
                raise SystemExit(
                    f"Cannot resume {scores_path}: row {line_no} differs on {mismatches}. "
                    "Use a new model sweep directory for different generation settings."
                )
            existing_score_ids.add(rid)
            solved_any += int(float(score_row.get("max_score", 0.0)) > 0)
        kept = len(existing_pair_rows)
        rows = [
            row for row in target_rows
            if (row.get("record_id") or row.get("hash_id")) not in existing_score_ids
        ]
        print(f"[dpo] resume: {len(existing_score_ids)} completed, {len(rows)} remaining")
    elif args.resume:
        if out_path.exists():
            raise SystemExit(
                f"Cannot resume: pair file {out_path} exists but score file {scores_path} is missing"
            )
        rows = target_rows

    if not args.resume and (out_path.exists() or (scores_path and scores_path.exists())):
        raise SystemExit("Output files already exist; use --resume or choose a new sweep directory")

    tokenizer = None
    completion_stream = iter(())
    if rows:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.model_revision,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if args.endpoint:
            completion_stream = _endpoint_completions(rows, args, tokenizer)
        elif args.backend == "vllm":
            completion_stream = _vllm_completions(rows, args, tokenizer)
        else:
            completion_stream = _hf_completions(rows, args, tokenizer)

    scores_fout = None
    if scores_path:
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_fout = scores_path.open("a" if args.resume else "w")
    # prompts where at least one sample passed something
    progress = tqdm(total=len(rows), desc="[dpo] sampling+scoring", unit="prompt")
    processed_rows = 0
    try:
        with out_path.open("a" if args.resume else "w") as fout:
            for i, (row, formatted_prompt, completions, finish_reasons) in enumerate(completion_stream):
                # Prompts come rendered through the chat template, identical to
                # GRPO training, so scores/pairs transfer across arms.
                if len(completions) != args.K:
                    raise RuntimeError(
                        f"Sampler returned {len(completions)} completions for "
                        f"{row.get('record_id')!r}; expected K={args.K}"
                    )
                candidates = score_candidates(
                    completions, row["verifier"], timeout=reward_timeout,
                    max_tests=max_tests, finish_reasons=finish_reasons,
                )
                # ``scores`` keeps its historical meaning for prompt_scores.jsonl and
                # the learnable-band filter: truncated candidates count as 0.0 there.
                # Pair SELECTION below deliberately does not use it.
                scores = [c["reward"] if c["reward"] is not None else 0.0 for c in candidates]
                statuses = [c["status"] for c in candidates]
                pair, pair_audit = build_preference_pair_from_candidates(
                    formatted_prompt, candidates, margin=args.margin
                )
                truncated_candidates += pair_audit["n_truncated"]
                total_candidates += pair_audit["n_candidates"]
                if pair is None and pair_audit["n_gradable"] < 2:
                    dropped_for_truncation += 1
                score_row = {
                        "schema": "prompt_score_v1",
                        "record_id": row.get("record_id") or row.get("hash_id") or f"row-{i}",
                        "dataset_id": row.get("dataset_id", "unknown"),
                        **gen_meta,
                        "scores": scores,
                        "statuses": statuses,
                        "finish_reasons": list(finish_reasons),
                        "min_score": min(scores) if scores else 0.0,
                        "max_score": max(scores) if scores else 0.0,
                        "mean_score": sum(scores) / len(scores) if scores else 0.0,
                        "n_gradable": pair_audit["n_gradable"],
                        "n_truncated": pair_audit["n_truncated"],
                    }
                if pair is not None:
                    record = DPOPreferenceRecord(
                        record_id=row.get("record_id") or row.get("hash_id") or f"row-{i}",
                        dataset_id=row.get("dataset_id", "unknown"),
                        prompt=pair["prompt"],
                        chosen=pair["chosen"],
                        rejected=pair["rejected"],
                        chosen_reward=pair["chosen_reward"],
                        rejected_reward=pair["rejected_reward"],
                        metadata={
                            "source_record_id": row.get("record_id") or row.get("hash_id"),
                            "margin": args.margin,
                            **gen_meta,
                        },
                    )
                    # Pair is prepared first; the durable score row below is the
                    # transaction's completion marker used by resume.
                    _durable_jsonl_write(fout, record.to_json())
                    kept += 1
                if scores_fout:
                    _durable_jsonl_write(scores_fout, score_row)
                solved_any += int(any(score > 0 for score in scores))
                processed_rows += 1
                progress.update(1)
                completed = len(existing_score_ids) + i + 1
                progress.set_postfix(pairs=kept, solved=f"{solved_any}/{completed}")
    finally:
        progress.close()
        if scores_fout:
            scores_fout.close()
    completed_rows = len(existing_score_ids) + processed_rows
    if completed_rows != len(target_rows):
        raise SystemExit(
            f"Sweep ended after {completed_rows}/{len(target_rows)} prompts; "
            "completion manifest was not written"
        )
    print(
        f"[dpo] done: {kept} pairs from {completed_rows}/{len(target_rows)} prompts "
        f"(margin>={args.margin}) -> {out_path}"
    )
    if total_candidates:
        truncated_rate = truncated_candidates / total_candidates
        print(
            f"[dpo] truncation: {truncated_candidates}/{total_candidates} candidates "
            f"({truncated_rate:.1%}) hit the {args.max_new_tokens}-token cap and were "
            f"excluded from pair selection; {dropped_for_truncation} prompts yielded "
            f"fewer than two finished candidates."
        )
        if truncated_rate > 0.10:
            print(
                f"    !! {truncated_rate:.0%} truncation means this cohort is selected "
                f"toward problems the model can finish inside {args.max_new_tokens} tokens, "
                f"which is a difficulty bias, not just lost data. Raise --max_new_tokens "
                f"until this is under ~5% before treating the pairs as representative."
            )
    if scores_path:
        print(f"[dpo] wrote score summaries -> {scores_path}")
    if args.manifest_out:
        manifest_path = Path(args.manifest_out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        prompts_path = Path(args.prompts).resolve()
        manifest = {
            "schema": "model_data_sweep_v1",
            "status": "complete",
            "generator": gen_meta,
            "inputs": {
                "prompts": str(prompts_path),
                "prompts_sha256": _sha256(prompts_path),
                "prompt_rows": len(target_rows),
                "limit": args.limit,
            },
            "outputs": {
                "pairs": str(out_path.resolve()),
                "pairs_sha256": _sha256(out_path),
                "pair_rows": kept,
                "scores": str(scores_path.resolve()) if scores_path else None,
                "scores_sha256": _sha256(scores_path) if scores_path else None,
                "score_rows": completed_rows if scores_path else 0,
            },
            "summary": {
                "prompts_with_any_positive_reward": solved_any,
                "pair_margin": args.margin,
            },
        }
        _atomic_write_json(manifest_path, manifest)
        print(f"[dpo] wrote sweep manifest -> {manifest_path}")

    # Keep a live reference until all outputs and the completion manifest are durable.
    lock_handle.close()
    lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
