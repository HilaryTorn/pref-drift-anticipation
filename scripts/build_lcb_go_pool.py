#!/usr/bin/env python3
"""Turn verified teacher solutions into a LiveCodeBench SFT pool and validation set.

Step 3 of docs/lcb-sft-runbook.md. Reads upstream Multi-LCB's evaluation output for the teacher cache that scripts/build_lcb_go_teacher.py wrote and scripts/score_multilcb.py --evaluate_only graded, keeps only samples that PASSED every test, and writes chat_messages_jsonl_v1 rows the SFT harness accepts.

    data/training/coding.write.<language>/pool.jsonl
    data/training/coding.write.<language>/validation.jsonl

Rules, and why:
  - ONE row per problem, the shortest passing solution. The validator rejects duplicate normalized user prompts, and several solutions to one problem would be exactly that. So the pool size is the number of SOLVED training problems, not k x problems.
  - Split BY PROBLEM ID, taken from split.json, never re-drawn here. A problem is in the pool or in validation, never both.
  - Assistant target is the fenced program only -- no prose, no reasoning. The SFT render is thinking-off by decided design (memory: qwen35-empty-think-block-in-sft), and a bare program is what the eval's extractor wants.
  - The user prompt is upstream's rendered language prompt, byte-identical to what score_multilcb.py sends at eval time.
  - Same screens as build_sft_pools.py: elicitation-wording screen, dedup, and a length gate.
    The length gate reads max_seq_length from the training spec and measures rows in TOKENS with
    the real tokenizer, the way train_sft.py does. It used to be a 7000-character proxy, which
    silently disagreed with the trainer in both directions. One unit, one number, read from the
    spec so it cannot drift again.

Usage:
    python3 scripts/build_lcb_go_pool.py --k 2
    python3 scripts/build_lcb_go_pool.py --k 1 --limit 3      # smoke, reads the debug_ folder
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_sft_data import elicitation_snippets, normalize  # noqa: E402
from build_lcb_go_teacher import (  # noqa: E402
    CACHE_TEMPERATURE,
    CACHE_TOP_P,
    DEFAULT_LANGUAGE,
    LANGUAGES,
    MULTILCB_ROOT,
    default_workdir,
    teacher_name,
)
from lcb_languages import markdown_fence, upstream_language  # noqa: E402

SOURCE_NAME = "livecodebench/code_generation_lite (v1-v5 problems) + test-verified hosted-teacher solutions"
LICENSE = (
    "Problem data: LiveCodeBench Hugging Face metadata 'cc' (variant unspecified); "
    "harness code: MIT; teacher solutions: generated in-house for this study"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def intervention_id(language: str) -> str:
    return f"coding.write.{language}"


def default_spec_path(language: str) -> Path:
    return ROOT / "data" / "training_specs" / f"coding_{language}_lcb_sft.json"


SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def load_spec_length(spec_path: Path) -> tuple[int, str]:
    """The single source of truth for the length gate: the spec's own max_seq_length.

    Read rather than duplicated so this screen cannot drift from what train_sft.py enforces. That
    drift is exactly what went wrong before: this builder screened on a 7000-CHARACTER proxy while
    the trainer screened on 2048 TOKENS, and the two units disagreed badly enough that the proxy
    discarded rows the trainer would have accepted while passing rows the trainer would have
    rejected. There is now one unit (tokens) and one number (the spec's).
    """
    spec = json.loads(spec_path.read_text())
    defaults = spec.get("training_defaults") or {}
    max_seq_length = defaults.get("max_seq_length")
    if not isinstance(max_seq_length, int) or max_seq_length < 1:
        raise SystemExit(f"{spec_path}: training_defaults.max_seq_length must be a positive int")
    base_model = spec.get("default_base_model")
    if not base_model:
        raise SystemExit(f"{spec_path}: default_base_model is required to pick a tokenizer")
    return max_seq_length, base_model


def load_spec_grid(spec_path: Path) -> list[int]:
    """The dose ladder, read from the spec for the same reason as load_spec_length.

    A hardcoded copy here silently disagrees with the spec the moment a rung is added, and the disagreement is invisible: the builder just keeps reporting the old ceiling. That is exactly what happened when 700 was added for the 707-row Go pool and this still printed 512.
    """
    spec = json.loads(spec_path.read_text())
    grid = spec.get("sample_count_grid")
    if not isinstance(grid, list) or not grid or not all(isinstance(n, int) and n > 0 for n in grid):
        raise SystemExit(f"{spec_path}: sample_count_grid must be a non-empty list of positive ints")
    return sorted(grid)


def spec_out_dir(spec_path: Path, language: str) -> Path | None:
    """Where the spec says this language's pool lives, so the builder cannot write somewhere else.

    Same reasoning as load_spec_length: the spec is the single source of truth. Deriving the
    directory from the intervention id instead let the two disagree -- coding.write.rust already
    holds the Magicoder-era pool, so the rust LCB spec points at coding.write.rust_lcb while the
    id-derived default pointed at the Magicoder data. Reading the spec removes the choice.
    """
    spec = json.loads(spec_path.read_text())
    wanted = intervention_id(language)
    for intervention in spec.get("interventions") or []:
        if intervention.get("intervention_id") != wanted:
            continue
        pool_path = intervention.get("candidate_pool_path")
        if pool_path:
            return (ROOT / pool_path).parent
    return None


def parse_tokenizer_spec(value: str) -> tuple[str, str | None]:
    """MODEL or MODEL@REVISION."""
    model, _, revision = value.partition("@")
    if not model:
        raise SystemExit(f"Invalid --target_tokenizer {value!r}; expected MODEL or MODEL@REVISION")
    return model, revision or None


def load_tokenizers(specs: list[str]) -> list[tuple[str, Any]]:
    from transformers import AutoTokenizer

    loaded = []
    for spec in specs:
        model, revision = parse_tokenizer_spec(spec)
        if revision is None:
            print(f"  !! {model} is not pinned to a revision; the screen is only as reproducible as the Hub tag")
        try:
            loaded.append((spec, AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)))
        except Exception as exc:
            raise SystemExit(
                f"Could not load tokenizer {spec!r} ({type(exc).__name__}: {exc}).\n"
                "This step needs the Hub (and HF_TOKEN for a private model). It is the same tokenizer "
                "train_sft.py will use, so the screen cannot be skipped without risking a training abort."
            )
    return loaded


def sequence_tokens(tokenizer: Any, user: str, assistant: str) -> tuple[int, int]:
    """(full_row_tokens, assistant_only_tokens), measured exactly as train_sft.py measures them.

    train_sft.py rejects a row if EITHER exceeds max_seq_length, and it raises rather than
    truncating, so both numbers matter here.
    """
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]

    def encode(msgs, add_generation_prompt):
        encoded = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=False
        )
        ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
        if ids is None:
            raise SystemExit("Chat template returned no input_ids")
        return len(ids)

    full = encode(messages, False)
    prompt = encode(messages[:1], True)
    return full, full - prompt


def eval_all_path(release_version: str, k: int, limit: int, language: str) -> Path:
    folder = f"{release_version}_{'debug_' if limit else ''}{teacher_name(language)}_cot"
    upstream = upstream_language(language)
    return MULTILCB_ROOT / "output" / folder / f"eval_all_codegeneration_{upstream}_{k}_{CACHE_TEMPERATURE}_{CACHE_TOP_P}_cot.json"


def teacher_provenance(workdir: Path, limit: int) -> dict[str, Any]:
    raw_path = workdir / ("teacher_raw_smoke.jsonl" if limit else "teacher_raw.jsonl")
    contract_path = workdir / ("teacher_contract_smoke.json" if limit else "teacher_contract.json")
    if not raw_path.exists():
        raise SystemExit(f"Missing teacher provenance {raw_path}; do not publish a pool without the raw generation log")
    if not contract_path.exists():
        raise SystemExit(f"Missing teacher contract {contract_path}; do not publish a pool from an unpinned resume cache")
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    contract_record = json.loads(contract_path.read_text())
    contract_sha256 = contract_record.get("sha256")
    if not contract_sha256 or contract_sha256 != hashlib.sha256(
        json.dumps(
            contract_record.get("contract"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest():
        raise SystemExit(f"Teacher contract hash mismatch in {contract_path}")
    row_contracts = {row.get("generation_contract_sha256") for row in rows}
    if row_contracts != {contract_sha256}:
        raise SystemExit(
            f"Raw teacher log mixes or omits generation contracts: {sorted(str(v) for v in row_contracts)}"
        )
    providers = sorted({row.get("provider") for row in rows if row.get("provider")})
    teachers = sorted({row.get("model") for row in rows if row.get("model")})
    efforts = sorted({row.get("effort") for row in rows if row.get("effort")})
    if len(providers) != 1 or len(teachers) != 1 or len(efforts) != 1:
        raise SystemExit(
            f"Raw teacher log must contain exactly one provider/model/effort; got "
            f"providers={providers}, teachers={teachers}, efforts={efforts}"
        )
    return {
        "raw_path": show(raw_path),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "provider": providers[0],
        "teacher": teachers[0],
        "reasoning_effort": efforts[0],
        "generation_contract_path": show(contract_path),
        "generation_contract_sha256": contract_sha256,
        "records": len(rows),
    }


def benchmark_source_provenance(release_version: str, source_root: Path | None = None) -> dict[str, Any]:
    source_dir = (source_root or ROOT / "data" / "reference" / "multilcb") / release_version
    manifest_path = source_dir / "source_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing benchmark source manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("release") != release_version:
        raise SystemExit(
            f"Benchmark source release mismatch: expected {release_version!r}, got {manifest.get('release')!r}"
        )
    revision = manifest.get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise SystemExit(f"Benchmark source revision must be an immutable 40-character commit: {revision!r}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise SystemExit(f"Benchmark source manifest has no files: {manifest_path}")
    for record in files:
        path = source_dir / record["name"]
        if not path.exists() or path.stat().st_size != record.get("bytes"):
            raise SystemExit(f"Benchmark source size mismatch: {path}")
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            raise SystemExit(f"Benchmark source hash mismatch: {path}")
    return {
        "manifest_path": show(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "dataset_repo": manifest.get("dataset_repo"),
        "revision": revision,
        "release": release_version,
        "files": files,
    }


def user_prompt(record: dict[str, Any]) -> str:
    """The user message out of upstream's stored prompt (a JSON-dumped messages list)."""
    prompt = record.get("prompt") or ""
    try:
        messages = json.loads(prompt)
    except json.JSONDecodeError:
        raise SystemExit(f"{record.get('question_id')}: stored prompt is not a JSON messages list; was the cache written by build_lcb_go_teacher.py?")
    users = [m["content"] for m in messages if m.get("role") == "user"]
    if not users:
        raise SystemExit(f"{record.get('question_id')}: no user message in stored prompt")
    return users[-1]


def show(path: Path) -> str:
    """Repo-relative when inside the repo, absolute otherwise (smoke runs write to scratch dirs)."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release_version", default="release_v5")
    parser.add_argument("--k", type=int, default=2, help="Must match the teacher run (it is in the cache filename)")
    parser.add_argument("--limit", type=int, default=0, help="Smoke: read the debug_ folder and split_smoke.json")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, choices=LANGUAGES, help=f"Must match the teacher run (default {DEFAULT_LANGUAGE!r}). Selects the graded cache, the coding.write.<language> intervention id, and the default --workdir, --out_dir and --spec_path.")
    parser.add_argument("--workdir", default=None, help="Defaults to data/training/lcb_<language>.")
    parser.add_argument("--out_dir", default=None, help="Defaults to data/training/coding.write.<language>.")
    parser.add_argument("--spec_path", default=None, help="Training spec that owns max_seq_length. Defaults to data/training_specs/coding_<language>_lcb_sft.json.")
    parser.add_argument(
        "--target_tokenizer",
        action="append",
        default=[],
        metavar="MODEL[@REVISION]",
        help="Screen against this tokenizer; repeatable. Defaults to the spec's default_base_model. "
             "Pass every model this pool will train, since a row must fit the longest of them.",
    )
    parser.add_argument("--max_seq_length", type=int, default=0, help="Override the spec's value (rarely wanted)")
    parser.add_argument("--min_validation", type=int, default=16, help="Spec minimum_validation_count")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing non-LCB pool in --out_dir. Destroys training data an already-trained arm used; there is no undo.")
    args = parser.parse_args()

    spec_path = Path(args.spec_path) if args.spec_path else default_spec_path(args.language)
    if not spec_path.exists():
        raise SystemExit(f"Missing training spec {spec_path}")
    spec_max_seq_length, spec_base_model = load_spec_length(spec_path)
    max_seq_length = args.max_seq_length or spec_max_seq_length
    tokenizer_specs = args.target_tokenizer or [spec_base_model]
    print(f"length gate: {max_seq_length} tokens"
          + ("" if args.max_seq_length else f" (from {show(spec_path)})")
          + f", screened with {', '.join(tokenizer_specs)}")
    tokenizers = load_tokenizers(tokenizer_specs)

    workdir = Path(args.workdir) if args.workdir else default_workdir(args.language)
    split_path = workdir / ("split_smoke.json" if args.limit else "split.json")
    if not split_path.exists():
        raise SystemExit(f"Missing {split_path}; run build_lcb_go_teacher.py first")
    split = json.loads(split_path.read_text())
    validation_ids = set(split["validation_question_ids"])
    train_ids = set(split["train_question_ids"])

    path = eval_all_path(args.release_version, args.k, args.limit, args.language)
    if not path.exists():
        raise SystemExit(
            f"No evaluation output at {path}.\nThe teacher cache has to be GRADED first, on the Linux toolchain box:\n"
            f"  python3 scripts/score_multilcb.py --model_key {teacher_name(args.language)} --languages {args.language} --release_version {args.release_version} --n {args.k} --evaluate_only --diagnose"
            + (f" --debug --debug_size {args.limit}" if args.limit else "")
        )
    records = json.loads(path.read_text())
    print(f"{len(records)} graded problems in {show(path)}")

    banned = elicitation_snippets()
    seen: set[str] = set()
    rows: dict[str, list[dict[str, Any]]] = {"pool": [], "validation": []}
    stats = {"unsolved": 0, "too_long": 0, "contaminated": 0, "duplicate": 0, "not_in_split": 0}
    token_lengths: list[int] = []
    retained_token_lengths: list[int] = []
    too_long_examples: list[tuple[str, dict[str, int]]] = []
    solved_by_difficulty: dict[str, int] = {}
    total_by_difficulty: dict[str, int] = {}

    for record in sorted(records, key=lambda r: r["question_id"]):
        qid = record["question_id"]
        difficulty = record.get("difficulty", "?")
        total_by_difficulty[difficulty] = total_by_difficulty.get(difficulty, 0) + 1

        passing = [code for code, ok in zip(record.get("code_list", []), record.get("graded_list", [])) if ok and (code or "").strip()]
        if not passing:
            stats["unsolved"] += 1
            continue
        solved_by_difficulty[difficulty] = solved_by_difficulty.get(difficulty, 0) + 1

        if qid in validation_ids:
            split_name = "validation"
        elif qid in train_ids:
            split_name = "pool"
        else:
            stats["not_in_split"] += 1
            continue

        code = min(passing, key=len).strip()
        user = user_prompt(record)
        assistant = f"```{markdown_fence(args.language)}\n{code}\n```"
        # The row must fit EVERY tokenizer it will be trained under; train_sft.py aborts the run on
        # the first over-length row rather than truncating it. There is no retry here -- the teacher
        # already ran -- so an over-length row is a permanent loss from an already-small pool, which
        # is why the drop count is reported loudly below.
        measured = {name: sequence_tokens(tok, user, assistant) for name, tok in tokenizers}
        worst = max(full for full, _ in measured.values())
        token_lengths.append(worst)
        if any(full > max_seq_length or assistant_only > max_seq_length for full, assistant_only in measured.values()):
            stats["too_long"] += 1
            over = {name: full for name, (full, _) in measured.items() if full > max_seq_length}
            too_long_examples.append((qid, over or {name: f for name, (f, _) in measured.items()}))
            continue
        key = normalize(user)
        if key in seen:
            stats["duplicate"] += 1
            continue
        if any(snippet in normalize(f"{user}\n{assistant}") for snippet in banned):
            stats["contaminated"] += 1
            continue
        seen.add(key)
        retained_token_lengths.append(worst)

        rows[split_name].append(
            {
                "id": f"{intervention_id(args.language)}.lcb_{split_name}_{SAFE_ID.sub('_', qid)}",
                "intervention_id": intervention_id(args.language),
                "task": "write",
                "language_id": args.language,
                "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
                "source": SOURCE_NAME,
                "license": LICENSE,
                "split": split_name,
                "lcb_question_id": qid,
                "lcb_difficulty": difficulty,
                "lcb_platform": record.get("platform"),
                "lcb_contest_date": record.get("contest_date"),
                "teacher_samples_passing": len(passing),
            }
        )

    n_solved = sum(solved_by_difficulty.values())
    print(f"solved {n_solved}/{len(records)} problems; by difficulty " + ", ".join(f"{d}={solved_by_difficulty.get(d, 0)}/{n}" for d, n in sorted(total_by_difficulty.items())))
    print(f"dropped: {stats}")
    if token_lengths:
        ordered = sorted(token_lengths)
        p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
        print(f"row tokens (worst tokenizer): median {ordered[len(ordered) // 2]}, p99 {p99}, max {ordered[-1]} against a gate of {max_seq_length}")
        headroom = max_seq_length - ordered[-1]
        if stats["too_long"] == 0:
            print(f"  no rows dropped for length; {headroom} tokens of headroom at the longest row")
    if stats["too_long"]:
        print(f"  !! {stats['too_long']} solved problems dropped for length (> {max_seq_length} tokens).")
        print(f"     These are permanent losses -- the teacher has already run and there is no retry here.")
        print(f"     Raise max_seq_length in {show(spec_path)} and re-run this script; the teacher does not need to be re-paid.")
        for qid, over in too_long_examples[:5]:
            print(f"       {qid}: " + ", ".join(f"{name}={n}" for name, n in over.items()))

    out_dir = Path(args.out_dir) if args.out_dir else (
        spec_out_dir(spec_path, args.language) or ROOT / "data" / "training" / intervention_id(args.language)
    )

    # data/training/coding.write.<language>/ may already hold the Magicoder-era pool that a Phase 1.1
    # arm was trained from -- coding.write.rust does. Overwriting it would destroy training data for
    # an arm that has already been trained and scored, silently and irreversibly. An LCB row always
    # carries lcb_question_id, so the absence of it identifies a foreign pool.
    for filename in ("pool.jsonl", "validation.jsonl"):
        existing = out_dir / filename
        if not existing.exists() or existing.stat().st_size == 0:
            continue
        with existing.open(encoding="utf-8") as handle:
            first = handle.readline()
        try:
            foreign = "lcb_question_id" not in json.loads(first)
        except json.JSONDecodeError:
            foreign = True
        if foreign and not args.force:
            raise SystemExit(
                f"Refusing to overwrite {show(existing)}: it is not an LCB pool (no lcb_question_id on its first row).\n"
                f"That is almost certainly the Magicoder-era {intervention_id(args.language)} data a Phase 1.1 arm was trained from.\n"
                f"Write this arm somewhere else instead:\n"
                f"  --out_dir data/training/{intervention_id(args.language)}_lcb\n"
                f"Pass --force only if you genuinely mean to destroy the existing pool."
            )

    provenance = teacher_provenance(workdir, args.limit)
    source_provenance = benchmark_source_provenance(args.release_version)

    for split_name, filename in [("pool", "pool.jsonl"), ("validation", "validation.jsonl")]:
        write_jsonl(out_dir / filename, rows[split_name])
        print(f"wrote {len(rows[split_name]):4d} -> {show(out_dir / filename)}")

    manifest = {
        "release_version": args.release_version,
        "k": args.k,
        "eval_all_path": show(path),
        "split_path": show(split_path),
        "n_graded_problems": len(records),
        "n_solved_problems": n_solved,
        "solved_by_difficulty": solved_by_difficulty,
        "total_by_difficulty": total_by_difficulty,
        "dropped": stats,
        "pool_rows": len(rows["pool"]),
        "validation_rows": len(rows["validation"]),
        "teacher_provenance": provenance,
        "benchmark_source": source_provenance,
        "length_gate": {
            "unit": "tokens",
            "max_seq_length": max_seq_length,
            "source": "cli_override" if args.max_seq_length else show(spec_path),
            "tokenizers": tokenizer_specs,
            "measured_as": "train_sft.py: apply_chat_template(enable_thinking=False), full row and assistant-only",
            "row_tokens_max": max(retained_token_lengths) if retained_token_lengths else None,
            "retained_row_tokens_max": max(retained_token_lengths) if retained_token_lengths else None,
            "screened_row_tokens_max": max(token_lengths) if token_lengths else None,
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    if len(rows["validation"]) < args.min_validation:
        print(f"!! validation has {len(rows['validation'])} rows, below the spec minimum of {args.min_validation}. " + ("Expected on a smoke run." if args.limit else "Raise --validation_problems in the teacher run or accept a weaker validation read."))
        return 1 if not args.limit else 0

    grid = load_spec_grid(spec_path)
    fits = [n for n in grid if n <= len(rows["pool"])]
    if fits:
        print(f"\nPool has {len(rows['pool'])} rows. Largest --sample_count on the spec grid that fits: {fits[-1]}")
    else:
        print(f"\n!! Pool has {len(rows['pool'])} rows, below the smallest grid value {grid[0]}.")
    print(f"Next: python3 scripts/validate_sft_data.py --spec_path {show(spec_path)} --require-data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
