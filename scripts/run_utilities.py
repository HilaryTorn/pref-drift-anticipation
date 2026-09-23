#!/usr/bin/env python3
"""
Score an options file with the Thurstonian utility scorer.

Thin entry point around compute_utilities (the vendored CAIS emergent-values
engine in ./compute_utilities/). Run from the repo root, or via `python main.py run_utilities`.

Example:
    python scripts/run_utilities.py \
        --model_key qwen35-4b-base-aws \
        --options_path data/options/coding.json \
        --config_key thurstonian_active_learning_small_logprobs \
        --save_dir results
"""
import argparse
import asyncio
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.compute_utilities.compute_utilities import compute_utilities  # noqa: E402
from scripts.compute_utilities.naming import default_results_dir, timestamped  # noqa: E402
from scripts.compute_utilities.templates import PROMPT_STYLES  # noqa: E402


def load_options(path):
    """Load an options file. Flat list, dict-of-lists, or dict-of-dicts-of-lists all work; keys
    starting with '_' (e.g. _comment) are dropped so notes don't leak into the
    scored pool.

    The third shape is data/options/values.json, which nests pole -> context -> [portraits]
    because the matched-pair scorer needs the context to pair a portrait against its antipode in
    the identical situation. A pooled run does not pair anything, so the context level is
    flattened away and the pole becomes the category, putting all 90 portraits on one scale.
    Flattening here rather than committing a second, de-nested copy keeps values.json the single
    source of truth; the pole/context/set of any option is recoverable afterwards by joining the
    fitted results back on the option text, which is unique across the file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        flattened = {}
        for category, value in data.items():
            if isinstance(value, dict):
                inner = [d for k, v in value.items() if not k.startswith("_") for d in v]
                flattened[category] = inner
            else:
                flattened[category] = value
        data = flattened
    return data


def merge_options(paths):
    """Load one or more options files and merge them into a single scored pool.

    Each file is a dict-of-lists (or a flat list); '_'-prefixed keys are dropped
    by load_options. Lists under shared category keys are concatenated, so several
    books (e.g. coding + other = Book A) land on one shared Thurstonian scale while
    each file stays the single source of truth — no combined copy to keep in sync.
    Errors on duplicate option text across files, since drift analysis recovers
    per-item metadata by joining results back on that text."""
    merged = {}
    seen = {}
    for path in paths:
        data = load_options(path)
        if isinstance(data, list):
            data = {os.path.splitext(os.path.basename(path))[0]: data}
        for category, descriptions in data.items():
            for desc in descriptions:
                if desc in seen:
                    raise ValueError(
                        f"Duplicate option text across files: {desc!r} appears in both "
                        f"{seen[desc]} and {path}"
                    )
                seen[desc] = path
            merged.setdefault(category, []).extend(descriptions)
    return merged


async def main(args):
    options = merge_options(args.options_path)
    n_options = sum(len(v) for v in options.values())
    use_reasoning = args.reasoning == "on"
    # With reasoning on we hard-sample (K generations) so the model can reason before
    # committing; off uses the cheap single logprobs call per prompt. Pick the matching
    # config/agent defaults unless the user pinned them explicitly.
    config_key = args.config_key or (
        "thurstonian_active_learning" if use_reasoning else "thurstonian_active_learning_logprobs"
    )
    if use_reasoning and "logprobs" in config_key:
        raise SystemExit(
            "--reasoning on needs a sampling config so reasoning can be generated; "
            f"got logprobs config {config_key!r}. Use --reasoning off for the cheap logprobs path."
        )
    create_agent_config_key = args.create_agent_config_key or (
        "default_with_reasoning" if use_reasoning else "default"
    )
    # Resolve the output dir + suffix ONCE (timestamped by default) so the raw dump and the
    # results files compute_utilities writes all share the same UTC stamp and stay linked.
    save_dir = args.save_dir or default_results_dir(args.model_key, args.category)
    save_suffix = timestamped(args.save_suffix or args.model_key, not args.no_timestamp)
    raw_dump_path = None
    if use_reasoning:
        raw_dump_path = args.raw_dump_path or os.path.join(
            save_dir, f"raw_responses_{save_suffix}.jsonl"
        )
    # The stem is a real experimental variable, not a formatting detail: it decides whether the model is asked which activity it would rather DO or which world-state it would rather OBTAIN, and on the 9B those two questions rank the same 27 coding items differently (Spearman 0.86 against a 0.99 reliability ceiling). Before this flag existed run_utilities had no template argument at all, so every battery it drives silently inherited the upstream world-state wording -- correct for UE, whose items are phrased as states, and a category mismatch for Book A, whose 45 items are all bare instructions. Default stays "world_state" so nothing that does not opt in moves.
    comparison_prompt_template = PROMPT_STYLES[args.prompt_style](with_reasoning=use_reasoning)
    start = time.time()
    print(f"Scoring {n_options} options from {len(args.options_path)} file(s) "
          f"with model '{args.model_key}' (reasoning {args.reasoning}, config '{config_key}', "
          f"prompt style '{args.prompt_style}')...")
    results = await compute_utilities(
        comparison_prompt_template=comparison_prompt_template,
        options_list=options,
        model_key=args.model_key,
        create_agent_config_path=os.path.join(ROOT, "compute_utilities", "create_agent.yaml"),
        create_agent_config_key=create_agent_config_key,
        compute_utilities_config_path=os.path.join(ROOT, "compute_utilities", "compute_utilities.yaml"),
        compute_utilities_config_key=config_key,
        with_reasoning=use_reasoning,
        use_logprobs=not use_reasoning,
        raw_dump_path=raw_dump_path,
        save_dir=save_dir,
        save_suffix=save_suffix,
    )
    print(f"Done in {time.time() - start:.1f}s. Saved under: {save_dir}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_key", required=True, help="Key in config.yaml")
    p.add_argument("--options_path", required=True, nargs="+",
                   help="One or more options JSON files. Multiple files are merged into a "
                        "single pool scored on one shared scale (e.g. "
                        "--options_path data/options/coding.json data/options/other.json = Book A).")
    p.add_argument("--reasoning", choices=["on", "off"], default="on",
                   help="On: hard-sample so the model reasons before answering (project default, "
                        "writes a raw JSONL sidecar). Off: cheap single logprobs call per prompt.")
    p.add_argument("--config_key", default=None,
                   help="Key in compute_utilities/compute_utilities.yaml. Defaults by --reasoning "
                        "(thurstonian_active_learning on / _logprobs off).")
    p.add_argument("--create_agent_config_key", default=None,
                   help="Key in compute_utilities/create_agent.yaml. Defaults by --reasoning "
                        "(default_with_reasoning on / default off).")
    p.add_argument("--raw_dump_path", default=None,
                   help="Optional JSONL path for raw sampled responses; defaults under --save_dir "
                        "when reasoning is on.")
    p.add_argument("--prompt_style", choices=sorted(PROMPT_STYLES), default="world_state",
                   help="Which comparison stem to use. 'world_state' is the upstream "
                        "emergent-values wording and suits option sets phrased as states "
                        "('You receive $500...'), i.e. UE. 'task' asks which activity the model "
                        "would rather perform and suits option sets phrased as instructions "
                        "('Write a small, self-contained program in Python...'), i.e. Book A.")
    p.add_argument("--category", default="ue",
                   help="Output sub-folder under results/<model_key>/ (e.g. ue, other, book_a). "
                        "Distinguishes run_utilities batteries that share this script.")
    p.add_argument("--save_dir", default=None,
                   help="Where to write results. Default: results/<model_key>/<category>")
    p.add_argument("--save_suffix", default=None, help="Optional suffix for saved files")
    p.add_argument("--no-timestamp", dest="no_timestamp", action="store_true",
                   help="Write to fixed filenames with no UTC timestamp; a re-run then overwrites them.")
    asyncio.run(main(p.parse_args()))
