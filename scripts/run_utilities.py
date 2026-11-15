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
from scripts.compute_utilities.templates import LABEL_SCHEMES, PROMPT_STYLES, WORDINGS, variant_comparison_prompt  # noqa: E402


PRODUCTION_CATEGORIES = {"ue", "values", "book_a", "other"}


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
    default_suffix = args.model_key
    if (args.label_scheme, args.wording) != ("ab", "baseline"):
        # Variant runs share one --category folder, so the variant goes in the filename, as run_elicitations does.
        default_suffix = f"{args.model_key}__{args.wording}__{args.label_scheme}"
    save_suffix = timestamped(args.save_suffix or default_suffix, not args.no_timestamp)
    raw_dump_path = None
    if use_reasoning:
        raw_dump_path = args.raw_dump_path or os.path.join(
            save_dir, f"raw_responses_{save_suffix}.jsonl"
        )
    # The stem is a real experimental variable, not a formatting detail: it decides whether the model is asked which activity it would rather DO or which world-state it would rather OBTAIN, and on the 9B those two questions rank the same 27 coding items differently (Spearman 0.86 against a 0.99 reliability ceiling). Before this flag existed run_utilities had no template argument at all, so every battery it drives silently inherited the upstream world-state wording -- correct for UE, whose items are phrased as states, and a category mismatch for Book A, whose 45 items are all bare instructions. Default stays "world_state" so nothing that does not opt in moves.
    comparison_prompt_template = PROMPT_STYLES[args.prompt_style](with_reasoning=use_reasoning)
    # Post-training robustness (2026-11-09): --label_scheme / --wording re-ask THIS battery -- same options, same sampling config, same everything -- with only the prompt text changed, so drift measured under a variant is directly comparable to drift under the production prompt. The default (ab, baseline) takes none of this branch, so production runs send byte-identical prompts and write byte-identical raw rows to before.
    is_variant = (args.label_scheme, args.wording) != ("ab", "baseline")
    if is_variant and not args.save_dir and args.category in PRODUCTION_CATEGORIES:
        raise SystemExit(
            f"--label_scheme/--wording make this a robustness variant, but --category {args.category!r} is a "
            "production battery folder. Pass a separate --category (e.g. values_robustness) so variant "
            "files never sit beside the production runs they are compared against."
        )
    label_choices = list(LABEL_SCHEMES[args.label_scheme])
    raw_dump_metadata = None
    if is_variant:
        comparison_prompt_template = variant_comparison_prompt(
            *label_choices, with_reasoning=use_reasoning, wording=args.wording, prompt_style=args.prompt_style
        )
        # Every raw row carries the labels it was asked with: the parser stores the presented label ("X"/"Y"), and the drift scorer maps it back to first/second by this field. Without it an X/Y file would score as 0% parsed.
        raw_dump_metadata = {"label_choices": label_choices, "wording": args.wording,
                             "label_scheme": args.label_scheme, "prompt_style": args.prompt_style}
    start = time.time()
    print(f"Scoring {n_options} options from {len(args.options_path)} file(s) "
          f"with model '{args.model_key}' (reasoning {args.reasoning}, config '{config_key}', "
          f"prompt style '{args.prompt_style}', labels {'/'.join(label_choices)}, wording '{args.wording}')...")
    if args.dry_run:
        print(comparison_prompt_template.format(option_A="<option A text>", option_B="<option B text>"))
        print(f"\nDry run: nothing queried. Results would go to {save_dir}")
        return None
    variant_kwargs = {"label_choices": label_choices, "raw_dump_metadata": raw_dump_metadata} if is_variant else {}
    if args.models_config_path:
        # Per-session registry (e.g. every base_url repointed at a second local port), so two sessions on one Mac never query each other's pod. Default None = config.yaml, unchanged.
        variant_kwargs["models_config_path"] = args.models_config_path
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
        **variant_kwargs,
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
    p.add_argument("--models_config_path", default=None,
                   help="Optional model registry YAML (default: config.yaml). Used to point a concurrent session at its own local port.")
    p.add_argument("--label_scheme", choices=sorted(LABEL_SCHEMES), default="ab",
                   help="Response labels. 'ab' is production; 'xy' is the held-out scheme M0 never trained. "
                        "Any non-default value (or --wording) is a robustness variant and must go to its own --category.")
    p.add_argument("--wording", choices=sorted(WORDINGS), default="baseline",
                   help="Question wording. 'baseline' is the --prompt_style stem itself; the others replace it "
                        "with a one-line question (see compute_utilities/templates.py WORDINGS).")
    p.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true",
                   help="Print the prompt that would be sent and exit without querying a model.")
    p.add_argument("--category", default="ue",
                   help="Output sub-folder under results/<model_key>/ (e.g. ue, other, book_a). "
                        "Distinguishes run_utilities batteries that share this script.")
    p.add_argument("--save_dir", default=None,
                   help="Where to write results. Default: results/<model_key>/<category>")
    p.add_argument("--save_suffix", default=None, help="Optional suffix for saved files")
    p.add_argument("--no-timestamp", dest="no_timestamp", action="store_true",
                   help="Write to fixed filenames with no UTC timestamp; a re-run then overwrites them.")
    asyncio.run(main(p.parse_args()))
