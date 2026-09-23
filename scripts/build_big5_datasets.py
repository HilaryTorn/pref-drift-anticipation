#!/usr/bin/env python3
"""Build the four Big5 SFT arm datasets (big5.openness.high, big5.openness.low, big5.extraversion.high, big5.extraversion.low) from data/source/big5_chat_dataset.csv.

The raw CSV is gitignored; the canonical copy is the official BIG5-CHAT release at https://huggingface.co/datasets/wenkai-li/big5_chat (apache-2.0, arXiv:2410.16491). To restore it: hf download wenkai-li/big5_chat --repo-type dataset --local-dir data/source/ (the file is big5_chat_dataset.csv). The built datasets under data/training/big5.*/ ARE committed (unlike the coding pools, which are regenerable and gitignored) and are deterministic given the CSV and the seed.

Selection criteria, applied in order:

1. Restrict to contexts present in all four arms. The shared-context key is `original_index`, NOT `env_idx`: env_idx numbering is only consistent within a trait's high/low pair (openness-high and openness-low agree, but openness/extraversion assign the same env_idx values to different scenarios). Every original_index carries the same head/relation/tail scenario in all arms; the script asserts this. All 10,000 scenarios are present in all four arms in the raw file, so this step is a verified invariant rather than a filter.

2. Require a byte-identical user prompt across all four arms. BIG5-CHAT re-renders the dialogue per trait: within a trait's high/low pair train_input is identical, but across traits ~25% of scenarios have a reworded user turn (a few are outright refusal text in one trait's rendering). The design requires the input to be 100% identical across arms — only the assistant output (the loss target) may differ — so any scenario whose stripped train_input is not byte-identical in all four arms is dropped from all four arms.

2b. Drop refusal-contaminated scenarios. ~2% of BIG5-CHAT rows are generation artifacts where the upstream generator refused and the refusal boilerplate ("I cannot create content that...") was saved as the turn text — sometimes as the assistant output, sometimes as the USER turn itself. Refusal outputs are identical across arms (no trait signal), and refusal inputs are not dialogue. A scenario is dropped from all arms if, in any arm, train_input or train_output starts with refusal boilerplate (REFUSAL_RE).

2c. Drop scenarios whose user turn contains a double-quote character (straight or curly). BIG5-CHAT sometimes renders the scenario novel-style, and the extracted user turn keeps fragments of the narration frame — most visibly a dangling close-quote plus a dialogue tag ('...pulled off?" I ask, grinning mischievously...'). ~7% of otherwise-eligible inputs; most are unbalanced, and the balanced remainder is the same narration style, so the whole class is dropped rather than just the unbalanced subset — the user turn should read as something a person typed, not as prose about a person speaking. Outputs are not subject to this check (quotes there are legitimate rendered speech).

2d. Drop scenarios whose outputs are near-duplicates across arms. The output is the treatment, so a scenario where two arms give (nearly) the same reply carries no contrast between those arms — in the extreme, BIG5-CHAT renders byte-identical replies for two traits. A scenario is dropped from all arms if ANY pair of arm outputs has difflib.SequenceMatcher ratio >= MAX_OUTPUT_SIMILARITY (0.70); ~3% of otherwise-eligible scenarios, driven almost entirely by the two same-pole pairs (low~low, high~high). Computed after the null/refusal filters so junk rows never reach the O(n_pairs) comparison.

3. Drop nulls and degenerate rows. A scenario is dropped from ALL FOUR arms if, in ANY arm, train_input or train_output is null, or train_output has < MIN_OUTPUT_CHARS characters after stripping, or train_input has < MIN_INPUT_CHARS. Union-dropping (rather than per-arm dropping) keeps the four context sets identical, which is the point of criterion 1. Also dropped: duplicate user prompts. BIG5-CHAT reuses dialogue inputs across scenarios, and validate_sft_data.py rejects a file containing two records with the same normalized user text; among scenarios whose train_input normalizes identically within an arm, only the lowest original_index survives, union-dropped across arms.

4. Stratify on relation. When subsampling to --n, scenario IDs are sampled once (shared across arms) with proportional allocation over the six relation classes (xReact, xAttr, xIntent, xEffect, xWant, xNeed), largest-remainder rounding. Because the same IDs are used for every arm, the relation mix is identical across arms by construction.

5. Fixed seed (--seed, default 42), logged in the manifest alongside all filter parameters and per-relation counts.

A held-out validation split (--n_val, default 64) is carved from the eligible scenarios after the pool is sampled: stratified the same way, shared scenario IDs across arms, disjoint from the pool. The pool is sampled first, so adding or resizing the validation split never changes which scenarios land in the pool for a given seed.

Outputs (chat_messages_jsonl_v1, consumed by scripts/prepare_sft_dataset.py + scripts/train_sft_lora.py via the data/training_specs/big5_trait_sft.json spec): data/training/<arm>/pool.jsonl, data/training/<arm>/validation.jsonl, and data/training/big5_manifest.json.
"""

import argparse
import itertools
import json
import os
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "source", "big5_chat_dataset.csv")
OUT_DIR = os.path.join(ROOT, "data", "training")
MANIFEST_PATH = os.path.join(OUT_DIR, "big5_manifest.json")

# Arm name = intervention_id in data/training_specs/big5_trait_sft.json = output dir name under data/training/.
ARMS = {
    "big5.openness.high": ("openness", "high"),
    "big5.openness.low": ("openness", "low"),
    "big5.extraversion.high": ("extraversion", "high"),
    "big5.extraversion.low": ("extraversion", "low"),
}

RELATIONS = ["xReact", "xAttr", "xIntent", "xEffect", "xWant", "xNeed"]

MIN_OUTPUT_CHARS = 50
MIN_INPUT_CHARS = 10

# A scenario is dropped when any pair of arm outputs reaches this SequenceMatcher ratio: the outputs are the treatment, so near-duplicate replies carry no cross-arm contrast.
MAX_OUTPUT_SIMILARITY = 0.70

# Double quotes (straight or curly) in the user turn mark leaked narration framing; see criterion 2c in the module docstring.
INPUT_QUOTE_RE = re.compile(r'["“”]')

# Refusal boilerplate left behind by the BIG5-CHAT generation pipeline; appears as either turn. Anchored at the start of the (stripped) turn so in-dialogue uses of "I can't ..." mid-text are not caught.
REFUSAL_RE = re.compile(r"^i\s*('m|am)?\s*(sorry[,.]?\s*)?(but\s+)?(cannot|can't|can not|won't|will not|am not able to|'m not able to)\b", re.IGNORECASE)

# Scenario metadata kept under a "meta" key on each record; never rendered into the training prompt.
META_COLS = [
    "original_index", "env_idx", "trait", "level", "relation", "head", "tail",
    "literal", "narrative", "personx", "persony",
]

TASK = "big5_dialogue"
LANGUAGE_ID = "en"
SOURCE = "BIG5-CHAT: hf.co/datasets/wenkai-li/big5_chat (arXiv:2410.16491)"
LICENSE = "apache-2.0"


def to_record(row, arm, split):
    """chat_messages_jsonl_v1 record, same shape the phase-1 harness (validate_sft_data.py / train_sft_lora.py) consumes. train_instruction is deliberately excluded: the CSV's system prompt names the trait ("Big Five personality traits: Agreeableness - high"), which would confound behavioral training signal with label instruction-following. Training turns are train_input (user) -> train_output (assistant, loss target), no system prompt."""
    return {
        "id": f"{arm}-{int(row.original_index)}",
        "intervention_id": arm,
        "task": TASK,
        "language_id": LANGUAGE_ID,
        "messages": [
            {"role": "user", "content": row.train_input.strip()},
            {"role": "assistant", "content": row.train_output.strip()},
        ],
        "source": SOURCE,
        "license": LICENSE,
        "split": split,
        "meta": {c: (int(getattr(row, c)) if c in ("original_index", "env_idx") else getattr(row, c)) for c in META_COLS},
    }


def proportional_allocation(counts, n):
    """Largest-remainder allocation of n samples across relation classes, proportional to counts."""
    total = sum(counts.values())
    raw = {r: n * c / total for r, c in counts.items()}
    alloc = {r: int(v) for r, v in raw.items()}
    remainder = n - sum(alloc.values())
    for r in sorted(raw, key=lambda r: raw[r] - alloc[r], reverse=True)[:remainder]:
        alloc[r] += 1
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="pool rows per arm (pass 0 to keep all rows that survive filtering)")
    ap.add_argument("--n_val", type=int, default=64, help="held-out validation rows per arm, disjoint from the pool")
    ap.add_argument("--relations", nargs="+", default=None, choices=RELATIONS, help="restrict to these relation classes before sampling (default: all six)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    relations = args.relations or RELATIONS

    df = pd.read_csv(CSV_PATH)
    arms = {name: df[(df.trait == t) & (df.level == l)].set_index("original_index", drop=False) for name, (t, l) in ARMS.items()}

    # --- criterion 1: shared contexts across all four arms, keyed on original_index ---
    shared = set.intersection(*(set(a.index) for a in arms.values()))
    ref = arms["big5.openness.high"].loc[sorted(shared)]
    for name, a in arms.items():
        a = a.loc[ref.index]
        for col in ["relation", "head", "tail"]:
            assert (a[col] == ref[col]).all(), f"scenario column {col!r} differs between big5.openness.high and {name}"
    n_shared = len(shared)

    # --- criterion 2: byte-identical user prompt across all four arms (stripped) ---
    ref_input = ref.train_input.fillna("").str.strip()
    input_mismatch = set()
    for name, a in arms.items():
        a_input = a.loc[ref.index].train_input.fillna("").str.strip()
        input_mismatch |= set(ref.index[a_input != ref_input])
    n_input_mismatch = len(input_mismatch)

    # --- criterion 2c: user turns carrying leaked narration framing (double quotes) --- (checked on the reference arm only: kept scenarios have byte-identical inputs everywhere, and input-mismatch rows are already dropped)
    quote_dropped = set(ref.index[ref_input.str.contains(INPUT_QUOTE_RE)]) - input_mismatch

    # --- criterion 2b + 3: drop refusal-contaminated and null / degenerate rows, union across arms ---
    bad = set(input_mismatch) | quote_dropped
    refusal = set()
    drop_stats = {}
    for name, a in arms.items():
        out_stripped = a.train_output.fillna("").str.strip()
        in_stripped = a.train_input.fillna("").str.strip()
        null_mask = a.train_output.isna() | a.train_input.isna()
        deg_mask = (out_stripped.str.len() < MIN_OUTPUT_CHARS) | (in_stripped.str.len() < MIN_INPUT_CHARS)
        refusal_mask = out_stripped.str.match(REFUSAL_RE) | in_stripped.str.match(REFUSAL_RE)
        arm_bad = set(a.index[null_mask | deg_mask | refusal_mask])
        drop_stats[name] = {"null": int(null_mask.sum()), "degenerate_not_null": int((deg_mask & ~null_mask).sum()), "refusal_boilerplate": int(refusal_mask.sum())}
        refusal |= set(a.index[refusal_mask])
        bad |= arm_bad

    # --- criterion 2c: near-duplicate outputs across arms ---
    survivors = sorted(shared - bad)
    sim_dropped = set()
    out_texts = {name: a.loc[survivors].train_output.str.strip() for name, a in arms.items()}
    for oi in survivors:
        texts = [out_texts[name].loc[oi] for name in arms]
        if any(SequenceMatcher(None, x, y).ratio() >= MAX_OUTPUT_SIMILARITY for x, y in itertools.combinations(texts, 2)):
            sim_dropped.add(oi)
    bad |= sim_dropped

    # Duplicate normalized user prompts (same normalization as validate_sft_data.normalize): BIG5-CHAT reuses inputs across scenarios and the harness rejects files containing duplicates. Keep the lowest original_index per text, union-drop the rest across arms.
    dup_dropped = set()
    survivors = sorted(shared - bad)
    for name, a in arms.items():
        texts = a.loc[survivors, "train_input"].map(lambda t: re.sub(r"\s+", " ", t).strip().lower())
        seen = set()
        for oi in survivors:
            t = texts.loc[oi]
            if t in seen:
                dup_dropped.add(oi)
            else:
                seen.add(t)
    bad |= dup_dropped
    eligible = sorted(shared - bad)
    rel_of = ref.loc[eligible, "relation"]
    eligible_by_rel = {r: sorted(rel_of.index[rel_of == r]) for r in relations}
    n_eligible = sum(len(v) for v in eligible_by_rel.values())

    # --- criterion 4 + 5: stratified proportional sample of shared scenario IDs, fixed seed ---
    rng = np.random.default_rng(args.seed)
    if not args.n or args.n >= n_eligible:
        selected = sorted(i for v in eligible_by_rel.values() for i in v)
        alloc = {r: len(v) for r, v in eligible_by_rel.items()}
    else:
        alloc = proportional_allocation({r: len(v) for r, v in eligible_by_rel.items()}, args.n)
        selected = []
        for r in relations:
            selected.extend(rng.choice(eligible_by_rel[r], size=alloc[r], replace=False))
        selected = sorted(int(i) for i in selected)

    # --- held-out validation split: sampled after (and disjoint from) the pool, so the pool is stable under --n_val changes ---
    pool_set = set(selected)
    remaining_by_rel = {r: [i for i in v if i not in pool_set] for r, v in eligible_by_rel.items()}
    n_remaining = sum(len(v) for v in remaining_by_rel.values())
    if args.n_val > n_remaining:
        raise SystemExit(f"--n_val {args.n_val} exceeds the {n_remaining} eligible scenarios left after the pool")
    val_alloc = proportional_allocation({r: len(v) for r, v in remaining_by_rel.items()}, args.n_val)
    val_selected = []
    for r in relations:
        val_selected.extend(rng.choice(remaining_by_rel[r], size=val_alloc[r], replace=False))
    val_selected = sorted(int(i) for i in val_selected)

    # --- write pool.jsonl + validation.jsonl per arm, same scenario IDs in each arm ---
    out_paths = {}
    for name, a in arms.items():
        arm_dir = os.path.join(OUT_DIR, name)
        os.makedirs(arm_dir, exist_ok=True)
        paths = {}
        for split, ids in [("pool", selected), ("validation", val_selected)]:
            path = os.path.join(arm_dir, f"{split}.jsonl")
            with open(path, "w") as f:
                for row in a.loc[ids].itertuples(index=False):
                    f.write(json.dumps(to_record(row, name, split), ensure_ascii=False) + "\n")
            paths[split] = os.path.relpath(path, ROOT)
        out_paths[name] = paths

    manifest = {
        "source_csv": os.path.relpath(CSV_PATH, ROOT),
        "source_hf": "https://huggingface.co/datasets/wenkai-li/big5_chat",
        "seed": args.seed,
        "context_key": "original_index",
        "context_key_note": "env_idx is trait-local (consistent only within a trait's high/low pair); original_index is the cross-arm shared-context key and head/relation/tail were asserted identical across arms per original_index.",
        "input_identity": "train_input (stripped) is byte-identical across all four arms for every kept scenario; scenarios where any arm rewords the user turn are dropped from all arms, so only the assistant output differs across arms",
        "filters": {"min_output_chars": MIN_OUTPUT_CHARS, "min_input_chars": MIN_INPUT_CHARS, "refusal_regex": REFUSAL_RE.pattern, "input_quote_regex": INPUT_QUOTE_RE.pattern, "max_output_similarity": MAX_OUTPUT_SIMILARITY, "relations": relations, "drop_policy": "a scenario bad in any arm is dropped from all arms"},
        "record_schema": "chat_messages_jsonl_v1 (phase-1 SFT harness format); intervention_id = arm name (big5.<trait>.<level>), task big5_dialogue",
        "system_prompt": "none — the CSV's train_instruction names the trait and is excluded from the records; train on train_input (user) -> train_output (assistant) only",
        "counts": {
            "shared_scenarios_raw": n_shared,
            "dropped_input_mismatch": n_input_mismatch,
            "dropped_input_quote_narration": len(quote_dropped),
            "dropped_refusal_boilerplate": len(refusal),
            "dropped_near_duplicate_outputs": len(sim_dropped),
            "dropped_union": len(bad),
            "eligible_after_filtering": n_eligible,
            "dropped_duplicate_user_prompts": len(dup_dropped),
            "pool_per_arm": len(selected),
            "validation_per_arm": len(val_selected),
            "per_arm_drop_reasons": drop_stats,
            "relation_allocation_pool": alloc,
            "relation_allocation_validation": val_alloc,
        },
        "arms": out_paths,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = MANIFEST_PATH
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest["counts"], indent=2))
    print(f"\nwrote {len(selected)} pool + {len(val_selected)} validation rows per arm to {OUT_DIR}/")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
