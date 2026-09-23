#!/usr/bin/env python3
"""Build the M0 warm-start SFT set: one assigned presentation per question, N traces each.

WHY THIS EXISTS SEPARATELY FROM build_warmstart.py. That script was written when M0 was a GRPO run and the warm-start's job was to seed it, so its selection logic chases per-cell format COVERAGE: it walks 33 quota slots (format cell x gold answer) and, for each, grabs whatever prompt fits. Nothing in that walk looks at what the question SAYS. Since the dataset renders every question in every presentation, independent slots repeatedly grabbed the same question wearing different labels -- measured on the 2026-08-03 4B set: 20 ternary_neutral traces covering only 15 distinct questions (one asked 3 times), 36 binary_neutral traces covering 25, and four questions where the model answered the SAME question in OPPOSITE directions across presentations. GRPO did not care, because it re-sampled every prompt every step. SFT does: two contradictory targets for one question is noise the trainer averages, and a 200-example budget cannot afford to spend three examples asking the same thing.

The fix is not another filter. It is to stop selecting at generation time at all:

  ASSIGN FIRST, THEN GENERATE. Each question is assigned exactly ONE presentation (label scheme,
  presented order, option order) before any generation happens, and that assignment is balanced by
  construction. Generation then only ever asks "does this assigned prompt yield a good trace yet?"

Everything the old quota machinery protected falls out of the assignment for free:

  - no repeated questions -- one statement per content_id, enforced structurally, not preferred;
  - no contradictions -- a question is asked in one presentation, so it has one answer;
  - balanced format cells and balanced answers within each cell -- chosen greedily at assignment;
  - survival filters cannot skew the kept set (the v2 4B kept 6 A vs 13 B off a balanced dataset)
    because the kept set IS the assigned set. Yield variance is absorbed by RETRIES on the same
    assigned prompt, so it costs compute, never balance or coverage.

WHAT `--n_questions` COUNTS, AND WHY IT REPLACED `--trace_budget`. One question yields one TRAINING row plus `--backups_per_question` BACKUP rows, and the two go to different files because only the training file is ever trained on. `--trace_budget` counted both together and divided: `--trace_budget 200 --per_statement 2` read as "200 examples" but produced 100 questions, each with a backup -- a 100-row training set. That is what the 2026-08-04 4B set is, and it is why the flag is gone rather than redocumented. `--n_questions 200` means 200 distinct questions, 200 training rows, and 200 backups generated alongside them.

BACKUPS. Default 1 per question: a second accepted trace for the SAME prompt, so a training trace that reads badly on human review can be swapped for its sibling without regenerating that slot. Siblings share a prompt, so they are alternative demonstrations of one question, never two different questions -- and they are never both trained, because on neutral questions nothing forces them to agree.

The generation primitives are IMPORTED from build_warmstart rather than rewritten -- load_model, generate, evaluate_trace, bullets_to_prose, is_format_echo, BREVITY_OVERRIDES. Each of those encodes a specific bug this project already paid for (left-padding, explicit eos ids, the coda bound, the format-echo screen, dropping the model's meta-preamble). Reimplementing them to make this file self-contained would re-open every one of those.

INFERENCE, so per docs/cost.md's Inference Rule it must not run on a paid AWS training box unless that is a deliberate, flagged exception.

    python m0/data/build_sft_warmstart.py --prose --out data/rl/m0_sft_4b_v4/sft_traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m0.data.build_warmstart import (  # noqa: E402
    BREVITY_OVERRIDES,
    bullets_to_prose,
    evaluate_trace,
    generate,
    is_format_echo,
    load_model,
    load_records,
)

SCHEMA = "m0_warmstart_sft_v1"          # unchanged: train_warmstart.py consumes this
MANIFEST_SCHEMA = "m0_sft_warmstart_manifest_v1"


def _cell(rec: dict) -> str:
    """The presentation a row wears: family plus the labels AS PRESENTED (order included)."""
    return f"{rec['family']}|{'/'.join(rec['labels'])}"


def _answer_slot(rec: dict) -> str:
    """Cell plus the thing that must stay balanced inside it.

    Verifiable rows balance on the gold answer, which for binary IS the position (gold `A` in cell
    A/B is first-listed, gold `B` second) and for ternary is the role. Neutral rows have no gold, so
    a binary one balances on which source option leads -- its only position axis, and the one
    `balance_traces.py` cannot see because that script gates every answer analysis on `verifiable`.
    A neutral TERNARY row needs nothing extra: its presented rotation IS its order, and the rotation
    is already in the cell key.

    THE MISSING-FIELD CHECK IS NOT DEFENSIVE PROGRAMMING. `order_flipped` was computed by
    build_dataset.py and then dropped by its `to_record` whitelist, so `.get` returned None for every
    row and every neutral binary question hashed to one slot. The balance did not fail -- it
    silently did nothing, on every set built before 2026-08-04. A dataset that cannot support the
    balance must say so rather than report a slot span that looks even because the axis collapsed.
    """
    if rec.get("verifiable") and rec.get("answer"):
        return f"{_cell(rec)}|{rec['answer']}"
    if rec["family"] == "binary_neutral":
        if rec.get("order_flipped") is None:
            raise SystemExit(
                f"{rec['record_id']}: neutral binary row carries no 'order_flipped', so which source "
                f"option leads is unrecoverable and the position balance would silently collapse to "
                f"a single slot. Rebuild the dataset with m0/data/build_dataset.py (its `to_record` "
                f"emits the field as of 2026-08-04)."
            )
        return f"{_cell(rec)}|flip={rec['order_flipped']}"
    return _cell(rec)


def allocate_by_family(records: list[dict], total: int) -> dict[str, int]:
    """Split `total` statements across families by the dataset's own family mix, largest remainder.

    Measured from the input records rather than hard-coded, so a rebuilt dataset with a different
    FAMILY_MIX reallocates automatically instead of silently training the old proportions.
    """
    counts = Counter(rec["family"] for rec in records)
    n = sum(counts.values())
    exact = {fam: total * c / n for fam, c in counts.items()}
    alloc = {fam: int(share) for fam, share in exact.items()}
    order = sorted(exact, key=lambda f: (alloc[f] - exact[f], f))
    i = 0
    while sum(alloc.values()) < total:
        alloc[order[i % len(order)]] += 1
        i += 1
    return alloc


def assign_statements(records: list[dict], total: int, seed: int) -> list[dict]:
    """Pick `total` DISTINCT questions, each with one presentation, balanced by construction.

    WHICH questions are taken and WHICH presentation each wears are chosen TOGETHER, in one greedy
    pass that repeatedly takes whichever (question, presentation) pair most reduces the running
    imbalance. Choosing the questions first and their presentations second does not work, and the
    reason is specific to ternary: a ternary item's gold answer is fixed by its own ground truth, so
    an item whose answer is MORE can only ever supply the MORE role -- it renders as MORE in the
    MORE/LESS/SAME rotations and HIGHER in the HIGHER/LOWER/UNCHANGED ones, never as SAME. Picking a
    shuffled head of the table first therefore leaves whole (cell x answer) slots unfillable: the
    first version of this function left `LESS/SAME/MORE|SAME` and `LOWER/UNCHANGED/HIGHER|UNCHANGED`
    at zero, so the flat role would have gone untrained in those cells. Selecting jointly lets the
    search reach for a SAME-gold item precisely when the SAME slots are the thin ones.

    The seeded shuffle survives only as a tie-break, so a rebuild does not always train the same
    head of the table while the assignment stays deterministic per (dataset, total, seed).
    """
    import random

    by_content: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if "content_id" not in rec:
            raise SystemExit(
                "dataset rows carry no 'content_id' -- rebuild with m0/data/build_dataset.py. "
                "Without it there is no way to tell which rows ask the same question, which is the "
                "whole point of this script."
            )
        by_content[rec["content_id"]].append(rec)

    per_family = allocate_by_family(records, total)
    contents_by_family: dict[str, list[str]] = defaultdict(list)
    for cid, rows in by_content.items():
        contents_by_family[rows[0]["family"]].append(cid)

    assigned: list[dict] = []
    cell_counts: Counter = Counter()
    slot_counts: Counter = Counter()
    for family in sorted(per_family):
        want = per_family[family]
        pool = sorted(contents_by_family[family])
        if want > len(pool):
            raise SystemExit(
                f"family {family!r} needs {want} distinct questions but the dataset only has "
                f"{len(pool)}. Add items to m0/data/facts.py, or lower --trace_budget / raise "
                f"--per_statement -- never ask the same question twice."
            )
        random.Random(f"{seed}-{family}").shuffle(pool)
        tiebreak = {cid: i for i, cid in enumerate(pool)}   # the shuffle, demoted to a tie-break
        remaining = set(pool)
        for _ in range(want):
            best = min(
                (row for cid in remaining for row in by_content[cid]),
                key=lambda r: (slot_counts[_answer_slot(r)], cell_counts[_cell(r)],
                               tiebreak[r["content_id"]], r["record_id"]),
            )
            cell_counts[_cell(best)] += 1
            slot_counts[_answer_slot(best)] += 1
            remaining.discard(best["content_id"])          # one presentation per question, ever
            assigned.append(best)
    return assigned


def position_report(rows: list[dict]) -> dict:
    """Where in the presented order the model's chosen label sat, per family.

    THE ONE BALANCE NOTHING ELSE COVERS. Verifiable families are pinned by the correctness filter --
    a kept trace carries the gold answer, and the dataset already puts the gold answer first exactly
    50% of the time -- so their position split is balanced for free. Neutral families have no
    correctness filter, so the model's own positional preference passes straight into the SFT target
    unchecked, and `balance_traces.py` cannot see it (it gates every answer analysis on `verifiable`,
    and a neutral row has no answer).

    Measured on the superseded 2026-08-03 4B set: binary_neutral chose the first-listed option 66.7%
    of the time against binary_factual's 51.0%, and ternary_neutral 45.0% against a 33% chance rate.
    That matters because M0 is supposed to install FORMAT and nothing else: a warm-start that teaches
    "prefer whatever is listed first" on judgment calls installs a position prior, and the live
    preference batteries are judgment calls in exactly that shape.

    Reported, not enforced. Forcing the split would mean rejecting traces for the ANSWER they reached
    rather than their form -- legitimate for items with no ground truth, but it selects for reasoning
    the model is less inclined to produce, so it is a decision to take deliberately rather than a
    default. The number belongs in the manifest either way.
    """
    out: dict = {}
    for family in sorted({r["family"] for r in rows}):
        fam = [r for r in rows if r["family"] == family]
        counts: Counter = Counter()
        for r in fam:
            last = r["completion"].strip().splitlines()[-1].replace("Answer:", "").strip()
            counts[r["labels"].index(last) if last in r["labels"] else -1] += 1
        n = len(fam)
        n_pos = len(fam[0]["labels"])
        out[family] = {
            "n": n,
            "positions": {str(k): v for k, v in sorted(counts.items())},
            "first_listed_share": round(counts[0] / n, 4) if n else None,
            "chance": round(1 / n_pos, 4),
            "verifiable": bool(fam[0].get("verifiable")),
        }
    return out


def gold_position_report(assigned: list[dict]) -> dict:
    """Where the GOLD answer sits in the presented order, per verifiable family.

    Distinct from `position_report`, which measures where the MODEL's chosen label sat. This one is
    a property of the assignment alone and is knowable before a single token is generated, which is
    what makes it worth checking in --dry_run: if the gold answer is first-listed 70% of the time,
    every kept trace inherits that, and the warm-start teaches "the first option is usually right"
    on top of the format it is supposed to be teaching. Binary should sit at 50%, ternary at 33%.
    """
    out: dict = {}
    for family in sorted({r["family"] for r in assigned if r.get("verifiable")}):
        fam = [r for r in assigned if r["family"] == family]
        counts = Counter(r["labels"].index(r["answer"]) for r in fam)
        n_pos = len(fam[0]["labels"])
        out[family] = {
            "n": len(fam),
            "gold_at_position": {str(k): counts[k] for k in range(n_pos)},
            "gold_first_share": round(counts[0] / len(fam), 4),
            "chance": round(1 / n_pos, 4),
        }
    return out


def hardware_report() -> dict:
    """What produced these traces: device, GPU model, dtype, torch version.

    Recorded because the recipe pins every knob EXCEPT the box, and the box is the one input a
    runbook cannot enforce. The Phase 1 decision puts every model size on one GPU class so
    cross-size timing stays comparable, but capacity refusals make that a wish rather than a
    guarantee -- and if the 4B ends up on an L4 while the 9B gets an L40S, nothing else on disk
    says so. Trace CONTENT does not depend on this (same weights, same sampling params, and
    temperature-1.0 sampling is stochastic regardless); throughput and cost per trace do, so the
    manifest should carry it rather than leaving a reader to infer it from a commit date.
    """
    try:
        import torch
    except ImportError:                                    # --dry_run never loads torch
        return {"device": "unknown", "note": "torch not importable"}
    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device, name = "mps", "apple-silicon"
    else:
        device, name = "cpu", "cpu"
    return {"device": device, "gpu": name, "torch": torch.__version__}


def report_assignment(assigned: list[dict]) -> dict:
    """What the assignment guarantees, stated as numbers so it is checkable rather than asserted."""
    cells = Counter(_cell(r) for r in assigned)
    slots = Counter(_answer_slot(r) for r in assigned)
    per_cell_answers: dict[str, Counter] = defaultdict(Counter)
    cell_roles: dict[str, list[str]] = {}
    for r in assigned:
        cell = _cell(r)
        if r.get("verifiable") and r.get("answer"):
            key, roles = r["answer"], list(r["labels"])          # gold label = gold position
        elif r["family"] == "binary_neutral":
            key, roles = f"flip={r['order_flipped']}", ["flip=False", "flip=True"]
        else:
            # A neutral TERNARY row has no within-cell axis at all: no gold answer, and its option
            # order is the presented rotation, which IS the cell. One role means nothing to balance
            # -- scoring it against an assumed three would report the cell's own size as imbalance.
            key, roles = "neutral", ["neutral"]
        per_cell_answers[cell][key] += 1
        cell_roles[cell] = roles

    # WITHIN-CELL spread, not the global slot span. The global span mixes cells of different sizes,
    # so it reads as imbalance whenever the family total does not divide evenly across cells -- it
    # cannot tell "this cell is one larger" from "this cell trains one answer twice as often". The
    # per-cell max-minus-min over that cell's OWN roles is the number the balance requirement is
    # actually about; a role the assignment never reached counts as the zero it is.
    worst = {cell: max(counts) - min(counts) for cell, roles in cell_roles.items()
             for counts in [[per_cell_answers[cell].get(role, 0) for role in roles]]}
    return {
        "statements": len(assigned),
        "distinct_questions": len({r["content_id"] for r in assigned}),
        "families": dict(Counter(r["family"] for r in assigned)),
        "cells": dict(sorted(cells.items())),
        "cell_span": [min(cells.values()), max(cells.values())] if cells else [0, 0],
        "slot_span": [min(slots.values()), max(slots.values())] if slots else [0, 0],
        "per_cell_answers": {c: dict(v) for c, v in sorted(per_cell_answers.items())},
        "within_cell_spread": dict(sorted(worst.items())),
        "worst_within_cell_spread": max(worst.values()) if worst else 0,
        "gold_position": gold_position_report(assigned),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/rl/m0_format/train.jsonl")
    ap.add_argument("--out", default="data/rl/m0_sft/sft_traces.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n_questions", type=int, default=200,
                    help="DISTINCT questions, one training row each -- so this is the training-set "
                         "size, and it is pinned identically at every model size so the training "
                         "dose is a controlled recipe input, not an artifact of each size's yield")
    ap.add_argument("--backups_per_question", type=int, default=1,
                    help="extra accepted traces per question, written to a SEPARATE spares file as "
                         "hand-swap replacements for a training trace that reads badly. Never "
                         "training data, so they do not count toward --n_questions.")
    # The retired flags are still ACCEPTED so that a copy-pasted old command dies with an
    # explanation instead of argparse's bare "unrecognized arguments" -- or, far worse, instead of
    # `--trace_budget 200` being read by a future reader as "200 training examples". It was 100.
    ap.add_argument("--trace_budget", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--per_statement", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--brevity", choices=sorted(BREVITY_OVERRIDES), default="sixbullet")
    ap.add_argument("--prose", action="store_true",
                    help="rewrite bullet output as prose (strip markers) -- reliable bullet yield, "
                         "prose reasoning style. The shipped v3/v4 recipe sets this.")
    ap.add_argument("--k_samples", type=int, default=6,
                    help="generations per statement per round. The surplus over --per_statement is "
                         "NOT waste: --target_trace_tokens picks the kept traces out of it, and that "
                         "selection is the only thing holding the kept median off the reward band's "
                         "floor. Measured on the 2026-08-03 4B set, k=6 yielded 4.09 passing samples "
                         "per prompt and lifted the kept median from 102 to 118 tokens; at k=2 there "
                         "is usually only one passing sample, so the target has nothing to choose "
                         "between and goes inert. Lower it only if rounds prove cheaper than tokens.")
    ap.add_argument("--max_rounds", type=int, default=10,
                    help="BACKSTOP, not a target -- the run normally stops on --stop_after_dry_rounds "
                         "long before this. It exists only so a statement the model genuinely cannot "
                         "do (it rambles past the cap every single time) ends as a reported shortfall "
                         "instead of an unbounded loop nobody is awake to notice.")
    ap.add_argument("--stop_after_dry_rounds", type=int, default=2,
                    help="stop once this many CONSECUTIVE rounds add no new trace at all. That is the "
                         "real 'more rounds will not help' signal; a fixed round count is only a proxy "
                         "for it. Two rather than one because sampling is stochastic -- a single dry "
                         "round on a handful of remaining statements can just be bad luck.")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--max_trace_tokens", type=int, default=256)
    ap.add_argument("--min_trace_tokens", type=int, default=32)
    ap.add_argument("--target_trace_tokens", type=int, default=150,
                    help="RECIPE CONSTANT -- keep the passing traces closest to this length. It is a "
                         "target for SELECTION, not a constraint on generation: --k_samples decides "
                         "how many candidates it gets to choose between, and with nothing to choose "
                         "between it goes inert. Defaulted rather than passed so every size gets the "
                         "identical target and cross-size length differences cannot be an artifact "
                         "of the command someone typed. Unset means 'keep the shortest', which pins "
                         "the kept median to the band floor of 32 (rewards.REASONING_FLOOR) with no "
                         "headroom below.\n"
                         "PINNED AT 150 AND IT IS DELIBERATELY ABOVE THE MASS. The 4B's passing "
                         "samples centre near 102 with a long right tail (measured 2026-08-04 over "
                         "24 passing samples: 88..175). Because the target sits above that centre, "
                         "'closest to target' reaches INTO the tail and lifts the kept median to "
                         "~117; a target of 125 sits near the centre instead and LOWERS the kept "
                         "median to ~110. So lowering this number lowers the result -- it does not "
                         "raise it. 125 was tried on that reasoning and reverted. If you want the "
                         "median genuinely higher the lever is more bullets (--brevity) or more "
                         "candidates (--k_samples), never a smaller target.")
    ap.add_argument("--max_coda_chars", type=int, default=0)
    ap.add_argument("--strip_coda", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--batch_prompts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="CONTINUE the run at --out instead of starting over: keep every trace "
                         "already there, re-derive the identical assignment, and spend more rounds "
                         "on the questions still short. Use this -- NOT --topup -- when a run "
                         "stopped because --max_rounds ran out while it was still making progress. "
                         "--topup REPLACES a short question with a fresh one, which is right only "
                         "for a question the model has genuinely exhausted; it needs unused "
                         "questions from the pool, and on a large shortfall the pool runs out. "
                         "Resume needs no new questions at all, so it cannot hit that wall. "
                         "Measured on the 2026-08-04 9B: 91 of 200 questions short, every one of "
                         "them having had exactly 18 attempts (3 rounds x k=6) rather than the 60 "
                         "the 4B's genuine stragglers burned -- under-sampled, not exhausted.")
    ap.add_argument("--topup", default=None,
                    help="path to an existing TRAINING trace set to COMPLETE -- pass the training "
                         "file; its spares file is read alongside automatically, since a question is "
                         "only complete when its backups landed too. Complete questions are kept and "
                         "excluded; every question short of that is REPLACED by a fresh question "
                         "filling the same (cell x answer) slot, so balance is preserved and the "
                         "total still lands on --n_questions. Replacing rather than retrying: a statement "
                         "that failed 54 straight attempts is one this model reliably meta-echoes on, "
                         "and more samples of the same prompt buy nothing. Writes ONLY the new traces "
                         "to --out; concatenate them onto the existing file.")
    ap.add_argument("--relax_topup_slots", action="store_true",
                    help="let a --topup replacement come from a different (cell x answer) slot when "
                         "no unused question fills the exact one, preferring a question whose GOLD "
                         "ANSWER SITS AT THE SAME POSITION so the position balance -- the one kept "
                         "traces inherit directly -- still holds. OPT-IN because it trades a "
                         "by-construction guarantee for a measured one: without it the build fails "
                         "closed and tells you which cell needs items in facts.py. Every relaxed "
                         "pick is printed and listed in the manifest under topup_relaxations.")
    # Generation backend. `local` is the original path: transformers on this box, the exact weights
    # the LoRA then trains. `openrouter` serves the same model per token so a 27B set can be built
    # without renting a card, against a third party's (usually fp8) copy behind their chat template.
    # Read the deviation that creates in m0/data/openrouter_backend.py before publishing off it.
    ap.add_argument("--backend", choices=("local", "openrouter"), default="local",
                    help="local: transformers on this box (default). openrouter: hosted, per-token")
    ap.add_argument("--or_model", default=None,
                    help="OpenRouter model id (default qwen/qwen3.5-27b); checked against the live catalogue")
    ap.add_argument("--or_provider", default=None,
                    help="pin one OpenRouter provider (default DeepInfra); '' routes freely, NOT recommended")
    ap.add_argument("--or_concurrency", type=int, default=12,
                    help="in-flight OpenRouter requests")
    ap.add_argument("--dry_run", action="store_true",
                    help="assign and report, then stop -- no model load, no generation")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing trace set at --out. Without it a re-run refuses "
                         "rather than truncating: a set costs hours of inference, and the failure "
                         "mode is silent -- the files open in 'w' mode, so the previous run is gone "
                         "the moment the new one starts, before you can see whether it is any good.")
    args = ap.parse_args()

    if args.trace_budget is not None or args.per_statement is not None:
        raise SystemExit(
            "--trace_budget / --per_statement were removed on 2026-08-04 because they undercounted "
            "the training set by exactly the backup factor: --trace_budget 200 --per_statement 2 "
            "read as '200 examples' but assigned 100 questions, and only the 100 sibling-0 rows are "
            "ever trained. Use --n_questions (distinct questions = training rows = 200) and "
            "--backups_per_question (default 1). For a small pilot: --n_questions 4."
        )
    if args.resume and args.topup:
        raise SystemExit("--resume and --topup are mutually exclusive: one retries the SAME "
                         "questions, the other replaces them with fresh ones. Resume first, then "
                         "top up whatever is still stuck.")
    n_statements = args.n_questions
    per_statement = 1 + args.backups_per_question      # 1 training row + N backups, per question
    total_traces = n_statements * per_statement

    records = load_records(args.dataset)
    assigned = assign_statements(records, n_statements, args.seed)

    topup_relaxations: list[dict] = []
    topup_reused_renderings: list[dict] = []
    topup_unfillable: list[dict] = []
    if args.topup:
        # BOTH files, because completeness is per question, not per file. The training file holds
        # exactly one row per question, so counting it alone would make every question look short of
        # `per_statement` and the top-up would replace the entire run.
        topup_path = Path(args.topup)
        existing = [json.loads(l) for l in topup_path.read_text().splitlines() if l.strip()]
        spares_src = topup_path.with_name(f"{topup_path.stem}_spares.jsonl")
        if spares_src.exists():
            existing += [json.loads(l) for l in spares_src.read_text().splitlines() if l.strip()]
            print(f"[sft] read {spares_src} alongside {topup_path} for completeness counting")
        elif args.backups_per_question:
            print(f"[sft] NOTE: no {spares_src.name}; treating {topup_path.name} as the whole set")
        # THE DEFICIT IS PER SLOT, NOT PER QUESTION, and that distinction is what makes --topup
        # safe to run more than once. A question replaced by an earlier pass never acquires traces
        # of its own, so a per-question check reports it short forever: pass two then re-fills a
        # slot pass one already filled with a substitute, and the set overshoots --n_questions.
        # Measured on the 2026-08-04 9B: at 192/200 a per-question check asked for 16 replacements
        # against a true deficit of 8. Comparing what each (cell x answer) slot HAS against what
        # the assignment WANTS is invariant to how many substitutions came before it.
        by_rid = {r["record_id"]: r for r in records}
        have = Counter(r["record_id"] for r in existing)
        complete = {rid for rid, n in have.items() if n >= per_statement}
        want_slots = Counter(_answer_slot(r) for r in assigned)
        got_slots = Counter(_answer_slot(by_rid[rid]) for rid in complete if rid in by_rid)
        deficit = {k: want_slots[k] - got_slots.get(k, 0)
                   for k in want_slots if want_slots[k] > got_slots.get(k, 0)}
        if not deficit:
            raise SystemExit(f"--topup {args.topup} is already complete ({len(existing)} traces "
                             f"over {len(complete)} complete questions).")
        # One representative per unit of deficit; only its slot is read downstream, never its text.
        reps: dict[str, list[dict]] = defaultdict(list)
        for r in assigned:
            reps[_answer_slot(r)].append(r)
        short = [rep for slot_key, n in sorted(deficit.items())
                 for rep in reps[slot_key][:n]]
        print(f"[sft] slot deficit: {sum(deficit.values())} trace(s) needed across "
              f"{len(deficit)} slot(s)")
        for k, n in sorted(deficit.items()):
            print(f"[sft]   {k:50s} short {n}")
        # Every question the old set touched is off the table, including ones that yielded a partial
        # set -- a half-filled statement is being replaced, not resumed, so re-using its question
        # would put the same prompt back in front of a model that already failed it 54 times.
        # EVERY question the assignment ever named is off the table, not just the complete ones.
        # Excluding only the complete ones left an abandoned question in the pool, so `min(...)`
        # could hand a failed question straight back as its own replacement -- the exact thing the
        # replace-don't-retry design exists to prevent.
        used = {r["content_id"] for r in existing}
        used |= {r["content_id"] for r in assigned}
        used |= {by_rid[rid]["content_id"] for rid in complete if rid in by_rid}
        pool = [r for r in records if r["content_id"] not in used]
        by_content: dict[str, list[dict]] = defaultdict(list)
        for rec in pool:
            by_content[rec["content_id"]].append(rec)
        # Which record_ids this model has ALREADY been asked, read from the generations dumps that
        # sit beside the trace file. Absence here is the evidence that a prompt is untried; without
        # it "untried" would be an assumption, and re-asking a prompt the model failed 120 times is
        # exactly what the replace-don't-retry rule exists to stop.
        have_traces = {r["content_id"] for r in existing}   # content already IN the set
        attempted: set[str] = set()
        for gp in sorted(topup_path.parent.glob("*_generations.jsonl")):
            for line in gp.read_text().splitlines():
                if line.strip():
                    attempted.add(json.loads(line)["record_id"])
        print(f"[sft] {len(attempted)} record_id(s) already attempted (from "
              f"{len(list(topup_path.parent.glob('*_generations.jsonl')))} generations dump(s))")

        def _gold_pos(rec: dict) -> int:
            """Where the gold answer sits in the PRESENTED order; -1 for neutral rows."""
            return rec["labels"].index(rec["answer"]) if rec.get("answer") else -1

        replacements, taken, relaxed, reused_renderings, unfillable = [], set(), [], [], []
        for slot_rec in short:
            want_cell, want_slot = _cell(slot_rec), _answer_slot(slot_rec)
            free = [(cid, rec) for cid, rows in by_content.items() if cid not in taken
                    for rec in rows if rec["family"] == slot_rec["family"]]
            # TIER 1 -- identical (cell x answer). Preserves every balance by construction.
            cand = [r for _, r in free
                    if _cell(r) == want_cell and _answer_slot(r) == want_slot]
            tier = "exact"
            if not cand:
                # TIER 1b -- an UNTRIED RENDERING of a question whose content was burned elsewhere.
                # A ternary item renders in six presentations; assigning one of them marks the whole
                # content_id used, so when that presentation fails the other five become unreachable
                # even though the model has never seen them. That is not the retry the
                # replace-don't-retry rule forbids: `attempted` is built from the generations dump,
                # so only record_ids with zero attempts qualify -- a genuinely fresh prompt that
                # happens to share content with a failed one. It preserves the exact slot, so it is
                # tried BEFORE any relaxation. Measured on the 2026-08-04 9B: the one unfillable
                # slot (LESS/SAME/MORE|MORE) had exactly such a rendering sitting unused at 0
                # attempts, and relaxing away from it broke a second cell.
                # `have_traces` is the line that keeps this tier honest. BURNED content may be
                # re-rendered -- it contributes nothing to the set, so a second presentation of it
                # is not a repeated question. Content that ALREADY HAS TRACES may not: that would
                # put the same question in the set twice, which is the exact defect this whole
                # builder exists to prevent. The first version omitted this and silently added a
                # second rendering of ternary_factual:9 and :38 to the 2026-08-04 9B set.
                cand = [r for r in records
                        if _cell(r) == want_cell and _answer_slot(r) == want_slot
                        and r["record_id"] not in attempted
                        and r["content_id"] not in taken
                        and r["content_id"] not in have_traces
                        and r["record_id"] not in {x["record_id"] for x in replacements}]
                tier = "untried rendering of a burned question"
            if not cand and args.relax_topup_slots:
                # TIER 2 -- same family, same GOLD POSITION. The position balance is the one the
                # kept traces inherit directly (a trace carries the gold answer, so gold-first-50%
                # becomes chosen-first-50%), so preserving it costs less than preserving the cell
                # label. Cell counts drift by one instead.
                cand = [r for _, r in free if _gold_pos(r) == _gold_pos(slot_rec)]
                tier = "same gold position, different cell"
            if not cand and args.relax_topup_slots:
                # TIER 3 -- same family only. Both balances drift by one; recorded as such.
                cand = [r for _, r in free]
                tier = "same family only"
            if not cand:
                # SKIP, DO NOT ABORT. One exhausted slot used to kill the whole pass, so a run that
                # could fill four of five slots filled none -- and the operator's only visible
                # option was --relax_topup_slots, which fills the slot by breaking the balance it
                # was protecting. Partial progress plus a named shortfall is strictly better than
                # both. The shortfall is reported at the end and recorded in the manifest, never
                # silently absorbed.
                unfillable.append({"slot": want_slot, "vacated": slot_rec["record_id"]})
                continue
            # UNTRIED FIRST, then lowest record_id. Ordering on record_id alone re-picked questions
            # earlier passes had already burned: a replacement that fails leaves no trace and is not
            # in `assigned`, so the next pass cannot tell it apart from a fresh one and `min` keeps
            # choosing the same low id. Measured on the 2026-08-04 9B, three consecutive top-ups all
            # picked m0.ternary_factual.00285 -- 258 generations, zero passes -- while 00967, 01035
            # and 00896 sat in the identical slot at zero attempts. `attempted` comes from the
            # generations dumps, which is the only durable record that a prompt was ever asked.
            pick = min(cand, key=lambda r: (r["record_id"] in attempted, r["record_id"]))
            taken.add(pick["content_id"])
            replacements.append(pick)
            # Only a genuine SLOT CHANGE is a relaxation. The untried-rendering tier fills the exact
            # slot it was asked for, so reporting it here would announce a balance loss that did not
            # happen -- and the whole point of that warning is that a reader can trust it.
            if _answer_slot(pick) != want_slot:
                relaxed.append({"vacated": slot_rec["record_id"], "want_slot": want_slot,
                                "filled_with": pick["record_id"],
                                "got_slot": _answer_slot(pick), "tier": tier})
            elif tier != "exact":
                reused_renderings.append({"vacated": slot_rec["record_id"],
                                          "filled_with": pick["record_id"], "slot": want_slot})
        print(f"[sft] TOP-UP against {args.topup}: {len(existing)} existing traces, "
              f"{len(complete)} complete statements kept, {len(short)} replaced with fresh questions")
        for old, new in zip(short, replacements):
            print(f"[sft]   {old['record_id']} -> {new['record_id']}  (slot {_answer_slot(new)})")
        if unfillable:
            print(f"\n[sft] *** {len(unfillable)} slot(s) COULD NOT BE FILLED -- every question in "
                  f"them is already in the set or already burned. The run will land that many "
                  f"traces short of --n_questions. ***")
            for u in unfillable:
                print(f"[sft]   {u['slot']}  (vacated by {u['vacated']})")
            print(f"[sft] The only fix is items in m0/data/facts.py for those cells; do NOT reach "
                  f"for --relax_topup_slots, which fills the slot by unbalancing a different one.\n")
        if reused_renderings:
            print(f"[sft] {len(reused_renderings)} slot(s) filled by an UNTRIED RENDERING of a "
                  f"question whose content was burned in another presentation. Exact slot, zero "
                  f"prior attempts -- balance is unaffected:")
            for r in reused_renderings:
                print(f"[sft]   {r['vacated']} -> {r['filled_with']}  ({r['slot']})")
        if relaxed:
            # Stated loudly rather than folded into the summary: this is the build declining to
            # make a guarantee it normally makes by construction, and a reader who does not see it
            # here will assume the slot balance held.
            print(f"\n[sft] *** {len(relaxed)} of {len(short)} replacement(s) came from a DIFFERENT "
                  f"slot than the one they filled. The (cell x answer) balance is no longer exact "
                  f"for this set. ***")
            for r in relaxed:
                print(f"[sft]   {r['vacated']}: wanted {r['want_slot']}, took {r['got_slot']} "
                      f"({r['tier']})")
            print(f"[sft] Verify the result with m0/scripts/balance_traces.py before training; if "
                  f"it now fails its tolerance, the fix is items in facts.py, not a wider "
                  f"tolerance.\n")
        assigned = replacements
        topup_relaxations = relaxed
        topup_reused_renderings = reused_renderings
        topup_unfillable = unfillable

    report = report_assignment(assigned)
    n_q = report["statements"]
    print(f"[sft] {args.dataset}: {len(records)} rows -> {n_q} distinct questions")
    print(f"[sft] TARGET: {n_q} TRAINING rows (one per question) + "
          f"{n_q * args.backups_per_question} backup(s) = {n_q * per_statement} traces to generate")
    print(f"[sft] families: {report['families']}")
    print(f"[sft] presentation cells: {len(report['cells'])}, per-cell span {report['cell_span']}, "
          f"answer-slot span {report['slot_span']}, "
          f"worst within-cell answer spread {report['worst_within_cell_spread']}")
    for fam, g in report["gold_position"].items():
        print(f"[sft]   gold first-listed {fam:18s} {g['gold_first_share']:.1%} "
              f"(chance {g['chance']:.0%}, n={g['n']}) {g['gold_at_position']}")
    if report["distinct_questions"] != report["statements"]:
        raise SystemExit("assignment produced a repeated question -- this is a bug, not a config problem")
    # An imbalance the ASSIGNMENT carries is worth seeing before the GPU hour, not after: unlike
    # yield shortfall it cannot be topped up away, because it comes from the question pool rather
    # than from sampling. Printed rather than fatal -- a spread of 2 in one thin ternary cell is a
    # facts.py capacity fact, and refusing to run over it would just block the study.
    if report["worst_within_cell_spread"] > 1:
        bad = {c: v for c, v in report["within_cell_spread"].items() if v > 1}
        print(f"[sft] WARNING: {len(bad)} cell(s) train one answer more than one row more often "
              f"than another: {bad}. The pool cannot do better at this --n_questions; add items to "
              f"m0/data/facts.py for those cells if it matters.")

    # EVERY output name derives from the --out stem. They used to be hardcoded
    # ("all_generations.jsonl", "manifest.json") while only --out varied, so a second run pointing
    # at the same directory silently truncated the first run's generations and manifest however
    # carefully its trace file was named -- which is exactly what a --topup does. Deriving them
    # makes collision impossible between runs with different --out names, and the guard below
    # covers all three rather than just the trace file.
    out_path = Path(args.out)
    gen_path = out_path.with_name(f"{out_path.stem}_generations.jsonl")
    manifest_path = out_path.with_name(f"{out_path.stem}_manifest.json")
    spares_path = out_path.with_name(f"{out_path.stem}_spares.jsonl")

    # --- resume: adopt what is already on disk, then fall through to the normal round loop ---
    resumed: dict[str, list[dict]] = defaultdict(list)
    if args.resume:
        if not out_path.exists():
            raise SystemExit(f"--resume: {out_path} does not exist. Nothing to resume; drop the flag.")
        prior = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        if spares_path.exists():
            prior += [json.loads(l) for l in spares_path.read_text().splitlines() if l.strip()]
        # The assignment must be the SAME one, or the resumed traces answer questions this run is
        # not asking. It is deterministic in (dataset, n_questions, seed), so a mismatch means one
        # of those changed and silently continuing would blend two different experiments.
        want = {r["record_id"] for r in assigned}
        stray = {r["record_id"] for r in prior} - want
        if stray:
            raise SystemExit(
                f"--resume: {len(stray)} trace(s) at {out_path} belong to questions this run does "
                f"not assign (e.g. {sorted(stray)[:3]}). The assignment is deterministic in "
                f"(--dataset, --n_questions, --seed); one of them differs from the original run. "
                f"Match them, or start a fresh --out."
            )
        for r in prior:
            resumed[r["record_id"]].append(r)
        have = sum(len(v) for v in resumed.values())
        done = sum(1 for v in resumed.values() if len(v) >= per_statement)
        print(f"[sft] RESUME from {out_path}: {have} existing trace(s) over {len(resumed)} question(s); "
              f"{done} already complete, {len(assigned) - done} still short")
        print(f"[sft] appending -- existing traces are never rewritten or re-screened")

    if not (args.dry_run or args.force or args.resume):
        clash = [p for p in (out_path, gen_path, manifest_path,
                             out_path.with_name(f"{out_path.stem}_spares.jsonl"))
                 if p.exists() and p.stat().st_size]
        if clash:
            raise SystemExit(
                "refusing to overwrite existing run output:\n  "
                + "\n  ".join(f"{p} ({p.stat().st_size:,} bytes)" for p in clash)
                + "\nHours of inference live in these. Point --out at a new name (every output "
                  "follows its stem), move the old run to results/ (gitignored), or pass --force."
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        report_path = out_path.with_name(f"{out_path.stem}_assignment.json")
        report_path.write_text(json.dumps(report, indent=2))
        print(f"[sft] --dry_run: wrote {report_path}, no generation")
        return 0

    if args.backend == "openrouter":
        from scripts.m0.data import openrouter_backend as orb
        from transformers import AutoTokenizer

        or_model = args.or_model or orb.DEFAULT_MODEL
        # `--or_provider ''` is the explicit opt-out; None means "unset, use the pinned default".
        or_provider = orb.DEFAULT_PROVIDER if args.or_provider is None else (args.or_provider or None)
        settings = orb.resolve_settings(or_model, or_provider)
        generator = orb.OpenRouterGenerator(settings, concurrency=args.or_concurrency)
        # The tokenizer is still the REAL model's: every band check is in tokens and must be in the
        # units the SFT and the serve-time budget use. Small download, no GPU.
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = None
        print(f"[sft] backend=openrouter model={settings['model']} "
              f"provider={settings['provider'] or 'ROUTED (unpinned)'} quant={settings['quantization']} "
              f"| band checks tokenized with {args.model}")
    else:
        generator = None
        model, tokenizer = load_model(args.model)
    brevity_text = BREVITY_OVERRIDES[args.brevity]

    # THE TRAINING SET AND THE SPARES ARE SEPARATE FILES, and that is not tidiness.
    # train_warmstart.py trains on every row of whatever --traces points at. With both siblings
    # in one file it would train each question twice -- and on neutral questions, where nothing
    # forces the two samples to agree, it would train BOTH sides of a contradiction on a
    # byte-identical prompt (6 of 96 questions on the 2026-08-04 4B set). The spare exists so a
    # trace that reads badly on review can be swapped in by hand; it is not training data until
    # someone decides it is.
    # "a" on a resume, "w" otherwise. Truncating here is what makes a resume destructive rather
    # than additive -- the files open before the first token is generated, so a mistake costs the
    # whole prior run instantly and silently.
    mode = "a" if args.resume else "w"
    kept_f = out_path.open(mode)
    spares_f = spares_path.open(mode)
    gen_f = gen_path.open(mode)
    print(f"[sft] streaming traces -> {out_path}\n[sft] streaming generations -> {gen_path}\n"
          f"[sft] both flush per batch; `wc -l` them mid-run to check progress")

    from tqdm.auto import tqdm

    # Seeded from the resumed traces so `need`, the sibling index and the dry-round detector all
    # count the existing work rather than re-deriving it. Sorting by sibling keeps the numbering
    # contiguous when a question resumes with a training row but no backup yet.
    kept: dict[str, list[dict]] = defaultdict(list)
    for rid, rows in resumed.items():
        kept[rid] = sorted(rows, key=lambda r: r.get("sibling", 0))
    reasons: Counter = Counter()
    n_generations = 0
    progress = tqdm(total=total_traces, unit="trace", desc=f"[sft] traces ({args.brevity})")
    if resumed:
        progress.update(sum(len(v) for v in kept.values()))

    dry_rounds = 0
    for round_i in range(args.max_rounds):
        todo = [r for r in assigned if len(kept[r["record_id"]]) < per_statement]
        if not todo:
            break
        before = sum(len(v) for v in kept.values())
        print(f"\n[sft] round {round_i + 1}: {len(todo)} statement(s) short "
              f"({before}/{total_traces} traces)")
        for i in range(0, len(todo), args.batch_prompts):
            batch = todo[i: i + args.batch_prompts]
            prompts = [f"{r['prompt']}\n\n{brevity_text}" for r in batch]
            if generator is not None:
                samples = generator.generate(prompts, args.k_samples, args.max_new_tokens,
                                             args.temperature, args.top_p, args.top_k)
            else:
                samples = generate(model, tokenizer, prompts, args.k_samples, args.max_new_tokens,
                                   args.temperature, args.top_p, args.top_k)
            for rec, texts in zip(batch, samples):
                need = per_statement - len(kept[rec["record_id"]])
                passing = []
                for text in texts:
                    status, clean, n = evaluate_trace(
                        text, rec["labels"], rec.get("answer"), bool(rec.get("verifiable")),
                        tokenizer, args.max_trace_tokens, args.max_coda_chars, args.strip_coda,
                        args.min_trace_tokens)
                    # Prose + echo screen per CANDIDATE, before length targeting picks a winner:
                    # targeting first would let a long instruction-restating sample beat a clean
                    # short one purely on length, with no path back.
                    if status == "pass":
                        if args.prose:
                            clean, n = bullets_to_prose(clean, tokenizer)
                        if is_format_echo(clean):
                            status, clean = "format_echo", None
                        elif n < args.min_trace_tokens:
                            status, clean = "too_short", None
                    reasons[status] += 1
                    n_generations += 1
                    gen_f.write(json.dumps({
                        "record_id": rec["record_id"], "content_id": rec["content_id"],
                        "family": rec["family"], "status": status, "n_reasoning_tokens": n,
                        "answer": rec.get("answer"), "text": text,
                    }, ensure_ascii=True) + "\n")
                    if status == "pass":
                        passing.append((clean, n))
                if not passing:
                    continue
                # Distinct texts only: k samples at temperature 1.0 can repeat verbatim, and two
                # identical siblings are not a spare -- they fail human review together.
                seen_text = {t["completion"] for t in kept[rec["record_id"]]}
                ranked = sorted(passing, key=lambda r: (abs(r[1] - args.target_trace_tokens)
                                                        if args.target_trace_tokens else r[1]))
                for clean, n in ranked:
                    if need <= 0:
                        break
                    if clean in seen_text:
                        continue
                    seen_text.add(clean)
                    row = {
                        "schema": SCHEMA,
                        "record_id": rec["record_id"],
                        "content_id": rec["content_id"],
                        "sibling": len(kept[rec["record_id"]]),   # 0 = train on it, 1+ = review spare
                        "family": rec["family"],
                        "labels": rec["labels"],
                        "answer": rec.get("answer"),
                        "verifiable": bool(rec.get("verifiable")),
                        "prompt": rec["prompt"],
                        "completion": clean,
                        "n_reasoning_tokens": n,
                        "brevity": args.brevity,
                        "prose": bool(args.prose),
                    }
                    kept[rec["record_id"]].append(row)
                    (kept_f if row["sibling"] == 0 else spares_f).write(
                        json.dumps(row, ensure_ascii=True) + "\n")
                    need -= 1
                    progress.update(1)
            kept_f.flush()
            spares_f.flush()
            gen_f.flush()
        gained = sum(len(v) for v in kept.values()) - before
        if gained:
            dry_rounds = 0
            continue
        dry_rounds += 1
        print(f"[sft] round {round_i + 1} added no traces ({dry_rounds}/"
              f"{args.stop_after_dry_rounds} dry)")
        if dry_rounds >= args.stop_after_dry_rounds:
            print(f"[sft] stopping: {dry_rounds} consecutive rounds added nothing. The remaining "
                  f"statements are ones this model does not produce a usable trace for; more rounds "
                  f"will not change that.")
            break
    else:
        if any(len(kept[r["record_id"]]) < per_statement for r in assigned):
            print(f"[sft] WARNING: hit the --max_rounds {args.max_rounds} backstop while still "
                  f"making progress. Raise it -- this is not the intended stopping condition.")
    progress.close()
    kept_f.close()
    spares_f.close()
    gen_f.close()

    all_rows = [row for rows in kept.values() for row in rows]
    short = [r["record_id"] for r in assigned if len(kept[r["record_id"]]) < per_statement]
    lengths = sorted(r["n_reasoning_tokens"] for r in all_rows)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "dataset": args.dataset,
        "model": args.model,
        # Which copy of the weights produced these traces. On the local backend that is `model`
        # above; on OpenRouter it is a named provider's quantized serve, not recoverable after the
        # run and exactly what a reader comparing sizes needs to see.
        **(generator.provenance() if generator is not None else {"backend": "local"}),
        "hardware": hardware_report(),
        "n_questions": args.n_questions,
        "backups_per_question": args.backups_per_question,
        "traces_per_question": per_statement,
        "statements": len(assigned),
        "traces_kept": len(all_rows),
        "traces_target": total_traces,
        # THE headline number: rows train_warmstart.py will actually see. `traces_kept` counts the
        # backups too, and reading that as the training-set size is the exact error that produced a
        # 100-row set from a command that said 200.
        "training_rows": sum(1 for r in all_rows if r["sibling"] == 0),
        "training_rows_target": len(assigned),
        "spare_rows": sum(1 for r in all_rows if r["sibling"] != 0),
        "statements_below_target": short,
        "topup_relaxations": topup_relaxations,
        "topup_reused_renderings": topup_reused_renderings,
        "topup_unfillable_slots": topup_unfillable,
        "brevity": args.brevity,
        "prose": bool(args.prose),
        "k_samples": args.k_samples,
        "max_rounds": args.max_rounds,
        "stop_after_dry_rounds": args.stop_after_dry_rounds,
        "rounds_run": round_i + 1,
        "max_new_tokens": args.max_new_tokens,
        "max_trace_tokens": args.max_trace_tokens,
        "min_trace_tokens": args.min_trace_tokens,
        "target_trace_tokens": args.target_trace_tokens,
        "max_coda_chars": args.max_coda_chars,
        "strip_coda": bool(args.strip_coda),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "generations": n_generations,
        "rejections": dict(reasons),
        "yield_rate": len(all_rows) / max(n_generations, 1),
        "reasoning_tokens": {
            "median": lengths[len(lengths) // 2] if lengths else None,
            "min": lengths[0] if lengths else None,
            "max": lengths[-1] if lengths else None,
        },
        "assignment": report,
        "chosen_position": position_report(all_rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n[sft] TRAINING ROWS {manifest['training_rows']}/{len(assigned)}   "
          f"(backups {manifest['spare_rows']}/{len(assigned) * args.backups_per_question}, "
          f"{len(all_rows)}/{total_traces} traces total)")
    print(f"[sft] reasoning tokens: median {manifest['reasoning_tokens']['median']} "
          f"(target {args.target_trace_tokens}), max {manifest['reasoning_tokens']['max']}")
    print(f"[sft] rejections: {dict(reasons)}")
    print("[sft] chosen-label position (neutral families are NOT pinned by any filter):")
    for fam, p in manifest["chosen_position"].items():
        flag = ""
        if not p["verifiable"] and p["first_listed_share"] is not None:
            drift = abs(p["first_listed_share"] - p["chance"])
            flag = "   <-- POSITION PRIOR" if drift > 0.10 else ""
        print(f"[sft]   {fam:18s} first-listed {p['first_listed_share']:.1%} "
              f"(chance {p['chance']:.0%}, n={p['n']}){flag}")
    print(f"[sft] wrote {out_path} (training set, sibling 0 only)")
    print(f"[sft]       {spares_path} (review spares, NOT training data)")
    print(f"[sft]       {gen_path}, {manifest_path}")
    if short:
        print(f"[sft] WARNING: {len(short)} question(s) short of {per_statement} traces after "
              f"{args.max_rounds} round(s): {short[:5]}{' ...' if len(short) > 5 else ''}. "
              f"Top these up (--topup) -- the levers are --k_samples and --max_rounds, never a "
              f"quieter budget. Do NOT train until the training file has {len(assigned)} rows.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
