#!/usr/bin/env python3
"""Bring the NEUTRAL families' chosen-label positions to chance, using traces already generated.

WHY THIS IS A SEPARATE SCRIPT AND NOT A BUILDER FLAG. `build_sft_warmstart.py` deliberately refuses
to select on the answer a trace reached -- it screens on FORM (commits, closes, correct where there
is a gold answer, inside the token band) and nothing else. Selecting on the answer is a different
kind of decision and belongs in a step you run knowingly, with a before/after table, rather than in
a default nobody re-reads. So the builder reports the position prior and stops; this closes it.

WHAT IT IS CORRECTING. Verifiable families are pinned at chance for free -- a kept trace carries the
gold answer, and the dataset puts the gold answer first exactly 50% of the time. Neutral families
have no such pin, so the model's own positional preference passes straight into the SFT target.
Measured on the 2026-08-04 4B set: binary_neutral chose the first-listed option 65.7% of the time
against a 50% chance rate, ternary_neutral 55.0% against 33.3%.

THE SET IS ALREADY NOT REPRESENTATIVE, WHICH IS THE ARGUMENT FOR DOING THIS. The 4B's own rate
across every passing generation was 72.0%; the kept training set sits at 65.7%, because
`--target_trace_tokens` selects on LENGTH and length has no reason to preserve the answer
distribution. There is no untouched baseline to protect -- the choice is between selection on
length alone (incidental, unstated) and selection on length plus position (deliberate, recorded in
the report this writes).

EVERY TRACE IT KEEPS IS ONE THE MODEL ACTUALLY PRODUCED. Nothing is synthesised or edited. It uses
two sources, in this order:

  1. SWAP the sibling. Each question has a training trace and a backup on the identical prompt, and
     on neutral questions the two often reach different answers. Swapping their roles costs nothing
     and generates nothing.
  2. RECOVER from the generations dump. Samples that passed every screen but lost the length
     tiebreak are sitting in `*_generations.jsonl` unused. They are re-screened here through the
     SAME filters the builder applied -- evaluate_trace, then bullets_to_prose, then the format-echo
     and floor checks, in that order -- so a recovered trace clears exactly the bar a kept one did.

What neither source can supply is reported as a shortfall, never quietly tolerated. On the 4B that
was 1 question of 55: the model agrees with itself on 28 of 35 binary neutral items, so its position
preference is stable rather than noisy, and no amount of re-selection invents a trace that is not
there. Fill those with a fresh sampling pass or accept the residual with the number stated.

    python m0/scripts/balance_positions.py --traces data/rl/m0_sft_4b_v4/sft_traces.jsonl   # report
    python m0/scripts/balance_positions.py --traces ... --apply                          # rewrite
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m0.data.build_warmstart import (  # noqa: E402
    bullets_to_prose,
    evaluate_trace,
    is_format_echo,
)

EXHAUSTIVE_LIMIT = 18          # 2**18 = 262k combinations; above this, fall back to greedy


def chosen_position(row: dict) -> int:
    """Index of the answered label in the order the prompt PRESENTED them. -1 if it did not commit."""
    last = row["completion"].strip().splitlines()[-1].replace("Answer:", "").strip()
    return row["labels"].index(last) if last in row["labels"] else -1


def _position_from_text(text: str, labels: list[str]) -> int:
    """Same, for a raw generation: read the answer AFTER the reasoning block, never inside it."""
    body = text.partition("</think>")[2] or text
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    if not lines:
        return -1
    last = lines[-1].replace("Answer:", "").strip()
    return labels.index(last) if last in labels else -1


def _deviation(counts: Counter, n_pos: int, total: int) -> float:
    ideal = total / n_pos
    return sum(abs(counts[i] - ideal) for i in range(n_pos))


def choose_swaps(pairs: list[tuple[str, int, int]], n_pos: int) -> set[str]:
    """Which questions to swap so the position histogram sits closest to uniform.

    NOT "swap every pair that disagrees" -- that overshoots. On the 4B's ternary_neutral, 8 pairs
    disagreed but only 5 swaps were wanted; the other 3 had the training row already on an
    under-represented position, so flipping them traded one imbalance for another.
    """
    fixed = Counter(train for _, train, spare in pairs if train == spare)
    movable = [(rid, train, spare) for rid, train, spare in pairs if train != spare]
    total = len(pairs)
    if not movable:
        return set()

    if len(movable) <= EXHAUSTIVE_LIMIT:
        best: tuple[float, set[str]] | None = None
        for mask in itertools.product((0, 1), repeat=len(movable)):
            counts = Counter(fixed)
            for bit, (_, train, spare) in zip(mask, movable):
                counts[spare if bit else train] += 1
            dev = _deviation(counts, n_pos, total)
            if best is None or dev < best[0]:
                best = (dev, {rid for bit, (rid, _, _) in zip(mask, movable) if bit})
        return best[1]

    # Greedy: repeatedly take the single swap that most reduces deviation. Deterministic via the
    # record_id tie-break so a re-run reproduces the same set.
    chosen: set[str] = set()
    counts = Counter(fixed)
    for _, train, _ in movable:
        counts[train] += 1
    while True:
        best_gain, best_rid = 0.0, None
        for rid, train, spare in sorted(movable):
            if rid in chosen:
                continue
            trial = Counter(counts)
            trial[train] -= 1
            trial[spare] += 1
            gain = _deviation(counts, n_pos, total) - _deviation(trial, n_pos, total)
            if gain > best_gain:
                best_gain, best_rid = gain, rid
        if best_rid is None:
            return chosen
        rid, train, spare = next(m for m in movable if m[0] == best_rid)
        counts[train] -= 1
        counts[spare] += 1
        chosen.add(rid)


def rederive(text: str, rec: dict, tokenizer, args) -> tuple[str, int] | None:
    """Re-screen a raw generation through the builder's filters, in the builder's order.

    The order is load-bearing and mirrors build_sft_warmstart: evaluate_trace first, THEN prose,
    THEN the echo screen and the floor re-check on the prose result. Screening after prose would
    let a sample that only passes because prose deleted its preamble slip through.
    """
    status, clean, n = evaluate_trace(
        text, rec["labels"], rec.get("answer"), bool(rec.get("verifiable")),
        tokenizer, args.max_trace_tokens, args.max_coda_chars, args.strip_coda,
        args.min_trace_tokens)
    if status != "pass":
        return None
    if args.prose:
        clean, n = bullets_to_prose(clean, tokenizer)
    if is_format_echo(clean) or n < args.min_trace_tokens:
        return None
    return clean, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="the TRAINING file; spares/generations are "
                                                    "found alongside it by --out stem convention")
    ap.add_argument("--dataset", default="data/rl/m0_format/train.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="tokenizer only -- no weights, no GPU")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the trace files. Without it nothing is written except the report, "
                         "so the numbers can be read before anything is touched.")
    ap.add_argument("--no_recover", action="store_true",
                    help="use sibling swaps only; do not pull unused generations back in")
    # Mirrors of the builder's screens. Defaults match its defaults; a recovered trace must clear
    # the identical bar, so these should only ever be changed together with the builder's.
    ap.add_argument("--max_trace_tokens", type=int, default=256)
    ap.add_argument("--min_trace_tokens", type=int, default=32)
    ap.add_argument("--max_coda_chars", type=int, default=0)
    ap.add_argument("--strip_coda", action="store_true")
    ap.add_argument("--prose", action="store_true", default=True)
    args = ap.parse_args()

    train_path = Path(args.traces)
    spares_path = train_path.with_name(f"{train_path.stem}_spares.jsonl")
    gens_path = train_path.with_name(f"{train_path.stem}_generations.jsonl")
    for p in (train_path, spares_path):
        if not p.exists():
            raise SystemExit(f"{p} not found -- pass the TRAINING file produced by "
                             f"build_sft_warmstart.py; its spares file must sit beside it.")

    train = [json.loads(l) for l in train_path.read_text().splitlines() if l.strip()]
    spares = [json.loads(l) for l in spares_path.read_text().splitlines() if l.strip()]
    by_rid_train = {r["record_id"]: r for r in train}
    spare_by_rid: dict[str, list[dict]] = defaultdict(list)
    for r in spares:
        spare_by_rid[r["record_id"]].append(r)

    families = sorted({r["family"] for r in train if not r.get("verifiable")})
    report: dict = {"traces": str(train_path), "families": {}}
    swapped: set[str] = set()
    recovered: dict[str, dict] = {}

    print(f"[balance] {train_path}: {len(train)} training rows, {len(spares)} spares")
    print(f"[balance] neutral families: {families}  (verifiable families are pinned by the "
          f"correctness filter and are left alone)\n")

    for family in families:
        fam_rows = [r for r in train if r["family"] == family]
        n_pos = len(fam_rows[0]["labels"])
        total = len(fam_rows)
        before = Counter(chosen_position(r) for r in fam_rows)

        pairs = []
        for r in fam_rows:
            sp = spare_by_rid.get(r["record_id"])
            pairs.append((r["record_id"], chosen_position(r),
                          chosen_position(sp[0]) if sp else chosen_position(r)))
        picks = choose_swaps(pairs, n_pos)
        swapped |= picks

        after = Counter()
        for rid, tr_pos, sp_pos in pairs:
            after[sp_pos if rid in picks else tr_pos] += 1

        report["families"][family] = {
            "n": total, "n_positions": n_pos, "chance": round(1 / n_pos, 4),
            "before": {str(k): before[k] for k in range(n_pos)},
            "after_swaps": {str(k): after[k] for k in range(n_pos)},
            "swaps": sorted(picks),
            "first_listed_before": round(before[0] / total, 4),
            "first_listed_after_swaps": round(after[0] / total, 4),
        }
        print(f"[balance] {family}: {total} questions, chance {1/n_pos:.1%}")
        print(f"[balance]   before      {dict(sorted(before.items()))}  "
              f"first-listed {before[0]/total:.1%}")
        print(f"[balance]   {len(picks)} swap(s) -> {dict(sorted(after.items()))}  "
              f"first-listed {after[0]/total:.1%}")

    # ---- recovery pass: only for positions still under-filled after swapping ----
    if not args.no_recover and gens_path.exists():
        recs = {json.loads(l)["record_id"]: json.loads(l)
                for l in Path(args.dataset).read_text().splitlines() if l.strip()}
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        gens = [json.loads(l) for l in gens_path.read_text().splitlines() if l.strip()]
        by_rid_gen: dict[str, list[dict]] = defaultdict(list)
        for g in gens:
            if g["status"] == "pass":
                by_rid_gen[g["record_id"]].append(g)

        for family, info in report["families"].items():
            n_pos, total = info["n_positions"], info["n"]
            counts = Counter({int(k): v for k, v in info["after_swaps"].items()})
            ideal = total / n_pos
            fam_rids = [r["record_id"] for r in train if r["family"] == family]
            moves = []
            # Take from the most over-filled position, give to the most under-filled, one at a time.
            while max(counts[i] - ideal for i in range(n_pos)) > 1:
                over = max(range(n_pos), key=lambda i: counts[i])
                under = min(range(n_pos), key=lambda i: counts[i])
                cand = None
                for rid in sorted(fam_rids):
                    if rid in recovered:
                        continue
                    cur = (chosen_position(spare_by_rid[rid][0]) if rid in swapped
                           else chosen_position(by_rid_train[rid]))
                    if cur != over:
                        continue
                    rec = recs.get(rid)
                    if rec is None:
                        continue
                    for g in by_rid_gen.get(rid, []):
                        if _position_from_text(g["text"], rec["labels"]) != under:
                            continue
                        out = rederive(g["text"], rec, tokenizer, args)
                        if out:
                            cand = (rid, out[0], out[1])
                            break
                    if cand:
                        break
                if cand is None:
                    break
                rid, clean, n = cand
                recovered[rid] = {"completion": clean, "n_reasoning_tokens": n}
                counts[over] -= 1
                counts[under] += 1
                moves.append(f"{rid}: pos {over} -> {under}")
            info["recovered"] = moves
            info["after_recovery"] = {str(k): counts[k] for k in range(n_pos)}
            info["first_listed_final"] = round(counts[0] / total, 4)
            info["residual_shortfall"] = max(0, int(round(max(
                counts[i] - ideal for i in range(n_pos)) - 1)))
            if moves:
                print(f"[balance] {family}: recovered {len(moves)} unused generation(s) -> "
                      f"{dict(sorted(counts.items()))}  first-listed {counts[0]/total:.1%}")
                for m in moves:
                    print(f"[balance]     {m}")
            if info["residual_shortfall"]:
                print(f"[balance] {family}: STILL {info['residual_shortfall']} short of chance -- "
                      f"neither sibling nor any unused generation answers the other way. Fill with a "
                      f"fresh sampling pass or accept it, but do not call this balanced.")

    report_path = train_path.with_name(f"{train_path.stem}_position_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[balance] wrote {report_path}")

    if not args.apply:
        print("[balance] REPORT ONLY -- nothing rewritten. Re-run with --apply to commit it.")
        return 0

    new_train, new_spares = [], []
    for r in train:
        rid = r["record_id"]
        if rid in recovered:
            # The recovered trace becomes the training row; the row it displaces becomes the spare,
            # so the question still carries a reviewable alternative rather than losing one.
            promoted = dict(r, **recovered[rid], sibling=0, position_source="recovered_generation")
            new_train.append(promoted)
            new_spares.append(dict(r, sibling=1))
        elif rid in swapped and spare_by_rid.get(rid):
            sp = spare_by_rid[rid][0]
            new_train.append(dict(sp, sibling=0, position_source="sibling_swap"))
            new_spares.append(dict(r, sibling=1))
        else:
            new_train.append(r)
            new_spares.extend(dict(s, sibling=1) for s in spare_by_rid.get(rid, []))

    backup = train_path.with_name(f"{train_path.stem}_prebalance.jsonl")
    backup_sp = spares_path.with_name(f"{spares_path.stem}_prebalance.jsonl")
    if not backup.exists():
        backup.write_text(train_path.read_text())
        backup_sp.write_text(spares_path.read_text())
        print(f"[balance] original preserved at {backup.name} / {backup_sp.name}")
    train_path.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in new_train))
    spares_path.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in new_spares))
    print(f"[balance] rewrote {train_path} ({len(new_train)} rows) and "
          f"{spares_path} ({len(new_spares)} rows)")
    print(f"[balance] {len(swapped)} swap(s), {len(recovered)} recovery(ies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
