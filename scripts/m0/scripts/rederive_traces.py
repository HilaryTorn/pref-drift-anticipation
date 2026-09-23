#!/usr/bin/env python3
"""Rebuild a warm-start trace set from its own all_generations.jsonl, on CPU, with the current filters.

WHY THIS EXISTS. Everything between "the model emitted this text" and "this is the SFT target" is deterministic post-processing: pick the best passing sample for each prompt, render its bullets to prose, screen it. Only the generation step needs a GPU. So when that post-processing is found to be wrong -- as it was on 2026-07-31, when bullets_to_prose recognised "- " but not "1. " and therefore treated a numbered response as having no bullets at all, falling back to every-line and carrying both the numerals and the model's meta-preamble into the target -- the fix does not require regenerating anything. `all_generations.jsonl` already holds every sample's raw text, which is the expensive part.

Replaying is sound because the current filters only ever REMOVE candidates relative to the run that produced the file: a sample the old build rejected cannot become acceptable under a stricter screen, so the old `pass` rows are the complete candidate pool. What replaying CAN do is change which of a prompt's passing samples wins (the format-echo screen now runs before length-targeting instead of after), and drop a prompt whose every sample fails the new screens -- which is why this reports per-slot shortfall rather than silently returning fewer traces.

    python m0/scripts/rederive_traces.py --run results/m0_warmstart_9b_v3 --model Qwen/Qwen3.5-9B

Writes `sft_traces_rederived.jsonl` beside the input and leaves the original untouched. Any resulting shortfall against the run's quota is what `build_warmstart.py --topup` then fills on the box.
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

# One source of truth: the selection and screening rules are imported from the builder, never
# reimplemented here. A copy would drift from the thing it is meant to reproduce.
from scripts.m0.data.build_warmstart import bullets_to_prose, is_format_echo, _quota_key
from scripts.m0.rewards import _count_tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="a warm-start output directory: all_generations.jsonl + manifest.json")
    ap.add_argument("--dataset", default="data/rl/m0_format/train.jsonl",
                    help="must be the SAME build the run used -- record_ids are only meaningful within one")
    ap.add_argument("--model", default=None, help="tokenizer to count reasoning tokens with; "
                                                  "defaults to the model recorded in the manifest")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text())
    out = Path(args.out) if args.out else run / "sft_traces_rederived.jsonl"

    dataset = {}
    for line in Path(args.dataset).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            dataset[rec["record_id"]] = rec

    # The main pool AND any --topup pool. A topup writes to `<run>/topup/all_generations.jsonl`
    # rather than appending, so reading only the main file silently loses every record the topup
    # rescued. Measured on results/m0_warmstart_4b_v3: m0.binary_neutral.00081 has zero rows in the
    # main pool and six passing rows in the topup, so replaying returned 199 traces against a
    # shipped set of 200 -- a shortfall that reads like a filter change rather than an unread file.
    pools = [run / "all_generations.jsonl"]
    topup = run / "topup" / "all_generations.jsonl"
    if topup.exists():
        pools.append(topup)
    generations = []
    for pool in pools:
        generations += [json.loads(line) for line in pool.read_text().splitlines() if line.strip()]
    print(f"[rederive] pools: {', '.join(str(p.relative_to(run)) for p in pools)} "
          f"-> {len(generations)} generations")
    unknown = {g["record_id"] for g in generations} - set(dataset)
    if unknown:
        raise SystemExit(
            f"{len(unknown)} record_ids in {run}/all_generations.jsonl are absent from {args.dataset} "
            f"(e.g. {sorted(unknown)[:3]}). That means the dataset has been rebuilt since the run: "
            f"record_ids are assigned per build and are not stable across generator changes. Rebuild "
            f"the dataset at the commit the run used before replaying."
        )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model or manifest["model"])

    target = manifest.get("target_trace_tokens")
    floor = manifest.get("min_trace_tokens") or 32
    # Prefer the RECORDED flag; fall back to the old inference only for manifests written before
    # build_warmstart started recording it (every shipped set through 2026-08-03). The inference is
    # right for every brevity variant shipped so far but is still a guess, and getting it wrong
    # silently changes what the replayed SFT target looks like -- prose vs literal "- " bullets.
    if "prose" in manifest:
        use_prose = bool(manifest["prose"])
    else:
        use_prose = bool(manifest.get("brevity")) and "bullet" in str(manifest.get("brevity"))
        print(f"[rederive] manifest predates the 'prose' field; inferring prose={use_prose} "
              f"from brevity={manifest.get('brevity')!r}")

    candidates: dict[str, list] = defaultdict(list)
    for gen in generations:
        if gen["status"] != "pass":
            continue
        # evaluate_trace returned text.rstrip() as the clean completion whenever the coda was empty,
        # which max_coda_chars=0 already guaranteed for every row that reached "pass".
        clean = gen["text"].rstrip()
        n = gen["n_reasoning_tokens"]
        if use_prose:
            clean, n = bullets_to_prose(clean, tokenizer)
        candidates[gen["record_id"]].append((clean, n))

    kept, dropped = [], Counter()
    lost_prompts = []
    for record_id, cands in candidates.items():
        screened = [(c, n) for c, n in cands if not is_format_echo(c) and n >= floor]
        if not screened:
            for c, n in cands:
                dropped["format_echo" if is_format_echo(c) else "too_short"] += 1
            lost_prompts.append(record_id)
            continue
        if target is not None:
            clean, n = min(screened, key=lambda r: abs(r[1] - target))
        else:
            clean, n = min(screened, key=lambda r: r[1])
        rec = dataset[record_id]
        kept.append({
            "schema": "m0_warmstart_sft_v1",
            "record_id": record_id,
            "family": rec["family"],
            "labels": rec["labels"],
            "answer": rec.get("answer"),
            "verifiable": bool(rec.get("verifiable")),
            "prompt": rec["prompt"],
            "completion": clean,
            "n_reasoning_tokens": n,
            "brevity": manifest.get("brevity"),
        })

    kept.sort(key=lambda r: r["record_id"])
    out.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in kept))

    print(f"[rederive] {run.name}: {len(candidates)} prompts with a passing sample -> {len(kept)} traces")
    if lost_prompts:
        print(f"[rederive]   {len(lost_prompts)} prompt(s) lost, every sample screened out: {lost_prompts}")
        for reason, n in dropped.most_common():
            print(f"[rederive]     {reason}: {n} sample(s)")

    # Compare against what the run originally shipped, so the effect of the replay is visible rather
    # than assumed. Same prompt, different target text = a re-selection or a re-render.
    original_path = run / "sft_traces.jsonl"
    if original_path.exists():
        original = {json.loads(l)["record_id"]: json.loads(l)
                    for l in original_path.read_text().splitlines() if l.strip()}
        changed = [r for r in kept if r["record_id"] in original
                   and original[r["record_id"]]["completion"] != r["completion"]]
        print(f"[rederive]   {len(changed)} trace(s) changed text vs the original set")
        for r in changed[:5]:
            print(f"[rederive]     {r['record_id']}")
            print(f"[rederive]       was: {original[r['record_id']]['completion'][:88]!r}")
            print(f"[rederive]       now: {r['completion'][:88]!r}")

    targets = manifest.get("quota_targets") or {}
    if targets:
        have = Counter(_quota_key(r) for r in kept)
        short = {slot: t - have[slot] for slot, t in targets.items() if have[slot] < t}
        if short:
            print(f"[rederive]   SHORT {sum(short.values())} trace(s) against the run's quota: {short}")
            print(f"[rederive]   fill with: build_warmstart.py --topup {out}")
        else:
            print(f"[rederive]   every quota slot met ({sum(targets.values())} traces)")
    print(f"[rederive] wrote {out}")


if __name__ == "__main__":
    main()
