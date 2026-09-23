#!/usr/bin/env python3
"""Re-report existing coding evaluations as (termination rate, quality | terminated).

A stored eval score is a PRODUCT of two things: how often the model produced a
finished program, and how good that program was. The verifier scores a truncated
completion exactly 0.0 — indistinguishable from a wrong answer — so a checkpoint
that merely learns to stop earlier posts a higher number without being any better
at coding.

That is not hypothetical. On the 4B Base DPO held-out set the published headline
was step 28 at +2.82pp over Base with a bootstrap interval excluding zero. Split
it and the entire gain is termination (+3.48pp) while quality-given-termination
actually FELL (-0.66pp). Step 125 is the same trade run further: +2.90pp
termination against -5.46pp quality.

This script needs no GPU and no retraining — it re-reads completions that already
exist. Use it before acting on any stored pass rate.

Truncation is read from ``finish_reason`` when the run recorded it. Older runs did
not, so it falls back to the unclosed-code-fence heuristic in
``rl_training.rewards.extract_code_with_status``; where both are available the
report includes the measured agreement between them, so the fallback's
trustworthiness is a number rather than an assumption.

    python scripts/rescore_truncation.py <dir-or-file> [...] --baseline base \\
        --out results/rescored.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rl_training.rewards import COMPLETION_OK, extract_code_with_status  # noqa: E402

BOOTSTRAP_DRAWS = 10000
# The budgets the arms have actually been trained and measured at, so the survival
# curve answers "what fraction of rollouts could have carried gradient at X".
LENGTH_GRID = (512, 1024, 1536, 2048, 4096, 8192, 16384)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_samples(path: Path) -> list[dict]:
    """Normalize both stored shapes into flat per-generation samples.

    ``rl_heldout_reward_completion_v1`` is one row per generation;
    ``livecodebench_qwen35_v1`` nests candidates under ``outputs``. LCB rows carry
    no reward (scoring happens in the LCB evaluator), so they contribute
    termination statistics only.
    """
    samples = []
    for row in read_jsonl(path):
        if "outputs" in row:
            for index, candidate in enumerate(row["outputs"]):
                samples.append({
                    "model": row.get("model_label") or path.parent.name,
                    "item": f'{row.get("question_id")}#{index}',
                    "reward": None,
                    "text": candidate.get("text", ""),
                    "finish_reason": candidate.get("finish_reason"),
                    "tokens": candidate.get("token_count"),
                    "cap": row.get("max_tokens"),
                })
        else:
            samples.append({
                "model": row.get("model") or row.get("served_model") or path.parent.name,
                "item": str(row.get("record_id") or row.get("dataset_index")),
                "reward": row.get("reward"),
                "text": row.get("completion", ""),
                "finish_reason": row.get("finish_reason"),
                "tokens": row.get("completion_token_count"),
                "cap": row.get("max_tokens"),
            })
    return samples


def classify(sample: dict) -> tuple[bool, str]:
    """Return (terminated, method). ``finish_reason`` wins when recorded."""
    reason = sample.get("finish_reason")
    if reason:
        return reason != "length", "finish_reason"
    _, status = extract_code_with_status(sample.get("text") or "")
    return status == COMPLETION_OK, "fence_heuristic"


def heuristic_agreement(samples: list[dict]) -> dict | None:
    """How often the fence heuristic matches ``finish_reason``, where both exist.

    This is what licenses (or doesn't) the fallback on older runs that never
    recorded a finish reason.
    """
    both = [s for s in samples if s.get("finish_reason")]
    if not both:
        return None
    agree = 0
    for sample in both:
        truth = sample["finish_reason"] != "length"
        _, status = extract_code_with_status(sample.get("text") or "")
        agree += int((status == COMPLETION_OK) == truth)
    return {"n": len(both), "agreement": agree / len(both)}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(samples: list[dict]) -> dict:
    n = len(samples)
    terminated = [s for s in samples if s["_terminated"]]
    scored = [s for s in samples if s["reward"] is not None]
    scored_terminated = [s for s in terminated if s["reward"] is not None]

    out = {
        "n_generations": n,
        "n_terminated": len(terminated),
        "termination_rate": len(terminated) / n if n else 0.0,
        "detection": sorted({s["_method"] for s in samples}),
    }
    if scored:
        out["mean_reward_merged"] = _mean([s["reward"] for s in scored])
        out["strict_full_pass_merged"] = _mean([float(s["reward"] == 1.0) for s in scored])
    if scored_terminated:
        out["mean_reward_given_termination"] = _mean([s["reward"] for s in scored_terminated])
        out["strict_full_pass_given_termination"] = _mean(
            [float(s["reward"] == 1.0) for s in scored_terminated]
        )
    tokens = [s["tokens"] for s in samples if isinstance(s.get("tokens"), int)]
    if tokens:
        tokens_sorted = sorted(tokens)
        # Generations censored at the run's own cap tell us nothing about budgets
        # ABOVE that cap — a completion cut at 8192 might have needed 9k or 90k. So
        # the curve stops at the cap rather than reporting a fictitious 100%.
        caps = [s.get("cap") for s in samples if isinstance(s.get("cap"), int)]
        run_cap = min(caps) if caps else None
        grid = [c for c in LENGTH_GRID if run_cap is None or c <= run_cap]
        out["completion_tokens"] = {
            "median": tokens_sorted[len(tokens_sorted) // 2],
            "max": tokens_sorted[-1],
            "run_cap": run_cap,
            # "What fraction would have finished inside a budget of X" — read this
            # against the arm's max_completion_length to see how much of the rollout
            # batch could have carried any gradient at all.
            "survival_at": {
                str(cap): sum(1 for t in tokens_sorted if t < cap) / len(tokens_sorted)
                for cap in grid
            },
        }
    return out


def paired_bootstrap(baseline: dict[str, float], candidate: dict[str, float],
                     seed: int = 0) -> dict | None:
    """Paired bootstrap over the items both models were scored on."""
    shared = sorted(set(baseline) & set(candidate))
    if len(shared) < 2:
        return None
    deltas = [candidate[item] - baseline[item] for item in shared]
    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP_DRAWS):
        means.append(_mean([deltas[rng.randrange(len(deltas))] for _ in deltas]))
    means.sort()
    return {
        "n_paired": len(shared),
        "mean_delta": _mean(deltas),
        "bootstrap_95_low": means[int(0.025 * BOOTSTRAP_DRAWS)],
        "bootstrap_95_high": means[int(0.975 * BOOTSTRAP_DRAWS)],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def decompose(baseline: dict, candidate: dict) -> dict | None:
    """Split the headline delta into a termination part and a quality part.

    merged = termination_rate * quality_given_termination (exactly, because a
    truncated generation scores 0). So the change decomposes as

        d_merged = (t1 - t0) * q0   +   t1 * (q1 - q0)
                    termination            quality

    Any checkpoint whose gain sits in the first term learned to stop, not to code.
    """
    keys = ("termination_rate", "mean_reward_given_termination")
    if not all(k in baseline and k in candidate for k in keys):
        return None
    t0, q0 = baseline["termination_rate"], baseline["mean_reward_given_termination"]
    t1, q1 = candidate["termination_rate"], candidate["mean_reward_given_termination"]
    return {
        "delta_merged": t1 * q1 - t0 * q0,
        "from_termination": (t1 - t0) * q0,
        "from_quality": t1 * (q1 - q0),
    }


def render(name: str, report: dict) -> None:
    print(f"\n=== {name}")
    models = report["models"]
    width = max(len(m) for m in models)
    # LCB generation dumps carry no reward (scoring happens in the LCB evaluator),
    # so those cells report termination only rather than a column of NaNs.
    scored = any("mean_reward_merged" in stats for stats in models.values())
    header = f"{'model':<{width}}  {'n':>5} {'finished':>9}"
    if scored:
        header += f" {'merged':>8} {'| finished':>10} {'strict':>7} {'strict|fin':>10}"
    print(header)
    for model, stats in models.items():
        line = (f"{model:<{width}}  {stats['n_generations']:>5} "
                f"{stats['termination_rate']:>8.1%}")
        if scored:
            def fmt(key, spec):
                value = stats.get(key)
                return format(value, spec) if value is not None else "-".rjust(int(spec.split(".")[0]))
            line += (f" {fmt('mean_reward_merged', '8.4f')}"
                     f" {fmt('mean_reward_given_termination', '10.4f')}"
                     f" {fmt('strict_full_pass_merged', '7.1%')}"
                     f" {fmt('strict_full_pass_given_termination', '10.1%')}")
        print(line)
    for model, comp in report.get("comparisons", {}).items():
        dec = comp.get("decomposition")
        if not dec:
            continue
        print(
            f"  {model} vs {report['baseline']}: "
            f"{100 * dec['delta_merged']:+.2f}pp total = "
            f"{100 * dec['from_termination']:+.2f}pp termination "
            f"{100 * dec['from_quality']:+.2f}pp quality"
        )
        boot = comp.get("bootstrap_merged")
        if boot:
            print(
                f"      merged 95% CI [{100 * boot['bootstrap_95_low']:+.2f}, "
                f"{100 * boot['bootstrap_95_high']:+.2f}]pp over {boot['n_paired']} items"
            )
    agree = report.get("heuristic_agreement")
    if agree:
        print(f"  fence heuristic agrees with finish_reason on "
              f"{agree['agreement']:.1%} of {agree['n']} generations")
    for model, stats in models.items():
        survival = stats.get("completion_tokens", {}).get("survival_at")
        if survival:
            shown = "  ".join(f"{cap}:{rate:.0%}" for cap, rate in survival.items())
            run_cap = stats["completion_tokens"].get("run_cap")
            suffix = f"  (measured at a {run_cap}-token cap; longer budgets unknown)" if run_cap else ""
            print(f"  {model} finished-within-budget: {shown}{suffix}")


def analyse(path: Path, baseline_hint: str | None) -> dict:
    samples = load_samples(path)
    if not samples:
        raise SystemExit(f"No samples found in {path}")
    agreement = heuristic_agreement(samples)
    for sample in samples:
        sample["_terminated"], sample["_method"] = classify(sample)

    by_model: dict[str, list[dict]] = {}
    for sample in samples:
        by_model.setdefault(sample["model"], []).append(sample)

    models = {name: summarize(rows) for name, rows in sorted(by_model.items())}
    baseline = baseline_hint if baseline_hint in models else next(iter(models))

    rewards_by_model = {
        name: {s["item"]: s["reward"] for s in rows if s["reward"] is not None}
        for name, rows in by_model.items()
    }
    comparisons = {}
    for name in models:
        if name == baseline:
            continue
        comparisons[name] = {
            "decomposition": decompose(models[baseline], models[name]),
            "bootstrap_merged": paired_bootstrap(
                rewards_by_model[baseline], rewards_by_model[name]
            ),
        }
    return {
        "schema": "truncation_rescore_v1",
        "source": str(path),
        "baseline": baseline,
        "models": models,
        "comparisons": comparisons,
        "heuristic_agreement": agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="completions/generations JSONL files or directories")
    parser.add_argument("--baseline", default=None, help="Model label to compare against")
    parser.add_argument("--out", default=None, help="Write the full report as JSON here")
    args = parser.parse_args()

    targets: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            targets.extend(sorted(path.rglob("completions.jsonl")))
            targets.extend(sorted(path.rglob("generations.jsonl")))
        else:
            targets.append(path)
    if not targets:
        raise SystemExit(f"No completions.jsonl / generations.jsonl found under {args.paths}")

    reports = {}
    for target in targets:
        name = str(target.parent)
        reports[name] = analyse(target, args.baseline)
        render(name, reports[name])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
        print(f"\n[rescore] wrote {out_path}")


if __name__ == "__main__":
    main()
