#!/usr/bin/env python3
"""Recover forced-choice votes that the strict reasoning-mode parser dropped, without re-running generation.

WHY THIS EXISTS. `parse_responses_forced_choice(with_reasoning=True)` accepts a vote only when the response's last non-empty line is a bare "Answer: X" (see `terminal_answer_label`). That rule is correct for the failure mode it was written against: a reasoning-on generation that runs past max_tokens and litters its scratchpad with format-instruction echoes and abandoned drafts, where a take-last-anywhere match scores a never-committed ramble as a real preference. But on the Big5 SFT arms the dominant failure is a *different* one. The trained checkpoints frequently skip the `<think>` block entirely and answer in ordinary prose ("I would rather be the one who questions uncertainties and ensures accuracy..."), then stop cleanly. Those are finished, committed answers with no scratchpad to be confused by, and the strict rule discards all of them. On qwen35-4b-big5sft-openness-low-step40-aws/values that is 27.7% of the battery against 0.4% for the M0 baseline, and the loss is item-dependent, so complete-case win rates compare a near-complete baseline against arms missing their most opinionated responses.

WHAT THIS DOES NOT DO. It does not relax the rule for responses that *do* contain a `<think>` block, because for those the original rationale still applies in full. It does not touch the original files. It does not call any model or API. Recovery is deterministic, tiered, and every recovered vote is written with the tier and the evidence that produced it so the whole thing can be audited or thrown away wholesale.

TIERS, most conservative first. Only `terminal_label` and `single_label` are enabled by default.
  terminal_label  - the final non-empty line names exactly one option label, possibly wrapped in markdown or prefixed by a stray word ("**B**", "I choose A."). Strict-parser-equivalent modulo formatting noise.
  single_label    - prose, no think block, terminates cleanly, and exactly one of the two labels appears anywhere as a standalone token. Nothing to disambiguate.
  terminal_endorse- prose naming BOTH labels ("Option A is good for X. Option B is better for Y."), where the final sentence names exactly one. Comparative write-ups that land on a choice. Opt-in via --enable-endorse; this is the tier most likely to be wrong, because a final sentence can name the option being rejected.
  semantic        - prose naming NEITHER label, paraphrasing one option's content instead. Requires matching against the option descriptions. Opt-in via --enable-semantic, and even then only fires on a decisive lexical margin; everything below the margin is left for an LLM judge. Export those with --judge-export.

REJECTED ALWAYS, never recovered under any tier:
  has_think       - a `<think>` block is present but no terminal answer: the original truncated-ramble case.
  degenerate_loop - low type-token ratio, the model looping on a phrase until it hit the cap.
  truncated       - ends mid-word or without terminal punctuation, i.e. the generation was cut off.
  instruction_echo- contains the prompt's own format instruction, which is what makes bare label matching unsafe.
  ambiguous       - the tier's disambiguation rule did not resolve to exactly one option.

OUTPUT. A sidecar `recovered_<original name>.jsonl` alongside the input, one record per *originally unparseable* row, carrying the original identifying fields plus `recovery_tier`, `recovered_response` (the presented label, matching `parsed_response_semantics`) and `recovery_evidence`. Plus a report to stdout, and with --win-rate-delta a before/after per-item win-rate comparison so the effect on the drift numbers is visible before anyone decides to trust this.

Usage:
  python scripts/recover_dropped_choices.py results/<arm>/values/raw_responses_*.jsonl --win-rate-delta
  python scripts/recover_dropped_choices.py results/<arm>/*/raw_responses_*.jsonl --enable-endorse --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# A label token standing on its own: not part of a word, not part of "Option A:" being quoted back
# as part of the instruction. Case-sensitive on purpose - the batteries use uppercase A/B, and
# lowercasing turns every indefinite article "a" in English prose into a vote.
LABEL_TOKEN = re.compile(r"(?<![A-Za-z0-9_])([AB])(?![A-Za-z0-9_])")

# Leading markdown/quote noise the strict parser already tolerates on an answer line.
ANSWER_LINE = re.compile(r"^[\s>*_#`\-]*(?:answer|final answer|choice)\s*[:\-]?\s*\**\s*([AB])\b", re.IGNORECASE)

# The prompt's own format instruction, echoed back. If this is present, a bare label match is
# reading the instruction rather than the model's answer.
INSTRUCTION_ECHO = re.compile(
    r"(answer\s*[:\-]?\s*[\"'`]?A[\"'`]?\s*(?:or|/|,)\s*[\"'`]?(?:answer\s*[:\-]?\s*)?B)"
    r"|(on (?:its|your) own line)"
    r"|(exactly as)",
    re.IGNORECASE,
)

SENTENCE_END = re.compile(r"[.!?\"')\]]\s*$")

STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had has have he her hers him his how i if in
into is it its me my no not of on or our ours she should so that the their theirs them then there these they this
to was we were what when where which who whom why will with would you your yours
""".split())


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def last_nonempty_line(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        if line.strip():
            return line.strip()
    return ""


def final_sentence(text: str) -> str:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return parts[-1] if parts else ""


def is_degenerate_loop(text: str) -> bool:
    words = text.split()
    return len(words) >= 40 and len(set(words)) / len(words) < 0.35


def looks_truncated(text: str) -> bool:
    """A cleanly finished prose answer ends on sentence punctuation. Anything else is very likely a
    generation that hit the token cap mid-thought, which is exactly the case the strict parser is
    right to refuse."""
    t = text.strip()
    return bool(t) and not SENTENCE_END.search(t)


def classify(record: dict, enable_endorse: bool, enable_semantic: bool, semantic_margin: int):
    """Return (tier, recovered_label_or_None, evidence). `tier` names the outcome even when nothing
    is recovered, so the report accounts for every dropped row."""
    raw = (record.get("raw_response") or "").strip()
    if not raw:
        return "empty", None, ""
    if "<think>" in raw:
        return "has_think", None, ""
    if is_degenerate_loop(raw):
        return "degenerate_loop", None, ""
    if looks_truncated(raw):
        return "truncated", None, raw[-60:]
    if INSTRUCTION_ECHO.search(raw):
        return "instruction_echo", None, ""

    # Tier 1: a real answer line at the end, just wearing formatting.
    tail = last_nonempty_line(raw)
    m = ANSWER_LINE.match(tail)
    if m:
        return "terminal_label", m.group(1).upper(), tail[:80]
    if len(tail) <= 4:
        bare = LABEL_TOKEN.findall(tail)
        if len(set(bare)) == 1:
            return "terminal_label", bare[0], tail

    found = set(LABEL_TOKEN.findall(raw))

    # Tier 2: prose mentioning exactly one label. Nothing to disambiguate.
    if len(found) == 1:
        return "single_label", found.pop(), ""

    # Tier 3: prose weighing both, where the closing sentence commits to one.
    if len(found) == 2:
        if not enable_endorse:
            return "both_labels_skipped", None, ""
        closing = final_sentence(raw)
        in_closing = set(LABEL_TOKEN.findall(closing))
        if len(in_closing) == 1:
            return "terminal_endorse", in_closing.pop(), closing[:100]
        return "ambiguous", None, closing[:100]

    # Tier 4: prose naming neither label, paraphrasing an option instead.
    if not enable_semantic:
        return "semantic_skipped", None, ""
    first = tokens(record.get("presented_first_description") or "")
    second = tokens(record.get("presented_second_description") or "")
    body = tokens(raw)
    # Score only on words that distinguish the two options; shared words say nothing about which
    # one the model picked.
    s_first = len(body & (first - second))
    s_second = len(body & (second - first))
    if abs(s_first - s_second) < semantic_margin:
        return "semantic_below_margin", None, f"{s_first}v{s_second}"
    return "semantic", ("A" if s_first > s_second else "B"), f"{s_first}v{s_second}"


RECOVERED_TIERS = {"terminal_label", "single_label", "terminal_endorse", "semantic"}

KEEP_FIELDS = (
    "split", "iteration", "prompt_idx", "sample_idx",
    "option_a_id", "option_b_id", "direction",
    "presented_first_id", "presented_second_id",
    "parsed_response_semantics",
)


def winner_id(record: dict, label: str):
    """Map a presented label back to an option id. Mirrors the semantics the batteries record in
    `parsed_response_semantics`; anything else would silently invert half the votes."""
    if record.get("parsed_response_semantics") not in (None, "presented_label"):
        return None
    return record["presented_first_id"] if label == "A" else record["presented_second_id"]


def per_item_win_rates(records, include_recovered: dict[int, str] | None = None):
    wins, total = defaultdict(int), defaultdict(int)
    for i, r in enumerate(records):
        label = None
        if r.get("parse_status") == "parsed" and r.get("parsed_response") in ("A", "B"):
            label = r["parsed_response"]
        elif include_recovered is not None:
            label = include_recovered.get(i)
        if label is None:
            continue
        w = winner_id(r, label)
        if w is None:
            continue
        for oid in (r["option_a_id"], r["option_b_id"]):
            total[oid] += 1
        wins[w] += 1
    return {o: wins[o] / total[o] for o in total if total[o]}


def process(path: Path, args) -> dict:
    records = [json.loads(line) for line in path.open()]
    tiers = Counter()
    recovered: dict[int, str] = {}
    out_rows = []

    for i, r in enumerate(records):
        if r.get("parse_status") == "parsed" and r.get("parsed_response") in ("A", "B"):
            continue
        tier, label, evidence = classify(r, args.enable_endorse, args.enable_semantic, args.semantic_margin)
        tiers[tier] += 1
        if tier in RECOVERED_TIERS and label is not None:
            recovered[i] = label
        row = {k: r.get(k) for k in KEEP_FIELDS if k in r}
        row.update({
            "record_index": i,
            "recovery_tier": tier,
            "recovered_response": label if tier in RECOVERED_TIERS else None,
            "recovery_evidence": evidence,
        })
        out_rows.append(row)

    n_total = len(records)
    n_parsed = sum(1 for r in records
                   if r.get("parse_status") == "parsed" and r.get("parsed_response") in ("A", "B"))
    n_dropped = n_total - n_parsed
    n_rec = len(recovered)

    print(f"\n{path}")
    print(f"  {n_total} responses | {n_parsed} parsed ({n_parsed/n_total:.1%}) | {n_dropped} dropped")
    if n_dropped:
        print(f"  recovered {n_rec}/{n_dropped} of the dropped ({n_rec/n_dropped:.1%})"
              f"  ->  parse rate {n_parsed/n_total:.1%} to {(n_parsed+n_rec)/n_total:.1%}")
        for tier, count in tiers.most_common():
            mark = "+" if tier in RECOVERED_TIERS else " "
            print(f"    {mark} {tier:<24}{count:>7} ({count/n_dropped:.1%})")

    if args.win_rate_delta and recovered:
        before = per_item_win_rates(records)
        after = per_item_win_rates(records, recovered)
        common = sorted(set(before) & set(after))
        if common:
            deltas = [abs(after[o] - before[o]) for o in common]
            worst = max(common, key=lambda o: abs(after[o] - before[o]))
            print(f"  win-rate impact over {len(common)} items:"
                  f" mean |delta| {sum(deltas)/len(deltas):.4f},"
                  f" max {max(deltas):.4f} (item {worst}:"
                  f" {before[worst]:.3f} -> {after[worst]:.3f})")

    if not args.dry_run and out_rows:
        out_path = path.with_name("recovered_" + path.name)
        with out_path.open("w") as f:
            for row in out_rows:
                f.write(json.dumps(row) + "\n")
        print(f"  wrote {out_path}")

    if args.judge_export and not args.dry_run:
        pending = [r for r in out_rows
                   if r["recovery_tier"] in ("semantic_skipped", "semantic_below_margin",
                                             "both_labels_skipped", "ambiguous")]
        if pending:
            jp = path.with_name("judge_queue_" + path.name)
            with jp.open("w") as f:
                for row in pending:
                    src = records[row["record_index"]]
                    f.write(json.dumps({
                        "record_index": row["record_index"],
                        "recovery_tier": row["recovery_tier"],
                        "presented_first_description": src.get("presented_first_description"),
                        "presented_second_description": src.get("presented_second_description"),
                        "raw_response": src.get("raw_response"),
                    }) + "\n")
            print(f"  wrote {jp} ({len(pending)} rows needing a judge)")

    return {"total": n_total, "parsed": n_parsed, "recovered": n_rec, "tiers": tiers}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="raw_responses_*.jsonl files")
    p.add_argument("--enable-endorse", action="store_true",
                   help="recover prose that names both labels when the final sentence commits to one")
    p.add_argument("--enable-semantic", action="store_true",
                   help="recover prose naming neither label by lexical match against the option descriptions")
    p.add_argument("--semantic-margin", type=int, default=3,
                   help="minimum distinguishing-word margin for the semantic tier (default 3)")
    p.add_argument("--win-rate-delta", action="store_true",
                   help="report how far per-item win rates move once recovered votes are counted")
    p.add_argument("--judge-export", action="store_true",
                   help="also write the rows this cannot resolve, for an LLM judge pass")
    p.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = p.parse_args()

    files = [f for f in args.paths if f.is_file()]
    if not files:
        sys.exit("no input files matched")

    agg = Counter()
    for f in sorted(files):
        s = process(f, args)
        agg["total"] += s["total"]
        agg["parsed"] += s["parsed"]
        agg["recovered"] += s["recovered"]

    if len(files) > 1:
        t, pa, rec = agg["total"], agg["parsed"], agg["recovered"]
        print(f"\n=== {len(files)} files: parse rate {pa/t:.1%} -> {(pa+rec)/t:.1%} "
              f"({rec} recovered of {t-pa} dropped) ===")


if __name__ == "__main__":
    main()
