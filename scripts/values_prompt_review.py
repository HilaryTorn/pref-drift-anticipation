#!/usr/bin/env python3
"""Round-trip the stage-1 candidate prompts through a human review pass.

Two subcommands, one file format. `export` writes a Markdown review document; you edit it by hand; `apply` reads it back and rebuilds the candidate set from what survived.

    python scripts/values_prompt_review.py export              # candidates_stage1.jsonl -> review.md
    #   ... edit review.md: delete rejected sections, fix wording in the ones you keep ...
    python scripts/values_prompt_review.py apply               # review.md -> candidates_reviewed.jsonl

Why a fenced block per prompt rather than bare prose. The document is meant to be edited, and an edited document has to parse back unambiguously. A ```text fence gives the prompt hard boundaries that survive rewrapping, blank lines, stray Markdown characters, and an editor that reflows paragraphs — none of which a heuristic "text between two headings" parser survives. The heading carries the candidate id, so deleting a whole section is a rejection and nothing else has to be recorded.

The Scruples title is shown for triage and is NOT part of the prompt. It is subreddit convention ("WIBTA for...") and including it in a training turn would stamp the AITA frame onto every example. `apply` asserts the title never leaked into the prompt body.

Hand edits are re-screened, not trusted. Every surviving prompt is re-run through the full stage-1 screen set — safety scope, banned domains, subreddit jargon, platform references, task requests, elicitation-stimulus echo, first-person density, length. An edit that reintroduces a violation is reported and the record is held back rather than written, because the screens are only a guarantee if they hold over the text that actually ships. Re-screening also catches the likelier accident: a section deleted halfway, leaving a truncated prompt behind.

Outputs from `apply`: <dir>/candidates_reviewed.jsonl (the kept records, with edits applied and an `edited` flag), and <dir>/review_manifest.json (counts of kept / rejected / edited / held-back, the rejected and held-back ids with reasons, and a per-record character delta so a large silent rewrite is visible).
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_values_prompt_candidates as B  # noqa: E402

DEFAULT_DIR = ROOT / "data" / "values_prompts"
MANIFEST_SCHEMA = "values_prompt_review_manifest_v1"

HEADING_RE = re.compile(r"^##\s+\d+\s+·\s+`([0-9a-f]+)`", re.MULTILINE)
FENCE_RE = re.compile(r"^```text\n(.*?)^```", re.MULTILINE | re.DOTALL)


BRIEF = """## What this is

Candidate **user turns** for a fine-tuning corpus. Each one is a real person describing an ordinary situation and asking what to do. They came from r/AmItheAsshole (via AI2's Scruples release) and have been filtered by script; you are the human pass.

Later, each of these gets four different assistant replies written for it — the same situation answered from four different value priorities. That is the experiment. **You do not need to judge that part**, and you do not need any background on the study.

## What to decide

Each candidate has a `**Verdict:**` line, pre-filled with `KEEP`. Change it only when something is wrong. Leave the prompt text alone unless you are fixing it (see below).

| Verdict | Use when |
|---|---|
| `KEEP` | Fine. An ordinary, low-stakes situation, and it reads like something a person actually sent. |
| `DROP: unsafe` | **Not low-stakes.** Anything where a confident recommendation could hurt someone — self-harm, abuse, a medical or mental-health decision, legal jeopardy, money trouble that is real hardship, or a situation turning on a child at risk. When in doubt, drop it. |
| `DROP: unnatural` | Doesn't work as a standalone message — incoherent, refers to context that isn't there, is a rant with no actual decision in it, or is so garbled it can't be answered. |
| `DROP: no fork` | There is only one sensible answer. A good candidate is one where reasonable people could genuinely advise different things. |
| `UNSURE` | Anything you'd want a second opinion on. These come back to the project lead — using it liberally is fine and costs nothing. |

**The `unsafe` pass is the important one.** Automated filters already ran and they leaked repeatedly — a roommate self-harming, a pet being put down, a fifteen-year-old — each caught only by someone reading it. That is the job here.

## Fixing rather than dropping

If a candidate is nearly fine, edit the text inside its ```` ```text ```` fence and leave the verdict as `KEEP`. Keep the fence markers. Every edit is automatically re-checked afterwards, so you cannot break anything silently.

## Notes

- The **Title** line is the original post title, shown to help you skim. It is not part of the prompt.
- Most prompts end with "Would I be wrong if...?" — that is generated from the title and is meant to be there.
- Reddit register (`[18F]`, "tl;dr", the confessional tone) is expected and is **not** a reason to drop.
- Target is roughly 250 keepers out of {total}. Being strict is the right instinct.
"""


def render(rows: list[dict], brief: bool = True, part: str = "") -> str:
    lines = [
        f"# Values-SFT candidate review{part}",
        "",
        f"{len(rows)} candidates. Built by `scripts/build_values_prompt_candidates.py`; provenance in `DECISIONS.md` (same directory).",
        "",
    ]
    if brief:
        lines += [BRIEF.replace("{total}", str(len(rows))), ""]
    lines += ["---", ""]
    for i, r in enumerate(rows, 1):
        flags = [f"ask={r['ask_source']}"] + (["scrubbed"] if r.get("jargon_scrubbed") else [])
        lines += [
            f"## {i:03d} · `{r['candidate_id']}` · {r['chars']}c · {' · '.join(flags)}",
            "",
            f"> **Title** (not part of the prompt): {r['meta'].get('title') or '—'}",
            "",
            "```text",
            r["prompt"],
            "```",
            "",
            "**Verdict:** KEEP",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def review_paths(directory: Path) -> list[Path]:
    """Every review document in the directory: the single review.md, or review_01.md ... from a split."""
    parts = sorted(directory.glob("review_[0-9][0-9].md"))
    single = directory / "review.md"
    return parts if parts else ([single] if single.exists() else [])


def export(args) -> None:
    directory = Path(args.dir)
    rows = [json.loads(line) for line in (directory / "candidates_stage1.jsonl").open()]
    # Shuffled deterministically by content hash, NOT sorted by ask_source or length. A reviewer
    # who runs out of time stops partway through, and with a sorted file that prefix is a biased
    # sample — all the short ones, or all the title-ask ones. Hash order makes any prefix
    # representative, so an unfinished pass is still usable, and it is reproducible without a seed.
    rows.sort(key=lambda r: r["candidate_id"])

    n = max(1, args.split)
    # Round-robin rather than contiguous blocks, so each reviewer sees the same mix of long and
    # short, title-ask and body-ask. Contiguous slices would hand one person every short prompt
    # and another every long one, and their keep-rates would not be comparable.
    chunks = [rows[i::n] for i in range(n)] if n > 1 else [rows]
    targets = [directory / (f"review_{i + 1:02d}.md" if n > 1 else "review.md") for i in range(n)]

    # review.md stops being build output the moment a human touches it: the deletions, edits and
    # verdicts in it exist nowhere else until `apply` runs. Overwriting an edited copy silently
    # destroys review work — which has happened once — so an export over a modified file refuses
    # unless forced.
    if not args.force:
        for target, chunk in zip(targets, chunks):
            part = f" — part {targets.index(target) + 1} of {n}" if n > 1 else ""
            if target.exists() and target.read_text() != render(chunk, part=part):
                raise SystemExit(
                    f"{target} has been modified since it was generated — exporting would discard those edits.\n"
                    f"  To keep them:    python scripts/values_prompt_review.py apply\n"
                    f"  To save a copy:  cp {target} {target.with_suffix('.md.bak')}\n"
                    f"  To overwrite:    re-run with --force"
                )
    if n > 1 and (directory / "review.md").exists():
        print(f"note: {directory / 'review.md'} still exists and is NOT part of a split review; delete it to avoid confusion.")

    for i, (target, chunk) in enumerate(zip(targets, chunks), 1):
        part = f" — part {i} of {n}" if n > 1 else ""
        target.write_text(render(chunk, part=part))
        print(f"{target}  —  {len(chunk)} candidates, {target.stat().st_size // 1024} KB")


def screen(text: str, snippets: list[str], min_chars: int, max_chars: int) -> list[str]:
    """Full stage-1 screen set over one prompt. Returns the violations; empty means clean."""
    problems = []
    if B.JARGON_RE.search(text):
        problems.append("subreddit jargon")
    if re.search(r"&[a-z]+;|&#x?\d", text, re.IGNORECASE):
        problems.append("html entity residue")
    if B.ZERO_WIDTH_RE.search(text):
        problems.append("zero-width characters")
    for label, matchers in (("safety", B.SAFETY_MATCHERS), ("task_request", B.TASK_REQUEST_MATCHERS), ("platform_reference", B.REDDIT_META_MATCHERS)):
        found = B.hits(matchers, text)
        if found:
            problems.append(f"{label}: {found}")
    domains = B.hits({k: B.BANNED_DOMAIN_MATCHERS[k] for k in B.DEFAULT_DOMAINS}, text)
    if domains:
        problems.append(f"banned_domain: {domains}")
    if any(s in B.normalize(text) for s in snippets):
        problems.append("elicitation stimulus echo")
    if len(B.FIRST_PERSON_RE.findall(text)) < B.MIN_FIRST_PERSON_HITS:
        problems.append("not first person enough")
    if not (min_chars <= len(text) <= max_chars):
        problems.append(f"length {len(text)} outside [{min_chars}, {max_chars}]")
    return problems


VERDICT_RE = re.compile(r"^\*\*Verdict:\*\*\s*(.+?)\s*$", re.MULTILINE)
SECTION_RE = re.compile(r"^##\s+\d+\s+·\s+`([0-9a-f]+)`", re.MULTILINE)


def parse_sections(text: str) -> list[tuple[str, str, str]]:
    """(candidate_id, prompt, verdict) per section. Splitting on the heading first means a broken fence is localised to its own section instead of swallowing the rest of the file."""
    parts = re.split(r"(?=^##\s+\d+\s+·\s+`[0-9a-f]+`)", text, flags=re.MULTILINE)
    out = []
    for part in parts:
        heading = SECTION_RE.match(part)
        if not heading:
            continue
        fence = FENCE_RE.search(part)
        verdict = VERDICT_RE.search(part)
        out.append((
            heading.group(1),
            fence.group(1).strip("\n").strip() if fence else "",
            (verdict.group(1).strip() if verdict else "KEEP"),
        ))
    return out


def apply_review(args) -> None:
    directory = Path(args.dir)
    original = {r["candidate_id"]: r for r in (json.loads(line) for line in (directory / "candidates_stage1.jsonl").open())}
    paths = review_paths(directory)
    if not paths:
        raise SystemExit(f"no review.md or review_NN.md in {directory} — run `export` first.")

    try:
        from validate_sft_data import elicitation_snippets

        snippets = elicitation_snippets()
    except Exception:  # noqa: BLE001
        snippets = []

    min_chars, max_chars = args.min_chars, B.SOURCES["scruples"]["max_chars"] + 400

    kept, held, edited, dropped, unsure = [], [], [], [], []
    seen: dict[str, str] = {}
    for path in paths:
        for candidate_id, prompt, verdict in parse_sections(path.read_text()):
            if candidate_id in seen:
                raise SystemExit(f"candidate {candidate_id} appears in both {seen[candidate_id]} and {path.name} — resolve the duplicate; nothing was written.")
            seen[candidate_id] = path.name
            record = original.get(candidate_id)
            if record is None:
                held.append({"candidate_id": candidate_id, "file": path.name, "problems": ["id not in candidates_stage1.jsonl"]})
                continue

            head = verdict.upper()
            if head.startswith("DROP"):
                dropped.append({"candidate_id": candidate_id, "reason": verdict, "file": path.name})
                continue
            if head.startswith("UNSURE"):
                unsure.append({"candidate_id": candidate_id, "note": verdict, "file": path.name, "prompt": prompt})
                continue
            if not head.startswith("KEEP"):
                held.append({"candidate_id": candidate_id, "file": path.name, "problems": [f"unrecognised verdict {verdict!r}; expected KEEP / DROP: <reason> / UNSURE"]})
                continue
            if not prompt:
                held.append({"candidate_id": candidate_id, "file": path.name, "problems": ["no ```text fence found in this section"]})
                continue

            title = record["meta"].get("title") or ""
            if title and B.normalize(title) in B.normalize(prompt):
                held.append({"candidate_id": candidate_id, "file": path.name, "problems": ["Scruples title leaked into the prompt body"]})
                continue

            problems = screen(prompt, snippets, min_chars, max_chars)
            if problems:
                held.append({"candidate_id": candidate_id, "file": path.name, "problems": problems, "chars": len(prompt)})
                continue

            was_edited = prompt != record["prompt"]
            if was_edited:
                edited.append({
                    "candidate_id": candidate_id,
                    "chars_before": record["chars"],
                    "chars_after": len(prompt),
                    "similarity": round(difflib.SequenceMatcher(None, record["prompt"], prompt).ratio(), 3),
                })
            kept.append({**record, "prompt": prompt, "chars": len(prompt), "edited": was_edited})

    deleted = [cid for cid in original if cid not in seen]

    out_path = directory / "candidates_reviewed.jsonl"
    with out_path.open("w") as fh:
        for record in kept:
            fh.write(json.dumps(record) + "\n")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "reviewed_from": "candidates_stage1.jsonl",
        "review_files": [p.name for p in paths],
        "counts": {
            "candidates_in": len(original),
            "sections_reviewed": len(seen),
            "kept": len(kept),
            "dropped_by_verdict": len(dropped),
            "deleted_from_file": len(deleted),
            "unsure": len(unsure),
            "held_back_by_rescreen": len(held),
            "edited": len(edited),
        },
        "rescreen": {
            "note": "every KEEP was re-run through the full stage-1 screen set; hand edits are not trusted",
            "domains_excluded": list(B.DEFAULT_DOMAINS),
            "min_chars": min_chars,
            "max_chars": max_chars,
        },
        "dropped": dropped,
        "unsure": unsure,
        "held_back": held,
        "edited": sorted(edited, key=lambda e: e["similarity"]),
        "deleted_ids": sorted(deleted),
    }
    (directory / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    c = manifest["counts"]
    print(f"kept {c['kept']}  ·  dropped {c['dropped_by_verdict']}  ·  deleted {c['deleted_from_file']}  ·  unsure {c['unsure']}  ·  edited {c['edited']}  ·  held back {c['held_back_by_rescreen']}")
    if unsure:
        print(f"\n{len(unsure)} marked UNSURE — these need your call (full text in review_manifest.json)")
    if held:
        print("\nheld back (edit reintroduced a violation, or section malformed):")
        for entry in held[:10]:
            print(f"  {entry['candidate_id']}  {entry['problems']}")
        if len(held) > 10:
            print(f"  ... and {len(held) - 10} more")
    print(f"\n-> {out_path}\n-> {directory / 'review_manifest.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="directory holding candidates_stage1.jsonl / review.md")
    sub = ap.add_subparsers(dest="command", required=True)
    ap_export = sub.add_parser("export", help="write review.md from candidates_stage1.jsonl")
    ap_export.add_argument("--split", type=int, default=1, help="split into N files (review_01.md ...) so several people can review in parallel without conflicting")
    ap_export.add_argument("--force", action="store_true", help="overwrite review.md even if it has been edited (destroys unapplied review work)")
    ap_apply = sub.add_parser("apply", help="rebuild the candidate set from an edited review.md")
    ap_apply.add_argument("--min_chars", type=int, default=150, help="length floor for reviewed prompts; relaxed from the mining floor since trimming a rambling post is a legitimate edit")
    args = ap.parse_args()
    (export if args.command == "export" else apply_review)(args)


if __name__ == "__main__":
    main()
