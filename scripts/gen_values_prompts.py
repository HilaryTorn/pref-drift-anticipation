#!/usr/bin/env python3
"""Generate the ~250 shared user prompts for the Phase 2 values-direct SFT corpus.

One prompt, four arm responses. This script writes ONLY the prompts — the value-pole responses are generated separately, per arm, and every arm sees a byte-identical user turn (the same invariant scripts/build_big5_datasets.py enforces by filtering, obtained here by construction).

Why authored rather than mined. WildChat was the planned source; the pull is implemented in scripts/build_values_prompt_candidates.py and was run over all 14 shards' worth of structure on shard 0. The corpus-wide ceiling for first-person / advice-framed / person-involving / non-technical opening turns is ~1,200, and a random sample of 18 from that ceiling contained zero first-person dilemmas — it is meme copypasta, fanfic narration, and task requests. The density the plan assumed (1 in 6-20 advice-seeking) holds for advice-SHAPED text and not for the personal dilemma with a value fork that this corpus needs. See the stage-1 manifest for the funnel. Bhatia et al. (arXiv:2510.26707) locates the transfer in assistant turns taking contentful positions at scale, not in prompt provenance, so authoring the prompts costs naturalness and not the mechanism.

The grid: DOMAINS x TENSIONS, one API call per cell, PROMPTS_PER_CELL prompts returned per call.

The tension axis is what makes a value fork available without ever naming a value. Each tension is a STRUCTURAL description of competing courses of action ("a standing arrangement is up for renewal", "something that advances your own plans collides with something owed to someone else") and carries no Schwartz vocabulary, no value label, and no indication of which way to resolve. The four arms then diverge on the same situation. A prompt generated FROM a value definition would leak the instrument; a prompt generated from a tension structure cannot, because the structure is shared by all four arms.

Domain separation from the instrument. The value portraits scored by the battery use a work / friendship / online triad (scripts/gen_value_portraits.py). This corpus deliberately uses family, money, everyday health routine, neighbours, and school, none of which are in that triad. Friendship and romantic adjacency are watched rather than banned: household and neighbour situations involve other people, but the generator is told not to center a friendship or a romance, and the shared BANNED_DOMAIN_MATCHERS screen (imported from build_values_prompt_candidates, so the puller and the generator cannot disagree about what is out of domain) fails any prompt that does.

Safety scope. Low-stakes ordinary dilemmas only: scheduling, obligations, minor conflicts, spending, plans. Abuse, self-harm, medical decisions, legal jeopardy, financial distress, and vulnerable minors are excluded. The screen is imported from build_values_prompt_candidates.SAFETY_MATCHERS rather than restated, so the WildChat pull and the authored corpus cannot drift apart on what "in scope" means. "Health" here is routine and habit (sleep, exercise, screen time, diet) and never a medical decision; the medical matcher enforces that.

Accept gates, all hard, all triggering a re-roll of the cell (same shape as gen_value_portraits.py):
  1. Value leak: no Schwartz pole label and no marker word from data/values_generation/value_gists.json.
  2. Instrument echo: no overlap with a live elicitation stimulus (validate_sft_data.elicitation_snippets()).
  3. Safety scope: no SAFETY_MATCHERS hit.
  4. Shape: first person, asks what to do (ADVICE_MATCHERS), inside the length band, no second-person portrait form.
  5. Banned domain: not centered on work, friendship, romance, or online life (shared screen).
  6. Diversity: SequenceMatcher ratio below MAX_SIMILARITY against every prompt already accepted, across all cells.

Usage:
    python scripts/gen_values_prompts.py --dry-run              # render prompts + cost estimate, no API calls
    python scripts/gen_values_prompts.py --cells 2              # smoke: first 2 cells only
    python scripts/gen_values_prompts.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts",):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from build_values_prompt_candidates import (  # noqa: E402
    ADVICE_MATCHERS,
    BANNED_DOMAIN_MATCHERS,
    FIRST_PERSON_RE,
    SAFETY_MATCHERS,
    hits,
    normalize,
)

GISTS_PATH = ROOT / "data" / "values_generation" / "value_gists.json"
OUT_PATH = ROOT / "data" / "values_prompts" / "prompts.json"
PROVENANCE_PATH = ROOT / "data" / "values_prompts" / "prompts_provenance.json"

SCHEMA = "values_sft_prompt_v1"

# Ordinary-life domains, disjoint from the portraits' work/friendship/online triad. The gloss is handed to the generator verbatim.
DOMAINS = {
    "family": "family and household life — relatives, holidays, chores, visits, shared possessions, who does what at home",
    "money": "ordinary personal spending and saving — a purchase, a shared bill, a gift, a subscription, a budget, lending a small sum",
    "health_routine": "everyday health habits and routine — sleep, exercise, food, screen time, a standing class or gym plan. NEVER a medical decision, symptom, diagnosis, medication, or treatment",
    "neighbours": "neighbours and the immediate local community — shared spaces, noise, borrowing, a residents' group, a local custom or event",
    "school": "school and study life — coursework, a club, a course choice, a group project, a teacher or classmate expectation, a schedule clash",
}

# Structural tensions. Deliberately value-neutral: each names a fork, not a resolution, and none uses Schwartz vocabulary. These are what guarantee all four arms have somewhere to go.
TENSIONS = {
    "expectation_vs_own_plan": "someone expects the writer to do what they have always done, and the writer would rather handle it a different way this time",
    "continuity_vs_change": "a long-standing arrangement is up for renewal and the writer is deciding whether to keep it going or let it change",
    "cost_vs_commitment": "an obligation the writer has taken on is consuming time or money they had earmarked for something else",
    "group_norm_vs_own_judgment": "everyone involved is doing it one way and the writer privately thinks a different way is better",
    "own_progress_vs_owed_to_others": "something that would advance what the writer is working toward collides with something they owe someone else",
    "defer_vs_act": "the writer can follow the established process and ask first, or simply decide it themselves and act",
}

PROMPTS_PER_CELL = 9  # 5 domains x 6 tensions x 9 = 270, trimmed to the ~250 target after review

# Length band in characters. Long enough to carry a situation an arm can bite on, short enough to read as something a person typed.
MIN_CHARS, MAX_CHARS = 220, 700
MIN_FIRST_PERSON_HITS = 4

# Cross-cell near-duplicate cap. The same tension in the same domain will produce similar prompts unless this binds; ratio is on the normalized prompt.
MAX_SIMILARITY = 0.55

# Generous because thinking is on by default on Opus 5 and max_tokens caps thinking PLUS response text: a budget sized to the visible JSON alone truncates the cell mid-prompt.
MAX_TOKENS = 8000

VALUE_LABELS = ["self-direction", "conformity", "tradition", "achievement", "security", "stimulation", "hedonism", "benevolence", "power", "universalism", "schwartz"]

# Value-adjacent vocabulary that would leak the construct even without the label. Extended at load time with the marker words in value_gists.json.
VALUE_WORDS = [
    "autonomy", "autonomous", "independence", "independent", "conform", "conformity", "obedience", "obedient",
    "tradition", "traditional", "custom", "heritage", "ancestral", "ambition", "ambitious", "achievement",
    "success", "successful", "self-reliant", "self-reliance", "individualism", "collectivism", "values",
    "value system", "principles", "moral", "morals", "ethics", "ethical", "virtue", "duty", "obligation to conform",
]


SECOND_PERSON_RE = re.compile(r"^\s*you\b", re.IGNORECASE)

SYSTEM_MSG = (
    "You write realistic first-person messages that ordinary people send to an assistant when they are stuck on a decision. "
    "You follow constraints exactly, never editorialize, and return only valid JSON."
)


def load_anthropic_key() -> None:
    """Same key path as scripts/gen_value_portraits.py — the project's own paid key, not the Azure credit (this run is well under the $10 threshold in docs/cost.md)."""
    path = ROOT / "api_keys" / "api_key_anthropic.txt"
    if not path.exists():
        sys.exit(f"No key file at {path} (see api_keys/README.md).")
    key = path.read_text().strip()
    if not key:
        sys.exit(f"{path} is empty. Add your Anthropic API key.")
    os.environ["ANTHROPIC_API_KEY"] = key


def load_value_words() -> list[str]:
    """Banned vocabulary = the hardcoded list above plus every marker word Schwartz's gists use, so the screen tracks the actual instrument rather than a guess about it."""
    gists = json.loads(GISTS_PATH.read_text())
    words = set(VALUE_LABELS) | {w.lower() for w in VALUE_WORDS}
    for pole, gist in gists.items():
        if pole.startswith("_"):
            continue
        words.add(pole.lower())
        for marker in re.findall(r"[\(\[]([^\)\]]*)[\)\]]", gist):
            for token in marker.split(","):
                token = token.strip().lower()
                if len(token) >= 5 and " " not in token:
                    words.add(token)
    return sorted(words)


def build_cell_prompt(domain: str, domain_gloss: str, tension: str, tension_gloss: str, n: int) -> str:
    return f"""Write {n} DIFFERENT first-person messages that someone might send to an assistant asking for advice.

Every message must:
- Be written in the first person, about the writer's own situation, in their own plain voice. Not a hypothetical, not a survey item, not a story about someone else.
- Describe an ordinary, low-stakes situation in this area: {domain_gloss}.
- Turn on this tension: {tension_gloss}.
- End by asking what to do. Phrasings vary: "What should I do?", "I don't know whether to...", "Am I being unreasonable?", "Help me decide."
- Be {MIN_CHARS}-{MAX_CHARS} characters, concrete, with enough specifics (who, what, when) that a reply has something to bite on.

Critical constraints:
- The message must leave a genuine fork. A reasonable adviser could recommend clearly different courses of action, and the message itself must not signal which is right. Do NOT have the writer state what they have already decided, and do NOT load the situation so one answer is obviously correct.
- Never name or characterize a value, principle, trait, or personality. No words like values, principles, tradition, independence, ambition, conformity, duty, moral. Describe only the situation and the competing pulls in it.
- Keep it low-stakes and ordinary. No abuse, no self-harm, no medical decisions, no legal trouble, no financial hardship, no situation turning on a child at risk. If the tension seems to require any of those, make it smaller and more everyday instead.
- Do not center the message on the writer's job or workplace, on a friendship or a romantic relationship, or on social media.
- Vary the writers: different ages, household shapes, and registers (some brisk, some rambling, some apologetic). Vary the specifics heavily between the {n} messages — different people, different objects, different timing.

Return ONLY valid JSON: {{"prompts": ["...", "..."]}} with exactly {n} strings."""


def gate(text: str, accepted: list[str], value_words: list[str], snippets: list[str]) -> list[str]:
    """Every accept gate for one prompt. Returns the list of problems; empty means clean."""
    problems = []
    lowered = f" {normalize(text)} "

    leaks = [w for w in value_words if re.search(rf"\b{re.escape(w)}\b", lowered)]
    if leaks:
        problems.append(f"value leak: {leaks[:3]}")
    echoes = [s for s in snippets if s in lowered]
    if echoes:
        problems.append(f"elicitation echo: {echoes[0][:40]!r}")
    safety = hits(SAFETY_MATCHERS, text)
    if safety:
        problems.append(f"safety scope: {safety}")
    if not hits(ADVICE_MATCHERS, text):
        problems.append("no advice framing")
    if len(FIRST_PERSON_RE.findall(text)) < MIN_FIRST_PERSON_HITS:
        problems.append("not first person enough")
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        problems.append(f"length {len(text)} outside [{MIN_CHARS}, {MAX_CHARS}]")
    if SECOND_PERSON_RE.match(text):
        problems.append("second-person portrait form")
    banned = hits(BANNED_DOMAIN_MATCHERS, text)
    if banned:
        problems.append(f"banned domain: {banned}")
    norm = normalize(text)
    for other in accepted:
        ratio = SequenceMatcher(None, norm, other).ratio()
        if ratio >= MAX_SIMILARITY:
            problems.append(f"near-duplicate (ratio {ratio:.2f})")
            break
    return problems


def call_cell(model, temperature, domain, tension, n, accepted, value_words, snippets, retries=3):
    """Returns (kept, raw, rejected). Unlike gen_value_portraits.py's all-or-nothing cell gate, prompts are independent, so a partial cell is kept and only the shortfall is re-rolled — a single bad prompt should not discard eight good ones."""
    import litellm

    # Opus 4.7 and later reject temperature/top_p/top_k with a 400; drop_params strips params the target model does not accept, so --temperature still works on models that do.
    litellm.drop_params = True
    prompt = build_cell_prompt(domain, DOMAINS[domain], tension, TENSIONS[tension], n)
    kept, rejected, raws = [], [], []
    for attempt in range(retries + 1):
        want = n - len(kept)
        if want <= 0:
            break
        resp = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": prompt if attempt == 0 else prompt.replace(f"exactly {n} strings", f"exactly {want} strings").replace(f"Write {n} DIFFERENT", f"Write {want} DIFFERENT")}],
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raws.append(raw)
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1].lstrip("json").strip()
        try:
            candidates = json.loads(raw)["prompts"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"    retry {attempt + 1}/{retries} (unparseable: {type(exc).__name__})", file=sys.stderr)
            continue
        for text in candidates:
            text = (text or "").strip()
            problems = gate(text, accepted + [normalize(k) for k in kept], value_words, snippets)
            if problems:
                rejected.append({"prompt": text, "problems": problems})
            else:
                kept.append(text)
            if len(kept) >= n:
                break
        if len(kept) < n:
            print(f"    retry {attempt + 1}/{retries} ({len(kept)}/{n} kept)", file=sys.stderr)
    return kept, raws, rejected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="anthropic/claude-opus-5", help="LiteLLM model id for the generator (never the scored model)")
    ap.add_argument("--temperature", type=float, default=1.0, help="dropped for models that reject sampling params (Opus 4.7+); litellm.drop_params handles this")
    ap.add_argument("--per_cell", type=int, default=PROMPTS_PER_CELL)
    ap.add_argument("--cells", type=int, default=None, help="generate only the first N grid cells (smoke runs)")
    ap.add_argument("--dry-run", action="store_true", help="render one cell prompt, report the grid and gates, make no API calls")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    if not args.dry_run:
        load_anthropic_key()
    value_words = load_value_words()
    try:
        from validate_sft_data import elicitation_snippets

        snippets = elicitation_snippets()
    except Exception as exc:  # noqa: BLE001
        snippets, snippet_error = [], repr(exc)
    else:
        snippet_error = None

    grid = [(d, t) for d in DOMAINS for t in TENSIONS]
    if args.cells:
        grid = grid[: args.cells]

    if args.dry_run:
        d, t = grid[0]
        print(build_cell_prompt(d, DOMAINS[d], t, TENSIONS[t], args.per_cell))
        print(f"\n--- grid: {len(grid)} cells x {args.per_cell} = {len(grid) * args.per_cell} prompts")
        print(f"--- value-leak vocabulary: {len(value_words)} terms, e.g. {value_words[:8]}")
        print(f"--- elicitation snippets loaded: {len(snippets)}{' (ERROR: ' + snippet_error + ')' if snippet_error else ''}")
        return

    accepted_norm: list[str] = []
    records, provenance = [], []
    for i, (domain, tension) in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] {domain} x {tension}", file=sys.stderr)
        kept, raws, rejected = call_cell(args.model, args.temperature, domain, tension, args.per_cell, accepted_norm, value_words, snippets)
        for j, text in enumerate(kept):
            accepted_norm.append(normalize(text))
            records.append({
                "schema": SCHEMA,
                "prompt_id": f"{domain}.{tension}.{j:02d}",
                "domain": domain,
                "tension": tension,
                "prompt": text,
                "chars": len(text),
            })
        provenance.append({
            "domain": domain, "tension": tension, "model": args.model, "temperature": args.temperature,
            "requested": args.per_cell, "kept": len(kept), "rejected": rejected, "raw_responses": raws,
        })
        print(f"    kept {len(kept)}/{args.per_cell}, rejected {len(rejected)}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "schema": SCHEMA,
        "generator_model": args.model,
        "temperature": args.temperature,
        "grid": {"domains": DOMAINS, "tensions": TENSIONS, "per_cell": args.per_cell},
        "gates": {
            "min_chars": MIN_CHARS, "max_chars": MAX_CHARS, "min_first_person_hits": MIN_FIRST_PERSON_HITS,
            "max_similarity": MAX_SIMILARITY, "value_leak_terms": len(value_words),
            "elicitation_snippets": len(snippets), "safety_matchers": sorted(SAFETY_MATCHERS),
        },
        "count": len(records),
        "prompts": records,
    }, indent=2, ensure_ascii=False) + "\n")
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    print(f"\n{len(records)} prompts -> {out_path}")
    print(f"provenance -> {PROVENANCE_PATH}")


if __name__ == "__main__":
    main()
