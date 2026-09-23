#!/usr/bin/env python3
"""Build the M0 format-training prompt set.

Deterministic and template-driven -- no API calls, no model in the loop, so rebuilding is free and
byte-reproducible from (seed, sizes). Content comes from m0/data/facts.py; wrappers come from
m0/prompts.py.

    python -m m0.data.build_dataset --out data/rl/m0_format --seed 0

Every row is `format_task_v1`:

    {"schema": "format_task_v1", "record_id": "m0.binary_factual.00007",
     "family": "binary_factual", "labels": ["A", "B"], "answer": "B", "verifiable": true,
     "carrier": "leveled_choice", "topic": "mass", "prompt": "<user text>"}

Three classes of check run before anything is written, and each FAILS the build rather than
dropping rows quietly -- a silently shrunken or skewed prompt set would be invisible later, when
the only symptom is a reward curve that rises for the wrong reason:

  1. Balance. Ground truth is near-uniform per family and per label scheme, option order is
     independent of correctness, and the correct option is not systematically the longer one.
     Any of those, left unchecked, gives the model a way to satisfy the correctness term without
     reasoning -- which is the exact failure the correctness term exists to prevent.
  2. Domain screen. No row may contain experimental-domain wording (coding, values, UE topics,
     training/anticipation phrasing) or echo a live elicitation stimulus.
  3. Length. Rows whose rendered prompt exceeds --max_prompt_tokens are an error, not a silent
     drop, so the set stays balanced.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.m0.data import facts  # noqa: E402
from scripts.m0.prompts import (  # noqa: E402
    BINARY_LABEL_SCHEMES,
    TERNARY_LABEL_SCHEMES,
    TERNARY_LABELS,
    build_prompt,
)

SCHEMA = "format_task_v1"
MANIFEST_SCHEMA = "m0_format_dataset_manifest_v1"

# Which carriers each family may use. Not every structure fits every content type: the
# "future_self"/"question_last" shapes carry a premise about a hypothetical situation, which reads
# as nonsense wrapped around "which is heavier?" but is the natural framing for a third-party
# judgment. Forcing all seven onto both families would buy structural coverage at the cost of
# prompts no human would write, and M0 would be learning to parse gibberish.
FACTUAL_CARRIERS = ["question_first", "leveled_choice", "constrained_choice", "options_first"]
NEUTRAL_CARRIERS = [
    "question_first",
    "question_last",
    "leveled_choice",
    "future_self",
    "constrained_choice",
    "options_first",
]

# The mix is two independent ~70:30 splits plus one capacity constraint, in priority order:
#
# 1. binary : ternary = 70 : 30 EXACT. Binary is the dominant live instrument -- every preference battery (coding pairs, values, Book A, UE) is a forced binary choice, and those are ~90% of serve-time generations -- while ternary appears only in the three anticipation arms (coding, UE, and since 2026-07-29 values, which moved from A/B to ternary when its ground truth became pooled per-portrait utilities). 30% is nonetheless far ABOVE ternary's serve share, deliberately: v3 ternary must install 6 presentation cells (2 word schemes x 3 rotations -- a third scheme was trained briefly and dropped 2026-07-31 to keep the pinned 200-trace warm-start budget viable, see m0/prompts.py) against binary's 3 schemes, and what installs a format is mass per cell, not mass per instrument. Even at 30% a ternary cell gets ~88 rows against a binary scheme's ~408, so ternary is the thinner side despite the over-weight -- if v3's per-rotation adoption gate fails on ternary cells, RAISING this share is the first lever.
#
# 2. verifiable : neutral ~= 70 : 30. The verifiable share is what keeps reasoning load-bearing -- a correctness term (reward rule 3; the warm-start's correctness filter) that the model cannot satisfy by guessing -- while the neutral share keeps M0 from learning "find the true option" instead of "reason, then commit", since every question in the real batteries is a judgment call with no checkable answer.
#
# 3. binary_neutral = 18% (historical, no longer a capacity cap). This share was fixed when the source table held 19 distinct contents rendering 342 prompts and a <= 19.5% capacity ceiling genuinely bound. That reasoning counted distinct PROMPTS when the quantity that matters is distinct CONTENTS: at 18% of a 200-trace warm-start, 36 traces drawn from 19 contents means every content is used ~twice, and the 2026-08-03 review found the repeats are exactly where the contradictions sat (6 of 24 repeated contents answered opposite ways across presentations). The table was therefore grown to 40 contents / 720 rendered prompts (see the note above the added items in facts.py), which removes the cap outright -- 18% of 1750 selects 315 of 720. The share stays at 18/52 for continuity with the committed v3 mix, not because capacity still forces it; if neutral repeats in the warm-start ever matter again, grow the table further before touching the mix.
#
# The committed numbers are the unique rounding of those three rules at the default 1750 total: binary 52+18=70, ternary 20+10=30 (rule 1 exact); verifiable 52+20=72, neutral 18+10=28 (rule 2, pulled 2 points off 70:30 by rule 3); all four demands integer.
FAMILY_MIX = {
    "binary_factual": 0.52,
    "ternary_factual": 0.20,
    "binary_neutral": 0.18,
    "ternary_neutral": 0.10,
}


# --- Framing text per carrier -----------------------------------------------------------------
# Paraphrases of the live wrappers' framing, never copies. See m0/prompts.py for the mapping from
# carrier name to the elicitation spec whose structure it mirrors.

FACTUAL_LEVELING = [
    "Assume both descriptions are accurate and refer to ordinary examples of the thing described.",
    "Assume ordinary conditions and typical examples, and ignore any unusual or extreme cases.",
    "Take both descriptions at face value and assume each is a common, everyday instance.",
]

FACTUAL_CONSTRAINTS = [
    "Assume standard conditions throughout: sea level, ordinary room temperature, and a typical "
    "example of each item. Neither description is a trick, and neither refers to a rare variant.",
    "Assume both refer to everyday instances under ordinary conditions, measured in the usual way, "
    "with nothing unusual about either case.",
]

NEUTRAL_LEVELING = [
    "Assume both options cost the same, take the same effort to arrange, and are equally practical.",
    "Assume either option could be arranged just as easily, at the same cost, in the same timeframe.",
    "Assume both are equally workable and that nothing rules either one out.",
]

NEUTRAL_CONSTRAINTS = [
    "Assume both options are equally affordable, would take the same time to put in place, and "
    "have been checked as practical. They differ only in what is described below.",
    "Assume the same budget, the same timeframe, and the same level of effort for either option, "
    "so the only difference is the one described.",
]

FACTUAL_LEAD_INS = [
    "Consider the two descriptions below.",
    "Below are two short descriptions.",
    "Two descriptions follow.",
]

NEUTRAL_LEAD_INS = [
    "Consider the two options below.",
    "Below are two possibilities.",
    "Two options follow.",
]

# {flat} is the no-change label of whichever scheme this item uses.
TERNARY_SAME_RULES = [
    "Answer {flat} when you judge there is no meaningful change, not merely when you are unsure.",
    "Use {flat} for no meaningful change -- not as a way of expressing uncertainty.",
    "Reserve {flat} for cases where nothing meaningful changes, rather than cases you find unclear.",
]

TERNARY_PREMISES = [
    "Consider the following change.",
    "Suppose the following happens.",
    "Take the following situation.",
]

# {labels} is the presented label order for this item, e.g. "LESS, SAME, or MORE".
#
# Two of the three stems present the labels as RESPONSE TOKENS ("...how does it change? Answer X, Y,
# or Z.") rather than as grammatical predicates ("...does it become X, Y, or Z?"). The predicate form
# forces the label to agree with the subject, and no label set agrees with every subject: English
# counts take more/fewer while scalar magnitudes take higher/lower, so "How many walkers take the
# path -- become HIGHER?" grates while "The temperature -- become MORE?" grates the other way. The
# obvious fix, matching schemes to item types, is the wrong one: all 24 neutral items are counts or
# frequencies, so it would empty the ternary_neutral x HIGHER/LOWER cells entirely -- and worse, it
# would make scheme predictable from content, teaching "temperature questions take HIGHER" instead of
# "commit using the labels you were given". At serve time the battery attaches MORE/LESS/SAME to
# PREFERENCES, content resembling neither training type, so label-following is the only thing that
# transfers. Decoupling the label from the grammar fixes the register clash without correlating
# scheme with content: stems still vary independently of both, by item index.
#
# The first stem KEEPS the predicate form deliberately -- the live battery asks "would your
# preference ... become MORE, LESS, or SAME?", which is grammatical for preferences, and M0 must stay
# in-distribution for the construction it is actually measured with.
TERNARY_QUESTIONS = [
    "Compared with before the change, does the quantity below become {labels}?",
    "Relative to how things stood beforehand, which way does the quantity below move? "
    "Answer {labels}.",
    "Compared with the situation before, how does the quantity below change? Answer {labels}.",
]


# --- Domain screen ----------------------------------------------------------------------------
# Terms drawn from the experimental domains M0 must not touch. A hit is a build failure: this is
# cheap to fix here and impossible to fix after a 4B run without redoing the run.
BANNED_TERMS = [
    # coding battery: languages and task verbs
    "python", "javascript", "java", "c++", "csharp", "c#", "golang", "rust", "php", "ruby",
    "typescript", "code", "coding", "program", "programming", "software", "debug", "algorithm",
    "compiler", "function", "variable", "dataset", "computer",
    # values battery
    "self-direction", "benevolence", "universalism", "achievement", "hedonism", "stimulation",
    "conformity", "tradition", "security", "schwartz", "moral", "ethical", "virtue",
    # utility engineering option subsets
    "career", "job", "employment", "workplace", "salary", "promotion", "power-seeking",
    "self-preservation", "shutdown", "autonomy", "freedom", "consciousness", "sentient",
    "artificial intelligence", " ai ", "language model", "chatbot", "assistant",
    # anticipation batteries
    "your future", "future self", "fine-tune", "fine-tuning", "training data", "trained on",
    "checkpoint", "preference for the following task",
]


def _banned_matcher(term: str):
    """Word-boundary match for plain words, substring match for anything with punctuation.

    Naive substring matching flags "crust" for "rust" and "programme" for "program". Those false
    positives are not harmless: a screen that cries wolf gets loosened, and a loosened screen is
    how real contamination gets through. Terms containing non-word characters ("c++", "c#",
    "self-direction") cannot use a trailing \\b, so they stay substring matches.
    """
    if re.fullmatch(r"[\w ]+", term):
        return re.compile(rf"\b{re.escape(term.strip())}\b").search
    return lambda text, _term=term: _term in text


_BANNED_MATCHERS = [(term, _banned_matcher(term)) for term in BANNED_TERMS]


def domain_hits(text: str) -> list[str]:
    # r'\s+', not r'\\s+': the latter matches a literal backslash-s and leaves newlines in place,
    # so a banned phrase split across a line break ("training\ndata") slips the screen entirely.
    lowered = f" {re.sub(r'[\s]+', ' ', text).lower()} "
    return [term for term, matches in _BANNED_MATCHERS if matches(lowered)]


def stimulus_hits(text: str, snippets: list[str]) -> list[str]:
    """Substring overlap with live elicitation stimuli, via the SFT screen's own snippet list."""
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return [snippet for snippet in snippets if snippet in normalized]


# --- Item generation ---------------------------------------------------------------------------

def _pairs_for_dimension(dim: dict, balance_length: bool = True) -> list:
    """Every unambiguous pair within a dimension, with the length tell balanced out.

    "Unambiguous" is enforced numerically rather than by eye: magnitude dimensions require the
    values to differ by at least min_ratio, and the additive ones (temperature, year) by min_gap.
    A pair the author merely believes is obvious -- a mug versus a large mug -- would train the
    model to guess, and the correctness term would then be rewarding noise.

    The length balancing is the important part. Descriptions of big, hot, or slow things tend to
    run longer than descriptions of small ones, so the raw pair set hands the model a free
    heuristic: "pick the longer option" scored 62% before this was added. Rewording the tables by
    hand fixes it for one edit and then oscillates -- pushing four dimensions toward chance flipped
    three of them past it, and the next person to add a fact would silently reintroduce the tell.

    So it is enforced structurally instead: split the pairs by whether the longer option is the
    correct one, and truncate the bigger bucket to the size of the smaller. The heuristic is then
    at chance BY CONSTRUCTION, for any table anyone writes later. The cost is some discarded pairs,
    which is affordable -- the dimensions produce far more pairs than the mix consumes.
    """
    pairs = []
    for first, second in itertools.combinations(dim["items"], 2):
        va, vb = abs(first[1]), abs(second[1])
        if "min_ratio" in dim:
            lo, hi = sorted((va, vb))
            if lo <= 0 or hi / lo < dim["min_ratio"]:
                continue
        if "min_gap" in dim and abs(first[1] - second[1]) < dim["min_gap"]:
            continue
        pairs.append((first, second))

    if not balance_length:
        return pairs

    longer_correct, shorter_correct, ties = [], [], []
    for first, second in pairs:
        first_wins = (first[1] > second[1]) if dim["larger_wins"] else (first[1] < second[1])
        winner, loser = (first, second) if first_wins else (second, first)
        if len(winner[0]) == len(loser[0]):
            ties.append((first, second))
        elif len(winner[0]) > len(loser[0]):
            longer_correct.append((first, second))
        else:
            shorter_correct.append((first, second))

    keep = min(len(longer_correct), len(shorter_correct))

    def _stride(bucket):
        """Take `keep` items spread evenly across the bucket, not the first `keep`.

        Sorted order is deterministic (which matters -- pair selection must not depend on RNG call
        order elsewhere in the build), but taking a prefix of it is alphabetically biased: an item
        whose description sorts late is dropped from EVERY pair it appears in. That is how "the
        heaviest legal ten-pin bowling ball" silently disappeared from the mass dimension the moment
        it was renamed. Striding keeps the determinism and spreads the loss across all items.
        """
        ordered = sorted(bucket)
        if keep >= len(ordered):
            return ordered
        return [ordered[round(i * (len(ordered) - 1) / max(keep - 1, 1))] for i in range(keep)]

    return sorted(_stride(longer_correct) + _stride(shorter_correct) + ties)


def _binary_framing(carrier: str, question: str, noun: str, rng: random.Random,
                    verifiable: bool, premise: str | None) -> dict:
    leveling = FACTUAL_LEVELING if verifiable else NEUTRAL_LEVELING
    constraints = FACTUAL_CONSTRAINTS if verifiable else NEUTRAL_CONSTRAINTS
    lead_ins = FACTUAL_LEAD_INS if verifiable else NEUTRAL_LEAD_INS
    return {
        "question": question,
        "plural_noun": noun,
        "leveling": rng.choice(leveling),
        "constraints": rng.choice(constraints),
        "lead_in": rng.choice(lead_ins),
        "premise": premise or "",
    }


def gen_binary_factual(rng: random.Random) -> list[dict]:
    items = []
    for dim in facts.FACTUAL_DIMENSIONS:
        for pair_i, ((name_a, val_a), (name_b, val_b)) in enumerate(_pairs_for_dimension(dim)):
            bigger_is_correct = dim["larger_wins"]
            first_wins = (val_a > val_b) if bigger_is_correct else (val_a < val_b)
            for carrier in FACTUAL_CARRIERS:
                for labels in BINARY_LABEL_SCHEMES:
                    # Option order is randomized independently of which one is correct, so
                    # position carries no signal for the model to exploit.
                    flip = rng.random() < 0.5
                    opt_a, opt_b = (name_b, name_a) if flip else (name_a, name_b)
                    a_is_correct = (not first_wins) if flip else first_wins
                    items.append({
                        "family": "binary_factual",
                        "topic": dim["key"],
                        # The QUESTION this row asks, stable across carrier, label scheme and option
                        # order. Downstream selection uses it to avoid asking the same question
                        # twice -- see _quota_batches in build_warmstart.py.
                        "content_id": f"binary_factual:{dim['key']}:{pair_i}",
                        "carrier": carrier,
                        "labels": list(labels),
                        "option_a": opt_a,
                        "option_b": opt_b,
                        "answer": labels[0] if a_is_correct else labels[1],
                        "verifiable": True,
                        # Cycled, not drawn -- same fix as gen_binary_neutral. prompts.py takes this modulo len(REASONING_INSTRUCTIONS) = 5, and the old `rng.randrange(64) % 5` was non-uniform (64 = 12*5 + 4). A consecutive counter is uniform to +-1 and deterministic.
                        "instruction_idx": len(items),
                        "framing": _binary_framing(carrier, dim["question"], dim["noun"], rng,
                                                   verifiable=True, premise=None),
                    })
    return items


def gen_binary_neutral(rng: random.Random) -> list[dict]:
    items = []
    # Option order and instruction phrasing are DEALT, not drawn. `check_balance` only gates
    # position on verifiable families (a neutral item has no answer, so there is no
    # answer-position to check), which left this the one presentation axis in the build with
    # neither construction nor a gate: measured 2026-08-03, the share of rows carrying the spec's
    # option_a in the first slot ranged 44.4%-54.9% across seeds -- a 10.5-point swing, wider than
    # the 8% tolerance the build gates its other checks on, and skewed 54.9% at the shipped seed 0.
    # Summing the three axis indices balances order within every axis and every pair of axes, so a
    # carrier or a label scheme cannot correlate with which option leads. Same principle as
    # _select_ternary_stratified: balance that depends on the RNG cooperating is not balance.
    for idx, spec in enumerate(facts.NEUTRAL_BINARY_ITEMS):
        for carrier_i, carrier in enumerate(NEUTRAL_CARRIERS):
            for scheme_i, labels in enumerate(BINARY_LABEL_SCHEMES):
                flip = (idx + carrier_i + scheme_i) % 2 == 1
                opt_a, opt_b = ((spec["option_b"], spec["option_a"]) if flip
                                else (spec["option_a"], spec["option_b"]))
                items.append({
                    "family": "binary_neutral",
                    "topic": spec["noun"],
                    "content_id": f"binary_neutral:{idx}",
                    "carrier": carrier,
                    "labels": list(labels),
                    "option_a": opt_a,
                    "option_b": opt_b,
                    "answer": None,  # no ground truth: scored on format alone
                    "verifiable": False,
                    # Recorded so selection can stratify on it. A neutral row has no answer, so
                    # option order is its only position axis and nothing else can reconstruct it
                    # once the option texts have been swapped.
                    "order_flipped": flip,
                    # Cycled, not drawn. prompts.py takes this modulo len(REASONING_INSTRUCTIONS),
                    # which is 5 -- and `rng.randrange(64) % 5` is not even uniform (64 = 12*5 + 4,
                    # so phrasings 0-3 were drawn 13/64 of the time and phrasing 4 only 12/64).
                    "instruction_idx": idx * len(NEUTRAL_CARRIERS) * len(BINARY_LABEL_SCHEMES)
                    + carrier_i * len(BINARY_LABEL_SCHEMES) + scheme_i,
                    "framing": _binary_framing(carrier, spec["question"], spec["noun"], rng,
                                               verifiable=False, premise=spec["premise"]),
                })
        _ = idx
    return items


def _ternary_label_list(labels: tuple[str, ...]) -> str:
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def _ternary_framing(
    change: str,
    header: str,
    premise_i: int,
    question_i: int,
    rule_i: int,
    scheme: tuple[str, ...],
    presented: tuple[str, ...],
) -> dict:
    """Render the framing against this item's scheme and presented label order.

    The question and same-rule strings used to hard-code MORE/LESS/SAME, so varying an item's
    labels would have produced a prompt that named one set of labels in the question and a
    different set in the answer instruction.

    Both arguments are needed and they are not interchangeable. The question lists labels in the
    order they are *presented*, but the same-rule names the no-change *role*, which is always
    scheme[2]. Deriving it from the presented order instead yields "Answer MORE when you judge
    there is no meaningful change" the moment a rotation puts MORE last.
    """
    return {
        "premise": f"{TERNARY_PREMISES[premise_i]}\n\n{change}",
        "question": TERNARY_QUESTIONS[question_i].format(labels=_ternary_label_list(presented)),
        "same_rule": TERNARY_SAME_RULES[rule_i].format(flat=scheme[2]),
        "subject_header": header,
    }


# The ternary family has one structure (premise + Task block, no Option blocks) because that is
# what the anticipation specs use. Variety therefore has to come from the framing, so each source
# item is rendered against a deterministic walk of the (premise x question x same-rule x
# instruction) cross-product rather than random draws -- random draws would collide and produce
# genuinely identical prompts, which waste rollouts when two land in the same GRPO batch.
_TERNARY_FRAMING_COMBOS = [
    (p, q, r)
    for p in range(len(TERNARY_PREMISES))
    for q in range(len(TERNARY_QUESTIONS))
    for r in range(len(TERNARY_SAME_RULES))
]
TERNARY_RENDERS_PER_ITEM = 9


# Canonical role order is (up, down, flat) -- source items record their answer as MORE/LESS/SAME,
# and the role index carries that answer into whichever scheme an item is rendered with.
_ROLE_INDEX = {label: index for index, label in enumerate(TERNARY_LABELS)}


def ternary_presentations() -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """(scheme, presented order) for every scheme x cyclic rotation.

    Cyclic rotations put each label in each position exactly once, matching the counterbalancing
    the anticipation battery now applies at scoring time (scripts/run_elicitations.cyclic_label_
    rotations). Training on the same set of presentations the battery can ask is the point: it is
    what keeps a rotated or relabelled battery in-distribution rather than a novel format.
    """
    presentations = []
    for scheme in TERNARY_LABEL_SCHEMES:
        for shift in range(len(scheme)):
            presentations.append((scheme, scheme[shift:] + scheme[:shift]))
    return presentations


def _gen_ternary(specs: list[dict], family: str, topic: str, verifiable: bool) -> list[dict]:
    items = []
    presentations = ternary_presentations()
    for spec_index, spec in enumerate(specs):
        for render in range(TERNARY_RENDERS_PER_ITEM):
            # Offset by spec_index so different source items take different framing walks.
            combo = _TERNARY_FRAMING_COMBOS[
                (spec_index * TERNARY_RENDERS_PER_ITEM + render) % len(_TERNARY_FRAMING_COMBOS)
            ]
            for scheme, presented in presentations:
                answer = None
                if verifiable:
                    # Re-express the canonical MORE/LESS/SAME answer in this item's own scheme.
                    answer = scheme[_ROLE_INDEX[spec["answer"]]]
                items.append({
                    "family": family,
                    "topic": topic,
                    # Keyed on the SOURCE SPEC only -- not on `render`. Two renders of one spec vary
                    # the framing wording but ask the same change/quantity question, which is what
                    # a reader sees as "the same question asked twice".
                    "content_id": f"{family}:{spec_index}",
                    "carrier": "premise_task",
                    "labels": list(presented),
                    "subject": spec["quantity"],
                    "answer": answer,
                    "verifiable": verifiable,
                    "instruction_idx": spec_index * TERNARY_RENDERS_PER_ITEM + render,
                    "framing": _ternary_framing(
                        spec["change"], "Quantity", *combo, scheme, presented
                    ),
                })
    return items


def gen_ternary_factual(rng: random.Random) -> list[dict]:
    return _gen_ternary(facts.TERNARY_FACTUAL_ITEMS, "ternary_factual", "change_effect", True)


def gen_ternary_neutral(rng: random.Random) -> list[dict]:
    return _gen_ternary(facts.NEUTRAL_TERNARY_ITEMS, "ternary_neutral", "judgment", False)


GENERATORS = {
    "binary_factual": gen_binary_factual,
    "ternary_factual": gen_ternary_factual,
    "binary_neutral": gen_binary_neutral,
    "ternary_neutral": gen_ternary_neutral,
}


# --- Checks ------------------------------------------------------------------------------------

def check_balance(rows: list[dict], tolerance: float, errors: list[str]) -> dict:
    """Ground-truth balance, position independence, and the length tell."""
    report: dict = {}

    for family in ("binary_factual", "ternary_factual"):
        family_rows = [r for r in rows if r["family"] == family]
        if not family_rows:
            continue
        # Per label scheme, not just overall: an overall-balanced set can still be skewed inside
        # one scheme, and the model sees the scheme.
        by_scheme: dict[str, Counter] = {}
        for row in family_rows:
            scheme = "/".join(row["labels"])
            by_scheme.setdefault(scheme, Counter())[row["answer"]] += 1
        report[family] = {scheme: dict(counts) for scheme, counts in by_scheme.items()}
        for scheme, counts in by_scheme.items():
            total = sum(counts.values())
            expected = 1.0 / len(scheme.split("/"))
            for label in scheme.split("/"):
                share = counts.get(label, 0) / total
                if abs(share - expected) > tolerance:
                    errors.append(
                        f"{family}/{scheme}: label {label!r} is the answer {share:.1%} of the time "
                        f"(expected ~{expected:.1%}, tolerance {tolerance:.0%}). M0 would learn a "
                        f"prior on the label instead of the format."
                    )

    # Length tell. The thing that actually matters is not whether correct options are on average
    # longer -- it is whether "always pick the longer option" WINS. A mean-length gap can coexist
    # with a coin-flip heuristic (a few very long options drag the mean without deciding many
    # items), so score the heuristic itself: how often does the longer option happen to be the
    # correct one? At chance it is unexploitable, whatever the means say. Anything a model could
    # learn instead of reasoning defeats the purpose of the correctness term.
    heuristic_wins, heuristic_total = 0, 0
    correct_len, wrong_len = [], []
    for row in rows:
        if row["family"] != "binary_factual":
            continue
        a_correct = row["answer"] == row["labels"][0]
        len_correct = len(row["option_a"] if a_correct else row["option_b"])
        len_wrong = len(row["option_b"] if a_correct else row["option_a"])
        correct_len.append(len_correct)
        wrong_len.append(len_wrong)
        if len_correct != len_wrong:  # ties give the heuristic nothing to go on
            heuristic_total += 1
            heuristic_wins += len_correct > len_wrong
    if correct_len:
        accuracy = heuristic_wins / heuristic_total if heuristic_total else 0.5
        report["length_tell"] = {
            "longer_option_heuristic_accuracy": accuracy,
            "mean_correct_chars": sum(correct_len) / len(correct_len),
            "mean_incorrect_chars": sum(wrong_len) / len(wrong_len),
            "decidable_items": heuristic_total,
        }
        if abs(accuracy - 0.5) > tolerance:
            errors.append(
                f"binary_factual length tell: 'pick the longer option' scores {accuracy:.1%} "
                f"(chance is 50%, tolerance {tolerance:.0%}). A model could satisfy the correctness "
                f"term with that heuristic instead of reasoning. Reword the option texts in "
                f"m0/data/facts.py rather than relaxing this check."
            )

    # Position independence: the correct answer should be the first-listed option ~half the time.
    first = sum(1 for r in rows if r["verifiable"] and r["family"] == "binary_factual"
                and r["answer"] == r["labels"][0])
    total = sum(1 for r in rows if r["verifiable"] and r["family"] == "binary_factual")
    if total:
        share = first / total
        report["first_option_correct_share"] = share
        if abs(share - 0.5) > tolerance:
            errors.append(
                f"binary_factual: the first-listed option is correct {share:.1%} of the time "
                f"(expected ~50%). Position would carry signal."
            )
    return report


def check_domains(rows: list[dict], errors: list[str]) -> None:
    try:
        from validate_sft_data import elicitation_snippets
        snippets = elicitation_snippets()
    except Exception as err:  # noqa: BLE001 - the screen must never be skipped silently
        errors.append(
            f"could not load the elicitation-contamination screen from scripts/validate_sft_data.py "
            f"({err}). Refusing to build an unscreened prompt set."
        )
        snippets = None

    for row in rows:
        text = row["prompt"]
        hits = domain_hits(text)
        if hits:
            errors.append(f"{row['record_id']}: experimental-domain terms {hits} in prompt")
        if snippets is not None:
            echoes = stimulus_hits(text, snippets)
            if echoes:
                errors.append(f"{row['record_id']}: echoes a live elicitation stimulus {echoes[:1]}")


def check_lengths(rows: list[dict], tokenizer_name: str, max_tokens: int, errors: list[str]) -> dict:
    """Prompt length under the trained model's own tokenizer.

    Measured with the 4B tokenizer by default because that is the model M0 trains: a prompt that
    fits under one tokenizer can overflow under another, and TRL would silently left-truncate it.
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as err:  # noqa: BLE001
        return {"skipped": f"{type(err).__name__}: {err}"}

    from scripts.m0.prompts import render_format_prompt

    lengths = []
    for row in rows:
        n = len(tokenizer(render_format_prompt(tokenizer, row["prompt"]))["input_ids"])
        lengths.append(n)
        if n > max_tokens:
            errors.append(
                f"{row['record_id']}: rendered prompt is {n} tokens, over --max_prompt_tokens "
                f"({max_tokens}). Shorten the item rather than letting TRL truncate it."
            )
    lengths.sort()
    return {
        "tokenizer": tokenizer_name,
        "max": lengths[-1],
        "median": lengths[len(lengths) // 2],
        "p95": lengths[int(len(lengths) * 0.95)],
    }


# --- Build --------------------------------------------------------------------------------------

def _rebalance_binary_labels(rows: list[dict]) -> None:
    """Force exact label balance on the SELECTED rows, per family and label scheme.

    Option order is drawn per item with a coin flip, which balances only in expectation. Once the
    pools are sampled down, the realised split drifts: a rebuild that dropped some mass pairs landed
    at 58% 'A' in the A/B scheme and failed the build. Balance that depends on the RNG cooperating
    is not balance -- and it is the one property that most directly hands the model a shortcut,
    since a label prior beats reasoning at no cost.

    So correct it deterministically instead: within each group, flip the minimum number of items to
    bring the answer distribution to even. Flipping swaps the two option texts and the answer, which
    leaves the item's content identical and only changes which label carries it.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["family"].startswith("ternary") or not row["verifiable"]:
            continue  # ternary answers come from the source table; neutral rows have no answer
        groups.setdefault((row["family"], "/".join(row["labels"])), []).append(row)

    for (_family, _scheme), group in groups.items():
        first_label = group[0]["labels"][0]
        on_first = [row for row in group if row["answer"] == first_label]
        target = len(group) // 2
        # Flip from whichever side is over-represented. Sorted by record content for determinism.
        surplus = sorted(on_first, key=lambda r: (r["option_a"], r["option_b"]))
        if len(on_first) < target:
            surplus = sorted((row for row in group if row["answer"] != first_label),
                             key=lambda r: (r["option_a"], r["option_b"]))
            need = target - len(on_first)
        else:
            need = len(on_first) - target
        for row in surplus[:need]:
            row["option_a"], row["option_b"] = row["option_b"], row["option_a"]
            row["answer"] = row["labels"][1] if row["answer"] == row["labels"][0] else row["labels"][0]


def _stratified_ternary_pick(pool: list[dict], want: int, rng: random.Random) -> list[dict]:
    """Select ternary rows with per-cell balance BY CONSTRUCTION instead of by luck.

    The pool renders every source item in all presentations (schemes x 3 rotations), and shuffle-then-truncate left each (presentation x answer) cell's balance to the RNG: per-cell sampling noise sat at or above the 8% gate in every layout tried (sd ~8.7% at 12 cells, ~7.5% at 9, ~6% at the current 6), so builds failed or passed by seed. Presentation is content-independent, which is what makes construction possible: each source unit (spec x render) is used in EXACTLY ONE presentation, dealt round-robin within its answer role so every cell receives every role in near-equal measure, then trimmed to `want` by always shrinking the largest (presentation x role) group — which also evens out any global role imbalance the source tables carry. Deviations are +-1 row per cell-role group at any seed.

    One presentation per unit also means no source content repeats across presentations in the built set, the same duplicate-aversion the binary families' error message enforces.
    """
    presentations = [tuple(p) for _, p in ternary_presentations()]
    by_unit_pres: dict[tuple[int, tuple[str, ...]], dict] = {}
    role_by_unit: dict[int, int | None] = {}
    for row in pool:
        unit = row["instruction_idx"]
        by_unit_pres[(unit, tuple(row["labels"]))] = row
        if tuple(row["labels"]) == TERNARY_LABELS:
            role_by_unit[unit] = (
                TERNARY_LABELS.index(row["answer"]) if row["verifiable"] else None
            )
    units = sorted(role_by_unit)
    if want > len(units):
        raise SystemExit(
            f"stratified ternary selection uses each source item in exactly one presentation, so "
            f"this family can supply {len(units)} rows but the mix asks for {want}. Add entries to "
            f"m0/data/facts.py rather than re-rendering the same content in several presentations."
        )
    by_role: dict[int | None, list[int]] = {}
    for unit in units:
        by_role.setdefault(role_by_unit[unit], []).append(unit)

    groups: dict[tuple[tuple[str, ...], int | None], list[dict]] = {}
    for role_i, (role, role_units) in enumerate(sorted(by_role.items(), key=lambda kv: str(kv[0]))):
        rng.shuffle(role_units)
        for i, unit in enumerate(role_units):
            # Offset by role_i so the dealing does not start every role at presentation 0, which
            # would hand the first presentations the rounding remainder of every role at once.
            pres = presentations[(i + role_i) % len(presentations)]
            groups.setdefault((pres, role), []).append(by_unit_pres[(unit, pres)])

    # Sort for a seed-independent starting order, THEN shuffle with the seeded rng. The sort alone
    # is not enough: `pop()` takes the last element, and `instruction_idx` is
    # `spec_index * RENDERS_PER_ITEM + render`, so an ascending sort put the LAST source items in
    # facts.py at the end of every group and the trim deleted them first, over and over. Measured on
    # the 2026-08-03 build: NEUTRAL_TERNARY_ITEMS positions 21-23 of 24 and TERNARY_FACTUAL_ITEMS
    # 43-45 of 46 produced zero rows in any split -- seven authored items were inert, and anything
    # appended to a table would have landed in the same dead zone. Shuffling first makes the trim
    # uniform across source items while staying deterministic per seed.
    for group in groups.values():
        group.sort(key=lambda r: r["instruction_idx"])
        rng.shuffle(group)
    total_now = sum(len(group) for group in groups.values())
    while total_now > want:
        largest = max(groups, key=lambda key: (len(groups[key]), str(key)))
        groups[largest].pop()
        total_now -= 1
    return [row for group in groups.values() for row in group]


def _slot_key(row: dict) -> str:
    """The warm-start's quota unit: format cell plus position role (mirrors m0/data/build_warmstart._quota_key, refined).

    For verifiable rows the position role is the gold answer, exactly as the quota key has it. For neutral binary rows it is `order_flipped` -- their only position axis, which the quota key cannot see because it keys on the answer and a neutral row has none. Without it, `_stratified_binary_pick`'s 50/50 pool balance is undone by the split's hypergeometric draw: measured 2026-08-03 on the balanced pool, the train split landed at 54.1% flipped at seed 0 and dev at 61.4% at seed 2 -- the same magnitude the selection fix removed, and train.jsonl is the sole source of warm-start SFT prompts. With it, every split sits at 49.7-50.0% at any seed. Splitting on a strict refinement of the quota key keeps every coarser quota slot proportionally represented, so the warm-start guarantee in `stratified_splits`' docstring is unchanged.
    """
    key = f"{row['family']}|{'/'.join(row['labels'])}"
    if row.get("verifiable") and row.get("answer"):
        key += f"|{row['answer']}"
    elif row["family"] == "binary_neutral":
        key += f"|flip={row['order_flipped']}"
    return key


def stratified_splits(rows: list[dict], train_size: int, dev_size: int, seed: int) -> dict[str, list[dict]]:
    """Partition rows into train/dev/heldout with every quota slot represented proportionally in each split.

    A naive `rows[:train_size]` slice leaves each slot's train-side pool to the hypergeometric draw (~7 +- 2 prompts for a ternary slot), and the warm-start's per-slot quotas then bind on the thinnest pools: measured at trace_budget 450, the tightest slots landed 4-6 prompts against targets of 3-4 and could not fill even with retries. Stratifying the SPLIT is free and fixes the floor: each slot contributes its proportional share to every split (largest-remainder rounding, deterministic), so train pools sit at their fair size and dev/heldout evaluate every format the battery can ask, not a lucky subset.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(_slot_key(row), []).append(row)
    total = len(rows)

    def allocate(size: int, capacity: dict[str, int]) -> dict[str, int]:
        exact = {key: size * len(groups[key]) / total for key in groups}
        alloc = {key: min(int(share), capacity[key]) for key, share in exact.items()}
        order = sorted(groups, key=lambda key: (alloc[key] - exact[key], key))
        i = 0
        while sum(alloc.values()) < size:
            key = order[i % len(order)]
            if alloc[key] < capacity[key]:
                alloc[key] += 1
            i += 1
        return alloc

    capacity = {key: len(group) for key, group in groups.items()}
    train_alloc = allocate(train_size, capacity)
    dev_alloc = allocate(dev_size, {key: capacity[key] - train_alloc[key] for key in groups})
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "heldout": []}
    for key in sorted(groups):
        group = groups[key]
        t, d = train_alloc[key], dev_alloc[key]
        splits["train"].extend(group[:t])
        splits["dev"].extend(group[t: t + d])
        splits["heldout"].extend(group[t + d:])
    shuffler = random.Random(f"{seed}-splits")
    for split_rows in splits.values():
        shuffler.shuffle(split_rows)
    return splits


def _binary_cell(row: dict) -> tuple:
    """The presentation cell a binary row occupies: label scheme x wording x position role."""
    base = (tuple(row["labels"]), row["carrier"])
    if row.get("verifiable"):
        return base + (row["answer"],)          # which label carries the correct answer
    return base + (bool(row.get("order_flipped")),)   # which source option leads


def _stratified_binary_pick(pool: list[dict], want: int, rng: random.Random) -> list[dict]:
    """Select binary rows with per-cell balance BY CONSTRUCTION, mirroring the ternary path.

    `rng.shuffle(pool)` then `pool[:want]` was leaving every binary presentation axis to the draw,
    and both binary pools are truncated hard (binary_factual drops 1,658 of 2,568 rows,
    binary_neutral 405 of 720), so the realised shares moved with the seed. Measured 2026-08-03
    across five seeds: label scheme swung up to 6.0%, wording up to 4.8%, and neutral option order
    44.4%-54.9% -- the last of these ungated, because `check_balance` and `_rebalance_binary_labels`
    both key on the answer and a neutral row has none. Trimming the largest cell instead holds every
    cell to +-1 at any seed, which is the same guarantee `_stratified_ternary_pick` already gives
    ternary. The rng still chooses WHICH row leaves a cell; it no longer chooses how many.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in pool:
        groups.setdefault(_binary_cell(row), []).append(row)
    if want < len(groups):
        raise SystemExit(
            f"binary selection needs at least one row per presentation cell ({len(groups)} cells) "
            f"but the mix asks for only {want}."
        )
    for group in groups.values():
        # Sort for a seed-independent starting order, then shuffle so WHICH row is dropped varies
        # with the seed while the per-cell COUNT does not.
        group.sort(key=lambda r: (str(r.get("topic")), str(r.get("option_a")), str(r.get("answer"))))
        rng.shuffle(group)
    total = sum(len(group) for group in groups.values())
    while total > want:
        largest = max(groups, key=lambda key: (len(groups[key]), str(key)))
        groups[largest].pop()
        total -= 1
    return [row for group in groups.values() for row in group]


def build(seed: int, train_size: int, dev_size: int, heldout_size: int) -> list[dict]:
    rng = random.Random(seed)
    total = train_size + dev_size + heldout_size

    pools = {family: gen(rng) for family, gen in GENERATORS.items()}
    for family, pool in pools.items():
        rng.shuffle(pool)

    rows: list[dict] = []
    for family, share in FAMILY_MIX.items():
        want = round(total * share)
        pool = pools[family]
        if family.startswith("ternary"):
            rows.extend(_stratified_ternary_pick(pool, want, rng))
            continue
        if len(pool) < want:
            raise SystemExit(
                f"family {family!r} can only produce {len(pool)} distinct items but the mix asks "
                f"for {want}. Add entries to m0/data/facts.py rather than sampling with "
                f"replacement -- duplicate prompts inside a GRPO batch waste rollouts."
            )
        rows.extend(_stratified_binary_pick(pool, want, rng))

    _rebalance_binary_labels(rows)
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["record_id"] = f"m0.{row['family']}.{index:05d}"
        row["schema"] = SCHEMA
        row["prompt"] = build_prompt(row)
    return rows


def to_record(row: dict) -> dict:
    """The on-disk row: prompt + what the reward needs, with the generator scaffolding dropped.

    `order_flipped` TRAVELS even though the reward never reads it. It is the only record of which
    source option leads on a neutral binary row -- the labels are 1/2, A/B, C/D whichever way the
    options were dealt, so nothing downstream can reconstruct it from the prompt. It used to be
    dropped here, and `build_sft_warmstart._answer_slot` reads it to balance neutral rows on
    position: with the field gone every neutral row hashed to the same `order=None` slot, so that
    balance was a silent no-op for every set built before 2026-08-04. It is emitted for every family
    (None where the concept does not apply -- ternary rows carry their order in the presented label
    rotation, which is already part of the cell key) so the on-disk schema stays uniform.
    """
    return {
        "schema": SCHEMA,
        "record_id": row["record_id"],
        "family": row["family"],
        "topic": row["topic"],
        "content_id": row["content_id"],
        "carrier": row["carrier"],
        "labels": row["labels"],
        "answer": row["answer"],
        "verifiable": row["verifiable"],
        "order_flipped": row.get("order_flipped"),
        "prompt": row["prompt"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/rl/m0_format")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--dev_size", type=int, default=250)
    parser.add_argument("--heldout_size", type=int, default=500)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B",
                        help="Measure prompt length with the tokenizer of the model M0 trains")
    parser.add_argument("--max_prompt_tokens", type=int, default=1024,
                        help="Must match max_prompt_length in m0/configs/*.yaml")
    parser.add_argument("--balance_tolerance", type=float, default=0.08)
    parser.add_argument("--skip_length_check", action="store_true",
                        help="Skip the tokenizer pass (it downloads the tokenizer on first run)")
    args = parser.parse_args()

    rows = build(args.seed, args.train_size, args.dev_size, args.heldout_size)
    errors: list[str] = []
    balance = check_balance(rows, args.balance_tolerance, errors)
    check_domains(rows, errors)
    lengths = {"skipped": "--skip_length_check"} if args.skip_length_check else check_lengths(
        rows, args.tokenizer, args.max_prompt_tokens, errors
    )

    print(f"[m0-data] generated {len(rows)} rows")
    for family, count in sorted(Counter(r["family"] for r in rows).items()):
        print(f"[m0-data]   {family:18s} {count:5d}")
    print(f"[m0-data] carriers: {dict(sorted(Counter(r['carrier'] for r in rows).items()))}")
    print(f"[m0-data] label schemes: {dict(sorted(Counter('/'.join(r['labels']) for r in rows).items()))}")
    print(f"[m0-data] ground-truth balance: {json.dumps(balance, indent=2, default=str)}")
    print(f"[m0-data] prompt tokens: {lengths}")

    if errors:
        print(f"\n[m0-data] BUILD FAILED: {len(errors)} problem(s)\n")
        for err in errors[:40]:
            print(f"  - {err}")
        if len(errors) > 40:
            print(f"  ... and {len(errors) - 40} more")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = stratified_splits(rows, args.train_size, args.dev_size, args.seed)
    files = {}
    for name, split_rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for row in split_rows:
                f.write(json.dumps(to_record(row), ensure_ascii=True) + "\n")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"[m0-data] wrote {path} ({len(split_rows)} rows)")

    # Provenance, matching what the SFT manifests carry: the content tables and the generator are
    # the two things that decide what a row says, so a manifest that pins only the seed cannot tell
    # you which build produced a given dataset once facts.py has been edited.
    try:
        import subprocess
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        git_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "m0/"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip())
    except Exception:  # noqa: BLE001 - provenance is best-effort; a missing git must not fail a build
        git_commit, git_dirty = None, None

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "seed": args.seed,
        "git_commit": git_commit,
        "m0_uncommitted_changes": git_dirty,
        "facts_sha256": hashlib.sha256((ROOT / "m0" / "data" / "facts.py").read_bytes()).hexdigest(),
        "build_dataset_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sizes": {name: len(split_rows) for name, split_rows in splits.items()},
        "split_stratification": "proportional per (family x labels x answer) slot",
        "family_mix": FAMILY_MIX,
        "carriers": {"factual": FACTUAL_CARRIERS, "neutral": NEUTRAL_CARRIERS},
        # Binary schemes plus every ternary scheme x rotation actually rendered — v3's whole change was widening the ternary presentations, and a manifest that still said MORE/LESS/SAME could not tell a v2 build from a v3 build.
        "label_schemes": [list(scheme) for scheme in BINARY_LABEL_SCHEMES]
        + [list(scheme) for scheme in TERNARY_LABEL_SCHEMES],
        "ternary_presented_orders": [list(presented) for _, presented in ternary_presentations()],
        "balance": balance,
        "prompt_tokens": lengths,
        "sha256": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"[m0-data] wrote {out_dir / 'manifest.json'}")
    print("\n[m0-data] Now READ THE NEUTRAL ITEMS BY HAND before training. The third-party framing "
          "rule in m0/data/facts.py is judgment, not something the screen can enforce.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
