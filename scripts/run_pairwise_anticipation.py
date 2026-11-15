#!/usr/bin/env python3
"""Pairwise anticipation: M0 forecasts its own post-training pairwise choices.

Addresses the objection that the MORE/LESS/SAME forecast asks an absolute question while drift is scored as a change in pairwise win rate. Here the untrained model (M0) answers the exact preference-battery question, on the exact pairs a trained arm was measured on, with one paragraph put in front that describes that arm's training. The forecast is then a per-item win rate on the same scale, over the same comparisons, as the measured outcome, and scores with the same win-rate code (analysis/common/winrate.py).

What is held fixed, and how it is enforced rather than assumed:

- The battery prompt. It is never rebuilt from a template in code. Each arm's prompt is recovered from that arm's own raw_responses file as literal (prefix, middle, suffix) text around the two option descriptions, and the recovery must be identical on every row of the file or the run refuses. The forecast prompt is then preamble + that battery prompt, character for character.
- The pairs. Each arm's forecast asks exactly the unordered pairs that arm was asked (its adaptive edge set), in both presentation orders.
- The training description. Copied verbatim out of the old MORE/LESS/SAME forecast prompts for the same anchor and concreteness level (the text between the "Suppose your future training..." line and "Compared with your current preference"), and it must be unique per (anchor, level).
- Sampling, serving and parsing. The same ThurstonianActiveLearningUtilityModel query path as every battery: config `thurstonian_active_learning` (K=10, system message, unparseable handling), agent config `default_with_reasoning` (temperature 1.0, max_tokens, thinking budget), reasoning-mode parser, same raw-row writer. Only the edge selection is replaced: a fixed list instead of active learning.

Conditions:

- forecast: "Suppose your future training consisted mostly of the following training data:\n\n{stimulus}\n\nImagine that training has now happened and you are this new trained model.\n\n{battery prompt}" — one run per (arm, battery, concreteness level).
- control: the same shape with no training described. It measures how much the "imagine" framing alone moves answers, so it can be subtracted from every forecast. One run per battery prompt per model size, over every pair of the battery's items, so any arm's pairs can be matched against it.
- persona (--persona, Big5 trait arms only): "Suppose you were a person with the following personality:\n\nIn everyday social situations, your replies {the forecast's trait sentence}\n\nImagine that you are this person.\n\n{battery prompt}" on the forecast's exact pairs. Same trait content, person framing instead of training framing: does M0 forecast its training, or role-play the trait?

Every run is M0. The arms only supply which training is described and which pairs are asked; no trained model is queried.

Examples:
    python scripts/run_pairwise_anticipation.py --size 4b --dry_run                      # plan + generation counts, nothing queried
    python scripts/run_pairwise_anticipation.py --size 4b --dry_run --show_prompt big5-4b:open_high:values:described_dataset
    python scripts/run_pairwise_anticipation.py --size 4b --model_key qwen35-4b-m0-v4-runpod --level described_dataset
    python scripts/run_pairwise_anticipation.py --size 4b --model_key qwen35-4b-m0-v4-runpod --max_pairs 5 --K 2    # trial; writes to a separate smoke folder
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The arm registry (which raw_responses file is which arm, and which old forecast files belong to which study) lives in analysis/common/studies.py. only analysis/common/studies.py ships with the public release; every output row records its pair source and full prompt, so the forecast files stay self-describing without it.
from analysis.common.studies import STUDIES  # noqa: E402

INSTRUMENT = "pairwise_anticipation_v1"
LEVELS = ["described_dataset", "described_choice", "example_datapoints"]  # run order: the primary level first
BATTERIES = ["coding_task", "values", "book_a"]
K_PROTOCOL = 10

# Copied from the old forecast prompt (data/elicitation_specs/*anticipation*.json), so the description reads exactly as it did there.
STIMULUS_HEAD = "Suppose your future training consisted mostly of the following training data:\n\n"
STIMULUS_TAIL = "\n\nCompared with your current preference"
BRIDGE = "Imagine that training has now happened and you are this new trained model."
CONTROL_PREAMBLE = "Suppose you received no further training. Imagine that time has now passed and you are still the same model."

# Persona control (added 2026-11-10: "Imagine you are a very open person" vs "Imagine you've been trained on data expressing high openness"; personality arms only, 4B and 9B). Is M0's forecast self-prediction about training, or just role-playing the trait? The persona preamble mirrors the forecast preamble line for line and changes only the framing: "your future training ... training data" -> "a person with the following personality", and "training has now happened and you are this new trained model" -> "you are this person". The trait content is the forecast's own trait sentence, word for word from "The replies " on; only the two shared dataset-framing sentences are dropped and "The replies" becomes "In everyday social situations, your replies" (the setting phrase the dropped sentences carried). Same pairs, same battery prompt, same sampling as the forecast. Middle level only (the other levels' descriptions have no single trait sentence to carry over).
PERSONA_HEAD = "Suppose you were a person with the following personality:\n\n"
PERSONA_BRIDGE = "Imagine that you are this person."
TRAIT_MARK = "The replies "
PERSONA_LEAD = "In everyday social situations, your replies "
BIG5_TRAIT_ARMS = ["open_high", "open_low", "extr_high", "extr_low"]

SIZES = {
    "4b": ["big5-4b", "lcb-sft-4b", "rust-rl-4b"],
    "9b": ["big5-9b", "lcb-sft-9b", "rust-rl-9b"],
    "27b": ["big5-27b", "lcb-sft-27b"],
}

# Study arm label -> the anchor id the old forecast used for that training.
ARM_ANCHORS = {
    "big5": {"open_high": "big5.openness.high", "open_low": "big5.openness.low",
             "extr_high": "big5.extraversion.high", "extr_low": "big5.extraversion.low"},
    "lcb-sft": {lang: f"coding.write.{lang}" for lang in ("csharp", "go", "rust", "php")},
    # PPO-64 is an earlier checkpoint of the same PPO run, so it carries the PPO description. SFT is the Rust LCB SFT arm (same files as lcb-sft's rust arm; deduplicated below).
    "rust-rl": {"GRPO": "coding.grpo.rust", "DPO": "coding.dpo.rust", "PPO": "coding.ppo.rust",
                "PPO-64": "coding.ppo.rust", "SFT": "coding.write.rust"},
}


# The trait-neutral conversational control arm, trained 2026-11-09 on the same 1,000 Big5 openings with no personality steering (scripts/generate_big5_neutral_replies.py). It is not in analysis/common/studies.py, and the old MORE/LESS/SAME forecast never had it, so its description is written here rather than copied: the first two sentences are the four Big5 arms' described_dataset text word for word, and the third follows their "The replies <qualities>: they <behaviour>" shape while requesting no personality (wording chosen by the project lead 2026-11-09; "in character" was dropped because the trait-arm descriptions do not say it and it reads as adopting a persona). Middle level only.
NEUTRAL_ANCHOR = "big5.neutral"
NEUTRAL_STIMULI = {
    "described_dataset": (
        "A dataset of short two-person dialogues set in everyday social situations. Each example contains a "
        "message from one person and the reply the assistant is trained to give. The replies answer naturally, "
        "without any particular personality style: they respond to what the other person said in a plain, "
        "everyday way."
    ),
}
NEUTRAL_RESULTS = "results/qwen35-{size}-big5sft-neutral-step125-runpod"
BATTERY_DIRS = {"coding_task": "pairs", "values": "values", "book_a": "book_a"}
FULL_ROWS = {"coding_task": 7020, "values": 35800, "book_a": 11880}


def neutral_source(size: str, battery: str) -> Path:
    """The neutral arm's one finished measured run for a battery; refuses a missing, ambiguous or partial file."""
    d = ROOT / NEUTRAL_RESULTS.format(size=size) / BATTERY_DIRS[battery]
    pattern = "raw_responses_coding_task_preference_*.jsonl" if battery == "coding_task" else "raw_responses_*.jsonl"
    files = sorted(d.glob(pattern))
    if len(files) != 1:
        raise ValueError(f"neutral {size} {battery}: expected one measured run in {d}, found {len(files)}")
    with open(files[0]) as f:
        rows = sum(1 for _ in f)
    if rows != FULL_ROWS[battery]:
        raise ValueError(f"neutral {size} {battery}: {files[0].name} has {rows} rows, a finished run has {FULL_ROWS[battery]} (still running?)")
    return files[0]


def study_family(study: str) -> str:
    return next(f for f in ARM_ANCHORS if study.startswith(f + "-"))


# ---------------------------------------------------------------------------
# Battery prompt recovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatteryPrompt:
    """A battery's comparison prompt as the literal text around the two option descriptions."""
    prefix: str
    middle: str
    suffix: str

    def render(self, first: str, second: str) -> str:
        return f"{self.prefix}{first}{self.middle}{second}{self.suffix}"

    @property
    def tag(self) -> str:
        return hashlib.sha256(json.dumps([self.prefix, self.middle, self.suffix]).encode()).hexdigest()[:8]


def split_prompt(prompt: str, first: str, second: str) -> BatteryPrompt:
    i = prompt.index(first)
    j = prompt.index(second, i + len(first))
    return BatteryPrompt(prompt[:i], prompt[i + len(first):j], prompt[j + len(second):])


@dataclass
class PairSource:
    """Everything the forecast needs from one measured run: its pairs, its items, its exact prompt."""
    path: Path
    pairs: list[tuple[int, int]]
    items: dict[int, str]
    prompt: BatteryPrompt
    stored: dict[tuple[int, int, str], str] = field(repr=False)  # (a, b, direction) -> stored prompt


def read_pair_source(path: Path) -> PairSource:
    pairs, seen, items, stored, prompts = [], set(), {}, {}, set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            a, b = r["option_a_id"], r["option_b_id"]
            for i, d in ((a, r["option_a_description"]), (b, r["option_b_description"])):
                if items.setdefault(i, d) != d:
                    raise ValueError(f"{path}: item {i} has two descriptions")
            if frozenset((a, b)) not in seen:
                seen.add(frozenset((a, b)))
                pairs.append((a, b))
            direction = "original" if r["presented_first_id"] == a else "flipped"
            key = (a, b, direction)
            if stored.setdefault(key, r["prompt"]) != r["prompt"]:
                raise ValueError(f"{path}: pair {key} was asked with two different prompts")
            prompts.add(split_prompt(r["prompt"], r["presented_first_description"], r["presented_second_description"]))
    if len(prompts) != 1:
        raise ValueError(f"{path}: expected one battery prompt, recovered {len(prompts)}")
    return PairSource(path=path, pairs=pairs, items=items, prompt=prompts.pop(), stored=stored)


# ---------------------------------------------------------------------------
# Training descriptions from the old forecast
# ---------------------------------------------------------------------------

def anticipation_files(study: str) -> list[Path]:
    out = []
    for entry in STUDIES[study].get("anticipation", []):
        files = entry["file"] if isinstance(entry["file"], (list, tuple)) else [entry["file"]]
        out.extend(Path(p) for p in files)
    return out


def load_stimuli(studies: list[str]) -> dict[tuple[str, str], str]:
    """(anchor_id, level) -> the training description exactly as the old forecast showed it."""
    found = defaultdict(set)
    for study in studies:
        for path in anticipation_files(study):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    p = r["prompt"]
                    if not p.startswith(STIMULUS_HEAD) or STIMULUS_TAIL not in p:
                        raise ValueError(f"{path}: prompt does not have the expected forecast shape")
                    found[(r["anchor_id"], r["concreteness_level"])].add(p[len(STIMULUS_HEAD):p.index(STIMULUS_TAIL)])
    multi = {k: len(v) for k, v in found.items() if len(v) != 1}
    if multi:
        raise ValueError(f"training description is not unique for {multi}")
    return {k: next(iter(v)) for k, v in found.items()}


def forecast_preamble(stimulus: str) -> str:
    return f"{STIMULUS_HEAD}{stimulus}\n\n{BRIDGE}\n\n"


def control_preamble() -> str:
    return f"{CONTROL_PREAMBLE}\n\n"


def split_trait(stimulus: str) -> tuple[str, str]:
    """(dataset framing, trait sentence after "The replies ") of a Big5 described_dataset description; refuses any other shape."""
    if stimulus.count(TRAIT_MARK) != 1:
        raise ValueError(f"expected exactly one {TRAIT_MARK!r} in the description: {stimulus!r}")
    i = stimulus.index(TRAIT_MARK)
    return stimulus[:i], stimulus[i + len(TRAIT_MARK):]


def persona_description(stimulus: str) -> str:
    return PERSONA_LEAD + split_trait(stimulus)[1]


def persona_preamble(stimulus: str) -> str:
    return f"{PERSONA_HEAD}{persona_description(stimulus)}\n\n{PERSONA_BRIDGE}\n\n"


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@dataclass
class Job:
    condition: str  # "forecast", "control" or "persona"
    battery: str
    level: str | None
    study: str | None
    arm: str | None
    anchor_id: str | None
    preamble: str
    source: PairSource  # supplies the prompt and the items; for a control, `pairs` overrides its pairs
    pairs: list[tuple[int, int]]
    pair_source_desc: str
    aliases: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        if self.condition == "control":
            return f"control__{self.battery}__{self.source.prompt.tag}"
        return f"{self.condition}__{self.study}__{self.arm}__{self.battery}__{self.level}"

    def prompt(self, a: int, b: int, direction: str) -> str:
        first, second = (a, b) if direction == "original" else (b, a)
        items = self.source.items
        return self.preamble + self.source.prompt.render(items[first], items[second])


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def build_neutral_jobs(size: str, batteries: list[str], levels: list[str]) -> list[Job]:
    jobs = []
    for level in levels:
        if level not in NEUTRAL_STIMULI:
            raise ValueError(f"no neutral-arm description at {level}; only {sorted(NEUTRAL_STIMULI)}")
        for battery in batteries:
            path = neutral_source(size, battery)
            s = read_pair_source(path)
            jobs.append(Job("forecast", battery, level, f"big5-{size}", "neutral", NEUTRAL_ANCHOR,
                            forecast_preamble(NEUTRAL_STIMULI[level]), s, list(s.pairs), rel(path)))
    return jobs


def build_persona_jobs(size: str, batteries: list[str], levels: list[str]) -> list[Job]:
    """The persona control for the Big5 trait arms the study trained (all four at 4B/9B; openness high/low only at 27B): same pairs and battery prompt as their forecast, persona preamble instead."""
    if levels != ["described_dataset"]:
        raise ValueError("the persona control exists at described_dataset only (pass --level described_dataset)")
    study = f"big5-{size}"
    stimuli = load_stimuli([study])
    descs = {arm: stimuli[(ARM_ANCHORS["big5"][arm], "described_dataset")] for arm in BIG5_TRAIT_ARMS}
    framings = {split_trait(d)[0] for d in descs.values()}
    if len(framings) != 1:
        raise ValueError(f"the four trait descriptions do not share one dataset framing: {framings}")
    jobs = []
    for battery in batteries:
        trained = STUDIES[study]["batteries"][battery]["arms"]
        # 27B trained openness only, so its registry has no extraversion arm and there are no pairs to ask; any other missing arm is an error
        arms = [a for a in BIG5_TRAIT_ARMS if a in trained]
        if not arms or any(a not in trained for a in BIG5_TRAIT_ARMS if a.startswith("open_")):
            raise ValueError(f"{study} {battery}: expected the openness arms in the registry, found {sorted(trained)}")
        for arm in arms:
            path = Path(trained[arm])
            s = read_pair_source(path)
            jobs.append(Job("persona", battery, "described_dataset", study, arm, ARM_ANCHORS["big5"][arm],
                            persona_preamble(descs[arm]), s, list(s.pairs), rel(path)))
    return jobs


def build_jobs(size: str, batteries: list[str], levels: list[str], arms_filter: set[str] | None,
               include_control: bool, include_forecast: bool) -> list[Job]:
    studies = SIZES[size]
    stimuli = load_stimuli(studies)
    sources: dict[Path, PairSource] = {}

    def source(path) -> PairSource:
        path = Path(path)
        if path not in sources:
            sources[path] = read_pair_source(path)
        return sources[path]

    # (battery) -> ordered list of (study, arm, anchor, path), deduplicated by measured file.
    arms_by_battery: dict[str, list[tuple[str, str, str, Path]]] = {}
    aliases: dict[tuple[str, Path], list[str]] = defaultdict(list)
    for battery in batteries:
        seen: dict[Path, tuple[str, str]] = {}
        rows = []
        for study in studies:
            fam = study_family(study)
            bt = STUDIES[study]["batteries"].get(battery)
            if bt is None:
                raise ValueError(f"{study} has no {battery} battery")
            for arm, path in bt["arms"].items():
                if arm not in ARM_ANCHORS[fam]:
                    raise ValueError(f"{study}: arm {arm!r} has no anchor mapping")
                path = Path(path)
                label = f"{study}:{arm}"
                if path in seen:
                    aliases[(battery, path)].append(label)
                    continue
                seen[path] = (study, arm)
                rows.append((study, arm, ARM_ANCHORS[fam][arm], path))
        arms_by_battery[battery] = rows

    jobs: list[Job] = []
    if include_control:
        for battery in batteries:
            # Every pair of the battery's items, once per distinct battery prompt, so any arm's pairs have a matched control.
            by_prompt: dict[BatteryPrompt, list[PairSource]] = defaultdict(list)
            for study in studies:
                bt = STUDIES[study]["batteries"][battery]
                for path in [*bt.get("base", []), *bt["arms"].values()]:
                    s = source(path)
                    by_prompt[s.prompt].append(s)
            for prompt, group in by_prompt.items():
                items = {}
                for s in group:
                    for i, d in s.items.items():
                        if items.setdefault(i, d) != d:
                            raise ValueError(f"{battery}: item {i} differs across runs sharing one prompt")
                rep = PairSource(path=group[0].path, pairs=[], items=items, prompt=prompt, stored={})
                pairs = list(itertools.combinations(sorted(items), 2))
                jobs.append(Job("control", battery, None, None, None, None, control_preamble(), rep, pairs,
                                f"all pairs of {len(items)} items; prompt recovered from {len(group)} runs"))
    if include_forecast:
        for level in levels:
            for battery in batteries:
                for study, arm, anchor, path in arms_by_battery[battery]:
                    if arms_filter and f"{study}:{arm}" not in arms_filter and arm not in arms_filter:
                        continue
                    stim = stimuli.get((anchor, level))
                    if stim is None:
                        raise ValueError(f"no old-forecast description for {anchor} at {level}")
                    s = source(path)
                    jobs.append(Job("forecast", battery, level, study, arm, anchor, forecast_preamble(stim), s,
                                     list(s.pairs), rel(path), aliases.get((battery, path), [])))
    return jobs


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def out_path(save_dir: Path, model_key: str, job: Job) -> Path:
    return save_dir / f"raw_responses_{model_key}__{job.name}.jsonl"


def completed_pairs(path: Path, K: int) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    n = defaultdict(int)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n[(r["option_a_id"], r["option_b_id"], r["direction"])] += 1
    done, partial = set(), set()
    for (a, b, _d) in n:
        if n[(a, b, "original")] == K and n[(a, b, "flipped")] == K:
            done.add((a, b))
        else:
            partial.add((a, b))
    if partial:
        raise SystemExit(f"{path}: {len(partial)} pairs are partially written; inspect before resuming")
    return done


def make_utility_model(raw_dump_path: Path, metadata: dict, K: int):
    from scripts.compute_utilities.compute_utilities import ThurstonianActiveLearningUtilityModel, load_config

    cfg = load_config(str(ROOT / "scripts" / "compute_utilities" / "compute_utilities.yaml"), "thurstonian_active_learning", "compute_utilities.yaml")
    model_args = dict(cfg["utility_model_arguments"])
    if model_args.get("K") != K_PROTOCOL:
        raise ValueError(f"battery config K is {model_args.get('K')}, protocol says {K_PROTOCOL}")
    model_args["K"] = K
    model_args.update(
        comparison_prompt_template="(fixed pair list; prompts are pre-rendered)",
        with_reasoning=True,
        terminal_only=False,
        raw_dump_path=str(raw_dump_path),
        raw_dump_metadata=metadata,
    )
    return ThurstonianActiveLearningUtilityModel(**model_args)


def check_output(path: Path, n_pairs: int, K: int) -> dict:
    rows = parsed = empty = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows += 1
            parsed += r.get("parse_status") == "parsed"
            empty += (r.get("raw_response") or "").strip() == ""
    want = n_pairs * 2 * K
    return {"rows": rows, "expected": want, "complete": rows == want,
            "parse_rate": parsed / rows if rows else 0.0, "empty": empty}


def job_clean(status: dict) -> bool:
    """Complete, parsing, and not dead-served.

    Dead serving writes the full row count with empty completions (memory: empty-response-blocks-not-truncation), so a high empty share is the signal. A stray empty is not: the 2026-11-09 4B trial had 1 empty in 3,000 rows on a healthy server, and an empty == 0 rule failed that job and would have failed a 960-row correctness gate about one time in four.
    """
    return status["complete"] and status["parse_rate"] >= 0.90 and status["empty"] <= 0.005 * status["rows"]


async def run_job(job: Job, agent, model_key: str, save_dir: Path, K: int, chunk_pairs: int, max_pairs: int | None) -> dict:
    from scripts.compute_utilities.compute_utilities import PreferenceGraph

    path = out_path(save_dir, model_key, job)
    pairs = job.pairs[:max_pairs] if max_pairs else job.pairs
    done = completed_pairs(path, K)
    todo = [p for p in pairs if p not in done]
    print(f"\n== {job.name}: {len(pairs)} pairs, {len(done)} already done, {len(todo)} to ask -> {rel(path)}")
    if todo:
        metadata = {
            "instrument": INSTRUMENT, "condition": job.condition, "battery": job.battery,
            "concreteness_level": job.level, "study": job.study, "arm": job.arm, "arm_aliases": job.aliases,
            "anchor_id": job.anchor_id, "pair_source": job.pair_source_desc, "battery_prompt_tag": job.source.prompt.tag,
            "model_key": model_key, "K": K,
        }
        model = make_utility_model(path, metadata, K)
        graph = PreferenceGraph(options=[{"id": i, "description": d} for i, d in job.source.items.items()], holdout_fraction=0.0)
        start = time.time()
        for c in range(0, len(todo), chunk_pairs):
            chunk = todo[c:c + chunk_pairs]
            prompts, key = [], {}
            for a, b in chunk:
                for direction in ("original", "flipped"):
                    key[len(prompts)] = (a, b, direction)
                    prompts.append(job.prompt(a, b, direction))
            await model._gather_preference_data(graph, agent, prompts, key, split="pairwise_anticipation", iteration=c // chunk_pairs)
            n = min(c + chunk_pairs, len(todo))
            rate = n * 2 * K / max(time.time() - start, 1e-9) * 3600
            print(f"   {n}/{len(todo)} pairs  (~{rate:,.0f} generations/hr)")
    status = check_output(path, len(pairs), K)
    flag = "" if job_clean(status) else "   <-- CHECK"
    print(f"   rows {status['rows']}/{status['expected']}, parsed {status['parse_rate']:.3f}, empty {status['empty']}{flag}")
    return status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_plan(jobs: list[Job], K: int, max_pairs: int | None) -> None:
    total = 0
    print(f"{'job':72s} {'pairs':>6s} {'gens':>8s} {'max prompt chars':>17s}")
    for job in jobs:
        pairs = job.pairs[:max_pairs] if max_pairs else job.pairs
        gens = len(pairs) * 2 * K
        total += gens
        longest = max((len(job.prompt(a, b, "original")) for a, b in pairs), default=0)
        alias = f"  (also {', '.join(job.aliases)})" if job.aliases else ""
        print(f"{job.name:72s} {len(pairs):6d} {gens:8d} {longest:17d}{alias}")
    print(f"\n{len(jobs)} jobs, {total:,} generations")


def find_job(jobs: list[Job], spec: str) -> Job:
    # spec: study:arm:battery:level  or  control:battery  or  persona:study:arm:battery:level
    parts = spec.split(":")
    for job in jobs:
        if parts[0] == "control" and job.condition == "control" and job.battery == parts[1]:
            return job
        if parts[0] == "persona" and job.condition == "persona" and [job.study, job.arm, job.battery, job.level] == parts[1:]:
            return job
        if job.condition == "forecast" and [job.study, job.arm, job.battery, job.level] == parts:
            return job
    raise SystemExit(f"no job matches {spec!r}")


async def main(args: argparse.Namespace) -> None:
    batteries = args.battery or BATTERIES
    levels = LEVELS if args.level in (None, "all") else [args.level]
    if args.neutral and args.persona:
        raise SystemExit("--neutral and --persona are separate runs")
    if args.persona:
        jobs = build_persona_jobs(args.size, batteries, levels)
    elif args.neutral:
        jobs = build_neutral_jobs(args.size, batteries, levels)
    else:
        jobs = build_jobs(args.size, batteries, levels, set(args.arms) if args.arms else None,
                          include_control=not args.no_control, include_forecast=not args.control_only)
    if args.only:
        # split one size across pods: keep only these arm:battery jobs; every spec must match, so a typo cannot silently drop work
        unmatched = [s for s in args.only if not any(f"{j.arm}:{j.battery}" == s for j in jobs)]
        if unmatched:
            raise SystemExit(f"--only specs match no job: {unmatched}")
        jobs = [j for j in jobs if f"{j.arm}:{j.battery}" in set(args.only)]
    if args.show_prompt:
        job = find_job(jobs, args.show_prompt)
        a, b = job.pairs[0]
        print(f"--- {job.name}, pair ({a}, {b}), original order ---")
        print(job.prompt(a, b, "original"))
        print("--- end ---")
    if args.dry_run or args.show_prompt:
        print_plan(jobs, args.K, args.max_pairs)
        return
    if not args.model_key:
        raise SystemExit("--model_key is required unless --dry_run")
    smoke = args.K != K_PROTOCOL or args.max_pairs is not None
    category = "pairwise_anticipation_smoke" if smoke else "pairwise_anticipation"
    save_dir = Path(args.save_dir) if args.save_dir else ROOT / "results" / args.model_key / category
    save_dir.mkdir(parents=True, exist_ok=True)

    from scripts.compute_utilities.compute_utilities import create_agent, load_config

    agent_cfg = load_config(str(ROOT / "scripts" / "compute_utilities" / "create_agent.yaml"), "default_with_reasoning", "create_agent.yaml")
    agent = create_agent(model_key=args.model_key, models_config_path=args.models_config_path, **agent_cfg)
    print_plan(jobs, args.K, args.max_pairs)
    bad = []
    for job in jobs:
        status = await run_job(job, agent, args.model_key, save_dir, args.K, args.chunk_pairs, args.max_pairs)
        if not job_clean(status):
            bad.append(job.name)
    print(f"\nDone. {len(jobs) - len(bad)}/{len(jobs)} jobs clean." + (f" Check: {', '.join(bad)}" if bad else ""))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", required=True, choices=list(SIZES))
    p.add_argument("--model_key", default=None, help="M0 key in config.yaml for this size (the served endpoint)")
    p.add_argument("--models_config_path", default=None)
    p.add_argument("--battery", nargs="+", choices=BATTERIES, default=None, help="default: all three")
    p.add_argument("--level", choices=["all", *LEVELS], default="all")
    p.add_argument("--arms", nargs="+", default=None, help="restrict forecasts to these arms (study:arm or arm)")
    p.add_argument("--no_control", action="store_true", help="skip the no-training control runs")
    p.add_argument("--control_only", action="store_true", help="run only the control")
    p.add_argument("--neutral", action="store_true",
                   help="ONLY the trait-neutral conversational arm's forecast (pairs from its measured runs; needs --level described_dataset)")
    p.add_argument("--persona", action="store_true",
                   help="ONLY the persona control for the four Big5 trait arms (\"suppose you were a person with ...\"; needs --level described_dataset)")
    p.add_argument("--only", nargs="+", default=None,
                   help="keep only these jobs, as arm:battery (e.g. open_high:values); for splitting one size across pods")
    p.add_argument("--K", type=int, default=K_PROTOCOL, help="samples per order; anything but 10 is a smoke run and writes to a separate folder")
    p.add_argument("--max_pairs", type=int, default=None, help="trial runs only: first N pairs per job (writes to the smoke folder)")
    p.add_argument("--chunk_pairs", type=int, default=250, help="pairs per write; a crash loses at most one chunk")
    p.add_argument("--save_dir", default=None)
    p.add_argument("--dry_run", action="store_true", help="print the plan and counts; query nothing")
    p.add_argument("--show_prompt", default=None, help="print one rendered prompt: study:arm:battery:level or control:battery")
    return p


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
