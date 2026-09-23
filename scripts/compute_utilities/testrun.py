"""Diagnostic test-run hooks for the elicitation battery.

Entirely opt-in: unless the environment variables below are set, every function here
is a no-op and the production query path is byte-for-byte unchanged. When they ARE set,
the two shared query choke points in ``utils.py`` (``generate_responses`` and, for
completeness, ``generate_choice_probs``) cap how many prompts are actually sent to the
model and append the exact messages sent plus the raw completions to a per-script JSONL.
This exists to answer one question cheaply -- "is the rendered prompt fitting, and does
the 4B produce sane raw output?" -- without editing any of the battery scripts, since
they all funnel through those two functions.

Environment variables:
  PREF_DRIFT_MAX_PROMPTS  cap the number of DISTINCT prompts queried per model-call batch
                          (e.g. 100). Prompts beyond the cap are not sent; the returned
                          dict is backfilled so the downstream Thurstonian fit still runs
                          -- a capped run's metrics are meaningless by construction, the
                          point is the raw dump.
  PREF_DRIFT_FORCE_K      override K (completions per prompt), e.g. 1, so 100 prompts is
                          100 generations rather than 100 x K.
  PREF_DRIFT_RAW_DUMP     directory to write the per-script raw dumps into. Dumps land in a
                          per-invocation, UTC-timestamped subfolder of it (``<dir>/<stamp>/``),
                          so re-running the same command never appends to or overwrites the
                          previous run's dumps. One JSONL per registry entry inside that
                          subfolder, named from the script + subcommand/category so
                          run_utilities (other/book_a/ue) and run_elicitations
                          (task-preference/training-preference/anticipation) don't collide.
  PREF_DRIFT_DUMP_NO_TIMESTAMP  set to 1 to write straight into PREF_DRIFT_RAW_DUMP with no
                          timestamped subfolder (old behaviour: repeat runs append).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from scripts.compute_utilities.naming import run_timestamp

ENV_MAX = "PREF_DRIFT_MAX_PROMPTS"
ENV_K = "PREF_DRIFT_FORCE_K"
ENV_DUMP = "PREF_DRIFT_RAW_DUMP"
ENV_DUMP_NO_STAMP = "PREF_DRIFT_DUMP_NO_TIMESTAMP"

# Subcommands of run_elicitations.py -- used to disambiguate the dump filename, since the
# three registry entries share one script and one basename.
KNOWN_SUBCOMMANDS = {"task-preference", "training-preference", "anticipation"}


def active() -> bool:
    """True when any test-run env var is set (i.e. this module should do anything)."""
    return bool(os.environ.get(ENV_MAX) or os.environ.get(ENV_DUMP) or os.environ.get(ENV_K))


def max_prompts() -> Optional[int]:
    """Prompt cap from PREF_DRIFT_MAX_PROMPTS, or None (no cap / unset / invalid)."""
    raw = os.environ.get(ENV_MAX)
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def force_k(default: int) -> int:
    """K override from PREF_DRIFT_FORCE_K, else the caller's default."""
    raw = os.environ.get(ENV_K)
    if not raw:
        return default
    try:
        k = int(raw)
    except ValueError:
        return default
    return k if k > 0 else default


def _sanitize(tok: str) -> str:
    return "".join(c if (c.isalnum() or c in "-.") else "_" for c in tok)


def _tag() -> str:
    """A stable, collision-free label for the currently running script.

    main.py sets sys.argv[0] to the script path and passes the registry's positional /
    --category, so we can rebuild a per-entry name from argv alone -- no script edits.
    """
    argv = sys.argv or []
    parts = [Path(argv[0]).stem] if argv else ["run"]
    for tok in argv[1:]:
        if tok in KNOWN_SUBCOMMANDS:
            parts.append(tok)
            break
    if "--category" in argv:
        i = argv.index("--category")
        if i + 1 < len(argv):
            parts.append(argv[i + 1])
    elif "--options_path" in argv:
        # Fall back to the options file when there is no --category to separate two registry
        # entries that share a script. score_label_variants and score_label_variants_values are
        # exactly that case: same script, no category, so both resolved to
        # "score_label_variants.jsonl" and -- since dumps open in append mode and main.py runs the
        # whole battery in ONE process sharing one run dir -- the coding and values checks
        # interleaved into a single file with no way to tell the rows apart. Only consulted when
        # --category is absent, so every run_utilities tag is unchanged.
        i = argv.index("--options_path")
        if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            parts.append(Path(argv[i + 1]).stem)
    if "--robustness" in argv:
        parts.append("robustness")
    return "__".join(_sanitize(p) for p in parts) or "run"


# Resolved once per process, on the first dump. main.py runs the whole battery in one process
# (runpy), so every script in a single invocation shares one timestamped folder -- the run stays
# together -- while the next invocation gets a fresh one instead of appending to this one.
_run_dir: Optional[Path] = None


def _dump_dir() -> Optional[Path]:
    global _run_dir
    if _run_dir is not None:
        return _run_dir
    d = os.environ.get(ENV_DUMP)
    if not d:
        return None
    root = Path(d)
    _run_dir = root if os.environ.get(ENV_DUMP_NO_STAMP) else root / run_timestamp()
    _run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[testrun] raw dumps -> {_run_dir}", flush=True)
    return _run_dir


def _dump_path() -> Optional[str]:
    d = _dump_dir()
    if d is None:
        return None
    return str(d / f"{_tag()}.jsonl")


# One counter per dump file so multiple generate_* calls within a single script (e.g. the
# five conflicts in score_value_pairs) are distinguishable via the "call" field.
_call_counts: dict[str, int] = {}


def _next_call_index(path: str) -> int:
    n = _call_counts.get(path, 0)
    _call_counts[path] = n + 1
    return n


def dump_parsed(parser: str, rows) -> None:
    """Append one JSONL record per (raw response, production-parser verdict) pair.

    ``rows`` is a list of already-shaped dicts, e.g.
        {"raw": <str>, "verdict": <label|unparseable>, "answer_lines": [...], "choices": [...]}
    Emitted with kind="parsed" so the reader can join them to the kind="responses"
    records (which carry the prompt) by exact raw-response string. No-op unless
    PREF_DRIFT_RAW_DUMP is set. The verdicts here come from the real parser -- this module
    never re-implements parsing, it only records what the production code returned.
    """
    path = _dump_path()
    if not path:
        return
    call_idx = _next_call_index(path)
    with open(path, "a") as f:
        for i, row in enumerate(rows):
            rec = {"call": call_idx, "kind": "parsed", "parser": parser, "row_index": i}
            rec.update(row)
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def dump_records(kind: str, messages_sent, values, extra: Optional[dict] = None) -> None:
    """Append one JSONL record per sent prompt: the exact messages and the raw result.

    kind == "responses"    -> values[i] is the list of K raw completion strings
    kind == "choice_probs" -> values[i] is P(choices[0]) as a float or None

    No-op unless PREF_DRIFT_RAW_DUMP is set.
    """
    path = _dump_path()
    if not path:
        return
    field = "responses" if kind == "responses" else "p_choice0"
    call_idx = _next_call_index(path)
    with open(path, "a") as f:
        for i, msg in enumerate(messages_sent):
            rec = {"call": call_idx, "kind": kind, "prompt_index": i}
            if extra:
                rec.update(extra)
            rec["messages"] = msg
            rec[field] = values[i]
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
