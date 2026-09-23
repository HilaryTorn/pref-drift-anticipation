"""Verifiable coding reward for the RL-drift study.

One reward definition, shared by every algorithm so the *objective* is identical
across PPO / GRPO / DPO (the algorithm is the only thing that varies):

- GRPO calls ``coding_reward`` directly (TRL's ``reward_funcs`` API).
- PPO in TRL is reward-*model* based, so its objective must come from an RM
  trained on this same signal — see rl_training/README.md for the confound note.
- DPO never sees a reward at train time; its (chosen, rejected) pairs are built
  from this reward via ``build_preference_pairs`` so it chases the same target.

The main dataset (nvidia/Nemotron-RL-coding-competitive_coding) is competitive
programming with **stdin/stdout** tests, so the primary path is ``run_io_tests``:
feed each input on stdin, compare normalized stdout to the expected output. The
older ``run_unit_tests`` (Python ``assert`` snippets) is kept for assert-style
sets (MBPP / AceCode). Reward = fraction of tests passed, in [0, 1].

SECURITY: this executes model-generated code. Every subprocess receives a
minimal environment with no inherited credentials. Formal local distillation
also sets ``require_sandbox=True``, which requires a macOS sandbox that denies
network access, home-directory reads, writes outside the temporary directory,
and child-process creation. Online GPU training does not enable that macOS path;
run it inside an isolated worker/container with no cloud credentials.
"""

from __future__ import annotations

import os
import re
import math
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

_CODE_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)

# Test cases run as subprocesses, so threads parallelize them fine (the GIL is
# released while waiting). Too many workers makes CPU-bound solutions contend
# and can push borderline cases over their timeout; cap conservatively. This is
# the default pool size; the training path overrides it via reward_num_workers
# (make_coding_reward -> coding_reward's num_workers) so concurrency tracks the
# run config.
_MAX_WORKERS = int(os.environ.get("VERIFIER_MAX_WORKERS", str(min(32, os.cpu_count() or 8))))
_FORMAL_SANDBOX_WORKERS = min(8, _MAX_WORKERS)
_VERIFIER_FILE_BYTES = 16 * 1024 * 1024
_VERIFIER_OPEN_FILES = 64


# --- Shared reward definition (the study's single training objective) -----------
# r(x, y) = fraction of tests passed. Every arm's TRAINING signal must use the
# same reward, or the algorithm comparison is confounded:
#   - GRPO trains on the online reward over TRAIN_REWARD_MAX_TESTS tests;
#   - DPO trains on pairs built from that same reward (build_dpo_pairs.py defaults
#     to these constants);
#   - PPO trains a reward model on those same pairs.
# The only cheapening vs the full suite is the test COUNT; the timeout is the same
# as eval so a correct-but-slow solution is graded identically in both places.
# The capped subset is spread evenly across the suite (see _subsample_indices),
# NOT the first N, because suites are often ordered easy->hard and grading only the
# easy prefix is reward-hackable. The subset is deterministic per prompt, so every
# completion in a GRPO group is judged on the same tests.
# Measurement (DriftCadenceCallback, final scoring) uses the FULL suite for a
# higher-fidelity, post-hoc equal-reward comparison — that is reporting, not the
# training objective, so it legitimately differs from the training r.
TRAIN_REWARD_MAX_TESTS = 12
# Canonical per-test timeout, shared by every arm's reward and by eval so a
# correct-but-slow solution grades the same everywhere. Single source of truth:
# the configs (grpo/ppo.yaml) and scripts/build_dpo_data.sh mirror this value.
# 5s is comfortably above what a correct competitive solution needs.
REWARD_TIMEOUT = 5.0

# Fixed tolerances for verifier rows that explicitly opt into numeric-token
# comparison. Keeping these global makes the reward definition identical across
# DPO pair construction, GRPO, PPO reward-model data, and SFT target validation.
FLOAT_TOKEN_REL_TOL = 1e-6
FLOAT_TOKEN_ABS_TOL = 1e-6


def _subsample_indices(n: int, max_tests: int | None) -> list[int]:
    """Evenly-spaced test indices spanning [0, n-1] inclusive (endpoints kept).

    Deterministic in ``n`` alone, so a prompt's reward is a stable function and
    all completions to that prompt are graded on the same cases. Spreads across
    the suite rather than taking a contiguous prefix, so an easy->hard ordering
    can't be gamed by solving only the easy end.
    """
    if max_tests is None or max_tests >= n:
        return list(range(n))
    if max_tests <= 1:
        return [0]
    return sorted({round(i * (n - 1) / (max_tests - 1)) for i in range(max_tests)})


def _map_checks(checks: list, max_workers: int | None = None) -> list[bool]:
    """Run zero-arg test-case callables, in parallel when there are several.

    ``max_workers`` defaults to ``_MAX_WORKERS``; the training reward threads the
    run config's worker count through here so grading concurrency is tunable.
    """
    workers = _MAX_WORKERS if max_workers is None else max_workers
    if len(checks) <= 1 or workers <= 1:
        return [check() for check in checks]
    with ThreadPoolExecutor(max_workers=min(workers, len(checks))) as pool:
        return list(pool.map(lambda check: check(), checks))


def extract_code(completion: str) -> str:
    """Pull the last fenced code block from a completion, else the raw text."""
    blocks = _CODE_FENCE.findall(completion or "")
    return blocks[-1].strip() if blocks else (completion or "").strip()


# Completion outcomes. ``TRUNCATED`` is deliberately NOT a score: a completion cut
# off by the generation cap is not evidence about quality, because we never saw the
# answer it was going to give.
#
# It used to be scored 0.0, identically to a wrong answer. ``_CODE_FENCE`` requires
# a CLOSING marker, so a completion truncated mid-code-block matched nothing, fell
# through to the raw-text branch, got executed as Python, raised SyntaxError and
# scored exactly 0.0. With ~49% of held-out generations hitting the cap, that made
# every reward and every eval metric a product of two things -- "did it terminate"
# and "was it correct" -- and training moved the first one.
#
# Detection here is a heuristic on the code fence, because the GRPO reward callable
# never sees generation metadata. Callers that DO have ``finish_reason`` (DPO pair
# building, the eval harnesses) should pass it: it is authoritative, and it also
# catches the case where the cap was hit before any fence was opened.
COMPLETION_OK = "ok"
COMPLETION_TRUNCATED = "truncated"
COMPLETION_EMPTY = "empty"

_FENCE_OPENER = re.compile(r"```[a-zA-Z0-9_+-]*\n")


def extract_code_with_status(completion: str, finish_reason: str | None = None) -> tuple[str, str]:
    """Extract code and classify the completion as ok / truncated / empty.

    ``finish_reason`` is the generator's own verdict ("length" means the cap was
    hit) and wins when supplied. Without it, an odd number of fence markers is the
    signature of a block that was opened and never closed.

    The returned code is still the best-effort extraction even when truncated, so a
    caller that wants to grade a partial answer anyway can; it just has to opt in.
    """
    text = completion or ""
    if not text.strip():
        return "", COMPLETION_EMPTY

    blocks = _CODE_FENCE.findall(text)
    openers = list(_FENCE_OPENER.finditer(text))
    unclosed = bool(openers) and text.count("```") % 2 == 1

    if unclosed:
        code = text[openers[-1].end():].strip()
    elif blocks:
        code = blocks[-1].strip()
    else:
        code = text.strip()

    if finish_reason == "length" or unclosed:
        return code, COMPLETION_TRUNCATED
    if not code:
        return "", COMPLETION_EMPTY
    return code, COMPLETION_OK


def _completion_text(completion) -> str:
    """TRL hands completions as str (plain) or [{'role','content'}] (chat)."""
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]


def verifier_sandbox_available() -> bool:
    """Whether the current host has the sandbox used by formal local builds."""
    return sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file()


def _sandbox_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _sandbox_profile(*, workdir: Path, python_executable: Path) -> str:
    python_prefix = python_executable.parent.parent
    return f"""(version 1)
(allow default)
(deny network*)
(deny process-fork)
(deny file-write*)
(allow file-write*
    (subpath "{_sandbox_string(str(workdir))}")
    (literal "/dev/null"))
(deny file-read* (subpath "{_sandbox_string(str(Path.home()))}"))
(allow file-read*
    (subpath "{_sandbox_string(str(python_prefix))}")
    (subpath "{_sandbox_string(str(workdir))}"))
"""


def _verifier_environment(workdir: Path) -> dict[str, str]:
    """Minimal deterministic environment; deliberately excludes all secrets."""
    return {
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _resource_runner(*, script: Path, timeout: float) -> str:
    cpu_seconds = max(1, math.ceil(timeout) + 1)
    return f"""import resource
import runpy
import sys

def cap(which, value):
    _, hard = resource.getrlimit(which)
    bounded = value if hard == resource.RLIM_INFINITY else min(value, hard)
    resource.setrlimit(which, (bounded, hard))
    resource.setrlimit(which, (bounded, bounded))

cap(resource.RLIMIT_CPU, {cpu_seconds})
cap(resource.RLIMIT_FSIZE, {_VERIFIER_FILE_BYTES})
cap(resource.RLIMIT_NOFILE, {_VERIFIER_OPEN_FILES})
sys.argv = [{str(script)!r}]
runpy.run_path({str(script)!r}, run_name="__main__")
"""


def _run(
    code: str,
    stdin: str | None,
    timeout: float,
    require_sandbox: bool = False,
) -> tuple[int, str]:
    """Execute ``code`` as a script in a fresh process; return (returncode, stdout)."""
    with tempfile.TemporaryDirectory(prefix="drift-verify-") as raw_workdir:
        workdir = Path(raw_workdir).resolve()
        script = workdir / "solution.py"
        runner = workdir / "runner.py"
        stdout_path = workdir / "stdout.txt"
        script.write_text(code)
        runner.write_text(_resource_runner(script=script, timeout=timeout))
        python_executable = Path(sys.executable).resolve()
        command = [str(python_executable), "-I", "-S", str(runner)]
        if require_sandbox:
            if not verifier_sandbox_available():
                raise RuntimeError(
                    "Formal verifier sandbox is unavailable; refusing to execute generated code"
                )
            profile = workdir / "verifier.sb"
            profile.write_text(
                _sandbox_profile(
                    workdir=workdir,
                    python_executable=python_executable,
                )
            )
            command = ["/usr/bin/sandbox-exec", "-f", str(profile), *command]
        try:
            with stdout_path.open("w") as stdout:
                result = subprocess.run(
                    command,
                    input=stdin,
                    stdout=stdout,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=timeout,
                    cwd=workdir,
                    env=_verifier_environment(workdir),
                )
            return result.returncode, stdout_path.read_text(errors="replace")
        except subprocess.TimeoutExpired:
            return -1, ""


def _normalize(text: str) -> str:
    """Canonicalize competitive-judge output: unify newlines, rstrip each line,
    drop trailing blank lines. Avoids false negatives from CRLF / trailing space."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _outputs_match(actual: str, expected: str, comparison: str) -> bool:
    if comparison == "tokens":
        return actual.split() == expected.split()
    if comparison == "exact_lines":
        return _normalize(actual) == _normalize(expected)
    if comparison == "case_insensitive_tokens":
        return [token.casefold() for token in actual.split()] == [
            token.casefold() for token in expected.split()
        ]
    if comparison == "float_tokens":
        actual_tokens = actual.split()
        expected_tokens = expected.split()
        if len(actual_tokens) != len(expected_tokens):
            return False
        for actual_token, expected_token in zip(actual_tokens, expected_tokens):
            try:
                actual_value = float(actual_token)
                expected_value = float(expected_token)
            except ValueError:
                if actual_token != expected_token:
                    return False
                continue
            if not (
                math.isfinite(actual_value)
                and math.isfinite(expected_value)
                and math.isclose(
                    actual_value,
                    expected_value,
                    rel_tol=FLOAT_TOKEN_REL_TOL,
                    abs_tol=FLOAT_TOKEN_ABS_TOL,
                )
            ):
                return False
        return True
    raise ValueError(f"Unsupported stdout comparison: {comparison!r}")


def _io_case_passes(
    code: str,
    stdin: str,
    expected: str,
    timeout: float,
    comparison: str,
    require_sandbox: bool,
) -> bool:
    rc, stdout = _run(code, stdin, timeout, require_sandbox)
    return rc == 0 and _outputs_match(stdout, expected, comparison)


def _unit_case_passes(
    code: str, test: str, timeout: float, require_sandbox: bool
) -> bool:
    rc, _ = _run(f"{code}\n\n{test}\n", None, timeout, require_sandbox)
    return rc == 0


def _case_checks(
    completion: str,
    verifier: dict,
    timeout: float,
    max_tests: int | None = None,
    require_sandbox: bool = False,
) -> list:
    """One zero-arg callable per test case of a ``coding_task_v1`` verifier.

    ``max_tests`` caps the cases via ``_subsample_indices`` (a deterministic,
    spread-out subset) BEFORE building callables, so the training reward's cheap
    budget flows through the same test-case-level parallel path as full grading.
    """
    code = extract_code(completion) if completion else ""
    verifier_type = verifier.get("type")
    if verifier_type in {"io_tests", "reference_io_tests"}:
        inputs, outputs = verifier["test_inputs"], verifier["test_outputs"]
        comparison = verifier.get("comparison", "tokens")
        if not code or not inputs:
            return []
        idx = _subsample_indices(len(inputs), max_tests)
        return [
            partial(
                _io_case_passes,
                code,
                inputs[i],
                outputs[i],
                timeout,
                comparison,
                require_sandbox,
            )
            for i in idx
        ]
    if verifier_type == "unit_tests":
        tests = verifier["tests"]
        if not code or not tests:
            return []
        idx = _subsample_indices(len(tests), max_tests)
        return [
            partial(_unit_case_passes, code, tests[i], timeout, require_sandbox)
            for i in idx
        ]
    raise ValueError(f"Unsupported verifier type: {verifier_type!r}")


def run_io_tests(
    code: str,
    inputs: list[str],
    outputs: list[str],
    timeout: float = 10.0,
    max_tests: int | None = None,
) -> float:
    """stdin/stdout verifier (Nemotron / CodeContests). Pass fraction over cases.

    ``max_tests`` caps how many cases are executed (``None`` = all). During
    training this is set low (see make_coding_reward) so grading is cheap; eval
    and DPO-pair building leave it ``None`` for full-fidelity pass rates.
    """
    verifier = {"type": "io_tests", "test_inputs": inputs, "test_outputs": outputs}
    return run_verifier(code, verifier, timeout, max_tests)


def run_unit_tests(
    code: str,
    tests: list[str],
    timeout: float = 10.0,
    max_tests: int | None = None,
) -> float:
    """assert-style verifier (MBPP / AceCode). Pass fraction over snippets."""
    return run_verifier(code, {"type": "unit_tests", "tests": tests}, timeout, max_tests)


def run_verifier(
    code: str,
    verifier: dict,
    timeout: float = 10.0,
    max_tests: int | None = None,
    require_sandbox: bool = False,
) -> float:
    """Dispatch a completion against a canonical ``coding_task_v1`` verifier."""
    checks = _case_checks(code, verifier, timeout, max_tests, require_sandbox)
    if not checks:
        return 0.0
    results = _map_checks(
        checks,
        max_workers=_FORMAL_SANDBOX_WORKERS if require_sandbox else None,
    )
    return sum(results) / len(checks)


def coding_reward(
    completions,
    verifier=None,
    test_inputs=None,
    test_outputs=None,
    completion_ids=None,
    eos_token_ids=None,
    timeout: float = 10.0,
    max_tests: int | None = None,
    num_workers: int | None = None,
    **kwargs,
) -> list[float | None]:
    """GRPO/RLOO-compatible reward function.

    TRL passes ``completions`` plus every dataset column as keyword arguments;
    canonical data uses a ``verifier`` column. Legacy top-level
    ``test_inputs``/``test_outputs`` are still accepted during migration.

    ``max_tests`` caps tests per completion (the shared training budget).
    ``num_workers`` sizes the grading pool; every (completion x test-case) pair
    across the whole rollout batch is flattened into one pool so the batch grades
    concurrently, not one completion at a time. Both are set by make_coding_reward
    from the run config; the bare defaults preserve the original full-suite,
    default-pool behavior.
    """
    from scripts.rl_training.completion_semantics import completion_ids_terminated

    if verifier is None:
        if test_inputs is None or test_outputs is None:
            raise ValueError("coding_reward requires either verifier or test_inputs/test_outputs")
        verifier = [
            {"type": "io_tests", "test_inputs": ti, "test_outputs": to}
            for ti, to in zip(test_inputs, test_outputs)
        ]
    if completion_ids is not None:
        if len(completion_ids) != len(completions):
            raise ValueError("completion_ids must have one row per completion")
        if not eos_token_ids:
            raise ValueError(
                "eos_token_ids are required when completion_ids are supplied; "
                "do not guess whether a capped rollout terminated"
            )
        terminated = [
            completion_ids_terminated(ids, eos_token_ids) for ids in completion_ids
        ]
    else:
        # Offline/legacy callers have no token metadata. Preserve their historic
        # behavior; online GRPO always supplies completion_ids and must bind EOS.
        terminated = [True] * len(completions)

    # Flatten every (completion, test case) pair into one worker pool so the
    # whole rollout batch verifies concurrently, not one completion at a time.
    # max_tests is applied per completion inside _case_checks (spread subset).
    per_completion = [
        _case_checks(_completion_text(c), v, timeout, max_tests) if is_finished else []
        for c, v, is_finished in zip(completions, verifier, terminated)
    ]
    flat_results = iter(_map_checks([check for checks in per_completion for check in checks], num_workers))
    rewards = []
    for checks, is_finished in zip(per_completion, terminated):
        results = [next(flat_results) for _ in checks]
        rewards.append(
            (sum(results) / len(checks) if checks else 0.0) if is_finished else None
        )
    return rewards


def make_coding_reward(
    timeout: float = REWARD_TIMEOUT,
    max_tests: int | None = TRAIN_REWARD_MAX_TESTS,
    num_workers: int | None = None,
    eos_token_ids: tuple[int, ...] | list[int] | None = None,
) -> callable:
    """Bind training-time grading knobs onto ``coding_reward`` for TRL.

    TRL calls the reward function with a fixed signature (no timeout / cap args),
    so the training config's cheap-grading settings are injected here instead.
    ``num_workers`` of ``None`` or ``<=0`` resolves to ``os.cpu_count()``. The
    returned callable keeps ``__name__ == 'coding_reward'`` because TRL uses it
    to name the reward's logged metric column.
    """
    resolved_workers = num_workers if num_workers and num_workers > 0 else os.cpu_count()

    def coding_reward_fn(completions, **kwargs):
        completion_ids = kwargs.get("completion_ids")
        if completion_ids is not None:
            kwargs["eos_token_ids"] = eos_token_ids
        return coding_reward(
            completions,
            timeout=timeout,
            max_tests=max_tests,
            num_workers=resolved_workers,
            **kwargs,
        )

    coding_reward_fn.__name__ = "coding_reward"
    return coding_reward_fn


def score_completions(
    completions: list[str],
    verifier: dict,
    timeout: float = 10.0,
    max_tests: int | None = None,
    num_workers: int | None = None,
) -> list[tuple[float, str]]:
    """Score candidate solutions with a canonical verifier.

    ``max_tests`` must match the training reward's budget when building DPO pairs
    / PPO reward-model data, so every arm shares one objective (build_dpo_pairs.py
    passes it). Left ``None`` (full suite) for eval/measurement.

    Every (completion x test-case) pair is flattened into one worker pool so the
    whole candidate set grades concurrently, not one completion at a time — this
    fully uses VERIFIER_MAX_WORKERS during DPO pair generation. ``num_workers``
    of ``None`` resolves to ``_MAX_WORKERS`` (the env-configured default).
    """
    texts = [_completion_text(c) for c in completions]
    per_completion = [_case_checks(t, verifier, timeout, max_tests) for t in texts]
    flat_results = iter(_map_checks([check for checks in per_completion for check in checks], num_workers))
    scored = []
    for text, checks in zip(texts, per_completion):
        results = [next(flat_results) for _ in checks]
        scored.append((sum(results) / len(checks) if checks else 0.0, text))
    return scored


def score_candidates(
    completions,
    verifier: dict,
    timeout: float = 10.0,
    max_tests: int | None = None,
    num_workers: int | None = None,
    finish_reasons: list[str | None] | None = None,
    require_sandbox: bool = False,
) -> list[dict]:
    """Score candidates, keeping "did not finish" separate from "scored zero".

    Returns one dict per candidate with ``reward``, ``text`` and ``status``.
    ``reward`` is ``None`` when the candidate is truncated or empty: those are not
    evidence and must never be averaged into a mean, used as a DPO ``rejected``, or
    fed to a policy-gradient advantage. ``score_completions`` is the older
    truncation-blind API and is kept for callers that predate this distinction.

    ``finish_reasons`` is the generator's verdict per candidate when available and
    is strictly better than the fence heuristic -- pass it whenever you have it.
    """
    texts = [_completion_text(c) for c in completions]
    reasons = list(finish_reasons or [None] * len(texts))
    if len(reasons) != len(texts):
        raise ValueError(
            f"finish_reasons has {len(reasons)} entries for {len(texts)} completions"
        )

    statuses = [extract_code_with_status(t, r)[1] for t, r in zip(texts, reasons)]
    gradable = [status == COMPLETION_OK for status in statuses]

    per_completion = [
        _case_checks(text, verifier, timeout, max_tests, require_sandbox) if ok else []
        for text, ok in zip(texts, gradable)
    ]
    flat_results = iter(
        _map_checks(
            [check for checks in per_completion for check in checks],
            _FORMAL_SANDBOX_WORKERS if require_sandbox else num_workers,
        )
    )
    scored = []
    for text, status, ok, checks in zip(texts, statuses, gradable, per_completion):
        results = [next(flat_results) for _ in checks]
        # A gradable completion with no runnable checks really did score zero (the
        # verifier had no cases, or extraction produced nothing runnable). Only the
        # truncated/empty statuses are withheld.
        reward = (sum(results) / len(checks) if checks else 0.0) if ok else None
        scored.append({"reward": reward, "text": text, "status": status})
    return scored


def build_preference_pairs(
    prompt: str,
    completions: list[str],
    verifier: dict,
    timeout: float = 10.0,
    margin: float = 0.5,
    max_tests: int | None = None,
) -> dict | None:
    """Turn scored candidates for one prompt into a DPO ``(chosen, rejected)`` row.

    Scores every candidate with the same I/O verifier, pairs best vs worst, and
    keeps the pair only when ``best - worst >= margin`` (default 0.5, per the
    study design) so ties/near-ties don't inject label noise into DPO.
    """
    scored = score_completions(completions, verifier, timeout, max_tests=max_tests)
    return build_preference_pair_from_scored(prompt, scored, margin=margin)


def build_preference_pair_from_scored(
    prompt: str,
    scored_completions: list[tuple[float, str]],
    margin: float = 0.5,
) -> dict | None:
    """Build a DPO pair from already-scored ``(score, completion)`` candidates.

    Truncation-blind: a candidate cut off at the generation cap arrives here with a
    0.0 score and becomes ``rejected``, which teaches DPO "finished beats
    unfinished" instead of "correct beats incorrect". Prefer
    ``build_preference_pair_from_candidates``.
    """
    scored = sorted(scored_completions)
    if not scored:
        return None
    worst_score, worst = scored[0]
    best_score, best = scored[-1]
    if best_score - worst_score < margin:
        return None
    return {
        "prompt": prompt,
        "chosen": _completion_text(best),
        "rejected": _completion_text(worst),
        "chosen_reward": best_score,
        "rejected_reward": worst_score,
    }


def build_preference_pair_from_candidates(
    prompt: str,
    candidates: list[dict],
    margin: float = 0.5,
) -> tuple[dict | None, dict]:
    """Build a DPO pair from ``score_candidates`` output, ignoring truncated ones.

    Returns ``(pair_or_none, audit)``. Both sides are drawn only from candidates
    that actually finished, so the contrast DPO learns is about correctness rather
    than termination. On the original cohorts 93% of ``rejected`` completions
    scored exactly 0.0 and most of those were simply cut off at 1024 tokens.

    Excluding truncated candidates costs pairs -- a prompt where nothing finished
    yields nothing -- which is the correct outcome: there was no evidence there.
    The audit counts make that loss visible instead of silent.
    """
    gradable = [c for c in candidates if c.get("reward") is not None]
    audit = {
        "n_candidates": len(candidates),
        "n_gradable": len(gradable),
        "n_truncated": sum(1 for c in candidates if c.get("status") == COMPLETION_TRUNCATED),
        "n_empty": sum(1 for c in candidates if c.get("status") == COMPLETION_EMPTY),
        "pair_eligible": False,
    }
    if len(gradable) < 2:
        return None, audit

    ordered = sorted(gradable, key=lambda c: (c["reward"], c["text"]))
    worst, best = ordered[0], ordered[-1]
    audit["margin"] = best["reward"] - worst["reward"]
    if audit["margin"] < margin:
        return None, audit

    audit["pair_eligible"] = True
    return (
        {
            "prompt": prompt,
            "chosen": best["text"],
            "rejected": worst["text"],
            "chosen_reward": best["reward"],
            "rejected_reward": worst["reward"],
        },
        audit,
    )
