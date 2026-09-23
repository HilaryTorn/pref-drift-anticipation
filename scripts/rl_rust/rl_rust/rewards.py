"""Multi-language verifiable coding reward. Same objective, more executors.

r(x, y) = fraction of tests passed, in [0, 1] -- identical to
``rl_training/rewards.py``. The ONLY thing that changes here is what runs the
program. That is deliberate: if the Rust arm and the Python arm disagree about
what counts as correct output, the language comparison is confounded and the
whole run is wasted.

So the pure semantics -- output normalization, the four comparison modes, the
deterministic spread-out test subsample, truncation classification -- are
IMPORTED from ``rl_training.rewards`` rather than copied. A copy would drift.
This module owns exactly one thing: dispatching source text to an executor
(see ``languages.py``), and threading a ``language`` argument through the call
chain that already existed.

Nothing in ``rl_training/`` is modified. The dependency is one-way and
read-only, so the Python pipeline is untouched and still runs on its own code.

Drop-in usage in train_rl.py -- change the import, nothing else:

    from rl_rust.rewards import make_coding_reward
    reward_fn = make_coding_reward(language="rust", eos_token_ids=eos)
"""

from __future__ import annotations

import os
import re
from functools import partial

# One source of truth for the reward SEMANTICS. Do not reimplement these.
from rl_training.rewards import (  # noqa: F401  (re-exported for callers)
    FLOAT_TOKEN_ABS_TOL,
    FLOAT_TOKEN_REL_TOL,
    REWARD_TIMEOUT,
    TRAIN_REWARD_MAX_TESTS,
    COMPLETION_EMPTY,
    COMPLETION_OK,
    COMPLETION_TRUNCATED,
    _completion_text,
    _map_checks,
    _normalize,
    _outputs_match,
    _subsample_indices,
    extract_code_with_status,
)

from scripts.rl_rust.rl_rust.languages import (
    BuildError,
    LanguageSpec,
    build,
    clear_build_cache,
    get_language,
    run_artifact,
    rust_toolchain_available,
)

DEFAULT_LANGUAGE = "rust"

_TAGGED_FENCE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def extract_code_for_language(
    completion: str,
    spec: LanguageSpec,
    finish_reason: str | None = None,
) -> tuple[str, str]:
    """Extract a code block, preferring one tagged with the target language.

    For Python this defers entirely to the original ``extract_code_with_status``
    so the Python arm grades byte-identically to the existing pipeline.

    For other languages the last *matching-tag* block wins, falling back to the
    original last-block rule when no tag matches. This matters because a Rust
    completion often ends with an untagged block showing expected output, and
    the plain last-block rule would grade that instead of the program.
    """
    if not spec.needs_build and spec.name == "python":
        return extract_code_with_status(completion, finish_reason)

    code, status = extract_code_with_status(completion, finish_reason)
    if status == COMPLETION_EMPTY:
        return code, status

    # A truncated completion keeps the original best-effort extraction: the
    # trailing block is the one that got cut off, and tag-matching an earlier
    # complete block would silently grade a different program.
    if status == COMPLETION_TRUNCATED:
        return code, status

    matches = [
        body.strip()
        for tag, body in _TAGGED_FENCE.findall(completion or "")
        if tag.lower() in spec.fence_tags
    ]
    if matches:
        return matches[-1], COMPLETION_OK
    return code, status


def _io_case_passes(
    artifact,
    spec: LanguageSpec,
    stdin: str,
    expected: str,
    timeout: float,
    comparison: str,
) -> bool:
    rc, stdout = run_artifact(artifact, spec, stdin, timeout)
    return rc == 0 and _outputs_match(stdout, expected, comparison)


def _failed_case() -> bool:
    """A case that cannot run (compile error). Scored 0.0, like a wrong answer."""
    return False


def _case_checks(
    completion: str,
    verifier: dict,
    timeout: float,
    max_tests: int | None = None,
    spec: LanguageSpec | None = None,
) -> list:
    """One zero-arg callable per test case, with the program built ONCE.

    The build happens here rather than inside each case callable, so a
    completion costs one ``rustc`` invocation instead of ``max_tests`` of them.
    A program that does not compile yields callables that all return False, so
    it scores 0.0 through the same path as a wrong answer -- no special-casing
    downstream, and the denominator stays the number of tests.
    """
    spec = spec or get_language(DEFAULT_LANGUAGE)
    code, _status = extract_code_for_language(completion, spec) if completion else ("", COMPLETION_EMPTY)

    verifier_type = verifier.get("type")
    if verifier_type not in {"io_tests", "reference_io_tests"}:
        raise ValueError(
            f"unsupported verifier type for {spec.name}: {verifier_type!r}. "
            "Only stdin/stdout suites are language-agnostic; assert-style "
            "unit_tests are Python-only by construction."
        )

    inputs, outputs = verifier["test_inputs"], verifier["test_outputs"]
    comparison = verifier.get("comparison", "tokens")
    if not code or not inputs:
        return []

    idx = _subsample_indices(len(inputs), max_tests)
    try:
        artifact = build(code, spec)
    except BuildError:
        return [_failed_case for _ in idx]

    return [
        partial(
            _io_case_passes,
            artifact,
            spec,
            inputs[i],
            outputs[i],
            timeout,
            comparison,
        )
        for i in idx
    ]


def run_verifier(
    code: str,
    verifier: dict,
    timeout: float = 10.0,
    max_tests: int | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> float:
    """Dispatch a completion against a canonical ``coding_task_v1`` verifier."""
    spec = get_language(language)
    checks = _case_checks(code, verifier, timeout, max_tests, spec)
    if not checks:
        return 0.0
    results = _map_checks(checks)
    return sum(results) / len(checks)


def run_io_tests(
    code: str,
    inputs: list[str],
    outputs: list[str],
    timeout: float = 10.0,
    max_tests: int | None = None,
    language: str = DEFAULT_LANGUAGE,
    comparison: str = "tokens",
) -> float:
    """stdin/stdout verifier. Pass fraction over cases."""
    verifier = {
        "type": "io_tests",
        "test_inputs": inputs,
        "test_outputs": outputs,
        "comparison": comparison,
    }
    return run_verifier(code, verifier, timeout, max_tests, language)


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
    language: str = DEFAULT_LANGUAGE,
    **kwargs,
) -> list[float | None]:
    """GRPO/RLOO-compatible reward function, parameterized by target language.

    Semantics match ``rl_training.rewards.coding_reward`` exactly, including the
    truncation contract: a rollout that hit the generation cap without
    terminating returns ``None`` (not 0.0), so DAPO overlong filtering can drop
    it instead of teaching the policy to stop early.
    """
    from rl_training.completion_semantics import completion_ids_terminated

    spec = get_language(language)

    if verifier is None:
        if test_inputs is None or test_outputs is None:
            raise ValueError(
                "coding_reward requires either verifier or test_inputs/test_outputs"
            )
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
        terminated = [True] * len(completions)

    # Builds happen here, one per distinct program, before the test-case pool is
    # flattened. Compiling inside the pool would serialize on the cache lock.
    per_completion = [
        _case_checks(_completion_text(c), v, timeout, max_tests, spec) if is_finished else []
        for c, v, is_finished in zip(completions, verifier, terminated)
    ]
    flat_results = iter(
        _map_checks([check for checks in per_completion for check in checks], num_workers)
    )
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
    language: str = DEFAULT_LANGUAGE,
) -> callable:
    """Bind training-time grading knobs onto ``coding_reward`` for TRL.

    Keeps ``__name__ == 'coding_reward'`` because TRL uses it to name the logged
    metric column -- so ``rewards/coding_reward/mean`` stays comparable to the
    Python arms' trainer_state without any downstream analysis change.
    """
    spec = get_language(language)
    if spec.needs_build and not rust_toolchain_available():
        raise RuntimeError(
            f"language {language!r} needs a compiler but rustc is not on PATH. "
            "Refusing to start: every rollout would score 0.0 and the run would "
            "look like a real null."
        )

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
            language=language,
            **kwargs,
        )

    coding_reward_fn.__name__ = "coding_reward"
    return coding_reward_fn


def score_candidates(
    completions,
    verifier: dict,
    timeout: float = 10.0,
    max_tests: int | None = None,
    num_workers: int | None = None,
    finish_reasons: list[str | None] | None = None,
    require_sandbox: bool = False,
    language: str = DEFAULT_LANGUAGE,
) -> list[dict]:
    """Truncation-aware candidate scoring, parameterized by target language.

    Mirrors ``rl_training.rewards.score_candidates`` exactly -- one dict per
    candidate with ``reward``, ``text`` and ``status``, where ``reward`` is
    ``None`` for truncated/empty candidates so they can never become a DPO
    ``rejected`` -- but grades through the language executor. This is what
    ``build_dpo_pairs.py`` calls, so the Rust DPO pair builder rebinds the
    upstream attribute to this function (see rl-rust/build_rust_dpo_pairs.py).
    On Rust this exclusion is load-bearing: 15-30% of candidates truncate, and
    the truncation-blind path would teach "finished beats unfinished".

    ``require_sandbox`` is the upstream formal-sandbox flag; that path is
    Python-only, so asking for it here is a configuration error, not a fallback.
    """
    if require_sandbox:
        raise ValueError(
            "require_sandbox grading is Python-only; the multi-language "
            "executor has no sandboxed worker pool"
        )
    spec = get_language(language)
    texts = [_completion_text(c) for c in completions]
    reasons = list(finish_reasons or [None] * len(texts))
    if len(reasons) != len(texts):
        raise ValueError(
            f"finish_reasons has {len(reasons)} entries for {len(texts)} completions"
        )

    statuses = [
        extract_code_for_language(t, spec, r)[1] for t, r in zip(texts, reasons)
    ]
    gradable = [status == COMPLETION_OK for status in statuses]

    per_completion = [
        _case_checks(text, verifier, timeout, max_tests, spec) if ok else []
        for text, ok in zip(texts, gradable)
    ]
    flat_results = iter(
        _map_checks(
            [check for checks in per_completion for check in checks], num_workers
        )
    )
    scored = []
    for text, status, ok, checks in zip(texts, statuses, gradable, per_completion):
        results = [next(flat_results) for _ in checks]
        # Same contract as upstream: a gradable completion with no runnable
        # checks really did score zero; only truncated/empty are withheld.
        reward = (sum(results) / len(checks) if checks else 0.0) if ok else None
        scored.append({"reward": reward, "text": text, "status": status})
    return scored


def score_completions(
    completions: list[str],
    verifier: dict,
    timeout: float = 10.0,
    max_tests: int | None = None,
    num_workers: int | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> list[tuple[float, str]]:
    """Score candidate solutions. Returns (reward, extracted_code) per candidate.

    ``max_tests`` must match the training reward's budget when building DPO
    pairs or PPO reward-model data, so every arm shares one objective.
    """
    spec = get_language(language)
    texts = [_completion_text(c) for c in completions]
    per_completion = [_case_checks(t, verifier, timeout, max_tests, spec) for t in texts]
    flat_results = iter(
        _map_checks([check for checks in per_completion for check in checks], num_workers)
    )
    scored = []
    for text, checks in zip(texts, per_completion):
        results = [next(flat_results) for _ in checks]
        reward = sum(results) / len(checks) if checks else 0.0
        code, _ = extract_code_for_language(text, spec)
        scored.append((reward, code))
    return scored


__all__ = [
    "BuildError",
    "DEFAULT_LANGUAGE",
    "REWARD_TIMEOUT",
    "TRAIN_REWARD_MAX_TESTS",
    "clear_build_cache",
    "coding_reward",
    "extract_code_for_language",
    "get_language",
    "make_coding_reward",
    "run_io_tests",
    "run_verifier",
    "rust_toolchain_available",
    "score_candidates",
    "score_completions",
]
