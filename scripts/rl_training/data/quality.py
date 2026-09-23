"""Paper-facing quality policy for competitive-programming records.

The Nemotron aggregate exposes fixed input/output pairs but not the semantic
checkers required by constructive, floating-point, or interactive problems.
Those tasks must not enter a formal exact-output cohort.  This module keeps the
policy deterministic, auditable, and shared by preparation and paid SFT builds.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any


QUALITY_POLICY = "coding_io_deterministic_v10"
FORMAL_MIN_UNIQUE_TESTS = 12
EXCLUDED_NATIVE_SOURCES = {"hackerearth"}

NEMOTRON_PROMPT_PREFIX = (
    "You are a helpful and harmless assistant. You should think step-by-step before "
    "responding to the instruction below.\n\n"
    "Please use python programming language only.\n\n"
    "You must use ```python for just the final solution code block with the following format:\n"
    "```python\n"
    "# Your code here\n"
    "```\n\n"
)

_NON_UNIQUE_OUTPUT = re.compile(
    r"if there (?:are|is) (?:multiple|more than one)"
    r"|more than one (?:solution|answer)"
    r"|multiple (?:valid )?(?:solutions|answers)"
    r"|if there (?:are|is) (?:many|several) (?:valid )?(?:solutions|answers|outputs).{0,100}\bany\b"
    r"|(?:output|print|find|choose).{0,80}\bany\b"
    r"|\bany (?:valid |possible )?(?:answer|solution|permutation|order|pair|sequence)\b"
    r"|\boutput an arbitrary schedule\b"
    r"|\bprint all .{0,120}\bin arbitrary order\b"
    r"|(?:-----Output-----|(?:^|\n)Output\b).{0,500}\bin (?:an )?arbitrary order\b"
    r"|\bif there (?:is|are) no valid "
    r"(?:arrangement|construction|permutation|ordering|sequence|assignment|schedule|path)\b"
    r".{0,180}\botherwise\b"
    r"|\bif it is possible find how to do that\b"
    r"|allowed to output nothing|(?:output|print) nothing",
    re.IGNORECASE | re.DOTALL,
)
_FLOAT_TOLERANCE = re.compile(
    r"absolute or relative error|relative or absolute error"
    r"|(?:absolute|relative) error(?: of)? (?:at most|up to)"
    r"|error (?:does not exceed|less than)"
    r"|(?:answer|output) will be considered (?:valid|correct|accepted) if it differs"
    r"|(?:at least|no less than)\s+\d+\s+digits?\s+after\s+(?:the\s+)?decimal",
    re.IGNORECASE,
)
_CASE_INSENSITIVE_OUTPUT = re.compile(
    r"(?:yes|no).{0,100}any (?:case|capitalization)"
    r"|any (?:case|capitalization).{0,100}(?:yes|no)"
    r"|you can (?:print|output) each letter in (?:any|arbitrary) case"
    r"|(?:print|output).{0,100}(?:yes|no).{0,60}case[- ]insensitive"
    r"|print either lowercase or uppercase letters in the answers",
    re.IGNORECASE | re.DOTALL,
)
_INTERACTIVE = re.compile(
    r"\binteractive\b|flush (?:the )?output"
    r"|search query equals to the whole blackboard",
    re.IGNORECASE,
)
_HACKEREARTH_TRAILING_SAMPLE_MARKER = re.compile(
    r"\n[ \t]*\n[ \t]*[A-Z@]{5,7}[ \t]*\Z"
)
_STARTS_WITH_INPUT_WITHOUT_STATEMENT = re.compile(
    r"^\s*(?:example\s*)?(?:-{2,}\s*)?(?:sample\s+)?input(?:\s*-+)?\b",
    re.IGNORECASE,
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def strip_nemotron_prompt_wrapper(prompt: str) -> tuple[str, bool]:
    """Remove only the exact known upstream generation wrapper."""
    prompt = prompt.strip()
    if prompt.startswith(NEMOTRON_PROMPT_PREFIX):
        return prompt[len(NEMOTRON_PROMPT_PREFIX) :].strip(), True
    return prompt, False


def prompt_exclusion_reasons(prompt: str) -> list[str]:
    checks = (
        ("non_unique_output", _NON_UNIQUE_OUTPUT),
        ("floating_point_tolerance", _FLOAT_TOLERANCE),
        ("case_insensitive_output", _CASE_INSENSITIVE_OUTPUT),
        ("interactive", _INTERACTIVE),
    )
    reasons = [name for name, pattern in checks if pattern.search(prompt)]
    if _STARTS_WITH_INPUT_WITHOUT_STATEMENT.match(prompt):
        reasons.append("missing_problem_statement")
    return reasons


def canonicalize_coding_row(
    row: dict[str, Any],
    *,
    min_unique_tests: int = FORMAL_MIN_UNIQUE_TESTS,
) -> tuple[dict[str, Any], list[str]]:
    """Return a canonical row and deterministic formal-cohort exclusions.

    Duplicate test inputs with identical outputs are removed. Conflicting
    outputs for one input are excluded because a deterministic stdout verifier
    cannot represent them.
    """
    cleaned = deepcopy(row)
    original_prompt = str(cleaned.get("prompt", ""))
    prompt, wrapper_removed = strip_nemotron_prompt_wrapper(original_prompt)
    cleaned["prompt"] = prompt

    reasons = prompt_exclusion_reasons(prompt)
    verifier = cleaned.get("verifier")
    if not isinstance(verifier, dict) or verifier.get("type") not in {
        "io_tests",
        "reference_io_tests",
    }:
        reasons.append("unsupported_verifier")
        return cleaned, sorted(set(reasons))

    inputs = verifier.get("test_inputs")
    outputs = verifier.get("test_outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        reasons.append("invalid_test_lists")
        return cleaned, sorted(set(reasons))

    metadata = cleaned.setdefault("metadata", {})
    if metadata.get("native_source") in EXCLUDED_NATIVE_SOURCES:
        reasons.append("unreliable_native_source")
    if metadata.get("native_source") == "hackerearth" and any(
        _HACKEREARTH_TRAILING_SAMPLE_MARKER.search(str(stdin)) for stdin in inputs
    ):
        reasons.append("trailing_sample_marker_noise")

    unique_inputs: list[str] = []
    unique_outputs: list[str] = []
    seen: dict[str, str] = {}
    for stdin, expected in zip(inputs, outputs):
        stdin = str(stdin)
        expected = str(expected)
        if not stdin.strip() or not expected.strip():
            reasons.append("blank_test_io")
        if stdin in seen:
            if seen[stdin] != expected:
                reasons.append("conflicting_test_outputs")
            continue
        seen[stdin] = expected
        unique_inputs.append(stdin)
        unique_outputs.append(expected)

    if len(unique_inputs) < min_unique_tests:
        reasons.append("insufficient_unique_tests")

    verifier["test_inputs"] = unique_inputs
    verifier["test_outputs"] = unique_outputs
    verifier["comparison"] = "tokens"

    upstream_wrapper_removed = bool(metadata.get("source_prompt_wrapper_removed"))
    metadata["quality"] = {
        "policy": QUALITY_POLICY,
        "prompt_wrapper_removed": wrapper_removed or upstream_wrapper_removed,
        "original_prompt_sha256": metadata.get(
            "source_original_prompt_sha256", sha256_text(original_prompt)
        ),
        "original_test_count": len(inputs),
        "unique_test_count": len(unique_inputs),
        "comparison": "tokens",
    }
    return cleaned, sorted(set(reasons))


def quality_policy_manifest(*, min_unique_tests: int) -> dict[str, Any]:
    return {
        "name": QUALITY_POLICY,
        "min_unique_tests": min_unique_tests,
        "prompt_wrapper": "strip_exact_nemotron_generation_prefix",
        "duplicate_tests": "deduplicate_identical_input_output; reject_conflicts",
        "stdout_comparison": "whitespace_token_equality",
        "excluded_prompt_classes": [
            "non_unique_output",
            "floating_point_tolerance",
            "case_insensitive_output",
            "interactive",
            "missing_problem_statement",
            "trailing_sample_marker_noise",
        ],
        "excluded_native_sources": sorted(EXCLUDED_NATIVE_SOURCES),
    }


def validate_formal_coding_row(
    row: dict[str, Any],
    *,
    min_unique_tests: int = FORMAL_MIN_UNIQUE_TESTS,
) -> list[str]:
    """Return policy violations without mutating an already canonical row."""
    reasons = prompt_exclusion_reasons(str(row.get("prompt", "")))
    if str(row.get("prompt", "")).startswith(NEMOTRON_PROMPT_PREFIX):
        reasons.append("source_prompt_wrapper_present")
    verifier = row.get("verifier")
    if not isinstance(verifier, dict) or verifier.get("type") not in {
        "io_tests",
        "reference_io_tests",
    }:
        reasons.append("unsupported_verifier")
        return sorted(set(reasons))
    if verifier.get("comparison") != "tokens":
        reasons.append("non_token_stdout_comparison")
    inputs = verifier.get("test_inputs")
    outputs = verifier.get("test_outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        reasons.append("invalid_test_lists")
        return sorted(set(reasons))
    if any(not str(i).strip() or not str(o).strip() for i, o in zip(inputs, outputs)):
        reasons.append("blank_test_io")
    metadata = row.get("metadata", {})
    if (
        isinstance(metadata, dict)
        and metadata.get("native_source") in EXCLUDED_NATIVE_SOURCES
    ):
        reasons.append("unreliable_native_source")
    if isinstance(metadata, dict) and metadata.get("native_source") == "hackerearth" and any(
        _HACKEREARTH_TRAILING_SAMPLE_MARKER.search(str(stdin)) for stdin in inputs
    ):
        reasons.append("trailing_sample_marker_noise")
    if len(inputs) != len(set(inputs)):
        reasons.append("duplicate_test_inputs")
    if len(inputs) < min_unique_tests:
        reasons.append("insufficient_unique_tests")
    return sorted(set(reasons))
