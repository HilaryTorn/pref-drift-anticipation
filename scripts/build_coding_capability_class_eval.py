#!/usr/bin/env python3
"""Fill remaining per-language quota gaps in data/reference/coding_capability_problems.json
using `class_based` candidates (single constructor + one usable public method), written as
`kind: "custom_driver"` problems.

This is the class-shaped sibling of build_coding_capability_pool_eval.py's `typed_function`
path: same idea (auto-derive a signature, generate test inputs, execute the row's own
reference solution twice to capture verified expected output, per-test resilience against
inputs that violate an implicit precondition), but for `ClassName(ctor_args).method(args)`
instead of a bare function call. Only ADDS problems and does not removes or regenerates the
typed_function problems already in the file for a language.

Usage:
    python3 scripts/build_coding_capability_class_eval.py --pool_n 36 --validation_n 4
    python3 scripts/build_coding_capability_class_eval.py --language java --dry_run
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scan_coding_capability_pool_candidates import strip_code_fence  # noqa: E402
from build_coding_capability_pool_eval import (  # noqa: E402
    CANDIDATES_PATH,
    OUT_PROBLEMS_PATH,
    OUT_REPORT_PATH,
    LANGUAGES,
    PARAM_TYPES,
    RETURN_TYPES,
    JAVA_CLASS_RE,
    JAVA_TYPE_MAP,
    RUST_TYPE_MAP,
    CPP_TYPE_MAP,
    _canon_annotation,
    _infer_param_type,
    _infer_return_type,
    _cpp_normalize_type,
    strip_c_style_comments,
    load_jsonl,
    render_literal,
    generate_test_inputs,
    derive_expected,
    N_CANDIDATE_TESTS,
)

CLASS_REPORT_PATH = ROOT / "data" / "reference" / "coding_capability_class_validation_report.json"


# --- Class signature extraction -------------------------------------------------------


def python_class_signature(solution: str) -> dict[str, Any] | None:
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return None
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        init = next(
            (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None
        )
        ctor_names: list[str] = []
        ctor_types: list[str] = []
        if init is not None:
            if init.args.vararg or init.args.kwarg or init.args.kwonlyargs or init.args.posonlyargs:
                continue
            init_source = ast.unparse(init)
            ok = True
            for arg in init.args.args[1:]:  # skip self
                t = (
                    _canon_annotation(arg.annotation)
                    if arg.annotation is not None
                    else _infer_param_type(init_source, arg.arg)
                )
                if t is None or t not in PARAM_TYPES:
                    ok = False
                    break
                ctor_names.append(arg.arg)
                ctor_types.append(t)
            if not ok:
                continue
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name.startswith("_"):
                continue
            if method.args.vararg or method.args.kwarg or method.args.kwonlyargs or method.args.posonlyargs:
                continue
            method_source = ast.unparse(method)
            m_names, m_types, ok = [], [], True
            for arg in method.args.args[1:]:  # skip self
                t = (
                    _canon_annotation(arg.annotation)
                    if arg.annotation is not None
                    else _infer_param_type(method_source, arg.arg)
                )
                if t is None or t not in PARAM_TYPES:
                    ok = False
                    break
                m_names.append(arg.arg)
                m_types.append(t)
            if not ok:
                continue
            ret = _canon_annotation(method.returns) if method.returns is not None else _infer_return_type(method)
            if ret is None or ret not in RETURN_TYPES:
                continue
            return {
                "class_name": cls.name,
                "constructor_param_names": ctor_names, "constructor_param_types": ctor_types,
                "method_name": method.name, "method_param_names": m_names, "method_param_types": m_types,
                "return_type": ret,
            }
    return None


JAVA_INSTANCE_METHOD_RE = re.compile(r"public\s+(?!static)([\w\[\]]+)\s+(\w+)\s*\(([^)]*)\)")


def _java_parse_params(params_raw: str) -> tuple[list[str], list[str]] | None:
    names, types = [], []
    for entry in (p.strip() for p in params_raw.split(",") if p.strip()):
        parts = entry.rsplit(None, 1)
        if len(parts) != 2:
            return None
        type_raw, pname = parts
        ptype = JAVA_TYPE_MAP.get(type_raw.strip())
        if ptype is None or ptype not in PARAM_TYPES:
            return None
        names.append(pname)
        types.append(ptype)
    return names, types


def java_class_signature(solution: str) -> dict[str, Any] | None:
    code = strip_c_style_comments(solution)
    class_match = JAVA_CLASS_RE.search(code)
    if not class_match:
        return None
    class_name = class_match.group(1)
    ctor_match = re.search(rf"public\s+{re.escape(class_name)}\s*\(([^)]*)\)", code)
    ctor_names, ctor_types = [], []
    if ctor_match and ctor_match.group(1).strip():
        parsed = _java_parse_params(ctor_match.group(1))
        if parsed is None:
            return None
        ctor_names, ctor_types = parsed
    for match in JAVA_INSTANCE_METHOD_RE.finditer(code):
        ret_raw, name, params_raw = match.groups()
        if name in (class_name, "main"):
            continue
        ret = JAVA_TYPE_MAP.get(ret_raw.strip())
        if ret is None or ret not in RETURN_TYPES:
            continue
        parsed = _java_parse_params(params_raw)
        if parsed is None:
            continue
        m_names, m_types = parsed
        return {
            "class_name": class_name,
            "constructor_param_names": ctor_names, "constructor_param_types": ctor_types,
            "method_name": name, "method_param_names": m_names, "method_param_types": m_types,
            "return_type": ret,
        }
    return None


RUST_NEW_FN_RE = re.compile(r"fn\s+new\s*\(([^)]*)\)\s*->\s*(?:Self|\w+)\s*\{")
RUST_METHOD_RE = re.compile(r"fn\s+(\w+)\s*\(\s*&(?:mut\s+)?self\s*(?:,\s*([^)]*))?\)\s*->\s*([\w<>&\[\]]+)\s*\{")
# Matches only an *inherent* impl (`impl StructName {`), not a trait impl (`impl Trait
# for StructName {`) -- the `\s*\{` right after the identifier fails to match when
# " for X {" follows, which is exactly the exclusion we want: a trait impl's methods
# have trait-defined (not necessarily f64/int/etc-typed) signatures, and single-struct
# files are the common case anyway, but Magicoder rows often define several unrelated
# structs (data holders, visitor-pattern participants, ...) in one solution -- pairing
# "the first struct name found" with "the first new() found anywhere" (the previous
# approach) silently associates a constructor with the wrong struct whenever more than
# one is present.
RUST_IMPL_HEADER_RE = re.compile(r"impl(?:<[^>]*>)?\s+(\w+)\s*\{")


def _extract_balanced_braces(code: str, open_brace_idx: int) -> str | None:
    depth = 0
    for i in range(open_brace_idx, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[open_brace_idx + 1 : i]
    return None


def _rust_impl_blocks(code: str) -> list[tuple[str, str]]:
    blocks = []
    for m in RUST_IMPL_HEADER_RE.finditer(code):
        body = _extract_balanced_braces(code, m.end() - 1)
        if body is not None:
            blocks.append((m.group(1), body))
    return blocks


def _rust_parse_params(params_raw: str) -> tuple[list[str], list[str]] | None:
    names, types = [], []
    for entry in (p.strip() for p in (params_raw or "").split(",") if p.strip()):
        if ":" not in entry:
            return None
        pname, type_raw = entry.split(":", 1)
        ptype = RUST_TYPE_MAP.get(type_raw.strip().removeprefix("mut ").strip())
        if ptype is None or ptype not in PARAM_TYPES:
            return None
        names.append(pname.strip())
        types.append(ptype)
    return names, types


def rust_class_signature(solution: str) -> dict[str, Any] | None:
    code = strip_c_style_comments(solution)
    for struct_name, body in _rust_impl_blocks(code):
        new_match = RUST_NEW_FN_RE.search(body)
        if not new_match:
            continue  # this struct's inherent impl has no `fn new(...) -> Self`-style constructor
        parsed = _rust_parse_params(new_match.group(1))
        if parsed is None:
            continue
        ctor_names, ctor_types = parsed
        for match in RUST_METHOD_RE.finditer(body):
            name, params_raw, ret_raw = match.groups()
            if name in ("new", "main"):
                continue
            ret = RUST_TYPE_MAP.get(ret_raw.strip())
            if ret is None or ret not in RETURN_TYPES:
                continue
            parsed = _rust_parse_params(params_raw)
            if parsed is None:
                continue
            m_names, m_types = parsed
            return {
                "class_name": struct_name,
                "constructor_param_names": ctor_names, "constructor_param_types": ctor_types,
                "method_name": name, "method_param_names": m_names, "method_param_types": m_types,
                "return_type": ret,
            }
    return None


CPP_CLASS_NAME_RE = re.compile(r"\b(?:class|struct)\s+(\w+)\s*\{")
# Methods commonly carry a trailing `const`/`noexcept`/`override` qualifier between the
# param list and `{`, and (unlike a free function) are always indented inside the class
# body -- `^` alone would only match column-0 text and silently find nothing.
CPP_METHOD_RE = re.compile(
    r"^\s*([\w:]+)\s+(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?\{", re.MULTILINE
)


def _cpp_parse_params(params_raw: str) -> tuple[list[str], list[str]] | None:
    names, types = [], []
    if not params_raw.strip():
        return names, types
    for entry in params_raw.split(","):
        entry = entry.strip()
        parts = entry.rsplit(None, 1)
        if len(parts) != 2:
            return None
        type_raw, pname = parts
        ptype = CPP_TYPE_MAP.get(_cpp_normalize_type(type_raw))
        if ptype is None or ptype not in PARAM_TYPES:
            return None
        names.append(pname.lstrip("&*"))
        types.append(ptype)
    return names, types


def cpp_class_signature(solution: str) -> dict[str, Any] | None:
    code = strip_c_style_comments(solution)
    class_match = CPP_CLASS_NAME_RE.search(code)
    if not class_match:
        return None
    class_name = class_match.group(1)
    # Allow an optional member-initializer list between the params and `{`, e.g.
    # `BankAccount(double b) : balance(b) {}` -- extremely common and otherwise
    # invisible to a plain `\(...\)\s*\{` match.
    ctor_match = re.search(
        rf"\b{re.escape(class_name)}\s*\(([^)]*)\)\s*(?::[^{{]*)?\{{", code
    )
    if not ctor_match:
        return None  # require an explicit constructor definition (defensive: don't assume a default ctor)
    parsed = _cpp_parse_params(ctor_match.group(1))
    if parsed is None:
        return None
    ctor_names, ctor_types = parsed
    for match in CPP_METHOD_RE.finditer(code):
        ret_raw, name, params_raw = match.groups()
        if name in (class_name, "if", "for", "while", "switch", "main"):
            continue
        ret = CPP_TYPE_MAP.get(_cpp_normalize_type(ret_raw))
        if ret is None or ret not in RETURN_TYPES:
            continue
        parsed = _cpp_parse_params(params_raw)
        if parsed is None:
            continue
        m_names, m_types = parsed
        return {
            "class_name": class_name,
            "constructor_param_names": ctor_names, "constructor_param_types": ctor_types,
            "method_name": name, "method_param_names": m_names, "method_param_types": m_types,
            "return_type": ret,
        }
    return None


CLASS_SIGNATURE_PARSERS = {
    "python": python_class_signature, "java": java_class_signature,
    "rust": rust_class_signature, "cpp": cpp_class_signature,
}


# --- Class-aware call building (construct-then-call) -----------------------------------


def build_class_call(sig: dict[str, Any], args: list[Any], lang: str) -> str:
    n_ctor = len(sig["constructor_param_types"])
    ctor_args, method_args = args[:n_ctor], args[n_ctor:]
    ctor_rendered = ", ".join(
        render_literal(a, t, lang) for a, t in zip(ctor_args, sig["constructor_param_types"])
    )
    method_rendered = ", ".join(
        render_literal(a, t, lang) for a, t in zip(method_args, sig["method_param_types"])
    )
    class_name, method_name = sig["class_name"], sig["method_name"]
    if lang == "python":
        return f"{class_name}({ctor_rendered}).{method_name}({method_rendered})"
    if lang == "java":
        return f"new {class_name}({ctor_rendered}).{method_name}({method_rendered})"
    if lang == "rust":
        return f"{class_name}::new({ctor_rendered}).{method_name}({method_rendered})"
    if lang == "cpp":
        return f"{class_name}({ctor_rendered}).{method_name}({method_rendered})"
    raise ValueError(lang)


def _sig_as_call_target(sig: dict[str, Any]) -> dict[str, Any]:
    """Adapts a class_signature dict into the shape derive_expected's harness-building
    expects (a flat `name`/`param_types` call target) by giving it a `name` derived
    from build_class_call instead -- see build_class_capture_harness below, which
    overrides build_call's dispatch for this shape via the `is_class` marker."""
    return {
        **sig,
        "param_types": sig["constructor_param_types"] + sig["method_param_types"],
        "is_class": True,
    }


def build_class_capture_harness(solution: str, sig: dict[str, Any], tests: list[list[Any]], lang: str) -> str:
    """Same shape/resilience as build_coding_capability_pool_eval.build_capture_harness,
    but the call expression constructs an instance first."""
    ret = sig["return_type"]
    if lang == "python":
        lines = [solution, "", "import json"]
        for args in tests:
            call = build_class_call(sig, args, lang)
            lines.append(
                f"try:\n    print('CAP_OUT:' + json.dumps({call}))\n"
                f"except Exception:\n    print('CAP_SKIP')"
            )
        return "\n".join(lines)
    if lang == "java":
        prints = []
        for args in tests:
            call = build_class_call(sig, args, lang)
            value_expr = call if ret == "str" else f"String.valueOf({call})"
            prints.append(
                f'        try {{ System.out.println("CAP_OUT:" + {value_expr}); }} '
                f'catch (Throwable e) {{ System.out.println("CAP_SKIP"); }}'
            )
        harness = (
            "\n\nclass _Capture {\n    public static void main(String[] args) {\n"
            + "\n".join(prints) + "\n    }\n}\n"
        )
        return solution + harness
    if lang == "rust":
        prints = []
        for args in tests:
            call = build_class_call(sig, args, lang)
            prints.append(
                f'    match std::panic::catch_unwind(|| {call}) {{\n'
                f'        Ok(v) => println!("CAP_OUT:{{}}", v),\n'
                f'        Err(_) => println!("CAP_SKIP"),\n'
                f'    }}'
            )
        harness = (
            "\n\nfn main() {\n    std::panic::set_hook(Box::new(|_| {}));\n"
            + "\n".join(prints) + "\n}\n"
        )
        return solution + harness
    if lang == "cpp":
        prints = []
        for args in tests:
            call = build_class_call(sig, args, lang)
            value_expr = f'(({call}) ? "true" : "false")' if ret == "bool" else f"({call})"
            prints.append(
                f'    try {{ std::cout << "CAP_OUT:" << {value_expr} << std::endl; }} '
                f'catch (...) {{ std::cout << "CAP_SKIP" << std::endl; }}'
            )
        harness = "\n\n#include <iostream>\nint main() {\n" + "\n".join(prints) + "\n    return 0;\n}\n"
        return solution + harness
    raise ValueError(lang)


def build_scoring_driver(sig: dict[str, Any], tests: list[list[Any]], expected: list[Any], lang: str) -> str:
    """The `driver` string stored on the problem: appended verbatim to a MODEL's
    completion at grading time (see run_custom_driver_* in score_coding_capability.py),
    so unlike build_class_capture_harness (executed against the real reference solution
    during the build) this compares against the already-verified expected values."""
    ret = sig["return_type"]
    if lang == "python":
        lines = ["__cap_passed = 0", "__cap_total = 0"]
        for args, exp in zip(tests, expected):
            call = build_class_call(sig, args, lang)
            exp_lit = render_literal(exp, ret, "python")
            lines.append(
                "__cap_total += 1\n"
                f"try:\n    __cap_passed += 1 if ({call}) == {exp_lit} else 0\n"
                "except Exception:\n    pass"
            )
        lines.append('print(f"CAPABILITY_RESULT {__cap_passed}/{__cap_total}")')
        return "\n".join(lines)
    if lang == "java":
        checks = []
        for args, exp in zip(tests, expected):
            call = build_class_call(sig, args, lang)
            exp_lit = render_literal(exp, ret, "java")
            checks.append(
                f'        total++; if (java.util.Objects.equals(_safe(() -> {call}), {exp_lit})) passed++;'
            )
        return (
            "\n\ninterface _Call { Object run(); }\n"
            "class _Harness {\n"
            "    static Object _safe(_Call c) {\n"
            "        try { return c.run(); } catch (Throwable e) { return new Object(); }\n"
            "    }\n"
            "    public static void main(String[] args) {\n"
            "        int passed = 0;\n        int total = 0;\n" + "\n".join(checks) + "\n"
            '        System.out.println("CAPABILITY_RESULT " + passed + "/" + total);\n'
            "    }\n}\n"
        )
    if lang == "rust":
        checks = []
        for args, exp in zip(tests, expected):
            call = build_class_call(sig, args, lang)
            exp_lit = render_literal(exp, ret, "rust")
            checks.append(
                f"    total += 1; if std::panic::catch_unwind(|| {call})"
                f".map(|v| v == {exp_lit}).unwrap_or(false) {{ passed += 1; }}"
            )
        return (
            "\n\nfn main() {\n    std::panic::set_hook(Box::new(|_| {}));\n"
            "    let mut passed: i32 = 0;\n    let mut total: i32 = 0;\n" + "\n".join(checks) + "\n"
            '    println!("CAPABILITY_RESULT {}/{}", passed, total);\n}\n'
        )
    if lang == "cpp":
        checks = []
        for args, exp in zip(tests, expected):
            call = build_class_call(sig, args, lang)
            exp_lit = render_literal(exp, ret, "cpp")
            checks.append(f"    _CAP_CHECK({call}, {exp_lit});")
        return (
            "\n\n#include <iostream>\nint _cap_passed = 0, _cap_total = 0;\n"
            "#define _CAP_CHECK(expr, expected) do { _cap_total++; "
            "try { if ((expr) == (expected)) _cap_passed++; } catch (...) {} } while (0)\n"
            "int main() {\n" + "\n".join(checks) + "\n"
            '    std::cout << "CAPABILITY_RESULT " << _cap_passed << "/" << _cap_total << std::endl;\n'
            "    return 0;\n}\n"
        )
    raise ValueError(lang)


def derive_expected_for_class(
    solution: str, sig: dict[str, Any], tests: list[list[Any]], lang: str
) -> tuple[list[list[Any]] | None, list[Any] | None, str | None]:
    """Mirrors build_coding_capability_pool_eval.derive_expected, but builds the
    class-aware capture harness instead of the plain-function one. Re-implemented
    (not parameterized into the original) because the harness-building call sites
    differ in shape; the run/parse/determinism logic is identical in spirit."""
    from build_coding_capability_pool_eval import run_capture, parse_cap_out, MIN_SUCCESSFUL_TESTS, MAX_KEPT_TESTS

    harness = build_class_capture_harness(solution, sig, tests, lang)
    ret = sig["return_type"]
    stdout1, err1 = run_capture(harness, lang)
    if err1:
        return None, None, f"first run failed: {err1}"
    values1 = parse_cap_out(stdout1, ret, lang, len(tests))
    stdout2, err2 = run_capture(harness, lang)
    if err2:
        return None, None, f"second run failed: {err2}"
    values2 = parse_cap_out(stdout2, ret, lang, len(tests))

    kept_tests, kept_expected = [], []
    for args, v1, v2 in zip(tests, values1, values2):
        if v1 is None or v2 is None or v1 != v2:
            continue
        kept_tests.append(args)
        kept_expected.append(v1)
        if len(kept_tests) >= MAX_KEPT_TESTS:
            break
    if len(kept_tests) < MIN_SUCCESSFUL_TESTS:
        return None, None, (
            f"only {len(kept_tests)}/{len(tests)} candidate inputs succeeded "
            f"deterministically (need >= {MIN_SUCCESSFUL_TESTS}): {values1}"
        )
    return kept_tests, kept_expected, None


# --- Orchestration -----------------------------------------------------------------------


def current_counts(problems: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {lang: {"pool": 0, "validation": 0} for lang in LANGUAGES}
    for p in problems:
        lang = p.get("language_id")
        split = p.get("split")
        if lang in counts and split in ("pool", "validation"):
            counts[lang][split] += 1
    return counts


def build_language(
    lang: str,
    candidates: dict[str, Any],
    needed: dict[str, int],
    seed: int,
    report: dict[str, Any],
    already_used_row_ids: set[str],
) -> list[dict[str, Any]]:
    problems = []
    for split in ("pool", "validation"):
        quota = needed[split]
        report["target_counts"].setdefault(lang, {})[split] = quota
        if quota <= 0:
            report["achieved_counts"].setdefault(lang, {})[split] = 0
            continue
        rows = load_jsonl(ROOT / "data" / "training" / f"coding.write.{lang}" / f"{split}.jsonl")
        # Exclude rows a prior pass (typed_function OR an earlier class-based run) already
        # turned into a problem -- otherwise a second fill-the-gap invocation reshuffles
        # the SAME candidate list with the SAME seed and can re-select and re-add a row
        # that's already present, producing a duplicate `{row_id}.capability` id.
        class_candidate_ids = [
            c["id"] for c in candidates[lang][split]["class_based"] if c["id"] not in already_used_row_ids
        ]
        rng = random.Random(seed)
        shuffled = class_candidate_ids[:]
        rng.shuffle(shuffled)

        accepted = 0
        for row_id in shuffled:
            if accepted >= quota:
                break
            row = rows[row_id]
            solution = strip_code_fence(row["messages"][1]["content"])
            sig = CLASS_SIGNATURE_PARSERS[lang](solution)
            if sig is None:
                report["signature_parse_failed"].append({"id": row_id, "language": lang})
                continue
            combined_types = sig["constructor_param_types"] + sig["method_param_types"]
            if not combined_types:
                # A zero-arg constructor + zero-arg method means every "test case" is a
                # bare repeat of the exact same call -- any input-independent variation
                # in the expected sequence (seen in practice: a static counter that
                # increments across constructions) reflects hidden global state, not the
                # model's ability to handle varying inputs. Not incorrect to grade, but a
                # degenerate, easily-gamed signal, so exclude these from the bank.
                report["signature_parse_failed"].append(
                    {"id": row_id, "language": lang, "reason": "zero_arg_degenerate"}
                )
                continue
            tests = generate_test_inputs(combined_types)
            kept_tests, expected, error = derive_expected_for_class(solution, sig, tests, lang)
            if error:
                report["execution_rejected"].append({"id": row_id, "language": lang, "reason": error})
                continue
            driver = build_scoring_driver(sig, kept_tests, expected, lang)
            problems.append(
                {
                    "id": f"{row_id}.capability",
                    "language_id": lang,
                    "split": split,
                    "kind": "custom_driver",
                    "source": "ise-uiuc/Magicoder-OSS-Instruct-75K",
                    "license": "mit",
                    "source_row_id": row_id,
                    "reference_solution_sha256": hashlib.sha256(solution.encode()).hexdigest(),
                    "prompt": row["messages"][0]["content"],
                    "entry_point": sig["class_name"],
                    "class_method": sig["method_name"],
                    "driver": driver,
                }
            )
            accepted += 1
        report["achieved_counts"].setdefault(lang, {})[split] = accepted
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=LANGUAGES, default=None)
    parser.add_argument("--pool_n", type=int, default=36, help="Target pool count per language (fills the gap only)")
    parser.add_argument("--validation_n", type=int, default=4, help="Target validation count per language")
    parser.add_argument("--seed", type=int, default=43)  # different seed than the typed_function pass
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not CANDIDATES_PATH.exists():
        raise SystemExit(f"Missing candidate manifest: {CANDIDATES_PATH}")
    candidates = json.loads(CANDIDATES_PATH.read_text())["languages"]
    languages = [args.language] if args.language else LANGUAGES

    existing: dict[str, Any] = {"schema": "coding_capability_pool_v1", "problems": []}
    if OUT_PROBLEMS_PATH.exists():
        existing = json.loads(OUT_PROBLEMS_PATH.read_text())
    existing_counts = current_counts(existing.get("problems", []))
    already_used_row_ids = {
        p["source_row_id"] for p in existing.get("problems", []) if "source_row_id" in p
    }

    report: dict[str, Any] = {
        "schema": "coding_capability_class_build_v1",
        "seed": args.seed,
        "target_counts": {},
        "achieved_counts": {},
        "signature_parse_failed": [],
        "execution_rejected": [],
    }

    all_new_problems = []
    for lang in languages:
        needed = {
            "pool": max(0, args.pool_n - existing_counts[lang]["pool"]),
            "validation": max(0, args.validation_n - existing_counts[lang]["validation"]),
        }
        print(f"Building {lang} class-based fill (need pool={needed['pool']}, validation={needed['validation']})...")
        new_problems = build_language(lang, candidates, needed, args.seed, report, already_used_row_ids)
        all_new_problems.extend(new_problems)
        already_used_row_ids.update(p["source_row_id"] for p in new_problems)
        achieved = report["achieved_counts"][lang]
        print(f"  {lang}: filled pool +{achieved['pool']}, validation +{achieved['validation']}")

    print(f"\nTotal custom_driver problems built: {len(all_new_problems)}")
    print(f"Signature parse failures: {len(report['signature_parse_failed'])}")
    print(f"Execution rejections: {len(report['execution_rejected'])}")

    if not args.dry_run:
        CLASS_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {CLASS_REPORT_PATH.relative_to(ROOT)}")
        # Purely additive: keep every existing problem (both typed_function and any
        # prior custom_driver rows), just append the newly built ones.
        existing["problems"] = existing.get("problems", []) + all_new_problems
        OUT_PROBLEMS_PATH.write_text(json.dumps(existing, indent=2, allow_nan=False) + "\n")
        print(f"Wrote {OUT_PROBLEMS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
