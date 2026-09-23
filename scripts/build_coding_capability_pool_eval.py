#!/usr/bin/env python3
"""Build data/reference/coding_capability_problems.json from real coding-SFT pool/
validation data (mechanical path: `typed_function` candidates only).

Reads the candidate manifest written by scan_coding_capability_pool_candidates.py,
samples up to --pool_n pool rows + --validation_n validation rows per language
(seed-reproducible, matching build_sft_pools.py's convention), parses each row's own
(explicitly typed) function/method signature, auto-generates test inputs per param
type, executes the row's OWN reference solution against those inputs to capture real
output as the graded expected value (run twice per row to reject nondeterministic
solutions), and writes verified `typed_function` problems.

`class_based` candidates are not handled here, those need a hand-authored driver per
problem and are appended to the same output file by a separate pass, not regenerated
by this script.

Usage:
    python3 scripts/scan_coding_capability_pool_candidates.py   # writes the manifest this reads
    python3 scripts/build_coding_capability_pool_eval.py --pool_n 36 --validation_n 4
    python3 scripts/build_coding_capability_pool_eval.py --language java --dry_run
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scan_coding_capability_pool_candidates import strip_code_fence  # noqa: E402

_IS_WINDOWS = platform.system() == "Windows"


def run_with_hard_timeout(cmd: list[str], cwd: str, timeout: float) -> dict:
    """subprocess.run(timeout=...) is not reliable here: on Windows, killing the
    direct child does not release output-pipe handles a grandchild process may have
    inherited (a JVM helper thread, a linker invoked by rustc/g++), and
    Popen.communicate()'s post-kill drain can then block forever waiting for a pipe
    close that never comes -- observed in practice as a `java _Capture` process from
    this exact pipeline still running many HOURS after its 20s timeout should have
    fired. Force-kill the whole process tree via `taskkill /F /T` on timeout instead
    of relying on Popen.kill(), which only touches the immediate process."""
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return {"stdout": stdout, "stderr": stderr, "returncode": proc.returncode, "error": None}
    except subprocess.TimeoutExpired:
        if _IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=10, check=False,
            )
        else:
            proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except Exception:
            stdout, stderr = "", ""
        return {"stdout": stdout, "stderr": stderr, "returncode": -1, "error": f"timeout after {timeout}s"}
    except OSError as exc:
        return {"stdout": "", "stderr": "", "returncode": -1, "error": f"failed to run {cmd[0]}: {exc}"}

CANDIDATES_PATH = ROOT / "data" / "reference" / "coding_capability_pool_candidates.json"
OUT_PROBLEMS_PATH = ROOT / "data" / "reference" / "coding_capability_problems.json"
OUT_REPORT_PATH = ROOT / "data" / "reference" / "coding_capability_pool_validation_report.json"

LANGUAGES = ["python", "java", "rust", "cpp"]

# Params: broadened beyond today's harness (adds float, list_str). Return: kept scalar
# and float-free -- cross-language float-to-string formatting differs (Java
# Double.toString vs Rust {} vs C++ cout precision), which would make exact-match
# grading of a *different* model's own float formatting unreliable. Params don't have
# this problem since we only ever construct literals from them, never parse them back.
PARAM_TYPES = {"int", "bool", "str", "float", "list_int", "list_str"}
# Tried and reverted 2026-08-21: admitting list_int/list_str as RETURN types looked like a +61
# candidate win (+17%), but that measured PARSEABILITY, which is only the first gate. Candidates
# must then survive execution-based expected-value derivation, and execution rejections rose
# 153 -> 210 -- almost exactly absorbing the newly-parseable signatures. Net yield was 4 problems
# out of 210. Not worth the added surface; the constraint is derivation, not the type whitelist.
# (Floats remain excluded for the separate, sound reason in the note above: cross-language
# float-to-string formatting makes exact-match grading of a model's own output unreliable.)
RETURN_TYPES = {"int", "bool", "str"}

C_STYLE_LINE_COMMENT_RE = re.compile(r"//.*")
C_STYLE_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_c_style_comments(code: str) -> str:
    return C_STYLE_LINE_COMMENT_RE.sub("", C_STYLE_BLOCK_COMMENT_RE.sub("", code))


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


# --- Signature extraction -------------------------------------------------------------


def _canon_annotation(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return {"int": "int", "bool": "bool", "str": "str", "float": "float"}.get(node.id)
    if isinstance(node, ast.Subscript):
        base = node.value.id if isinstance(node.value, ast.Name) else None
        if base not in ("list", "List"):
            return None
        inner = node.slice
        if isinstance(inner, ast.Name):
            if inner.id == "int":
                return "list_int"
            if inner.id == "str":
                return "list_str"
        return None
    return None


# Most Magicoder Python solutions carry no type hints at all (unlike java/rust/cpp,
# which are always explicitly typed), so requiring annotations alone rejects the
# large majority of otherwise-usable rows. Infer from usage instead when annotations
# are absent. A wrong guess is not a correctness risk: it just produces test inputs
# the reference solution errors on, which derive_expected()'s per-test resilience
# then drops -- worst case the row gets rejected for insufficient successful tests,
# same as an unannotated row is rejected today.
_STR_METHOD_SIGNALS = (
    "upper", "lower", "strip", "lstrip", "rstrip", "split", "replace", "startswith",
    "endswith", "join", "isalpha", "isdigit", "isupper", "islower", "format", "capitalize",
)


_DICT_METHOD_SIGNALS = ("items", "keys", "values", "get", "setdefault", "update", "pop")


def _infer_param_type(source: str, param: str) -> str | None:
    # Dict subscript access (`d[key]`) is syntactically identical to list indexing
    # (`lst[i]`) -- `x[y]` alone can't tell them apart. A dict-specific method call is
    # the one reliable signal, and dict is not a supported type here, so treat it as a
    # hard exclusion rather than falling through to the list-indexing guess below.
    if any(re.search(rf"\b{re.escape(param)}\s*\.\s*{m}\s*\(", source) for m in _DICT_METHOD_SIGNALS):
        return None
    if re.search(rf"\b{re.escape(param)}\s*==\s*(True|False)\b", source) or re.search(
        rf"\b{re.escape(param)}\s+is\s+(True|False)\b", source
    ):
        return "bool"
    if any(re.search(rf"\b{re.escape(param)}\s*\.\s*{m}\s*\(", source) for m in _STR_METHOD_SIGNALS):
        return "str"
    if re.search(rf'["\']\s*\+\s*{re.escape(param)}\b', source) or re.search(
        rf'\b{re.escape(param)}\s*\+\s*["\']', source
    ):
        return "str"
    is_indexable = bool(
        re.search(rf"\b{re.escape(param)}\s*\[", source)
        or re.search(rf"\bfor\s+\w+\s+in\s+{re.escape(param)}\b", source)
        or re.search(rf"\b{re.escape(param)}\s*\.\s*append\s*\(", source)
    )
    if is_indexable:
        # Disambiguate element type: does an indexed/iterated element get treated as a
        # string (method calls, string concat) or a number (arithmetic)? Default to
        # list_int when there's no signal either way -- Magicoder's algorithmic
        # problems skew numeric.
        elem_str_signal = any(
            re.search(rf"\b{re.escape(param)}\s*\[[^\]]*\]\s*\.\s*{m}\s*\(", source) for m in _STR_METHOD_SIGNALS
        )
        return "list_str" if elem_str_signal else "list_int"
    if re.search(rf"\brange\s*\(\s*{re.escape(param)}\b", source) or re.search(
        rf"\b{re.escape(param)}\s*[+\-*/%]{{1,2}}=?\s*[\w(]", source
    ):
        return "int"
    if re.search(rf"\b{re.escape(param)}\s*(==|!=|<=|>=|<|>)\s*-?\d", source):
        return "int"
    return None


def _infer_return_type(fn: ast.FunctionDef) -> str | None:
    kinds: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, bool):
            kinds.add("bool")
        elif isinstance(value, ast.Compare):
            kinds.add("bool")
        elif isinstance(value, ast.Constant) and isinstance(value.value, str):
            kinds.add("str")
        elif isinstance(value, ast.JoinedStr):  # f-string
            kinds.add("str")
        elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            # Ambiguous (numeric add vs string concat) without more context -- bail.
            return None
        elif isinstance(value, ast.Constant) and isinstance(value.value, int):
            kinds.add("int")
        elif isinstance(value, (ast.Call,)) and isinstance(value.func, ast.Name) and value.func.id in (
            "len", "sum", "int", "abs", "min", "max", "round",
        ):
            kinds.add("int")
        else:
            return None
    if len(kinds) == 1:
        return kinds.pop()
    return None


def python_signature(solution: str) -> dict[str, Any] | None:
    """Requires exactly one top-level function. Uses explicit annotations where
    present, falls back to usage-based inference (see _infer_param_type /
    _infer_return_type) where they're missing."""
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        return None
    fn = funcs[0]

    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs:
        return None
    param_names, param_types = [], []
    for arg in fn.args.args:
        if arg.annotation is not None:
            # An explicit annotation is an authoritative signal, even when it names a
            # type we don't support (dict, Optional[...], a custom class, ...) --
            # never fall through to guessing in that case, only when there is truly
            # no annotation to begin with.
            t = _canon_annotation(arg.annotation)
        else:
            t = _infer_param_type(solution, arg.arg)
        if t is None or t not in PARAM_TYPES:
            return None
        param_names.append(arg.arg)
        param_types.append(t)
    if fn.returns is not None:
        ret = _canon_annotation(fn.returns)
    else:
        ret = _infer_return_type(fn)
    if ret is None or ret not in RETURN_TYPES:
        return None
    return {"name": fn.name, "param_names": param_names, "param_types": param_types, "return_type": ret}


JAVA_STATIC_SIG_RE = re.compile(
    r"public\s+static\s+([\w\[\]]+)\s+(\w+)\s*\(([^)]*)\)"
)
JAVA_CLASS_RE = re.compile(r"\b(?:public\s+)?(?:final\s+)?(?:abstract\s+)?class\s+([A-Z]\w*)")
JAVA_TYPE_MAP = {
    "int": "int", "Integer": "int",
    "boolean": "bool", "Boolean": "bool",
    "String": "str",
    "double": "float", "Double": "float", "float": "float", "Float": "float",
    "int[]": "list_int",
    "String[]": "list_str",
}


def java_signature(solution: str) -> dict[str, Any] | None:
    code = strip_c_style_comments(solution)
    # A row's public static methods often include a `main(String[] args)` demo/driver
    # method alongside (or instead of) the real target -- skip it and any other match
    # whose types don't fit our supported set, rather than taking the first match.
    for match in JAVA_STATIC_SIG_RE.finditer(code):
        ret_raw, name, params_raw = match.groups()
        if name == "main":
            continue
        ret = JAVA_TYPE_MAP.get(ret_raw.strip())
        if ret is None or ret not in RETURN_TYPES:
            continue
        param_names, param_types, ok = [], [], True
        for entry in (p.strip() for p in params_raw.split(",") if p.strip()):
            parts = entry.rsplit(None, 1)
            if len(parts) != 2:
                ok = False
                break
            type_raw, pname = parts
            ptype = JAVA_TYPE_MAP.get(type_raw.strip())
            if ptype is None or ptype not in PARAM_TYPES:
                ok = False
                break
            param_names.append(pname)
            param_types.append(ptype)
        if ok:
            # The scanner only classifies a row typed_function when it has exactly one
            # class, so this is unambiguous -- but the class is whatever the original
            # Magicoder row named it (e.g. "CombinationGenerator"), never "Solution".
            # "Solution" is a grading-time convention told to the *model*, not a fact
            # about the reference solution being executed here.
            class_match = JAVA_CLASS_RE.search(code)
            class_name = class_match.group(1) if class_match else "Solution"
            return {
                "name": name, "param_names": param_names, "param_types": param_types,
                "return_type": ret, "class_name": class_name,
            }
    return None


RUST_FN_SIG_RE = re.compile(
    r"fn\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([\w<>&\[\]]+))?\s*\{"
)
RUST_TYPE_MAP = {
    "i32": "int", "i64": "int", "u32": "int", "u64": "int", "usize": "int", "isize": "int",
    "bool": "bool",
    "&str": "str", "String": "str",
    "f32": "float", "f64": "float",
    "&[i32]": "list_int", "Vec<i32>": "list_int",
    "&[&str]": "list_str", "Vec<String>": "list_str", "Vec<&str>": "list_str",
}


def rust_signature(solution: str) -> dict[str, Any] | None:
    code = strip_c_style_comments(solution)
    # Rows often carry a `fn main() { ... }` demo/test driver alongside the real target
    # function -- skip it (and any match with an unsupported signature) instead of
    # bailing on the first regex match, mirroring the cpp parser below.
    for match in RUST_FN_SIG_RE.finditer(code):
        name, params_raw, ret_raw = match.groups()
        if name == "main":
            continue
        ret = RUST_TYPE_MAP.get((ret_raw or "").strip()) if ret_raw else None
        if ret is None or ret not in RETURN_TYPES:
            continue
        param_names, param_types, ok = [], [], True
        for entry in (p.strip() for p in params_raw.split(",") if p.strip()):
            if entry in ("self", "&self", "&mut self") or ":" not in entry:
                ok = False
                break
            pname, type_raw = entry.split(":", 1)
            ptype = RUST_TYPE_MAP.get(type_raw.strip().removeprefix("mut ").strip())
            if ptype is None or ptype not in PARAM_TYPES:
                ok = False
                break
            param_names.append(pname.strip())
            param_types.append(ptype)
        if ok:
            return {"name": name, "param_names": param_names, "param_types": param_types, "return_type": ret}
    return None


CPP_FREE_FN_SIG_RE = re.compile(
    r"^([\w:]+)\s+(\w+)\s*\(([^)]*)\)\s*\{", re.MULTILINE
)
CPP_TYPE_MAP = {
    "int": "int",
    "bool": "bool",
    "std::string": "str", "string": "str",
    "double": "float", "float": "float",
    "std::vector<int>": "list_int", "vector<int>": "list_int",
    "std::vector<std::string>": "list_str", "vector<string>": "list_str",
    "std::vector<string>": "list_str",
}


def _cpp_normalize_type(raw: str) -> str:
    return raw.replace("const ", "").strip().rstrip("&").strip()


def cpp_signature(solution: str) -> dict[str, Any] | None:
    code = strip_c_style_comments(solution)
    for match in CPP_FREE_FN_SIG_RE.finditer(code):
        ret_raw, name, params_raw = match.groups()
        if name in ("if", "for", "while", "switch", "main"):
            continue
        ret = CPP_TYPE_MAP.get(_cpp_normalize_type(ret_raw))
        if ret is None or ret not in RETURN_TYPES:
            continue
        param_names, param_types, ok = [], [], True
        if params_raw.strip():
            for entry in params_raw.split(","):
                entry = entry.strip()
                parts = entry.rsplit(None, 1)
                if len(parts) != 2:
                    ok = False
                    break
                type_raw, pname = parts
                ptype = CPP_TYPE_MAP.get(_cpp_normalize_type(type_raw))
                if ptype is None or ptype not in PARAM_TYPES:
                    ok = False
                    break
                param_names.append(pname.lstrip("&*"))
                param_types.append(ptype)
        if ok:
            return {"name": name, "param_names": param_names, "param_types": param_types, "return_type": ret}
    return None


SIGNATURE_PARSERS = {
    "python": python_signature, "java": java_signature,
    "rust": rust_signature, "cpp": cpp_signature,
}


# --- Test input generation --------------------------------------------------------------

BASE_VALUES = {
    "int": [0, 1, -1, 42, -17, 1000, 7, -256],
    "bool": [True, False, True, False, True, False, True, False],
    "str": ["", "a", "hello", "Hello World", "racecar", "12345", "test string", "x"],
    "float": [0.0, 1.5, -2.75, 100.0, -0.5, 3.0, 42.25, -10.125],
    "list_int": [[], [1], [1, 2, 3], [-1, -2, -3], [5, 3, 8, 1, 9, 2], [0, 0, 0], [10], [-5, 5, 0]],
    "list_str": [[], ["a"], ["hello", "world"], ["x", "y", "z"], ["ab", "cd"], ["one"], ["p", "q", "r"], ["z"]],
}
# Generate more candidate inputs than we need: a reference solution often assumes an
# implicit precondition the problem prose states but a generic input violates (e.g. a
# roman-numeral parser given "a"), so some inputs will legitimately error. Rather than
# rejecting the whole row over one bad input, generate a surplus and keep whichever
# subset both executes cleanly and reproduces deterministically.
N_CANDIDATE_TESTS = 10
MIN_SUCCESSFUL_TESTS = 4
MAX_KEPT_TESTS = 6


def generate_test_inputs(param_types: list[str]) -> list[list[Any]]:
    tests = []
    for i in range(N_CANDIDATE_TESTS):
        args = [BASE_VALUES[t][i % len(BASE_VALUES[t])] for t in param_types]
        tests.append(args)
    return tests


# --- Literal rendering (adapted from scripts/score_coding_capability.py) ----------------


def render_literal(value: Any, type_: str, lang: str) -> str:
    if type_ == "bool":
        if lang == "python":
            return "True" if value else "False"
        return "true" if value else "false"
    if type_ in ("int",):
        return str(value)
    if type_ == "float":
        return f"{value}" if lang != "cpp" else f"{value}"
    if type_ == "str":
        return json.dumps(value)
    if type_ == "list_int":
        items = ", ".join(str(v) for v in value)
        return {
            "python": f"[{items}]",
            "java": f"new int[]{{{items}}}",
            "rust": f"&[{items}]",
            "cpp": f"std::vector<int>{{{items}}}",
        }[lang]
    if type_ == "list_str":
        items = ", ".join(json.dumps(v) for v in value)
        return {
            "python": f"[{items}]",
            "java": f"new String[]{{{items}}}",
            "rust": f"&[{items}]",
            "cpp": f"std::vector<std::string>{{{items}}}",
        }[lang]
    raise ValueError(f"unsupported {type_!r} for {lang!r}")


def build_call(sig: dict[str, Any], args: list[Any], lang: str) -> str:
    rendered = ", ".join(render_literal(a, t, lang) for a, t in zip(args, sig["param_types"]))
    if lang == "java":
        return f"{sig['class_name']}.{sig['name']}({rendered})"
    return f"{sig['name']}({rendered})"


# --- Reference-solution execution: derive expected outputs by running the real code ------


def build_capture_harness(solution: str, sig: dict[str, Any], tests: list[list[Any]], lang: str) -> str:
    """Each call is wrapped so one bad input (violates an implicit precondition the
    reference solution assumes, e.g. a roman-numeral parser given "a") prints CAP_SKIP
    and the harness keeps going, instead of the whole process dying on the first
    exception -- derive_expected() below then keeps only the calls that succeeded."""
    ret = sig["return_type"]
    if lang == "python":
        lines = [solution, "", "import json"]
        for args in tests:
            call = build_call(sig, args, lang)
            lines.append(
                f"try:\n    print('CAP_OUT:' + json.dumps({call}))\n"
                f"except Exception:\n    print('CAP_SKIP')"
            )
        return "\n".join(lines)
    if lang == "java":
        prints = []
        for args in tests:
            call = build_call(sig, args, lang)
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
            call = build_call(sig, args, lang)
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
            call = build_call(sig, args, lang)
            value_expr = f'(({call}) ? "true" : "false")' if ret == "bool" else f"({call})"
            prints.append(
                f'    try {{ std::cout << "CAP_OUT:" << {value_expr} << std::endl; }} '
                f'catch (...) {{ std::cout << "CAP_SKIP" << std::endl; }}'
            )
        harness = (
            "\n\n#include <iostream>\nint main() {\n" + "\n".join(prints) + "\n    return 0;\n}\n"
        )
        return solution + harness
    raise ValueError(lang)


def _matches_return_type(value: Any, ret: str) -> bool:
    """Python's `json.loads` happily round-trips values that were never actually valid
    for the declared return_type -- e.g. a reference solution's `int`-inferred function
    returning `float('-inf')` as an empty-input sentinel round-trips as a Python float,
    and `json.dumps` will silently emit it as the literal `-Infinity`, which is not
    valid JSON at all (this is exactly how one leaked into the output file). Java/
    Rust/C++ are naturally protected since `int(raw)` raises cleanly on non-integer
    text, but the python branch must check the parsed value's actual type explicitly."""
    if ret == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if ret == "bool":
        return isinstance(value, bool)
    if ret == "str":
        return isinstance(value, str)
    return False


def parse_cap_out(stdout: str, ret: str, lang: str, n_expected: int) -> list[Any | None]:
    """Returns a list aligned with the input tests, None wherever the harness printed
    CAP_SKIP (that test errored) or output was otherwise unparseable/mistyped."""
    lines = [ln for ln in stdout.splitlines() if ln.startswith(("CAP_OUT:", "CAP_SKIP"))]
    values: list[Any | None] = []
    for ln in lines:
        if ln == "CAP_SKIP":
            values.append(None)
            continue
        raw = ln[len("CAP_OUT:"):]
        if lang == "python":
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                values.append(None)
                continue
            values.append(parsed if _matches_return_type(parsed, ret) else None)
            continue
        if ret == "int":
            try:
                values.append(int(raw))
            except ValueError:
                values.append(None)
        elif ret == "bool":
            text = raw.strip().lower()
            values.append(text == "true" if text in ("true", "false") else None)
        else:  # str
            values.append(raw)
    while len(values) < n_expected:
        values.append(None)
    return values[:n_expected]


def run_capture(
    harness: str, lang: str, compile_timeout: float = 30.0, run_timeout: float = 8.0
) -> tuple[str, str | None]:
    """Returns (stdout, error). Compiles+runs (or just runs, for python). The actual
    execution step uses a short timeout -- these are tiny generated calls, so anything
    still running past a few seconds is almost certainly a genuine infinite loop in a
    (LLM-generated, not guaranteed correct) Magicoder reference solution, not a slow
    but legitimate computation."""
    if lang == "python":
        with tempfile.TemporaryDirectory(prefix="cap-build-py-") as d:
            path = Path(d) / "candidate.py"
            path.write_text(harness, encoding="utf-8")
            result = run_with_hard_timeout([sys.executable, str(path)], d, run_timeout)
            if result["error"]:
                return "", result["error"]
            if result["returncode"] != 0:
                return "", (result["stderr"] or "")[-500:]
            return result["stdout"], None
    if lang == "java":
        class_match = re.search(r"public\s+class\s+(\w+)", strip_c_style_comments(harness))
        filename = f"{class_match.group(1)}.java" if class_match else "Solution.java"
        with tempfile.TemporaryDirectory(prefix="cap-build-java-") as d:
            (Path(d) / filename).write_text(harness, encoding="utf-8")
            compiled = run_with_hard_timeout(["javac", "-nowarn", filename], d, compile_timeout)
            if compiled["error"]:
                return "", f"javac: {compiled['error']}"
            if compiled["returncode"] != 0:
                return "", (compiled["stderr"] or "")[-500:]
            result = run_with_hard_timeout(["java", "_Capture"], d, run_timeout)
            if result["error"]:
                return "", result["error"]
            if result["returncode"] != 0:
                return "", (result["stderr"] or "")[-500:]
            return result["stdout"], None
    if lang == "rust":
        with tempfile.TemporaryDirectory(prefix="cap-build-rs-") as d:
            src = Path(d) / "candidate.rs"
            src.write_text(harness, encoding="utf-8")
            bin_path = Path(d) / "candidate_bin"
            compiled = run_with_hard_timeout(
                ["rustc", "--edition", "2021", "-O", str(src), "-o", str(bin_path)], d, compile_timeout
            )
            if compiled["error"]:
                return "", f"rustc: {compiled['error']}"
            if compiled["returncode"] != 0:
                return "", (compiled["stderr"] or "")[-500:]
            result = run_with_hard_timeout([str(bin_path)], d, run_timeout)
            if result["error"]:
                return "", result["error"]
            if result["returncode"] != 0:
                return "", (result["stderr"] or "")[-500:]
            return result["stdout"], None
    if lang == "cpp":
        with tempfile.TemporaryDirectory(prefix="cap-build-cpp-") as d:
            src = Path(d) / "candidate.cpp"
            src.write_text(harness, encoding="utf-8")
            bin_path = Path(d) / "candidate_bin"
            compiled = run_with_hard_timeout(
                ["g++", "-O0", "-std=c++17", str(src), "-o", str(bin_path)], d, compile_timeout
            )
            if compiled["error"]:
                return "", f"g++: {compiled['error']}"
            if compiled["returncode"] != 0:
                return "", (compiled["stderr"] or "")[-500:]
            result = run_with_hard_timeout([str(bin_path)], d, run_timeout)
            if result["error"]:
                return "", result["error"]
            if result["returncode"] != 0:
                return "", (result["stderr"] or "")[-500:]
            return result["stdout"], None
    raise ValueError(lang)


def derive_expected(
    solution: str, sig: dict[str, Any], tests: list[list[Any]], lang: str
) -> tuple[list[list[Any]] | None, list[Any] | None, str | None]:
    """Returns (kept_tests, kept_expected, error). A row is accepted if at least
    MIN_SUCCESSFUL_TESTS of its N_CANDIDATE_TESTS candidate inputs both execute cleanly
    (no CAP_SKIP) and agree between two independent runs (determinism check) -- the
    rest are dropped rather than failing the whole row, since a generic input violating
    one function's implicit precondition says nothing about the other inputs."""
    harness = build_capture_harness(solution, sig, tests, lang)
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


def build_language(
    lang: str, candidates: dict[str, Any], pool_n: int, validation_n: int, seed: int, report: dict[str, Any]
) -> list[dict[str, Any]]:
    problems = []
    for split, quota in [("pool", pool_n), ("validation", validation_n)]:
        rows = load_jsonl(ROOT / "data" / "training" / f"coding.write.{lang}" / f"{split}.jsonl")
        typed_candidate_ids = [c["id"] for c in candidates[lang][split]["typed_function"]]
        rng = random.Random(seed)
        shuffled = typed_candidate_ids[:]
        rng.shuffle(shuffled)

        accepted = 0
        for row_id in shuffled:
            if accepted >= quota:
                break
            row = rows[row_id]
            solution = strip_code_fence(row["messages"][1]["content"])
            sig = SIGNATURE_PARSERS[lang](solution)
            if sig is None:
                report["signature_parse_failed"].append({"id": row_id, "language": lang})
                continue
            if not sig["param_types"]:
                # A zero-arg function has nothing for a test to vary -- every call is
                # identical, so all it can measure is "does this constant-returning
                # function get reproduced," not any real input-handling ability.
                report["signature_parse_failed"].append(
                    {"id": row_id, "language": lang, "reason": "zero_arg_degenerate"}
                )
                continue
            tests = generate_test_inputs(sig["param_types"])
            kept_tests, expected, error = derive_expected(solution, sig, tests, lang)
            if error:
                report["execution_rejected"].append({"id": row_id, "language": lang, "reason": error})
                continue
            tests = kept_tests
            problem_id = f"{row_id}.capability"
            problems.append(
                {
                    "id": problem_id,
                    "language_id": lang,
                    "split": split,
                    "kind": "typed_function",
                    "source": "ise-uiuc/Magicoder-OSS-Instruct-75K",
                    "license": "mit",
                    "source_row_id": row_id,
                    "reference_solution_sha256": hashlib.sha256(solution.encode()).hexdigest(),
                    "prompt": row["messages"][0]["content"],
                    "entry_point": sig["name"],
                    "param_names": sig["param_names"],
                    "param_types": sig["param_types"],
                    "return_type": sig["return_type"],
                    "tests": [[args, exp] for args, exp in zip(tests, expected)],
                }
            )
            accepted += 1
        report["achieved_counts"].setdefault(lang, {})[split] = accepted
        report["quota_counts"].setdefault(lang, {})[split] = quota
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=LANGUAGES, default=None)
    parser.add_argument("--pool_n", type=int, default=36)
    parser.add_argument("--validation_n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Print counts only, don't write output files")
    args = parser.parse_args()

    if not CANDIDATES_PATH.exists():
        raise SystemExit(f"Missing candidate manifest: {CANDIDATES_PATH}. Run scan_coding_capability_pool_candidates.py first.")
    candidates = json.loads(CANDIDATES_PATH.read_text())["languages"]
    languages = [args.language] if args.language else LANGUAGES

    report: dict[str, Any] = {
        "schema": "coding_capability_pool_mechanical_build_v1",
        "seed": args.seed,
        "pool_n_requested": args.pool_n,
        "validation_n_requested": args.validation_n,
        "achieved_counts": {},
        "quota_counts": {},
        "signature_parse_failed": [],
        "execution_rejected": [],
    }

    all_problems = []
    for lang in languages:
        print(f"Building {lang}...")
        problems = build_language(lang, candidates, args.pool_n, args.validation_n, args.seed, report)
        all_problems.extend(problems)
        achieved = report["achieved_counts"][lang]
        print(f"  {lang}: pool {achieved['pool']}/{args.pool_n}, validation {achieved['validation']}/{args.validation_n}")

    print(f"\nTotal typed_function problems built: {len(all_problems)}")
    print(f"Signature parse failures: {len(report['signature_parse_failed'])}")
    print(f"Execution rejections: {len(report['execution_rejected'])}")

    if not args.dry_run:
        OUT_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {OUT_REPORT_PATH.relative_to(ROOT)}")
        # NOTE: this only ever contains typed_function problems from the languages
        # requested this run. Step 3 (class_based / custom_driver) appends to this
        # same file rather than this script owning the full merge.
        existing: dict[str, Any] = {"schema": "coding_capability_pool_v1", "problems": []}
        if OUT_PROBLEMS_PATH.exists():
            try:
                existing = json.loads(OUT_PROBLEMS_PATH.read_text())
            except json.JSONDecodeError:
                pass
        # Only preserve rows belonging to the new pool_v1 schema (has language_id) for
        # a language NOT requested this run -- e.g. hand-authored custom_driver rows
        # from a Step 3 pass, or another language's mechanical run. Anything without
        # language_id is old-schema (the original 27 hand-written cross-language
        # problems) and must never survive a rebuild: this is a full replace, not a
        # merge with the pre-pool_v1 bank.
        kept = [
            p for p in existing.get("problems", [])
            if "language_id" in p and p["language_id"] not in languages
        ]
        existing["problems"] = kept + all_problems
        existing["schema"] = "coding_capability_pool_v1"
        existing["_comment"] = (
            "Problems sampled from the real coding-SFT source data "
            "(data/training/coding.write.<lang>/{pool,validation}.jsonl), not hand-written. "
            "Each problem is single-language (language_id) and carries its own `split`: "
            "'pool' rows can have been directly trained on (prepare_sft_dataset.py samples "
            "train.jsonl straight out of pool.jsonl); 'validation' rows are structurally "
            "guaranteed to never appear in any train.jsonl (see build_sft_pools.py). "
            "`kind` is 'typed_function' (graded against (args, expected) tuples mechanically "
            "derived by executing the row's own Magicoder reference solution -- see "
            "scripts/build_coding_capability_pool_eval.py) or 'custom_driver' (a hand-authored, "
            "execution-verified driver snippet appended to the model's completion, for "
            "class-based rows a typed-function harness can't express). Built by "
            "scripts/scan_coding_capability_pool_candidates.py + "
            "scripts/build_coding_capability_pool_eval.py; see "
            "data/reference/coding_capability_pool_validation_report.json for provenance."
        )
        # allow_nan=False: fail loudly at write time if a NaN/Infinity ever slips
        # through _matches_return_type's filtering, rather than silently emitting
        # invalid JSON (Python's json module accepts these as an extension; strict
        # JSON, and every other JSON parser, does not).
        OUT_PROBLEMS_PATH.write_text(json.dumps(existing, indent=2, allow_nan=False) + "\n")
        print(f"Wrote {OUT_PROBLEMS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
