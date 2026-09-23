#!/usr/bin/env python3
"""Scan the coding SFT pool/validation data for rows usable as capability-eval problems.

Classifies every row in data/training/coding.write.<lang>/{pool,validation}.jsonl as:

    typed_function - a single self-contained pure function (or, for java, a single
                      `public static` method) with primitive params, no I/O/randomness.
    class_based    - defines a class/struct with a small self-contained public
                      interface, compiles standalone.
    unsupported    - references an external framework/type the row doesn't itself
                      define (Spring, JPA, Lua C API, ...), does I/O, or is
                      otherwise non-deterministic.

Every non-unsupported row is then actually compiled/parsed standalone (javac / rustc
--emit=metadata / g++ -fsyntax-only / python ast+exec-at-module-level) to catch hidden
external dependencies the keyword heuristic misses. This is the feasibility check that
must run before data/reference/coding_capability_problems.json is rebuilt from real
pool/validation data. It answers "how many usable rows does each language actually
have" before any per-problem driver-authoring effort is spent.

Usage:
    python3 scripts/scan_coding_capability_pool_candidates.py
    python3 scripts/scan_coding_capability_pool_candidates.py --language java
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ["python", "java", "rust", "cpp"]
SPLITS = ["pool", "validation"]

OUT_PATH = ROOT / "data" / "reference" / "coding_capability_pool_candidates.json"

_IS_WINDOWS = platform.system() == "Windows"


def run_with_hard_timeout(cmd: list[str], cwd: str, timeout: float) -> dict:
    """subprocess.run(timeout=...) is not reliable here: on Windows, killing the
    direct child does not release output-pipe handles a grandchild process may have
    inherited (a JVM helper thread, a linker invoked by rustc/g++), and
    Popen.communicate()'s post-kill drain can then block indefinitely waiting for a
    pipe close that never comes. Force-kill the whole process tree via `taskkill /F
    /T` on timeout instead of relying on Popen.kill(), which only touches the
    immediate process. See build_coding_capability_pool_eval.py for the incident this
    was written after (a `java` candidate hung for hours past its 20s timeout)."""
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

# --- Non-determinism / I/O bans, shared across all four languages -------------------

BANNED_SNIPPETS = {
    "python": [
        "open(", "requests", "socket", "subprocess", "threading", "asyncio",
        "input(", "os.system", "random.", "time.time", "datetime.now",
        "uuid.uuid4", "__import__", "multiprocessing",
    ],
    "java": [
        "Scanner", "System.in", "Socket", "new Thread", "ExecutorService",
        "new Random(", "System.currentTimeMillis", "new File", "FileReader",
        "FileWriter", "System.nanoTime",
    ],
    "rust": [
        "std::net", "std::thread", "std::fs::File", "rand::", "SystemTime::now",
        "std::io::stdin", "std::env",
    ],
    "cpp": [
        "socket(", "fstream", "ifstream", "ofstream", "std::thread", "rand()",
        "time(NULL)", "time(nullptr)", "std::cin", "system(",
    ],
}

# Python: allow only stdlib imports (3.10+ exposes this directly).
STDLIB_MODULES = set(sys.stdlib_module_names)  # type: ignore[attr-defined]

# C++ standard headers we allow via #include.
CPP_STD_HEADERS = {
    "iostream", "vector", "string", "algorithm", "map", "unordered_map", "set",
    "unordered_set", "queue", "stack", "deque", "cmath", "cstdint", "cstring",
    "numeric", "utility", "tuple", "array", "list", "bitset", "climits",
    "functional", "sstream", "iomanip", "cctype", "cstdlib", "memory", "optional",
    "variant", "initializer_list",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


CODE_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*\n(.*?)\n?```", re.DOTALL)


def strip_code_fence(solution: str) -> str:
    """The pool/validation `solution` field is stored verbatim from Magicoder: a fenced
    code block followed by trailing prose (which itself often contains more inline
    backticks), e.g. '```java\\n...\\n```\\n\\nIn this solution, the `Foo` class ...'.
    Extract just the first fenced block; every downstream consumer needs the bare code."""
    match = CODE_FENCE_RE.search(solution)
    return match.group(1) if match else solution


def has_banned_snippet(solution: str, lang: str) -> str | None:
    for snippet in BANNED_SNIPPETS[lang]:
        if snippet in solution:
            return snippet
    return None


# --- Python classification -----------------------------------------------------------


def classify_python(solution: str) -> tuple[str, str]:
    import ast

    banned = has_banned_snippet(solution, "python")
    if banned:
        return "unsupported", f"banned snippet: {banned!r}"
    try:
        tree = ast.parse(solution)
    except SyntaxError as exc:
        return "unsupported", f"syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = (node.module or "").split(".")[0] if isinstance(node, ast.ImportFrom) else None
            names = [module] if module else [alias.name.split(".")[0] for alias in node.names]
            for name in names:
                if name and name not in STDLIB_MODULES:
                    return "unsupported", f"non-stdlib import: {name}"

    top_classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    top_funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if top_classes:
        return "class_based", f"{len(top_classes)} class(es), {len(top_funcs)} loose function(s)"
    if top_funcs:
        return "typed_function", f"{len(top_funcs)} top-level function(s)"
    return "unsupported", "no top-level function or class found"


def compile_check_python(solution: str, timeout: float = 10.0) -> str | None:
    """Execute the module in a subprocess with a timeout. Some Magicoder rows have
    top-level statements beyond def/class (not just definitions), so this can run
    real code -- a subprocess + timeout keeps a pathological row (infinite loop, huge
    computation) from hanging the whole scan, matching the sandboxing posture already
    used for java/rust/cpp below."""
    with tempfile.TemporaryDirectory(prefix="cap-scan-py-") as d:
        path = Path(d) / "candidate.py"
        path.write_text(solution, encoding="utf-8")
        result = run_with_hard_timeout([sys.executable, str(path)], d, timeout)
        if result["error"]:
            return result["error"]
        if result["returncode"] != 0:
            return (result["stderr"] or result["stdout"] or "python failed")[-500:]
    return None


# --- Java classification --------------------------------------------------------------

JAVA_IMPORT_RE = re.compile(r"^\s*import\s+([\w.]+)(?:\.\*)?\s*;", re.MULTILINE)
# Java class names are conventionally capitalized; requiring that avoids matching the
# bare word "class" inside comments/docstrings (e.g. "/** ... this class for X */"
# would otherwise be mis-parsed as `class for` -> a phantom class named "for").
JAVA_CLASS_RE = re.compile(r"\b(?:public\s+)?(?:final\s+)?(?:abstract\s+)?class\s+([A-Z]\w*)")
JAVA_PUBLIC_CLASS_RE = re.compile(r"\bpublic\s+(?:final\s+)?(?:abstract\s+)?class\s+([A-Z]\w*)")
JAVA_STATIC_METHOD_RE = re.compile(r"\bpublic\s+static\s+[\w<>\[\],\s]+?\s+\w+\s*\(")
JAVA_FIELD_RE = re.compile(r"^\s*(?:private|protected)\s+(?!static\s+final)[\w<>\[\]]+\s+\w+\s*[=;]", re.MULTILINE)

JAVA_LINE_COMMENT_RE = re.compile(r"//.*")
JAVA_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_java_comments(solution: str) -> str:
    return JAVA_LINE_COMMENT_RE.sub("", JAVA_BLOCK_COMMENT_RE.sub("", solution))


def classify_java(solution: str) -> tuple[str, str]:
    banned = has_banned_snippet(solution, "java")
    if banned:
        return "unsupported", f"banned snippet: {banned!r}"
    code = strip_java_comments(solution)
    for match in JAVA_IMPORT_RE.finditer(code):
        pkg = match.group(1)
        if not pkg.startswith("java."):
            return "unsupported", f"non-JDK import: {pkg}"
    classes = JAVA_CLASS_RE.findall(code)
    if not classes:
        return "unsupported", "no class definition found (java requires a wrapper class)"
    static_methods = JAVA_STATIC_METHOD_RE.findall(code)
    instance_fields = JAVA_FIELD_RE.findall(code)
    if len(classes) == 1 and static_methods and not instance_fields:
        return "typed_function", f"1 class, {len(static_methods)} public static method(s), 0 instance fields"
    return "class_based", f"{len(classes)} class(es), {len(static_methods)} static method(s), {len(instance_fields)} instance field(s)"


def compile_check_java(solution: str, timeout: float = 20.0) -> str | None:
    code = strip_java_comments(solution)
    # javac requires the filename to match the PUBLIC top-level class specifically, not
    # just any class in the file -- fall back to the first class only if none is public.
    match = JAVA_PUBLIC_CLASS_RE.search(code) or JAVA_CLASS_RE.search(code)
    if not match:
        return "no class found to name the file after"
    class_name = match.group(1)
    with tempfile.TemporaryDirectory(prefix="cap-scan-java-") as d:
        path = Path(d) / f"{class_name}.java"
        path.write_text(solution, encoding="utf-8")
        result = run_with_hard_timeout(["javac", "-nowarn", str(path)], d, timeout)
        if result["error"]:
            return result["error"]
        if result["returncode"] != 0:
            return (result["stderr"] or result["stdout"] or "javac failed")[:500]
    return None


# --- Rust classification ----------------------------------------------------------------

RUST_USE_RE = re.compile(r"^\s*use\s+([\w:]+)", re.MULTILINE)
RUST_ALLOWED_USE_PREFIXES = ("std", "core", "alloc")
RUST_STRUCT_RE = re.compile(r"\bstruct\s+\w+")
RUST_IMPL_RE = re.compile(r"\bimpl\b")
RUST_FN_RE = re.compile(r"\bfn\s+(\w+)\s*\(")


def classify_rust(solution: str) -> tuple[str, str]:
    banned = has_banned_snippet(solution, "rust")
    if banned:
        return "unsupported", f"banned snippet: {banned!r}"
    for match in RUST_USE_RE.finditer(solution):
        first_seg = match.group(1).split("::")[0]
        if first_seg not in RUST_ALLOWED_USE_PREFIXES:
            return "unsupported", f"external crate use: {match.group(1)}"
    has_struct = bool(RUST_STRUCT_RE.search(solution)) or bool(RUST_IMPL_RE.search(solution))
    fns = [m for m in RUST_FN_RE.findall(solution) if m != "main"]
    if has_struct:
        return "class_based", f"struct/impl present, {len(fns)} fn(s)"
    if fns:
        return "typed_function", f"{len(fns)} top-level fn(s)"
    return "unsupported", "no fn found"


def compile_check_rust(solution: str, timeout: float = 30.0) -> str | None:
    with tempfile.TemporaryDirectory(prefix="cap-scan-rs-") as d:
        src = Path(d) / "candidate.rs"
        src.write_text(solution, encoding="utf-8")
        out = Path(d) / "candidate.meta"
        result = run_with_hard_timeout(
            ["rustc", "--edition", "2021", "--crate-type", "lib", "--emit=metadata", "-o", str(out), str(src)],
            d, timeout,
        )
        if result["error"]:
            return result["error"]
        if result["returncode"] != 0:
            return (result["stderr"] or result["stdout"] or "rustc failed")[:500]
    return None


# --- C++ classification ------------------------------------------------------------------

CPP_INCLUDE_RE = re.compile(r'#include\s*[<"]([^">]+)[">]')
CPP_CLASS_RE = re.compile(r"\b(?:class|struct)\s+\w+\s*\{")
CPP_FN_DEF_RE = re.compile(r"^[\w:<>&*,\s]+\s+\w+\s*\([^;{}]*\)\s*\{", re.MULTILINE)


def classify_cpp(solution: str) -> tuple[str, str]:
    banned = has_banned_snippet(solution, "cpp")
    if banned:
        return "unsupported", f"banned snippet: {banned!r}"
    for match in CPP_INCLUDE_RE.finditer(solution):
        header = match.group(1)
        base = header.rsplit("/", 1)[-1].removesuffix(".h")
        if base not in CPP_STD_HEADERS and header not in CPP_STD_HEADERS:
            return "unsupported", f"non-standard include: {header}"
    has_class = bool(CPP_CLASS_RE.search(solution))
    fns = CPP_FN_DEF_RE.findall(solution)
    if has_class:
        return "class_based", f"class/struct present, {len(fns)} free function(s)"
    if fns:
        return "typed_function", f"{len(fns)} free function(s)"
    return "unsupported", "no function or class body found"


def compile_check_cpp(solution: str, timeout: float = 30.0) -> str | None:
    with tempfile.TemporaryDirectory(prefix="cap-scan-cpp-") as d:
        src = Path(d) / "candidate.cpp"
        src.write_text(solution, encoding="utf-8")
        result = run_with_hard_timeout(["g++", "-fsyntax-only", "-std=c++17", str(src)], d, timeout)
        if result["error"]:
            return result["error"]
        if result["returncode"] != 0:
            return (result["stderr"] or result["stdout"] or "g++ failed")[:500]
    return None


CLASSIFIERS = {
    "python": classify_python,
    "java": classify_java,
    "rust": classify_rust,
    "cpp": classify_cpp,
}
COMPILE_CHECKS = {
    "python": compile_check_python,
    "java": compile_check_java,
    "rust": compile_check_rust,
    "cpp": compile_check_cpp,
}


def scan_language(lang: str, split: str) -> dict[str, Any]:
    path = ROOT / "data" / "training" / f"coding.write.{lang}" / f"{split}.jsonl"
    rows = load_jsonl(path)
    result: dict[str, Any] = {
        "typed_function": [],
        "class_based": [],
        "rejected": {"unsupported": 0, "compile_failed": 0},
        "rejection_reasons": [],
        "total_rows": len(rows),
    }
    for row in rows:
        row_id = row["id"]
        solution = strip_code_fence(row["messages"][1]["content"])
        kind, reason = CLASSIFIERS[lang](solution)
        if kind == "unsupported":
            result["rejected"]["unsupported"] += 1
            result["rejection_reasons"].append({"id": row_id, "reason": reason})
            continue
        compile_error = COMPILE_CHECKS[lang](solution)
        if compile_error:
            result["rejected"]["compile_failed"] += 1
            result["rejection_reasons"].append(
                {"id": row_id, "reason": f"compile failed ({kind}): {compile_error}"}
            )
            continue
        result[kind].append({"id": row_id, "classification_note": reason})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=LANGUAGES, default=None)
    args = parser.parse_args()
    languages = [args.language] if args.language else LANGUAGES

    output: dict[str, Any] = {"languages": {}}
    print(f"{'lang':8} {'split':10} {'total':>6} {'typed_fn':>9} {'class':>6} {'unsupp':>7} {'compfail':>9}")
    for lang in languages:
        output["languages"][lang] = {}
        for split in SPLITS:
            scanned = scan_language(lang, split)
            output["languages"][lang][split] = scanned
            print(
                f"{lang:8} {split:10} {scanned['total_rows']:6d} "
                f"{len(scanned['typed_function']):9d} {len(scanned['class_based']):6d} "
                f"{scanned['rejected']['unsupported']:7d} {scanned['rejected']['compile_failed']:9d}"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
