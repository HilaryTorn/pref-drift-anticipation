#!/usr/bin/env python3
"""Functional-correctness robustness check for the coding trajectory.

This check uses problems sampled from the real coding-SFT source data
(data/training/coding.write.<lang>/{pool,validation}.jsonl).
~90% of each language's problems come from `pool.jsonl` (can have been
directly trained on, since prepare_sft_dataset.py samples train.jsonl straight out
of pool.jsonl) and ~10% from `validation.jsonl` (structurally guaranteed to never
appear in any train.jsonl -- see build_sft_pools.py). This measures in-distribution
vs. genuinely-held-out post-SFT performance, which a hand-written, corpus-independent
bank cannot. Every problem is single-language (no more shared-across-4-languages
rendering): each problem carries its own `language_id` and one of two `kind`s:

    typed_function - graded against fixed (args, expected) test tuples, mechanically
                      derived by executing the row's own real Magicoder reference
                      solution (see build_coding_capability_pool_eval.py).
    custom_driver   - a hand-authored, execution-verified driver snippet (for
                      class-based rows a typed-function harness can't express) that
                      gets appended to the model's completion and parses its own
                      CAPABILITY_RESULT line.

Grading executes real code (Python directly; Java/Rust/C++ via their compilers)
against each problem's stored expectations.

SECURITY: this executes model-generated code, so run this on a throwaway box, per
rl_training/rewards.py's existing caution about the same tradeoff.

Toolchains: Java/Rust/C++ need javac+java / rustc / g++ on PATH respectively.
A language whose toolchain is missing is skipped (and reported).

Usage:
    uv run scripts/score_coding_capability.py --model_key qwen35-4b-codingsft-python-step0-aws --dry-run
    uv run scripts/score_coding_capability.py --model_key qwen35-4b-codingsft-python-step125-aws
    uv run scripts/score_coding_capability.py --model_key <key> --languages python,java,rust,cpp

HuggingFace mode (bypasses config.yaml / a hand-run `vllm serve` endpoint entirely --
loads the base model, and optionally one LoRA adapter checkpoint, in-process via vLLM,
tensor-parallel across every visible GPU; see docs/serving-vllm-aws.md for the manual
equivalent this replaces). Run on the box with the GPUs -- this downloads real weights:
    uv run scripts/score_coding_capability.py --hf_base_repo prism-drift/qwen35-4b-m0-v4 --languages python
    uv run scripts/score_coding_capability.py --hf_base_repo prism-drift/qwen35-4b-m0-v4 --hf_adapter_repo prism-drift/qwen35-4b-m0-v4-phase-1-sft-adapters --hf_adapter_checkpoint checkpoint-40 --languages python
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_utilities.naming import default_results_dir, timestamped  # noqa: E402
from scripts.compute_utilities.utils import create_agent, create_hf_agent, generate_responses  # noqa: E402

DEFAULT_PROBLEMS_PATH = ROOT / "data" / "reference" / "coding_capability_problems.json"

LANGUAGES = ["python", "java", "rust", "cpp"]
DEFAULT_LANGUAGES = ["python", "java", "rust", "cpp"]
LANGUAGE_LABELS = {"python": "Python", "java": "Java", "rust": "Rust", "cpp": "C++"}

# Binaries that must resolve on PATH for a language to be scorable. Python needs
# nothing beyond the interpreter already running this script.
TOOLCHAIN_BINARIES = {
    "python": (),
    "java": ("javac", "java"),
    "rust": ("rustc",),
    "cpp": ("g++",),
}

PARAM_TYPE_DISPLAY = {
    "python": {
        "int": "int", "bool": "bool", "str": "str", "float": "float",
        "list_int": "list[int]", "list_str": "list[str]",
    },
    "java": {
        "int": "int", "bool": "boolean", "str": "String", "float": "double",
        "list_int": "int[]", "list_str": "String[]",
    },
    "rust": {
        "int": "i32", "bool": "bool", "str": "&str", "float": "f64",
        "list_int": "&[i32]", "list_str": "&[&str]",
    },
    "cpp": {
        "int": "int",
        "bool": "bool",
        "str": "std::string",
        "float": "double",
        "list_int": "std::vector<int>",
        "list_str": "std::vector<std::string>",
    },
}

RETURN_TYPE_DISPLAY = {
    "python": {"int": "int", "bool": "bool", "str": "str", "list_int": "list[int]", "list_str": "list[str]"},
    "java": {"int": "int", "bool": "boolean", "str": "String", "list_int": "int[]", "list_str": "String[]"},
    "rust": {"int": "i32", "bool": "bool", "str": "String", "list_int": "Vec<i32>", "list_str": "Vec<String>"},
    "cpp": {"int": "int", "bool": "bool", "str": "std::string",
            "list_int": "std::vector<int>", "list_str": "std::vector<std::string>"},
}

LANGUAGE_PROMPT_NOTE = {
    "python": "Return only the function (plus any needed imports), in one fenced code block. "
    "Do not call the function or print anything.",
    "java": "Define it as a `public static` method inside a class named exactly `Solution`. "
    "Return only that class, in one fenced code block. Do not include a `main` method.",
    "rust": "Return only the function (plus any needed `use` statements), in one fenced code "
    "block. Do not define a `main` function.",
    "cpp": "Return only the function (plus any needed `#include`s), in one fenced code block. "
    "Do not define a `main` function.",
}

CODE_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
FENCE_OPENER = re.compile(r"```[a-zA-Z0-9_+-]*\n")
RESULT_LINE = re.compile(r"CAPABILITY_RESULT (\d+)/(\d+)")

# Line-initial markers that mean "real code starts here", used only to recover an unfenced
# completion. Deliberately conservative: reasoning prose ("I'll use `fn main`...") does not start
# a line with these, so a completion that is pure reasoning falls through to no_code instead of
# being handed to a compiler.
CODE_START_MARKERS = {
    "python": ("def ", "import ", "from ", "class ", "@"),
    "java": ("public ", "private ", "static ", "class ", "import "),
    "rust": ("fn ", "pub fn ", "pub struct", "use ", "struct ", "impl ", "#["),
    "cpp": ("#include", "template", "struct ", "class ", "std::", "auto ", "int ", "bool "),
}


class UserInputError(ValueError):
    """Raised for CLI/data validation errors that should print without a traceback."""


def extract_code(completion: str, language: str) -> tuple[str, str]:
    """Return (code, extraction_status) for one completion.

    Status is one of:
        fenced         - a closed ``` block was found (the normal path; last block wins).
        unclosed_fence - the completion opened a fence and was cut before closing it. The body
                          after the opener is still graded, since a truncated-mid-function sample
                          is a real (failing) attempt rather than a measurement artifact.
        recovered      - no fence at all, but a line-initial code marker was found; graded from
                          that line on.
        no_code        - nothing code-shaped. Returns "" so the caller records the sample as a
                          non-attempt instead of compiling prose.

    The old behaviour was "no fence -> grade the whole completion", which fed reasoning text to
    rustc/g++/javac and produced compiler errors like ``error: prefix `doesn` is unknown``. Those
    surfaced as capability regressions when they were really generation-length failures, so the
    distinction between "wrote bad code" and "never got to the code" has to be recorded, not
    collapsed. See the 2026-08-20 rust arm result for the incident this was written after.
    """
    text = completion or ""
    blocks = CODE_FENCE.findall(text)
    if blocks:
        return blocks[-1].strip(), "fenced"

    # An odd number of fence markers means one was opened and never closed -- the signature of a
    # completion cut off by max_tokens partway through the code block.
    openers = list(FENCE_OPENER.finditer(text))
    if openers and text.count("```") % 2 == 1:
        body = text[openers[-1].end() :].strip()
        if body:
            return body, "unclosed_fence"

    markers = CODE_START_MARKERS[language]
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().startswith(markers):
            return "\n".join(lines[index:]).strip(), "recovered"

    return "", "no_code"


# Probing a toolchain means RUNNING it, not just locating it. macOS ships a `javac`/`java` shim that
# resolves on PATH with no JDK behind it and fails every compile with "Unable to locate a Java
# Runtime" -- so a which()-only check reports the toolchain present and every java problem then
# records as a model capability failure rather than a missing-toolchain skip. That is precisely the
# environment-masquerading-as-capability failure this eval was already burned by, so the guard has
# to execute the binary.
TOOLCHAIN_VERSION_FLAG = {"javac": "-version", "java": "-version", "rustc": "--version", "g++": "--version"}

_toolchain_probe_cache: dict[str, bool] = {}


def _binary_works(binary: str) -> bool:
    if binary in _toolchain_probe_cache:
        return _toolchain_probe_cache[binary]
    ok = False
    if shutil.which(binary) is not None:
        try:
            completed = subprocess.run(
                [binary, TOOLCHAIN_VERSION_FLAG.get(binary, "--version")],
                capture_output=True,
                timeout=30,
            )
            ok = completed.returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
    _toolchain_probe_cache[binary] = ok
    return ok


def toolchain_missing(lang: str) -> list[str]:
    """Binaries that are absent OR present-but-non-functional. Both mean the language cannot be
    scored, and conflating them with a working toolchain silently fabricates capability failures."""
    return [b for b in TOOLCHAIN_BINARIES[lang] if not _binary_works(b)]


def infer_language_from_model_key(model_key: str | None) -> str | None:
    """Recover the trained arm from a phase-1 checkpoint key, e.g.
    qwen35-4b-codingsft-python-step40-aws -> "python". Lets a bare --model_key
    default to scoring only the language that checkpoint was trained on, so the
    default invocation stays cheap and on-topic instead of scoring all four."""
    if not model_key:
        return None
    for lang in LANGUAGES:
        if f"-{lang}-" in model_key or model_key.endswith(f"-{lang}"):
            return lang
    return None


def resolve_hf_adapter_path(
    adapter_repo: str,
    language: str,
    checkpoint: str,
    cache_dir: str | None,
    revision: str | None,
) -> str:
    """Pull one PEFT adapter checkpoint out of a `*-phase-1-sft-adapters` Hub repo and
    return its local directory.

    Those repos hold every language arm and every training-step checkpoint in one repo
    (`coding.write.<lang>/checkpoint-<n>/...`, plus the final adapter at
    `coding.write.<lang>/...`), so there's no single adapter to resolve by repo id alone --
    unlike `--hf_base_repo`, which vLLM downloads whole and automatically. `allow_patterns`
    limits the download to just the two files LoRA loading actually needs, mirroring the
    `snapshot_download(..., allow_patterns=[...])` recipe in docs/serving-vllm-aws.md rather
    than pulling every arm/checkpoint (each ~130 MB) in the repo.
    """
    from huggingface_hub import snapshot_download

    subdir = f"coding.write.{language}"
    checkpoint_part = "" if checkpoint == "final" else checkpoint
    patterns = [
        f"{subdir}/{checkpoint_part}/adapter_config.json".replace("//", "/"),
        f"{subdir}/{checkpoint_part}/adapter_model.safetensors".replace("//", "/"),
    ]
    snapshot_dir = snapshot_download(
        repo_id=adapter_repo,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=patterns,
    )
    adapter_path = str(Path(snapshot_dir) / subdir / checkpoint_part)
    if not (Path(adapter_path) / "adapter_config.json").exists():
        raise UserInputError(
            f"No adapter found at {adapter_repo}/{subdir}"
            f"{'/' + checkpoint if checkpoint != 'final' else ''} "
            f"(looked in {adapter_path})"
        )
    return adapter_path


def render_literal(value, type_: str, lang: str) -> str:
    if type_ == "bool":
        if lang == "python":
            return "True" if value else "False"
        return "true" if value else "false"
    if type_ == "int":
        return str(value)
    if type_ == "float":
        return str(value)
    if type_ == "str":
        # Test strings are plain ASCII with no quotes/backslashes, so a Python
        # double-quoted literal is valid source text in all four languages.
        return json.dumps(value)
    if type_ == "list_int":
        items = ", ".join(str(v) for v in value)
        if lang == "python":
            return f"[{items}]"
        if lang == "java":
            return f"new int[]{{{items}}}"
        if lang == "rust":
            return f"&[{items}]"
        if lang == "cpp":
            return f"std::vector<int>{{{items}}}"
    if type_ == "list_str":
        items = ", ".join(json.dumps(v) for v in value)
        if lang == "python":
            return f"[{items}]"
        if lang == "java":
            return f"new String[]{{{items}}}"
        if lang == "rust":
            return f"&[{items}]"
        if lang == "cpp":
            return f"std::vector<std::string>{{{items}}}"
    raise ValueError(f"Unsupported type {type_!r} for language {lang!r}")


def build_signature(problem: dict, lang: str) -> str:
    names, types = problem["param_names"], problem["param_types"]
    ret = RETURN_TYPE_DISPLAY[lang][problem["return_type"]]
    entry_point = problem["entry_point"]
    if lang == "python":
        params = ", ".join(
            f"{n}: {PARAM_TYPE_DISPLAY[lang][t]}" for n, t in zip(names, types)
        )
        return f"def {entry_point}({params}) -> {ret}:"
    if lang == "java":
        params = ", ".join(
            f"{PARAM_TYPE_DISPLAY[lang][t]} {n}" for n, t in zip(names, types)
        )
        return f"public static {ret} {entry_point}({params})"
    if lang == "rust":
        params = ", ".join(
            f"{n}: {PARAM_TYPE_DISPLAY[lang][t]}" for n, t in zip(names, types)
        )
        return f"fn {entry_point}({params}) -> {ret}"
    if lang == "cpp":
        params = ", ".join(
            f"{PARAM_TYPE_DISPLAY[lang][t]} {n}" for n, t in zip(names, types)
        )
        return f"{ret} {entry_point}({params})"
    raise ValueError(lang)


def build_prompt(problem: dict) -> str:
    lang = problem["language_id"]
    if problem["kind"] == "typed_function":
        return (
            f"Write a {LANGUAGE_LABELS[lang]} function with EXACTLY this signature:\n\n"
            f"{build_signature(problem, lang)}\n\n"
            f"Task: {problem['prompt']}\n\n"
            f"{LANGUAGE_PROMPT_NOTE[lang]}"
        )
    if problem["kind"] == "custom_driver":
        return (
            f"Task ({LANGUAGE_LABELS[lang]}): {problem['prompt']}\n\n"
            f"Implement exactly the interface described above (entry point: "
            f"`{problem['entry_point']}`) so it can be called as described.\n\n"
            f"{LANGUAGE_PROMPT_NOTE[lang]}"
        )
    raise ValueError(f"Unknown problem kind {problem['kind']!r}")


# --- Execution ---------------------------------------------------------------


def _run_process(cmd: list[str], cwd: str, timeout: float) -> dict:
    """subprocess.run(timeout=...) is not reliable here: on Windows, killing the
    direct child does not release output-pipe handles a grandchild process may have
    inherited (a JVM helper thread, a linker invoked by rustc/g++), and
    Popen.communicate()'s post-kill drain can then block indefinitely waiting for a
    pipe close that never comes -- this executes model-generated code, so a hang here
    would stall a real scoring run, not just a one-off data-build script. Force-kill
    the whole process tree via `taskkill /F /T` on timeout instead of relying on
    Popen.kill(), which only touches the immediate process. See
    scripts/build_coding_capability_pool_eval.py for the incident this was written
    after (a `java` candidate hung for hours past its 20s timeout)."""
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        if os.name == "nt":
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
        return {"stdout": stdout or "", "error": f"timeout running {cmd[0]}"}
    except OSError as exc:
        return {"stdout": "", "error": f"failed to run {cmd[0]}: {exc}"}
    if returncode != 0 and not RESULT_LINE.search(stdout or ""):
        return {
            "stdout": stdout or "",
            "error": f"{cmd[0]} exited {returncode}: {(stderr or '')[:1000]}",
        }
    return {"stdout": stdout or "", "error": None}


def _score_from_stdout(stdout: str, total: int, error: str | None) -> dict:
    if error:
        return {"passed": 0, "total": total, "error": error}
    match = RESULT_LINE.search(stdout)
    if not match:
        return {
            "passed": 0,
            "total": total,
            "error": f"unparseable harness output: {stdout[:500]!r}",
        }
    return {"passed": int(match.group(1)), "total": int(match.group(2)), "error": None}


def _python_checks(problem: dict) -> list[str]:
    lines = []
    entry_point = problem["entry_point"]
    for args, expected in problem["tests"]:
        args_lit = ", ".join(
            render_literal(a, t, "python") for a, t in zip(args, problem["param_types"])
        )
        expected_lit = render_literal(expected, problem["return_type"], "python")
        lines.append(
            f"__cap_passed += 1 if __cap_safe_call({entry_point}, {args_lit}) == {expected_lit} else 0"
        )
    return lines


def run_python(code: str, problem: dict, timeout: float) -> dict:
    total = len(problem["tests"])
    script = "\n".join(
        [
            code,
            "",
            "def __cap_safe_call(_fn, *_args):",
            "    try:",
            "        return _fn(*_args)",
            "    except Exception:",
            "        return object()",
            "",
            "__cap_passed = 0",
            *_python_checks(problem),
            f'print(f"CAPABILITY_RESULT {{__cap_passed}}/{total}")',
        ]
    )
    with tempfile.TemporaryDirectory(prefix="coding-cap-py-") as d:
        path = Path(d) / "solution.py"
        path.write_text(script)
        result = _run_process([sys.executable, str(path)], d, timeout)
        return _score_from_stdout(result["stdout"], total, result["error"])


def run_java(code: str, problem: dict, timeout: float) -> dict:
    total = len(problem["tests"])
    checks = []
    for args, expected in problem["tests"]:
        args_lit = ", ".join(
            render_literal(a, t, "java") for a, t in zip(args, problem["param_types"])
        )
        expected_lit = render_literal(expected, problem["return_type"], "java")
        # deepEquals, not equals: Java arrays inherit Object.equals, which is REFERENCE equality, so
        # `Objects.equals(new int[]{1,2}, new int[]{1,2})` is false and every list-returning problem
        # would grade 0/N no matter how correct the answer -- silently, with no error to notice.
        # deepEquals dispatches to Arrays.deepEquals for arrays and falls back to equals otherwise,
        # so scalar return types behave exactly as before.
        checks.append(
            f"        total++; if (java.util.Objects.deepEquals("
            f'_safe(() -> Solution.{problem["entry_point"]}({args_lit})), {expected_lit})) passed++;'
        )
    harness = (
        "\n\ninterface _Call { Object run(); }\n"
        "class _Harness {\n"
        "    static Object _safe(_Call c) {\n"
        "        try { return c.run(); } catch (Throwable e) { return new Object(); }\n"
        "    }\n"
        "    public static void main(String[] args) {\n"
        "        int passed = 0;\n"
        "        int total = 0;\n" + "\n".join(checks) + "\n"
        '        System.out.println("CAPABILITY_RESULT " + passed + "/" + total);\n'
        "    }\n"
        "}\n"
    )
    with tempfile.TemporaryDirectory(prefix="coding-cap-java-") as d:
        (Path(d) / "Solution.java").write_text(code + harness)
        compiled = _run_process(["javac", "Solution.java"], d, timeout)
        if compiled["error"]:
            return {
                "passed": 0,
                "total": total,
                "error": f"compile: {compiled['error']}",
            }
        result = _run_process(["java", "_Harness"], d, timeout)
        return _score_from_stdout(result["stdout"], total, result["error"])


def run_rust(code: str, problem: dict, timeout: float) -> dict:
    total = len(problem["tests"])
    checks = []
    for args, expected in problem["tests"]:
        args_lit = ", ".join(
            render_literal(a, t, "rust") for a, t in zip(args, problem["param_types"])
        )
        expected_lit = render_literal(expected, problem["return_type"], "rust")
        checks.append(
            f"    total += 1; if std::panic::catch_unwind(|| {problem['entry_point']}({args_lit}))"
            f".map(|v| v == {expected_lit}).unwrap_or(false) {{ passed += 1; }}"
        )
    harness = (
        "\n\nfn main() {\n"
        "    let mut passed: i32 = 0;\n"
        "    let mut total: i32 = 0;\n" + "\n".join(checks) + "\n"
        '    println!("CAPABILITY_RESULT {}/{}", passed, total);\n'
        "}\n"
    )
    bin_name = "solution_bin.exe" if os.name == "nt" else "solution_bin"
    with tempfile.TemporaryDirectory(prefix="coding-cap-rs-") as d:
        (Path(d) / "solution.rs").write_text(code + harness)
        compiled = _run_process(
            ["rustc", "--edition", "2021", "-O", "solution.rs", "-o", bin_name],
            d,
            timeout,
        )
        if compiled["error"]:
            return {
                "passed": 0,
                "total": total,
                "error": f"compile: {compiled['error']}",
            }
        result = _run_process([str(Path(d) / bin_name)], d, timeout)
        return _score_from_stdout(result["stdout"], total, result["error"])


def run_cpp(code: str, problem: dict, timeout: float) -> dict:
    total = len(problem["tests"])
    checks = []
    for args, expected in problem["tests"]:
        args_lit = ", ".join(
            render_literal(a, t, "cpp") for a, t in zip(args, problem["param_types"])
        )
        expected_lit = render_literal(expected, problem["return_type"], "cpp")
        checks.append(f"    _CAP_CHECK({problem['entry_point']}({args_lit}), {expected_lit});")
    harness = (
        "\n\n#include <iostream>\n"
        "int _cap_passed = 0, _cap_total = 0;\n"
        "#define _CAP_CHECK(expr, expected) do { _cap_total++; "
        "try { if ((expr) == (expected)) _cap_passed++; } catch (...) {} } while (0)\n"
        "int main() {\n" + "\n".join(checks) + "\n"
        '    std::cout << "CAPABILITY_RESULT " << _cap_passed << "/" << _cap_total << std::endl;\n'
        "    return 0;\n"
        "}\n"
    )
    bin_name = "solution_bin.exe" if os.name == "nt" else "solution_bin"
    with tempfile.TemporaryDirectory(prefix="coding-cap-cpp-") as d:
        (Path(d) / "solution.cpp").write_text(code + harness)
        compiled = _run_process(
            ["g++", "-O0", "-std=c++17", "solution.cpp", "-o", bin_name], d, timeout
        )
        if compiled["error"]:
            return {
                "passed": 0,
                "total": total,
                "error": f"compile: {compiled['error']}",
            }
        result = _run_process([str(Path(d) / bin_name)], d, timeout)
        return _score_from_stdout(result["stdout"], total, result["error"])


JAVA_PUBLIC_CLASS_RE = re.compile(r"\bpublic\s+(?:final\s+)?(?:abstract\s+)?class\s+([A-Z]\w*)")


def run_custom_driver_python(code: str, problem: dict, timeout: float) -> dict:
    script = code + "\n\n" + problem["driver"]
    with tempfile.TemporaryDirectory(prefix="coding-cap-py-") as d:
        path = Path(d) / "solution.py"
        path.write_text(script)
        result = _run_process([sys.executable, str(path)], d, timeout)
        return _score_from_stdout(result["stdout"], 0, result["error"])


def run_custom_driver_java(code: str, problem: dict, timeout: float) -> dict:
    # Unlike typed_function (always "Solution"), a custom_driver's expected public
    # class name varies per problem (e.g. "FileManager") -- detect it from the
    # model's own completion so the file gets named correctly for javac.
    match = JAVA_PUBLIC_CLASS_RE.search(code)
    filename = f"{match.group(1)}.java" if match else "Solution.java"
    source = code + "\n\n" + problem["driver"]
    with tempfile.TemporaryDirectory(prefix="coding-cap-java-") as d:
        (Path(d) / filename).write_text(source)
        compiled = _run_process(["javac", filename], d, timeout)
        if compiled["error"]:
            return {"passed": 0, "total": 0, "error": f"compile: {compiled['error']}"}
        # By convention, every custom_driver defines its main entry point in a class
        # named `_Harness` (mirrors the typed_function harness below), independent of
        # whatever public class the model itself defines.
        result = _run_process(["java", "_Harness"], d, timeout)
        return _score_from_stdout(result["stdout"], 0, result["error"])


def run_custom_driver_rust(code: str, problem: dict, timeout: float) -> dict:
    source = code + "\n\n" + problem["driver"]
    bin_name = "solution_bin.exe" if os.name == "nt" else "solution_bin"
    with tempfile.TemporaryDirectory(prefix="coding-cap-rs-") as d:
        (Path(d) / "solution.rs").write_text(source)
        compiled = _run_process(
            ["rustc", "--edition", "2021", "-O", "solution.rs", "-o", bin_name], d, timeout
        )
        if compiled["error"]:
            return {"passed": 0, "total": 0, "error": f"compile: {compiled['error']}"}
        result = _run_process([str(Path(d) / bin_name)], d, timeout)
        return _score_from_stdout(result["stdout"], 0, result["error"])


def run_custom_driver_cpp(code: str, problem: dict, timeout: float) -> dict:
    source = code + "\n\n" + problem["driver"]
    bin_name = "solution_bin.exe" if os.name == "nt" else "solution_bin"
    with tempfile.TemporaryDirectory(prefix="coding-cap-cpp-") as d:
        (Path(d) / "solution.cpp").write_text(source)
        compiled = _run_process(
            ["g++", "-O0", "-std=c++17", "solution.cpp", "-o", bin_name], d, timeout
        )
        if compiled["error"]:
            return {"passed": 0, "total": 0, "error": f"compile: {compiled['error']}"}
        result = _run_process([str(Path(d) / bin_name)], d, timeout)
        return _score_from_stdout(result["stdout"], 0, result["error"])


EXECUTORS = {"python": run_python, "java": run_java, "rust": run_rust, "cpp": run_cpp}
CUSTOM_DRIVER_EXECUTORS = {
    "python": run_custom_driver_python,
    "java": run_custom_driver_java,
    "rust": run_custom_driver_rust,
    "cpp": run_custom_driver_cpp,
}


def score_problem(code: str, problem: dict, timeout: float) -> dict:
    lang = problem["language_id"]
    if problem["kind"] == "typed_function":
        return EXECUTORS[lang](code, problem, timeout)
    if problem["kind"] == "custom_driver":
        return CUSTOM_DRIVER_EXECUTORS[lang](code, problem, timeout)
    raise ValueError(f"Unknown problem kind {problem['kind']!r}")


# --- Orchestration -------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    problems = json.loads(Path(args.problems_path).read_text())["problems"]

    if args.hf_base_repo and args.model_key:
        raise UserInputError("--hf_base_repo and --model_key are mutually exclusive.")
    if args.hf_adapter_repo and not args.hf_base_repo:
        raise UserInputError("--hf_adapter_repo requires --hf_base_repo.")

    if args.languages:
        requested = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
        unknown = [lang for lang in requested if lang not in LANGUAGES]
        if unknown:
            raise UserInputError(f"Unknown language(s) {unknown}; known: {LANGUAGES}")
    elif args.hf_adapter_repo:
        raise UserInputError(
            "--hf_adapter_repo needs --languages naming exactly the one arm to load -- "
            "there's no --model_key to infer it from in HuggingFace mode."
        )
    else:
        inferred = infer_language_from_model_key(args.model_key)
        requested = [inferred] if inferred else DEFAULT_LANGUAGES
        print(
            f"No --languages given; defaulting to {requested} "
            f"({'inferred from --model_key' if inferred else 'no language inferable from --model_key'})."
        )

    if args.hf_adapter_repo and len(requested) != 1:
        raise UserInputError(
            "--hf_adapter_repo loads one LoRA adapter for one arm; --languages resolved to "
            f"{requested}, need exactly one."
        )

    skipped = {}
    available = []
    for lang in requested:
        missing = toolchain_missing(lang)
        if missing:
            skipped[lang] = f"missing toolchain binaries on PATH: {', '.join(missing)}"
            print(f"Skipping {LANGUAGE_LABELS[lang]}: {skipped[lang]}")
        else:
            available.append(lang)
    if not available:
        raise UserInputError(
            "No requested language has its toolchain available; nothing to run."
        )

    if args.dry_run:
        for lang in available:
            for problem in problems:
                if problem["language_id"] != lang:
                    continue
                print(f"\n=== {lang} / {problem['id']} ({problem['kind']}, split={problem['split']}) ===")
                print(build_prompt(problem))
        return

    if not args.model_key and not args.hf_base_repo:
        raise UserInputError("--model_key or --hf_base_repo is required unless --dry-run")

    if args.hf_base_repo:
        adapter_path = None
        if args.hf_adapter_repo:
            adapter_path = resolve_hf_adapter_path(
                args.hf_adapter_repo,
                available[0],
                args.hf_adapter_checkpoint,
                args.hf_cache_dir,
                args.hf_revision,
            )
            print(
                f"Loaded adapter coding.write.{available[0]}/{args.hf_adapter_checkpoint} "
                f"from {args.hf_adapter_repo} -> {adapter_path}"
            )
        agent = create_hf_agent(
            base_model=args.hf_base_repo,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            lora_path=adapter_path,
            cache_dir=args.hf_cache_dir,
        )
        run_label = args.hf_base_repo.rstrip("/").split("/")[-1]
        if args.hf_adapter_repo:
            run_label += f"-{available[0]}-{args.hf_adapter_checkpoint}"
    else:
        agent = create_agent(
            model_key=args.model_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            concurrency_limit=args.concurrency_limit,
            base_timeout=args.request_timeout,
        )
        run_label = args.model_key

    save_dir = Path(
        args.save_dir or default_results_dir(run_label, "coding_capability")
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    run_base = timestamped(
        args.save_suffix or "coding_capability", not args.no_timestamp
    )

    per_language_results = {}
    # Raw completions, persisted as a sidecar the way the elicitation batteries do. Until
    # 2026-08-21 this script saved only post-extraction text, so when the extractor turned out to
    # be grading prose there was nothing to re-analyse -- diagnosing it needed a fresh GPU run.
    # Anything that changes only extraction or parsing can now be re-derived offline from this file.
    raw_records: list[dict[str, Any]] = []
    for lang in available:
        lang_problems = [p for p in problems if p["language_id"] == lang]
        if not lang_problems:
            print(f"  {LANGUAGE_LABELS[lang]}: no problems in bank for this language, skipping")
            continue
        prompts = [build_prompt(problem) for problem in lang_problems]
        print(
            f"\nScoring {LANGUAGE_LABELS[lang]}: {len(prompts)} problem(s) x K={args.k} sample(s)"
        )
        responses = await generate_responses(
            agent, prompts, K=args.k, timeout=args.request_timeout
        )

        problem_results = []
        for idx, problem in enumerate(lang_problems):
            samples = []
            for sample_idx, completion in enumerate(responses.get(idx, [])):
                code, extraction_status = extract_code(completion, lang)
                raw_records.append(
                    {
                        "language_id": lang,
                        "problem_id": problem["id"],
                        "sample_index": sample_idx,
                        "completion": completion,
                        "extraction_status": extraction_status,
                    }
                )
                if extraction_status == "no_code":
                    # Never hand prose to a compiler: record the non-attempt and move on, so it
                    # cannot masquerade as a syntax error in the failure breakdown.
                    outcome = {"passed": 0, "total": 0, "error": "no_code_extracted"}
                else:
                    outcome = score_problem(code, problem, args.exec_timeout)
                samples.append(
                    {
                        "passed": outcome["passed"],
                        "total": outcome["total"],
                        "all_passed": outcome["total"] > 0
                        and outcome["passed"] == outcome["total"],
                        "error": outcome["error"],
                        "extracted_code": code,
                        "extraction_status": extraction_status,
                        "completion_chars": len(completion or ""),
                        # Heuristic, not the API's finish_reason: the shared generate_responses in
                        # compute_utilities returns bare strings, and plumbing finish_reason through
                        # it would touch every battery in the project. An unclosed fence or a
                        # completion with no code in it at all is overwhelmingly a max_tokens cut;
                        # a refusal would also land here, so treat this as a rate to watch rather
                        # than a per-sample verdict.
                        "likely_truncated": extraction_status in ("unclosed_fence", "no_code"),
                    }
                )
            n = len(samples)
            full_pass_rate = (
                (sum(1 for s in samples if s["all_passed"]) / n) if n else 0.0
            )
            mean_test_fraction = (
                (
                    sum(
                        (s["passed"] / s["total"]) if s["total"] else 0.0
                        for s in samples
                    )
                    / n
                )
                if n
                else 0.0
            )
            problem_results.append(
                {
                    "id": problem["id"],
                    "kind": problem["kind"],
                    "split": problem["split"],
                    "prompt": problem["prompt"],
                    "n_samples": n,
                    "full_pass_rate": full_pass_rate,
                    "mean_test_fraction": mean_test_fraction,
                    "samples": samples,
                }
            )

        lang_full = sum(p["full_pass_rate"] for p in problem_results) / len(
            problem_results
        )
        lang_frac = sum(p["mean_test_fraction"] for p in problem_results) / len(
            problem_results
        )
        pool_results = [p for p in problem_results if p["split"] == "pool"]
        validation_results = [p for p in problem_results if p["split"] == "validation"]
        # Generation health, reported next to the score rather than buried per-sample. A pass rate
        # that fell because the model never finished writing code is a different finding from one
        # that fell because the code was wrong, and only this rate separates them.
        all_samples = [s for p in problem_results for s in p["samples"]]
        extraction_counts: dict[str, int] = {}
        for sample in all_samples:
            status = sample["extraction_status"]
            extraction_counts[status] = extraction_counts.get(status, 0) + 1
        truncated_rate = (
            sum(1 for s in all_samples if s["likely_truncated"]) / len(all_samples)
            if all_samples
            else 0.0
        )
        # Two numbers, never one. A truncated sample scores 0 exactly like a wrong
        # answer, so the headline rate is the PRODUCT of "did it finish" and "was it
        # right" — and training moves the first factor far more easily than the
        # second. On the RL held-out set an apparently significant +2.8pp gain was
        # +3.5pp termination and -0.7pp quality once split this way.
        #
        # These are sample-level while full_pass_rate is problem-level (mean over K,
        # then over problems), so they are a companion diagnostic rather than a
        # redefinition and will not match it exactly.
        finished = [s for s in all_samples if not s["likely_truncated"]]
        conditional_full = (
            sum(1 for s in finished if s["all_passed"]) / len(finished) if finished else None
        )
        conditional_frac = (
            sum(s["passed"] / s["total"] for s in finished if s["total"]) / len(finished)
            if finished
            else None
        )
        per_language_results[lang] = {
            "full_pass_rate": lang_full,
            "mean_test_fraction": lang_frac,
            "extraction_status_counts": extraction_counts,
            "likely_truncated_rate": truncated_rate,
            "termination_rate": 1.0 - truncated_rate,
            "full_pass_rate_given_termination": conditional_full,
            "mean_test_fraction_given_termination": conditional_frac,
            "n_samples": len(all_samples),
            "n_terminated": len(finished),
            "pool_full_pass_rate": (
                sum(p["full_pass_rate"] for p in pool_results) / len(pool_results) if pool_results else None
            ),
            "validation_full_pass_rate": (
                sum(p["full_pass_rate"] for p in validation_results) / len(validation_results)
                if validation_results
                else None
            ),
            "problems": problem_results,
        }
        print(
            f"  {LANGUAGE_LABELS[lang]}: full_pass_rate={lang_full:.2f} mean_test_fraction={lang_frac:.2f} "
            f"(pool={per_language_results[lang]['pool_full_pass_rate']}, "
            f"validation={per_language_results[lang]['validation_full_pass_rate']})"
        )
        print(f"    extraction={extraction_counts} likely_truncated={truncated_rate:.1%}")
        if finished:
            print(
                f"    finished={1 - truncated_rate:.1%} of {len(all_samples)} samples; "
                f"given termination: full_pass={conditional_full:.2f} "
                f"mean_test_fraction={conditional_frac:.2f}"
            )
        if truncated_rate > 0.05:
            print(
                f"    !! {truncated_rate:.1%} of {LANGUAGE_LABELS[lang]} samples produced no usable code. "
                f"Raise --max_tokens (currently {args.max_tokens}) before reading this score as capability."
            )

    summary = {
        "model_key": run_label,
        "hf_base_repo": args.hf_base_repo,
        "hf_adapter_repo": args.hf_adapter_repo,
        "hf_adapter_checkpoint": args.hf_adapter_checkpoint if args.hf_adapter_repo else None,
        "k": args.k,
        "temperature": args.temperature,
        "problems_path": str(args.problems_path),
        "languages_requested": requested,
        "languages_scored": available,
        "languages_skipped": skipped,
        "results": per_language_results,
    }
    out_path = save_dir / f"coding_capability_{run_base}.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out_path}")

    raw_path = save_dir / f"raw_responses_coding_capability_{run_base}.jsonl"
    with raw_path.open("w") as handle:
        for record in raw_records:
            handle.write(json.dumps(record) + "\n")
    print(f"Wrote {raw_path} ({len(raw_records)} completions)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model_key",
        default=None,
        help="Key in config.yaml; omit only with --dry-run or --hf_base_repo",
    )
    parser.add_argument(
        "--hf_base_repo",
        default=None,
        help="HF repo id (or local path) of a base model, e.g. prism-drift/qwen35-4b-m0-v4. "
        "Alternative to --model_key: loads the model in-process via vLLM (downloading it "
        "from the Hub if needed), tensor-parallel across every visible GPU -- no config.yaml "
        "entry or `vllm serve` endpoint required. Mutually exclusive with --model_key.",
    )
    parser.add_argument(
        "--hf_adapter_repo",
        default=None,
        help="HF repo id of a phase-1 SFT adapters repo, e.g. "
        "prism-drift/qwen35-4b-m0-v4-phase-1-sft-adapters. Requires --hf_base_repo. These "
        "repos hold every language arm (coding.write.<lang>/) and every training-step "
        "checkpoint in one repo, so --languages must resolve to exactly the one arm to load.",
    )
    parser.add_argument(
        "--hf_adapter_checkpoint",
        default="final",
        help="Which checkpoint to load from --hf_adapter_repo's arm subfolder: 'final' (the "
        "arm-root adapter, the run's last saved weights) or a specific 'checkpoint-N'.",
    )
    parser.add_argument(
        "--hf_revision",
        default=None,
        help="Optional pinned revision/tag, applied to both --hf_base_repo and "
        "--hf_adapter_repo.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="Local download directory for --hf_base_repo/--hf_adapter_repo. Default: the "
        "standard HuggingFace cache (~/.cache/huggingface).",
    )
    parser.add_argument(
        "--problems_path",
        default=str(DEFAULT_PROBLEMS_PATH),
        help="Path to the problem-bank JSON",
    )
    parser.add_argument(
        "--languages",
        default=None,
        help="Comma-separated subset of python,java,rust,cpp. Default: infer the single "
        "trained arm from --model_key (e.g. '...-python-step40-...' -> python); "
        "falls back to all four languages if nothing can be inferred.",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Samples per problem (pass-rate estimate)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max generation tokens per sample. Was 1024 through 2026-08-20, which truncated "
        "reasoning-on checkpoints mid-thought before they emitted any code -- those samples were "
        "then graded as compiler errors and read as capability regressions (worst on rust). Scores "
        "taken at different values of this flag are NOT comparable; re-score every arm and every "
        "baseline together after changing it.",
    )
    parser.add_argument(
        "--concurrency_limit",
        type=int,
        default=20,
        help="Client-side in-flight request cap",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=60.0,
        help="Per-LLM-request base timeout (s)",
    )
    parser.add_argument(
        "--exec_timeout",
        type=float,
        default=20.0,
        help="Per-sample compile+run timeout (s)",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="Output dir. Default: results/<model_key>/coding_capability",
    )
    parser.add_argument("--save_suffix", default="coding_capability")
    parser.add_argument(
        "--no-timestamp",
        dest="no_timestamp",
        action="store_true",
        help="Write to a fixed filename with no UTC timestamp; a re-run then overwrites it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts for available languages, query nothing",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    try:
        asyncio.run(run(cli_args))
    except UserInputError as exc:
        raise SystemExit(str(exc)) from exc
