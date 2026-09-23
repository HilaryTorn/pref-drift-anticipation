"""Language backends for the multi-language verifiable coding reward.

The Python pipeline in ``rl_training/rewards.py`` hardcodes one executor: write
``solution.py``, run it with ``sys.executable``. Everything ABOVE that point --
output normalization, comparison modes, test subsampling, the flattened worker
pool, truncation semantics -- is already language-agnostic. This module is the
dispatch point that was missing, so the reward definition itself is untouched.

Two properties matter and neither is free:

**Compile once per program, not once per test case.** ``_io_case_passes`` is
called once per test, so a naive swap would invoke ``rustc`` 12 times for a
single completion. Builds are therefore cached by (language, sha256(source))
and shared across the test cases of a completion -- and across duplicate
completions within a GRPO group, which happen often at low temperature.

**A cached build artifact is not a shared working directory.** The Python path
gives every test case a fresh temp dir that doubles as ``HOME``/``TMPDIR``/cwd,
so a program that writes files cannot leak state into the next case. Caching the
whole directory would quietly break that. So the build artifact (read-only, in
the cache) and the run workdir (fresh per case, disposable) are separate.

Resource limits mirror the Python path exactly: RLIMIT_CPU, RLIMIT_FSIZE and
RLIMIT_NOFILE, applied by a small launcher. Address space is deliberately NOT
capped -- the multi-LCB harness found that ``limit_memory`` breaks native
binaries on macOS, and the Python path does not cap it either.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

_VERIFIER_FILE_BYTES = 16 * 1024 * 1024
_VERIFIER_OPEN_FILES = 64

# rustc on a cold cache is slow; a pathological program can be slower still.
# This bounds the build, separately from the per-test run timeout.
BUILD_TIMEOUT = 30.0

# How many compiled artifacts to keep. A GRPO step at K=8 x 8 prompts produces
# at most 64 distinct programs, so this holds several steps' worth and still
# stays small on disk (a debug rustc binary is a few MB).
_BUILD_CACHE_MAX = 256


class BuildError(Exception):
    """Raised when a program fails to compile. Callers score this as 0.0."""


@dataclass(frozen=True)
class LanguageSpec:
    """How to turn source text into something runnable, and how to run it."""

    name: str
    source_filename: str
    #: Fenced-block tags that identify this language in a model completion.
    fence_tags: tuple[str, ...]
    #: False for interpreted languages, where "build" is just writing the file.
    needs_build: bool


PYTHON = LanguageSpec(
    name="python",
    source_filename="solution.py",
    fence_tags=("python", "python3", "py"),
    needs_build=False,
)

RUST = LanguageSpec(
    name="rust",
    source_filename="solution.rs",
    fence_tags=("rust", "rs"),
    needs_build=True,
)

_LANGUAGES = {spec.name: spec for spec in (PYTHON, RUST)}


def get_language(name: str) -> LanguageSpec:
    try:
        return _LANGUAGES[name]
    except KeyError:
        raise ValueError(
            f"unsupported verifier language {name!r}; known: {sorted(_LANGUAGES)}"
        ) from None


def rust_toolchain_available() -> bool:
    return shutil.which("rustc") is not None


# --- build cache ---------------------------------------------------------------
#
# Keyed by (language, sha256(source)). The value is a directory holding the
# artifact, or a BuildError for a program that failed to compile -- negative
# results are cached too, because a GRPO group full of the same syntax error
# should pay for rustc once, not eight times.

_cache_lock = threading.Lock()
_cache: "OrderedDict[tuple[str, str], Path | BuildError]" = OrderedDict()
_key_locks: dict[tuple[str, str], threading.Lock] = {}
_cache_root: Path | None = None


def _root() -> Path:
    global _cache_root
    if _cache_root is None or not _cache_root.is_dir():
        _cache_root = Path(tempfile.mkdtemp(prefix="rl-rust-build-"))
    return _cache_root


def clear_build_cache() -> None:
    """Drop every cached artifact. Call between runs; safe to call anytime."""
    global _cache_root
    with _cache_lock:
        _cache.clear()
        _key_locks.clear()
        root, _cache_root = _cache_root, None
    if root is not None and root.is_dir():
        shutil.rmtree(root, ignore_errors=True)


def _evict_if_needed() -> None:
    while len(_cache) > _BUILD_CACHE_MAX:
        _, victim = _cache.popitem(last=False)
        if isinstance(victim, Path):
            shutil.rmtree(victim, ignore_errors=True)


def _build_rust(source: str, workdir: Path) -> Path:
    """Compile ``source`` to a native binary. Raises BuildError on failure.

    Flags match multi-LCB's ``eval_script_rust`` (``-C debuginfo=2``, no
    optimization) so a program graded here and a program graded there get the
    same verdict. Switching on ``-O`` would make correct-but-slow solutions pass
    more often and silently change the reward, so it is not a free win.
    """
    src = workdir / RUST.source_filename
    src.write_text(source)
    binary = workdir / "solution"
    try:
        result = subprocess.run(
            ["rustc", str(src), "-C", "debuginfo=2", "-o", str(binary)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=BUILD_TIMEOUT,
            cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        raise BuildError("rustc timed out") from None
    except FileNotFoundError:
        raise RuntimeError(
            "rustc not found on PATH; the Rust reward cannot grade without a toolchain"
        ) from None
    if result.returncode != 0 or not binary.is_file():
        raise BuildError(result.stderr[-2000:] or "rustc failed")
    return binary


def build(source: str, spec: LanguageSpec) -> Path:
    """Return a path to the runnable artifact for ``source``, building if needed.

    Concurrent callers with the same source block on one lock so the program is
    compiled exactly once. Raises BuildError for a program that does not compile.
    """
    key = (spec.name, hashlib.sha256(source.encode("utf-8", "replace")).hexdigest())

    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
        else:
            key_lock = _key_locks.setdefault(key, threading.Lock())
    if hit is not None:
        if isinstance(hit, BuildError):
            raise hit
        return hit

    with key_lock:
        # Another thread may have finished while we waited for the lock.
        with _cache_lock:
            hit = _cache.get(key)
        if hit is not None:
            if isinstance(hit, BuildError):
                raise hit
            return hit

        workdir = Path(tempfile.mkdtemp(prefix="build-", dir=_root()))
        try:
            if spec.needs_build:
                artifact = _build_rust(source, workdir)
            else:
                artifact = workdir / spec.source_filename
                artifact.write_text(source)
        except BuildError as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            with _cache_lock:
                _cache[key] = exc
                _evict_if_needed()
            raise

        with _cache_lock:
            _cache[key] = artifact
            _evict_if_needed()
        return artifact


# --- execution -----------------------------------------------------------------


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


def _python_runner(*, script: Path, timeout: float) -> str:
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


def _native_runner(*, binary: Path, timeout: float) -> str:
    """Apply the same rlimits to a native binary, then exec it in place.

    ``os.execv`` replaces the launcher, so the limits carry over and no extra
    process sits between the harness and the program under test.
    """
    cpu_seconds = max(1, math.ceil(timeout) + 1)
    return f"""import os
import resource

def cap(which, value):
    _, hard = resource.getrlimit(which)
    bounded = value if hard == resource.RLIM_INFINITY else min(value, hard)
    resource.setrlimit(which, (bounded, hard))
    resource.setrlimit(which, (bounded, bounded))

cap(resource.RLIMIT_CPU, {cpu_seconds})
cap(resource.RLIMIT_FSIZE, {_VERIFIER_FILE_BYTES})
cap(resource.RLIMIT_NOFILE, {_VERIFIER_OPEN_FILES})
os.execv({str(binary)!r}, [{str(binary)!r}])
"""


def run_artifact(
    artifact: Path,
    spec: LanguageSpec,
    stdin: str | None,
    timeout: float,
) -> tuple[int, str]:
    """Execute a built artifact on one test case in a fresh working directory.

    Returns ``(returncode, stdout)``; a timeout is reported as ``(-1, "")``,
    matching the Python pipeline so a hung program grades as a failed case
    rather than crashing the batch.
    """
    with tempfile.TemporaryDirectory(prefix="rl-rust-run-") as raw_workdir:
        workdir = Path(raw_workdir).resolve()
        runner = workdir / "runner.py"
        stdout_path = workdir / "stdout.txt"
        if spec.needs_build:
            runner.write_text(_native_runner(binary=artifact, timeout=timeout))
        else:
            runner.write_text(_python_runner(script=artifact, timeout=timeout))
        command = [str(Path(sys.executable).resolve()), "-I", "-S", str(runner)]
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


def run_source(
    source: str,
    spec: LanguageSpec,
    stdin: str | None,
    timeout: float,
) -> tuple[int, str]:
    """Build (cached) then run. A program that will not compile returns (-1, "")."""
    try:
        artifact = build(source, spec)
    except BuildError:
        return -1, ""
    return run_artifact(artifact, spec, stdin, timeout)
