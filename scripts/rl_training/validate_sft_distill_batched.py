#!/usr/bin/env python3
"""Validate a formal SFT dataset in fresh, bounded sandboxed processes.

Long verifier sweeps can create transient timeout failures when thousands of
short-lived child processes share one long-running validator. This runner keeps
the formal acceptance criterion unchanged while limiting resource accumulation:
it validates fixed-size batches in fresh processes and rechecks any apparent
verifier failure in isolated fresh processes. A row is accepted after a failed
batch only when every isolated confirmation passes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from scripts.rl_training.rewards import verifier_sandbox_available


FAILURE_LINE_RE = re.compile(r":(?P<line>[1-9][0-9]*): verifier pass rate is ")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonblank_rows(path: Path) -> int:
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validator_command(
    *,
    train: Path,
    source: Path,
    verifier_timeout: float,
    start_row: int,
    end_row: int,
    skip_verifier: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "rl_training.validate_sft_distill",
        "--train",
        str(train),
        "--train_source",
        str(source),
        "--split",
        "train",
        "--verifier_timeout",
        str(verifier_timeout),
        "--progress_every",
        "0",
        "--start_row",
        str(start_row),
        "--end_row",
        str(end_row),
    ]
    if skip_verifier:
        command.append("--skip_verifier")
    return command


def run_validator(
    command: list[str],
    *,
    verifier_workers: int,
    process_timeout: float,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["VERIFIER_MAX_WORKERS"] = str(verifier_workers)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            timeout=process_timeout,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": "validator process timeout",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train_source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--confirmation_runs", type=int, default=3)
    parser.add_argument("--verifier_workers", type=int, default=8)
    parser.add_argument("--verifier_timeout", type=float, default=10.0)
    parser.add_argument("--process_timeout", type=float, default=3600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = args.train.expanduser().resolve()
    source = args.train_source.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not train.is_file() or not source.is_file():
        raise SystemExit("--train and --train_source must exist")
    if args.batch_size < 1 or args.confirmation_runs < 1:
        raise SystemExit("--batch_size and --confirmation_runs must be >= 1")
    if not 1 <= args.verifier_workers <= 8:
        raise SystemExit("--verifier_workers must be between 1 and 8")
    if args.verifier_timeout <= 0 or args.process_timeout <= 0:
        raise SystemExit("timeouts must be > 0")
    if not verifier_sandbox_available():
        raise SystemExit("Batched formal validation requires the macOS sandbox-exec sandbox")

    rows = nonblank_rows(train)
    source_rows = nonblank_rows(source)
    report: dict[str, Any] = {
        "schema": "batched_sft_validation_v1",
        "status": "running",
        "started_at": utc_now(),
        "train": {"path": str(train), "rows": rows, "sha256": sha256_file(train)},
        "source": {"path": str(source), "rows": source_rows, "sha256": sha256_file(source)},
        "policy": {
            "sandbox_required": True,
            "batch_size": args.batch_size,
            "confirmation_runs": args.confirmation_runs,
            "verifier_workers": args.verifier_workers,
            "verifier_timeout": args.verifier_timeout,
            "process_timeout": args.process_timeout,
            "transient_failure_rule": (
                "A failed batch row is accepted only after every isolated fresh-process "
                "confirmation returns a complete verifier pass."
            ),
        },
        "structural_preflight": None,
        "batches": [],
        "isolated_confirmations": [],
        "transient_rows": [],
        "failed_rows": [],
    }
    atomic_write_json(report_path, report)

    if rows != source_rows or rows < 1:
        report["status"] = "failed"
        report["failure"] = f"row-count mismatch: train={rows}, source={source_rows}"
        report["completed_at"] = utc_now()
        atomic_write_json(report_path, report)
        raise SystemExit(report["failure"])

    structural = run_validator(
        validator_command(
            train=train,
            source=source,
            verifier_timeout=args.verifier_timeout,
            start_row=1,
            end_row=rows,
            skip_verifier=True,
        ),
        verifier_workers=args.verifier_workers,
        process_timeout=args.process_timeout,
    )
    report["structural_preflight"] = structural
    atomic_write_json(report_path, report)
    if structural["returncode"] != 0:
        report["status"] = "failed"
        report["failure"] = "structural preflight failed"
        report["completed_at"] = utc_now()
        atomic_write_json(report_path, report)
        raise SystemExit(report["failure"])

    for start in range(1, rows + 1, args.batch_size):
        end = min(rows, start + args.batch_size - 1)
        print(f"[batched-validate] rows {start}-{end}/{rows}", flush=True)
        result = run_validator(
            validator_command(
                train=train,
                source=source,
                verifier_timeout=args.verifier_timeout,
                start_row=start,
                end_row=end,
            ),
            verifier_workers=args.verifier_workers,
            process_timeout=args.process_timeout,
        )
        failed_lines = sorted(
            {
                int(match.group("line"))
                for match in FAILURE_LINE_RE.finditer(result["stderr"])
            }
        )
        batch = {
            "start_row": start,
            "end_row": end,
            **result,
            "failed_verifier_rows": failed_lines,
        }
        report["batches"].append(batch)
        atomic_write_json(report_path, report)
        if result["returncode"] == 0:
            continue
        if not failed_lines:
            report["status"] = "failed"
            report["failure"] = f"batch {start}-{end} failed without parseable verifier rows"
            report["completed_at"] = utc_now()
            atomic_write_json(report_path, report)
            raise SystemExit(report["failure"])

        for line_no in failed_lines:
            confirmations = []
            for run_index in range(1, args.confirmation_runs + 1):
                confirmation = run_validator(
                    validator_command(
                        train=train,
                        source=source,
                        verifier_timeout=args.verifier_timeout,
                        start_row=line_no,
                        end_row=line_no,
                    ),
                    verifier_workers=args.verifier_workers,
                    process_timeout=args.process_timeout,
                )
                confirmation["run"] = run_index
                confirmations.append(confirmation)
            row_result = {"line": line_no, "runs": confirmations}
            report["isolated_confirmations"].append(row_result)
            if all(item["returncode"] == 0 for item in confirmations):
                report["transient_rows"].append(line_no)
            else:
                report["failed_rows"].append(line_no)
            atomic_write_json(report_path, report)

    report["transient_rows"] = sorted(set(report["transient_rows"]))
    report["failed_rows"] = sorted(set(report["failed_rows"]))
    report["completed_at"] = utc_now()
    if report["failed_rows"]:
        report["status"] = "failed"
        report["failure"] = "one or more rows failed isolated confirmation"
        atomic_write_json(report_path, report)
        raise SystemExit(f"Batched SFT validation failed: rows {report['failed_rows']}")
    report["status"] = "complete"
    atomic_write_json(report_path, report)
    print(
        f"Batched SFT validation complete: rows={rows}, "
        f"transient_rows={report['transient_rows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
