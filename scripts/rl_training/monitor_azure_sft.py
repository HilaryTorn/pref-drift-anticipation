#!/usr/bin/env python3
"""Print compact, non-secret progress for an Azure SFT dataset build."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if line_no == len(lines):
                break
            raise
        if isinstance(row, dict):
            rows.append(row)
    return rows


def token_usage(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    input_tokens = output_tokens = reasoning_tokens = 0
    for row in rows:
        usage = row.get("usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        details = usage.get("output_tokens_details") or {}
        reasoning_tokens += int(details.get("reasoning_tokens") or 0)
    return input_tokens, output_tokens, reasoning_tokens


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def format_duration(hours: float | None) -> str:
    if hours is None or hours < 0:
        return "unknown"
    seconds = int(round(hours * 3600))
    return str(timedelta(seconds=seconds))


def snapshot(
    output_dir: Path,
    *,
    input_usd_per_million: float = 5.0,
    output_usd_per_million: float = 30.0,
) -> tuple[str, bool]:
    accepted = read_jsonl(output_dir / "accepted.jsonl")
    attempts = read_jsonl(output_dir / "attempts.jsonl")
    failed = read_jsonl(output_dir / "failed_candidates.jsonl")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    requested = int((manifest.get("counts") or {}).get("requested_rows") or 0)
    status = str(manifest.get("status") or "starting")
    reasons = Counter(str(row.get("reason") or "unknown") for row in failed)
    api_errors = sum("api_error" in row for row in attempts)
    rate_limit_errors = sum(
        "api_error" in row
        and any(marker in str(row.get("api_error")).lower() for marker in ("429", "rate_limit"))
        for row in attempts
    )
    interpreter_requested = sum(bool(row.get("code_interpreter_requested")) for row in attempts)
    interpreter_calls = sum(int(row.get("code_interpreter_call_count") or 0) for row in attempts)
    interpreter_containers_created = sum(
        bool(row.get("code_interpreter_requested"))
        and not bool(row.get("code_interpreter_container_reused"))
        and int(row.get("code_interpreter_call_count") or 0) > 0
        for row in attempts
    )
    interpreter_containers_reused = sum(
        bool(row.get("code_interpreter_container_reused"))
        and int(row.get("code_interpreter_call_count") or 0) > 0
        for row in attempts
    )
    input_tokens, output_tokens, reasoning_tokens = token_usage(attempts)
    latest = accepted[-1].get("record_id") if accepted else None
    imported = int((manifest.get("imported_accepted") or {}).get("retained_rows") or 0)
    attempted_ids = {row.get("record_id") for row in attempts if row.get("record_id")}
    remaining_unattempted = max(0, requested - imported - len(attempted_ids))
    cost_by_id: dict[str, float] = {}
    timestamps: list[datetime] = []
    for row in attempts:
        timestamp = parse_timestamp(row.get("started_at"))
        if timestamp is not None:
            timestamps.append(timestamp)
        record_id = row.get("record_id")
        usage = row.get("usage") or {}
        if not isinstance(record_id, str) or not isinstance(usage, dict):
            continue
        row_cost = (
            int(usage.get("input_tokens") or 0) * input_usd_per_million
            + int(usage.get("output_tokens") or 0) * output_usd_per_million
        ) / 1_000_000
        cost_by_id[record_id] = cost_by_id.get(record_id, 0.0) + row_cost
    estimated_cost = sum(cost_by_id.values())
    elapsed_minutes = (
        (max(timestamps) - min(timestamps)).total_seconds() / 60
        if len(timestamps) > 1
        else 0.0
    )
    hourly_burn = estimated_cost / elapsed_minutes * 60 if elapsed_minutes else 0.0
    mean_cost = estimated_cost / len(cost_by_id) if cost_by_id else 0.0
    p90_cost = 0.0
    if cost_by_id:
        costs = sorted(cost_by_id.values())
        p90_cost = costs[int(0.9 * (len(costs) - 1))]
    mean_remaining = remaining_unattempted * mean_cost
    p90_remaining = remaining_unattempted * p90_cost
    max_workers = int((manifest.get("execution") or {}).get("max_workers") or 1)
    max_attempts = int((manifest.get("generation_parameters") or {}).get("max_attempts") or 3)

    latest_run_id = attempts[-1].get("run_id") if attempts else None
    phase_attempts = [row for row in attempts if row.get("run_id") == latest_run_id]
    phase_rows_by_id: dict[str, list[dict[str, Any]]] = {}
    phase_cost = 0.0
    phase_starts: list[datetime] = []
    phase_ends: list[datetime] = []
    for row in phase_attempts:
        record_id = row.get("record_id")
        if isinstance(record_id, str):
            phase_rows_by_id.setdefault(record_id, []).append(row)
        usage = row.get("usage") or {}
        phase_cost += (
            int(usage.get("input_tokens") or 0) * input_usd_per_million
            + int(usage.get("output_tokens") or 0) * output_usd_per_million
        ) / 1_000_000
        started = parse_timestamp(row.get("started_at"))
        if started is not None:
            phase_starts.append(started)
            phase_ends.append(started + timedelta(seconds=float(row.get("elapsed_seconds") or 0.0)))
    completed_phase_ids = {
        record_id
        for record_id, rows in phase_rows_by_id.items()
        if any(float(row.get("teacher_pass_rate") or 0.0) == 1.0 for row in rows)
        or max(int(row.get("attempt_in_session") or 0) for row in rows) >= max_attempts
    }
    phase_wall_minutes = (
        (max(phase_ends) - min(phase_starts)).total_seconds() / 60
        if phase_starts and phase_ends
        else 0.0
    )
    records_per_minute = (
        len(completed_phase_ids) / phase_wall_minutes
        if completed_phase_ids and phase_wall_minutes > 0
        else 0.0
    )
    phase_hourly_burn = phase_cost / phase_wall_minutes * 60 if phase_wall_minutes else 0.0
    eta_hours = (
        remaining_unattempted / records_per_minute / 60
        if records_per_minute > 0
        else None
    )
    text = (
        f"status={status} accepted={len(accepted)}/{requested or '?'} "
        f"attempts={len(attempts)} failed_candidates={len(failed)} "
        f"api_errors={api_errors} rate_limit_errors={rate_limit_errors}\n"
        f"tokens input={input_tokens} output={output_tokens} reasoning={reasoning_tokens}\n"
        f"code_interpreter requested={interpreter_requested} calls={interpreter_calls} "
        f"containers_created={interpreter_containers_created} "
        f"container_reuses={interpreter_containers_reused}\n"
        f"estimated_usd={estimated_cost:.2f} historical_burn_usd_per_hour={hourly_burn:.2f} "
        f"remaining_unattempted={remaining_unattempted}\n"
        f"workers={max_workers} latest_phase_usd={phase_cost:.2f} "
        f"latest_phase_burn_usd_per_hour={phase_hourly_burn:.2f} "
        f"latest_phase_completed={len(completed_phase_ids)} "
        f"latest_phase_records_per_min={records_per_minute:.3f} "
        f"eta={format_duration(eta_hours)}\n"
        f"projected_remaining_usd mean={mean_remaining:.2f} p90_record_rate={p90_remaining:.2f} "
        f"prices_per_million input={input_usd_per_million:.2f} output={output_usd_per_million:.2f}\n"
        f"failure_reasons={dict(sorted(reasons.items()))} latest_accepted={latest}"
    )
    complete = status == "complete"
    return text, complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--watch", type=float, default=0.0, metavar="SECONDS")
    parser.add_argument("--input-usd-per-million", type=float, default=5.0)
    parser.add_argument("--output-usd-per-million", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_usd_per_million < 0 or args.output_usd_per_million < 0:
        raise SystemExit("token prices must be non-negative")
    while True:
        text, complete = snapshot(
            args.output_dir.expanduser().resolve(),
            input_usd_per_million=args.input_usd_per_million,
            output_usd_per_million=args.output_usd_per_million,
        )
        print(time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        print(text, flush=True)
        if not args.watch or complete:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
