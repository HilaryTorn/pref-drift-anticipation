#!/usr/bin/env python3
"""Serve a formal checkpoint bank and supervise a checkpoint trajectory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_rl_checkpoint_elicitation import (  # noqa: E402
    build_registry,
    discover_adapter_paths,
)


def require_available_port(port: int) -> None:
    """Fail before launch instead of accidentally attaching to a stale server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.settimeout(1.0)
        if candidate.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} has a listener; refusing stale-server reuse")


def raise_shutdown(_signum, _frame) -> None:
    """Turn tmux/SIGTERM shutdown into normal unwinding so vLLM is reaped."""
    raise KeyboardInterrupt


def wait_for_server(base_url: str, api_key: str, process: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    last_error = "server has not responded"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"vLLM exited during startup with code {return_code}")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(10)
    raise RuntimeError(f"vLLM did not become ready within {timeout}s: {last_error}")


def terminate_process(process: subprocess.Popen, timeout: int = 60) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run_once(args: argparse.Namespace) -> None:
    checkpoint_root = args.checkpoint_root.resolve()
    base_model = args.base_model.resolve()
    if not (base_model / "config.json").is_file():
        raise RuntimeError(f"Invalid base model snapshot: {base_model}")
    adapters = discover_adapter_paths(checkpoint_root)
    require_available_port(args.port)
    base_url = f"http://localhost:{args.port}/v1"
    registry, modules = build_registry(args.size, base_url, adapters)

    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.runtime_dir / f"models-{args.size}.yaml"
    manifest_path = args.runtime_dir / f"loras-{args.size}.json"
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False))
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "rl_checkpoint_vllm_lora_manifest_v1",
                "model_size": args.size,
                "base_model": str(base_model),
                "baseline_alias": f"qwen35-{args.size}-m0-v4",
                "modules": modules,
            },
            indent=2,
        )
        + "\n"
    )

    api_key_path = ROOT / "api_keys" / "api_key_vllm_endpoint.txt"
    if not api_key_path.is_file() or not api_key_path.read_text().strip():
        raise RuntimeError(f"Missing vLLM API key: {api_key_path}")
    api_key = api_key_path.read_text().strip()
    server_command = [
        "vllm",
        "serve",
        str(base_model),
        "--served-model-name",
        f"qwen35-{args.size}-m0-v4",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--api-key",
        api_key,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--language-model-only",
        "--reasoning-parser",
        "qwen3",
        "--enable-lora",
        "--max-lora-rank",
        "16",
        "--max-loras",
        "1",
        "--max-cpu-loras",
        str(len(modules)),
        "--lora-modules",
        *(f"{module['name']}={module['path']}" for module in modules),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    server_log_path = args.log_dir / f"rl-checkpoint-vllm-{args.size}.log"
    trajectory_label = args.trajectory_kind.replace("-", "_")
    trajectory_log_path = args.log_dir / f"{trajectory_label}-trajectory-{args.size}.log"
    with server_log_path.open("a") as server_log:
        server_log.write(f"\n=== starting vLLM at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
        server_log.flush()
        process = subprocess.Popen(
            server_command,
            cwd=ROOT,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_server(base_url, api_key, process, args.startup_timeout)
            trajectory_script = (
                "run_dpo_grpo_rl_trajectory.py"
                if args.trajectory_kind == "rl-method"
                else "run_coding_task_preference_trajectory.py"
            )
            trajectory_command = [
                sys.executable,
                str(ROOT / "scripts" / trajectory_script),
                "--size",
                args.size,
                "--models-config-path",
                str(registry_path),
                "--state-dir",
                str(args.state_dir / args.size),
                "--command-timeout",
                str(args.command_timeout),
            ]
            with trajectory_log_path.open("a") as trajectory_log:
                trajectory_log.write(
                    f"\n=== starting trajectory at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n"
                )
                trajectory_log.flush()
                trajectory_process = subprocess.Popen(
                    trajectory_command,
                    cwd=ROOT,
                    env=environment,
                    stdout=trajectory_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    return_code = trajectory_process.wait()
                except BaseException:
                    terminate_process(trajectory_process)
                    raise
                if return_code == 75:
                    raise subprocess.TimeoutExpired(
                        trajectory_command, args.command_timeout
                    )
                if return_code:
                    raise subprocess.CalledProcessError(
                        return_code, trajectory_command
                    )
        finally:
            terminate_process(process)


def main() -> int:
    signal.signal(signal.SIGHUP, raise_shutdown)
    signal.signal(signal.SIGTERM, raise_shutdown)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["4b", "9b"], required=True)
    parser.add_argument(
        "--trajectory-kind",
        choices=["rl-method", "coding-task"],
        default="rl-method",
        help="Choose the dedicated RL-method battery or realized coding-task preference.",
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("/ndata/xianglin/ai_drift/elicitation_cache/runtime"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/ndata/xianglin/ai_drift/logs"),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=96,
        help="Match the RL scorer's per-endpoint concurrency ceiling.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=1800,
        help=(
            "Maximum seconds without trajectory completion before the current "
            "client/server process groups are recycled. Each individual command "
            "normally completes in minutes; 1800 implements the 30-minute watchdog."
        ),
    )
    parser.add_argument(
        "--restart-delay",
        type=int,
        default=1800,
        help="Seconds between supervised retries; default is the requested 30 minutes.",
    )
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument(
        "--wait-for-state-dir",
        type=Path,
        default=None,
        help="Do not allocate a GPU until this trajectory state directory is complete.",
    )
    parser.add_argument("--expected-wait-markers", type=int, default=388)
    args = parser.parse_args()
    if args.state_dir is None:
        state_name = (
            "rl_checkpoint_trajectory_state"
            if args.trajectory_kind == "rl-method"
            else "coding_task_preference_trajectory_state"
        )
        args.state_dir = ROOT / "results" / state_name

    if args.wait_for_state_dir is not None:
        while True:
            completed = len(list(args.wait_for_state_dir.glob("*.done.json")))
            if completed >= args.expected_wait_markers:
                print(
                    f"Dependency complete: {completed}/{args.expected_wait_markers} markers",
                    flush=True,
                )
                break
            print(
                f"Waiting for dependency: {completed}/{args.expected_wait_markers}; "
                f"checking again in {args.restart_delay} seconds",
                flush=True,
            )
            time.sleep(args.restart_delay)

    while True:
        try:
            run_once(args)
            print(
                f"{args.size} {args.trajectory_kind} checkpoint trajectory complete",
                flush=True,
            )
            return 0
        except KeyboardInterrupt:
            raise
        except subprocess.TimeoutExpired as exc:
            print(
                f"{args.size} trajectory watchdog fired after "
                f"{args.command_timeout}s: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if not args.supervise:
                return 1
            print(
                "Recycling vLLM and retrying immediately from command markers",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            print(f"{args.size} trajectory attempt failed: {exc}", file=sys.stderr, flush=True)
            if not args.supervise:
                return 1
            print(
                f"Retrying from command markers in {args.restart_delay} seconds",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(args.restart_delay)


if __name__ == "__main__":
    raise SystemExit(main())
