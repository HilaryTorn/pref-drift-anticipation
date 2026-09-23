#!/usr/bin/env python3
"""Isolated OpenAI Responses API worker for SFT distillation.

The distillation builder launches this module in a subprocess so request
timeouts can be enforced without Python multiprocessing. API keys are read only
from the environment; request payloads arrive on stdin; the result is one JSON
object on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_RESPONSES_URL = "https://api.openai.com/v1/responses"


def write_result(status: str, result: Any) -> None:
    sys.stdout.write(json.dumps({"status": status, "result": result}, ensure_ascii=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--api_key_env", required=True)
    parser.add_argument(
        "--responses_url",
        default=os.environ.get("OPENAI_RESPONSES_URL", DEFAULT_RESPONSES_URL),
        help="Responses API URL. Defaults to OpenAI; Azure callers pass <endpoint>/responses.",
    )
    parser.add_argument(
        "--auth_header",
        choices=["bearer", "api-key"],
        default="bearer",
        help="Credential header style: OpenAI uses bearer; Azure OpenAI API-key calls use api-key.",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        write_result("worker_error", f"missing environment variable: {args.api_key_env}")
        return 0

    payload = sys.stdin.buffer.read()
    headers = {"Content-Type": "application/json"}
    if args.auth_header == "api-key":
        headers["api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(args.responses_url, data=payload, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=args.timeout) as response:
            write_result("ok", json.loads(response.read().decode("utf-8")))
    except HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        write_result("http_error", {"code": exc.code, "body": body_text[:500]})
    except URLError as exc:
        write_result("url_error", str(exc.reason))
    except TimeoutError as exc:
        write_result("timeout_error", str(exc))
    except Exception as exc:  # pragma: no cover - defensive boundary around API worker.
        write_result("worker_error", f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
