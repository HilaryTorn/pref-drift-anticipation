#!/usr/bin/env python3
"""OpenRouter generation backend for the M0 warm-start trace build.

WHY THIS EXISTS. `build_warmstart.py` generates traces with transformers on a GPU box. For the 27B that means renting a card for the inference half of the build, which the Inference Rule (docs/cost.md) discourages and the budget does not want. OpenRouter serves `qwen/qwen3.5-27b` per token, so the trace set can be built from a laptop for a couple of dollars. Everything downstream of generation -- the integrity filters, the (cell x answer) quotas, the refill rounds, the manifest -- is untouched and shared with the local backend.

WHAT IS AND IS NOT PRESERVED. The recipe is identical: same prompt file, same brevity override, same k, same sampling parameters, same token band, same filters. What changes is the numerics: the served endpoint is a third party's copy of the weights, usually fp8-quantized, behind that provider's own chat template. So the traces are the 27B's own reasoning in the sense that matters for self-distillation (same weights, same family -- not a foreign teacher), but they are NOT byte-for-byte what `Qwen/Qwen3.5-27B` would emit locally in bf16. That is a real deviation from `m0/README.md`'s "the traces are the model's OWN reasoning" and belongs in the limitations of any write-up that compares 27B against the 4B and 9B builds, which were generated on-stack.

THE COMPLETION IS REASSEMBLED, NOT SAMPLED. This is the part to understand before trusting the output. A local generation is one string: `...reasoning...\\n</think>\\n\\nAnswer: 2` (no opening tag -- the chat template leaves the block open, see m0/rewards.py::split_completion). The chat-completions API instead hands back the reasoning in `message.reasoning` and the post-`</think>` text in `message.content`, so this module joins them back into that exact shape. Two consequences:

  1. The `coda` filter goes nearly vacuous. Locally it rejects a trace that wrote prose between `</think>` and the answer line -- the failure that ruined the first 9B set. Here the provider has already split that prose off into `content` alongside the answer, so a coda shows up as a multi-line `content` rather than as the thing the filter was written to catch. It still gets rejected (the answer must be the last non-empty line), but by a different mechanism than on the local path.
  2. Provider disagreement is real and measured. On 2026-09-03, one identical request: DeepInfra (fp8) returned 178 reasoning tokens and a bare `Answer: 1`; Alibaba spent 1046 reasoning tokens restating the instructions; **Novita (bf16) mis-split the two fields**, returning a `content` that began mid-sentence inside the reasoning. Novita's output would be corrupt traces. Hence `--or_provider` pins one provider with fallbacks off, by default, and records it in the manifest. Do not turn fallbacks on to chase throughput.

Nothing here silently repairs a bad response. A missing `reasoning`, an empty `content` or a provider that returns its own `</think>` produces a string that the existing filters reject and tally (`no_close`, `no_commit`, `meta_echo`), so a misbehaving provider shows up as a yield collapse in the run's own diagnostics rather than as quietly degraded training data.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

KEY_ENV = "OPENROUTER_API_KEY"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"

DEFAULT_MODEL = "qwen/qwen3.5-27b"
# Measured on 2026-09-03 against the M0 prompt file: the only one of the three sampled providers that both honoured the six-bullet brevity override and split reasoning from content correctly. Re-check with --or_probe before a full run rather than trusting this constant -- endpoints change under a model id without the id changing.
DEFAULT_PROVIDER = "DeepInfra"


def load_env_files() -> None:
    """Make the repo's gitignored .env visible, as scripts/build_lcb_go_teacher.py does."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def resolve_key() -> str:
    """Process environment first, then .env, then api_keys/api_key_openrouter.txt."""
    load_env_files()
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        key_path = ROOT / "api_keys" / "api_key_openrouter.txt"
        if key_path.exists():
            key = key_path.read_text().strip()
    if not key:
        raise SystemExit(
            f"Missing {KEY_ENV}.\n"
            "Put it in the repo's gitignored .env (same file as HF_TOKEN), as one line:\n"
            f"  {KEY_ENV}=sk-or-...\n"
            "or in api_keys/api_key_openrouter.txt. An exported environment variable wins over .env."
        )
    return key


def _get_json(url: str, key: str, timeout: int = 60) -> Any:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def resolve_settings(model: str, provider: str | None) -> dict[str, Any]:
    """Validate the model id and the pinned provider against the live catalogue BEFORE any spend.

    A mistyped id or a provider that has since dropped the model is the obvious way to lose an
    afternoon, so both fail here with the available alternatives listed rather than after N
    requests. The catalogue entry and the endpoint's quantization go into the manifest: "which
    copy of the weights produced these traces" is exactly the provenance a reader of the 27B
    results will want, and it is not recoverable afterwards.
    """
    key = resolve_key()
    try:
        catalogue = _get_json(MODELS_URL, key)["data"]
    except Exception as exc:  # network/auth problems must be legible, not a traceback
        raise SystemExit(f"Could not reach OpenRouter's model catalogue ({type(exc).__name__}: {exc}). Check the key and connectivity.")

    entry = next((item for item in catalogue if item.get("id") == model), None)
    if entry is None:
        needle = model.split("/")[-1].lower()[:12]
        close = [item["id"] for item in catalogue if needle in item.get("id", "").lower()][:12]
        raise SystemExit(
            f"Model {model!r} is not in OpenRouter's catalogue.\n"
            + ("Close matches:\n  " + "\n  ".join(close) if close else "No close matches; browse https://openrouter.ai/models")
            + "\nPass the exact id with --or_model."
        )

    endpoints = _get_json(ENDPOINTS_URL.format(model=model), key)["data"].get("endpoints", [])
    chosen = None
    if provider:
        chosen = next((e for e in endpoints if e.get("provider_name") == provider), None)
        if chosen is None:
            names = sorted({e.get("provider_name", "?") for e in endpoints})
            raise SystemExit(
                f"Provider {provider!r} does not serve {model!r}.\n"
                f"Available: {', '.join(names)}\n"
                "Pass one of those with --or_provider, or --or_provider '' to let OpenRouter route "
                "(NOT recommended: providers disagree on the reasoning/content split -- see this "
                "module's docstring)."
            )

    pricing = entry.get("pricing") or {}
    return {
        "key": key,
        "model": model,
        "name": entry.get("name") or model,
        "provider": provider or None,
        "quantization": (chosen or {}).get("quantization"),
        "context_length": entry.get("context_length"),
        # OpenRouter quotes USD per token, as strings.
        "price_in": float(pricing.get("prompt") or 0.0),
        "price_out": float(pricing.get("completion") or 0.0),
    }


class OpenRouterGenerator:
    """Drop-in replacement for build_warmstart.generate()'s model+tokenizer pair.

    Holds the settings, the thread pool and the running spend. `generate()` matches the local
    function's signature and return shape (one list of k strings per prompt, in prompt order) so
    the call site branches on the backend and nothing else changes.
    """

    def __init__(self, settings: dict[str, Any], concurrency: int = 8, max_retries: int = 5,
                 timeout: int = 240):
        self.settings = settings
        self.concurrency = max(1, concurrency)
        self.max_retries = max_retries
        self.timeout = timeout
        self._lock = threading.Lock()
        self.cost_usd = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.calls = 0
        self.errors = 0

    # -- one request -------------------------------------------------------------------------

    def _body(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float,
              top_k: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.settings["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            # The local path renders the chat template with enable_thinking=True. This is the
            # nearest API equivalent: ask the server for the reasoning block and to return it.
            "reasoning": {"enabled": True},
            "include_reasoning": True,
            "usage": {"include": True},
        }
        if self.settings.get("provider"):
            # allow_fallbacks off ON PURPOSE. Silently failing over to another provider mid-run
            # would mix two different copies of the weights into one trace set, and the manifest
            # would name only the one that was asked for.
            body["provider"] = {"only": [self.settings["provider"]], "allow_fallbacks": False}
        return body

    def _call_once(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float,
                   top_k: int) -> str:
        """Return one completion in LOCAL SHAPE, or "" when the response is unusable.

        "" is deliberate rather than an exception: an empty string fails the `</think>` check and
        lands in the run's `no_close` tally, so a provider having a bad day shows up as a yield
        collapse in the existing diagnostics instead of aborting a part-finished run.
        """
        payload = json.dumps(self._body(prompt, max_new_tokens, temperature, top_p, top_k)).encode()
        headers = {
            "Authorization": f"Bearer {self.settings['key']}",
            "Content-Type": "application/json",
        }
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                request = urllib.request.Request(CHAT_URL, data=payload, headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                # 429 (rate limit) and 5xx (provider hiccup) are worth waiting out; a 4xx is a bad
                # request and will fail identically every time, so it stops here.
                if exc.code != 429 and exc.code < 500:
                    detail = exc.read()[:300].decode("utf-8", "replace")
                    raise SystemExit(f"OpenRouter rejected the request (HTTP {exc.code}): {detail}")
                if attempt == self.max_retries - 1:
                    with self._lock:
                        self.errors += 1
                    return ""
                time.sleep(delay)
                delay *= 2
            except Exception:
                if attempt == self.max_retries - 1:
                    with self._lock:
                        self.errors += 1
                    return ""
                time.sleep(delay)
                delay *= 2
        else:  # pragma: no cover - loop always breaks or returns
            return ""

        usage = data.get("usage") or {}
        with self._lock:
            self.calls += 1
            self.cost_usd += float(usage.get("cost") or 0.0)
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            self.reasoning_tokens += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)

        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        reasoning = (message.get("reasoning") or "").strip()
        content = (message.get("content") or "").strip()
        if not reasoning or not content:
            # Truncated mid-reasoning (finish_reason "length"), a refusal, or a provider that did
            # not return the block. Nothing to salvage -- an answer with no reasoning is not a
            # trace, and reasoning with no answer never committed.
            return ""
        # LOCAL SHAPE. No opening <think>: the chat template leaves the block open, so a local
        # completion carries only the closing tag (m0/rewards.py::split_completion). Getting this
        # wrong would put an unmatched tag in every SFT target.
        return f"{reasoning}\n</think>\n\n{content}"

    # -- the batch API build_warmstart calls --------------------------------------------------

    def generate(self, prompt_texts: list[str], k: int, max_new_tokens: int, temperature: float,
                 top_p: float = 0.95, top_k: int = 20) -> list[list[str]]:
        """k samples for each prompt, preserving prompt order.

        The API has no `n`, so k samples is k requests. They are fanned out together with the
        batch's prompts because the whole batch is one quota step: the caller cannot proceed until
        every prompt in it has been sampled, so there is nothing to gain by serialising the k.
        """
        jobs = [(i, p) for i, p in enumerate(prompt_texts) for _ in range(k)]
        results: list[list[str]] = [[] for _ in prompt_texts]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [
                (i, pool.submit(self._call_once, p, max_new_tokens, temperature, top_p, top_k))
                for i, p in jobs
            ]
            for i, future in futures:
                results[i].append(future.result())
        return results

    def provenance(self) -> dict[str, Any]:
        """What went into the manifest: which copy of the weights, and what it cost."""
        return {
            "backend": "openrouter",
            "or_model": self.settings["model"],
            "or_model_name": self.settings["name"],
            "or_provider": self.settings["provider"],
            "or_quantization": self.settings["quantization"],
            "or_price_per_token": {"in": self.settings["price_in"], "out": self.settings["price_out"]},
            "or_calls": self.calls,
            "or_failed_calls": self.errors,
            "or_prompt_tokens": self.prompt_tokens,
            "or_completion_tokens": self.completion_tokens,
            "or_reasoning_tokens": self.reasoning_tokens,
            "or_cost_usd": round(self.cost_usd, 6),
        }


def probe(model: str, provider: str | None, prompt: str, max_new_tokens: int = 400) -> None:
    """One request per available provider, printed side by side. Costs a fraction of a cent.

    Run this before a full build. The point is not the answer -- it is whether a provider returns
    a short reasoning block and a bare answer line, or 1000 tokens of restated instructions, or a
    reasoning/content split that lands mid-sentence. All three were observed on 2026-09-03.
    """
    key = resolve_key()
    endpoints = _get_json(ENDPOINTS_URL.format(model=model), key)["data"].get("endpoints", [])
    names = [provider] if provider else sorted({e.get("provider_name") for e in endpoints if e.get("provider_name")})
    for name in names:
        settings = resolve_settings(model, name)
        generator = OpenRouterGenerator(settings, concurrency=1)
        text = generator._call_once(prompt, max_new_tokens, 1.0, 0.95, 20)
        quant = settings.get("quantization")
        print(f"\n=== {name} (quant={quant}) cost=${generator.cost_usd:.5f} "
              f"reasoning_tokens={generator.reasoning_tokens} ===")
        if not text:
            print("  UNUSABLE (no reasoning, no content, or the request failed)")
            continue
        reasoning, _, visible = text.partition("</think>")
        print(f"  reasoning[:200]: {reasoning[:200]!r}")
        print(f"  after </think>:  {visible.strip()[:200]!r}")
