all: toolchain

# Install Java, Rust, C++, and Python + uv toolchains.
toolchain:
	SUDO=""; if [ "$$(id -u)" != "0" ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi; \
	if command -v apt-get >/dev/null 2>&1; then \
		$$SUDO apt-get update && \
		$$SUDO apt-get install -y default-jdk build-essential python3 python3-venv python3-pip curl ca-certificates; \
	elif command -v dnf >/dev/null 2>&1; then \
		$$SUDO dnf install -y java-17-amazon-corretto-devel gcc gcc-c++ make python3 python3-pip curl ca-certificates; \
	elif command -v yum >/dev/null 2>&1; then \
		$$SUDO yum install -y java-11-amazon-corretto-devel gcc gcc-c++ make python3 python3-pip curl ca-certificates; \
	else \
		echo "No known package manager (apt-get/dnf/yum) found -- install a JDK, gcc, g++, make, and python3 manually." >&2; \
		exit 1; \
	fi
	command -v rustc >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "Toolchain installed. Run 'source \$$HOME/.cargo/env' (rustc) and 'source \$$HOME/.local/bin/env' (uv), or start a new shell, to pick them up on PATH."

# Default cache dir for model/adapter downloads -- `/home/sagemaker-user` is too small.
HF_CACHE_DIR ?= /mnt/sagemaker-nvme/sagemaker-user/.cache/huggingface

VLLM_USE_FLASHINFER_SAMPLER ?= 0
export VLLM_USE_FLASHINFER_SAMPLER

# data/reference/coding_capability_problems.json is sampled from the real coding-SFT
# training data (data/training/coding.write.<lang>/{pool,validation}.jsonl)
# pool_n/validation_n are pinned to 16/1 because that's C++'s ceiling
coding-capability-data:
	python scripts/scan_coding_capability_pool_candidates.py
	python scripts/build_coding_capability_pool_eval.py --pool_n 16 --validation_n 1
	python scripts/build_coding_capability_class_eval.py --pool_n 16 --validation_n 1

coding-checks-base:
	@python scripts/score_coding_capability.py --hf_base_repo prism-drift/qwen35-4b-m0-v4 $(if $(HF_CACHE_DIR),--hf_cache_dir "$(HF_CACHE_DIR)")
	@python scripts/score_coding_capability.py --hf_base_repo prism-drift/qwen35-9b-m0-v4 $(if $(HF_CACHE_DIR),--hf_cache_dir "$(HF_CACHE_DIR)")

# --hf_adapter_repo holds one LoRA arm per language and --languages must resolve to
# exactly one arm per invocation, so this reloads vLLM once per language.
coding-checks-sft:
	@for lang in python java rust cpp; do \
		python scripts/score_coding_capability.py --hf_base_repo prism-drift/qwen35-4b-m0-v4 --hf_adapter_repo prism-drift/qwen35-4b-m0-v4-phase-1-sft-adapters --languages $$lang $(if $(HF_CACHE_DIR),--hf_cache_dir "$(HF_CACHE_DIR)"); \
		python scripts/score_coding_capability.py --hf_base_repo prism-drift/qwen35-9b-m0-v4 --hf_adapter_repo prism-drift/qwen35-9b-m0-v4-phase-1-sft-adapters --languages $$lang $(if $(HF_CACHE_DIR),--hf_cache_dir "$(HF_CACHE_DIR)"); \
	done

coding-checks: coding-checks-base coding-checks-sft

# Multi-LCB coding-capability instrument (docs/multilcb-eval.md).
MULTILCB_CONDA_ENV ?= multi_lcb_env
MULTILCB_LANGUAGES ?= python,rust,go,csharp,php
MULTILCB_N ?= 1
MULTILCB_4B_HF_REPO ?= prism-drift/qwen35-4b-m0-v4
MULTILCB_4B_SERVED_NAME ?= qwen35-4b-m0-v4
MULTILCB_4B_MODEL_KEY ?= qwen35-4b-m0-v4-aws
MULTILCB_9B_HF_REPO ?= prism-drift/qwen35-9b-m0-v4
MULTILCB_9B_SERVED_NAME ?= qwen35-9b-m0-v4
MULTILCB_9B_MODEL_KEY ?= qwen35-9b-m0-v4-aws
MULTILCB_SERVE_PORT ?= 8000
MULTILCB_MAX_MODEL_LEN ?= 32768
MULTILCB_API_KEY ?= dummy-key
# Concurrency is bounded by the endpoint's KV cache, NOT by CPU. Read the real number off the vLLM
# startup line "GPU KV cache size: N tokens" and set MULTILCB_KV_CACHE_TOKENS before a real run; the
# default is the small (~24 GB card) value measured on 2026-08-24, so it under- rather than
# over-commits. score_multilcb.py refuses a batch size that does not fit -- see docs/multilcb-eval.md.
MULTILCB_KV_CACHE_TOKENS ?= 287464
MULTILCB_BATCH_SIZE ?= 8
MULTILCB_PREFLIGHT_N ?= 5
# vLLM's own --tensor-parallel-size defaults to 1 GPU. Only raise this on a multi-GPU box (e.g.
# ml.g6.12xlarge's 4x L4) when you actually want the model sharded across all of them -- for a
# smoke/debug run on a 4B model, 1 GPU is enough and the rest sit idle either way, so leave it at 1
# unless you're doing a real timed run and want the throughput.
MULTILCB_TENSOR_PARALLEL_SIZE ?= 1
# Dedicated venv for the GENERATION side of score_multilcb.py (requirements-multilcb-gen.txt) --
# deliberately separate from the vllm-env used to *serve* the model (that one needs vllm==0.25.0
# itself; this one only needs an OpenAI client and never touches vllm). Point this at wherever that
# venv lives on your box.
MULTILCB_GEN_VENV ?= $(HOME)/mlcb-gen
MULTILCB_MAX_TOKENS ?= 16384
MULTILCB_DEBUG_SIZE ?= 60

toolchain-multilcb:
	@HARD_STACK_KB="$$(ulimit -Hs)"; \
	if [ "$$HARD_STACK_KB" != "unlimited" ] && [ "$$HARD_STACK_KB" -lt 8388608 ] 2>/dev/null; then \
		echo "This shell's hard stack-size ulimit ($$HARD_STACK_KB KB) is below the 8 GB" >&2; \
		echo "multi-lcb/lcb_runner/evaluation/testing_plang.py's limit_virtual_memory() requires," >&2; \
		echo "and a non-root process cannot raise its own hard rlimit (confirmed via 'prlimit" >&2; \
		echo "--stack=... : Operation not permitted' -- seen on both a SageMaker Code Editor" >&2; \
		echo "container AND a classic ml.g6.12xlarge Notebook Instance on 2026-08-27, same 10240 KB" >&2; \
		echo "value both times, so this is not container-specific -- assume it applies to any" >&2; \
		echo "SageMaker-managed box until proven otherwise. Evaluation will fail on every language" >&2; \
		echo "except rust with 'Exception occurred in preexec_fn' -- this is a platform limit, not" >&2; \
		echo "something --debug_size or more disk space fixes. Run evaluation on a plain, non-" >&2; \
		echo "SageMaker EC2 instance instead -- see docs/multilcb-eval.md's runbook and the" >&2; \
		echo "known-limitation notes in INSTRUCTIONS_CODE_EDITOR.md / INSTRUCTIONS_NOTEBOOK.md." >&2; \
		exit 1; \
	fi
	@if ! command -v conda >/dev/null 2>&1 && [ ! -x "$$HOME/miniconda3/bin/conda" ]; then \
		echo "conda not found -- installing Miniconda to $$HOME/miniconda3"; \
		curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
		bash /tmp/miniconda.sh -b -p "$$HOME/miniconda3" && \
		rm -f /tmp/miniconda.sh; \
	fi
# Every conda call below passes --override-channels so the Anaconda 'defaults' channels
# (repo.anaconda.com/pkgs/main and /pkgs/r) are never consulted. Two reasons, one of which will
# stop a fresh box dead:
#   1. Miniconda installers since ~2025 gate those channels behind a Terms of Service acceptance.
#      A non-interactive `conda create` against them fails with CondaToSNonInteractiveError and no
#      env is created -- seen on a fresh g6e box on 2026-09-08. Without --override-channels the
#      fix is a manual `conda tos accept` on every new box, forever.
#   2. conda's default channel priority is flexible, not strict, so with 'defaults' in the list a
#      compiler can be resolved from either channel run to run. Evaluation verdicts are a property
#      of the compilers, so that is a silent comparability risk, not just untidiness.
# If you ever drop --override-channels, re-export multilcb-env.lock.yml and diff it before
# trusting the scores.
	CONDA_BASE="$$(conda info --base 2>/dev/null || echo $$HOME/miniconda3)"; \
	export PATH="$$CONDA_BASE/bin:$$PATH"; \
	conda install -y -n base -c conda-forge --override-channels conda-libmamba-solver; \
	conda env list | grep -qE "^$(MULTILCB_CONDA_ENV)[[:space:]]" || conda create -y -n $(MULTILCB_CONDA_ENV) --solver=libmamba -c conda-forge --override-channels python=3.11; \
	conda install -y -n $(MULTILCB_CONDA_ENV) --solver=libmamba -c conda-forge --override-channels --file requirements-multilcb.txt; \
	conda run -n $(MULTILCB_CONDA_ENV) pip install pyyaml
	CONDA_BASE="$$(conda info --base 2>/dev/null || echo $$HOME/miniconda3)"; \
	export PATH="$$CONDA_BASE/bin:$$PATH"; \
	@echo "Toolchain env '$(MULTILCB_CONDA_ENV)' ready (python, rust, go, scala, csharp, php)."

# NO --reasoning-parser HERE, for any size -- removed 2026-09-03, do not add it back. It is correct
# for the preference batteries in docs/serving-vllm-aws.md and WRONG for this instrument, which is
# why it looks like an omission. vLLM's qwen3 parser moves the think block out of
# `message.content` into `message.reasoning_content`; upstream reads only `message.content`
# (multi-lcb/lcb_runner/lm_styles/vllm_async_runner.py:129) and its extractor strips an INLINE
# `</think>` from that same string (utils/extraction_utils.py:66). Nothing in the vendored harness
# ever reads `reasoning_content`. On 2026-08-26 that returned 112 of 175 completions as empty
# strings and made pass@1 a pure artifact -- see docs/multilcb-9b-baseline-runbook.md, trap #1.
serve-multilcb-4b:
	VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve $(MULTILCB_4B_HF_REPO) \
		--served-model-name $(MULTILCB_4B_SERVED_NAME) \
		--dtype bfloat16 \
		--max-model-len $(MULTILCB_MAX_MODEL_LEN) \
		--tensor-parallel-size $(MULTILCB_TENSOR_PARALLEL_SIZE) \
		--api-key $(MULTILCB_API_KEY) \
		--host 0.0.0.0 --port $(MULTILCB_SERVE_PORT)

serve-multilcb-9b:
	VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve $(MULTILCB_9B_HF_REPO) \
		--served-model-name $(MULTILCB_9B_SERVED_NAME) \
		--dtype bfloat16 \
		--max-model-len $(MULTILCB_MAX_MODEL_LEN) \
		--tensor-parallel-size $(MULTILCB_TENSOR_PARALLEL_SIZE) \
		--api-key $(MULTILCB_API_KEY) \
		--host 0.0.0.0 --port $(MULTILCB_SERVE_PORT)

run-multilcb-eval:
	@test -n "$(MODEL_KEY)" || { echo "MODEL_KEY is required, e.g. make run-multilcb-eval MODEL_KEY=qwen35-4b-m0-v4-aws"; exit 1; }
	CONDA_BASE="$$(conda info --base 2>/dev/null || echo $$HOME/miniconda3)"; \
	export PATH="$$CONDA_BASE/bin:$$PATH"; \
	conda run -n $(MULTILCB_CONDA_ENV) --no-capture-output python3 scripts/fetch_multilcb_dataset.py; \
	conda run -n $(MULTILCB_CONDA_ENV) --no-capture-output python3 scripts/score_multilcb.py --model_key $(MODEL_KEY) --languages $(MULTILCB_LANGUAGES) --n $(MULTILCB_N) --batch_size $(MULTILCB_BATCH_SIZE) --kv_cache_tokens $(MULTILCB_KV_CACHE_TOKENS) --num_process_evaluate $$(nproc) $(MULTILCB_EVAL_ARGS)

# Run this FIRST on any new box, against an already-serving endpoint. Reports how many tokens a real
# completion takes and whether the model stops on its own, which is what every time and cost estimate
# for this instrument depends on. Minutes, not hours.
run-multilcb-preflight:
	@test -n "$(MODEL_KEY)" || { echo "MODEL_KEY is required, e.g. make run-multilcb-preflight MODEL_KEY=qwen35-4b-m0-v4-aws"; exit 1; }
	CONDA_BASE="$$(conda info --base 2>/dev/null || echo $$HOME/miniconda3)"; \
	export PATH="$$CONDA_BASE/bin:$$PATH"; \
	conda run -n $(MULTILCB_CONDA_ENV) --no-capture-output python3 scripts/fetch_multilcb_dataset.py; \
	conda run -n $(MULTILCB_CONDA_ENV) --no-capture-output python3 scripts/score_multilcb.py --model_key $(MODEL_KEY) --languages python --preflight $(MULTILCB_PREFLIGHT_N) --batch_size $(MULTILCB_BATCH_SIZE) --kv_cache_tokens $(MULTILCB_KV_CACHE_TOKENS)

run-multilcb-eval-all:
	@for triple in "$(MULTILCB_4B_HF_REPO)|$(MULTILCB_4B_SERVED_NAME)|$(MULTILCB_4B_MODEL_KEY)" \
	              "$(MULTILCB_9B_HF_REPO)|$(MULTILCB_9B_SERVED_NAME)|$(MULTILCB_9B_MODEL_KEY)"; do \
		HF_REPO="$${triple%%|*}"; rest="$${triple#*|}"; SERVED_NAME="$${rest%%|*}"; EVAL_MODEL_KEY="$${rest#*|}"; \
		echo "=== Serving $$HF_REPO as $$SERVED_NAME on port $(MULTILCB_SERVE_PORT) ==="; \
		VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "$$HF_REPO" \
			--served-model-name "$$SERVED_NAME" \
			--dtype bfloat16 \
			--max-model-len $(MULTILCB_MAX_MODEL_LEN) \
			--tensor-parallel-size $(MULTILCB_TENSOR_PARALLEL_SIZE) \
			--api-key $(MULTILCB_API_KEY) \
			--host 0.0.0.0 --port $(MULTILCB_SERVE_PORT) \
			> "/tmp/vllm-$$SERVED_NAME.log" 2>&1 & \
		VLLM_PID=$$!; \
		echo "vLLM PID $$VLLM_PID, log at /tmp/vllm-$$SERVED_NAME.log"; \
		UP=0; \
		for i in $$(seq 1 180); do \
			if curl -sf -H "Authorization: Bearer $(MULTILCB_API_KEY)" http://localhost:$(MULTILCB_SERVE_PORT)/v1/models >/dev/null 2>&1; then UP=1; break; fi; \
			kill -0 $$VLLM_PID 2>/dev/null || break; \
			sleep 5; \
		done; \
		if [ "$$UP" != "1" ]; then \
			echo "vLLM for $$SERVED_NAME never came up -- see /tmp/vllm-$$SERVED_NAME.log"; \
			kill $$VLLM_PID 2>/dev/null; exit 1; \
		fi; \
		echo "=== Endpoint up, running eval for $$EVAL_MODEL_KEY ==="; \
		$(MAKE) --no-print-directory run-multilcb-eval MODEL_KEY="$$EVAL_MODEL_KEY"; \
		EVAL_STATUS=$$?; \
		echo "=== Stopping vLLM (PID $$VLLM_PID) ==="; \
		kill $$VLLM_PID 2>/dev/null; wait $$VLLM_PID 2>/dev/null; \
		[ "$$EVAL_STATUS" = "0" ] || exit $$EVAL_STATUS; \
	done

.PHONY: all toolchain coding-capability-data coding-checks coding-checks-base coding-checks-sft toolchain-multilcb serve-multilcb-4b serve-multilcb-9b run-multilcb-eval run-multilcb-preflight run-multilcb-debug-generate run-multilcb-debug-evaluate run-multilcb-eval-all
