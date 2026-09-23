# AI Preference Drift

Studying task & training preferences in open-source LMs, whether they drift under training, and whether models anticipate their own drift.

## Layout

| Path                          | What it is                                                                                                                                                                                                                                                                                        | Ours?        |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `compute_utilities/`          | Thurstonian preference **scorer**, vendored from the CAIS [emergent-values](https://github.com/centerforaisafety/emergent-values) fork (Apache-2.0, see `LICENSE-emergent-values`). Stripped to the agents we'll plausibly use (vLLM/HuggingFace + LiteLLM); wired for our vLLM OpenAI-compatible checkpoint endpoints. | vendored     |
| `config.yaml`                 | Model registry and experiment registry the scorer reads (must sit next to `compute_utilities/`). Current entries target the Qwen3.5 base/M0/SFT endpoints used by the study, plus disabled dataset/training helpers.                                                                                                      | ours         |
| `api_keys/`                   | Optional provider or endpoint keys for hosted-model checks. Gitignored.                                                                                                                                                                                              | —            |
| `data/`                       | The preference **option sets** we're building + the UE reference subset. See `data/README.md`.                                                                                                                                                                                                    | **ours**     |
| `main.py`                     | Single entry point for every runnable script under `scripts/`, dispatched per `config.yaml`'s `experiments:` section. See Run below.                                                                                                                                                              | ours         |
| `scripts/run_utilities.py`    | Thin CLI to score an options file with the scorer. No bare `run_utilities` entry — run via `main.py run_utilities_book_a`, `run_utilities_values`, `run_utilities_ue`, or `run_utilities_other`, each of which pins its own options set and prompt style.                                                                                                                                                                                                               | ours         |
| `scripts/run_elicitations.py` | CLI for coding task-preference / training-preference / anticipation elicitation. Run via `main.py run_elicitations_task`, `run_elicitations_training`, or `run_elicitations_anticipation`.                                                                                                        | ours         |
| `scripts/train_sft_lora.py`   | TRL/PEFT supervised fine-tuning runner for prepared intervention datasets. Uses `SFTTrainer`, `LoraConfig`, assistant-only masking, and an optional Hugging Face Hub push callback.                                                                                                               | ours         |
| `multi-lcb/`                  | Second coding-capability instrument, vendored **unmodified** from [Multi-LCB](https://github.com/Multi-LCB/Multi-LCB) at `d80be9f` (MIT, see `LICENSE-multi-lcb`). Competitive-programming problems graded by stdin/stdout execution across 12 languages; we score rust, go, csharp, scala, php and python by default. Do not patch it in place — the published numbers are a property of this exact code. See `docs/multilcb-eval.md`. | vendored     |
| `scripts/score_multilcb.py`   | Adapter over `multi-lcb/`: resolves a `config.yaml` model key to its vLLM endpoint, invokes the upstream runner, and normalises the output into the `results/<model_key>/multilcb/` summary schema. Generation runs in the repo `venv/` (`requirements-multilcb-gen.txt`); only evaluation needs the toolchain conda env (`requirements-multilcb.txt`). Runbook in `docs/multilcb-eval.md`. | ours         |
| `emergent-values/`            | Full upstream fork, kept locally for reference. Gitignored.                                                                                                                                                                                                                                       | reference    |

## What was stripped from the vendored scorer

Only the parts the study uses were kept. Removed: the six direct-SDK API agent classes (OpenAI/Anthropic/Gemini/Grok/Fireworks) and their heavy imports, the other paper experiments, the SLURM orchestration, and the figures notebook. The fork's **logprobs forced-choice** elicitation (cheaper, ideal for local checkpoints + small option pools) is kept.

## Inference / serving — self-hosted vLLM endpoint

We're fine-tuning on AWS, so scoring runs against our **own** checkpoints — which no hosted API serves. Each teammate stands up a **vLLM OpenAI-compatible endpoint** for the checkpoint and points the scorer at it. The scorer side is wired: a `model_type: vllm_endpoint` entry in `config.yaml` (with a `base_url`) is served by `VLLMEndpointAgent`, a thin subclass of `LiteLLMAgent` routing through LiteLLM's native `hosted_vllm` provider. Scoring uses UE's **sampling** protocol (pairwise forced choice, K=10, temperature 1.0, order-counterbalanced — config `thurstonian_active_learning`), so the endpoint only needs plain chat completions, not token logprobs.

## Run

Dependencies and the virtualenv are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

`main.py` is the single entry point for every runnable script under `scripts/`. It dispatches them per the `experiments:` section of `config.yaml`, which controls which scripts run, whether each is `enabled`, and their default CLI flags (and, for `run_elicitations.py`'s subcommands, which one via `positional:`).

Current live reruns omit Book A (`run_utilities_book_a`) unless a Book A ablation
is explicitly intended. Values are still in scope: value preferences are scored
as pairwise A/B utilities with `run_utilities_values`, while values anticipation
is a separate ternary `MORE` / `LESS` / `SAME` forecast via
`run_values_anticipation`. Use named entries for this current scope; the broad
default registry run is historical/baseline-oriented and includes Book A.

```bash
uv run main.py                # run the default baseline registry, including Book A
uv run main.py --list         # show the registry and what runs by default

uv run scripts/run_utilities.py \
    --model_key <aws-endpoint-entry-in-config.yaml> \
    --options_path data/options/coding.json \
    --config_key thurstonian_active_learning_small_logprobs \
    --save_dir results

uv run main.py run_elicitations_task -- --model_key <model> --save_dir results/task_preference

uv run main.py run_utilities_values                              # run one entry by name
uv run main.py run_utilities_values -- --model_key qwen35-08b-local # append raw flags

# Current live scope without Book A:
uv run main.py \
    --model_key <model> \
    run_elicitations_task \
    run_elicitations_training \
    run_utilities_values \
    run_utilities_ue \
    run_elicitations_anticipation \
    run_elicitations_ue_anticipation \
    run_values_anticipation
```

In that command, `run_utilities_values` is the value-preference A/B utility run;
`run_values_anticipation` is the values forecast run with `MORE` / `LESS` /
`SAME` labels.

SFT runs are intentionally separate from elicitation/scoring. First prepare a
seeded dataset with `scripts/prepare_sft_dataset.py`, then train:

```bash
uv run python scripts/train_sft_lora.py \
    --spec_path data/training_specs/coding_axis_sft.json \
    --intervention_id coding.write.python \
    --dataset_dir results/sft_datasets/coding.write.python_n1000_seed42 \
    --output_dir results/sft_runs/qwen35-4b-m0-v4/coding.write.python_n1000_seed42
```

Via `main.py` both steps are one command, and `--arm` / `--size` keep the
intervention id, dataset dir, output dir, and hub repo ids consistent with each
other: `python main.py --sft --arm java --size 9b`.

The SFT spec defaults to bf16 for the AWS/L4 path; pass `--no-bf16` only when
testing on hardware that cannot run bfloat16.

Dry-run validates the base-model config before training. The SFT example uses
`Qwen/Qwen3.5-0.8B`, the methodology's debug model: baseline, training, and
post-training scoring must all use the same Qwen3.5 base-model family, or the
before/after comparison measures nothing. Use `uv sync` for the pinned SFT
environment, or `pip install -r requirements-sft.txt` if `uv` is unavailable.
The checked-in stack is verified by `scripts/preflight_qwen35.py` before GPU
training.

There are three SFT-adjacent paths, and they should not be mixed:

- `scripts/train_sft_lora.py` is the Phase 1 coding-preference SFT runner over the prepared Magicoder language-arm datasets.
- `rl_training/build_sft_distill.py` and `rl_training/build_sft_dataset_azure.py` generate verifier-filtered teacher distillation datasets; they write data, not model checkpoints.
- `rl_training/train_sft.py` is the RL-comparison SFT policy arm, matched to PPO/DPO/GRPO cadence for the formal policy comparison.

## Datasets

The coding dataset build is driven by `config.yaml` (`datasets.coding`) at the project root.

Run the build script to regenerate:

```bash
uv run scripts/build_coding_dataset.py
```
