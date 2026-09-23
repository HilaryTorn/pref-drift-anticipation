#!/usr/bin/env python3
"""Run RL-method preferences and RL anticipation via a vLLM endpoint.

This runner intentionally does not extend the established ``training-preference``
or ``anticipation`` CLI modes. Those measure training-data preference and the
existing coding anticipation construct. Here:

* ``rl-preferences`` measures stated preferences over SFT/GRPO/DPO/PPO methods;
* ``rl-anticipation`` forecasts task-preference drift under those methods;
* ``rl-preference-anticipation`` forecasts the model's later preference over
  the RL methods themselves, on the same scale as ``rl-preferences``.

Both modes reuse the repository's pairwise scorer, response parsers, result
directories, and vLLM endpoint agent. Native thinking is switched per request.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_utilities.naming import default_results_dir, run_timestamp, timestamped  # noqa: E402
from scripts.elicitation_experiment_specs import LEVELS  # noqa: E402
from scripts.run_elicitations import (  # noqa: E402
    default_pairwise_raw_dump_path,
    grouped_option_records,
    load_experiment_spec,
    load_spec,
    load_training_stimuli,
    render_anticipation_records,
    run_anticipation,
    slug,
    write_pairwise_manifest,
)


COMPARISONS = {
    "all": {
        "preferences_spec": "rl_preferences",
        "anticipation_spec": "rl_anticipation",
        "experiment_spec": "data/experiment_specs/rl_anticipation.json",
    },
    "dpo-grpo": {
        "preferences_spec": "rl_dpo_grpo_preferences",
        "preference_anticipation_spec": "rl_dpo_grpo_preference_anticipation",
        "anticipation_spec": "rl_dpo_grpo_anticipation",
        "experiment_spec": "data/experiment_specs/rl_dpo_grpo_anticipation.json",
    },
    "dpo-grpo-9b": {
        "preferences_spec": "rl_dpo_grpo_9b_preferences",
        "preference_anticipation_spec": "rl_dpo_grpo_preference_anticipation",
        "anticipation_spec": "rl_dpo_grpo_9b_anticipation",
        "experiment_spec": "data/experiment_specs/rl_dpo_grpo_9b_anticipation.json",
    },
    "rl-cartesian": {
        "preferences_spec": "rl_cartesian_preferences",
        "preference_anticipation_spec": "rl_cartesian_preference_anticipation",
        "anticipation_spec": None,
        "experiment_spec": None,
    },
    "rl-cartesian-9b": {
        "preferences_spec": "rl_cartesian_9b_preferences",
        "preference_anticipation_spec": "rl_cartesian_9b_preference_anticipation",
        "anticipation_spec": None,
        "experiment_spec": None,
    },
}
FORMAL_PPO_RECOVERY_KIND = (
    "live_final_policy_rollout_scored_immediately_by_formal_reward_model"
)
MIN_FORMAL_REWARD_MODEL_ACCURACY = 0.60
ACTUAL_TRAINING_DESCRIPTIONS = {
    "sft": "supervised fine-tuning on one verifier-perfect teacher target per frozen prompt",
    "dpo": "direct preference optimization on verifier-ranked chosen/rejected solution pairs",
    "grpo": "group-relative policy optimization with eight verifier-scored rollouts per frozen prompt",
    "ppo": "proximal policy optimization using the formal verifier-derived reward model",
}
AGENT_PROFILES_PATH = (
    ROOT / "rl_training" / "configs" / "anticipation_preferences.yaml"
)


def load_agent_profile(native_thinking: bool) -> tuple[str, dict]:
    profile_name = "thinking" if native_thinking else "non_thinking"
    profiles = yaml.safe_load(AGENT_PROFILES_PATH.read_text())
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        raise ValueError(f"Missing RL agent profile {profile_name!r} in {AGENT_PROFILES_PATH}")
    if profile.get("enable_thinking") is not native_thinking:
        raise ValueError(
            f"RL agent profile {profile_name!r} must set "
            f"enable_thinking={native_thinking}"
        )
    return profile_name, profile


def load_vllm_model_config(model_key: str, models_config_path: str | None = None) -> dict:
    config_path = Path(models_config_path) if models_config_path else ROOT / "config.yaml"
    models = yaml.safe_load(config_path.read_text())
    model_config = models.get(model_key)
    if model_config is None:
        raise ValueError(f"Model {model_key!r} not found in {config_path}")
    if model_config.get("model_type") != "vllm_endpoint":
        raise ValueError(
            f"{model_key!r} uses model_type={model_config.get('model_type')!r}; "
            "RL anticipation/preferences require a self-hosted vllm_endpoint model"
        )
    if not model_config.get("base_url") or not model_config.get("model_name"):
        raise ValueError(f"vllm_endpoint model {model_key!r} needs model_name and base_url")
    return model_config


def create_rl_vllm_agent(
    model_key: str,
    native_thinking: bool,
    models_config_path: str | None = None,
    max_tokens_override: int | None = None,
):
    """Compose the existing endpoint agent with an RL-local native-thinking switch."""
    model_config = load_vllm_model_config(model_key, models_config_path)
    profile_name, profile = load_agent_profile(native_thinking)
    api_key_path = ROOT / "api_keys" / "api_key_vllm_endpoint.txt"
    if not api_key_path.is_file():
        raise ValueError(
            f"No vLLM endpoint API key at {api_key_path}; use the key passed to vllm serve"
        )
    os.environ["HOSTED_VLLM_API_KEY"] = api_key_path.read_text().strip()

    # Lazy import keeps prompt rendering and unit tests usable in the lightweight
    # environment where the training/inference torch stack is not installed.
    from scripts.compute_utilities.llm_agent import VLLMEndpointAgent

    class RLNativeThinkingEndpointAgent(VLLMEndpointAgent):
        def __init__(self, *args, enable_thinking: bool, **kwargs):
            self.enable_thinking = enable_thinking
            super().__init__(*args, **kwargs)

        def _reasoning_extra_body(self) -> dict:
            body = super()._reasoning_extra_body() or {}
            chat_template_kwargs = dict(body.get("chat_template_kwargs") or {})
            chat_template_kwargs["enable_thinking"] = self.enable_thinking
            body["chat_template_kwargs"] = chat_template_kwargs
            return body

    agent_kwargs = dict(profile)
    enable_thinking = bool(agent_kwargs.pop("enable_thinking"))
    if max_tokens_override is not None:
        agent_kwargs["max_tokens"] = max_tokens_override
    agent = RLNativeThinkingEndpointAgent(
        served_model=model_config["model_name"],
        base_url=model_config["base_url"],
        accepts_system_message=model_config.get("accepts_system_message", True),
        enable_thinking=enable_thinking,
        **agent_kwargs,
    )
    return agent, profile_name


def rl_preferences_options(level: str, spec_name: str) -> dict[str, list[str]]:
    if level not in LEVELS:
        raise ValueError(f"Unknown level: {level}. Must be one of {LEVELS}")
    items = load_training_stimuli(spec_name)
    return {"Training methods": [item["stimuli"][level] for item in items]}


def rl_preferences_manifest(options: dict, level: str, spec_name: str) -> list[dict]:
    spec = load_spec(spec_name)
    items = load_training_stimuli(spec_name)
    actual_ids = [item["id"] for item in items]
    if actual_ids != spec["anchor_ids"]:
        raise ValueError(
            f"RL stimulus IDs/order {actual_ids} do not match anchor_ids "
            f"{spec['anchor_ids']}"
        )
    by_text = {item["stimuli"][level]: item for item in items}
    if len(by_text) != len(items):
        raise ValueError(f"RL preference stimuli must be unique at level {level}")

    records = []
    for scorer_id, option_record in enumerate(grouped_option_records(options)):
        description = option_record["description"]
        item = by_text.get(description)
        if item is None:
            raise ValueError(f"RL preferences option {scorer_id} has no source stimulus")
        records.append(
            {
                "scorer_option_id": scorer_id,
                "canonical_id": item["id"],
                "training_method": item["training_method"],
                "source_cohort_id": item["source_cohort_id"],
                "example_datapoints_source": item["example_datapoints_source"],
                "example_datapoints_provenance": item["example_datapoints_provenance"],
                "concreteness_level": level,
                "category": option_record["category"],
                "description": description,
            }
        )
    return records


def formal_rl_example_source_error(record: dict) -> str | None:
    """Return why a stimulus is not backed by an allowed formal RL artifact."""
    source = record.get("example_datapoints_source")
    if source == "real_training_data":
        return None
    if source != "live_policy_rollout":
        return f"unsupported example_datapoints_source={source!r}"
    if (
        record.get("canonical_id") != "coding.ppo"
        or record.get("training_method") != "ppo"
    ):
        return "live_policy_rollout is allowed only for coding.ppo"

    provenance = record.get("example_datapoints_provenance")
    if not isinstance(provenance, dict):
        return "live PPO rollout is missing provenance"
    if provenance.get("schema") != "rl_example_provenance_v1":
        return "live PPO rollout has an invalid provenance schema"
    if provenance.get("recovery_kind") != FORMAL_PPO_RECOVERY_KIND:
        return "live PPO rollout is not a formal final-policy rollout"

    accuracy = provenance.get("reward_model_held_out_pairwise_accuracy")
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        return "live PPO rollout is missing numeric reward-model accuracy"
    if float(accuracy) < MIN_FORMAL_REWARD_MODEL_ACCURACY:
        return (
            "live PPO rollout reward-model accuracy "
            f"{float(accuracy):.3f} is below {MIN_FORMAL_REWARD_MODEL_ACCURACY:.2f}"
        )

    artifacts = provenance.get("source_artifacts")
    if not isinstance(artifacts, list):
        return "live PPO rollout is missing source artifacts"
    rollout_stems: set[str] = set()
    manifest_stems: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path = Path(str(artifact.get("path") or ""))
        digest = artifact.get("sha256")
        if len(str(digest or "")) != 64:
            continue
        if path.name.startswith("rl_ppo_online_rollouts_") and path.name.endswith(
            ".jsonl"
        ):
            rollout_stems.add(path.name.removesuffix(".jsonl"))
        if path.name.startswith("rl_ppo_online_rollouts_") and path.name.endswith(
            ".manifest.json"
        ):
            manifest_stems.add(path.name.removesuffix(".manifest.json"))
    if not rollout_stems.intersection(manifest_stems):
        return "live PPO rollout lacks a matched online-rollout data/manifest pair"
    return None


def validate_formal_rl_example_sources(manifest: list[dict], run_label: str) -> None:
    invalid = []
    for record in manifest:
        error = formal_rl_example_source_error(record)
        if error:
            invalid.append(f"{record.get('canonical_id')} ({error})")
    if invalid:
        raise ValueError(
            f"Refusing formal {run_label} with invalid example provenance: "
            + "; ".join(invalid)
        )


def validate_model_size(spec: dict, model_key: str) -> None:
    expected = spec.get("model_size")
    if expected and f"-{expected}-" not in model_key.lower():
        raise ValueError(
            f"Comparison spec requires a {expected.upper()} model, but model_key={model_key!r}"
        )


def render_rl_anticipation_records(
    experiment_spec: dict,
    levels: list[str],
    target_ids: list[str],
    target_set: str,
    spec_name: str = "rl_anticipation",
) -> list[dict]:
    return render_anticipation_records(
        anchor_ids=experiment_spec["anchor_ids"],
        levels=levels,
        target_ids=target_ids,
        target_set=target_set,
        experiment_spec=experiment_spec,
        spec_name=spec_name,
        training_stimuli_spec_name=spec_name,
    )


def variant_name(args: argparse.Namespace) -> str:
    return "thinking" if args.thinking == "on" else "non_thinking"


async def run_rl_preferences(args: argparse.Namespace) -> dict:
    spec_name = COMPARISONS[args.comparison]["preferences_spec"]
    spec = load_spec(spec_name)
    validate_model_size(spec, args.model_key)
    options = rl_preferences_options(args.level, spec_name)
    manifest = rl_preferences_manifest(options, args.level, spec_name)
    variant = variant_name(args)
    save_dir = args.save_dir or default_results_dir(args.model_key, "pairs")
    default_suffix = f"{spec['mode']}_{args.level}_{variant}"
    save_suffix = timestamped(args.save_suffix or default_suffix, not args.no_timestamp)
    config_key = args.config_key or {
        "dpo-grpo": "thurstonian_binary_sampled",
        "dpo-grpo-9b": "thurstonian_binary_sampled",
        "rl-cartesian": "thurstonian_threeway_sampled",
        "rl-cartesian-9b": "thurstonian_threeway_sampled",
    }.get(args.comparison, "thurstonian_active_learning")
    if "logprobs" in config_key:
        raise ValueError("rl-preferences always uses sampled vLLM completions, not logprobs")
    validate_formal_rl_example_sources(manifest, "RL preferences")

    write_pairwise_manifest(save_dir, save_suffix, spec["mode"], manifest, args.level)
    native_thinking = args.thinking == "on"
    agent, profile_name = create_rl_vllm_agent(
        args.model_key, native_thinking, args.models_config_path
    )
    prompt_template = spec["prompt_template"]
    raw_dump_path = args.raw_dump_path or default_pairwise_raw_dump_path(
        save_dir, save_suffix
    )

    from scripts.compute_utilities.compute_utilities import compute_utilities

    return await compute_utilities(
        options_list=options,
        model_key=args.model_key,
        agent=agent,
        compute_utilities_config_path=str(
            ROOT / "compute_utilities" / "compute_utilities.yaml"
        ),
        compute_utilities_config_key=config_key,
        comparison_prompt_template=prompt_template,
        with_reasoning=False,
        terminal_only=native_thinking,
        use_logprobs=False,
        utility_model_seed=args.seed,
        raw_dump_path=raw_dump_path,
        raw_dump_metadata={
            "mode": spec["mode"],
            "model_key": args.model_key,
            "save_suffix": save_suffix,
            "concreteness_level": args.level,
            "reasoning": False,
            "prompt_reasoning": False,
            "parse_terminal_only": native_thinking,
            "native_thinking": native_thinking,
            "answer_format": "bare_label",
            "inference_mode": "sample",
            "run_variant": variant,
            "agent_profile": profile_name,
            "compute_utilities_config_key": config_key,
        },
        save_dir=save_dir,
        save_suffix=save_suffix,
    )


async def run_rl_preference_anticipation(args: argparse.Namespace) -> dict:
    forecast_spec_name = COMPARISONS[args.comparison].get(
        "preference_anticipation_spec"
    )
    if not forecast_spec_name:
        raise ValueError(
            "rl-preference-anticipation requires the matched DPO/GRPO or RL Cartesian design"
        )
    preferences_spec_name = COMPARISONS[args.comparison]["preferences_spec"]
    preferences_spec = load_spec(preferences_spec_name)
    validate_model_size(preferences_spec, args.model_key)
    forecast_spec = load_spec(forecast_spec_name)
    for field in ("training_stimuli_source", "anchor_ids", "model_size"):
        if (
            field in forecast_spec
            and forecast_spec.get(field) != preferences_spec.get(field)
        ):
            raise ValueError(
                f"Preference and anticipation specs disagree on {field}: "
                f"{preferences_spec.get(field)!r} != {forecast_spec.get(field)!r}"
            )
    options = rl_preferences_options(args.level, preferences_spec_name)
    manifest = rl_preferences_manifest(options, args.level, preferences_spec_name)
    stimuli = load_training_stimuli(preferences_spec_name)
    actual_training_methods = forecast_spec.get("actual_training_methods") or []
    if args.anticipated_training_method not in actual_training_methods:
        raise ValueError(
            f"{args.anticipated_training_method!r} is not an actual-training arm "
            f"in {forecast_spec_name}: {actual_training_methods}"
        )
    matching_anchors = [
        item
        for item in stimuli
        if item.get("training_method") == args.anticipated_training_method
    ]
    if args.anticipated_training_method == "sft" and not matching_anchors:
        anticipated_training_anchor_id = "coding.sft"
    elif len(matching_anchors) == 1:
        anticipated_training_anchor_id = matching_anchors[0]["id"]
    else:
        raise ValueError(
            "Expected one anticipation anchor for method "
            f"{args.anticipated_training_method!r}, got {len(matching_anchors)}"
        )
    prompt_template = forecast_spec["prompt_template"].replace(
        "{actual_training_method}", args.anticipated_training_method.upper()
    )
    prompt_template = prompt_template.replace(
        "{actual_training_description}",
        ACTUAL_TRAINING_DESCRIPTIONS[args.anticipated_training_method],
    )

    variant = variant_name(args)
    save_dir = args.save_dir or default_results_dir(args.model_key, "pairs")
    default_suffix = (
        f"{forecast_spec['mode']}_{args.anticipated_training_method}_"
        f"{args.level}_{variant}"
    )
    save_suffix = timestamped(args.save_suffix or default_suffix, not args.no_timestamp)
    config_key = args.config_key or (
        "thurstonian_threeway_sampled"
        if args.comparison in {"rl-cartesian", "rl-cartesian-9b"}
        else "thurstonian_binary_sampled"
    )
    if "logprobs" in config_key:
        raise ValueError(
            "rl-preference-anticipation uses sampled vLLM completions, not logprobs"
        )
    validate_formal_rl_example_sources(manifest, "RL preference anticipation")

    write_pairwise_manifest(
        save_dir, save_suffix, forecast_spec["mode"], manifest, args.level
    )
    native_thinking = args.thinking == "on"
    agent, profile_name = create_rl_vllm_agent(
        args.model_key,
        native_thinking,
        args.models_config_path,
        max_tokens_override=512,
    )
    raw_dump_path = args.raw_dump_path or default_pairwise_raw_dump_path(
        save_dir, save_suffix
    )

    from scripts.compute_utilities.compute_utilities import compute_utilities

    return await compute_utilities(
        options_list=options,
        model_key=args.model_key,
        agent=agent,
        compute_utilities_config_path=str(
            ROOT / "compute_utilities" / "compute_utilities.yaml"
        ),
        compute_utilities_config_key=config_key,
        comparison_prompt_template=prompt_template,
        with_reasoning=False,
        terminal_only=native_thinking,
        use_logprobs=False,
        utility_model_seed=args.seed,
        raw_dump_path=raw_dump_path,
        raw_dump_metadata={
            "mode": forecast_spec["mode"],
            "model_key": args.model_key,
            "save_suffix": save_suffix,
            "concreteness_level": args.level,
            "anticipated_training_method": args.anticipated_training_method,
            "anticipated_training_anchor_id": anticipated_training_anchor_id,
            "reasoning": False,
            "prompt_reasoning": False,
            "parse_terminal_only": native_thinking,
            "native_thinking": native_thinking,
            "answer_format": "bare_label",
            "inference_mode": "sample",
            "run_variant": variant,
            "agent_profile": profile_name,
            "compute_utilities_config_key": config_key,
        },
        save_dir=save_dir,
        save_suffix=save_suffix,
    )


def default_rl_anticipation_results_path(
    model_key: str,
    experiment_spec: dict,
    level: str,
    target_set: str,
    variant: str,
    stamp: str | None,
) -> Path:
    filename = (
        f"rl_anticipation_{slug(experiment_spec['id'])}_{slug(target_set)}_"
        f"{slug(level)}_{slug(model_key)}_{slug(variant)}"
    )
    if stamp:
        filename += f"_{stamp}"
    return Path(default_results_dir(slug(model_key), "anticipation")) / f"{filename}.json"


async def run_rl_anticipation(args: argparse.Namespace) -> None:
    spec_name = COMPARISONS[args.comparison]["anticipation_spec"]
    if spec_name is None:
        raise ValueError(
            "RL Cartesian comparisons define same-scale RL preference anticipation; "
            "use the rl-preference-anticipation subcommand"
        )
    spec = load_spec(spec_name)
    if args.experiment_spec is None:
        args.experiment_spec = COMPARISONS[args.comparison]["experiment_spec"]
    native_thinking = args.thinking == "on"
    variant = variant_name(args)
    if args.model_key and not args.results_path:
        experiment_spec = load_experiment_spec(args.experiment_spec)
        args.results_path = str(
            default_rl_anticipation_results_path(
                model_key=args.model_key,
                experiment_spec=experiment_spec,
                level=args.level,
                target_set=args.target_set,
                variant=variant,
                stamp=None if args.no_timestamp else run_timestamp(),
            )
        )

    def agent_factory(model_key: str, _prompt_reasoning: bool):
        agent, profile_name = create_rl_vllm_agent(
            model_key, native_thinking, args.models_config_path
        )
        return agent, {"agent_profile": profile_name}

    await run_anticipation(
        args,
        spec_name=spec_name,
        training_stimuli_spec_name=spec_name,
        agent_factory=agent_factory,
        record_metadata={
            "mode": spec["mode"],
            "prompt_reasoning": False,
            "parse_terminal_only": native_thinking,
            "native_thinking": native_thinking,
            "answer_format": "bare_label",
            "inference_mode": "sample",
            "run_variant": variant,
        },
        raw_dump_metadata={
            "prompt_reasoning": False,
            "parse_terminal_only": native_thinking,
            "native_thinking": native_thinking,
            "answer_format": "bare_label",
            "inference_mode": "sample",
            "run_variant": variant,
        },
        formal_source_error=(
            "Refusing formal RL anticipation with synthetic/missing examples"
        ),
        output_label="RL anticipation",
        prompt_reasoning=False,
        parse_terminal_only=native_thinking,
    )


def add_mode_arguments(parser: argparse.ArgumentParser, model_required: bool) -> None:
    parser.add_argument("--model_key", required=model_required)
    parser.add_argument(
        "--comparison",
        choices=sorted(COMPARISONS),
        default="all",
        help="Use the four-method, matched DPO-vs-GRPO, or RL-only Cartesian design.",
    )
    parser.add_argument(
        "--thinking",
        choices=["on", "off"],
        default="on",
        help="Toggle only Qwen native thinking; prompt/parser/sampling stay fixed.",
    )
    parser.add_argument(
        "--models-config-path",
        default=None,
        help="Optional model registry YAML; defaults to repository config.yaml.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preferences = sub.add_parser(
        "rl-preferences", help="score stated preferences over SFT/GRPO/DPO/PPO"
    )
    add_mode_arguments(preferences, model_required=True)
    preferences.add_argument("--level", required=True, choices=[*LEVELS, "all"])
    preferences.add_argument("--config_key", default=None)
    preferences.add_argument("--save_dir", default=None)
    preferences.add_argument("--save_suffix", default=None)
    preferences.add_argument("--seed", type=int, default=42)
    preferences.add_argument("--raw_dump_path", default=None)
    preferences.add_argument(
        "--no-timestamp", "--no_timestamp", dest="no_timestamp", action="store_true"
    )

    anticipation = sub.add_parser(
        "rl-anticipation", help="export or query RL-method drift forecasts"
    )
    add_mode_arguments(anticipation, model_required=False)
    anticipation.add_argument(
        "--experiment_spec",
        default=None,
    )
    anticipation.add_argument(
        "--level", default="all", choices=["primary", "all", *LEVELS]
    )
    anticipation.add_argument(
        "--target_set", default="all", choices=["primary", "all", "spillover"]
    )
    anticipation.add_argument("--output_path", default=None)
    anticipation.add_argument("--system_message", default="You are a helpful assistant.")
    anticipation.add_argument("--K", type=int, default=4)
    anticipation.add_argument(
        "--label_rotations",
        type=int,
        default=3,
        help="Ask each forecast under this many cyclic label rotations and pool the samples. The default 3 puts each of MORE/LESS/SAME in each position exactly once, matching the executed coding/UE/values forecast design (3 rotations x K=4 = 12 samples per forecast). Valid only against M0-v3+ endpoints, which trained all rotations; pass 1 to reproduce the old single-order design. Total samples per forecast = K x rotations.",
    )
    anticipation.add_argument("--timeout", type=int, default=5)
    anticipation.add_argument("--results_path", default=None)
    anticipation.add_argument("--raw_dump_path", default=None)
    anticipation.add_argument(
        "--no-timestamp", "--no_timestamp", dest="no_timestamp", action="store_true"
    )

    method_anticipation = sub.add_parser(
        "rl-preference-anticipation",
        help="forecast the model's own post-training RL-method preference",
    )
    add_mode_arguments(method_anticipation, model_required=True)
    method_anticipation.add_argument("--level", required=True, choices=[*LEVELS, "all"])
    method_anticipation.add_argument(
        "--anticipated-training-method",
        choices=["sft", "dpo", "grpo", "ppo"],
        required=True,
    )
    method_anticipation.add_argument("--config_key", default=None)
    method_anticipation.add_argument("--save_dir", default=None)
    method_anticipation.add_argument("--save_suffix", default=None)
    method_anticipation.add_argument("--seed", type=int, default=42)
    method_anticipation.add_argument("--raw_dump_path", default=None)
    method_anticipation.add_argument(
        "--no-timestamp", "--no_timestamp", dest="no_timestamp", action="store_true"
    )
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"rl-preferences", "rl-preference-anticipation"}:
        if args.level == "all":
            if args.raw_dump_path:
                raise ValueError(
                    "--raw_dump_path cannot name one file with --level all; "
                    "omit it to get one level-specific raw file per run"
                )
            requested_suffix = args.save_suffix
            for level in LEVELS:
                args.level = level
                args.save_suffix = (
                    f"{requested_suffix}_{level}" if requested_suffix else None
                )
                if args.command == "rl-preferences":
                    await run_rl_preferences(args)
                else:
                    await run_rl_preference_anticipation(args)
        else:
            if args.command == "rl-preferences":
                await run_rl_preferences(args)
            else:
                await run_rl_preference_anticipation(args)
    elif args.command == "rl-anticipation":
        await run_rl_anticipation(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
