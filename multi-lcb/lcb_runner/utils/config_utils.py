from typing import List, Literal, Optional
import argparse
from lcb_runner.utils.scenarios import Scenario
from pathlib import Path
from dataclasses import dataclass, field, fields, MISSING


FormatVersions = Literal["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]


@dataclass(kw_only=True)
class ConfigDataset:
    """Load and prefilter dataset."""
    dataset_path: str
    dataset_cache: str
    release_version: str # LCB release version
    dataset_fast_load: bool
    start_date: Optional[str] = None  # format - YYYY-MM-DD
    end_date: Optional[str] = None  # format - YYYY-MM-DD
    ds_name: Optional[str] = None  # example: ds_name = "lcb_dataset"


@dataclass(kw_only=True)
class ConfigEvaluation:
    evaluate: bool
    continue_existing_eval: bool
    num_process_evaluate: int
    eval_timeout: int  # timeout for evaluation
    eval_restarts: int


@dataclass(kw_only=True)
class ConfigGeneration:
    continue_existing: bool
    n: int
    temperature: float
    top_p: float
    max_tokens: int
    max_seq_length: int
    gen_timeout: int  # timeout for generation
    cot_code_execution: bool  # enable reasoning for cot models
    chat_template_kwargs: dict  # params passed to chat_template
    enable_reasoning_kw: str  # will be passed to activate cot in chat_template

    def __post_init__(self):

        if not self.chat_template_kwargs:
            self.chat_template_kwargs = {}

        if not isinstance(self.chat_template_kwargs, dict):
            raise ValueError("Argument 'chat_template_kwargs' must be a dict")

        if self.cot_code_execution:
            self.chat_template_kwargs[self.enable_reasoning_kw] = True


@dataclass(kw_only=True)
class ConfigDebug:
    debug: bool = False
    enhance_prompts: bool = True
    format_version: FormatVersions = "v2"
    debug_size: Optional[int] = 10  # number of samples to use in debug run


@dataclass(kw_only=True)
class ConfigLCB(ConfigDataset, ConfigEvaluation, ConfigGeneration, ConfigDebug):
    """Provides typing hints for parsed LCB config."""

    scenario: Scenario
    output_dir: str  # output folder name (for all generations and evaluations)
    plangs: List[str]
    model: str
    local_model_path: Optional[str]
    tokenizer_path: Optional[str]
    num_process: int  #
    use_cache: bool
    cache_batch_size: int
    batch_size: int
    stop_token: str
    plang: str = "python"  # plang that is being processed, auto overwritten

    @staticmethod
    def parse_arguments(args: argparse.Namespace):
        kwgs = {}
        missing_fields = []
        for slot in fields(ConfigLCB):
            if not hasattr(args, slot.name):
                if slot.default == MISSING and slot.init:
                    missing_fields.append(slot.name)
            else:
                kwgs[slot.name] = getattr(args, slot.name)

        assert len(missing_fields) == 0, ValueError(
            f"Arguments missing following keys: {missing_fields}"
        )
        return ConfigLCB(**kwgs)  # pylint: disable=E1125


@dataclass
class ConfigEvalScores:
    """Provides typing hints for parsed compute_scores.py config."""

    scenario: Scenario
    output_dir: str  # output folder name (for all generations and evaluations)
    eval_all_file: str | Path | None  # redefine output filename
    model: str
    n: int
    temperature: float
    top_p: float
    platform: Optional[Literal["leetcode", "codeforces", "atcoder"]]
    start_date: Optional[str]  # format - YYYY-MM-DD
    end_date: Optional[str]  # format - YYYY-MM-DD
    cot_code_execution: bool  # enable reasoning for cot models
    debug: bool
    plang: dict = field(init=False)  # plang that is being processed, auto overwritten

    @staticmethod
    def parse_arguments(args: argparse.Namespace):
        kwgs = {}
        missing_fields = []
        for slot in fields(ConfigEvalScores):
            if not hasattr(args, slot.name):
                if slot.default == MISSING and slot.init:
                    missing_fields.append(slot.name)
            else:
                kwgs[slot.name] = getattr(args, slot.name)

        assert len(missing_fields) == 0, ValueError(
            f"Arguments missing following keys: {missing_fields}"
        )
        return ConfigEvalScores(**kwgs)  # pylint: disable=E1125
