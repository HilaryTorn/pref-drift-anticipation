import os
import argparse

from typing import Sequence
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.utils import ConfigLCB, Scenario, PLANGS, PLang, get_output_folder_name
import json


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0301",
        help="Name of the model to use matching `models_store.py`",
    )

    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="If you have a local model, specify it here in conjunction with --model",
    )

    parser.add_argument(
        "--cot_code_execution",
        action="store_true",
        help="Enable/disable thinking. Disabled by default. Whether to use CoT in code execution scenario",
    )

    parser.add_argument(
        "--n", type=int, default=10, help="Number of samples to generate"
    )

    parser.add_argument("--N", dest="n", type=int, help="Alias for 'n' parameter")

    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for sampling"
    )

    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for sampling")
    parser.add_argument(
        "--max_tokens", type=int, default=2000, help="Max tokens for sampling"
    )

    parser.add_argument(
        "--max_seq_length", type=int, default=32768, help="Max seq length for VLLM"
    )

    parser.add_argument(
        "--num_process",
        default=-1,
        type=int,
        help="Number of processes to use for loading data and for generation (vllm/sglang runs do not use this)",
    )

    parser.add_argument(
        "--stop_token",
        default="###",
        type=str,
        help="Stop token (use `,` to separate multiple tokens)",
    )

    parser.add_argument(
        "--continue_existing", action="store_true", help="use existing generations"
    )

    parser.add_argument(
        "--continue_existing_eval", action="store_true", help="use existing evaluations"
    )

    parser.add_argument(
        "--use_cache", action="store_true", help="Use cache for generation"
    )
    parser.add_argument(
        "--cache_batch_size", type=int, default=100, help="Batch size for caching"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--debug_size",
        type=int,
        default=10,
        help="Count problems that will be used in debug",
    )

    parser.add_argument("--evaluate", action="store_true", help="Evaluate the results")
    parser.add_argument(
        "--num_process_evaluate",
        type=int,
        default=min(os.cpu_count(), 60),
        help="Number of processes to use for evaluation. High numbers may lead to TimeOut errors during evaluation. We use <=60",
    )

    parser.add_argument(
        "--timeout",
        dest="eval_timeout",
        type=int,
        default=6,
        help="Timeout for evaluation",
    )
    parser.add_argument(
        "--eval_timeout", type=int, default=6, help="Timeout for evaluation"
    )

    parser.add_argument(
        "--eval_restarts",
        type=int,
        default=3,
        help="Number of times to restart failed jobs in evaluation. At least 3 recommended.",
    )

    parser.add_argument(
        "--openai_timeout",
        dest="gen_timeout",
        type=int,
        default=300,
        help="Timeout for requests to OpenAI. Recommended to be >=300s for cot models.",
    )

    parser.add_argument(
        "--gen_timeout",
        type=int,
        default=300,
        help="Timeout for requests to OpenAI. Recommended to be >=300s for cot models.",
    )

    # Added to avoid running extra generations (it's slow for reasoning models)
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )

    parser.add_argument(
        "--not_fast",
        action="store_true",
        help="whether to use full set of tests (slower and more memory intensive evaluation)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Generation queue depth. Used for generation in async vllm.",
    )

    plangs = ", ".join(PLANGS)
    parser.add_argument(
        "--plangs",
        type=str,
        default="python",
        help=f"'all' or list of programming languages to use {plangs}",
    )

    parser.add_argument(
        "--enable_reasoning_kw",
        type=str,
        default="enable_thinking",
        help="Keyword used in chat template that enables 'thinking' token",
    )

    parser.add_argument(
        "--chat_template_kwargs",
        type=str,
        default="{}",
        help="JSON-encoded string representing dictionary with data.",
    )

    return parser


def clean_parsed_args(args: argparse.Namespace) -> ConfigLCB:

    args.scenario = Scenario.codegeneration

    args.chat_template_kwargs = json.loads(args.chat_template_kwargs)

    args.dataset_path = os.environ.get("DATASET_PATH", "")
    args.dataset_cache = os.environ.get("DATASET_CACHE", "")
    args.release_version = os.environ.get("RELEASE_VERSION", "release_v6")
    args.ds_name = os.environ.get("DATASET_NAME", "")  # will be appended to the output folder name
    args.dataset_fast_load = not args.not_fast

    # developer args
    args.enhance_prompts = os.environ.get("ENHANCE_PROMPTS", "True") == "True"
    args.format_version = os.environ.get("FORMAT_VERSION", "v2")

    model = LanguageModelStore[args.model]

    if os.path.exists(args.local_model_path):
        args.tokenizer_path = args.local_model_path
    else:
        args.tokenizer_path = None

    args.output_dir = get_output_folder_name(model, args)
    args.stop_token = args.stop_token.split(",")

    if args.plangs.lower() == "all":
        args.plangs = PLANGS
    else:
        fuzzy_match = [PLang.fuzzy_match(x) for x in args.plangs.split(",")]
        args.plangs = [x for x in PLANGS if x in set(fuzzy_match)]

    assert args.plangs, ValueError("No programming languages specified")

    if args.num_process <= 0:
        args.num_process = os.cpu_count()

    if args.num_process_evaluate <= 0:
        args.num_process_evaluate = os.cpu_count()

    conf = ConfigLCB.parse_arguments(args)
    return conf


def get_args(cmd: Sequence[str] = None) -> ConfigLCB:
    """Parse arguments passed in the command line.

    Args:
        cmd (Sequence[str], optional): List with command line arguments. Defaults to None.
        save_log (bool, optional): save command line arguments into the file

    Returns:
        ConfigLCB: mLCB config object
    """

    parser = create_parser()
    args = parser.parse_args(cmd)
    conf = clean_parsed_args(args)

    return conf
