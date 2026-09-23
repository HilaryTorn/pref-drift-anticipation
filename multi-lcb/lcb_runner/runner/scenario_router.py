from typing import Union, List, Tuple, Callable, Dict
import time
import os

from lcb_runner.evaluation import (
    codegen_metrics,
)
from lcb_runner.prompts import get_format_prompt_generation, ChatMessages

from lcb_runner.evaluation.metrics_typing import (
    CodeEvalScores,
    CodeEvalMetadaJson,
    PassAtKStats,
)

from lcb_runner.utils import (
    ConfigLCB,
    Scenario,
    truncate_reasoning,
    extract_code,
)

from lcb_runner.benchmarks import (
    CodeGenerationProblem,
    load_code_generation_dataset,
    load_code_generation_dataset_not_fast,
)
from lcb_runner.lm_styles import LMStyle


BenchMarkType = Union[List[CodeGenerationProblem]]
PLang = str
PromptFormatter = Callable[[CodeGenerationProblem, LMStyle, PLang], ChatMessages]


def build_prompt_benchmark(
    args: ConfigLCB,
) -> tuple[List[CodeGenerationProblem], PromptFormatter]:

    if args.scenario != Scenario.codegeneration:
        raise ValueError(f"Scenario {args.scenario} not implemented")

    if os.path.exists(args.dataset_path) or args.dataset_fast_load:
        benchmark = load_code_generation_dataset(args)
    else:
        benchmark = load_code_generation_dataset_not_fast(args)

    benchmark = sorted(benchmark, key=lambda x: x.question_id)
    format_prompt = get_format_prompt_generation(args)
    return benchmark, format_prompt


def combine_results(scenario: Scenario, results: list[list[str]]):

    # because we save full answers it might be reasonble to exclude long thinking
    results = [
        [truncate_reasoning(output) for output in outputs_list]
        for outputs_list in results
    ]

    if scenario == Scenario.codegeneration:
        combined_results = [
            (
                outputs_list,
                [extract_code(output) for output in outputs_list],
            )
            for outputs_list in results
        ]
    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    return combined_results


def sort_and_extract_save_results(
    scenario: Scenario, save_results: list[dict]
) -> Tuple[List[dict], List[Tuple[List[str], List[str]]]]:

    if scenario == Scenario.codegeneration:
        save_results = sorted(save_results, key=lambda x: x["question_id"])
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["code_list"])
            for save_result_instance in save_results
        ]
    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    return save_results, combined_results


def get_metrics(
    scenario: Scenario,
    args: ConfigLCB,
    benchmark: BenchMarkType,
    combined_results: List[Tuple[List[str], List[str]]],
) -> Tuple[
    PassAtKStats, Dict[int, List[CodeEvalScores]], List[List[CodeEvalMetadaJson]]
]:
    """Evaluate generated code on given inputs/outputs.
    All datasets are sorted by question_id.

     combined_results: list with (output_text, extracted_code_block_text)
    """

    assert len(benchmark) == len(combined_results), ValueError(
        "Size and order of questions must be aligned"
    )

    st_time = time.time()
    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]

    # lists with extracted code blocks
    generations = [extracted for _, extracted in combined_results]

    if scenario == Scenario.codegeneration:
        metrics = codegen_metrics(
            eval_samples,
            generations,
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.eval_timeout,
            plang=args.plang,
            n_restarts=args.eval_restarts,
        )
    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    ## ---> Output main stats after evaluation

    # pass1_val is less than 1, for example val = 0.1842
    pass1_val = round(metrics[0]["pass@1"], 4)

    delta_time = round(time.time() - st_time, 1)
    metrics[0]["total_eval_time"] = delta_time
    short_metrics = {k: v for k, v in metrics[0].items() if k not in ("detail")}

    print(short_metrics)
    print("pass@1:", args.plang, pass1_val)
    print(f"Time passed: {delta_time}s")

    ## <----

    return metrics
