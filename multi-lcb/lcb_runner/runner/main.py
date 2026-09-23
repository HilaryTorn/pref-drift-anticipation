import os
import json
from typing import List

from lcb_runner.runner.parser import get_args, ConfigLCB
from lcb_runner.utils import (
    Scenario,
    Timer,
    get_gen_output_path,
    get_eval_output_path,
    get_eval_all_output_path,
)

from lcb_runner.lm_styles import LanguageModelStore, LanguageModel
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.evaluation.compute_scores import combine_eval_results
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    combine_results,
    sort_and_extract_save_results,
    get_metrics,
    BenchMarkType,
    PromptFormatter,
)


# LLM server that will process requests
runner = None


def generate_plang(
    model: LanguageModel,
    format_prompt: PromptFormatter,
    benchmark: BenchMarkType,
    args: ConfigLCB,
) -> None:
    global runner

    gen_output_path = get_gen_output_path(args)

    if not os.path.exists(gen_output_path):
        print(f"File {gen_output_path} does not exist, starting from scratch")

    if args.continue_existing and os.path.exists(gen_output_path):

        with open(gen_output_path, "r", encoding="utf-8") as f:
            old_save_results = json.load(f)

        old_save_results = [
            instance
            for instance in old_save_results
            if instance["output_list"] and [x for x in instance["output_list"] if x]
        ]
        old_save_results_question_ids = [
            instance["question_id"] for instance in old_save_results
        ]
        # Keep only those cached results that are still present in current benchmark
        current_benchmark_qids = {inst.question_id for inst in benchmark}
        old_save_results = [
            inst
            for inst in old_save_results
            if inst["question_id"] in current_benchmark_qids
        ]

        remaining_benchmark = [
            instance
            for instance in benchmark
            if instance.question_id not in old_save_results_question_ids
        ]
        print(
            f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining"
        )
    else:
        old_save_results = []
        remaining_benchmark = benchmark

    if len(remaining_benchmark) > 0:
        if not runner:
            runner = build_runner(args, model)
        results, prompts = runner.run_main(
            remaining_benchmark, format_prompt, args.plang
        )
    else:
        prompts = []
        results = []

    # extract code text from outputs
    combined_results = combine_results(
        args.scenario,
        results,
    )

    # merge code and outputs with questions
    save_results = [
        instance.insert_output(outputs_list, extracted_list, prompt)
        for instance, (outputs_list, extracted_list), prompt in zip(
            remaining_benchmark, combined_results, prompts
        )
    ]

    if len(old_save_results) > 0:
        save_results += old_save_results

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    with open(gen_output_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=4, ensure_ascii=False)


def eval_plang(args: ConfigLCB, benchmark: BenchMarkType):

    gen_output_path = get_gen_output_path(args)
    eval_all_file = get_eval_all_output_path(args)
    eval_file = get_eval_output_path(args)

    with open(gen_output_path, "r", encoding="utf-8") as f:
        saved_results: List[dict] = json.load(f)

    # Map prompts by question_id from generation results (if present)
    prompts_by_qid = {
        inst["question_id"]: inst.get("prompt", "") for inst in saved_results
    }

    # Align cache to current benchmark set (e.g., when converter/filtering changes)
    bench_qids = set([inst.question_id for inst in benchmark])
    generation_qids = set([inst["question_id"] for inst in saved_results])
    eval_qids = bench_qids.intersection(generation_qids)

    if len(eval_qids) != len(bench_qids):
        missing_qied = bench_qids - eval_qids
        n_missing = len(missing_qied)
        print(
            f"ERROR missing generations for {n_missing} problems. Skipping missing problems."
        )

    # filter benchmark and saved questions and align their order
    benchmark = [bench for bench in benchmark if bench.question_id in eval_qids]
    benchmark.sort(key=lambda x: x.question_id)

    saved_results = [inst for inst in saved_results if inst["question_id"] in eval_qids]
    saved_results.sort(key=lambda x: x["question_id"])

    _, combined_results = sort_and_extract_save_results(args.scenario, saved_results)

    if args.continue_existing_eval and os.path.exists(eval_all_file):
        with open(eval_all_file, encoding="utf-8") as fp:
            old_eval_all_results = json.load(fp)

        if os.path.exists(eval_file):
            with open(eval_file, encoding="utf-8") as fp:
                old_eval_results = json.load(fp)
        else:
            old_eval_results = None

        old_eval_results_question_ids = [
            instance["question_id"] for instance in old_eval_all_results
        ]

        remaining_indices = [
            idx
            for idx in range(len(benchmark))
            if benchmark[idx].question_id not in old_eval_results_question_ids
        ]
        benchmark = [benchmark[idx] for idx in remaining_indices]
        combined_results = [combined_results[idx] for idx in remaining_indices]

        old_eval_size = len(old_eval_results_question_ids)
        new_eval_size = len(benchmark)

        if new_eval_size == 0:
            return

        print(f"Found {old_eval_size}, running evals for {new_eval_size} problems")
        metrics = get_metrics(args.scenario, args, benchmark, combined_results)
        graded = extract_instance_results(metrics[1])

        if args.scenario == Scenario.codegeneration:
            metadatas = metrics[2]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list,
                    extracted_list,
                    graded_list,
                    prompt=prompts_by_qid.get(instance.question_id, ""),
                    metadata=meta,
                )
                for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                    benchmark, combined_results, graded, metadatas
                )
            ]
        else:
            raise NotImplementedError(f"Unknown scenario {args.scenario}")

        if old_eval_results:
            for key in metrics[0]:
                if key in old_eval_results[0]:
                    if key != "detail":
                        metrics[0][key] = (
                            old_eval_size * old_eval_results[0][key]
                            + new_eval_size * metrics[0][key]
                        )
                        metrics[0][key] /= old_eval_size + new_eval_size

            for key in metrics[0]["detail"]:
                if key in old_eval_results[0]["detail"]:
                    metrics[0]["detail"][key] = {
                        **metrics[0]["detail"][key],
                        **old_eval_results[0]["detail"][key],
                    }
            metrics[1] = {**metrics[1], **old_eval_results[1]}
        else:
            print("Old eval file not present, cannot update eval file")
            metrics = {}

    else:
        metrics = get_metrics(args.scenario, args, benchmark, combined_results)
        graded = extract_instance_results(metrics[1])
        old_eval_all_results = []

        if args.scenario == Scenario.codegeneration:
            if metrics:
                metadatas = metrics[2]
            else:
                metadatas = [[] for _ in benchmark]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list,
                    extracted_list,
                    graded_list,
                    prompt=prompts_by_qid.get(instance.question_id, ""),
                    metadata=meta,
                )
                for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                    benchmark, combined_results, graded, metadatas
                )
            ]
            if metrics and old_eval_all_results:
                metrics[2] = old_eval_all_results[2] + metrics[2]

        else:
            raise NotImplementedError(f"Unknown scenario {args.scenario}")

    save_eval_results = old_eval_all_results + save_eval_results

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    with open(eval_all_file, "w", encoding="utf-8") as f:
        json.dump(save_eval_results, f, indent=4, ensure_ascii=False)


def main():
    global runner

    args = get_args()
    timer = Timer(args)

    print(f"Processing langs: {args.plangs}")
    print(f"Output directory: 'output/{args.output_dir}'")

    timer.start("load_data")
    benchmark, format_prompt = build_prompt_benchmark(args)
    timer.stop("load_data", verbose=True)

    # remove benchmarks that can't be converted into stdin format
    benchmark = [bench for bench in benchmark if not bench.is_functional()]

    if args.debug:
        benchmark = benchmark[: args.debug_size]
        print(f"Running with {len(benchmark)} problems in debug mode")
    else:
        print(f"Running with {len(benchmark)} problems")

    n_plangs = len(args.plangs)

    # Step 1. generate solutions
    for i, plang in enumerate(args.plangs):
        model = LanguageModelStore[args.model]
        print(f"Generating for {plang}, cnt lang processed {i}/{n_plangs}")
        args.plang = plang.lower()
        timer.gen_start(plang)
        generate_plang(model, format_prompt, benchmark, args)
        timer.gen_stop(plang)

    if runner:
        # Turning off VLLM and freeing GPUs
        del runner

    # Step 2. evaluate solutions
    if args.evaluate:
        for i, plang in enumerate(args.plangs):
            print(f"Evaluating for {plang}, cnt lang processed {i}/{n_plangs}")
            args.plang = plang.lower()
            timer.eval_start(plang)
            eval_plang(args, benchmark)
            timer.eval_stop(plang)

        # aggregate all evaluations in the single file
        combine_eval_results(args)

    timer.finalize()


if __name__ == "__main__":
    main()
