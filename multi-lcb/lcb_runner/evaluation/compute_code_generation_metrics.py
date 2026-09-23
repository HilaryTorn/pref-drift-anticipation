from typing import List, Tuple, Dict
import json
from collections import defaultdict
from multiprocessing.pool import Pool
from tqdm import tqdm

from lcb_runner.evaluation.metrics_typing import (
    CodeEvalScores,
    CodeEvalMetada,
    SolutionCode,
    ProblemDataJson,
    CodeEvalMetadaJson,
    PassAtKStats,
)
from lcb_runner.evaluation.testing_plang import (
    eval_plang_code,
    prepare_plang_env,
    TestScore,
)
from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results

from dataclasses import asdict


def evaluate_generations_by_problem(
    args: list,
) -> Tuple[List[CodeEvalScores], List[CodeEvalMetada]]:
    problem_generations: List[SolutionCode] = args[0]
    sample = args[1]
    timeout: int = args[2]
    plang: int = args[3]

    in_outs = json.loads(sample["input_output"])
    all_inputs: List[str] = in_outs["inputs"]  # given
    all_outputs: List[str] = in_outs["outputs"]  # expected outputs
    _ = in_outs["fn_name"]  # always None

    res = []
    metadata = []
    for problem_code in problem_generations:
        cur_res, cur_metadata = eval_plang_code(
            problem_code, all_inputs, all_outputs, plang=plang, timeout=timeout
        )

        cur_metadata = asdict(cur_metadata)
        if "error" in cur_metadata:
            # converting field to string for pickle purpose
            cur_metadata["error"] = str(cur_metadata["error"])

        # non ints will cause errors later in the code
        cur_res = [
            int(v.value) if isinstance(v, TestScore) else int(v) for v in cur_res
        ]

        res.append(cur_res)
        metadata.append(cur_metadata)

    return res, metadata


def _worker(args: list):
    idx: int = args[0]
    return idx, evaluate_generations_by_problem(args[1:])


def evaluate_generations(
    samples_list: list[ProblemDataJson],
    generations_list: list[list[SolutionCode]],
    num_process_evaluate: int = 16,
    timeout: int = 6,
    plang: str = "python",
    n_restarts: int = 3,
) -> Tuple[Dict[int, List[CodeEvalScores]], Dict[int, List[CodeEvalMetada]]]:
    """We take the list of code generations and try to compile them
     and the run their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS dataset)
        samples_list: list of inputs/outputs for tests for each problem

    Returns:
        results: dictionary of results, key is the problem index, value is a list of results for each generation
    """

    # generations are code generations in the same order of the dataset
    # inputs are flattened generations, i.e. each generations_list consist of list with single element [generation]
    n_inputs = len(generations_list)

    inputs = [
        (index, generations_list[index], samples_list[index], timeout, plang)
        for index in range(n_inputs)
    ]

    results = {}  # map: idx -> list[list[int]]
    metadata = {}  # map: idx -> list[dict]

    cnt_runs = 0
    max_runs = max(1, n_restarts)
    while cnt_runs < max_runs:
        # restart evaluation with failed tasks

        cnt_runs += 1
        tasks_not_done = [w for w in inputs if w[0] not in results]

        # Tasks failed with Timeout may have failed because of errors in a thread/pool
        tasks_for_restart = [
            inputs[idx]
            for idx, m in metadata.items()
            if m[-1].get("error", "NoErrors") in ("TimeoutExpired", "BuildTimeOut")
        ]
        total_tasks = tasks_not_done + tasks_for_restart
        n_tasks = len(total_tasks)
        if n_tasks == 0:
            break

        # each restart reduce num cores in evaluator
        cpu_scale_perc = 1 - 0.2 * (cnt_runs - 1)
        n_proc = max(1, int(num_process_evaluate * cpu_scale_perc))
        n_proc = min(n_proc, n_tasks)

        prepare_plang_env(plang)

        print(
            f"Running in {n_proc} threads. Run N {cnt_runs}/{max_runs}. Timeout={timeout}s"
        )
        with Pool(n_proc) as pool:
            results_iterator = pool.imap_unordered(_worker, total_tasks)
            for idx, (r, m) in tqdm(
                results_iterator, total=n_tasks, desc="Evaluating code gens"
            ):
                results[idx] = r
                metadata[idx] = m

    assert (
        len(results) == n_inputs
    ), f"results = {len(results)} inputs = {n_inputs} {results=}"

    return results, metadata


def codegen_metrics(
    samples_list: List[ProblemDataJson],
    generations_list: list[List[SolutionCode]],
    k_list: List[int] = (1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000),
    num_process_evaluate: int = 16,
    timeout: int = 6,
    plang: str = "python",
    n_restarts: int = 3,
) -> Tuple[
    PassAtKStats, Dict[int, List[CodeEvalScores]], List[List[CodeEvalMetadaJson]]
]:
    """
    generations_list - list with extracted code blocks for each task
    samples_list - List[dict] with tests data, each sample contains keys 'input','output','function_name'
    """

    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, (sample, generation_list) in enumerate(
        zip(samples_list, generations_list)
    ):
        assert isinstance(generation_list, list), generations_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)

    print(f"Evaluating {plang} on {len(samples_linear)} samples...")
    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
        plang=plang,
        n_restarts=n_restarts,
    )

    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        results[remap_index[idx]].append(sub_results[0])

    for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        metadatas[remap_index[idx]].append(sub_metadatas[0])

    pass_at_k = compute_metrics_from_results(results, k_list=k_list)
    pass_at_k["plang"] = plang

    final_metadata = []
    for key in sorted(list(metadatas.keys())):
        final_metadata.append(metadatas[key])
    for i in range(len(final_metadata)):
        if not isinstance(final_metadata[i], list):
            final_metadata[i] = [json.dumps(final_metadata[i])]
        else:
            final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]

        assert len(final_metadata[i]) == len(
            generations_list[0]
        ), f"{len(final_metadata[i])=}"

    return pass_at_k, results, final_metadata
