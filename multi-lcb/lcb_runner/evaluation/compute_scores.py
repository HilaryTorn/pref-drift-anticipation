import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
import os


from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.evaluation.pass_k_utils import (
    estimate_pass_at_k,
)
from lcb_runner.utils import (
    ConfigLCB,
    ConfigEvalScores,
    PLANGS,
    Scenario,
    get_eval_all_output_path,
    get_final_scores_path,
    get_output_folder_name,
)


def get_parser() -> ConfigEvalScores:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0301",
        help="Name of the model to use matching `lm_styles.py`",
    )

    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="If you have a local model, specify it here in conjunction with --model",
    )

    parser.add_argument(
        "--n", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument("--N", dest="n", type=int, help="Alias for 'n' parameter")

    parser.add_argument(
        "--eval_all_file",
        type=str,
        default=None,
        help="Alternative way to provide the evaluation file. Use with '.csv' suffix. ",
    )

    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for sampling"
    )

    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for sampling")

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
        "--platform",
        type=str,
        default=None,
        help="Platform to filter the evaluation file",
    )

    parser.add_argument(
        "--cot_code_execution",
        action="store_true",
        help="whether to use CoT in code execution scenario",
    )

    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()
    args.scenario = Scenario.codegeneration

    model = LanguageModelStore[args.model]
    args.output_dir = get_output_folder_name(model, args)

    if hasattr(args, "eval_all_file") and args.eval_all_file is not None:
        args.eval_all_file = Path(
            "output", args.output_dir, args.eval_all_file
        ).with_suffix(".csv")

    conf = ConfigEvalScores.parse_arguments(args)
    return conf


def compute_plang_scores(args: ConfigLCB | ConfigEvalScores) -> pd.DataFrame:

    plang = args.plang
    eval_all_file = get_eval_all_output_path(args)

    if not os.path.exists(eval_all_file):
        return pd.DataFrame()

    metrics_dict = {}
    metrics_dict["plang"] = plang

    with open(eval_all_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    for res in results:
        res["contest_date"] = datetime.fromisoformat(res["contest_date"])

    if args.start_date is not None:
        p_start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

        results = [
            result for result in results if p_start_date <= result["contest_date"]
        ]

    if args.end_date is not None:
        p_end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

        results = [result for result in results if result["contest_date"] <= p_end_date]

    if hasattr(args, "platform") and args.platform is not None:
        results = [result for result in results if result["platform"] == args.platform]

    totals = [len(x["graded_list"]) for x in results]
    corrects = [sum(x["graded_list"]) for x in results]

    easy_totals = [len(x["graded_list"]) for x in results if x["difficulty"] == "easy"]
    med_totals = [len(x["graded_list"]) for x in results if x["difficulty"] == "medium"]
    hard_totals = [len(x["graded_list"]) for x in results if x["difficulty"] == "hard"]
    easy_corrects = [
        sum(x["graded_list"]) for x in results if x["difficulty"] == "easy"
    ]
    med_corrects = [
        sum(x["graded_list"]) for x in results if x["difficulty"] == "medium"
    ]
    hard_corrects = [
        sum(x["graded_list"]) for x in results if x["difficulty"] == "hard"
    ]

    print(f"Scores for {plang}")
    for k in [1, 5, 10, 25, 50, 100, 150, 200]:
        if min(totals) < k:
            continue

        print(
            f"-- Pass@{k} = ",
            estimate_pass_at_k(totals, corrects, k).mean(),
        )
        print(
            f"-- Easy Pass@{k} = ",
            estimate_pass_at_k(easy_totals, easy_corrects, k).mean(),
        )
        print(
            f"-- Medium Pass@{k} = ",
            estimate_pass_at_k(med_totals, med_corrects, k).mean(),
        )
        print(
            f"-- Hard Pass@{k} = ",
            estimate_pass_at_k(hard_totals, hard_corrects, k).mean(),
        )

    pass_1_list = [result["pass@1"] for result in results]
    pass_1_val = sum(pass_1_list) / len(pass_1_list)
    # print(f"Pass@1: {pass_1_val}")
    metrics_dict["Pass@1"] = pass_1_val

    easy_pass_1_list = [
        result["pass@1"]
        for result in results
        if "difficulty" in result and result["difficulty"] == "easy"
    ]
    if len(easy_pass_1_list) > 0:
        pass_1_val = sum(easy_pass_1_list) / len(easy_pass_1_list)
        # print(f"Easy Pass@1: {pass_1_val}")
        metrics_dict["Easy Pass@1"] = pass_1_val

    medium_pass_1_list = [
        result["pass@1"]
        for result in results
        if "difficulty" in result and result["difficulty"] == "medium"
    ]
    if len(medium_pass_1_list) > 0:
        pass_1_val = sum(medium_pass_1_list) / len(medium_pass_1_list)
        # print(f"Medium Pass@1: {pass_1_val}")
        metrics_dict["Medium Pass@1"] = pass_1_val

    hard_pass_1_list = [
        result["pass@1"]
        for result in results
        if "difficulty" in result and result["difficulty"] == "hard"
    ]
    if len(hard_pass_1_list) > 0:
        pass_1_val = sum(hard_pass_1_list) / len(hard_pass_1_list)
        # print(f"Hard Pass@1: {pass_1_val}")
        metrics_dict["Hard Pass@1"] = pass_1_val

    df = pd.DataFrame([metrics_dict])
    return df


def combine_eval_results(args: ConfigLCB | ConfigEvalScores):

    df = pd.DataFrame()
    scored_plangs = []
    for _, plang in enumerate(PLANGS):
        args.plang = plang.lower()
        plang_df = compute_plang_scores(args)
        df = pd.concat([df, plang_df], axis=0)

        if plang_df.shape[0] > 0:
            scored_plangs.append(plang)

    if hasattr(args, "eval_all_file") and args.eval_all_file is not None:
        output_file_path = args.eval_all_file
    else:
        output_file_path = get_final_scores_path(args)

    df.to_csv(output_file_path, index=False)

    print(f"Plangs in the output: {scored_plangs}")
    print(f"Final outputs saved: {output_file_path}")


if __name__ == "__main__":
    combine_eval_results(get_parser())
