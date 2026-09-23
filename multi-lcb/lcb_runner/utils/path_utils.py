import os
from lcb_runner.lm_styles.models_store import LanguageModel, LMStyle
from lcb_runner.utils.config_utils import ConfigLCB
from pathlib import Path


def get_output_folder_name(model: LanguageModel, args: ConfigLCB) -> str:
    """Returns output folder name.

    Args:
        model (LanguageModel): model that is being processsed
        args (ConfigLCB): LCB config

    Returns:
        output_folder (str): folder name where all results will be saved

    Example:
        "Qwen3-32B_cot"
    """

    output_folder = model.model_repr

    if (
        model.model_style in (LMStyle.SGLangAsync, LMStyle.VLLMAsync)
        and args.local_model_path
    ):

        checkpoint_path: str = args.local_model_path
        parts = checkpoint_path.split("/")
        if parts[-1].startswith(("step-", "step_")):
            #  ../model_name/step_500 -> ../output/model_name_step_500
            output_folder = f"{'_'.join(parts[-2:])}"
        elif parts[-1].startswith(("epoch-", "epoch_", "ep_", "ep-")):
            #  ../model_name/epoch-100 -> ../output/model_name_epoch-100
            output_folder = f"{'_'.join(parts[-2:])}"
        else:
            output_folder = f"{parts[-1]}"

    if args.cot_code_execution:
        # ../output/model_name -> ../output/model_name_cot
        output_folder += "_cot"

    if hasattr(args, "debug") and args.debug:
        # ../output/model_name -> ../output/debug_model_name
        output_folder = "debug_" + output_folder

    if hasattr(args, "ds_name") and args.ds_name:
        output_folder = f"{args.ds_name}_{output_folder}"

    return output_folder


def ensure_dir(path: str, is_file=True):
    """Generates missing directory given input path.

    Args:
        path (str): path to file or folder
        is_file (bool, optional): Is path pointing to file. Defaults to True.
    """

    if is_file:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)
    return


def get_cache_path(args: ConfigLCB) -> str:
    scenario = "codegeneration"
    dir_name = args.output_dir
    n = args.n
    top_p = args.top_p
    temperature = args.temperature
    path = f"cache/{dir_name}/{scenario}_{n}_{temperature}_{top_p}.json"
    ensure_dir(path)
    return path


def get_gen_output_path(args: ConfigLCB) -> str:
    """Returns filename where generations will be stored.

    Example:
        .../LiveCodeBench-mult/output/Qwen3-32B_cot/codegeneration_c++_10_0.6_0.95_cot.json
    """

    scenario = "codegeneration"
    dir_name = args.output_dir
    n = args.n
    top_p = args.top_p
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    plang = args.plang.lower()
    path = f"output/{dir_name}/{scenario}_{plang}_{n}_{temperature}_{top_p}{cot_suffix}.json"
    ensure_dir(path)
    return path


def get_final_scores_path(args: ConfigLCB) -> str:
    """Returns filename.

    Example:
        .../LiveCodeBench-mult/output/Qwen3-32B_cot/main_codegeneration_10_0.6_0.95_cot.csv
    """

    scenario = "generation"
    dir_name = args.output_dir
    n = args.n
    top_p = args.top_p
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""

    path = (
        f"output/{dir_name}/main_{scenario}_{n}_{temperature}_{top_p}{cot_suffix}.csv"
    )
    ensure_dir(path)
    return path


def get_eval_all_output_path(args: ConfigLCB) -> str:
    """Returns filename.

    Example:
        .../LiveCodeBench-mult/output/Qwen3-32B_cot/eval_all_codegeneration_c++_10_0.6_0.95_cot.json
    """
    output_path = get_gen_output_path(args)
    head, tail = os.path.split(output_path)
    path = os.path.join(head, "eval_all_" + tail)
    return path


def get_eval_output_path(args: ConfigLCB) -> str:
    """Returns filename.

    Example:
        .../LiveCodeBench-mult/output/Qwen3-32B_cot/eval_codegeneration_c++_10_0.6_0.95_cot.json
    """

    output_path = get_gen_output_path(args)
    head, tail = os.path.split(output_path)
    path = os.path.join(head, "eval_" + tail)
    return path


def get_timer_path(args: ConfigLCB) -> str:
    """Returns filename.

    Example:
        .../LiveCodeBench-mult/output/Qwen3-32B_cot/debug_timer.txt
    """
    dir_name = args.output_dir
    path = f"output/{dir_name}/debug_timer.txt"
    ensure_dir(path)
    return path


def get_cmd_path(args: ConfigLCB) -> str:
    """Returns filename.

    Example:
        .../LiveCodeBench-mult/output/Qwen3-32B_cot/debug_cmd.txt
    """
    dir_name = args.output_dir
    path = f"output/{dir_name}/debug_cmd.txt"
    ensure_dir(path)
    return path
