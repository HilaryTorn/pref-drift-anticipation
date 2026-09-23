from lcb_runner.utils import ConfigLCB
from lcb_runner.runner.parser import get_args


def create_default_config(
    model_type: str = "VLLMAsync",
    local_model_path: str = "chat_gpht",
    debug: bool = None,
    not_fast: str = None,
    plangs="all",
    enable_reasoning_kw: str = None,
    chat_template_kwargs: str = None,
    cot_code_execution: bool = True,
) -> ConfigLCB:

    cmd = [
        "--model",
        f"{model_type}",
        "--local_model_path",
        f"{local_model_path}",
        "--max_tokens",
        "30000",
        "--max_seq_length",
        "32768",
        "--evaluate",
        "--temperature",
        "0.2",
        "--top_p",
        "0.95",
        "--N",
        "10",
        "--start_date",
        "2024-07-01",
        "--batch_size",
        "50",
        "--plangs",
        f"{plangs}",
        "--num_process_evaluate",
        "60",
    ]

    if cot_code_execution:
        cmd.append("--cot_code_execution")

    if chat_template_kwargs:
        cmd.append("--chat_template_kwargs")
        cmd.append(chat_template_kwargs)

    if enable_reasoning_kw:
        cmd.append("--enable_reasoning_kw")
        cmd.append(enable_reasoning_kw)

    if not_fast:
        cmd.append("--not_fast")

    if debug:
        cmd.append("--debug")

    conf = get_args(cmd)
    return conf
