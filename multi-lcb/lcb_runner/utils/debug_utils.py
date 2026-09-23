from typing import List
from lcb_runner.utils.path_utils import get_cmd_path
import time
from lcb_runner.utils.config_utils import ConfigLCB, ConfigEvalScores
import sys


def save_cmd_args(cmd: List[str], args: ConfigLCB | ConfigEvalScores) -> None:
    """Saves arguments passed in the input."""

    if not cmd:
        cmd = sys.argv[1:]

    out_path = get_cmd_path(args)
    begin_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"Run time: {begin_ts}\n")
        f.write(" ".join(cmd) + "\n")
