import time
import json
from hashlib import sha256
from lcb_runner.utils.config_utils import ConfigLCB
from lcb_runner.utils.path_utils import get_timer_path
from typing import List, Optional
import sys


def dict_to_hash(data_dict: dict) -> str:
    json_string = json.dumps(data_dict)
    hex_string = json_string.encode("utf-8")
    return sha256(hex_string).hexdigest()


class Timer:
    """Logs time of execution of each step."""

    _temp_ts = {}
    data = {}

    def __init__(self, args: ConfigLCB, cmd: Optional[List[str]] = None):

        self.path = get_timer_path(args)
        self.begin_time = time.time()  # unix value
        self.begin_ts = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.gmtime()
        )  # human readable time
        self.meta = {
            "begin_ts": self.begin_ts,
            "temp": args.temperature,
            "top_p": args.top_p,
            "cot": args.cot_code_execution,
            "eval_num_cores": args.num_process_evaluate,
            "N": args.n,
        }

        if not cmd:
            self.cmd = sys.argv[1:]
        else:
            self.cmd = cmd

        self.run_id = dict_to_hash(self.meta)
        self._save_meta()

    def _save_meta(self):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("-" * 30 + "\n")
            f.write(f"run_id: {self.run_id}" + "\n")
            f.write(f"meta: {json.dumps(self.meta)}" + "\n")
            f.write("args: " + " ".join(self.cmd) + "\n")

    def _save_key(self, key: str):
        if key not in self.data:
            return

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{key}: {self.data[key]}" + "\n")

    def finalize(self):
        self.data["total_time"] = round(time.time() - self.begin_time, 2)
        self._save_key("total_time")

        print("Timer stats:")
        for k, v in self.data.items():
            print(f"{k}: {v}")

    def start(self, key: str):
        self._temp_ts[key] = time.time()

    def stop(self, key: str, verbose=False):
        if key not in self._temp_ts:
            return
        self.data[key] = round(time.time() - self._temp_ts[key], 2)
        self._save_key(key)
        if verbose:
            print(f"Time spent '{key}': {self.data[key]}")

    def gen_start(self, plang: str):
        self.start(f"gen_{plang}")

    def gen_stop(self, plang: str):
        self.stop(f"gen_{plang}")

    def eval_start(self, plang: str):
        self.start(f"eval_{plang}")

    def eval_stop(self, plang: str):
        self.stop(f"eval_{plang}")
