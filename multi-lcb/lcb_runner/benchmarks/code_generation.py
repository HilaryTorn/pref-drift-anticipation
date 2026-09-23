import json
import zlib
import pickle
import base64
import os

from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Union, List, TypedDict
from functools import partial

from datasets import load_dataset
from multiprocessing import Pool

from prompt_enhancer.unified_converter import ConverterFactory
from lcb_runner.utils.config_utils import ConfigLCB


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    __test__ = False  # not a pytest
    STDIN = "stdin"
    FUNCTIONAL = "functional"


class EvalSample(TypedDict):
    # JsonEncoded tests input/outuputs data
    input_output: str


@dataclass(kw_only=True)
class Test:
    __test__ = False  # not a pytest
    input: str
    output: str
    testtype: TestType

    @staticmethod
    def parse_raw_json(data: str) -> Dict[str, Any] | Any:
        """
        Parses raw JSON string data into a Python object or a dictionary of parameters.

        This method handles multi-line JSON inputs by treating each line as a separate
        parameter and single-line inputs as a direct value.

        :param data: The raw input string, potentially containing multiple JSON lines.
        :returns: A dictionary with named parameters if multiple JSON values are found,
                  the single parsed JSON value if only one is found, or an empty dictionary
                  if the input is empty or represents an empty list/string.
        :rtype: Dict[str, Any] | Any

        :Example:
        >>> Test.parse_raw_json("3\n[1,2,3]")
        {'param_0': 3, 'param_1': [1, 2, 3]}
        >>> Test.parse_raw_json("123")
        {'param_0': 123}
        >>> Test.parse_raw_json('""')
        {'param_0': ''}
        >>> Test.parse_raw_json('[]')
        {'param_0': []}
        """
        if not data.strip():
            return {"param_0": data}

        lines = [raw for raw in data.split("\n") if raw.strip() != ""]
        parsed_values = [json.loads(raw) for raw in lines]

        if len(parsed_values) > 1:
            input_data = {f"param_{i}": val for i, val in enumerate(parsed_values)}
            return input_data
        elif len(parsed_values) == 1:
            return {"param_0": parsed_values[0]}
        else:
            return {"param_0": ""}

    def __post_init__(self):
        self.testtype = TestType(self.testtype)


@dataclass(kw_only=True)
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict
    format_version: str = "v2"

    @staticmethod
    def worker(args: dict, format_version=None):
        if format_version:
            args["format_version"] = format_version
        return CodeGenerationProblem(**args)

    def __str__(self):
        return f"{self.__class__.__name__}(q_id={self.question_id})"

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)

        if isinstance(self.contest_date, str):
            self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except (json.JSONDecodeError, TypeError):
            self.private_test_cases = pickle.loads(
                zlib.decompress(
                    base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                )
            )  # type: ignore
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore

        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

        version = self.format_version.lower()

        assert all(
            test.testtype == self.public_test_cases[0].testtype
            for test in self.public_test_cases
        ), f"Public test cases have different test types: {self.public_test_cases[0].testtype} != {self.public_test_cases[1].testtype}"

        assert all(
            test.testtype == self.private_test_cases[0].testtype
            for test in self.private_test_cases
        ), f"Private test cases have different test types: {self.public_test_cases[0].testtype} != {self.public_test_cases[1].testtype}"

        tests_obj: List[Test] = []
        tests_data: List[dict] = []
        for test in self.public_test_cases + self.private_test_cases:
            if test.testtype == TestType.FUNCTIONAL:
                new_input = Test.parse_raw_json(test.input)
                new_output = Test.parse_raw_json(test.output)
                tests_obj.append(test)
                tests_data.append({"input": new_input, "output": new_output["param_0"]})

        converter = ConverterFactory.create_converter(version)
        converted_tests = converter.convert_all_tests(tests_data)

        # fix all FUNCTIONAL tests
        for i, test in enumerate(tests_obj):
            converted_test = converted_tests[i]
            test.input = converted_test["stdin"]
            if test.input and not test.input.endswith("\n"):
                test.input += "\n"
            test.output = converted_test["stdout"]
            if test.output and not test.output.endswith("\n"):
                test.output += "\n"
            test.testtype = TestType.STDIN

    # merge code and outputs with questions
    @staticmethod
    def _prompt_to_str(prompt: Union[None, str, List[str]]) -> str:
        if prompt is None:
            return ""

        if isinstance(prompt, list):
            try:
                return json.dumps(prompt, ensure_ascii=False)
            except TypeError:
                return str(prompt)
        return prompt

    def insert_output(
        self,
        output_list: List[str],
        code_list: List[str],
        prompt: Union[None, str, List[str]] = "",
    ) -> dict:

        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "prompt": self._prompt_to_str(prompt),
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        prompt: Union[str, List[str]] = "",
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list, prompt=prompt)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def is_functional(self) -> bool:
        return any(
            t.testtype == TestType.FUNCTIONAL
            for t in self.public_test_cases + self.private_test_cases
        )

    def get_evaluation_sample(self) -> EvalSample:
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                        if t.testtype == TestType.STDIN
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                        if t.testtype == TestType.STDIN
                    ],
                    "fn_name": None,  # not used
                }
            ),
        }


def convert_codegen_data(
    dataset, n_cores: int, format_version: str
) -> list[CodeGenerationProblem]:
    """Converts codegen dataset into list of codegen problems."""
    n_cores = min(n_cores, len(dataset))

    worker = partial(CodeGenerationProblem.worker, format_version=format_version)

    if n_cores == 1:
        dataset = [worker(x) for x in dataset]
    else:
        with Pool(processes=n_cores) as pool:
            dataset = pool.map(worker, dataset)

    return dataset


def load_code_generation_dataset(args: ConfigLCB) -> list[CodeGenerationProblem]:

    debug = args.debug
    start_date = args.start_date
    end_date = args.end_date
    n_proc = args.num_process

    print("Loading code generation problems")
    if not os.path.exists(args.dataset_path):
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            split="test",
            version_tag=args.release_version,
            trust_remote_code=True,
            num_proc=n_proc,
        )
    else:
        print(f"Dataset found in {args.dataset_path}")
        debug_data_path = os.path.join(args.dataset_path, "test6.jsonl")
        if debug and os.path.exists(debug_data_path):
            dataset = load_dataset(
                args.dataset_path,
                split="train",
                data_files=["test6.jsonl"],
                # trust_remote_code=True,
                cache_dir=args.dataset_cache,
                num_proc=n_proc,
            )
        else:
            dataset = load_dataset(
                args.dataset_path,
                split="test",
                # trust_remote_code=True,
                cache_dir=args.dataset_cache,
                num_proc=n_proc,
            )

    dataset = convert_codegen_data(dataset, n_proc, format_version=args.format_version)

    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")

    return dataset


def load_code_generation_dataset_not_fast(
    args: ConfigLCB,
) -> list[CodeGenerationProblem]:

    print("Loading code generation problems (not fast)")

    num_proc = args.num_process
    dataset = load_dataset(
        "livecodebench/code_generation", split="test", num_proc=num_proc
    )
    dataset = convert_codegen_data(
        dataset, num_proc, format_version=args.format_version
    )

    print(f"Loaded {len(dataset)} problems")
    return dataset
