from typing import Dict, List, TypedDict, Optional, NotRequired, Any
import numpy as np


problem_id = int  # pylint: disable=C0103
lcb_pass_rate = np.float64  # percent LLM generations that passed all tests
task_pass_rate = np.float64


ProblemScores = Dict[problem_id, task_pass_rate]

ScoresDetailed = TypedDict(
    "ScoresDetailed",
    {
        "pass@1": ProblemScores,
        "pass@5": NotRequired[ProblemScores],
        "pass@10": NotRequired[ProblemScores],
        "pass@20": NotRequired[ProblemScores],
        "pass@40": NotRequired[ProblemScores],
        "pass@50": NotRequired[ProblemScores],
    },
)

PassAtKStats = TypedDict(
    "PassAtKStats",
    {
        "plang": NotRequired[str],  # 'python','c++',...
        "total_eval_time": NotRequired[float],  # in seconds
        "pass@1": lcb_pass_rate,
        "pass@5": NotRequired[lcb_pass_rate],
        "pass@10": NotRequired[lcb_pass_rate],
        "pass@20": NotRequired[lcb_pass_rate],
        "pass@40": NotRequired[lcb_pass_rate],
        "pass@50": NotRequired[lcb_pass_rate],
        "detail": ScoresDetailed,
    },
)


ProblemTestsPassed = bool  # True if all tests passed, False otherwise
SolutionCode = str
TestScore = int | bool  # True or >0 if test passed, othewise negative
CodeEvalScores = List[TestScore]  # list with scores for each evaluated test
CodeEvalMetada = Dict[str, Any]

CodeEvalMetadaJson = str  # json encoded 'CodeEvalMetada'


ProblemTestsData = TypedDict(
    "ProblemTestsData",
    {"inputs": List[str], "outputs": List[str], "fn_name": Optional[str]},
)

ProblemDataJson = TypedDict(
    "ProblemDataJson",
    {  # json encoded 'ProblemTestsData' (dict with keys 'inputs','outputs','fn_name')
        "input_output": str
    },
)
