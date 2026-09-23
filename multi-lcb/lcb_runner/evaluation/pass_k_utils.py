import numpy as np
from typing import Dict, List
from numpy.typing import NDArray
from itertools import repeat
from lcb_runner.evaluation.metrics_typing import (
    PassAtKStats,
    CodeEvalScores,
    ProblemTestsPassed,
)


def estimate_pass_at_k(
    num_samples: int | List[int], num_correct: int | List[int], k: int
) -> NDArray[np.float64]:
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """
        n - cnt total samples
        c - cnt correct samples
        k - if we take top_k

        comb(n, k) = n! / (k! * (n-k)!)

        Calculates 1 - comb(n - c, k) / comb(n, k).
        """

        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)],
        dtype=float,
    )


def compute_metrics_from_results(
    results: Dict[str, List], k_list=(1, 5)
) -> PassAtKStats:
    """
    Example outputs:
        {'pass@1': np.float64(0.1),
        'detail': {
                    'pass@1': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 0.0}
        }}

    'detail' gives information about performance for each task_id.
    """

    total: List[int] = []
    correct: List[int] = []
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist()
        for k in ks
        if (total >= k).all()
    }
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }
    detail_metrics = {k: dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k


def extract_instance_results(
    results: Dict[int, List[CodeEvalScores]],
) -> List[List[ProblemTestsPassed]]:
    """For each problem checks if code in generation passed all tests.

    Args:
        results (Dict[int, List[CodeEvalScores]]): maps 'task_id' to code eval results
    Returns:
        List[List[ProblemSolved]]: grades for each problem
    """
    instance_wise_grades = {}
    for task_id, res in results.items():
        instance_wise_grades[task_id] = []
        for generation in res:
            instance_wise_grades[task_id].append(all([g > 0 for g in generation]))

    instance_wise_grades = [
        v for _, v in sorted(instance_wise_grades.items(), key=lambda item: item[0])
    ]
    return instance_wise_grades
