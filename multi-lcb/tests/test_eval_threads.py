import os

from lcb_runner.evaluation.testing_plang import TestScore
from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations
import json


TIMEOUT = 15
N_GENERATIONS = 100
MIN_CORES = 2  # tests must be done with at least two cores
MAX_CORES = 10


class TestPoolEvaluation:
    """Test class for Python evaluation functions"""

    def test_multiprocessing(self):
        """Test that Pool of workers can execute the problem without errors."""

        plang = "python"
        program = """
import sys
import numpy as np

def main():
    # triger OpenBLAS by doing matrix multiplication
    arr1 = np.arange(1,10001)
    arr1 = arr1.reshape([100,100])

    arr2 = np.arange(10000,0,-1)
    arr2 = arr2.reshape([100,100])

    c = arr1.dot(arr2)
    print(c.mean())
            
main()
"""

        tests = {
            "inputs": ["", "", ""],
            "outputs": ["2492167525", "2492167525", "2492167525"],
            "fn_name": [None, None, None],
        }

        samples_list = [{"input_output": json.dumps(tests)}] * N_GENERATIONS
        generations_list = [[program]] * N_GENERATIONS

        n_cores = min(os.cpu_count(), MAX_CORES)
        n_cores = max(n_cores, MIN_CORES)

        assert n_cores >= 2

        results, _ = evaluate_generations(
            samples_list,
            generations_list,
            num_process_evaluate=n_cores,
            plang=plang,
            n_restarts=3,
        )

        results_flat = []
        for _, r in results.items():
            results_flat.extend(*r)

        assert all([x == TestScore.PASSED for x in results_flat]), ValueError(
            "One of the tests not passed"
        )
