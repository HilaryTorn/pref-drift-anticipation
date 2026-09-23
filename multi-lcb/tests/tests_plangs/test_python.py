import os
from lcb_runner.evaluation.testing_plang import eval_plang_code
from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations
from dataclasses import asdict
import json


TIMEOUT = 15
PLANG = "python"


class TestPythonEvaluation:
    """Test class for Python evaluation functions"""

    def test_simple_stdin_program(self):
        """Test successful execution"""
        program = """
import sys

lines = []
for line in sys.stdin:
    lines.append(line.strip())

for line in lines:
    print(line)
"""

        input_data = ["0", "55555"]
        output_data = ["0", "55555"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, PLANG, TIMEOUT
        )

        metadata = asdict(metadata)

        assert result == [True, True]
        assert "execution_time" in metadata

    def test_numpy_openblas(self):
        """Test that numpy doesn't cause openblas errors when running code in multithreads."""
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

        samples_list = [{"input_output": json.dumps(tests)}] * 10
        generations_list = [[program]] * 10
        n_cores = min(os.cpu_count(), 10)
        n_restarts = 1

        results, metadata = evaluate_generations(
            samples_list,
            generations_list,
            num_process_evaluate=n_cores,
            plang=PLANG,
            n_restarts=n_restarts,
        )

        results_flat = []
        for _, r in results.items():
            results_flat.extend(*r)

        err = metadata[0][0].get("error_message", "")
        assert not "Error importing numpy" in err

        assert all([x > 0 for x in results_flat]), ValueError(
            "One of the test didn't pass"
        )
