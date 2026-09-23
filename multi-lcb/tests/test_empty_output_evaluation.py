"""
Test suite for empty string/array evaluation scenarios at the CodeGenerationProblem level.

This module tests the critical requirement that the evaluation system should
accept various representations of "empty" as equivalent when the target expects
empty output. This tests the full pipeline from dataset format to final evaluation.

Key Requirements:
- Target '\"\"' should accept empty string output from code

These tests ensure that models are not incorrectly penalized for outputting
semantically correct but differently formatted empty results.
"""

import os
import json
import pytest
from lcb_runner.evaluation.testing_plang import TestScore
from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations
from lcb_runner.benchmarks import CodeGenerationProblem


class TestEmptyOutputEvaluation:
    """Test class for empty output evaluation scenarios using CodeGenerationProblem."""

    def test_empty_string_functional_to_stdin_conversion(self):
        """
        Test that CodeGenerationProblem correctly converts functional tests with empty strings.

        This tests the conversion from dataset format (functional tests) to STDIN format
        and ensures the evaluation pipeline handles empty strings correctly.
        """
        # Create a problem with functional test that has empty string output
        problem_data = {
            "question_title": "Empty String Test",
            "question_content": "Return empty string when condition is met.",
            "platform": "leetcode",
            "question_id": "empty_test_1",
            "contest_id": "test_contest",
            "contest_date": "2023-01-01T00:00:00",
            "starter_code": "def solution(): pass",
            "difficulty": "easy",
            "public_test_cases": json.dumps(
                [
                    {
                        "input": "1",  # Simple valid JSON input
                        "output": '""\n',  # Dataset contains quoted empty string
                        "testtype": "functional",
                    }
                ]
            ),
            "private_test_cases": json.dumps([]),
            "metadata": json.dumps({}),
        }

        # Create CodeGenerationProblem - this will convert functional to STDIN
        problem = CodeGenerationProblem(**problem_data)

        # Check that conversion happened correctly
        assert len(problem.public_test_cases) == 1
        test_case = problem.public_test_cases[0]

        print(f"Original functional output: '\"\"'")
        print(f"Converted STDIN output: {repr(test_case.output)}")

        # Now test evaluation with code that outputs empty string
        program_that_outputs_empty = """
def main():
    print('')  # Model outputs actual empty string

main()
"""

        # Prepare for evaluation using the converted test case
        tests = {
            "inputs": [test_case.input],
            "outputs": [test_case.output],  # This is the converted STDIN format
            "fn_name": [None],
        }

        samples_list = [{"input_output": json.dumps(tests)}]
        generations_list = [[program_that_outputs_empty]]

        results, debug_info = evaluate_generations(
            samples_list,
            generations_list,
            num_process_evaluate=1,
            plang="python",
            n_restarts=1,
        )

        results_flat = []
        for _, r in results.items():
            results_flat.extend(*r)

        # This SHOULD pass but currently fails - this is the bug we need to fix
        assert all(
            [x == TestScore.PASSED for x in results_flat]
        ), f"Empty string output should be accepted for converted target: {results_flat}"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
