import pytest
from lcb_runner.evaluation.testing_plang import match_tests_groud_truth, TestScore
from typing import List


class TestEvalMatchOutputs:
    """Test cases for Test.parse_raw_json"""

    @pytest.mark.parametrize(
        "input_data, code_outputs, tests_answers, expected_result",
        [
            (["abcdef"], ["10.01\n20.02"], ["10.01\n20.02"], True),
            (["abcdef"], ["20.02\n10.01"], ["10.01\n20.02"], False),
            (["abcdef"], ["10.01\n20.02"], ["10.01"], False),
            (["abcdef"], ["10.01"], ["10.01\n20.02"], False),
            (["abcdef"], ["10.01 20.02 30.03"], ["10.01\n20.02\n30.03"], False),
            (["abcdef"], ["10.01, 20.02, 30.03"], ["10.01 20.02 30.03"], False),
            (["abcdef"], ["10.01\n20.02\n30.03"], ["10.01 20.02 30.03"], False),
            (["abcdef"], [" 10.01 \n"], ["10.01"], True),
            (["abcdef"], ["10.0001"], ["10.0002"], False),
            (["abcdef"], ["10.00001"], ["10.00002"], True),
            (["abcdef"], ["10.00001"], ["10"], True),
            (["abcdef"], ["10"], ["10.00001"], True),
            (["abcdef"], ["10.01"], ["10.001"], False),
            (["abcdef"], ["true"], ["True"], True),
            (["abcdef"], ["True"], ["True"], True),
            (["abcdef"], ["True"], ["false"], False),
            (["abcdef"], ["False"], ["false"], True),
            (["abcdef"], ["false"], ["False"], True),
            (
                ["abcdef"],
                ["1000000.1"],
                ["1000000.2"],
                False,
            ),  # True for np.isclose(a,b)
            (["abcdef"], ["1000000.00001"], ["1000000.000002"], True),
            (["abcdef"], ["ABCDEF"], ["aBCDEF"], False),
            (["abcdef"], ["ABCDEF"], ["ABCDEF"], True),
            (
                ["abcdef"] * 10,
                ["1", "2\n1\n0", "3", "4", "5", "6", "7", "8", "9", "10"],
                ["1", "2\n1\n0", "3", "4", "5", "6", "7", "8", "9", "10"],
                True,
            ),  # all correct
            (
                ["abcdef"] * 5,
                ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                True,
            ),  # not enough inputs
            (
                ["abcdef"] * 10,
                ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
                False,
            ),  # not enough outputs
            (
                ["abcdef"] * 10,
                ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
                ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                False,
            ),  # not enough code outputs
            (
                ["abcdef"] * 10,
                ["1", "2", "3", "4", "0", "6", "7", "8", "9", "10"],
                ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                False,
            ),  # one of the tests failed
        ],
    )
    def test_code_outputs(
        self,
        input_data: List[str],
        code_outputs: List[str],
        tests_answers: List[str],
        expected_result: bool,
    ):
        results, _ = match_tests_groud_truth(code_outputs, input_data, tests_answers)
        tests_passed = all([x == TestScore.PASSED for x in results])
        assert tests_passed == expected_result
