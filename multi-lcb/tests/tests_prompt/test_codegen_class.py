"""Module tests how tests in problems convert into stdin/stdout format from json."""

import pytest
import json

from lcb_runner.benchmarks import (
    Test,
    TestType,
    CodeGenerationProblem,
)


# Test cases for Test.parse_raw_json
@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        ("3\n[1,2,3]", {"param_0": 3, "param_1": [1, 2, 3]}),
        ("123", {"param_0": 123}),
        ('""', {"param_0": ""}),
        ("[]", {"param_0": []}),
        ("", {"param_0": ""}),
        ("  \n\t ", {"param_0": "  \n\t "}),  # Test with only whitespace
        ('{"key": "value"}', {"param_0": {"key": "value"}}),
        ('[1, "hello", 3.0]', {"param_0": [1, "hello", 3.0]}),
        ("true", {"param_0": True}),
        ("null", {"param_0": None}),
        ('"hello"', {"param_0": "hello"}),
        ("1\n2\n3", {"param_0": 1, "param_1": 2, "param_2": 3}),
        ('{"a":1}\n{"b":2}', {"param_0": {"a": 1}, "param_1": {"b": 2}}),
        ("[1,2]\n[3,4]", {"param_0": [1, 2], "param_1": [3, 4]}),  # 2D array test 1
        (
            '["a","b"]\n["c","d"]',
            {"param_0": ["a", "b"], "param_1": ["c", "d"]},
        ),  # 2D array test 2
        (
            '""',
            {"param_0": ""},
        ),  # empty encoded json string
        (
            '"12  3"\n"45    6"',
            {"param_0": "12  3", "param_1": "45    6"},
        ),  # string with newlines
    ],
)
def test_parse_raw_json(input_data, expected_output):
    """
    Test cases for the parse_raw_json static method of the Test class.

    :param input_data: The raw string input to the method.
    :type input_data: str
    :param expected_output: The expected parsed output.
    :type expected_output: Dict[str, Any] | Any
    """
    assert Test.parse_raw_json(input_data) == expected_output


# Test cases for CodeGenerationProblem __post_init__ with functional tests
def test_code_generation_problem_functional_conversion():
    """
    Tests the __post_init__ method of CodeGenerationProblem to ensure
    functional test cases are correctly converted to STDIN using parse_raw_json.
    """
    problem_data = {
        "question_title": "Test Title",
        "question_content": "Test Content",
        "platform": "leetcode",
        "question_id": "1",
        "contest_id": "1",
        "contest_date": "2023-01-01T00:00:00",
        "starter_code": "def func(): pass",
        "difficulty": "easy",
        "public_test_cases": json.dumps(
            [{"input": "1\n2", "output": "3", "testtype": "functional"}]
        ),
        "private_test_cases": json.dumps(
            [{"input": "4\n5", "output": "9", "testtype": "functional"}]
        ),
        "metadata": json.dumps({}),
    }

    problem = CodeGenerationProblem(**problem_data)

    # Assert that test cases are converted to STDIN and inputs/outputs are parsed
    assert len(problem.public_test_cases) == 1
    assert problem.public_test_cases[0].testtype == TestType.STDIN
    assert problem.public_test_cases[0].input == "1\n2\n"
    assert problem.public_test_cases[0].output == "3\n"

    assert len(problem.private_test_cases) == 1
    assert problem.private_test_cases[0].testtype == TestType.STDIN
    assert problem.private_test_cases[0].input == "4\n5\n"
    assert problem.private_test_cases[0].output == "9\n"


# New test cases for CodeGenerationProblem with 2D array inputs/outputs
@pytest.mark.parametrize(
    "input_json, output_json, expected_input_stdin, expected_output_stdin",
    [
        ("[1,2]\n[3,4]", "[[5,6],[7,8]]", "1 2\n3 4\n", "5 6\n7 8\n"),
        ("[1]\n[2,3]", "[[4,5,6]]", "1\n2 3\n", "4 5 6\n"),
        ('[1,"a"]\n[2.0,"b"]', '[["c",3.0],["d",4]]', "1 a\n2.0 b\n", "c 3.0\nd 4\n"),
        # ("[]\n[]", "[]\n[]", "\n\n", "\n\n"),  # Empty inner lists
        ("[1]\n[2]", "[[3],[4]]", "1\n2\n", "3\n4\n"),
        (
            "[1,2,3]\n[4,5,6]\n[7,8,9]",
            "[[10,11,12]]",
            "1 2 3\n4 5 6\n7 8 9\n",
            "10 11 12\n",
        ),
        (
            "[true,false]\n[true,true]",
            "[[false,true]]",
            "true false\ntrue true\n",
            "false true\n",
        ),
        ("[1,null]\n[null,4]", "[[null,2]]", "1 null\nnull 4\n", "null 2\n"),
        (
            '["hello","world"]\n["foo","bar"]',
            '[["new","test"]]',
            "hello world\nfoo bar\n",
            "new test\n",
        ),
        ('""', '""', "", ""),  # expty str
        # ("[1,2]\n3", "[[4,5],6]", "1 2\n3\n", "4 5\n6\n"), # Mixed-depth array test: Excluded for now
    ],
)
def test_code_generation_problem_2d_functional_conversion(
    input_json: str,
    output_json: str,
    expected_input_stdin: str,
    expected_output_stdin: str,
):
    """
    Tests the CodeGenerationProblem initialization and functional test conversion
    for various two-dimensional array inputs and outputs.

    :param input_json: Raw JSON string for the test input.
    :type input_json: str
    :param output_json: Raw JSON string for the test output.
    :type output_json: str
    :param expected_input_stdin: Expected STDIN format for the input.
    :type expected_input_stdin: str
    :param expected_output_stdin: Expected STDIN format for the output.
    :type expected_output_stdin: str
    """
    problem_data = {
        "question_title": "2D Array Test",
        "question_content": "Content for 2D array test.",
        "platform": "leetcode",
        "question_id": "2",
        "contest_id": "2",
        "contest_date": "2023-01-02T00:00:00",
        "starter_code": "def solve(arr): return arr",
        "difficulty": "medium",
        "public_test_cases": json.dumps(
            [{"input": input_json, "output": output_json, "testtype": "functional"}]
        ),
        "private_test_cases": json.dumps([]),
        "metadata": json.dumps({}),
    }

    problem = CodeGenerationProblem(**problem_data, format_version="v1")

    assert len(problem.public_test_cases) == 1
    assert problem.public_test_cases[0].testtype == TestType.STDIN
    assert problem.public_test_cases[0].input == expected_input_stdin
    assert problem.public_test_cases[0].output == expected_output_stdin


# New test cases for CodeGenerationProblem with 2D array inputs/outputs for V2 converter
@pytest.mark.parametrize(
    "input_json, output_json, expected_input_stdin, expected_output_stdin",
    [
        ("[1,2]\n[3,4]", "[[5,6],[7,8]]", "1 2\n3 4\n", "2\n5 6\n7 8\n"),
        ("[1]\n[2,3]", "[[4,5,6]]", "1\n2 3\n", "1\n4 5 6\n"),
        (
            '[1,"a"]\n[2.0,"b"]',
            '[["c",3.0],["d",4]]',
            "1 a\n2.0 b\n",
            "2\nc 3.0\nd 4\n",
        ),
        # Empty inner lists: V2 converter behavior for empty lists might be same as V1, check if this is still failing
        # ("[]\n[]","[]\n[]","\n\n","\n\n"),
        ("[1]\n[2]", "[[3],[4]]", "1\n2\n", "2\n3\n4\n"),
        (
            "[[1,2,3],[4,5,6],[7,8,9]]",
            "[[10,11,12]]",
            "3\n1 2 3\n4 5 6\n7 8 9\n",
            "1\n10 11 12\n",
        ),
        (
            "[true,false]\n[true,true]",
            "[[false,true]]",
            "true false\ntrue true\n",
            "1\nfalse true\n",
        ),
        ("[1,null]\n[null,4]", "[[null,2]]", "1 null\nnull 4\n", "1\nnull 2\n"),
        (
            '["hello","world"]\n["foo","bar"]',
            '[["new","test"]]',
            "hello world\nfoo bar\n",
            "1\nnew test\n",
        ),
        # empty 2d_array
        # ("[[]]", "[[]]", "\n\n", "0\n"),
        # ("[1,2]\n3", "[[4,5],6]", "2\n1 2\n3\n", "2\n4 5\n6\n"), # Mixed-depth array test: Excluded for now
    ],
)
def test_code_generation_problem_2d_functional_conversion_v2(
    input_json: str,
    output_json: str,
    expected_input_stdin: str,
    expected_output_stdin: str,
):
    """
    Tests the CodeGenerationProblem initialization and functional test conversion
    for various two-dimensional array inputs and outputs using V2 converter.

    :param input_json: Raw JSON string for the test input.
    :type input_json: str
    :param output_json: Raw JSON string for the test output.
    :type output_json: str
    :param expected_input_stdin: Expected STDIN format for the input.
    :type expected_input_stdin: str
    :param expected_output_stdin: Expected STDIN format for the output.
    :type expected_output_stdin: str
    """

    problem_data = {
        "question_title": "2D Array V2 Test",
        "question_content": "Content for 2D array V2 test.",
        "platform": "leetcode",
        "question_id": "3",
        "contest_id": "3",
        "contest_date": "2023-01-03T00:00:00",
        "starter_code": "def solve_v2(arr): return arr",
        "difficulty": "medium",
        "public_test_cases": json.dumps(
            [{"input": input_json, "output": output_json, "testtype": "functional"}]
        ),
        "private_test_cases": json.dumps([]),
        "metadata": json.dumps({}),
    }

    problem = CodeGenerationProblem(**problem_data, format_version="v2")

    assert len(problem.public_test_cases) == 1
    assert problem.public_test_cases[0].testtype == TestType.STDIN

    # print(problem_data)
    # print(problem.public_test_cases[0].input)
    # print(problem.public_test_cases[0].output)

    assert problem.public_test_cases[0].input == expected_input_stdin
    assert problem.public_test_cases[0].output == expected_output_stdin
