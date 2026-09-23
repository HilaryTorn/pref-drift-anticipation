import pytest
import json
from prompt_enhancer.example_parser import ExampleParser

import os


@pytest.fixture(name="example_parser")
def create_example_parser():
    """Fixture to provide an ExampleParser instance."""
    parser = ExampleParser()
    yield parser


def test_extract_examples_basic(example_parser):
    """Test basic extraction of examples from problem text."""
    problem_text = """
Example 1:
Input: nums = [1,2,3], target = 6
Output: 2

Example 2:
Input: nums = [1,1,1], target = 3
Output: 3
"""
    examples = example_parser.extract_examples(problem_text)
    assert len(examples) == 2
    assert examples[0]["example_id"] == 1
    assert examples[0]["input"] == {"nums": [1, 2, 3], "target": 6}
    assert examples[0]["output"] == 2
    assert examples[1]["example_id"] == 2
    assert examples[1]["input"] == {"nums": [1, 1, 1], "target": 3}
    assert examples[1]["output"] == 3


def test_parse_input_line_multiple_types(example_parser):
    """Test parsing of an input line with multiple variable assignments and types."""
    line = 'target = "abc", words = ["a","b"], costs = [1,2]'
    parsed = example_parser.parse_input_line(line)
    assert parsed == {"target": "abc", "words": ["a", "b"], "costs": [1, 2]}


def test_parse_input_line_single_value(example_parser):
    """Test parsing of an input line with a single variable assignment."""
    line = "n = 5"
    parsed = example_parser.parse_input_line(line)
    assert parsed == {"n": 5}


def test_parse_input_line_complex_values(example_parser):
    """Test parsing of an input line with complex variable assignments."""
    line = 'grid = [[1,2],[3,4]], s = "hello", b = true, f = 3.14'
    parsed = example_parser.parse_input_line(line)
    assert parsed == {"grid": [[1, 2], [3, 4]], "s": "hello", "b": True, "f": 3.14}


def test_parse_value_various_types(example_parser):
    """Test parsing of various data types (string, int, float, bool, null, list, dict)."""
    assert example_parser.parse_value('"test_string" ') == "test_string"
    assert example_parser.parse_value("123") == 123
    assert example_parser.parse_value("-4.5") == -4.5
    assert example_parser.parse_value("true") is True
    assert example_parser.parse_value("false") is False
    assert example_parser.parse_value("null") is None
    assert example_parser.parse_value("[1,2,3]") == [1, 2, 3]
    assert example_parser.parse_value("[[1],[2]]") == [[1], [2]]
    assert example_parser.parse_value('{"key": "value"}') == {"key": "value"}
    assert example_parser.parse_value("[true,true,true]") == [True, True, True]
    assert example_parser.parse_value('[["abc","a a"],[2,4]]') == [
        ["abc", "a a"],
        [2, 4],
    ]


def test_parse_input_line_no_explicit_variable_names(example_parser):
    """
    Test parsing of complex input data without explicit variable names.
    """
    line = "[1,2],[3,4]"  # Example for _parse_list_of_values
    parsed = example_parser.parse_input_line(line)
    assert parsed == {"_param_0": [1, 2], "_param_1": [3, 4]}


def test_parse_input_line_single_complex_value_no_explicit_variable_names(
    example_parser,
):
    """
    Test parsing of a single complex input value without explicit variable names.
    """
    line = "[1,2,3]"
    parsed = example_parser.parse_input_line(line)
    assert parsed == {"_param_0": [1, 2, 3]}


def test_parse_input_line_mixed_complex_and_scalar_no_explicit_variable_names(
    example_parser,
):
    """
    Test parsing of mixed complex and scalar input values without explicit variable names.
    """
    line = '[1,2],"hello",42'
    parsed = example_parser.parse_input_line(line)
    assert parsed == {"_param_0": [1, 2], "_param_1": "hello", "_param_2": 42}


@pytest.mark.skip(reason="TODO: missing file with input data")
def test_extract_examples_from_tasks_json(example_parser):
    """Test extraction of examples from a JSON file containing task data."""
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "assets", "tasks.json")
    with open(file_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    problem_content = task_data["question_content"]
    examples = example_parser.extract_examples(problem_content)

    assert len(examples) == 2

    # Example 1
    assert examples[0]["example_id"] == 1
    expected_input_1 = {
        "target": "abcdef",
        "words": ["abdef", "abc", "d", "def", "ef"],
        "costs": [100, 1, 1, 10, 5],
    }
    assert examples[0]["input"] == expected_input_1
    assert examples[0]["output"] == 7

    # Example 2
    assert examples[1]["example_id"] == 2
    expected_input_2 = {
        "target": "aaaa",
        "words": ["z", "zz", "zzz"],
        "costs": [1, 10, 100],
    }
    assert examples[1]["input"] == expected_input_2
    assert examples[1]["output"] == -1


def test_extract_examples_no_examples(example_parser):
    """Test extraction of examples from text with no example blocks."""
    problem_text = """
No examples here.
"""
    examples = example_parser.extract_examples(problem_text)
    assert len(examples) == 0


def test_extract_examples_malformed_example(example_parser):
    """Test extraction of examples from text with malformed example blocks."""
    problem_text = """
Example 1:
Input: a = 1
# Missing output

Example 2:
Output: 2
# Missing input
"""
    examples = example_parser.extract_examples(problem_text)
    assert len(examples) == 0
