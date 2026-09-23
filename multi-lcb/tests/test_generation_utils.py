"""Module tests how code is extracted from model generations and reasoning content truncation."""

from lcb_runner.utils import extract_code, truncate_reasoning
import pytest


@pytest.mark.parametrize(
    "content, extracted_code",
    [
        ("<think>abcddeefggh</think>", ""),
        (None, ""),
        ("abcddeefggh</think>", ""),
        ("</think>", ""),
        ("<think>abcddeefggh</think>```code```", "code"),
        ("<think>abcddeefggh</think>```\n\ncode\n\ncode\n```", "code\n\ncode"),
        ("<think>abcddeefggh</think>```python\na=4```", "a=4"),
        ("<think>abcddeefggh</think>```python\n\n\n```", ""),
        ("<think>```rust\ncode_block1```</think>```rust\na=14```", "a=14"),
        ("<think>qwertg</think>```# YOUR CODE HERE\nv=55```", "v=55"),
        ("<think>qwertg</think>```## YOUR CODE HERE\nc=d```", "c=d"),
        ("<think>qwertg</think>```// YOUR CODE HERE\nww='1234'```", "ww='1234'"),
        ("<think>qwertg</think>```//// YOUR CODE HERE\nif k=4:```", "if k=4:"),
        (
            "```python\n//// YOUR CODE HERE\nabc=10``` and second code block ```rust//// YOUR CODE HERE\ncc=10```",
            "abc=10",
        ),
        ("```   python\ncc=11```", "cc=11"),
        ("```   python  \ncc=44\n```", "cc=44"),
        ("```python\ncc=44\n## YOUR CODE HERE```", "cc=44"),
    ],
)
def test_code_extract(content: str | None, extracted_code: str):
    out = extract_code(content)
    assert out.strip() == extracted_code.strip()


@pytest.mark.parametrize(
    "content, expected_output, l",
    [
        ("<think>abcddeefggh</think>", "<think>abcddeefggh</think>", 100),
        (None, None, 100),
        ("", "", 100),
        ("abcddeefggh</think>", "abcddeefggh</think>", 100),
        ("</think>", "</think>", 100),
        (
            "<think>abcddeefggh</think>```code```",
            "<think>abcddeefggh</think>```code```",
            100,
        ),
        (
            "<think>" + "abc" * 300 + "</think>",
            "<think>... [total 900 symbols]</think>",
            20,
        ),
        ("abc" * 300 + "</think>", "... [total 900 symbols]</think>", 20),
    ],
)
def test_response_truncate(content: str | None, expected_output: str | None, l: int):
    out = truncate_reasoning(content, l=l)

    if out:
        out = out.strip()

    if expected_output:
        expected_output = expected_output.strip()

    assert out == expected_output


@pytest.mark.parametrize(
    "content, l",
    [
        ("<think>" + "abc" * 300 + "</think>", 45),
        ("<think>" + "abc" * 300 + "</think>", 45),
        ("<think>" + "abc" * 300 + "</think>", 50),
        ("<think>" + "abc" * 300 + "</think>", 100),
        ("<think>" + "abc" * 300 + "</think>", 200),
        ("<think>" + "abc" * 300 + "</think>", 300),
        ("<think>" + "abc" * 300 + "</think>", 400),
        ("abc" * 300 + "</think>", 400),
        ("abc" * 300 + "</think>", 234),
        ("abc" * 300 + "</think>", 45),
        ("<think>" + "abc" * 300, 45),
        ("<think>" + "abc" * 300, 200),
    ],
)
def test_response_truncate_length(content: str | None, l: int):

    out = truncate_reasoning(content, l=l)

    assert "[total 900 symbols]" in out
    assert len(out) == l, print(out)
