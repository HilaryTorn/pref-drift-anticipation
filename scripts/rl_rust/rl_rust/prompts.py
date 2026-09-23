"""Coding prompts, parameterized by target language.

Mirrors ``rl_training/prompts.py`` exactly in structure and chat-template
handling. The Python format string is reproduced verbatim so a run configured
for ``python_io`` here is token-identical to one from the existing pipeline --
that is what makes a Python-vs-Rust RL comparison legitimate.

The Rust prompt is the Python one with the language swapped and one addition:
an explicit note that the program must be a complete ``fn main`` reading stdin.
Without it the model tends to emit a bare function, which compiles to nothing
runnable and scores 0.0 for a reason that has nothing to do with its reasoning.
"""

from __future__ import annotations

CODING_PYTHON_IO_PROMPT = """You are solving a competitive programming problem.

Write a complete Python 3 program that reads from standard input and writes to standard output.
Return only the solution code. Do not include explanations.

Problem:
{problem}"""

CODING_RUST_IO_PROMPT = """You are solving a competitive programming problem.

Write a complete Rust program that reads from standard input and writes to standard output.
The program must compile as a standalone file with a `fn main()` entry point and use only the standard library.
Return only the solution code. Do not include explanations.

Problem:
{problem}"""

_FORMATS = {
    "python_io": CODING_PYTHON_IO_PROMPT,
    "rust_io": CODING_RUST_IO_PROMPT,
}

#: Which verifier language each prompt format targets. Keeping the mapping here
#: means a run cannot prompt for Rust and grade as Python.
FORMAT_LANGUAGE = {
    "python_io": "python",
    "rust_io": "rust",
}


def format_coding_prompt(problem: str, *, format_name: str = "rust_io") -> str:
    problem = problem.strip()
    if format_name == "raw":
        return problem
    try:
        return _FORMATS[format_name].format(problem=problem)
    except KeyError:
        raise ValueError(f"unknown coding prompt format {format_name!r}") from None


def render_coding_prompt(tokenizer, problem: str, *, format_name: str = "rust_io") -> str:
    """Render the exact prompt string the policy is trained/evaluated on.

    Chat-tuned models (all Qwen3.5 sizes) must see their chat template -- feeding
    them raw text puts them in continuation mode and they ramble instead of
    answering. ``enable_thinking=False`` pins non-thinking mode across model
    sizes whose templates default differently; templates without that variable
    ignore it. Tokenizers without a chat template fall back to the raw format.

    Every consumer of coding prompts (GRPO rollouts, the per-checkpoint reward
    callback, prompt-length filtering) must go through this one function so the
    token stream is identical everywhere.
    """
    user_text = format_coding_prompt(problem, format_name=format_name)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return user_text
