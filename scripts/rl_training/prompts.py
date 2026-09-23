"""Prompt formats shared by RL training, DPO pair generation, and reward eval."""

from __future__ import annotations

CODING_PYTHON_IO_PROMPT = """You are solving a competitive programming problem.

Write a complete Python 3 program that reads from standard input and writes to standard output.
Return only the solution code. Do not include explanations.

Problem:
{problem}"""


def format_coding_prompt(problem: str, *, format_name: str = "python_io") -> str:
    problem = problem.strip()
    if format_name == "raw":
        return problem
    if format_name == "python_io":
        return CODING_PYTHON_IO_PROMPT.format(problem=problem)
    raise ValueError(f"unknown coding prompt format {format_name!r}")


def render_coding_prompt(tokenizer, problem: str, *, format_name: str = "python_io") -> str:
    """Render the exact prompt string the policy is trained/evaluated on.

    Chat-tuned models (all Qwen3.5 sizes) must see their chat template — feeding
    them raw text puts them in continuation mode and they ramble instead of
    answering. ``enable_thinking=False`` pins non-thinking mode across model
    sizes whose templates default differently; templates without that variable
    ignore it. Tokenizers without a chat template fall back to the raw format.

    Every consumer of coding prompts (GRPO rollouts, the per-checkpoint reward
    callback, DPO pair generation, prompt-length filtering) must go through this
    one function so the token stream is identical everywhere.
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
