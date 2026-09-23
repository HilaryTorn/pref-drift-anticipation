from lcb_runner.lm_styles.models_store import LMStyle
from lcb_runner.runner.parser import ConfigLCB
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem, Platform
from copy import deepcopy
from lcb_runner.utils import PLang
from prompt_enhancer import create_version_specific_enhancer
from typing import Optional, List, TypedDict, Literal
import os

PROMPT_ENHANCER_AVAILABLE = True


class ChatMessage(TypedDict):
    role: Literal["assistant", "system", "user"]
    content: str


ChatMessages = List[ChatMessage]


class PromptConstants:
    # pylint: disable=C0301
    SYSTEM_MESSAGE_GENERIC = "You are an expert <param:Plang> programmer. You will be given a question (problem specification) and will generate a correct <param:Plang> program that matches the specification and passes all tests."

    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the <param:plang> program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n\n"


def get_enhanced_question_template_answer(
    question: CodeGenerationProblem,
    enhance_prompts: bool = False,
    format_version: str = "v1",
):

    base_prompt = f"### Question:\n{question.question_content}\n\n"
    if (
        enhance_prompts
        and PROMPT_ENHANCER_AVAILABLE
        and question.platform == Platform.LEETCODE
    ):
        enhancer = create_version_specific_enhancer(format_version)
        enhanced_prompt = enhancer.enhance_prompt(
            base_prompt, question.question_content
        )
        enhanced_prompt += "```<param:plang>\n<param:comment> YOUR CODE HERE\n```\n\n"
        enhanced_prompt += "### Answer: (use the provided format with backticks)\n\n"
        final_prompt = enhanced_prompt
    else:
        final_prompt = base_prompt

    # If not enhanced, or enhancement was skipped (e.g., due to platform),
    # append the default formatting block.
    if not (
        enhance_prompts
        and PROMPT_ENHANCER_AVAILABLE
        and question.platform == Platform.LEETCODE
    ):
        format_block = ""
        if question.starter_code:
            format_block += (
                f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
                f"```<param:plang>\n{question.starter_code}\n```\n\n"
            )
        else:
            format_block += (
                f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
            )
            format_block += "```<param:plang>\n<param:comment> YOUR CODE HERE\n```\n\n"
        format_block += "### Answer: (use the provided format with backticks)\n\n"
        final_prompt += format_block

    return final_prompt


def _format_prompt_generation_enhanced(
    question: CodeGenerationProblem,
    lang_model_style: LMStyle,
    enhance_prompts: bool = False,  # New parameter
    format_version: str = "v1",  # New parameter
    tokenizer_path: Optional[str] = None,
) -> ChatMessages:

    if tokenizer_path:
        # may be used later for applying chat template
        assert os.path.exists(tokenizer_path)

    if lang_model_style in (
        LMStyle.VLLMBase,
        LMStyle.VLLMAsync,
        LMStyle.SGLangAsync,
        LMStyle.OpenAIChat,
        LMStyle.Claude,
        LMStyle.Grok,
        LMStyle.Gemini,
        LMStyle.GeminiThinking,
        LMStyle.DeepSeek,
        LMStyle.Mistral,
        LMStyle.Cohere,
    ):
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_enhanced_question_template_answer(
                    question, enhance_prompts, format_version
                ),
            },
        ]
        return chat_messages

    raise NotImplementedError(f"LanguageModelStyle {lang_model_style} not implemented")


def get_format_prompt_generation(args: ConfigLCB):

    tokenizer_path = args.tokenizer_path
    enhance_prompts = args.enhance_prompts
    format_version = args.format_version

    def format_prompt_generation(
        question: CodeGenerationProblem, lang_model_style: LMStyle, plang: str
    ) -> ChatMessages:
        """
        plang - "c++", "c#", "Python", ...
        """

        plang = plang.lower()
        params_map = {
            "<param:Plang>": PLang.get_display_lang_name(plang),
            "<param:plang>": PLang.get_markdown_lang_name(plang),
            "<param:PLANG>": plang.upper(),
            "<param:comment>": PLang.get_single_line_comment(plang),
        }
        question = deepcopy(question)

        # starter_code was provided only in Python, thus it is irrelevant.
        question.starter_code = None

        # promt is either a string or list of messages
        prompt = _format_prompt_generation_enhanced(
            question=question,
            lang_model_style=lang_model_style,
            tokenizer_path=tokenizer_path,
            enhance_prompts=enhance_prompts,
            format_version=format_version,
        )

        if isinstance(prompt, str):
            for k, v in params_map.items():
                prompt = prompt.replace(k, v)
        elif isinstance(prompt, tuple):
            system_message, chat_messages = prompt
            for k, v in params_map.items():
                system_message = system_message.replace(k, v)
            for msg in chat_messages:
                for k, v in params_map.items():
                    msg["content"] = msg["content"].replace(k, v)
            return (system_message, chat_messages)
        elif isinstance(prompt, list):
            for msg in prompt:
                for k, v in params_map.items():
                    msg["content"] = msg["content"].replace(k, v)
        else:
            raise NotImplementedError()

        return prompt

    return format_prompt_generation
