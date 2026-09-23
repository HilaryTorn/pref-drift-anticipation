#!/usr/bin/env python3
"""
Enhanced Prompt Enhancer with versioned formatting support.
"""

from typing import Dict, List, Any

try:
    from .example_parser import ExampleParser
    from .unified_converter import ConverterFactory
except ImportError:
    from example_parser import ExampleParser
    from unified_converter import ConverterFactory


class EnhancedPromptEnhancer:
    """Enhanced PromptEnhancer supporting multiple formatting versions.

    This class is responsible for enhancing an original prompt with example
    input/output formats, using a specified data converter version.
    """

    def __init__(
        self, format_version: str = "v1_no_dimensions", enable_examples: bool = True
    ):
        """Initializes the EnhancedPromptEnhancer.

        :param format_version: The version of the data converter to use (e.g., 'v1_no_dimensions').
        :type format_version: str
        :param enable_examples: If True, examples will be extracted and added to the prompt.
        :type enable_examples: bool
        """
        self.enable_examples = enable_examples
        self.format_version = format_version
        self.example_parser = ExampleParser()
        self.converter = ConverterFactory.create_converter(format_version)
        self.external_format_hints = {}

    def enhance_prompt(self, original_prompt: str, problem_content: str) -> str:
        """Enhances the original prompt by extracting and adding example input/output formats.

        If `enable_examples` is False, the original prompt is returned unchanged.
        Any errors during example extraction will result in the original prompt being returned.

        :param original_prompt: The initial prompt string to be enhanced.
        :type original_prompt: str
        :param problem_content: The full problem description text from which to extract examples.
        :type problem_content: str
        :return: The enhanced prompt string with examples, or the original prompt if examples are disabled or an error occurs.
        :rtype: str
        """
        if not self.enable_examples:
            return original_prompt

        try:
            examples = self._extract_examples(problem_content)
            if examples:
                return self._add_examples_to_prompt(original_prompt, examples)
        except Exception as e:
            print(f"⚠️ PromptEnhancer error: {e}")
        return original_prompt

    def _extract_examples(self, problem_content: str) -> List[Dict[str, Any]]:
        """Extracts and processes examples from the problem content.

        This method uses the `ExampleParser` to get raw examples and then converts them
        into stdin/stdout formats using the configured `converter`.

        :param problem_content: The full problem description text.
        :type problem_content: str
        :return: A list of dictionaries, each containing raw and processed example data.
        :rtype: List[Dict[str, Any]]
        """
        raw_examples = self.example_parser.extract_examples(problem_content)
        processed_examples = self.converter.convert_all_tests(raw_examples)
        result = []
        for example in zip(raw_examples, processed_examples):
            raw_example, processed_example = example
            result.append(
                {
                    "example_id": raw_example["example_id"],
                    "raw_input": raw_example["input"],
                    "raw_output": raw_example["output"],
                    "stdin_format": processed_example["stdin"],
                    "stdout_format": processed_example["stdout"],
                }
            )
        return result

    def _add_examples_to_prompt(
        self, original_prompt: str, examples: List[Dict[str, Any]]
    ) -> str:
        """Adds formatted examples to the original prompt string.

        This method appends a section with input/output format examples to the prompt,
        including general instructions and specifically formatted examples.

        :param original_prompt: The base prompt string.
        :type original_prompt: str
        :param examples: A list of processed example dictionaries.
        :type examples: List[Dict[str, Any]]
        :return: The prompt string with examples appended.
        :rtype: str
        """
        enhanced_prompt = original_prompt + "\n\n"
        enhanced_prompt += self._get_default_prompt_instructions() + "\n\n"
        for example in examples:
            enhanced_prompt += self._format_single_example(example) + "\n"
        return enhanced_prompt

    def _format_single_example(self, example: Dict[str, Any]) -> str:
        """Formats a single example block with stdin and stdout.

        :param example: A dictionary containing example data, including 'example_id', 'stdin_format', and 'stdout_format'.
        :type example: Dict[str, Any]
        :return: A formatted string representing a single example block.
        :rtype: str
        """
        enhanced_block = f"""Sample Input {example['example_id']}:

{example['stdin_format']}

Sample Output {example['example_id']}:

{example['stdout_format']}

"""
        return enhanced_block

    def _get_2d_array_instructions(self) -> str:
        """Returns 2D array specific input/output format instructions based on the format version.

        :return: A string containing 2D array format instructions.
        :rtype: str
        """
        if self.format_version in ["v2_rows_only", "v2"]:
            return "For 2D arrays, the first line indicates the number of rows, followed by newline-separated rows."
        elif self.format_version in ["v3_smart_matrix", "v3"]:
            return "For 2D matrices, the first line indicates the number of rows and columns, followed by newline-separated rows. For jagged 2D arrays, the first line indicates the number of rows, followed by newline-separated rows."
        else:  # v1_no_dimensions or any other default
            return "2D arrays are newline-separated rows."

    def _get_default_prompt_instructions(self) -> str:
        """Returns general input/output format instructions for the prompt.

        :return: A string containing general format instructions.
        :rtype: str
        """
        from lcb_runner.prompts.code_generation import PromptConstants

        base_instructions = (
            f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        )
        two_d_instructions = self._get_2d_array_instructions()

        if self.format_version == "v1_no_dimensions" or self.format_version == "v1":
            return base_instructions
        else:
            return f"{base_instructions} {two_d_instructions}"

    def set_format_version(self, version: str):
        """Sets the format version for the converter.

        This method updates the `format_version` attribute and re-initializes
        the `converter` with the new version.

        :param version: The new version identifier for the converter.
        :type version: str
        """
        self.format_version = version
        self.converter = ConverterFactory.create_converter(version)


def create_version_specific_enhancer(version_name: str) -> EnhancedPromptEnhancer:
    """Creates an EnhancedPromptEnhancer instance with a specific format version.

    :param version_name: The name of the version to use for the enhancer.
            It will be converted to lowercase.
    :type version_name: str
    :return: An instance of EnhancedPromptEnhancer configured with the specified version.
    :rtype: EnhancedPromptEnhancer
    """
    return EnhancedPromptEnhancer(
        format_version=version_name.lower(), enable_examples=True
    )
