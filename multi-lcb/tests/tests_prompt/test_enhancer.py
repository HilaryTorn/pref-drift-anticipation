#!/usr/bin/env python3
"""
Tests for EnhancedPromptEnhancer with unified architecture.
"""

import unittest

from prompt_enhancer.unified_converter import (
    V1NoDimensionsConverter,
    V3SmartMatrixConverter,
    UnifiedDataAnalyzer,
    DataType,
)
from prompt_enhancer.enhancer import EnhancedPromptEnhancer
from prompt_enhancer.example_parser import ExampleParser
from lcb_runner.prompts.code_generation import PromptConstants


class TestEnhancedPromptEnhancer(unittest.TestCase):
    """Tests for the EnhancedPromptEnhancer class."""

    def setUp(self):
        self.problem_content_simple_array = """Your task is to sum a list of numbers.\nExample 1:\nInput: colors = [1,2,3]\nOutput: 6"""
        self.problem_content_matrix_v3 = """You are given a 2D array, return the sum of its elements.\nExample 1:\nInput: grid = [[1,2],[3,4]]\nOutput: [[5,6],[7,8]]"""

    def test_init_default(self):
        """Tests default initialization and specific version initialization."""
        # Test default initialization
        enhancer = EnhancedPromptEnhancer()
        self.assertTrue(enhancer.enable_examples)
        self.assertEqual(enhancer.format_version, "v1_no_dimensions")
        self.assertIsInstance(enhancer.example_parser, ExampleParser)
        self.assertIsInstance(enhancer.converter, V1NoDimensionsConverter)

        # Test with specific format version
        enhancer_v3 = EnhancedPromptEnhancer(format_version="v3_smart_matrix")
        self.assertEqual(enhancer_v3.format_version, "v3_smart_matrix")
        self.assertIsInstance(enhancer_v3.converter, V3SmartMatrixConverter)

    def test_extract_examples_simple_array(self):
        """Tests example extraction and prompt enhancement for a simple array problem (V1)."""
        enhancer = EnhancedPromptEnhancer(format_version="v1_no_dimensions")
        problem_content = self.problem_content_simple_array
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)

        # Assertions for the updated prompt structure
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "Sample Input 1:\n\n1 2 3\n\nSample Output 1:\n\n6\n", enhanced_prompt
        )

    def test_extract_examples_matrix_v3(self):
        """Tests example extraction and prompt enhancement for a matrix problem (V3)."""
        enhancer = EnhancedPromptEnhancer(format_version="v3_smart_matrix")
        problem_content = self.problem_content_matrix_v3
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)

        # Assertions for the updated prompt structure
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "For 2D matrices, the first line indicates the number of rows and columns, followed by newline-separated rows. For jagged 2D arrays, the first line indicates the number of rows, followed by newline-separated rows.",
            enhanced_prompt,
        )
        self.assertIn(
            "Sample Input 1:\n\n2 2\n1 2\n3 4\n\nSample Output 1:\n\n2 2\n5 6\n7 8\n",
            enhanced_prompt,
        )

    def test_set_format_version(self):
        """Tests setting a new format version and converter instance."""
        enhancer = EnhancedPromptEnhancer(format_version="v1_no_dimensions")
        self.assertEqual(enhancer.format_version, "v1_no_dimensions")

        enhancer.set_format_version("v3_smart_matrix")
        self.assertEqual(enhancer.format_version, "v3_smart_matrix")
        self.assertIsInstance(enhancer.converter, V3SmartMatrixConverter)


class TestRealWorldProblems(unittest.TestCase):
    """Tests with real-world problem examples for various formats."""

    def test_string_array_problem_seniors(self):
        """Tests a problem with string and array input (V1)."""
        problem_content = """Your task is to calculate the final score...\nExample 1:\nInput: seniors = [[4,1],[3,2]]\nOutput: 10"""
        enhancer = EnhancedPromptEnhancer(format_version="v1_no_dimensions")
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "Sample Input 1:\n\n4 1\n3 2\n\nSample Output 1:\n\n10\n", enhanced_prompt
        )

    def test_2d_array_problem_variables(self):
        """Tests a 2D array problem with multiple input variables (V3)."""
        problem_content = (
            "Example 1:\nInput: n = 4, edges = [[0,1],[1,2],[2,3]], source = 0, destination = 3\nOutput: true\n\n"
            + "Example 2:\nInput: n = 6, edges = [[0,1],[0,2],[3,5],[5,4],[4,3]], source = 0, destination = 5\nOutput: false"
        )
        enhancer = EnhancedPromptEnhancer(format_version="v3_smart_matrix")
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "For 2D matrices, the first line indicates the number of rows and columns, followed by newline-separated rows. For jagged 2D arrays, the first line indicates the number of rows, followed by newline-separated rows.",
            enhanced_prompt,
        )
        self.assertIn(
            "Sample Input 1:\n\n4\n3 2\n0 1\n1 2\n2 3\n0\n3\n\nSample Output 1:\n\ntrue\n\n\n",
            enhanced_prompt,
        )
        self.assertIn(
            "Sample Input 2:\n\n6\n5 2\n0 1\n0 2\n3 5\n5 4\n4 3\n0\n5\n\nSample Output 2:\n\nfalse\n\n\n",
            enhanced_prompt,
        )

    def test_tree_boolean_problem(self):
        """Tests a tree-like boolean problem (V1)."""
        problem_content = """You are given the roots of two binary trees...\nExample 1:\nInput: root1 = [1,2,3], root2 = [1,2,3]\nOutput: true"""
        enhancer = EnhancedPromptEnhancer(format_version="v1_no_dimensions")
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "Sample Input 1:\n\n1 2 3\n1 2 3\n\nSample Output 1:\n\ntrue\n",
            enhanced_prompt,
        )

    def test_parameter_order_preservation(self):
        """Tests that parameter order is preserved in the enhanced prompt (V1)."""
        problem_content = """You are given an integer array nums and an integer k...\nExample 1:\nInput: nums = [1,2,3], k = 2\nOutput: 5"""
        enhancer = EnhancedPromptEnhancer(format_version="v1_no_dimensions")
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "Sample Input 1:\n\n1 2 3\n2\n\nSample Output 1:\n\n5\n", enhanced_prompt
        )

    def test_complex_data_types_v2(self):
        """Tests complex data types and their conversion with V2 (rows only)."""
        problem_content = """Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.\nExample 1:\nInput: val = ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"], a = [[2], [1,1],[2,2],[1],[3,3],[2],[4,4],[1],[3],[4]]\nOutput: [null, null, null, 1, null, -1, null, -1, 3, 4]"""
        enhancer = EnhancedPromptEnhancer(format_version="v2_rows_only")
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "For 2D arrays, the first line indicates the number of rows, followed by newline-separated rows.",
            enhanced_prompt,
        )
        self.assertIn(
            "Sample Input 1:\n\nLRUCache put put get put get put get get get\n10\n2\n1 1\n2 2\n1\n3 3\n2\n4 4\n1\n3\n4\n\nSample Output 1:\n\nnull null null 1 null -1 null -1 3 4\n\n",
            enhanced_prompt,
        )

    def test_mixed_numeric_types(self):
        """Tests mixed integer and float numeric types (V1)."""
        problem_content = """You are given an array of integers `nums`...\nExample 1:\nInput: nums = [1.0, 2, 3.5]\nOutput: 6.5"""
        enhancer = EnhancedPromptEnhancer(format_version="v1_no_dimensions")
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "Sample Input 1:\n\n1.0 2 3.5\n\nSample Output 1:\n\n6.5\n", enhanced_prompt
        )

    def test_edge_case_empty_arrays(self):
        """Tests edge case with empty 2D arrays (V3)."""
        problem_content = """Given a 2D integer array `grid`...\nExample 1:\nInput: grid = [[]]\nOutput: [[]]"""
        enhancer = EnhancedPromptEnhancer(format_version="v3_smart_matrix")
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "For 2D matrices, the first line indicates the number of rows and columns, followed by newline-separated rows. For jagged 2D arrays, the first line indicates the number of rows, followed by newline-separated rows.",
            enhanced_prompt,
        )
        self.assertIn(
            "Sample Input 1:\n\n1 0\n\n\nSample Output 1:\n\n1 0\n\n\n", enhanced_prompt
        )  # 1 row, 0 cols

    def test_boolean_and_null_conversion(self):
        """Tests boolean and null value conversion (V1)."""
        problem_content = (
            """Example 1:\nInput: is_active = true, user_id = null\nOutput: true"""
        )
        enhancer = EnhancedPromptEnhancer(format_version="v1_no_dimensions")
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "Sample Input 1:\n\ntrue\nnull\n\nSample Output 1:\n\ntrue\n",
            enhanced_prompt,
        )


class TestVersionSpecificBehavior(unittest.TestCase):
    """Tests for version-specific formatting behaviors."""

    def test_v1_no_dimensions_format(self):
        """Tests V1 converter for no dimensions format."""
        enhancer = EnhancedPromptEnhancer(format_version="v1_no_dimensions")
        problem_content = "Example 1:\nInput: arr = [1,2,3]\nOutput: 1 2 3"
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "Sample Input 1:\n\n1 2 3\n\nSample Output 1:\n\n1 2 3\n", enhanced_prompt
        )

    def test_v2_rows_only_format(self):
        """Tests V2 converter for rows-only format."""
        enhancer = EnhancedPromptEnhancer(format_version="v2_rows_only")
        problem_content = (
            "Example 1:\nInput: matrix = [[1,2],[3,4]]\nOutput: 2\n1 2\n3 4"
        )
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "For 2D arrays, the first line indicates the number of rows, followed by newline-separated rows.",
            enhanced_prompt,
        )
        self.assertIn(
            "Sample Input 1:\n\n2\n1 2\n3 4\n\nSample Output 1:\n\n2\n\n",
            enhanced_prompt,
        )  # Corrected expected output

    def test_v3_smart_matrix_format(self):
        """Tests V3 converter for smart matrix format."""
        enhancer = EnhancedPromptEnhancer(format_version="v3_smart_matrix")
        problem_content = (
            "Example 1:\nInput: matrix = [[1,2],[3,4]]\nOutput: 2 2\n1 2\n3 4"
        )
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "For 2D matrices, the first line indicates the number of rows and columns, followed by newline-separated rows. For jagged 2D arrays, the first line indicates the number of rows, followed by newline-separated rows.",
            enhanced_prompt,
        )
        self.assertIn(
            "Sample Input 1:\n\n2 2\n1 2\n3 4\n\nSample Output 1:\n\n2 2\n\n",
            enhanced_prompt,
        )  # Corrected expected output

    def test_v2_rows_only_format_for_grid_problem(self):
        """
        Test to specifically check the v2_rows_only format for a 2D character matrix
        similar to 'count-submatrices-with-equal-frequency-of-x-and-y'.
        It should output only the number of rows, then the matrix content.
        """
        enhancer = EnhancedPromptEnhancer(format_version="v2_rows_only")
        problem_content = (
            "Example 1:\n" 'Input: grid = [["X","Y","."],["Y",".","."]]\n' "Output: 3"
        )
        enhanced_prompt = enhancer.enhance_prompt("Original prompt.", problem_content)
        self.assertIn(PromptConstants.FORMATTING_WITHOUT_STARTER_CODE, enhanced_prompt)
        self.assertIn(
            "For 2D arrays, the first line indicates the number of rows, followed by newline-separated rows.",
            enhanced_prompt,
        )
        self.assertIn(
            "Sample Input 1:\n\n2\nX Y .\nY . .\n\nSample Output 1:\n\n3\n",
            enhanced_prompt,
        )


class TestIntegrationWithUnifiedConverter(unittest.TestCase):

    def test_unified_analyzer_integration(self):
        """Tests integration with DataAnalyzer from unified_converter."""

        test_cases_raw = [
            {"input": {"grid": [[1, 2], [3, 4]]}, "output": [[5, 6], [7, 8]]},
            {
                "input": {"grid": [[1, 2], [3, 4, 5]]},  # Jagged!
                "output": [[9, 10], [11, 12]],
            },
        ]

        # Convert raw dict to TestCase for easier analysis
        parsed_test_cases = [
            UnifiedDataAnalyzer.analyze_test_case(tc) for tc in test_cases_raw
        ]

        # Use UnifiedConverterService for analysis
        unified_format = UnifiedDataAnalyzer.analyze_unified_test_case_format(
            parsed_test_cases
        )

        self.assertEqual(len(unified_format.input_format), 1)
        self.assertEqual(unified_format.input_format[0].var_name, "grid")
        self.assertEqual(unified_format.input_format[0].data_type, DataType.ARRAY_2D)

        self.assertEqual(unified_format.output_format.var_name, "output")
        self.assertEqual(
            unified_format.output_format.data_type, DataType.ARRAY_2D_MATRIX
        )
