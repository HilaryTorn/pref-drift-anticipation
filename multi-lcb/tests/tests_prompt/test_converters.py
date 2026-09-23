#!/usr/bin/env python3
"""
Tests for all converter versions.
"""

import pytest

from prompt_enhancer.unified_converter import (
    DataType,
    DataItem,
    TestCase,
    V1NoDimensionsConverter,
    V2RowsOnlyConverter,
    V3SmartMatrixConverter,
    ConverterFactory,
    UnifiedParameterFormat,
    UnifiedTaskFormat,
    UnifiedDataAnalyzer,
)


class TestV1NoDimensionsConverter:
    """Tests for V1 converter."""

    def setup_method(self):
        self.processor = ConverterFactory.create_converter("v1_no_dimensions")

    def test_convert_input_matrix(self):
        """
        Test conversion of an input matrix.
        """
        input_data = {"matrix": [[1, 2], [3, 4]], "scalar": 5}
        output_data = 0  # Placeholder output, not used for input conversion tests

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "1 2\n3 4\n5"
        assert result["stdin"] == expected_stdin

    def test_convert_input_jagged(self):
        """
        Test conversion of a jagged input array.
        """
        input_data = {"jagged": [[1, 2], [3, 4, 5]]}
        output_data = 0

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "1 2\n3 4 5"
        assert result["stdin"] == expected_stdin

    def test_convert_input_1d_array(self):
        """
        Test conversion of a 1D input array.
        """
        input_data = {"array": [1, 2, 3]}
        output_data = 0

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "1 2 3"
        assert result["stdin"] == expected_stdin

    def test_convert_input_string(self):
        """
        Test conversion of an input string.
        """
        input_data = {"string": "hello"}
        output_data = 0

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "hello"
        assert result["stdin"] == expected_stdin

    def test_convert_input_empty_list(self):
        """
        Test conversion of an empty input list.
        """
        input_data = {"empty": []}
        output_data = 0

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = ""
        assert result["stdin"] == expected_stdin

    def test_convert_output_matrix(self):
        """
        Test conversion of an output matrix.
        """
        input_data = {}
        output_data = [[1, 2], [3, 4]]

        result = self.processor.convert_test(input_data, output_data)
        expected_stdout = "1 2\n3 4"
        assert result["stdout"] == expected_stdout

    def test_convert_output_scalar(self):
        """
        Test conversion of a scalar output.
        """
        input_data = {}
        output_data = 42

        result = self.processor.convert_test(input_data, output_data)
        expected_stdout = "42"
        assert result["stdout"] == expected_stdout

    def test_convert_output_boolean(self):
        """
        Test conversion of a boolean output.
        """
        input_data = {}
        output_data = True

        result = self.processor.convert_test(input_data, output_data)
        expected_stdout = "true"
        assert result["stdout"] == expected_stdout

    def test_convert_input_hard_test(self):
        """
        Test conversion of mixed data types in input arrays.
        """
        input_data = {"matrix": [[1, "2"], [3, "4"]], "array": [1, 2, 3], "scalar": 42}
        output_data = 0

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "1 2\n3 4\n1 2 3\n42"
        assert result["stdin"] == expected_stdin

    def test_convert_input_array_of_booleans(self):
        """
        Test conversion of an input array of booleans.
        """
        input_data = {"bool_array": [True, False, True]}
        output_data = 0

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "true false true"
        assert result["stdin"] == expected_stdin

    def test_convert_input_array_of_booleans_strings(self):
        """
        Test conversion of an input array of booleans as strings.
        """
        input_data = {"bool_array": ["True", "False", "True"]}
        output_data = 0

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "True False True"
        assert result["stdin"] == expected_stdin

    def test_convert_input_array_of_strings(self):
        """
        Test conversion of an input array of strings.
        """
        input_data = {"string_array": ["hello", "world", 1]}
        output_data = 0

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "hello world 1"
        assert result["stdin"] == expected_stdin

    def test_boolean_and_null_conversion_v1(self):
        """
        Test conversion of boolean and null values in V1NoDimensionsConverter.
        """
        input_data = {"bool_true": True, "bool_false": False, "none_val": None}
        # output_data = {"bool_true": True, "bool_false": False, "none_val": None}

        result_input = self.processor.convert_test(input_data, {})["stdin"]
        assert "true" in result_input
        assert "false" in result_input
        assert "null" in result_input

        # Test individual boolean and null output conversions
        result_output_true = self.processor.convert_test({}, True)["stdout"]
        assert result_output_true == "true"

        result_output_false = self.processor.convert_test({}, False)["stdout"]
        assert result_output_false == "false"

        result_output_null = self.processor.convert_test({}, None)["stdout"]
        assert result_output_null == "null"

    def test_boolean_and_null_in_list_conversion(self):
        """
        Test list of booleans and nulls in V1NoDimensionsConverter.
        """
        input_data = {
            "bool_array": [True, False, True],
            "none_array": [None, None, None],
        }
        output_data = 0

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "true false true\nnull null null"
        assert result["stdin"] == expected_stdin


class TestV2RowsOnlyConverter:
    """Tests for V2 converter."""

    def setup_method(self):
        self.processor = ConverterFactory.create_converter("v2_rows_only")

    def test_convert_input_matrix(self):
        """Test conversion of an input matrix with row count."""
        input_data = {"matrix": [[1, 2], [3, 4]]}
        output_data = [[1, 2], [3, 4]]

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "2\n1 2\n3 4"
        assert result["stdin"] == expected_stdin

    def test_convert_input_jagged(self):
        """Test conversion of a jagged input array with row count."""
        input_data = {"jagged": [[1, 2], [3, 4, 5]]}
        output_data = [[1, 2], [3, 4, 5]]

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "2\n1 2\n3 4 5"
        assert result["stdin"] == expected_stdin

    def test_convert_input_empty_matrix(self):
        """Test conversion of an empty input matrix."""
        input_data = {"empty": []}
        output_data = []

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = ""
        assert result["stdin"] == expected_stdin

    def test_convert_output_matrix(self):
        """Test conversion of an output matrix."""
        input_data = {}
        output_data = [[1, 2, 3], [4, 5, 6]]

        result = self.processor.convert_test(input_data, output_data)
        expected_stdout = "2\n1 2 3\n4 5 6"
        assert result["stdout"] == expected_stdout

    def test_convert_mixed_input(self):
        """Test conversion of mixed input types."""
        input_data = {"matrix": [[1, 2]], "array": [7, 8, 9], "scalar": 10}
        output_data = [[1, 2]]

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "1\n1 2\n7 8 9\n10"
        assert result["stdin"] == expected_stdin


class TestV3SmartMatrixConverter:
    """Tests for V3 converter."""

    def setup_method(self):
        self.processor = ConverterFactory.create_converter(version="v3_smart_matrix")

    def test_convert_input_matrix(self):
        """Test conversion of an input matrix with rows and columns."""
        input_data = {"matrix": [[1, 2, 3], [4, 5, 6]]}
        output_data = [[1, 2, 3], [4, 5, 6]]

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "2 3\n1 2 3\n4 5 6"
        assert result["stdin"] == expected_stdin

    def test_convert_input_jagged(self):
        """Test conversion of a jagged input array with row count."""
        input_data = {"jagged": [[1, 2], [3, 4, 5]]}
        output_data = [[1, 2], [3, 4, 5]]

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "2\n1 2\n3 4 5"
        assert result["stdin"] == expected_stdin

    def test_convert_input_single_element_matrix(self):
        """Test conversion of a single-element input matrix."""
        input_data = {"single": [[42]]}
        output_data = [[42]]

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "1 1\n42"
        assert result["stdin"] == expected_stdin

    def test_convert_output_matrix(self):
        """
        Test conversion of an output matrix with rows and columns.
        """
        input_data = {}
        output_data = [[1, 2], [3, 4]]

        result = self.processor.convert_test(input_data, output_data)
        expected_stdout = "2 2\n1 2\n3 4"
        assert result["stdout"] == expected_stdout

    def test_convert_output_jagged(self):
        """
        Test conversion of a jagged output array with row count.
        """
        input_data = {}
        output_data = [[1, 2], [3, 4, 5]]

        result = self.processor.convert_test(input_data, output_data)
        expected_stdout = "2\n1 2\n3 4 5"
        assert result["stdout"] == expected_stdout

    def test_convert_square_matrix(self):
        """
        Test conversion of a square input matrix.
        """
        input_data = {"square": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
        output_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        result = self.processor.convert_test(input_data, output_data)
        expected_stdin = "3 3\n1 2 3\n4 5 6\n7 8 9"
        assert result["stdin"] == expected_stdin


class TestConverterFactory:
    """Tests for the converter factory."""

    def test_create_v1_converter(self):
        """Test creating a V1 converter."""
        converter = ConverterFactory.create_converter("v1_no_dimensions")
        assert isinstance(converter, V1NoDimensionsConverter)

    def test_create_v2_converter(self):
        """Test creating a V2 converter."""
        converter = ConverterFactory.create_converter("v2_rows_only")
        assert isinstance(converter, V2RowsOnlyConverter)

    def test_create_v3_converter(self):
        """Test creating a V3 converter."""
        converter = ConverterFactory.create_converter("v3_smart_matrix")
        assert isinstance(converter, V3SmartMatrixConverter)

    def test_create_unknown_converter(self):
        """Test creating an unknown converter."""
        with pytest.raises(ValueError, match="Unknown converter version"):
            ConverterFactory.create_converter("unknown_version")

    def test_create_none_converter(self):
        """Test creating a converter with None version."""
        with pytest.raises(ValueError):
            ConverterFactory.create_converter(None)

    def test_create_empty_string_converter(self):
        """Test creating a converter with an empty string version."""
        with pytest.raises(ValueError):
            ConverterFactory.create_converter("")


class TestConverterEdgeCases:
    """Tests for edge cases across all converters."""

    def test_negative_numbers_all_converters(self):
        """Test negative numbers across all converters using `UnifiedDataProcessor`."""
        input_data = {"matrix": [[-1, -2], [-3, -4]]}
        output_data = [[-1, -2], [-3, -4]]

        # V1
        processor_v1 = ConverterFactory.create_converter("v1_no_dimensions")
        result_v1 = processor_v1.convert_test(input_data, output_data)
        assert "-1 -2" in result_v1["stdin"]
        assert "-3 -4" in result_v1["stdin"]

        # V2
        processor_v2 = ConverterFactory.create_converter("v2_rows_only")
        result_v2 = processor_v2.convert_test(input_data, output_data)
        assert result_v2["stdin"].startswith("2")
        assert "-1 -2" in result_v2["stdin"]

        # V3
        processor_v3 = ConverterFactory.create_converter("v3_smart_matrix")
        result_v3 = processor_v3.convert_test(input_data, output_data)
        assert result_v3["stdin"].startswith("2 2")
        assert "-1 -2" in result_v3["stdin"]

    def test_floating_point_numbers(self):
        """Test floating point numbers using `ConverterFactory`."""
        input_data = {"float_matrix": [[1.5, 2.7], [3.14, 4.0]]}
        output_data = [[1.5, 2.7], [3.14, 4.0]]

        processor = ConverterFactory.create_converter(version="v3_smart_matrix")
        result = processor.convert_test(input_data, output_data)

        assert "1.5 2.7" in result["stdin"]
        assert "3.14 4.0" in result["stdin"]

    def test_large_matrix(self):
        """Test a large matrix using `ConverterFactory.create_converter`."""
        large_matrix = [[i * j for j in range(100)] for i in range(50)]
        input_data = {"large_matrix": large_matrix}
        output_data = large_matrix

        processor = ConverterFactory.create_converter(version="v3_smart_matrix")
        result = processor.convert_test(input_data, output_data)

        assert result["stdin"].startswith("50 100")
        lines = result["stdin"].split("\n")
        assert len(lines) == 51  # Header + 50 rows

    def test_zero_values(self):
        """Test zero values using `ConverterFactory.create_converter`."""
        input_data = {"zero_matrix": [[0, 0], [0, 0]]}
        output_data = [[0, 0], [0, 0]]

        processor = ConverterFactory.create_converter(version="v2_rows_only")
        result = processor.convert_test(input_data, output_data)

        assert "2\n0 0\n0 0" == result["stdin"]


class TestUnifiedDataAnalyzerAnalysis:
    """Tests for UnifiedDataAnalyzer's analysis methods."""

    @pytest.mark.parametrize(
        "type_counts, context, expected_type",
        [
            ({DataType.SCALAR: 5}, "scalar_context", DataType.SCALAR),
            ({DataType.STRING: 3}, "string_context", DataType.STRING),
            ({DataType.ARRAY_1D: 2}, "array_1d_context", DataType.ARRAY_1D),
            ({DataType.ARRAY_2D_MATRIX: 4}, "matrix_context", DataType.ARRAY_2D_MATRIX),
            ({DataType.ARRAY_2D: 1}, "array_2d_context", DataType.ARRAY_2D),
            (
                {DataType.ARRAY_2D: 2, DataType.ARRAY_2D_MATRIX: 3},
                "mixed_2d_context",
                DataType.ARRAY_2D,
            ),
        ],
    )
    def test_resolve_unified_type_valid_cases(
        self, type_counts, context, expected_type
    ):
        """Tests _resolve_unified_type with valid, consistent type counts."""
        assert (
            UnifiedDataAnalyzer.resolve_unified_type(type_counts, context)
            == expected_type
        )

    @pytest.mark.parametrize(
        "type_counts, context, expected_error_message_part",
        [
            ({}, "empty_context", "no data provided"),
            (
                {DataType.SCALAR: 1, DataType.ARRAY_1D: 1},
                "inconsistent_context",
                "Inconsistent data types",
            ),
            (
                {DataType.STRING: 1, DataType.ARRAY_2D_MATRIX: 1},
                "another_inconsistent",
                "Inconsistent data types",
            ),
        ],
    )
    def test_resolve_unified_type_invalid_cases(
        self, type_counts, context, expected_error_message_part
    ):
        """Tests _resolve_unified_type with invalid or inconsistent type counts, expecting ValueError."""
        with pytest.raises(ValueError) as excinfo:
            UnifiedDataAnalyzer.resolve_unified_type(type_counts, context)
        assert expected_error_message_part in str(excinfo.value)

    def test_analyze_unified_test_case_format_empty_list(self):
        """Tests analyze_unified_test_case_format with an empty list of test cases."""
        unified_format = UnifiedDataAnalyzer.analyze_unified_test_case_format([])
        assert unified_format == UnifiedTaskFormat(input_format=[], output_format=None)

    def test_analyze_unified_test_case_format_single_test_case_scalar(self):
        """Tests analyze_unified_test_case_format with a single test case having scalar types."""
        test_cases = [
            TestCase(
                input=[
                    DataItem(
                        value=10,
                        data_type=DataType.SCALAR,
                        metadata={"var_name": "param_0"},
                    ),
                    DataItem(
                        value="hello",
                        data_type=DataType.STRING,
                        metadata={"var_name": "param_1"},
                    ),
                ],
                output=DataItem(
                    value=20, data_type=DataType.SCALAR, metadata={"var_name": "output"}
                ),
            )
        ]
        unified_format = UnifiedDataAnalyzer.analyze_unified_test_case_format(
            test_cases
        )
        expected_input = [
            UnifiedParameterFormat(var_name="param_0", data_type=DataType.SCALAR),
            UnifiedParameterFormat(var_name="param_1", data_type=DataType.STRING),
        ]
        expected_output = UnifiedParameterFormat(
            var_name="output", data_type=DataType.SCALAR
        )
        assert unified_format == UnifiedTaskFormat(
            input_format=expected_input, output_format=expected_output
        )

    def test_analyze_unified_test_case_format_multiple_test_cases_consistent(self):
        """Tests analyze_unified_test_case_format with multiple consistent test cases."""
        test_cases = [
            TestCase(
                input=[
                    DataItem(
                        value=[1, 2],
                        data_type=DataType.ARRAY_1D,
                        metadata={"var_name": "arr_in"},
                    )
                ],
                output=DataItem(
                    value=[3, 4],
                    data_type=DataType.ARRAY_1D,
                    metadata={"var_name": "arr_out"},
                ),
            ),
            TestCase(
                input=[
                    DataItem(
                        value=[5, 6, 7],
                        data_type=DataType.ARRAY_1D,
                        metadata={"var_name": "arr_in"},
                    )
                ],
                output=DataItem(
                    value=[8, 9],
                    data_type=DataType.ARRAY_1D,
                    metadata={"var_name": "arr_out"},
                ),
            ),
        ]
        unified_format = UnifiedDataAnalyzer.analyze_unified_test_case_format(
            test_cases
        )
        expected_input = [
            UnifiedParameterFormat(var_name="arr_in", data_type=DataType.ARRAY_1D)
        ]
        expected_output = UnifiedParameterFormat(
            var_name="arr_out", data_type=DataType.ARRAY_1D
        )
        assert unified_format == UnifiedTaskFormat(
            input_format=expected_input, output_format=expected_output
        )

    def test_analyze_unified_test_case_format_multiple_test_cases_mixed_2d(self):
        """Tests analyze_unified_test_case_format with multiple test cases having mixed 2D array types."""
        test_cases = [
            TestCase(
                input=[
                    DataItem(
                        value=[[1, 2], [3, 4]],
                        data_type=DataType.ARRAY_2D_MATRIX,
                        metadata={"var_name": "mat_in"},
                    )
                ],
                output=DataItem(
                    value=[[1, 2, 3], [4, 5, 6]],
                    data_type=DataType.ARRAY_2D,
                    metadata={"var_name": "jagged_out"},
                ),
            ),
            TestCase(
                input=[
                    DataItem(
                        value=[[5, 6], [7, 8], [9, 10]],
                        data_type=DataType.ARRAY_2D_MATRIX,
                        metadata={"var_name": "mat_in"},
                    )
                ],
                output=DataItem(
                    value=[[7, 8]],
                    data_type=DataType.ARRAY_2D_MATRIX,
                    metadata={"var_name": "jagged_out"},
                ),
            ),
            TestCase(
                input=[
                    DataItem(
                        value=[[11], [12, 13]],
                        data_type=DataType.ARRAY_2D,
                        metadata={"var_name": "mat_in"},
                    )
                ],
                output=DataItem(
                    value=[[9]],
                    data_type=DataType.ARRAY_2D,
                    metadata={"var_name": "jagged_out"},
                ),
            ),
        ]
        unified_format = UnifiedDataAnalyzer.analyze_unified_test_case_format(
            test_cases
        )
        expected_input = [
            UnifiedParameterFormat(var_name="mat_in", data_type=DataType.ARRAY_2D)
        ]
        expected_output = UnifiedParameterFormat(
            var_name="jagged_out", data_type=DataType.ARRAY_2D
        )
        assert unified_format == UnifiedTaskFormat(
            input_format=expected_input, output_format=expected_output
        )

    def test_analyze_unified_test_case_format_inconsistent_input_parameters_count(self):
        """Tests analyze_unified_test_case_format with inconsistent number of input parameters across tests."""
        test_cases = [
            TestCase(
                input=[DataItem(value=1, data_type=DataType.SCALAR)],
                output=DataItem(value=2, data_type=DataType.SCALAR),
            ),
            TestCase(
                input=[
                    DataItem(value=3, data_type=DataType.SCALAR),
                    DataItem(value=4, data_type=DataType.SCALAR),
                ],
                output=DataItem(value=5, data_type=DataType.SCALAR),
            ),
        ]
        with pytest.raises(ValueError) as excinfo:
            UnifiedDataAnalyzer.analyze_unified_test_case_format(test_cases)
        assert "Number of input parameters differs between test cases." in str(
            excinfo.value
        )

    def test_analyze_unified_test_case_format_input_inconsistent_types(self):
        """Tests analyze_unified_test_case_format with inconsistent input data types for the same parameter."""
        test_cases = [
            TestCase(
                input=[
                    DataItem(
                        value=1,
                        data_type=DataType.SCALAR,
                        metadata={"var_name": "param_0"},
                    )
                ],
                output=DataItem(value=2, data_type=DataType.SCALAR),
            ),
            TestCase(
                input=[
                    DataItem(
                        value=[1, 2],
                        data_type=DataType.ARRAY_1D,
                        metadata={"var_name": "param_0"},
                    )
                ],
                output=DataItem(value=3, data_type=DataType.SCALAR),
            ),
        ]
        with pytest.raises(ValueError) as excinfo:
            UnifiedDataAnalyzer.analyze_unified_test_case_format(test_cases)
        assert "Inconsistent data types for input param param_0" in str(excinfo.value)

    def test_analyze_unified_test_case_format_output_inconsistent_types(self):
        """Tests analyze_unified_test_case_format with inconsistent output data types."""
        test_cases = [
            TestCase(
                input=[],
                output=DataItem(
                    value=1, data_type=DataType.SCALAR, metadata={"var_name": "output"}
                ),
            ),
            TestCase(
                input=[],
                output=DataItem(
                    value="hello",
                    data_type=DataType.STRING,
                    metadata={"var_name": "output"},
                ),
            ),
        ]
        with pytest.raises(ValueError) as excinfo:
            UnifiedDataAnalyzer.analyze_unified_test_case_format(test_cases)
        assert "Inconsistent data types for output" in str(excinfo.value)

    def test_analyze_unified_test_case_format_no_output(self):
        """Tests analyze_unified_test_case_format when no output is present in test cases."""
        test_cases = [
            TestCase(
                input=[
                    DataItem(
                        value=1,
                        data_type=DataType.SCALAR,
                        metadata={"var_name": "param_0"},
                    )
                ],
                output=None,
            ),
            TestCase(
                input=[
                    DataItem(
                        value=2,
                        data_type=DataType.SCALAR,
                        metadata={"var_name": "param_0"},
                    )
                ],
                output=None,
            ),
        ]
        unified_format = UnifiedDataAnalyzer.analyze_unified_test_case_format(
            test_cases
        )
        expected_input = [
            UnifiedParameterFormat(var_name="param_0", data_type=DataType.SCALAR)
        ]
        assert unified_format == UnifiedTaskFormat(
            input_format=expected_input, output_format=None
        )
