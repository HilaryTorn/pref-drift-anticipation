import pytest

from lcb_runner.evaluation.testing_plang import eval_plang_code, Status
from dataclasses import asdict


class TestPHPEvaluation:
    """Test class for PHP evaluation functions"""

    def test_accepted_verdict(self):
        """Test successful execution"""
        program = "<?php\n$input = trim(fgets(STDIN));\necho $input;\n?>"
        input_data = ["0", "55555"]
        output_data = ["0", "55555"]

        result, metadata = eval_plang_code(program, input_data, output_data, "php", 15)
        metadata = asdict(metadata)

        assert result == [True, True]
        assert "execution_time" in metadata

    def test_wrong_answer_content_mismatch(self):
        """Test wrong answer content"""
        program = "<?php\n$input = trim(fgets(STDIN));\necho $input + 1;\n?>"
        input_data = ["5"]
        output_data = ["5"]

        result, metadata = eval_plang_code(program, input_data, output_data, "php", 15)
        metadata = asdict(metadata)

        assert result == [-2]
        assert metadata["error"] == Status.WrongAnswer

    def test_empty_code_verdict(self):
        """Test empty program"""
        program = ""
        input_data = ["5"]
        output_data = ["5"]

        result, metadata = eval_plang_code(program, input_data, output_data, "php", 15)
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] == Status.EmptyCode
        assert "Empty string instead of a program" in metadata["error_message"]

    def test_runtime_error_verdict(self):
        """Test runtime error"""
        program = "<?php\nundefined_function();\n?>"
        input_data = ["5"]
        output_data = ["5"]

        result, metadata = eval_plang_code(program, input_data, output_data, "php", 15)
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] == Status.Exception

    def test_bool_values(self):
        """Test bool values comparison"""
        program = "<?php\necho 'true';\n?>"
        input_data = [""]
        output_data = ["True"]

        result, metadata = eval_plang_code(program, input_data, output_data, "php", 15)
        metadata = asdict(metadata)

        assert result == [True]

    def test_time_limit_verdict(self):
        """Test time limit exceeded during execution (not compilation)"""
        program = "<?php\nwhile(true) {sleep(1);}\n?>"
        input_data = ["5"]
        output_data = ["5"]

        result, metadata = eval_plang_code(program, input_data, output_data, "php", 1)
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] in [Status.TimeoutExpired, Status.BuildFailed]

    @pytest.mark.skip(reason="Compile timeout tests are flaky")
    def test_compile_timeout(self):
        """Test compilation timeout with complex code"""
        program = (
            '<?php\necho "hello";\n?>'
            + "\nfunction extraFunction$i() { echo $i; }" * 1000
        )
        input_data = [""]
        output_data = ["hello"]

        result, metadata = eval_plang_code(program, input_data, output_data, "php", 1)
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] in [Status.BuildTimeOut, Status.BuildFailed]

    def test_multiple_test_cases(self):
        """Test multiple test cases with mixed results"""
        program = "<?php\n$input = trim(fgets(STDIN));\necho $input;\n?>"
        input_data = ["1", "2", "3"]
        output_data = ["1", "wrong", "3"]

        result, metadata = eval_plang_code(program, input_data, output_data, "php", 15)
        metadata = asdict(metadata)

        assert -2 in result
        assert metadata["error"] == Status.WrongAnswer

    def test_various_input_types(self):
        """Test various input types"""
        test_cases = [
            ("5", "5"),
            ("hello", "hello"),
            ("3.14", "3.14"),
            ("", ""),
        ]

        program = "<?php\n$input = trim(fgets(STDIN));\necho $input;\n?>"

        for input_val, expected_output in test_cases:
            result, metadata = eval_plang_code(
                program, [input_val], [expected_output], "php", 15
            )
            metadata = asdict(metadata)

            assert result == [True]

    def test_numeric_precision(self):
        """Test numeric precision handling"""
        program = '<?php\necho "3.141592653589793";\n?>'
        input_data = [""]
        output_data = ["3.14159"]

        result, metadata = eval_plang_code(program, input_data, output_data, "php", 15)
        metadata = asdict(metadata)

        assert result == [True]

    def test_integer_operations(self):
        """Test integer operations"""
        program = "<?php\n$input = trim(fgets(STDIN));\necho $input * 2;\n?>"
        input_data = ["10", "25"]
        output_data = ["20", "50"]

        result, metadata = eval_plang_code(program, input_data, output_data, "php", 15)
        metadata = asdict(metadata)

        assert result == [True, True]


class TestPHPEdgeCases:
    """Test edge cases for PHP evaluation"""

    def test_large_input(self):
        """Test with large input"""
        program = "<?php\n$input = trim(fgets(STDIN));\necho strlen($input);\n?>"
        large_input = "a" * 100
        expected_output = str(len(large_input))

        result, metadata = eval_plang_code(
            program, [large_input], [expected_output], "php", 15
        )
        metadata = asdict(metadata)

        assert result == [True]

    def test_special_characters(self):
        """Test with special characters"""
        program = "<?php\n$input = trim(fgets(STDIN));\necho $input;\n?>"
        special_input = "hello world!@#$%^&*()"
        expected_output = special_input

        result, metadata = eval_plang_code(
            program, [special_input], [expected_output], "php", 15
        )
        metadata = asdict(metadata)

        assert result == [True]

    def test_null_handling(self):
        """Test PHP's null handling"""
        program = "<?php\n$input = trim(fgets(STDIN));\necho $input === '' ? 'empty' : 'not_empty';\n?>"
        input_data = ["test", ""]
        output_data = ["not_empty", "empty"]

        result, metadata = eval_plang_code(program, input_data, output_data, "php", 15)
        metadata = asdict(metadata)

        assert result == [True, True]

    def test_syntax_error(self):
        """Test syntax error detection"""
        program = '<?php\n$input = unclosed string"\n?>'
        input_data = ["test"]
        output_data = ["test"]

        result, metadata = eval_plang_code(program, input_data, output_data, "php", 15)
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] == Status.Exception
