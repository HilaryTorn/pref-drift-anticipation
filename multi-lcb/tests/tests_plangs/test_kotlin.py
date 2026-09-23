import pytest
from lcb_runner.evaluation.testing_plang import eval_plang_code, Status
from dataclasses import asdict


TIMEOUT = 15


class TestKotlinEvaluation:
    """Test class for Kotlin evaluation functions"""

    def test_accepted_verdict(self):
        """Test successful execution"""
        program = "fun main() {\nval input = readln()\nprintln(input)\n}"
        input_data = ["0", "55555"]
        output_data = ["0", "55555"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "kotlin", TIMEOUT
        )

        metadata = asdict(metadata)

        assert result == [True, True]
        assert "execution_time" in metadata

    def test_wrong_answer_content_mismatch(self):
        """Test wrong answer content"""
        program = "fun main() {\nval input = readln().toInt()\nprintln(input + 1)\n}"
        input_data = ["5"]
        output_data = ["5"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "kotlin", TIMEOUT
        )
        metadata = asdict(metadata)

        assert result == [-2]
        assert metadata["error"] == Status.WrongAnswer

    def test_empty_code_verdict(self):
        """Test empty program"""
        program = ""
        input_data = ["5"]
        output_data = ["5"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "kotlin", TIMEOUT
        )
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] == Status.EmptyCode
        assert "Empty string instead of a program" in metadata["error_message"]

    def test_runtime_error_verdict(self):
        """Test runtime error"""
        program = 'fun main() {\nval x: Int = "not_a_number".toInt()\n}'
        input_data = ["5"]
        output_data = ["5"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "kotlin", TIMEOUT
        )
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] == Status.Exception

    def test_bool_values(self):
        """Test bool values comparison"""
        program = 'fun main() {\nprintln("true")\n}'
        input_data = [""]
        output_data = ["True"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "kotlin", TIMEOUT
        )
        metadata = asdict(metadata)

        assert result == [True]

    def test_time_limit_verdict(self):
        """Test time limit exceeded during execution (not compilation)"""
        program = """
        fun main() {
            while (true) {
                println("This is an infinite loop!")
                Thread.sleep(1000) // Pause for 1 second (1000 milliseconds)
            }
            println("done")
        }
        """
        input_data = [""]
        output_data = ["done"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "kotlin", 1
        )

        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] in [Status.TimeoutExpired, Status.BuildFailed]

    @pytest.mark.skip(reason="Compile timeout tests are flaky")
    def test_compile_timeout(self):
        """Test compilation timeout with complex code"""
        program = (
            'fun main() {\nprintln("hello")\n}'
            + "\nfun extraFunction$i() { println($i) }" * 10000
        )
        input_data = [""]
        output_data = ["hello"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "kotlin", 1
        )
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] in [Status.BuildTimeOut, Status.BuildFailed]

    def test_multiple_test_cases(self):
        """Test multiple test cases with mixed results"""
        program = "fun main() {\nval input = readln()\nprintln(input)\n}"
        input_data = ["1", "2", "3"]
        output_data = ["1", "wrong", "3"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "kotlin", TIMEOUT
        )
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

        program = "fun main() {\nval input = readln()\nprintln(input)\n}"

        for input_val, expected_output in test_cases:
            result, metadata = eval_plang_code(
                program, [input_val], [expected_output], "kotlin", TIMEOUT
            )
            metadata = asdict(metadata)

            assert result == [True]

    def test_numeric_precision(self):
        """Test numeric precision handling"""
        program = 'fun main() {\nprintln("3.141592653589793")\n}'
        input_data = [""]
        output_data = ["3.14159"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "kotlin", TIMEOUT
        )
        metadata = asdict(metadata)

        assert result == [True]

    def test_integer_operations(self):
        """Test integer operations"""
        program = "fun main() {\nval input = readln().toInt()\nprintln(input * 2)\n}"
        input_data = ["10", "25"]
        output_data = ["20", "50"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "kotlin", TIMEOUT
        )
        metadata = asdict(metadata)

        assert result == [True, True]


class TestKotlinEdgeCases:
    """Test edge cases for Kotlin evaluation"""

    def test_large_input(self):
        """Test with large input"""
        program = "fun main() {\nval input = readln()\nprintln(input.length)\n}"
        large_input = "a" * 100
        expected_output = str(len(large_input))

        result, metadata = eval_plang_code(
            program, [large_input], [expected_output], "kotlin", TIMEOUT
        )
        metadata = asdict(metadata)

        assert result == [True]

    def test_special_characters(self):
        """Test with special characters"""
        program = "fun main() {\nval input = readln()\nprintln(input)\n}"
        special_input = "hello world!@#$%^&*()"
        expected_output = special_input

        result, metadata = eval_plang_code(
            program, [special_input], [expected_output], "kotlin", TIMEOUT
        )
        metadata = asdict(metadata)

        assert result == [True]

    def test_null_safety(self):
        """Test Kotlin's null safety features"""
        program = "fun main() {\nval input: String? = readln()\nprintln(input?.length ?: -1)\n}"
        input_data = ["test", ""]
        output_data = ["4", "0"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "kotlin", TIMEOUT
        )
        metadata = asdict(metadata)

        assert result == [True, True]

    def test_syntax_error(self):
        """Test syntax error detection"""
        program = 'fun main() {\nval x = unclosed string"\n}'
        input_data = ["test"]
        output_data = ["test"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "kotlin", TIMEOUT
        )
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] in [Status.SyntaxError, Status.BuildFailed]
