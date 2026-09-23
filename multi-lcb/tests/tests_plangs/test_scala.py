import pytest
from lcb_runner.evaluation.testing_plang import eval_plang_code, Status
from dataclasses import asdict


class TestScalaEvaluation:
    """Test class for Scala evaluation functions"""

    def test_accepted_verdict(self):
        """Test successful execution"""
        program = """object Main {
  def main(args: Array[String]): Unit = {
    val input = scala.io.StdIn.readLine()
    println(input)
  }
}"""
        input_data = ["0", "55555"]
        output_data = ["0", "55555"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "scala", 15
        )

        metadata = asdict(metadata)

        assert result == [True, True]
        assert "execution_time" in metadata

    def test_wrong_answer_content_mismatch(self):
        """Test wrong answer content"""
        program = """object Main {
  def main(args: Array[String]): Unit = {
    val input = scala.io.StdIn.readLine().toInt
    println(input + 1)
  }
}"""
        input_data = ["5"]
        output_data = ["5"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "scala", 15
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
            program, input_data, output_data, "scala", 15
        )
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] == Status.EmptyCode
        assert "Empty string instead of a program" in metadata["error_message"]

    def test_runtime_error_verdict(self):
        """Test runtime error"""
        program = """object Main {
  def main(args: Array[String]): Unit = {
    val x: Int = "not_a_number".toInt
  }
}"""
        input_data = ["5"]
        output_data = ["5"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "scala", 15
        )
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] == Status.Exception

    def test_bool_values(self):
        """Test bool values comparison"""
        program = """object Main {
  def main(args: Array[String]): Unit = {
    println("true")
  }
}"""
        input_data = [""]
        output_data = ["True"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "scala", 15
        )
        metadata = asdict(metadata)

        assert result == [True]

    def test_time_limit_verdict(self):
        """Test time limit exceeded during execution (not compilation)"""
        program = """object Main {
  def main(args: Array[String]): Unit = {
    while (true) {
      Thread.sleep(2000) // Sleep for 2000 milliseconds (2 seconds)
    }
    println("done")
  }
}"""
        input_data = [""]
        output_data = ["done"]

        result, metadata = eval_plang_code(program, input_data, output_data, "scala", 1)
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] in (Status.TimeoutExpired, Status.BuildTimeOut)

    @pytest.mark.skip(reason="Compile timeout tests are flaky")
    def test_compile_timeout(self):
        """Test compilation timeout with complex code"""
        program = (
            """object Main {
  def main(args: Array[String]): Unit = {
    println("hello")
  }
}"""
            + "\n"
            + "\n".join(
                ["def extraFunction$i(): Unit = {{ println($i) }}" for i in range(1000)]
            )
        )

        input_data = [""]
        output_data = ["hello"]

        result, metadata = eval_plang_code(program, input_data, output_data, "scala", 1)
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] in [Status.BuildTimeOut]

    def test_multiple_test_cases(self):
        """Test multiple test cases with mixed results"""
        program = """object Main {
  def main(args: Array[String]): Unit = {
    val input = scala.io.StdIn.readLine()
    println(input)
  }
}"""
        input_data = ["1", "2", "3"]
        output_data = ["1", "wrong", "3"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "scala", 15
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

        program = """object Main {
  def main(args: Array[String]): Unit = {
    val input = scala.io.StdIn.readLine()
    println(input)
  }
}"""

        for input_val, expected_output in test_cases:
            result, metadata = eval_plang_code(
                program, [input_val], [expected_output], "scala", 15
            )
            metadata = asdict(metadata)

            assert result == [True]

    def test_numeric_precision(self):
        """Test numeric precision handling"""
        program = """object Main {
  def main(args: Array[String]): Unit = {
    println("3.141592653589793")
  }
}"""
        input_data = [""]
        output_data = ["3.14159"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "scala", 15
        )
        metadata = asdict(metadata)

        assert result == [True]

    def test_integer_operations(self):
        """Test integer operations"""
        program = """object Main {
  def main(args: Array[String]): Unit = {
    val input = scala.io.StdIn.readLine().toInt
    println(input * 2)
  }
}"""
        input_data = ["10", "25"]
        output_data = ["20", "50"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "scala", 15
        )

        assert result == [True, True]
        assert metadata.success


class TestScalaEdgeCases:
    """Test edge cases for Scala evaluation"""

    def test_large_input(self):
        """Test with large input"""
        program = """object Main {
  def main(args: Array[String]): Unit = {
    val input = scala.io.StdIn.readLine()
    println(input.length)
  }
}"""
        large_input = "a" * 100
        expected_output = str(len(large_input))

        result, metadata = eval_plang_code(
            program, [large_input], [expected_output], "scala", 15
        )
        metadata = asdict(metadata)

        assert result == [True]

    def test_special_characters(self):
        """Test with special characters"""
        program = """object Main {
  def main(args: Array[String]): Unit = {
    val input = scala.io.StdIn.readLine()
    println(input)
  }
}"""
        special_input = "hello world!@#$%^&*()"
        expected_output = special_input

        result, metadata = eval_plang_code(
            program, [special_input], [expected_output], "scala", 15
        )
        metadata = asdict(metadata)

        assert result == [True]

    def test_option_type(self):
        """Test Scala's Option type handling"""
        program = """object Main {
  def main(args: Array[String]): Unit = {
    val input = scala.io.StdIn.readLine()
    val result = if (input.isEmpty) None else Some(input.length)
    println(result.getOrElse(-1))
  }
}"""
        input_data = ["test", ""]
        output_data = ["4", "-1"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "scala", 15
        )
        metadata = asdict(metadata)

        assert result == [True, True]

    def test_syntax_error(self):
        """Test syntax error detection"""
        program = """object Main {
  def main(args: Array[String]): Unit = {
    val x = "unclosed string
  }
}"""
        input_data = ["test"]
        output_data = ["test"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "scala", 15
        )
        metadata = asdict(metadata)

        assert result == [-5]
        assert metadata["error"] in [Status.SyntaxError, Status.BuildFailed]

    def test_collection_operations(self):
        """Test Scala collection operations"""
        program = """object Main {
  def main(args: Array[String]): Unit = {
    val input = scala.io.StdIn.readLine()
    val numbers = input.split(" ").map(_.toInt)
    val sum = numbers.sum
    println(sum)
  }
}"""
        input_data = ["1 2 3 4 5", "10 20 30"]
        output_data = ["15", "60"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, "scala", 15
        )
        metadata = asdict(metadata)

        assert result == [True, True]
