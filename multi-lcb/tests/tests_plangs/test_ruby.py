from lcb_runner.evaluation.testing_plang import (
    eval_plang_code,
)
from dataclasses import asdict


TIMEOUT = 15
PLANG = "ruby"


class TestRubyEvaluation:
    """Test class for Ruby evaluation functions"""

    def test_simple_stdin_program(self):
        """Test successful execution"""
        program = """
def read_and_display_variables

  first_variable = gets.chomp
  second_variable = gets.chomp
  
  puts first_variable
  puts second_variable
end

read_and_display_variables
"""

        input_data = ["0\n0", "55555\n55555"]
        output_data = ["0\n0", "55555\n55555"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, PLANG, TIMEOUT
        )

        metadata = asdict(metadata)
        print(metadata)

        assert result == [True, True]
        assert "execution_time" in metadata
