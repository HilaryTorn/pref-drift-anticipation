from lcb_runner.evaluation.testing_plang import eval_plang_code
from dataclasses import asdict


TIMEOUT = 15
PLANG = "c++"


class TestCppEvaluation:
    """Test class for C++ evaluation functions"""

    def test_simple_stdin_program(self):
        """Test successful execution"""
        program = """
#include <iostream> // Required for input/output operations

int main() {
    int num1; // Declare an integer variable to store the first number
    int num2; // Declare an integer variable to store the second number

    // Read the two numbers from standard input
    std::cin >> num1 >> num2;

    // Print the two numbers to standard output
    std::cout << num1 << std::endl;
    std::cout << num2 << std::endl;

    return 0; // Indicate successful program execution
}
"""

        input_data = ["0\n0", "55555\n55555"]
        output_data = ["0\n0", "55555\n55555"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, PLANG, TIMEOUT
        )

        metadata = asdict(metadata)

        assert result == [True, True]
        assert "execution_time" in metadata
