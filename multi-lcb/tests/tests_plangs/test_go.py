from lcb_runner.evaluation.testing_plang import (
    eval_plang_code,
    prepare_plang_env,
)
from dataclasses import asdict


TIMEOUT = 15
PLANG = "go"


class TestGoEvaluation:
    """Test class for Go evaluation functions"""

    def test_cache_clean(self):
        prepare_plang_env(PLANG, timeout=3)

    def test_simple_stdin_program(self):
        """Test successful execution"""
        program = """
package main

import "fmt"

func main() {
	var num1, num2 int

	_, _ = fmt.Scan(&num1)
	_, _ = fmt.Scan(&num2)
    
    fmt.Println(num1)
    fmt.Println(num2)

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
