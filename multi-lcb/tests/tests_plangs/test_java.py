from lcb_runner.evaluation.testing_plang import eval_plang_code
from dataclasses import asdict


TIMEOUT = 15
PLANG = "java"


class TestJavaEvaluation:
    """Test class for Java evaluation functions"""

    def test_simple_stdin_program(self):
        """Test successful execution"""
        program = """
import java.util.Scanner;

public class ReadAndPrintLines {

    public static void main(String[] args) {
        // Create a Scanner object to read input from stdin
        Scanner scanner = new Scanner(System.in);

        // read user input
        int v1 = scanner.nextInt();
        int v2 = scanner.nextInt();

        // Close the scanner to release system resources
        scanner.close();

        System.out.println(v1);
        System.out.println(v2);
        
    }
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
