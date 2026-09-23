from lcb_runner.evaluation.testing_plang import eval_plang_code
from dataclasses import asdict
import resource

TIMEOUT = 15
PLANG = "c#"


class TestCsharpEvaluation:
    """Test class for C# evaluation functions"""

    def test_simple_stdin_program(self):
        """Test successful execution"""
        program = """
using System;

public class ReadAndPrintNumbers
{
    public static void Main(string[] args)
    {
        string input1 = Console.ReadLine();
        string input2 = Console.ReadLine();

        int number1 = int.Parse(input1);
        int number2 = int.Parse(input2);

        Console.WriteLine(number1);
        Console.WriteLine(number2);
    }
}
"""

        resource.setrlimit(resource.RLIMIT_AS, (-1, -1))

        input_data = ["0\n0", "55555\n55555"]
        output_data = ["0\n0", "55555\n55555"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, PLANG, TIMEOUT
        )

        metadata = asdict(metadata)

        print(metadata)

        assert result == [True, True]
        assert "execution_time" in metadata
