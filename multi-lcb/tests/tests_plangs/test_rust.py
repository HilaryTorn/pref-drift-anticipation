from lcb_runner.evaluation.testing_plang import (
    eval_plang_code,
    Status,
    Result,
    TestScore,
    get_build_status,
)


TIMEOUT = 15
PLANG = "rust"


class TestRustEvaluation:
    """Test class for Rust evaluation functions"""

    def test_simple_stdin_program(self):
        """Test successful execution"""
        program = """
use std::io;

fn read_and_display_variables() {
    
    let mut input1 = String::new();
    io::stdin()
        .read_line(&mut input1)
        .expect("Error");
    let first_variable = input1.trim();
    
    let mut input2 = String::new();

    io::stdin()
        .read_line(&mut input2)
        .expect("Error");
    let second_variable = input2.trim();
    
    println!("{}", first_variable);
    println!("{}", second_variable);
}

fn main() {
    read_and_display_variables();
}

"""

        input_data = ["0\n0", "55555\n55555"]
        output_data = ["0\n0", "55555\n55555"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, PLANG, TIMEOUT
        )

        assert result == [True, True]
        assert metadata.success is True

    def test_task(self):
        program = """
        
use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut lines = input.lines();
    
    let start: Vec<i32> = lines.next().unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    let d: i32 = lines.next().unwrap().parse().unwrap();
    
    let result = max_score(start, d);
    println!("{}", result);
}

fn max_score(start: Vec<i32>, d: i32) -> i32 {
    let mut intervals = start.into_iter().map(|s| (s, s + d)).collect::<Vec<_>>();
    intervals.sort_unstable();
    
    let mut left = 0;
    let mut right = 1;
    let mut max_score = 0;
    
    while right < intervals.len() {
        let (left_start, left_end) = intervals[left];
        let (right_start, right_end) = intervals[right];
        
        if left_end >= right_start {
            max_score = max_score.max((right_start - left_start).min((left_end - right_start).min((right_end - left_end))));
            left = right;
            right += 1;
        } else {
            max_score = max_score.max(right_start - left_start);
            right += 1;
        }
    }
    
    max_score
}        
        """

        input_data = ["6 0 3\n2"]
        output_data = ["4"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, PLANG, TIMEOUT
        )

        assert result == [TestScore.FAILED]
        assert metadata.error == Status.WrongAnswer


class TestErrorMessages:
    def test_rust_memory_err(self):
        """Test that Pool of workers can execute the problem without errors."""
        exit_code = -1
        stdout = ""
        err_msg = """
        ...
        error: linking with `cc` failed: exit status: 1
        |
        = note:  "cc" "-m64" ...
        = note: some arguments are omitted. use `--verbose` to show all linker arguments
        = note: terminate called after throwing an instance of 'std::system_error'
            what():  Resource temporarily unavailable
          PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
          Stack dump:
          ... 
          error: aborting due to 1 previous error
          """

        result = Result(
            exit_code=exit_code, stdout=stdout, stderr=err_msg, plang="rust"
        )
        status = get_build_status(result)
        assert status == Status.OutOfMemory
