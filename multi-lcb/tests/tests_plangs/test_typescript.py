from pathlib import Path
from lcb_runner.evaluation.testing_plang import (
    eval_plang_code,
    install_npm_packages,
    SubprocessConfig,
)
from dataclasses import asdict
import tempfile

TIMEOUT = 15
PLANG = "typescript"


class TestTypescriptEvaluation:
    """Test class for Typescript evaluation functions."""

    def test_install_npm(self):

        program = """
        import * as readline from 'readline';
        """

        sconf = SubprocessConfig(plang="ts")

        with tempfile.TemporaryDirectory() as tmpdirname:
            sconf.set_cwd(tmpdirname)

            fname = Path(tmpdirname, "Main.js")
            with open(fname, "w", encoding="utf8") as f:
                f.write(program)
                f.flush()

            result = install_npm_packages(sconf)

        assert result.exit_code == 0

    def test_simple_stdin_program(self):
        """Test successful execution"""
        program = """
        
import * as readline from 'readline';

const rl = readline.createInterface({
 input: process.stdin,
 output: process.stdout
});

async function main() {
    const array: string[] = [];
    
    for await (const line of rl) {
        array.push(line);
    }
    
    for (const item of array) {
        console.log(item);
    }
}

main();
"""

        input_data = ["0\n0", "55555\n55555"]
        output_data = ["0\n0", "55555\n55555"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, PLANG, TIMEOUT
        )

        metadata = asdict(metadata)

        assert result == [True, True]
        assert "execution_time" in metadata

    def test_simple_stdin_deno(self):
        """Test successful execution"""
        program = """
import { readLines } from "https://deno.land/std/io/mod.ts";

async function main() {
    const array: string[] = [];
    
    for await (const line of readLines(Deno.stdin)) {
        array.push(line)
    }    
    
    for (const item of array) {
        console.log(item);
    }        
}

main();
"""

        input_data = ["0\n0", "55555\n55555"]
        output_data = ["0\n0", "55555\n55555"]

        result, metadata = eval_plang_code(
            program, input_data, output_data, PLANG, TIMEOUT
        )

        metadata = asdict(metadata)

        assert result == [True, True]
        assert "execution_time" in metadata
