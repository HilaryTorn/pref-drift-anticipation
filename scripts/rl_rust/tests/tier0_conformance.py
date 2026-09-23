"""Tier 0: prove the Rust reward is correct before spending a GPU hour.

Three parts, in increasing order of how much they would cost to discover later:

A. **Conformance** -- every code path against a known expected reward. Correct,
   wrong, compile error, panic, timeout, partial credit, all four comparison
   modes, truncation, extraction. Synthetic, fast, exhaustive. A3/A4 add the
   DPO and PPO launcher plumbing (rebinding, config injection, the PPO rollout
   memory patch, the RM head rescale) on tiny random models, CPU only.

B. **Python parity** -- the new module's Python path must agree with
   ``rl_training.rewards`` on the same inputs. If it does not, the Python and
   Rust arms are not graded by the same definition and the comparison is dead.

C. **Real verified data** -- the 700-row Rust SFT pool holds GPT-5.6 teacher
   solutions that were executable-verified at build time, tagged with their
   LiveCodeBench question id. Joined against the v1-v5 test cases, those are
   known-correct Rust programs with real tests. They must score ~1.0. Anything
   else is the harness being wrong, not the programs.

Run:  .venv/bin/python rl-rust/tests/tier0_conformance.py
"""

from __future__ import annotations

import base64
import json
import random
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-rust"))

from rl_rust import languages  # noqa: E402
from scripts.rl_rust.rl_rust.rewards import (  # noqa: E402
    clear_build_cache,
    coding_reward,
    extract_code_for_language,
    get_language,
    run_io_tests,
    rust_toolchain_available,
)

RUST = get_language("rust")
PY = get_language("python")

_FAILURES: list[str] = []
_PASSES = 0


def check(name: str, got, want, tol: float = 1e-9) -> None:
    global _PASSES
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) and isinstance(got, (int, float)) else (got == want)
    if ok:
        _PASSES += 1
        print(f"  PASS  {name}  (= {got!r})")
    else:
        _FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  got {got!r}, want {want!r}")


def rust(body: str) -> str:
    return f"```rust\n{body}\n```"


# --- A. conformance -----------------------------------------------------------

ECHO = """use std::io::{self, Read};
fn main() {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let n: i64 = s.trim().parse().unwrap();
    println!("{}", n * 2);
}
"""

WRONG = ECHO.replace("n * 2", "n * 3")

PANIC = """fn main() { panic!("boom"); }"""

NO_COMPILE = """fn main() { let x: i32 = "not an int"; }"""

HANG = """fn main() { loop {} }"""

# Passes only when the input is even; used for exact partial credit.
HALF = """use std::io::{self, Read};
fn main() {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let n: i64 = s.trim().parse().unwrap();
    if n % 2 == 0 { println!("{}", n * 2); } else { println!("wrong"); }
}
"""


def part_a() -> None:
    print("\n== A. conformance ==")
    ins = ["1", "2", "3", "4"]
    outs = ["2", "4", "6", "8"]

    check("correct program", run_io_tests(rust(ECHO), ins, outs, timeout=5.0, language="rust"), 1.0)
    check("wrong output", run_io_tests(rust(WRONG), ins, outs, timeout=5.0, language="rust"), 0.0)
    check("compile error", run_io_tests(rust(NO_COMPILE), ins, outs, timeout=5.0, language="rust"), 0.0)
    check("runtime panic", run_io_tests(rust(PANIC), ins, outs, timeout=5.0, language="rust"), 0.0)

    # 2 of 4 inputs are even -> exactly half the tests pass.
    check("partial credit", run_io_tests(rust(HALF), ins, outs, timeout=5.0, language="rust"), 0.5)

    t0 = time.time()
    check("infinite loop times out", run_io_tests(rust(HANG), ["1"], ["2"], timeout=2.0, language="rust"), 0.0)
    elapsed = time.time() - t0
    check("timeout is bounded (<15s)", elapsed < 15.0, True)

    check("empty completion", run_io_tests("", ins, outs, timeout=5.0, language="rust"), 0.0)
    check("prose, no code", run_io_tests("I cannot solve this.", ins, outs, timeout=5.0, language="rust"), 0.0)

    # Comparison modes. The program prints a fixed string; only the mode varies.
    check(
        "case_insensitive_tokens",
        run_io_tests(rust('fn main() { println!("YES"); }'), ["x"], ["yes"], timeout=5.0,
                     language="rust", comparison="case_insensitive_tokens"),
        1.0,
    )
    check(
        "tokens ignores whitespace",
        run_io_tests(rust('fn main() { println!("1   2\\n3"); }'), ["x"], ["1 2 3"], timeout=5.0,
                     language="rust", comparison="tokens"),
        1.0,
    )
    check(
        "exact_lines is strict",
        run_io_tests(rust('fn main() { println!("1 2"); }'), ["x"], ["1  2"], timeout=5.0,
                     language="rust", comparison="exact_lines"),
        0.0,
    )
    check(
        "float_tokens rejects beyond 1e-6",
        run_io_tests(rust('fn main() { println!("1.001"); }'), ["x"], ["1.0"], timeout=5.0,
                     language="rust", comparison="float_tokens"),
        0.0,
    )
    check(
        "float_tokens accepts within 1e-6",
        run_io_tests(rust('fn main() { println!("1.0000000001"); }'), ["x"], ["1.0"], timeout=5.0,
                     language="rust", comparison="float_tokens"),
        1.0,
    )
    check(
        "float_tokens still compares non-numeric tokens",
        run_io_tests(rust('fn main() { println!("No 1.0"); }'), ["x"], ["Yes 1.0"], timeout=5.0,
                     language="rust", comparison="float_tokens"),
        0.0,
    )

    # Extraction: a trailing untagged block must not shadow the tagged program.
    completion = f"Here is my solution:\n```rust\n{ECHO}\n```\nExample output:\n```\n2\n```"
    code, status = extract_code_for_language(completion, RUST)
    check("prefers rust-tagged block", "fn main()" in code, True)
    check("extraction status ok", status, "ok")

    # Truncation contract: a capped rollout is None, not 0.0.
    rewards = coding_reward(
        [rust(ECHO)],
        verifier=[{"type": "io_tests", "test_inputs": ins, "test_outputs": outs}],
        completion_ids=[[1, 2, 3]],
        eos_token_ids=[999],
        timeout=5.0,
        language="rust",
    )
    check("truncated rollout -> None", rewards[0], None)

    rewards = coding_reward(
        [rust(ECHO)],
        verifier=[{"type": "io_tests", "test_inputs": ins, "test_outputs": outs}],
        completion_ids=[[1, 2, 999]],
        eos_token_ids=[999],
        timeout=5.0,
        language="rust",
    )
    check("terminated rollout -> 1.0", rewards[0], 1.0)


def part_a2_build_once() -> None:
    """The performance property: one rustc call per program, not per test case."""
    print("\n== A2. compile-once ==")
    clear_build_cache()
    calls = {"n": 0}
    original = languages._build_rust

    def counting(source, workdir):
        calls["n"] += 1
        return original(source, workdir)

    languages._build_rust = counting
    try:
        ins = [str(i) for i in range(1, 13)]
        outs = [str(2 * i) for i in range(1, 13)]
        t0 = time.time()
        reward = run_io_tests(rust(ECHO), ins, outs, timeout=5.0, language="rust")
        elapsed = time.time() - t0
        check("12 test cases still correct", reward, 1.0)
        check("rustc invoked exactly once for 12 tests", calls["n"], 1)

        # A GRPO group repeats the same program; the cache must span completions.
        before = calls["n"]
        coding_reward(
            [rust(ECHO)] * 8,
            verifier=[{"type": "io_tests", "test_inputs": ins, "test_outputs": outs}] * 8,
            timeout=5.0,
            language="rust",
        )
        check("identical group of 8 compiles 0 more times", calls["n"] - before, 0)
        print(f"  note: 12-test grade took {elapsed:.2f}s wall")
    finally:
        languages._build_rust = original


def part_a3_dpo_candidates() -> None:
    """The DPO pair path: truncation-aware scoring and the launcher rebinding.

    The Rust DPO arm builds its pairs through rl_training/build_dpo_pairs.py
    with ``score_candidates`` rebound to the Rust one. This proves (1) the Rust
    ``score_candidates`` honors the truncation contract, (2) pair selection
    excludes withheld candidates, and (3) the rebinding in
    build_rust_dpo_pairs.py actually redirects the attribute build_dpo_pairs
    resolves -- the failure mode where it doesn't is a sweep that grades Rust as
    Python and completes normally with zero pairs.
    """
    print("\n== A3. DPO candidate scoring ==")
    from rl_training.rewards import build_preference_pair_from_candidates
    from scripts.rl_rust.rl_rust.rewards import score_candidates

    ins = ["1", "2", "3", "4"]
    outs = ["2", "4", "6", "8"]
    verifier = {"type": "io_tests", "test_inputs": ins, "test_outputs": outs}

    scored = score_candidates(
        [rust(ECHO), rust(WRONG), rust(ECHO), "", "no code"],
        verifier,
        timeout=5.0,
        finish_reasons=["stop", "stop", "length", "stop", "stop"],
        language="rust",
    )
    check("correct candidate scores 1.0", scored[0]["reward"], 1.0)
    check("wrong candidate scores 0.0", scored[1]["reward"], 0.0)
    check("truncated candidate withheld (None)", scored[2]["reward"] is None, True)
    check("truncated status recorded", scored[2]["status"], "truncated")
    check("empty candidate withheld (None)", scored[3]["reward"] is None, True)

    pair, audit = build_preference_pair_from_candidates("p", scored, margin=0.5)
    check("pair built from finished candidates only", pair is not None, True)
    if pair:
        check("chosen is the correct program", pair["chosen_reward"], 1.0)
        check("rejected is the wrong one, not the truncated one",
              pair["rejected"], scored[1]["text"])
    check("audit counts one truncated", audit["n_truncated"], 1)

    # Python parity for the candidate path, same standard as part B.
    from rl_training import rewards as upstream

    py_ok = "```python\nimport sys\nn=int(sys.stdin.read())\nprint(n*2)\n```"
    py_wrong = "```python\nimport sys\nn=int(sys.stdin.read())\nprint(n*3)\n```"
    mine = score_candidates(
        [py_ok, py_wrong], verifier, timeout=5.0,
        finish_reasons=["stop", "length"], language="python",
    )
    theirs = upstream.score_candidates(
        [py_ok, py_wrong], verifier, timeout=5.0, finish_reasons=["stop", "length"],
    )
    check("python parity: rewards", [c["reward"] for c in mine], [c["reward"] for c in theirs])
    check("python parity: statuses", [c["status"] for c in mine], [c["status"] for c in theirs])

    # The launcher rebinding, end to end, then restored so part B still tests
    # the pristine upstream module.
    import scripts.rl_rust.build_rust_dpo_pairs as build_rust_dpo_pairs
    from rl_training import prompts as up_prompts

    original_scorer = upstream.score_candidates
    original_renderer = up_prompts.render_coding_prompt
    try:
        build_rust_dpo_pairs.install_scoring("rust", "rust_io")
        rebound = upstream.score_candidates(
            [rust(ECHO)], verifier, timeout=5.0, finish_reasons=["stop"]
        )
        check("rebound upstream score_candidates grades Rust", rebound[0]["reward"], 1.0)
        rendered = up_prompts.render_coding_prompt(None, "A+B")
        check("rebound renderer asks for Rust", "Rust" in rendered, True)
    finally:
        upstream.score_candidates = original_scorer
        up_prompts.render_coding_prompt = original_renderer


def part_a4_ppo_plumbing() -> None:
    """The PPO arm's three launcher patches, on a tiny random model, CPU only.

    (1) The chunk-wise rollout replacement must give the same query_responses
    and the same log-probs as TRL's stock batch_generation + the trainer's own
    selective_log_softmax_token_chunks -- it exists only to stop TRL stacking
    ~80 GB of float32 logits for a 64 x 2048 rollout batch. (2) The PPOConfig
    wrapper must actually deliver the YAML extras train_ppo does not forward.
    (3) Rescaling the RM's bias-free score head by a positive scalar must leave
    every pairwise ranking unchanged -- the property calibrate_reward_head.py
    relies on. All patches are restored afterwards.
    """
    print("\n== A4. PPO plumbing (rollout memory patch, config injection, head rescale) ==")
    import torch
    import yaml
    from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM, Qwen2ForSequenceClassification

    import scripts.rl_rust.train_rust_ppo as train_rust_ppo
    from rl_training import train_rl
    from rl_training import masked_ppo_trainer as masked

    train_rl._patch_trl_optional_imports()
    ppo_impl = masked.ppo_impl
    stock_batch_generation = ppo_impl.batch_generation
    stock_selective = masked.selective_log_softmax_token_chunks

    torch.manual_seed(0)
    pad, vocab = 0, 97
    lm = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        pad_token_id=pad, eos_token_id=1, bos_token_id=2,
    )).eval()
    batch, context = 11, 7  # not a multiple of the chunk (4): exercises the ragged last chunk
    queries = torch.randint(3, vocab, (batch, context))
    queries[0, :3] = pad  # left-padded row, as TRL's collator produces
    # Many EOS ids so chunks stop at different lengths and the padding path is real.
    gen = GenerationConfig(max_new_tokens=9, temperature=1.0 + 1e-7, top_k=0.0, top_p=1.0,
                           do_sample=True, eos_token_id=list(range(3, 60)), pad_token_id=pad)

    def rollout_logprobs(batch_generation_fn, selective_fn):
        torch.manual_seed(123)
        query_responses, store = batch_generation_fn(lm, queries, 4, pad, gen)
        parts = []
        for i in range(0, batch, 4):
            response = query_responses[i : i + 4, context:]
            parts.append(selective_fn(store[i : i + 4], response))
        return query_responses, torch.cat(parts, 0)

    try:
        qr_stock, lp_stock = rollout_logprobs(stock_batch_generation, stock_selective)
        # gen chunk == the trainer chunk (4): generate calls line up one-to-one
        # with stock's, so RNG consumption -- hence every sampled token -- is
        # identical, and the log-probs must match up to recomputation numerics
        # (KV-cache incremental forward vs one full forward; fp32 tiny model).
        train_rust_ppo.install_rollout_logprob_fix({"generation_chunk_size": 4})
        train_rust_ppo.install_rollout_logprob_fix({"generation_chunk_size": 4})  # idempotent no-op
        check("rollout patch installed", ppo_impl.batch_generation is not stock_batch_generation, True)
        qr_new, lp_new = rollout_logprobs(ppo_impl.batch_generation, masked.selective_log_softmax_token_chunks)
        check("patched rollout: same query_responses (no-scores generation samples identically)",
              torch.equal(qr_stock, qr_new), True)
        # Stock pads a short chunk's logits with zeros, giving a garbage logprob
        # there that the trainer masks later; compare the real positions only.
        real = (qr_stock[:, context:] != pad)
        check("patched rollout: logprobs match stock on real positions (forward == generation scores)",
              float((lp_stock - lp_new).abs()[real].max()), 0.0, tol=1e-4)
        check("padding exercised (some chunk stopped early)", bool((~real).any()), True)

        # Whole-batch generation (generation_chunk_size 0, the production
        # default). Sampled tokens legitimately differ from stock -- one big
        # generate draws the RNG in a different order -- so the check is
        # against ground truth instead: recompute the log-probs of the RETURNED
        # sequences with one direct full-batch forward and the ref-path recipe,
        # and the store must agree. Verifies the slicing, the temperature
        # scalar, and the sub-chunked recovery loop.
        ppo_impl.batch_generation = stock_batch_generation  # clear the patch flag path
        masked.selective_log_softmax_token_chunks = stock_selective
        train_rust_ppo.install_rollout_logprob_fix({"generation_chunk_size": 0})
        torch.manual_seed(321)
        qr_wb, store_wb = ppo_impl.batch_generation(lm, queries, 4, pad, gen)
        with torch.no_grad():
            ref_out = ppo_impl.forward(lm, qr_wb, pad)
        ref_logits = ref_out.logits[:, context - 1 : -1] / float(gen.temperature)
        lp_truth = masked.selective_log_softmax(ref_logits, qr_wb[:, context:]) \
            if hasattr(masked, "selective_log_softmax") else stock_selective(ref_logits, qr_wb[:, context:])
        lp_wb = torch.cat([masked.selective_log_softmax_token_chunks(store_wb[i : i + 4], qr_wb[i : i + 4, context:])
                           for i in range(0, batch, 4)], 0)
        real_wb = (qr_wb[:, context:] != pad)
        check("whole-batch generation: store matches direct-forward ground truth",
              float((lp_truth - lp_wb).abs()[real_wb].max()), 0.0, tol=1e-4)
        check("whole-batch generation: shapes preserved",
              tuple(qr_wb.shape[:1]) == (batch,) and lp_wb.shape[0] == batch, True)
        logits = torch.randn(2, 5, vocab)
        labels = torch.randint(0, vocab, (2, 5))
        check("real logits still route to the original",
              torch.allclose(masked.selective_log_softmax_token_chunks(logits, labels),
                             stock_selective(logits, labels)), True)

        cfg = yaml.safe_load((REPO / "rl-rust" / "configs" / "ppo_rust.yaml").read_text())
        import trl.experimental.ppo as ppo_module
        stock_config = ppo_module.PPOConfig
        extras = train_rust_ppo.install_ppo_config_extras(cfg)
        built = ppo_module.PPOConfig(output_dir=str(REPO / "rl-rust" / "out" / "_tier0_ppo"),
                                     report_to="none", bf16=False)
        check("PPOConfig receives temperature from YAML", built.temperature, cfg["temperature"])
        check("PPOConfig receives local_rollout_forward_batch_size",
              built.local_rollout_forward_batch_size, cfg["local_rollout_forward_batch_size"])
        check("every YAML extra landed", all(getattr(built, k) == v for k, v in extras.items()), True)
        check("forwarded keys are not double-injected",
              set(extras) & set(train_rust_ppo._PPO_CONFIG_FORWARDED_KEYS), set())
        ppo_module.PPOConfig = stock_config

        torch.manual_seed(1)
        rm = Qwen2ForSequenceClassification(Qwen2Config(
            vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
            num_labels=1, pad_token_id=pad,
        )).eval()
        check("Qwen seq-classification head is bias-free (mean cannot be shifted; scale only)",
              rm.score.bias is None, True)
        seqs = torch.randint(3, vocab, (20, 12))
        with torch.no_grad():
            before = rm(input_ids=seqs).logits.squeeze(-1)
            rm.score.weight.mul_(1.0 / before.std())
            after = rm(input_ids=seqs).logits.squeeze(-1)
        chosen, rejected = before[:10], before[10:]
        check("head rescale keeps every pairwise ranking",
              torch.equal(chosen > rejected, after[:10] > after[10:]), True)
        check("head rescale gives unit spread", float(after.std()), 1.0, tol=1e-4)
    finally:
        ppo_impl.batch_generation = stock_batch_generation
        masked.selective_log_softmax_token_chunks = stock_selective


# --- B. python parity ---------------------------------------------------------


def part_b() -> None:
    print("\n== B. python-path parity vs rl_training.rewards ==")
    from rl_training import rewards as upstream

    py_ok = "```python\nimport sys\nn=int(sys.stdin.read())\nprint(n*2)\n```"
    py_wrong = "```python\nimport sys\nn=int(sys.stdin.read())\nprint(n*3)\n```"
    py_broken = "```python\ndef f(:\n```"
    ins = ["1", "2", "3", "4"]
    outs = ["2", "4", "6", "8"]

    cases = [("correct", py_ok), ("wrong", py_wrong), ("syntax error", py_broken), ("prose", "no code here")]
    for name, completion in cases:
        mine = run_io_tests(completion, ins, outs, timeout=5.0, language="python")
        theirs = upstream.run_io_tests(
            upstream.extract_code(completion), ins, outs, timeout=5.0
        )
        check(f"python parity: {name}", mine, theirs)


# --- C. real verified rust ----------------------------------------------------


def _load_lcb_tests() -> dict[str, list[tuple[str, str]]]:
    """question_id -> [(stdin, expected_stdout)] for stdin-style LCB problems."""
    index: dict[str, list[tuple[str, str]]] = {}
    root = REPO / "data/reference/multilcb/release_v5"
    for path in sorted(root.glob("test*.jsonl")):
        for line in open(path):
            row = json.loads(line)
            cases = []
            try:
                public = json.loads(row.get("public_test_cases") or "[]")
            except json.JSONDecodeError:
                public = []
            private = []
            raw = row.get("private_test_cases")
            if raw:
                try:
                    private = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        private = json.loads(
                            zlib.decompress(base64.b64decode(raw.encode())).decode()
                        )
                    except Exception:
                        private = []
            for case in list(public) + list(private):
                if not isinstance(case, dict):
                    continue
                if case.get("testtype") != "stdin":
                    continue
                cases.append((case.get("input", ""), case.get("output", "")))
            if cases:
                index[row["question_id"]] = cases
    return index


def part_c(sample: int = 40, seed: int = 0) -> None:
    print("\n== C. real verified Rust teacher solutions vs real LCB tests ==")
    pool_path = REPO / "results/sft_datasets/coding.write.rust_lcb_n700_seed42/train.jsonl"
    if not pool_path.is_file():
        print("  SKIP  Rust SFT pool not found")
        return

    tests = _load_lcb_tests()
    print(f"  indexed {len(tests)} stdin-style LCB problems with test cases")

    rows = []
    for line in open(pool_path):
        row = json.loads(line)
        qid = row.get("lcb_question_id")
        if qid in tests:
            solution = row["messages"][-1]["content"]
            rows.append((qid, solution, tests[qid]))

    print(f"  {len(rows)} of 700 teacher solutions joined to a stdin test suite")
    if not rows:
        print("  SKIP  no join (LCB pool problems may be leetcode-style only)")
        return

    random.Random(seed).shuffle(rows)
    rows = rows[:sample]

    t0 = time.time()
    scores = []
    for qid, solution, cases in rows:
        ins = [c[0] for c in cases]
        outs = [c[1] for c in cases]
        r = run_io_tests(solution, ins, outs, timeout=5.0, max_tests=12, language="rust")
        scores.append((qid, r))
    elapsed = time.time() - t0

    perfect = sum(1 for _, r in scores if r >= 0.999)
    zero = [q for q, r in scores if r <= 0.001]
    mean = sum(r for _, r in scores) / len(scores)

    print(f"  graded {len(scores)} verified solutions in {elapsed:.1f}s "
          f"({elapsed/len(scores):.2f}s each)")
    print(f"  mean reward {mean:.3f}   fully passing {perfect}/{len(scores)}")
    if zero:
        print(f"  scored 0.0: {zero[:10]}")

    # These solutions were executable-verified when the pool was built, so a
    # correct harness should pass the large majority. A low rate here means the
    # harness is broken -- do not proceed to a GPU run.
    check("mean reward on verified solutions >= 0.80", mean >= 0.80, True)
    check("majority fully pass", perfect >= 0.7 * len(scores), True)


def main() -> int:
    if not rust_toolchain_available():
        print("rustc not found on PATH -- cannot run tier 0")
        return 2
    print(f"rustc: {languages.shutil.which('rustc')}")
    part_a()
    part_a2_build_once()
    part_a3_dpo_candidates()
    part_a4_ppo_plumbing()
    part_b()
    part_c()
    clear_build_cache()

    print(f"\n{'=' * 60}")
    print(f"passed {_PASSES}, failed {len(_FAILURES)}")
    for failure in _FAILURES:
        print(f"  FAIL  {failure}")
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
