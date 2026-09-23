#!/usr/bin/env python3
"""Tier 0 for the PPO example_datapoints live batch. Mac, no GPU, no model.

Covers the parts that decide whether the PPO option means what it claims: the guards that refuse a malformed batch, and the rendering that turns a scored batch into stimulus text. The GPU stages in between (merge, generate, score) are exercised for real on the pod, but everything that can be wrong for free is wrong here first.

Run:  python3 rl-rust/tests/tier0_ppo_live_batch.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl-rust"))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.rl_rust.score_ppo_live_batch import load_calibrated_manifest, validate_batch  # noqa: E402
import build_rust_rl_stimuli as builder  # noqa: E402


_FAILURES: list[str] = []
_PASSES = 0


def check(name: str, got, want) -> None:
    global _PASSES
    if got == want:
        _PASSES += 1
        print(f"  PASS  {name}")
    else:
        _FAILURES.append(name)
        print(f"  FAIL  {name}\n        got:  {got!r}\n        want: {want!r}")


def check_raises(name: str, fn, needle: str) -> None:
    global _PASSES
    try:
        fn()
    except SystemExit as exc:
        if needle in str(exc):
            _PASSES += 1
            print(f"  PASS  {name}")
        else:
            _FAILURES.append(name)
            print(f"  FAIL  {name}\n        raised, but message lacked {needle!r}: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        _FAILURES.append(name)
        print(f"  FAIL  {name}\n        wrong exception type: {type(exc).__name__}: {exc}")
        return
    _FAILURES.append(name)
    print(f"  FAIL  {name}\n        did not raise")


PROMPTS = [
    {"record_id": "aaa", "prompt": "problem one"},
    {"record_id": "bbb", "prompt": "problem two"},
]


def completions(**overrides) -> list[dict]:
    rows = [
        {"record_id": "aaa", "text": "fn main() { println!(\"a\"); }", "finish_reason": "stop"},
        {"record_id": "bbb", "text": "fn main() { println!(\"b\"); }", "finish_reason": "stop"},
    ]
    for row in rows:
        row.update(overrides)
    return rows


def live_batch_artifact(size: str = "4b", **row_overrides) -> dict:
    rows = []
    for n, (rid, score) in enumerate((("aaa", 1.2345), ("bbb", -0.5)), start=1):
        row = {
            "record_id": rid,
            "problem": f"problem {n}",
            "rendered_prompt": f"<|im_start|>user\nproblem {n}<|im_end|>\n<|im_start|>assistant\n",
            "completion": f"fn main() {{ println!(\"{rid}\"); }}",
            "rm_score": score,
            "finish_reason": "stop",
        }
        row.update(row_overrides)
        rows.append(row)
    return {
        "schema": "ppo_live_batch_v1",
        "size": size,
        "policy_checkpoint": "prism-drift/qwen35-4b-m0-v4-rust-rl-adapters/ppo.rust.s0/checkpoint-124",
        "policy_dir": "/root/merged-4b-ppo-step124",
        "reward_model": "/root/runs/qwen35-4b-rm-rust-s0-calibrated",
        "reward_model_calibration": {"schema": "score_head_calibration_v1", "sigma": 3.21},
        "reward_model_held_out_pairwise_accuracy": 0.72,
        "source_artifacts": [],
        "rows": rows,
    }


def section(title: str) -> None:
    print(f"\n{title}")


def main() -> int:
    section("A. batch guards (score_ppo_live_batch.validate_batch)")
    ok = validate_batch(PROMPTS, completions(), allow_truncated=False)
    check("a clean batch pairs one completion per pinned id", sorted(ok), ["aaa", "bbb"])

    dupes = completions() + [{"record_id": "aaa", "text": "second", "finish_reason": "stop"}]
    check_raises(
        "two completions for one problem is refused, not silently picked from",
        lambda: validate_batch(PROMPTS, dupes, allow_truncated=False),
        "expected exactly 1",
    )
    check_raises(
        "a pinned problem with no completion is refused",
        lambda: validate_batch(PROMPTS, completions()[:1], allow_truncated=False),
        "no completion for pinned ids",
    )
    check_raises(
        "a truncated completion is refused by default",
        lambda: validate_batch(PROMPTS, completions(finish_reason="length"), allow_truncated=False),
        "hit the length limit",
    )
    kept = validate_batch(PROMPTS, completions(finish_reason="length"), allow_truncated=True)
    check("--allow_truncated keeps it deliberately", sorted(kept), ["aaa", "bbb"])

    section("B. the reward model must be the calibrated one")
    with tempfile.TemporaryDirectory() as td:
        rm = Path(td)
        check_raises(
            "an RM directory with no manifest is refused",
            lambda: load_calibrated_manifest(rm),
            "no reward_model_manifest.json",
        )
        (rm / "reward_model_manifest.json").write_text(json.dumps({"schema": "reward_model_manifest_v1"}))
        check_raises(
            "an UNcalibrated RM is refused: PPO never saw that scale",
            lambda: load_calibrated_manifest(rm),
            "not calibrated",
        )
        (rm / "reward_model_manifest.json").write_text(
            json.dumps({"schema": "reward_model_manifest_v1", "score_head_calibration": {"sigma": 3.2}})
        )
        check("a calibrated RM passes", load_calibrated_manifest(rm)["score_head_calibration"], {"sigma": 3.2})

    section("C. rendering (build_rust_rl_stimuli.ppo_example)")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "4b.scored.json"
        path.write_text(json.dumps(live_batch_artifact()))
        # ppo_example records the artifact path relative to the repo root, so the fixture has to
        # live under it; a temp dir elsewhere would fail on relative_to, not on anything real.
        repo_path = REPO / "rl-rust" / "out" / "ppo-live-batch" / "tier0-fixture.scored.json"
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        repo_path.write_text(json.dumps(live_batch_artifact()))
        try:
            text, prov = builder.ppo_example("4b", ["aaa", "bbb"], repo_path)

            check("renders one block per pinned problem", text.count("Example "), 2)
            check("shows exactly one program per problem (PPO's shape)", text.count("wrote this program"), 2)
            check("says the program was not run", text.count("It was not run"), 2)
            check("quotes the reward-model score to 3dp", "rated it 1.234" in text, True)
            check("keeps a negative score's sign", "rated it -0.500" in text, True)
            check("strips chat markup from the prompt", "<|im_start|>" in text, False)
            check("never claims a test result", "of the tests" in text, False)
            check("provenance marks the source kind honestly", prov["source_kind"], "generated_from_trained_policy")
            check("provenance carries the policy checkpoint", "checkpoint-124" in prov["policy_checkpoint"], True)
            check("provenance carries the RM calibration", prov["reward_model_calibration"]["sigma"], 3.21)
            check("training_coverage records the eight passes", prov["training_coverage"]["passes"], 8)

            # A truncated program must SAY it was cut off; a silently cut-off program in a stimulus
            # reads as a model that writes broken code.
            repo_path.write_text(json.dumps(live_batch_artifact(finish_reason="length")))
            cut_text, _ = builder.ppo_example("4b", ["aaa", "bbb"], repo_path)
            check("a truncated program is labelled as cut off", cut_text.count("cut off before it finished"), 2)

            repo_path.write_text(json.dumps(live_batch_artifact(size="9b")))
            check_raises(
                "a 9B artifact is refused when building 4B",
                lambda: builder.ppo_example("4b", ["aaa", "bbb"], repo_path),
                "built for size",
            )

            bad = live_batch_artifact()
            bad["schema"] = "something_else"
            repo_path.write_text(json.dumps(bad))
            check_raises(
                "an unexpected schema is refused",
                lambda: builder.ppo_example("4b", ["aaa", "bbb"], repo_path),
                "unexpected schema",
            )

            repo_path.write_text(json.dumps(live_batch_artifact()))
            check_raises(
                "an artifact missing a pinned problem is refused, not re-pinned",
                lambda: builder.ppo_example("4b", ["aaa", "zzz"], repo_path),
                "no scored program for pinned ids",
            )
        finally:
            repo_path.unlink(missing_ok=True)

    section("D. the whole 30-item build with PPO filled")
    with tempfile.TemporaryDirectory() as td:
        fixture = REPO / "rl-rust" / "out" / "ppo-live-batch" / "tier0-build.scored.json"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        pinned = builder.shared_record_ids("4b", 0)
        artifact = live_batch_artifact()
        for row, rid in zip(artifact["rows"], pinned):
            row["record_id"] = rid
        fixture.write_text(json.dumps(artifact))
        try:
            items = builder.build("4b", 0, fixture)
            by_id = {item["id"]: item for item in items}
            check("30 items", len(items), 30)
            v2 = {x["id"]: x for x in json.loads((REPO / "data/source/coding_training_stimuli_v2.json").read_text())}
            check("the 27 corpus items are still byte-identical to v2", all(by_id[k] == v2[k] for k in v2), True)
            ppo = by_id["coding.ppo.rust"]
            check("PPO example_datapoints is populated", ppo["stimuli"]["example_datapoints"] is not None, True)
            check("PPO source is no longer blocked", ppo["example_datapoints_source"], "generated_from_trained_policy")
            check(
                "PPO illustrates the same problems as GRPO",
                ppo["example_datapoints_provenance"]["selected_record_ids"],
                by_id["coding.grpo.rust"]["example_datapoints_provenance"]["selected_record_ids"],
            )
            check("the described levels stay non-null", all(ppo["stimuli"][lvl] for lvl in ("described_choice", "described_dataset")), True)
            check("described_dataset states eight passes", "eight passes" in ppo["stimuli"]["described_dataset"], True)

            blocked = builder.build("4b", 0, None)
            blocked_ppo = {item["id"]: item for item in blocked}["coding.ppo.rust"]
            check("without the artifact PPO stays null", blocked_ppo["stimuli"]["example_datapoints"], None)
            check("without the artifact PPO stays blocked", blocked_ppo["example_datapoints_source"], "blocked_pending_training_run")
        finally:
            fixture.unlink(missing_ok=True)

    print(f"\n{_PASSES} passed, {len(_FAILURES)} failed")
    if _FAILURES:
        for name in _FAILURES:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
