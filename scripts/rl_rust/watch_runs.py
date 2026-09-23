"""Print the Rust GRPO learning curve from both pods.

The kill/continue decision rests on three columns, in this order:

  zero_std (frac_reward_zero_std)
      GRPO learns from disagreement WITHIN a group of 8. If this is near 1.0,
      almost every group scored identically -- usually all-zero -- so the
      advantage is 0 and there is NO gradient. The run cannot learn regardless
      of how long it goes. This is the kill signal and it shows within ~10 steps.

  reward
      What we came for. Noisy step to step; read the trend across blocks.

  kl
      Whether the policy is moving at all. The flat Python arms sat at ~1e-3.
      Rising is good. The phase matters: >0.05 in the first ~20 steps is
      overshoot (drop LR to 5e-6), but the run being copied ENDED at kl 0.0734,
      so 0.05-0.08 late is what winning looked like. Do not kill a late run for
      passing 0.05.

Also: clip (truncation) and s/step (the budget).

REFERENCE -- final metrics of the successful truncfix 9B run, so these columns
have something real to be compared against rather than invented thresholds:

    kl 0.0734 | reward 0.4687 | reward_std 0.4559
    frac_reward_zero_std 0.175 | clipped_ratio 0.3375

zero_std should FALL toward ~0.2 as the model improves; clip ~0.34 is normal.

Usage: .venv/bin/python rl-rust/watch_runs.py [n_recent]
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys

PODS = [("9B  (A100)", "64.247.196.119", 10329), ("4B  (A40)", "69.30.85.25", 22071)]
METRIC = re.compile(r"\{[^{}]*'reward'[^{}]*\}")

REMOTE = r"""
if [ -f /root/train.done ]; then echo "STATE|finished $(cat /root/train.done)";
elif pgrep -f train_rust_grpo.py >/dev/null; then echo "STATE|running $(ps -o etime= -C python|head -1|tr -d ' ')";
else echo "STATE|NOT RUNNING"; fi
echo "GPU|$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
echo "ERRS|$(grep -ciE 'traceback|out of memory|runtimeerror' /root/train.log 2>/dev/null || echo 0)"
echo "CKPT|$(ls -d /root/runs/*/checkpoint-* 2>/dev/null | wc -l)"
echo '---METRICS---'
# Prefer trainer_state.json: stdout is BLOCK-BUFFERED when redirected to a file,
# so the metric dicts lag the (stderr) progress bar by several steps. The
# checkpoint's log_history is written directly and is always current.
python3 -c "
import json,glob,sys
cks=sorted(glob.glob('/root/runs/*/checkpoint-*/trainer_state.json'), key=lambda p:int(p.split('checkpoint-')[1].split('/')[0]))
if cks:
    for e in json.load(open(cks[-1]))['log_history']:
        if 'reward' in e: print(json.dumps(e))
" 2>/dev/null || tr '\r' '\n' < /root/train.log | grep -oE "\{[^{}]*'reward'[^{}]*\}" | tail -60
echo '---EVAL---'
cat /root/runs/*/reward_log.jsonl 2>/dev/null | tail -5
"""

COLS = [
    ("step", "step", "{:.0f}"),
    ("reward", "reward", "{:.4f}"),
    ("rw_std", "reward_std", "{:.4f}"),
    ("zero_std", "frac_reward_zero_std", "{:.3f}"),
    ("kl", "kl", "{:.5f}"),
    ("clip", "completions/clipped_ratio", "{:.3f}"),
    ("len", "completions/mean_length", "{:.0f}"),
    ("s/step", "step_time", "{:.0f}"),
]


def fetch(host: str, port: int) -> str:
    try:
        r = subprocess.run(
            ["ssh", "-n", "-o", "ConnectTimeout=25", "-o", "StrictHostKeyChecking=accept-new",
             "-p", str(port), f"root@{host}", REMOTE],
            capture_output=True, text=True, timeout=180,
        )
        return r.stdout
    except Exception as exc:
        return f"STATE|UNREACHABLE ({type(exc).__name__})\n"


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    for label, host, port in PODS:
        out = fetch(host, port)
        print(f"=============== {label} ===============")
        head, _, rest = out.partition("---METRICS---")
        metrics_txt, _, eval_txt = rest.partition("---EVAL---")
        for line in head.strip().splitlines():
            k, _, v = line.partition("|")
            if k == "ERRS" and v.strip() not in ("0", ""):
                print(f"  !! ERRORS IN LOG: {v}")
            elif k in ("STATE", "GPU", "CKPT"):
                print(f"  {k:6s} {v}")

        rows = []
        for line in metrics_txt.strip().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))      # trainer_state path
            except Exception:
                try:
                    rows.append(ast.literal_eval(line))   # stdout fallback
                except Exception:
                    pass
        rows = rows[-n:]
        if not rows:
            print("  (no step metrics logged yet)\n")
            continue
        print("  " + "".join(f"{h:>10s}" for h, _, _ in COLS))
        for r in rows:
            cells = []
            for _, key, fmt in COLS:
                # TRL logs every value as a STRING ('reward': '0.08575'), so these
                # must be coerced -- an isinstance(float) check silently blanks
                # the entire table. There is also no 'step' key; step = epoch*124.
                v = r.get(key)
                if key == "step" and v is None:
                    try: v = round(float(r.get("epoch", 0)) * 124)
                    except Exception: v = None
                try: v = float(v)
                except (TypeError, ValueError): v = None
                cells.append(fmt.format(v) if v is not None else "-")
            print("  " + "".join(f"{c:>10s}" for c in cells))

        if len(rows) >= 4:
            half = len(rows) // 2
            f = lambda rr: sum(float(x.get("reward", 0) or 0) for x in rr) / max(1, len(rr))
            a, b = f(rows[:half]), f(rows[half:])
            print(f"  reward: first half {a:.4f} -> last half {b:.4f}  ({b - a:+.4f})")

        ev = [l for l in eval_txt.strip().splitlines() if l.strip().startswith("{")]
        if ev:
            print("  held-out:")
            for l in ev:
                print(f"    {l}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
