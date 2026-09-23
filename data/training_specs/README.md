# Training Specs

Training specs define reproducible intervention conditions for SFT. They do not
claim that data already exists. A spec can be:

- `planned`: design is defined, but real candidate pools may still be absent.
- `ready`: required pool and validation files exist and pass validation.
- `retired`: kept for provenance but not used.

Validate specs and available data with:

```bash
python3 scripts/validate_sft_data.py --spec_path data/training_specs/coding_axis_sft.json
```

When data exists, require it:

```bash
python3 scripts/validate_sft_data.py \
  --spec_path data/training_specs/coding_axis_sft.json \
  --require-data
```

Prepare an exact sampled training subset:

```bash
python3 scripts/prepare_sft_dataset.py \
  --spec_path data/training_specs/coding_axis_sft.json \
  --intervention_id coding.write.python \
  --sample_count 1000 \
  --seed 42 \
  --output_dir results/sft_datasets/coding.write.python_n1000_seed42
```

Train a LoRA adapter from that prepared subset:

```bash
python3 scripts/train_sft_lora.py \
  --spec_path data/training_specs/coding_axis_sft.json \
  --intervention_id coding.write.python \
  --dataset_dir results/sft_datasets/coding.write.python_n1000_seed42 \
  --output_dir results/sft_runs/coding.write.python_n1000_seed42
```

The current spec default is `Qwen/Qwen3.5-0.8B`, with bf16 enabled for the
AWS/L4 path. Use `--no-bf16` for local smoke tests on hardware without bfloat16
support. Use `uv sync`, `pyproject.toml`, or `requirements-sft.txt` for pinned
SFT dependencies. The checked-in training stack must pass
`scripts/preflight_qwen35.py` before spending GPU time.
