# SWE-bench Train Sample 100

Seeded raw sample of 100 tasks from the Hugging Face dataset [SWE-bench/SWE-bench](https://huggingface.co/datasets/SWE-bench/SWE-bench).

- Source split: `train`
- Source rows: `19008`
- Sample seed: `42`
- Sampling: `random.Random(42).sample(range(19008), 100)`, sorted by source index
- Data: `sample.jsonl`
- Provenance: `manifest.json`

This is a raw reference/prototyping sample, not a held-out evaluation set. It intentionally uses the train split to avoid committing SWE-bench dev/test tasks as training-adjacent data. Before any actual SFT run, convert selected rows into the project chat-message schema and verify source/license policy.
