# API keys

The scorer reads a hosted model's key from `api_key_<model_type>.txt` in this directory (see `compute_utilities/utils.py`), where `model_type` is the value set for that entry in `config.yaml`. These `api_key_*.txt` files are gitignored, so real keys are never committed.

For our self-hosted vLLM endpoint (`model_type: vllm_endpoint`), the key lives in `api_key_vllm_endpoint.txt` and is whatever you pass to `vllm serve --api-key`. This one file is **intentionally committed** (a `.gitignore` exception) because its value is a non-secret dummy (`dummy-key`) shared by the whole team — it grants nothing, it just has to match `--api-key`. Committing it means it travels with `git clone`, so a freshly-cloned box can score without recreating it. Do **not** put a real secret here. See `docs/serving-vllm-aws.md` for standing up the endpoint.
