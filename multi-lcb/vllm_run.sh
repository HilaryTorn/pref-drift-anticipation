MODEL_TAG="Qwen/Qwen3-32B"

vllm serve "$MODEL_TAG" \
    --host "127.0.0.1" \
    --port 5000 \
    --tensor-parallel-size 8 \
    --enable-prefix-caching \
    --seed 187 \
    --max-model-len 32768
