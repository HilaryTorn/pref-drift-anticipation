
MODEL_TAG="Qwen/Qwen3-4B-Thinking-2507"

python3 -m sglang.launch_server \
    --model-path "$MODEL_TAG" \
    --host 0.0.0.0 \
    --port 5000 \
    --dp 8 \
    --tp-size 1 \
    --context-length 32768 \
    --max-running-requests 100 \
    --nnodes 1