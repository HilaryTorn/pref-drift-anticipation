# Start SGlang/VLLM as a separate process
# SGlang/VLLM server host and port
export OPENAI_BASE_URL=http://localhost:5000/v1
export OPENAI_KEY=dummy

# sglang model name
MODEL_TAG="Qwen/Qwen3-4B-Thinking-2507"

start_date="2024-07-01" # most used default start_date

temp=0.2 # 0.2 is default LCB 
top_p=0.95 # 0.95 is default LCB 
num=10  # 10 is default LCB 

## --n=10  (use n=10 for non-reasoning and n=1 for reasoning models)
## --batch_size=50  (async queue depth)
## --cot_code_execution  (turn on reasoning, better use with n=1)
## --openai_timeout=1200  (timeout for async request client)
python -m lcb_runner.runner.main \
    --model "VLLMAsync" \
    --local_model_path ${MODEL_TAG} \
    --max_tokens 30000 \
    --continue_existing \
    --continue_existing_eval \
    --max_seq_length 32768 \
    --evaluate \
    --temperature ${temp} \
    --top_p ${top_p} \
    --n ${num} \
    --start_date $start_date \
    --batch_size 40 \
    --openai_timeout 1200 \
    --plangs "all" \
    --num_process_evaluate 60 \
    --cot_code_execution 