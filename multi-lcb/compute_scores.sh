MODEL_TAG="Qwen/Qwen2.5-Coder-3B-Instruct"
temp=0.2 
top_p=0.95
num=10

# Use the same parameters that were used for generation and evaluation
python -m lcb_runner.evaluation.compute_scores \
    --model "VLLMAsync" \
    --local_model_path ${MODEL_TAG} \
    --cot_code_execution \
    --temperature ${temp} \
    --top_p ${top_p} \
    --n ${num} \
    --start_date "2024-10-01" \
    --end_date "2025-05-01" \
    --eval_all_file "final_scores.csv" \
    --platform "leetcode"
