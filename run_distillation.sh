#!/bin/bash

set -e

format_duration() {
  local total_seconds=$1
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))

  printf "%02dh:%02dm:%02ds" "$hours" "$minutes" "$seconds"
}

run_step() {
  local label="$1"
  local success_message="$2"
  shift 2

  local start_ts
  local end_ts
  local elapsed

  start_ts=$(date +%s)

  echo ""
  echo "[$label] started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

  "$@"

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))

  echo "[$label] ${success_message}"
  echo "[$label] finished at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[$label] elapsed: $(format_duration "$elapsed")"
}

pipeline_start_ts=$(date +%s)

cd /workspace
mkdir -p outputs

echo "== COL772 A4 pipeline =="
echo "started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

# part A: generate teacher reasoning traces
run_step \
  "Part A" \
  "done. saved to outputs/train.jsonl" \
  python3 src/dataset_generation.py \
    --teacher_model /models/Qwen2.5-7B-Instruct \
    --output_file outputs/train.jsonl \
    --top_k_logits 10

# carve out a small val set for monitoring training
run_step \
  "Val" \
  "done. saved to outputs/val.jsonl" \
  python3 src/make_val_set.py \
    --train_data outputs/train.jsonl \
    --output_file outputs/val.jsonl \
    --num_samples 300,200,150,100,100

# part B: train qwen student (in-family, uses kd loss)
run_step \
  "Part B / Qwen" \
  "qwen training done. adapter at outputs/qwen_lora" \
  python3 src/train_distill.py \
    --student_model /models/Qwen2.5-1.5B-Instruct \
    --train_data outputs/train.jsonl \
    --output_dir outputs/qwen_lora \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --epochs 5 \
    --lr 2e-4 \
    --mask_prompt_tokens \
    --top_k_logits 10 \
    --kd_alpha 0.5 \
    --kd_temperature 2.0

# part B: train llama student (cross-family, sft only)
run_step \
  "Part B / LLaMA" \
  "llama training done. adapter at outputs/llama_lora" \
  python3 src/train_distill.py \
    --student_model /models/Llama-3.2-1B-Instruct \
    --train_data outputs/train.jsonl \
    --output_dir outputs/llama_lora \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --epochs 5 \
    --lr 2e-4 \
    --mask_prompt_tokens

# part C: eval qwen
run_step \
  "Part C / Qwen Eval" \
  "qwen eval done. report at outputs/metrics_Qwen2.5-1.5B-Instruct.txt" \
  python3 src/inference_eval.py \
    --base_model /models/Qwen2.5-1.5B-Instruct \
    --adapter_path outputs/qwen_lora \
    --test_data outputs/val.jsonl \
    --output_predictions outputs/predictions_Qwen2.5-1.5B-Instruct.jsonl \
    --report_file outputs/metrics_Qwen2.5-1.5B-Instruct.txt \
    --max_new_tokens 2048 \
    --batch_size 16 \
    --gpu_memory_utilization 0.85

# part C: eval llama
run_step \
  "Part C / LLaMA Eval" \
  "llama eval done. report at outputs/metrics_Llama-3.2-1B-Instruct.txt" \
  python3 src/inference_eval.py \
    --base_model /models/Llama-3.2-1B-Instruct \
    --adapter_path outputs/llama_lora \
    --test_data outputs/val.jsonl \
    --output_predictions outputs/predictions_Llama-3.2-1B-Instruct.jsonl \
    --report_file outputs/metrics_Llama-3.2-1B-Instruct.txt \
    --max_new_tokens 2048 \
    --batch_size 16 \
    --gpu_memory_utilization 0.85

pipeline_end_ts=$(date +%s)
pipeline_elapsed=$((pipeline_end_ts - pipeline_start_ts))

echo ""
echo "== pipeline done =="
echo "finished at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "total elapsed: $(format_duration "$pipeline_elapsed")"
