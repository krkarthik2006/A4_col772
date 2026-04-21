#!/bin/bash

set -e

cd /workspace
mkdir -p outputs

echo "======================================================"
echo "  COL772 A4 — Multilingual Knowledge Distillation"
echo "======================================================"
date

# ── Part A: Dataset generation ─────────────────────────────────────────────
echo ""
echo "[Part A] Generating teacher distillation data..."
python3 src/dataset_generation.py \
  --teacher_model /models/Qwen2.5-7B-Instruct \
  --output_file outputs/train.jsonl \
  --top_k_logits 10
echo "[Part A] Done. Saved to outputs/train.jsonl"

# ── Val set creation ────────────────────────────────────────────────────────
echo ""
echo "[Val]   Creating held-out validation set..."
python3 src/make_val_set.py \
  --train_data outputs/train.jsonl \
  --output_file outputs/val.jsonl \
  --num_samples 300,200,150,100,100
echo "[Val]   Done. Saved to outputs/val.jsonl"

# ── Part B: Train Qwen student ──────────────────────────────────────────────
echo ""
echo "[Part B] Training in-family student: Qwen2.5-1.5B-Instruct..."
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
echo "[Part B] Qwen training done. Adapter at outputs/qwen_lora"

# ── Part B: Train LLaMA student ─────────────────────────────────────────────
echo ""
echo "[Part B] Training cross-family student: Llama-3.2-1B-Instruct..."
python3 src/train_distill.py \
  --student_model /models/Llama-3.2-1B-Instruct \
  --train_data outputs/train.jsonl \
  --output_dir outputs/llama_lora \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --epochs 5 \
  --lr 2e-4 \
  --mask_prompt_tokens
echo "[Part B] LLaMA training done. Adapter at outputs/llama_lora"

# ── Part C: Evaluate Qwen student ───────────────────────────────────────────
echo ""
echo "[Part C] Inference + evaluation: Qwen2.5-1.5B-Instruct..."
python3 src/inference_eval.py \
  --base_model /models/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/qwen_lora \
  --test_data outputs/val.jsonl \
  --output_predictions outputs/predictions_Qwen2.5-1.5B-Instruct.jsonl \
  --report_file outputs/metrics_Qwen2.5-1.5B-Instruct.txt \
  --max_new_tokens 2048 \
  --batch_size 16 \
  --gpu_memory_utilization 0.85
echo "[Part C] Qwen eval done. Report at outputs/metrics_Qwen2.5-1.5B-Instruct.txt"

# ── Part C: Evaluate LLaMA student ──────────────────────────────────────────
echo ""
echo "[Part C] Inference + evaluation: Llama-3.2-1B-Instruct..."
python3 src/inference_eval.py \
  --base_model /models/Llama-3.2-1B-Instruct \
  --adapter_path outputs/llama_lora \
  --test_data outputs/val.jsonl \
  --output_predictions outputs/predictions_Llama-3.2-1B-Instruct.jsonl \
  --report_file outputs/metrics_Llama-3.2-1B-Instruct.txt \
  --max_new_tokens 2048 \
  --batch_size 16 \
  --gpu_memory_utilization 0.85
echo "[Part C] LLaMA eval done. Report at outputs/metrics_Llama-3.2-1B-Instruct.txt"

echo ""
echo "======================================================"
echo "  Pipeline complete!"
date
echo "======================================================"
