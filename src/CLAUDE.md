# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COL772 Assignment 4 — Knowledge Distillation for a multilingual MCQ benchmark (MMLU Pro) covering English, Hindi, Bengali, Kannada, and Tamil. The pipeline has three parts:

- **Part A** (`dataset_generation.py`) — **Complete**: Uses a teacher LLM (via vLLM) to generate reasoning traces for sampled MCQ data, saving structured JSONL for distillation.
- **Part B** (`train_distill.py`) — **Stub**: Student model fine-tuning via CoT/SFT distillation; argument parsing exists, training loop is empty.
- **Part C** (`inference_eval.py`) — **Stub**: Inference and per-language accuracy evaluation; argument parsing exists, logic is empty.

## Models

- **Teacher**: `Qwen/Qwen2.5-7B-Instruct`
- **Student (in-family)**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Student (cross-family)**: `meta-llama/Llama-3.2-1B-Instruct`

Cross-family distillation (Llama ↔ Qwen) cannot use logit-level KD due to different vocabularies — use text-based CoT SFT only.

## Allowed Libraries (grader constraint)

`transformers`, `datasets`, `vllm`, `peft`, `trl`, `torch`, `tqdm`, `pandas`, `numpy`, and Python standard library. Any additional package requires Piazza approval.

## Evaluation Environment

NVIDIA V100, Python 3.10. Time limits: dataset generation 240 min, distillation 240 min, evaluation 60 min.

## Running the Code

### Part A: Dataset Generation
```bash
python dataset_generation.py \
  --teacher_model Qwen/Qwen2.5-7B-Instruct \
  --num_samples 3000,2200,1800,1600,1400 \
  --output_file data/train.jsonl \
  --batch_size 32 \
  --max_new_tokens 1024 \
  --filter_incorrect \
  --skip_unparsed
```
`--num_samples` is comma-separated counts for `[english, hindi, bengali, kannada, tamil]`; total must be ≤ 10,000.

### Part B: Training
```bash
python train_distill.py \
  --student_model Qwen/Qwen2.5-1.5B-Instruct \
  --teacher_model Qwen/Qwen2.5-7B-Instruct \
  --train_data data/train.jsonl \
  --output_dir outputs/distilled_qwen \
  --batch_size 4 --epochs 3 --lr 2e-5 \
  --max_length 2048 --mask_prompt_tokens
```
`--teacher_model` is optional (for online distillation). `--mask_prompt_tokens` masks question tokens so loss is only on reasoning + answer.

### Part C: Inference & Evaluation
```bash
python inference_eval.py \
  --base_model <model-id> \
  --adapter_path "" \
  --test_data <test-file.jsonl> \
  --output_predictions outputs/predictions.jsonl \
  --report_file outputs/metrics.txt \
  --max_new_tokens 2048
```

## Architecture

### Data Flow
```
data/dataset.jsonl
  → MMLUPro.load_mmlu_pro()       # loads + normalizes multilingual JSONL
  → sample_datasets()              # per-language sampling with fixed seed
  → format_teacher_prompt()        # language-aware prompt with structured tags
  → generate_and_parse_batch()     # batched vLLM inference
  → _parse_teacher_generation()    # defensive 3-step answer extraction
  → data/train.jsonl               # teacher-distilled reasoning traces
```

### Teacher Prompt Format
Responses must wrap reasoning in `<reasoning>...</reasoning>` and end with `#### ANSWER: [A-J]`. The `_extract_answer_letter()` function has a 3-step fallback cascade: strict format → loose regex on last 400 chars → scan last 5 lines.

### Output JSONL Schema
Each record contains: `question`, `reasoning`, `final_answer`, `gold_answer`, `language`, `subject`, `source_dataset`, `prompt`, `teacher_generation`, `teacher_answer_parsed` (bool), `teacher_correct` (bool).

### Key Constants (dataset_generation.py)
- `LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]`
- `MAX_TOTAL_SAMPLES = 10_000`
- Answer options: letters A–J (10-way MCQ)

### MMLUPro Dataset Loader (data/mmlupro.py)
Handles normalization across answer formats (letter, index, digit string), language code canonicalization, and chat-template-compatible message conversion.

## Submission Checklist

- `dataset_generation.py` ✓
- `train_distill.py` — needs implementation
- `inference_eval.py` — needs implementation
- `predictions_<student_model>.jsonl` — format: `{language, subject, question, gold_answer, predicted_answer, generation}`
- `metrics_<student_model>.txt` — format: `<LANGUAGE> ACCURACY: xx.xx` per language
- `run_distillation.sh` — end-to-end pipeline shell script
- `requirements.txt`
- Project report (max 4 pages): distillation strategy, cross-family challenges, metrics comparison, honour code

## Design Notes (PART_A_DESIGN.md)
The `PART_A_DESIGN.md` file documents every major function and decision in Part A in detail — consult it before modifying `dataset_generation.py`.
