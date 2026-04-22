# Assignment Constraints (COL772 A4)

Extracted from `a4_col772.pdf`. Violating any of these will cost marks or cause grading failure.

## Data Constraints

- **Source**: Sample exclusively from `data/dataset.jsonl`. Downloading pre-distilled datasets is NOT allowed.
- **Total samples**: ≤ 10,000 training samples across all languages combined. Enforced in `sample_datasets()`.
- **Per-language availability** (hard ceiling):
  | Language | Max available |
  |----------|--------------|
  | English  | 8,200        |
  | Hindi    | 4,200        |
  | Bengali  | 3,100        |
  | Kannada  | 2,500        |
  | Tamil    | 2,000        |
- A validation set is allowed (counts toward the 10K total if sampled from `dataset.jsonl`).

## Generation Constraints

- **Max tokens per question**: 2,048 (`--max_new_tokens` must be ≤ 2048). Enforced in `main()`.
- **Single-pass only**: No multi-call strategies (e.g., separate translation + solving calls) — each row must consume exactly one generation slot.
- **No rejection sampling / retries**: `DEFAULT_REJECTION_SAMPLING_RETRIES = 0`, `DEFAULT_MAX_RETRIES = 0`. Retries burn the 10K budget.

## Models (fixed — do not change)

| Role               | Model                         |
|--------------------|-------------------------------|
| Teacher            | `Qwen/Qwen2.5-7B-Instruct`    |
| Student (in-family)| `Qwen/Qwen2.5-1.5B-Instruct`  |
| Student (cross-family) | `meta-llama/Llama-3.2-1B-Instruct` |

Cross-family KD (Qwen→LLaMA) cannot use logit-level distillation due to different vocabularies; use text-based CoT SFT only.

## Allowed Libraries

Only these packages are permitted without Piazza approval:
`transformers`, `datasets`, `vllm`, `peft`, `trl`, `torch`, `tqdm`, `pandas`, `numpy`, Python standard library.

Any additional package requires explicit Piazza approval before use.

## Time Limits (on NVIDIA V100, Python 3.10)

| Script                   | Limit      |
|--------------------------|------------|
| `dataset_generation.py`  | 240 minutes |
| `train_distill.py`       | 240 minutes |
| `inference_eval.py`      | 60 minutes  |

Scripts exceeding these limits are terminated by the grader.

## Output Format Requirements

### `predictions_<student_model>.jsonl`
One JSON per line:
```json
{"language": "en", "subject": "...", "question": "...", "gold_answer": "H", "predicted_answer": "H", "generation": "...#### ANSWER: (H)"}
```

### `metrics_<student_model>.txt`
```
English ACCURACY: xx.xx
Hindi ACCURACY: xx.xx
Bengali ACCURACY: xx.xx
Kannada ACCURACY: xx.xx
Tamil ACCURACY: xx.xx
Overall ACCURACY: xx.xx
```

### `data/train.jsonl` (teacher output schema)
Required fields: `question`, `final_answer`, `gold_answer`, `language`, `subject`, `teacher_generation`.

## Submission Checklist

1. `dataset_generation.py` — must support `--teacher_model`, `--num_samples`, `--output_file`
2. `train_distill.py` — must support `--teacher_model`, `--student_model`, `--train_data`, `--output_dir`
3. `inference_eval.py` — must support `--base_model`, `--adapter_path`, `--test_data`, `--output_predictions`, `--report_file`
4. `predictions_<student_model>.jsonl` × 2 (one per student)
5. `metrics_<student_model>.txt` × 2
6. `run_distillation.sh` — end-to-end shell script
7. `requirements.txt`
8. Project report (max 4 pages) including honour code

## Rationalization Strategy (TA-clarified)

Because we are strictly capped at 10,000 total generations, **Gold-Guided Rationalization** is used:
- The teacher prompt in `format_teacher_prompt()` injects the gold answer letter so the teacher always produces correct reasoning traces.
- The student prompt in `_format_student_prompt()` must **never** include the gold answer — the student must learn to derive the answer from scratch.
- This is standard Knowledge Distillation; the student sees the teacher's reasoning but not the hint that produced it.
