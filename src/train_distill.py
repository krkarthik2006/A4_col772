from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


LOGGER = logging.getLogger(__name__)

# Response templates used by DataCollatorForCompletionOnlyLM to locate the
# start of the assistant turn so prompt tokens can be masked out of the loss.
RESPONSE_TEMPLATES: dict[str, str] = {
    "qwen":  "<|im_start|>assistant\n",
    "llama": "<|start_header_id|>assistant<|end_header_id|>\n\n",
}


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline CoT distillation: SFT student on teacher reasoning traces"
    )

    # ── required / core ──────────────────────────────────────────────────────
    parser.add_argument("--student_model", required=True,
                        help="HuggingFace model ID or path to the student model")
    parser.add_argument("--teacher_model", required=False,
                        help="HuggingFace model ID or path to the teacher model "
                             "(reserved for online distillation; unused in offline mode)")
    parser.add_argument("--train_data", default="data/train.jsonl",
                        help="Path to training JSONL produced by dataset_generation.py")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save LoRA adapter (and optionally merged model)")

    # ── basic training knobs ─────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Peak learning rate (2e-4 is appropriate for LoRA)")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum tokenized sequence length")
    parser.add_argument(
        "--mask_prompt_tokens",
        action="store_true",
        help="Mask question tokens so loss is computed only on reasoning + answer",
    )

    # ── LoRA / PEFT ──────────────────────────────────────────────────────────
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA scaling factor (alpha = 2*r is a good default)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout probability on LoRA layers")
    parser.add_argument("--use_qlora", action="store_true",
                        help="Load base model in 4-bit (NF4) for QLoRA training")
    parser.add_argument("--save_merged", action="store_true",
                        help="After training, merge LoRA weights into base and save "
                             "the full model to <output_dir>/merged/")

    # ── optimiser schedule ───────────────────────────────────────────────────
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps "
                             "(effective_batch = batch_size × accum)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Fraction of total steps used for LR warmup")

    # ── data filtering / validation ──────────────────────────────────────────
    parser.add_argument("--filter_correct_only", action="store_true",
                        help="Only train on examples where teacher_correct==True")
    parser.add_argument("--val_data", default="",
                        help="Optional path to validation JSONL for eval during training")

    # ── soft-label KD (top-K logits) ─────────────────────────────────────────
    parser.add_argument("--top_k_logits", type=int, default=0,
                        help="Use top-K soft-label KD loss (0 = disabled, paper default: 10). "
                             "Requires teacher_top_k_logits field in train_data.")
    parser.add_argument("--kd_alpha", type=float, default=0.5,
                        help="Weight of the KD loss: total = (1-α)·CE + α·T²·KL")
    parser.add_argument("--kd_temperature", type=float, default=1.0,
                        help="Temperature T for teacher/student distribution sharpening")

    # ── logging ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def detect_model_family(model_path: str) -> str:
    """Return 'llama' or 'qwen' based on the model name/path."""
    return "llama" if "llama" in model_path.lower() else "qwen"


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [
            json.loads(line)
            for raw_line in fh
            if (line := raw_line.strip())
        ]


def _fit_question(question: str, final_answer: str, tokenizer, max_tokens: int) -> str:
    """
    Fit the question+options into max_tokens, guaranteeing the correct option is kept.

    Strategy:
      1. If it already fits, return as-is.
      2. Otherwise split into stem and option lines. Always keep the correct option.
         Fill remaining budget with other options (in original order). Truncate the
         stem last if still needed.
    """
    ids = tokenizer.encode(question, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return question

    # Split stem from options block (separated by first \n\n)
    parts = question.split("\n\n", 1)
    if len(parts) != 2:
        # No options block detected — just truncate the stem
        return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

    stem, options_block = parts
    option_lines = [l for l in options_block.split("\n") if l.strip()]

    correct_line, other_lines = None, []
    for line in option_lines:
        if line.startswith(f"({final_answer})"):
            correct_line = line
        else:
            other_lines.append(line)

    # Build with correct option guaranteed; greedily add others that still fit
    kept = [correct_line] if correct_line else []
    base = f"{stem}\n\n" + "\n".join(kept)
    budget = max_tokens - len(tokenizer.encode(base, add_special_tokens=False))

    for line in other_lines:
        line_cost = len(tokenizer.encode("\n" + line, add_special_tokens=False))
        if budget >= line_cost:
            kept.append(line)
            budget -= line_cost

    # Restore original option order (sort by letter)
    kept.sort(key=lambda l: l[1] if l.startswith("(") else "Z")
    options_text = "\n".join(kept)

    # If stem itself is too long after options are fixed, truncate it
    stem_budget = max_tokens - len(tokenizer.encode("\n\n" + options_text, add_special_tokens=False))
    stem_ids = tokenizer.encode(stem, add_special_tokens=False)
    if len(stem_ids) > stem_budget:
        stem = tokenizer.decode(stem_ids[:stem_budget], skip_special_tokens=True)

    return f"{stem}\n\n{options_text}"


def build_training_text(row: dict, tokenizer, max_question_tokens: int = 1200) -> str:
    """
    Construct a full chat-templated string for one training example.

    Input format (from dataset_generation.py output):
      - row["question"]       : full MCQ text with options
      - row["reasoning"]      : teacher reasoning extracted from <reasoning>…</reasoning>
      - row["final_answer"]   : parsed answer letter (A–J)
      - row["gold_answer"]    : ground-truth letter (fallback)

    Output: tokenizer.apply_chat_template result (no generation prompt suffix).
    """
    final_answer = row.get("final_answer") or row.get("gold_answer", "")
    reasoning = row.get("reasoning", "").strip()

    question = _fit_question(row["question"], final_answer, tokenizer, max_question_tokens)

    assistant_content = (
        f"<reasoning>\n{reasoning}\n</reasoning>\n"
        f"#### ANSWER: ({final_answer})"
    )

    messages = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": assistant_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def prepare_dataset(
    data_path: str,
    tokenizer,
    filter_correct_only: bool = False,
    max_length: int = 2048,
) -> Dataset:
    """Load and format JSONL into a HuggingFace Dataset with a single 'text' column."""
    rows = load_training_rows(data_path, filter_correct_only=filter_correct_only)
    # Reserve 848 tokens for response (reasoning + answer); question gets the rest.
    max_question_tokens = max_length - 848
    texts = [build_training_text(r, tokenizer, max_question_tokens=max_question_tokens) for r in rows]
    return Dataset.from_dict({"text": texts})


def load_training_rows(
    data_path: str,
    *,
    filter_correct_only: bool = False,
) -> list[dict]:
    rows = load_jsonl(data_path)
    LOGGER.info("Loaded %d rows from %s", len(rows), data_path)

    if not filter_correct_only:
        return rows

    before = len(rows)
    rows = [row for row in rows if row.get("teacher_correct", True)]
    LOGGER.info(
        "filter_correct_only: kept %d / %d rows (dropped %d)",
        len(rows), before, before - len(rows),
    )
    return rows


# ── Top-K soft-label KD helpers ──────────────────────────────────────────────

def _find_sublist(haystack: list, needle: list) -> int:
    """Return the start index of needle in haystack, or -1 if not found."""
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def tokenize_with_kd(
    row: dict,
    tokenizer,
    response_template_ids: list,
    top_k: int,
    max_length: int,
) -> dict:
    """Tokenize one example and align stored teacher top-K logprobs with token positions.

    teacher_top_k_ids / teacher_top_k_vals shape: [seq_len, top_k].
    Index i in those tensors corresponds to model logits[:, i, :], i.e. the
    distribution that predicts input_ids[i+1].  Positions without teacher data
    are filled with -1 / 0.0 and masked out in the KD loss.
    """
    text = build_training_text(row, tokenizer)
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    input_ids: list = enc["input_ids"]
    seq_len = len(input_ids)

    # Build labels: -100 everywhere; unmask from the first response token onward.
    labels = [-100] * seq_len
    template_pos = _find_sublist(input_ids, response_template_ids)
    logit_offset = -1
    if template_pos >= 0:
        resp_start = template_pos + len(response_template_ids)
        labels[resp_start:] = input_ids[resp_start:]
        # logits[logit_offset] predicts input_ids[resp_start] — the first
        # response token — so teacher logit j maps to logit position logit_offset+j.
        logit_offset = resp_start - 1

    # Initialise teacher tensors with "no data" sentinel.
    t_ids = [[-1] * top_k for _ in range(seq_len)]
    t_vals = [[0.0] * top_k for _ in range(seq_len)]

    raw_logits = row.get("teacher_top_k_logits")
    if raw_logits and logit_offset >= 0:
        for j, tok_topk in enumerate(raw_logits):
            pos = logit_offset + j
            # pos >= seq_len-1: no label exists for the next token, skip.
            if pos >= seq_len - 1:
                break
            for k_idx, pair in enumerate(tok_topk[:top_k]):
                t_ids[pos][k_idx] = pair[0]
                t_vals[pos][k_idx] = pair[1]

    return {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "teacher_top_k_ids": t_ids,
        "teacher_top_k_vals": t_vals,
    }


def prepare_kd_dataset(
    data_path: str,
    tokenizer,
    response_template_ids: list,
    top_k: int,
    max_length: int,
    filter_correct_only: bool = False,
) -> Dataset:
    rows = load_training_rows(data_path, filter_correct_only=filter_correct_only)
    has_logits = sum(1 for r in rows if r.get("teacher_top_k_logits"))
    LOGGER.info("Rows with teacher top-K logits: %d / %d", has_logits, len(rows))

    tokenized = [
        tokenize_with_kd(r, tokenizer, response_template_ids, top_k, max_length)
        for r in rows
    ]
    return Dataset.from_list(tokenized)


class TopKLogitCollator:
    """Pads pre-tokenised KD examples and stacks teacher top-K tensors."""

    def __init__(self, tokenizer, top_k: int) -> None:
        self.pad_id = tokenizer.pad_token_id
        self.top_k = top_k

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids, batch_mask, batch_labels = [], [], []
        batch_t_ids, batch_t_vals = [], []

        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            batch_input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            batch_mask.append(f["attention_mask"] + [0] * pad)
            batch_labels.append(f["labels"] + [-100] * pad)
            batch_t_ids.append(f["teacher_top_k_ids"] + [[-1] * self.top_k] * pad)
            batch_t_vals.append(f["teacher_top_k_vals"] + [[0.0] * self.top_k] * pad)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "teacher_top_k_ids": torch.tensor(batch_t_ids, dtype=torch.long),
            "teacher_top_k_vals": torch.tensor(batch_t_vals, dtype=torch.float),
        }


class KDTrainer(Trainer):
    """Trainer that mixes cross-entropy with top-K soft-label KL divergence.

    Loss = (1 - kd_alpha) * CE  +  kd_alpha * T² * KL(p_T || p_S)

    where p_T and p_S are renormalised over the top-K teacher positions with
    temperature T, following MiniLLM / DistiLLM conventions.
    """

    def __init__(self, *args, kd_alpha: float = 0.5, kd_temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        t_ids = inputs.pop("teacher_top_k_ids", None)   # [B, L, K]
        t_vals = inputs.pop("teacher_top_k_vals", None)  # [B, L, K]  (log-probs)

        outputs = model(**inputs)
        ce_loss = outputs.loss

        if t_ids is None or self.kd_alpha == 0.0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # has_teacher: True at positions where teacher data was stored.
        has_teacher = t_ids[:, :, 0] >= 0  # [B, L]
        if not has_teacher.any():
            return (ce_loss, outputs) if return_outputs else ce_loss

        T = self.kd_temperature
        student_logits = outputs.logits  # [B, L, V]

        # Gather student logits at the teacher's top-K vocabulary positions.
        safe_ids = t_ids.clamp(min=0)                              # [B, L, K]
        s_topk = student_logits.gather(dim=-1, index=safe_ids)    # [B, L, K]

        # Renormalise both distributions over top-K with temperature T.
        # t_vals are log-probs; dividing by T and applying softmax gives
        # p_T(k) ∝ p_teacher(k)^(1/T), consistent with the paper's formula.
        t_log_probs = F.log_softmax(t_vals / T, dim=-1)   # [B, L, K]
        s_log_probs = F.log_softmax(s_topk / T, dim=-1)   # [B, L, K]

        # KL(p_T || p_S) per position, using F.kl_div which handles 0·log0 = 0.
        kl = F.kl_div(s_log_probs, t_log_probs.exp(), reduction="none").sum(dim=-1)  # [B, L]

        mask = has_teacher.float()
        kl_loss = (kl * mask).sum() / mask.sum().clamp(min=1)

        loss = (1.0 - self.kd_alpha) * ce_loss + self.kd_alpha * (T ** 2) * kl_loss
        return (loss, outputs) if return_outputs else loss


# ── Model / config builders ───────────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantisation config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(args: argparse.Namespace) -> LoraConfig:
    """
    LoRA applied to all attention + FFN projection layers.
    The 7-module list is identical for Qwen2.5 and LLaMA-3.2.
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        inference_mode=False,
    )


def load_student(args: argparse.Namespace):
    """Load student tokenizer + model, with optional QLoRA quantisation."""
    LOGGER.info("Loading tokenizer from %s", args.student_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    # Some models (e.g. LLaMA-3.2) ship without a pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right-padding keeps attention masks aligned with labels during SFT.
    tokenizer.padding_side = "right"

    model_kwargs: dict = dict(
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if args.use_qlora:
        model_kwargs["quantization_config"] = get_bnb_config()

    LOGGER.info("Loading student model from %s", args.student_model)
    model = AutoModelForCausalLM.from_pretrained(args.student_model, local_files_only=True, **model_kwargs)
    # Disable KV-cache so gradient checkpointing works correctly.
    model.config.use_cache = False

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_family = detect_model_family(args.student_model)
    LOGGER.info("Detected model family: %s", model_family)

    model, tokenizer = load_student(args)

    lora_config = get_lora_config(args)

    # Common training argument values shared by both training paths.
    common_training_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataloader_drop_last=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    resp_template = RESPONSE_TEMPLATES[model_family]
    resp_template_ids = tokenizer.encode(resp_template, add_special_tokens=False)

    train_dataset: Dataset
    eval_dataset: Optional[Dataset] = None

    if args.top_k_logits > 0:
        # ── Soft-label KD path ────────────────────────────────────────────────
        LOGGER.info(
            "KD mode: top_k=%d, alpha=%.2f, temperature=%.2f",
            args.top_k_logits, args.kd_alpha, args.kd_temperature,
        )

        train_dataset = prepare_kd_dataset(
            args.train_data, tokenizer, resp_template_ids,
            args.top_k_logits, args.max_length,
            filter_correct_only=args.filter_correct_only,
        )
        LOGGER.info("Training examples: %d", len(train_dataset))

        if args.val_data:
            eval_dataset = prepare_kd_dataset(
                args.val_data, tokenizer, resp_template_ids,
                args.top_k_logits, args.max_length,
            )
            LOGGER.info("Validation examples: %d", len(eval_dataset))
        has_eval = eval_dataset is not None

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        collator = TopKLogitCollator(tokenizer, args.top_k_logits)

        training_args = TrainingArguments(
            **common_training_kwargs,
            remove_unused_columns=False,
            eval_strategy="epoch" if has_eval else "no",
            load_best_model_at_end=has_eval,
        )

        trainer = KDTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            kd_alpha=args.kd_alpha,
            kd_temperature=args.kd_temperature,
        )

    else:
        # ── Standard SFT path ─────────────────────────────────────────────────
        train_dataset = prepare_dataset(
            args.train_data, tokenizer,
            filter_correct_only=args.filter_correct_only,
            max_length=args.max_length,
        )
        LOGGER.info("Training examples: %d", len(train_dataset))

        if args.val_data:
            eval_dataset = prepare_dataset(args.val_data, tokenizer, max_length=args.max_length)
            LOGGER.info("Validation examples: %d", len(eval_dataset))
        has_eval = eval_dataset is not None

        collator = None
        if args.mask_prompt_tokens:
            collator = DataCollatorForCompletionOnlyLM(
                response_template=resp_template_ids,
                tokenizer=tokenizer,
            )
            LOGGER.info(
                "Prompt masking ON — response template: %r (%d tokens)",
                resp_template, len(resp_template_ids),
            )
        else:
            LOGGER.info("Prompt masking OFF — loss computed on full sequence")

        training_args = SFTConfig(
            **common_training_kwargs,
            max_seq_length=args.max_length,
            dataset_text_field="text",
            remove_unused_columns=True,
            eval_strategy="epoch" if has_eval else "no",
            load_best_model_at_end=has_eval,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
            data_collator=collator,
            processing_class=tokenizer,
        )

    LOGGER.info(
        "Starting training: %d examples, %d epochs, lr=%.2e, "
        "batch=%d×%d (effective %d), LoRA r=%d α=%d",
        len(train_dataset), args.epochs, args.lr,
        args.batch_size, args.gradient_accumulation_steps,
        args.batch_size * args.gradient_accumulation_steps,
        args.lora_r, args.lora_alpha,
    )
    trainer.train()

    # ── save adapter ─────────────────────────────────────────────────────────
    LOGGER.info("Saving LoRA adapter to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ── optional: merge LoRA into base and save full model ───────────────────
    if args.save_merged:
        LOGGER.info("Merging LoRA weights into base model…")
        merged_model = trainer.model.merge_and_unload()
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(exist_ok=True)
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        LOGGER.info("Merged model saved to %s", merged_dir)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
