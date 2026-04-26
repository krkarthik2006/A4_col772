from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from utils import load_vllm_llm, prompt_vllm


LOGGER = logging.getLogger(__name__)

LANGUAGES = ["en", "hindi", "bengali", "kannada", "tamil"]
LANGUAGE_LABELS = {
    "en":      "English",
    "english": "English",
    "hindi":   "Hindi",
    "bengali": "Bengali",
    "kannada": "Kannada",
    "tamil":   "Tamil",
}

GENERATION_SYSTEM_MESSAGE = (
    "You are a careful multilingual reasoning assistant that follows "
    "the requested output format exactly."
)

ANSWER_TAG_RE = re.compile(r"####\s*ANSWER\s*:\s*\(?([A-J])\)?", re.IGNORECASE)
FREEFORM_RE   = re.compile(
    r"(?:final\s+answer|answer|option)\s*[:\-]?\s*\(?([A-J])\)?",
    re.IGNORECASE,
)
LAST_LETTER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference + eval on test JSONL")
    parser.add_argument("--base_model", required=True,
                        help="Path to distilled student model (or base model if using adapter)")
    parser.add_argument("--adapter_path", default="",
                        help="Optional path to LoRA adapter directory")
    parser.add_argument("--test_data", required=True,
                        help="Path to test JSONL")
    parser.add_argument("--output_predictions", required=True,
                        help="Path to write predictions JSONL")
    parser.add_argument("--report_file", required=True,
                        help="Path to write per-language accuracy report")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def canonical_language(lang: str) -> str:
    tag = lang.strip().lower()
    if tag in {"en", "eng", "english"}:
        return "en"
    return tag


def extract_answer(text: str) -> str:
    all_matches = ANSWER_TAG_RE.findall(text)
    if all_matches:
        return all_matches[-1].upper()

    tail = text[-800:]
    m = FREEFORM_RE.search(tail)
    if m:
        return m.group(1).upper()

    lines = [l for l in text.splitlines() if l.strip()][-5:]
    for line in reversed(lines):
        letters = LAST_LETTER_RE.findall(line)
        if letters:
            return letters[-1].upper()

    return ""


def _format_student_prompt(instruction: str, language: str = "") -> str:
    lang_instruction = (
        "1. Identify the core concept or formula needed to solve this question."
        if language in ("en", "english") else
        "1. First, translate the question to English. Then, identify the core concept needed."
    )

    return (
        "You are an expert multilingual reasoning assistant.\n"
        "Your task is to provide the step-by-step reasoning that leads to the correct answer.\n\n"
        "Output rules:\n"
        f"{lang_instruction}\n"
        "2. Write all your reasoning and analysis entirely in English.\n"
        "3. Put all reasoning inside <reasoning> and </reasoning> tags.\n"
        "4. After the reasoning block, write exactly one final line in the format "
        "'#### ANSWER: (LETTER)' where LETTER is a single capital letter A-J. "
        "Do not write anything else on that line or after it.\n"
        "5. Stop immediately after writing the answer line.\n\n"
        "Question:\n"
        f"{instruction}"
    )


def build_inference_prompt(row: dict) -> list[dict]:
    options = row.get("options", [])
    question_text = row.get("question", "")
    language = row.get("language", "en")

    if options:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        choices = "\n".join(f"({letters[i]}) {opt}" for i, opt in enumerate(options))
        instruction = f"{question_text}\n\n{choices}"
    else:
        instruction = question_text

    user_content = _format_student_prompt(instruction, language)

    return [
        {"role": "system", "content": GENERATION_SYSTEM_MESSAGE},
        {"role": "user",   "content": user_content},
    ]


def _batched(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    model_path = args.base_model

    if args.adapter_path:
        LOGGER.info(
            "Merging LoRA adapter %s into base model %s before loading with vLLM",
            args.adapter_path, args.base_model,
        )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(
            args.adapter_path, trust_remote_code=True, local_files_only=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            local_files_only=True,
        )
        peft_model = PeftModel.from_pretrained(base, args.adapter_path)
        merged = peft_model.merge_and_unload()

        merged_tmp = Path(args.output_predictions).parent / "_merged_tmp"
        merged_tmp.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving merged model to %s", merged_tmp)
        merged.save_pretrained(str(merged_tmp))
        tokenizer.save_pretrained(str(merged_tmp))

        del merged, peft_model, base
        torch.cuda.empty_cache()

        model_path = str(merged_tmp)

    LOGGER.info("Loading model for inference via vLLM: %s", model_path)
    llm, tokenizer = load_vllm_llm(
        model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    test_rows = load_jsonl(args.test_data)
    LOGGER.info("Loaded %d test examples from %s", len(test_rows), args.test_data)

    predictions: list[dict] = []
    all_messages = [build_inference_prompt(row) for row in test_rows]

    LOGGER.info(
        "Running inference: batch_size=%d, max_new_tokens=%d",
        args.batch_size, args.max_new_tokens,
    )
    for batch_rows, batch_msgs in tqdm(
        zip(_batched(test_rows, args.batch_size),
            _batched(all_messages, args.batch_size)),
        total=(len(test_rows) + args.batch_size - 1) // args.batch_size,
        desc="Inference",
    ):
        generations = prompt_vllm(
            llm,
            tokenizer,
            batch_messages=batch_msgs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stop=[s for l in "ABCDEFGHIJ" for s in (f"#### ANSWER: ({l})\n", f"#### ANSWER: ({l})")],
            use_tqdm=False,
        )
        for row, gen in zip(batch_rows, generations):
            predicted = extract_answer(gen)
            gold = row.get("answer", row.get("gold_answer", ""))
            if isinstance(gold, int):
                gold = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[gold]
            gold = str(gold).strip().upper()

            predictions.append({
                "language":         canonical_language(row.get("language", "en")),
                "subject":          row.get("subject", ""),
                "question":         row.get("question", ""),
                "gold_answer":      gold,
                "predicted_answer": predicted,
                "generation":       gen,
            })

    out_path = Path(args.output_predictions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for pred in predictions:
            fh.write(json.dumps(pred, ensure_ascii=False) + "\n")
    LOGGER.info("Predictions saved to %s", out_path)

    lang_correct: dict[str, int] = defaultdict(int)
    lang_total:   dict[str, int] = defaultdict(int)

    for pred in predictions:
        lang = pred["language"]
        lang_total[lang] += 1
        if pred["predicted_answer"] == pred["gold_answer"]:
            lang_correct[lang] += 1

    canonical_groups: dict[str, list[str]] = {
        "en":      ["en"],
        "hindi":   ["hindi"],
        "bengali": ["bengali"],
        "kannada": ["kannada"],
        "tamil":   ["tamil"],
    }

    report_lines: list[str] = []
    overall_correct = 0
    overall_total   = 0

    for canonical, aliases in canonical_groups.items():
        correct = sum(lang_correct.get(a, 0) for a in aliases)
        total   = sum(lang_total.get(a, 0)   for a in aliases)
        if total == 0:
            continue
        acc = 100.0 * correct / total
        label = LANGUAGE_LABELS.get(canonical, canonical.capitalize())
        line = f"{label} ACCURACY: {acc:.2f}"
        report_lines.append(line)
        LOGGER.info(line)
        overall_correct += correct
        overall_total   += total

    if overall_total > 0:
        overall_acc = 100.0 * overall_correct / overall_total
        line = f"Overall ACCURACY: {overall_acc:.2f}"
        report_lines.append(line)
        LOGGER.info(line)

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines) + "\n")
    LOGGER.info("Metrics report saved to %s", report_path)


if __name__ == "__main__":
    main()
