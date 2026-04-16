from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, concatenate_datasets
from tqdm.auto import tqdm

from data.mmlupro import MMLUPro
from utils import load_vllm_llm, prompt_vllm


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
DATASET_LANGUAGE_CODES = {
    "english": "en",
    "hindi": "hindi",
    "bengali": "bengali",
    "kannada": "kannada",
    "tamil": "tamil",
}
LANGUAGE_NAMES = {
    "english": "English",
    "hindi": "Hindi",
    "bengali": "Bengali",
    "kannada": "Kannada",
    "tamil": "Tamil",
}
ANSWER_RE = re.compile(r"####\s*ANSWER\s*:\s*([A-J])", re.IGNORECASE)
REASONING_BLOCK_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>",
    re.IGNORECASE | re.DOTALL,
)
FREEFORM_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer|option)\s*[:\-]?\s*\(?([A-J])\)?",
    re.IGNORECASE,
)
TAIL_LETTER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def _options_to_text(options: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(
        f"({letters[idx]}) {choice}" for idx, choice in enumerate(options)
    )


def sample_datasets(
        samples_per_language: list[int],
        split: str = "test",
        seed: int = 42,
) -> Dataset:
    """Fetch and sample the requested number of rows for each language."""
    if len(samples_per_language) != len(LANGUAGES):
        raise ValueError(
            "--num_samples must contain 5 comma-separated integers for "
            "english,hindi,bengali,kannada,tamil"
        )
    if any(count < 0 for count in samples_per_language):
        raise ValueError("--num_samples values must be >= 0")
    if sum(samples_per_language) <= 0:
        raise ValueError("--num_samples must request at least one sample")
    if sum(samples_per_language) > 10_000:
        raise ValueError(
            "The total requested samples exceed the assignment limit of 10,000"
        )

    sampled_datasets: list[Dataset] = []
    for idx, (language_name, sample_count) in enumerate(
        zip(LANGUAGES, samples_per_language)
    ):
        if sample_count == 0:
            LOGGER.info("Skipping %s because requested sample count is 0", language_name)
            continue

        dataset_language = DATASET_LANGUAGE_CODES[language_name]
        dataset = MMLUPro(
            language=dataset_language,
            split=split,
            logger=LOGGER,
        ).load_mmlu_pro(unified=True)

        available = len(dataset)
        if sample_count > available:
            raise ValueError(
                f"Requested {sample_count} samples for {language_name}, "
                f"but only {available} are available"
            )

        sampled_language_dataset = dataset.shuffle(seed=seed + idx).select(
            range(sample_count)
        )
        LOGGER.info(
            "Sampled %d/%d rows for %s",
            sample_count,
            available,
            language_name,
        )
        sampled_datasets.append(sampled_language_dataset)

    if not sampled_datasets:
        raise ValueError("No samples were selected after language-wise sampling")

    if len(sampled_datasets) == 1:
        return sampled_datasets[0].shuffle(seed=seed)

    return concatenate_datasets(sampled_datasets).shuffle(seed=seed)


def format_teacher_prompt(instruction: str, language: str) -> str:
    """Build a language-aware prompt that enforces reasoning and final answer."""
    canonical_language = _canonical_prompt_language(language)
    language_name = LANGUAGE_NAMES.get(canonical_language, canonical_language.title())
    return (
        "You are creating multilingual knowledge-distillation training data.\n"
        "Solve the following multiple-choice question carefully.\n\n"
        "Output rules:\n"
        f"1. Write the full reasoning only in {language_name}.\n"
        "2. Put all reasoning inside <reasoning> and </reasoning> tags.\n"
        "3. After the reasoning block, write exactly one final line in the format "
        "'#### ANSWER: [LETTER]'.\n"
        "4. The final answer letter must be one of A-J.\n"
        "5. Do not output anything after the final answer line.\n\n"
        "Question:\n"
        f"{instruction}"
    )


def generate_and_parse(
        model,
        tokenizer,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
) -> dict[str, str]:
    """Query teacher model once and extract instruction/reasoning/final answer."""
    return generate_and_parse_batch(
        model,
        tokenizer,
        [prompt],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )[0]


def generate_and_parse_batch(
        model,
        tokenizer,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
) -> list[dict[str, str]]:
    """Query the teacher on a batch of prompts and parse the outputs."""
    batch_messages = [
        [
            {
                "role": "system",
                "content": (
                    "You are a careful multilingual reasoning assistant that follows "
                    "the requested output format exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]

    raw_generations = prompt_vllm(
        model,
        tokenizer,
        batch_messages=batch_messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_tqdm=False,
    )

    return [_parse_teacher_generation(text) for text in raw_generations]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query teacher and build train corpus JSONL"
    )
    parser.add_argument(
        "--teacher_model",
        required=True,
        help="Hugging Face path to the teacher model",
    )
    parser.add_argument(
        "--num_samples",
        type=str,
        required=True,
        help=(
            "Comma-separated sample counts for english,hindi,bengali,"
            "kannada,tamil"
        ),
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSONL path for train corpus",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Teacher prompting batch size",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per question (<= 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Teacher decoding temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Teacher decoding top-p",
    )
    parser.add_argument(
        "--filter_incorrect",
        action="store_true",
        help="Drop samples where the parsed teacher answer != gold answer",
    )
    parser.add_argument(
        "--skip_unparsed",
        action="store_true",
        help="Drop samples if the answer cannot be parsed from the teacher output",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.6,
        help="Target fraction of GPU memory for vLLM; lower if startup fails",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def _parse_num_samples(raw_value: str) -> list[int]:
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(parts) != len(LANGUAGES):
        raise ValueError(
            "--num_samples must contain exactly 5 comma-separated integers "
            "for english,hindi,bengali,kannada,tamil"
        )

    try:
        counts = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(
            "--num_samples must contain only integers"
        ) from exc

    if any(count < 0 for count in counts):
        raise ValueError("--num_samples values must be >= 0")

    return counts


def _build_instruction(row: dict[str, Any]) -> str:
    options = row["options"]
    if not isinstance(options, list):
        options = list(options)
    return f"{row['question']}\n\n{_options_to_text(options)}"


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    samples_per_language = _parse_num_samples(args.num_samples)
    if args.max_new_tokens <= 0 or args.max_new_tokens > 2048:
        raise ValueError("--max_new_tokens must be in the range [1, 2048]")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sampled = sample_datasets(
        samples_per_language=samples_per_language,
        split=args.split,
        seed=args.seed,
    )
    LOGGER.info("Collected %d samples", len(sampled))

    teacher, tokenizer = load_vllm_llm(
        model_id=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    written = 0
    skipped_unparsed = 0
    skipped_incorrect = 0
    with output_path.open("w", encoding="utf-8") as fp:
        sampled_rows = list(sampled)
        total_batches = (len(sampled_rows) + args.batch_size - 1) // args.batch_size
        for batch_rows in tqdm(
            _batched(sampled_rows, args.batch_size),
            desc="Generating teacher traces",
            total=total_batches,
        ):
            prompts = []
            questions_with_choices = []
            for row in batch_rows:
                question_with_choices = _build_instruction(row)
                questions_with_choices.append(question_with_choices)
                prompts.append(
                    format_teacher_prompt(question_with_choices, row["language"])
                )

            parsed_batch = generate_and_parse_batch(
                teacher,
                tokenizer,
                prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            for row, question_with_choices, prompt, parsed in zip(
                batch_rows, questions_with_choices, prompts, parsed_batch
            ):
                gold_answer = str(row.get("answer", "")).upper()[:1]
                teacher_answer = parsed["final_answer"]
                answer_parsed = bool(teacher_answer)
                teacher_correct = answer_parsed and teacher_answer == gold_answer

                if args.skip_unparsed and not answer_parsed:
                    skipped_unparsed += 1
                    continue
                if args.filter_incorrect and answer_parsed and not teacher_correct:
                    skipped_incorrect += 1
                    continue

                record = {
                    "question": question_with_choices,
                    "reasoning": parsed["reasoning"],
                    "final_answer": teacher_answer,
                    "gold_answer": gold_answer,
                    "language": row["language"],
                    "subject": row.get("subject"),
                    "source_dataset": row.get("source_dataset"),
                    "prompt": prompt,
                    "teacher_generation": parsed["raw_generation"],
                    "teacher_answer_parsed": answer_parsed,
                    "teacher_correct": teacher_correct,
                }
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    LOGGER.info(
        "Saved %d rows to %s (skipped_unparsed=%d, skipped_incorrect=%d)",
        written,
        output_path,
        skipped_unparsed,
        skipped_incorrect,
    )


def _canonical_prompt_language(language: str) -> str:
    normalized = str(language).strip().lower()
    for canonical, dataset_code in DATASET_LANGUAGE_CODES.items():
        if normalized in {canonical, dataset_code}:
            return canonical
    return normalized


def _parse_teacher_generation(raw_generation: str) -> dict[str, str]:
    raw_generation = raw_generation.strip()
    reasoning_match = REASONING_BLOCK_RE.search(raw_generation)
    answer_match = ANSWER_RE.search(raw_generation)
    final_answer = _extract_answer_letter(raw_generation)

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    elif answer_match:
        reasoning = raw_generation[:answer_match.start()].strip()
    else:
        reasoning = raw_generation.strip()

    return {
        "reasoning": reasoning,
        "final_answer": final_answer,
        "raw_generation": raw_generation,
    }


def _extract_answer_letter(text: str) -> str:
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).upper()

    tail = text[-400:]
    match = FREEFORM_ANSWER_RE.search(tail)
    if match:
        return match.group(1).upper()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-5:]):
        match = TAIL_LETTER_RE.search(line)
        if match:
            return match.group(1).upper()
    return ""


def _batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start_idx in range(0, len(items), batch_size):
        yield items[start_idx: start_idx + batch_size]


if __name__ == "__main__":
    main()
