from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from data.mmlupro import MMLUPro


LOGGER = logging.getLogger(__name__)

LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]

# default val sizes per language (total 850)
DEFAULT_NUM_SAMPLES = "300,200,150,100,100"
DEFAULT_SEED = 100   # different from dataset_generation.py seed (42)


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible validation set from held-out dataset rows"
    )
    parser.add_argument(
        "--train_data", default="outputs/train.jsonl",
        help="Path to train.jsonl — its questions are excluded from the val pool",
    )
    parser.add_argument(
        "--output_file", default="outputs/val.jsonl",
        help="Output path for the validation JSONL",
    )
    parser.add_argument(
        "--num_samples", default=DEFAULT_NUM_SAMPLES,
        help=f"Comma-separated val counts for english,hindi,bengali,kannada,tamil "
             f"(default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def _parse_counts(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != len(LANGUAGES):
        raise ValueError(
            f"--num_samples must have exactly {len(LANGUAGES)} comma-separated integers "
            f"for {','.join(LANGUAGES)}"
        )
    try:
        counts = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError("--num_samples values must be integers") from exc
    if any(c < 0 for c in counts):
        raise ValueError("--num_samples values must be >= 0")
    return counts


def _load_train_raw_questions(train_path: str) -> set[str]:
    used: set[str] = set()
    p = Path(train_path)
    if not p.exists():
        LOGGER.warning(
            "train_data file %s not found — using full dataset as val pool",
            train_path,
        )
        return used

    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            formatted_q = row.get("question", "")
            raw_q = formatted_q.split("\n\n")[0].strip()
            if raw_q:
                used.add(raw_q)

    LOGGER.info("Loaded %d train questions to exclude from val pool", len(used))
    return used


def _sample_language(
    language_name: str,
    count: int,
    train_questions: set[str],
    rng: random.Random,
) -> list[dict]:
    canonical = MMLUPro._canonical_language(language_name)
    dataset = MMLUPro(language=canonical, logger=LOGGER).load_mmlu_pro(unified=True)
    all_rows = list(dataset)

    remaining = [
        row for row in all_rows
        if row.get("question", "").strip() not in train_questions
    ]

    LOGGER.info(
        "%-10s  total=%d  held-out=%d  requested=%d",
        language_name, len(all_rows), len(remaining), count,
    )

    if count > len(remaining):
        LOGGER.warning(
            "%s: requested %d but only %d held-out rows available — using all of them",
            language_name, count, len(remaining),
        )
        count = len(remaining)

    if count == 0:
        return []

    rng.shuffle(remaining)
    return remaining[:count]


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    counts = _parse_counts(args.num_samples)
    LOGGER.info(
        "Val counts requested — %s",
        ", ".join(f"{lang}:{n}" for lang, n in zip(LANGUAGES, counts)),
    )

    train_questions = _load_train_raw_questions(args.train_data)
    rng = random.Random(args.seed)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    lang_written: dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as fh:
        for lang_name, count in zip(LANGUAGES, counts):
            if count == 0:
                LOGGER.info("Skipping %s (count=0)", lang_name)
                continue

            rows = _sample_language(lang_name, count, train_questions, rng)

            for row in rows:
                record = {
                    "question":    row["question"],
                    "options":     row["options"],
                    "answer":      row["answer"],
                    "gold_answer": row["answer"],
                    "language":    row["language"],
                    "subject":     row.get("subject", ""),
                    "source":      row.get("source_dataset", ""),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            lang_written[lang_name] = len(rows)
            total_written += len(rows)

    LOGGER.info("Validation set saved to %s", output_path)
    LOGGER.info("Total rows: %d", total_written)
    for lang, n in lang_written.items():
        LOGGER.info("  %-10s  %d rows", lang, n)


if __name__ == "__main__":
    main()
