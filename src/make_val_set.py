"""
make_val_set.py — Create a reproducible validation set from held-out rows
                  (rows in dataset.jsonl that were NOT sampled into train.jsonl).

Why:
  - The val set lets you tune hyperparameters and detect overfitting during
    training via inference_eval.py without touching the private test set.
  - Excluding train rows prevents data leakage.
  - A fixed seed makes the split reproducible across machines and runs.

Typical usage:
    # Build the val set after running dataset_generation.py
    python make_val_set.py \\
        --train_data data/train.jsonl \\
        --output_file data/val.jsonl \\
        --num_samples 300,200,150,100,100

    # Then evaluate a trained model on it:
    python inference_eval.py \\
        --base_model outputs/distilled_qwen \\
        --test_data data/val.jsonl \\
        --output_predictions outputs/val_predictions.jsonl \\
        --report_file outputs/val_metrics.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from data.mmlupro import MMLUPro


LOGGER = logging.getLogger(__name__)

LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]

# Default val size per language (total 500, kept small so train budget is preserved).
# Adjust freely — just don't exceed available held-out rows.
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
        "--train_data", default="data/train.jsonl",
        help="Path to train.jsonl produced by dataset_generation.py. "
             "Its questions are excluded from the val pool. "
             "If the file does not exist the whole dataset is used as val pool.",
    )
    parser.add_argument(
        "--output_file", default="data/val.jsonl",
        help="Output path for the validation JSONL",
    )
    parser.add_argument(
        "--num_samples", default=DEFAULT_NUM_SAMPLES,
        help=(
            "Comma-separated val sample counts for "
            "english,hindi,bengali,kannada,tamil "
            f"(default: {DEFAULT_NUM_SAMPLES})"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """
    Return the set of raw question strings that were used in training.

    train.jsonl stores the question in "formatted" form:
        "<raw_question>\\n\\n(A) opt1\\n(B) opt2\\n..."
    So we split on the first double-newline to recover the raw question.
    """
    used: set[str] = set()
    p = Path(train_path)
    if not p.exists():
        LOGGER.warning(
            "train_data file %s not found — using full dataset as val pool "
            "(no train/val deduplication).",
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
            # Split on first blank line separating question from choices
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
    """
    Load all rows for one language, exclude those in train, sample `count` rows.
    Returns a list of raw dataset dicts (with original question + options fields).
    """
    canonical = MMLUPro._canonical_language(language_name)
    dataset = MMLUPro(language=canonical, logger=LOGGER).load_mmlu_pro(unified=True)
    all_rows = list(dataset)

    # Exclude rows whose raw question string appears in the train set
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


# ── Main ──────────────────────────────────────────────────────────────────────

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
                # Store both raw question + options so inference_eval.py can
                # call format_mmlu() — and gold_answer for accuracy computation.
                record = {
                    "question":    row["question"],
                    "options":     row["options"],
                    "answer":      row["answer"],       # letter A–J
                    "gold_answer": row["answer"],       # alias used by inference_eval.py
                    "language":    row["language"],     # canonical code (en, hindi, …)
                    "subject":     row.get("subject", ""),
                    "source":      row.get("source_dataset", ""),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            lang_written[lang_name] = len(rows)
            total_written += len(rows)

    LOGGER.info("─" * 50)
    LOGGER.info("Validation set saved to %s", output_path)
    LOGGER.info("Total rows: %d", total_written)
    for lang, n in lang_written.items():
        LOGGER.info("  %-10s  %d rows", lang, n)


if __name__ == "__main__":
    main()
