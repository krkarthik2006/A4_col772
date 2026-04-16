from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_vllm_llm, prompt_vllm


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
LANGUAGE_LABELS = {
    "english": "English",
    "hindi": "Hindi",
    "bengali": "Bengali",
    "kannada": "Kannada",
    "tamil": "Tamil",
}
ANSWER_TAG_RE = re.compile(r"####\s*ANSWER\s*:\s*([A-J])", re.IGNORECASE)
LAST_LINE_LETTER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference + eval on test JSONL")
    parser.add_argument("--base_model", required=True,
                        help="Base student model path")
    parser.add_argument("--adapter_path", default="",
                        help="Optional PEFT adapter path")
    parser.add_argument("--test_data", required=True, help="Test JSONL path")
    parser.add_argument("--output_predictions", required=True,
                        help="Predictions JSONL path")
    parser.add_argument("--report_file", required=True,
                        help="Metrics report text file")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)


if __name__ == "__main__":
    main()
