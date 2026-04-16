from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger(__name__)


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Starter distillation training loop")
    parser.add_argument("--student_model", required=True,
                        help="Base student model path")
    parser.add_argument("--teacher_model", required=False,
                        help="Teacher model path")
    parser.add_argument("--train_data", default="data/train.jsonl",
                        help="Path to train JSONL with prompt and teacher_generation")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save trained weights")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Epoch count")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int,
                        default=2048, help="Max sequence length")
    parser.add_argument(
        "--mask_prompt_tokens",
        action="store_true",
        help="Mask prompt tokens so loss is only on reasoning + final answer",
    )
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
