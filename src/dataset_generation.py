from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, concatenate_datasets
from tqdm.auto import tqdm

from data.mmlupro import MMLUPro
from utils import load_vllm_llm, prompt_vllm, prompt_vllm_with_logprobs


# ── HYPERPARAMETERS ────────────────────────────────────────────────────────
# Edit values here. CLI flags override these when explicitly passed.

# Samples per language: english, hindi, bengali, kannada, tamil.
# Equal split (2000 each) gives every language identical training signal.
# Tamil has exactly 2000 rows available — raising any value above its
# language cap will raise an error at runtime.
DEFAULT_NUM_SAMPLES = "2000,2000,2000,2000,2000"

DEFAULT_BATCH_SIZE        = 32     # teacher inference batch size
DEFAULT_MAX_NEW_TOKENS    = 1024   # per-question token budget (hard cap: 2048)
DEFAULT_TEMPERATURE       = 0.0    # 0.0 = greedy / deterministic
DEFAULT_TOP_P             = 1.0
DEFAULT_MAX_RETRIES       = 2      # re-query attempts for unparsed outputs
DEFAULT_RETRY_TEMPERATURE = 0.4    # temperature injected for retry diversity
DEFAULT_GPU_MEM_UTIL      = 0.6    # vLLM GPU memory fraction (lower if OOM)
DEFAULT_TENSOR_PARALLEL   = 1      # set > 1 for multi-GPU tensor parallelism
DEFAULT_SEED              = 42

# True  → discard rows where teacher answer ≠ gold (recommended)
# False → keep them; teacher_correct flag remains set for post-hoc analysis
DEFAULT_FILTER_INCORRECT  = True

# True  → discard rows where no answer letter could be extracted after retries
DEFAULT_SKIP_UNPARSED     = True

# Number of top-K logprobs to save per generated token (0 = disabled).
# When > 0 vLLM stores the top-K (token_id, log_prob) pairs alongside each row
# so the student can be trained with soft-label KD instead of pure SFT.
DEFAULT_TOP_K_LOGITS      = 0

# ───────────────────────────────────────────────────────────────────────────

LOGGER = logging.getLogger(__name__)
LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
GENERATION_SYSTEM_MESSAGE = (
    "You are a careful multilingual reasoning assistant that follows "
    "the requested output format exactly."
)

# Human-readable names keyed by MMLUPro canonical codes ("en", "hindi", …)
LANGUAGE_NAMES = {
    "en":      "English",
    "hindi":   "Hindi",
    "bengali": "Bengali",
    "kannada": "Kannada",
    "tamil":   "Tamil",
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

# Ordered (category, pattern) pairs derived from the actual subject strings in
# data/dataset.jsonl. First match wins; unmatched rows fall into "other".
#
# Ordering rules that prevent false matches:
#   • law before everything (professional_law = 17% of dataset)
#   • business_econ before chemistry (avoid "chemical engineering" going to chem)
#   • chemistry before physics (PhysicalChemistry must not go to physics)
#   • computer_science before engineering (machine_learning before machine)
#   • engineering before physics (FluidMechanics stays in engineering, not physics)
SUBJECT_CATEGORY_PATTERNS: list[tuple[str, re.Pattern]] = [
    # professional_law = 1 726 rows (~17 %); jurisprudence, international_law
    ("law",             re.compile(r"law|juris", re.I)),
    # stemez-Business/Economics, macroeconomics, accounting, Finance, marketing,
    # management, public_relations, econometrics
    ("business_econ",   re.compile(r"business|econom|financ|account|market|management|public_relat", re.I)),
    # stemez-Chemistry/PhysicalChemistry/OrganicChemistry, scibench-atkins/chemmc
    ("chemistry",       re.compile(r"chem|atkins", re.I)),
    # before engineering so machine_learning goes here, not to "machine"
    ("computer_science", re.compile(r"comput|program|algorithm|machine.learn|software|database|computer.secur", re.I)),
    # stemez-Electric*/Electromagnetics/HeatTransfer/MachineDesign/FluidMechanics/
    # TransportPhenomena, theoremQA-EECS, ori_mmlu-electrical_engineering
    ("engineering",     re.compile(r"electr|heat|machine|eecs|fluid|transport", re.I)),
    # after engineering so Fluid/Electric subjects stay there; picks up
    # stemez-Physics/Optics/Thermodynamics, scibench-thermo/class/quan/fund/matter,
    # ori_mmlu-astronomy, stemez-Mechanics
    ("physics",         re.compile(r"physic|optic|thermo|quan|astronom|mechanic|matter|fund|\bclass\b|relativity", re.I)),
    # math, algebra, calculus, scibench-stat/diff, formal_logic, statistics
    ("mathematics",     re.compile(r"math|algebra|calcul|stat|arithmetic|formal.logic|\bdiff\b", re.I)),
    # stemez-Biology/Genetics, ori_mmlu-high_school_biology/college_biology/
    # medical_genetics/anatomy/virology
    ("biology",         re.compile(r"bio|anatomy|genetic|physiolog|ecolog|evolution|virology", re.I)),
    # professional_medicine, nutrition, clinical_knowledge, human_aging, sexuality
    ("medicine",        re.compile(r"medic|clinical|health|pharmac|nurs|disease|diagno|surgery|aging|sexuality|nutrit", re.I)),
    ("psychology",      re.compile(r"psycholog|cognitive|behavior|mental|neurosci", re.I)),
    # prehistory matches via "histor" substring
    ("history",         re.compile(r"histor|ancient|medieval|civiliz", re.I)),
    # global_facts, security_studies, government_and_politics, geography,
    # world_religions, us_foreign_policy, sociology
    ("social_science",  re.compile(r"social|sociolog|anthropolog|politic|geograph|global|religion|securit|foreign.polic", re.I)),
    # philosophy, moral_disputes, logical_fallacies; formal_logic caught by math above
    ("philosophy",      re.compile(r"philosoph|ethic|logic|moral|metaphys|fallaci", re.I)),
]


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


def _categorize_subject(subject: str) -> str:
    """Map a raw subject string to a broad category using regex. Returns 'other' if no match."""
    subject = subject or ""
    for category, pattern in SUBJECT_CATEGORY_PATTERNS:
        if pattern.search(subject):
            return category
    return "other"


def _subject_balanced_sample(dataset: Dataset, n: int, seed: int) -> Dataset:
    """
    Sample n rows with even representation across subject categories.
    Uses round-robin so no single subject dominates.
    """
    import random
    rng = random.Random(seed)

    category_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(dataset):
        cat = _categorize_subject(row.get("subject", ""))
        category_indices[cat].append(i)

    for indices in category_indices.values():
        rng.shuffle(indices)

    categories = sorted(category_indices.keys())
    cat_cursors = {cat: 0 for cat in categories}
    selected: list[int] = []

    while len(selected) < n:
        made_progress = False
        for cat in categories:
            if len(selected) >= n:
                break
            cursor = cat_cursors[cat]
            indices = category_indices[cat]
            if cursor < len(indices):
                selected.append(indices[cursor])
                cat_cursors[cat] += 1
                made_progress = True
        if not made_progress:
            break

    return dataset.select(selected[:n])


def sample_datasets(
        samples_per_language: list[int],
        split: str = "test",
        seed: int = 42,
) -> Dataset:
    """Fetch and sample the requested number of rows for each language."""
    total_requested = sum(samples_per_language)
    if len(samples_per_language) != len(LANGUAGES):
        raise ValueError(
            "--num_samples must contain 5 comma-separated integers for "
            "english,hindi,bengali,kannada,tamil"
        )
    if any(count < 0 for count in samples_per_language):
        raise ValueError("--num_samples values must be >= 0")
    if total_requested <= 0:
        raise ValueError("--num_samples must request at least one sample")
    if total_requested > 10_000:
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

        dataset_language = MMLUPro._canonical_language(language_name)
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

        sampled_language_dataset = _subject_balanced_sample(
            dataset, sample_count, seed=seed + idx
        )
        LOGGER.info(
            "Sampled %d/%d rows for %s (subject-balanced)",
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
    canonical = MMLUPro._canonical_language(language)
    language_name = LANGUAGE_NAMES.get(canonical, canonical.title())
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


def _build_batch_messages(prompts: list[str]) -> list[list[dict[str, str]]]:
    return [
        [
            {"role": "system", "content": GENERATION_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]


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
    raw_generations = prompt_vllm(
        model,
        tokenizer,
        batch_messages=_build_batch_messages(prompts),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_tqdm=False,
    )

    return [_parse_teacher_generation(text) for text in raw_generations]


def generate_and_parse_batch_with_logprobs(
        model,
        tokenizer,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
) -> tuple[list[dict[str, str]], list[list], list[list[int]]]:
    """Like generate_and_parse_batch but also returns per-token top-K log-probs
    and the exact token IDs generated by the teacher for BPE-aligned KD."""
    raw_generations, all_logprobs, all_token_ids = prompt_vllm_with_logprobs(
        model,
        tokenizer,
        batch_messages=_build_batch_messages(prompts),
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_tqdm=False,
    )
    return [_parse_teacher_generation(text) for text in raw_generations], all_logprobs, all_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query teacher and build train corpus JSONL"
    )
    # ── required infrastructure args (no top-of-file default) ──
    parser.add_argument("--teacher_model", required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output_file", required=True,
                        help="Output JSONL path for train corpus")

    # ── hyperparameter overrides (default=None → falls back to top-of-file constants) ──
    parser.add_argument("--num_samples", type=str, default=None,
                        help="Comma-separated counts for en,hi,bn,kn,ta "
                             f"(default: {DEFAULT_NUM_SAMPLES})")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_retries", type=int, default=None)
    parser.add_argument("--retry_temperature", type=float, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    # store_true with default=None: not passed → None (use constant); passed → True
    parser.add_argument("--filter_incorrect", action="store_true", default=None)
    parser.add_argument("--skip_unparsed", action="store_true", default=None)
    parser.add_argument("--top_k_logits", type=int, default=None,
                        help="Save top-K logprobs per token for soft-label KD "
                             f"(0 = disabled, default: {DEFAULT_TOP_K_LOGITS})")

    # ── non-hyperparameter flags ──
    parser.add_argument("--split", default="test")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
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
    options = list(row["options"])
    return f"{row['question']}\n\n{_options_to_text(options)}"


def _resolve_arg(value, default):
    return default if value is None else value


def _retry_unparsed(
        teacher,
        tokenizer,
        parsed_batch: list[dict[str, str]],
        prompts: list[str],
        *,
        max_new_tokens: int,
        retry_temperature: float,
        top_p: float,
        max_retries: int,
        top_k_logits: int = 0,
        batch_logprobs: list | None = None,
        batch_token_ids: list | None = None,
) -> list[dict[str, str]]:
    """Re-query teacher for any outputs where the answer could not be parsed.

    When top_k_logits > 0 and batch_logprobs is provided, the retry uses
    generate_and_parse_batch_with_logprobs and updates batch_logprobs in-place
    so logit arrays stay aligned with their corresponding text outputs.
    """
    for attempt in range(max_retries):
        failed = [i for i, p in enumerate(parsed_batch) if not p["final_answer"]]
        if not failed:
            break
        LOGGER.debug(
            "Retry %d/%d: re-querying %d unparsed outputs",
            attempt + 1, max_retries, len(failed),
        )
        if top_k_logits > 0 and batch_logprobs is not None:
            retry_results, retry_logprobs, retry_token_ids = generate_and_parse_batch_with_logprobs(
                teacher,
                tokenizer,
                [prompts[i] for i in failed],
                max_new_tokens=max_new_tokens,
                temperature=retry_temperature,
                top_p=top_p,
                top_k=top_k_logits,
            )
            for i, result, lp, tid in zip(failed, retry_results, retry_logprobs, retry_token_ids):
                if result["final_answer"]:
                    parsed_batch[i] = result
                    batch_logprobs[i] = lp
                    if batch_token_ids is not None:
                        batch_token_ids[i] = tid
        else:
            retry_results = generate_and_parse_batch(
                teacher,
                tokenizer,
                [prompts[i] for i in failed],
                max_new_tokens=max_new_tokens,
                temperature=retry_temperature,
                top_p=top_p,
            )
            for i, result in zip(failed, retry_results):
                if result["final_answer"]:
                    parsed_batch[i] = result
    return parsed_batch


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    # Resolve each hyperparameter: CLI value if explicitly passed, else top-of-file constant.
    num_samples_str = _resolve_arg(args.num_samples, DEFAULT_NUM_SAMPLES)
    batch_size = _resolve_arg(args.batch_size, DEFAULT_BATCH_SIZE)
    max_new_tokens = _resolve_arg(args.max_new_tokens, DEFAULT_MAX_NEW_TOKENS)
    temperature = _resolve_arg(args.temperature, DEFAULT_TEMPERATURE)
    top_p = _resolve_arg(args.top_p, DEFAULT_TOP_P)
    max_retries = _resolve_arg(args.max_retries, DEFAULT_MAX_RETRIES)
    retry_temperature = _resolve_arg(
        args.retry_temperature, DEFAULT_RETRY_TEMPERATURE
    )
    gpu_mem_util = _resolve_arg(
        args.gpu_memory_utilization, DEFAULT_GPU_MEM_UTIL
    )
    tensor_parallel = _resolve_arg(
        args.tensor_parallel_size, DEFAULT_TENSOR_PARALLEL
    )
    seed = _resolve_arg(args.seed, DEFAULT_SEED)
    filter_incorrect = _resolve_arg(
        args.filter_incorrect, DEFAULT_FILTER_INCORRECT
    )
    skip_unparsed = _resolve_arg(args.skip_unparsed, DEFAULT_SKIP_UNPARSED)
    top_k_logits = _resolve_arg(args.top_k_logits, DEFAULT_TOP_K_LOGITS)

    samples_per_language = _parse_num_samples(num_samples_str)
    if max_new_tokens <= 0 or max_new_tokens > 2048:
        raise ValueError("--max_new_tokens must be in the range [1, 2048]")
    if batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sampled = sample_datasets(
        samples_per_language=samples_per_language,
        split=args.split,
        seed=seed,
    )
    LOGGER.info("Collected %d samples", len(sampled))

    teacher, tokenizer = load_vllm_llm(
        model_id=args.teacher_model,
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_mem_util,
    )

    written = 0
    skipped_unparsed = 0
    skipped_incorrect = 0
    with output_path.open("w", encoding="utf-8") as fp:
        sampled_rows = list(sampled)
        total_batches = (len(sampled_rows) + batch_size - 1) // batch_size
        for batch_rows in tqdm(
            _batched(sampled_rows, batch_size),
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

            if top_k_logits > 0:
                parsed_batch, batch_logprobs, batch_token_ids = generate_and_parse_batch_with_logprobs(
                    teacher,
                    tokenizer,
                    prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k_logits,
                )
            else:
                parsed_batch = generate_and_parse_batch(
                    teacher,
                    tokenizer,
                    prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                batch_logprobs = [None] * len(parsed_batch)
                batch_token_ids = [None] * len(parsed_batch)

            parsed_batch = _retry_unparsed(
                teacher,
                tokenizer,
                parsed_batch,
                prompts,
                max_new_tokens=max_new_tokens,
                retry_temperature=retry_temperature,
                top_p=top_p,
                max_retries=max_retries,
                top_k_logits=top_k_logits,
                batch_logprobs=batch_logprobs,
                batch_token_ids=batch_token_ids,
            )

            for row, question_with_choices, prompt, parsed, token_logprobs, token_ids in zip(
                batch_rows, questions_with_choices, prompts, parsed_batch, batch_logprobs, batch_token_ids
            ):
                gold_answer = str(row.get("answer", "")).upper()[:1]
                teacher_answer = parsed["final_answer"]
                answer_parsed = bool(teacher_answer)
                teacher_correct = answer_parsed and teacher_answer == gold_answer

                if skip_unparsed and not answer_parsed:
                    skipped_unparsed += 1
                    continue
                if filter_incorrect and answer_parsed and not teacher_correct:
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
                    "teacher_top_k_logits": token_logprobs,
                    "teacher_generated_ids": token_ids,
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


def _parse_teacher_generation(raw_generation: str) -> dict[str, str]:
    raw_generation = raw_generation.strip()
    reasoning_match = REASONING_BLOCK_RE.search(raw_generation)
    answer_match = ANSWER_RE.search(raw_generation)
    final_answer = _extract_answer_letter(raw_generation)

    reasoning = raw_generation
    if reasoning_match:
        reasoning = reasoning_match.group(1)
    elif answer_match:
        reasoning = raw_generation[:answer_match.start()]

    return {
        "reasoning": reasoning.strip(),
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
