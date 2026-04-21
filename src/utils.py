from __future__ import annotations
from typing import Iterable
from vllm import LLM, SamplingParams

MMLU_TEMPLATE = '''Answer the following multiple choice question. The last line of your response should be of the following format: '#### ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}'''
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def format_mmlu(question, choices):
    choices_str = '\n'.join(
        f'({letter}) {choice}' for letter, choice in zip(LETTERS, choices)
    )
    return MMLU_TEMPLATE.format(question=question, choices=choices_str, letters=LETTERS[:len(choices)])


def build_vllm_prompt(tokenizer, messages):
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].capitalize()}: {message['content']}")
    rendered.append('Assistant:')
    return '\n'.join(rendered)


def load_vllm_llm(model_id, tensor_parallel_size: int = 1, **kwargs):
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        **kwargs,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def prompt_vllm(
    llm,
    tokenizer,
    batch_messages: Iterable[list[dict]],
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_p: float = 1.0,
    use_tqdm: bool = True,
):
    prompts = [build_vllm_prompt(tokenizer, messages) for messages in batch_messages]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    return [output.outputs[0].text if output.outputs else '' for output in outputs]


def prompt_vllm_with_logprobs(
    llm,
    tokenizer,
    batch_messages: Iterable[list[dict]],
    top_k: int,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_p: float = 1.0,
    use_tqdm: bool = True,
):
    """Like prompt_vllm but also returns top-K log-probs per generated token.

    Returns:
        texts: list of generated strings
        all_logprobs: list of lists; each inner list has one entry per generated
            token, and each entry is a list of (token_id, log_prob) pairs sorted
            by log_prob descending (length <= top_k).
    """
    prompts = [build_vllm_prompt(tokenizer, messages) for messages in batch_messages]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        logprobs=top_k,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)

    texts: list[str] = []
    all_logprobs: list[list[list[tuple[int, float]]]] = []
    all_token_ids: list[list[int]] = []
    for output in outputs:
        if not output.outputs:
            texts.append('')
            all_logprobs.append([])
            all_token_ids.append([])
            continue
        out = output.outputs[0]
        texts.append(out.text)
        # Capture the exact token IDs the teacher generated so callers can
        # bypass re-tokenization and guarantee 1:1 logit alignment.
        all_token_ids.append(list(out.token_ids))
        token_logprobs: list[list[tuple[int, float]]] = []
        for step_lps in (out.logprobs or []):
            # step_lps: dict[int, Logprob]; Logprob.logprob is the log-probability
            pairs = sorted(
                [(tok_id, lp.logprob) for tok_id, lp in step_lps.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
            token_logprobs.append(pairs)
        all_logprobs.append(token_logprobs)

    return texts, all_logprobs, all_token_ids
