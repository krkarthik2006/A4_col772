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
