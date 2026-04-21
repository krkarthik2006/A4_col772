# Part B Design Notes

This note explains the code and design decisions for **Part B: Knowledge Distillation** of the COL772 A4 assignment. It focuses mainly on [train_distill.py](train_distill.py), but also explains how it connects to [dataset_generation.py](dataset_generation.py), [make_val_set.py](make_val_set.py), and [inference_eval.py](inference_eval.py), because the distillation pipeline only makes sense end to end.

## Assignment Goal

Part B asks us to transfer reasoning behavior from a larger **teacher model** into smaller **student models**. The assignment explicitly highlights a difficult setting:

- cross-architecture distillation
- cross-tokenizer distillation
- multilingual reasoning
- compute constraints

Because Qwen2.5 and Llama-3.2 do not share a tokenizer or output vocabulary, **classical logits-based KD is inconvenient here**. Your implementation therefore makes a very sensible choice:

**use text-based offline chain-of-thought distillation**

Instead of matching teacher probabilities token by token, the code first generates teacher reasoning traces as text and then trains the student with supervised fine-tuning on those traces.

That is the central design idea of your Part B solution.

## What Your Pipeline Actually Does

Your current pipeline is:

1. [dataset_generation.py](dataset_generation.py) samples multilingual MCQs and asks the teacher to produce structured reasoning plus a final answer.
2. [make_val_set.py](make_val_set.py) builds a held-out validation set that does not overlap with train questions.
3. [train_distill.py](train_distill.py) converts teacher outputs into chat-formatted examples and SFT-trains a student with LoRA or QLoRA.
4. [inference_eval.py](inference_eval.py) runs the student with vLLM, extracts final answers, and reports language-wise accuracy.

So even though the assignment names Part B as "training pipeline", your actual implementation is better understood as an **offline distillation system** spanning data generation, training, and evaluation.

## Core Distillation Strategy

The strategy implemented in [train_distill.py](train_distill.py) is:

- train on teacher-generated reasoning traces
- preserve the same output structure used during teacher generation
- optionally keep only teacher-correct rows
- adapt the base student using LoRA/QLoRA instead of full fine-tuning
- support both Qwen-style and Llama-style chat templates

This is a good fit for the assignment because it avoids tokenizer projection across model families. The teacher supervises the student through plain text, which both architectures can consume.

## How The Data Is Prepared For Distillation

### 1. Teacher traces are generated in a controlled format

In [dataset_generation.py](dataset_generation.py), `format_teacher_prompt(...)` forces the teacher to:

- reason in the same language as the question
- wrap reasoning inside `<reasoning>...</reasoning>`
- end with `#### ANSWER: [LETTER]`

This is an important design choice. It makes the teacher output easy to parse and also gives the student a stable response format to imitate later.

### 2. Sampling is balanced across languages and roughly balanced across subjects

The script uses:

- `LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]`
- `sample_datasets(...)` for language-wise control
- `_subject_balanced_sample(...)` for round-robin category balancing

This is a strong decision because multilingual datasets often become English-heavy or dominated by a few subject clusters. Your sampler tries to reduce both problems before training even starts.

### 3. The code filters bad teacher data early

Each saved row contains:

- the parsed teacher reasoning
- the teacher final answer
- the gold answer
- whether the teacher answer was parsed
- whether the teacher was correct

Then `dataset_generation.py` can skip:

- unparsed examples
- incorrect teacher examples

This matters because distillation quality depends heavily on teacher quality. A small model can easily overfit bad reasoning if the training set is noisy.

## Detailed Explanation Of `train_distill.py`

### 1. CLI design

`parse_args()` exposes the main controls:

- base student model path
- train JSONL path
- output directory
- LoRA settings
- learning rate, epochs, batch size, max length
- optional prompt masking
- optional validation set
- optional QLoRA

One thing to notice: `--teacher_model` is accepted, but in the current implementation it is **not used during training**. The help string already says it is reserved for online distillation. So this script is purely **offline distillation**, not online KD.

That is completely valid for the assignment, but it is worth stating clearly in your report.

### 2. Model family detection

`detect_model_family(model_path)` returns either `"llama"` or `"qwen"`.

Why this exists:

- Qwen and Llama use different chat templates
- prompt masking depends on finding the assistant-turn boundary correctly
- `DataCollatorForCompletionOnlyLM` needs the exact response prefix token ids

So the code stores:

- Qwen response template: `<|im_start|>assistant\n`
- Llama response template: `<|start_header_id|>assistant<|end_header_id|>\n\n`

This is a small but important cross-architecture design decision. Without it, masked-loss training can silently fail because the collator would mask the wrong span.

### 3. JSONL loading and text construction

`load_jsonl(...)` reads the teacher-produced training rows.

`build_training_text(row, tokenizer)` is the key formatting function. It builds a two-turn chat example:

- user: the full MCQ question with options
- assistant: the teacher reasoning plus the final answer

The assistant text is reconstructed as:

```text
<reasoning>
...
</reasoning>
#### ANSWER: (X)
```

This is one of the best choices in the script.

Why:

- the student is trained on the exact format expected at inference time
- reasoning and answer are not learned as unrelated outputs
- the model learns both "how to think" and "how to finish"

It is essentially sequence-level imitation learning over teacher traces.

### 4. Dataset preparation

`prepare_dataset(...)` loads rows and optionally drops examples where `teacher_correct == False`.

Then it converts every row into a single `text` field for TRL SFT training.

Why this design is reasonable:

- TRL's `SFTTrainer` works naturally with one text field
- chat-template rendering is delegated to the tokenizer, so Qwen and Llama stay compatible
- correctness filtering is handled before training, which keeps the trainer simple

The key trade-off is:

- filtering incorrect rows improves supervision quality
- but removes data volume

In your setup, this trade-off is exposed cleanly as `--filter_correct_only`.

### 5. LoRA and QLoRA configuration

`get_bnb_config()` sets 4-bit NF4 quantization for QLoRA.

`get_lora_config(args)` attaches LoRA to:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

This is a strong default because it covers:

- attention projections
- MLP projections

So the student can adapt both attention behavior and feed-forward transformations without full fine-tuning.

Why PEFT is a good choice here:

- student models are small, but training with long CoT sequences is still memory-heavy
- assignment compute is limited
- LoRA lets you adapt the model cheaply
- QLoRA further reduces memory pressure

This is especially useful for sequence lengths like `2048`, where full fine-tuning would be much harder on limited hardware.

### 6. Student loading

`load_student(args)` does a few practical things:

- loads tokenizer and model with `trust_remote_code=True`
- inserts a pad token if the tokenizer does not define one
- sets right-padding
- loads BF16 weights
- optionally applies 4-bit quantization
- disables `use_cache`
- calls `prepare_model_for_kbit_training(...)` when QLoRA is used

Each of these has a reason:

- missing pad tokens are common in chat LLMs
- right-padding is safer for batched causal LM SFT
- `use_cache=False` is needed when gradient checkpointing is enabled
- `prepare_model_for_kbit_training` makes low-bit training stable

These are not flashy design decisions, but they are the kind that prevent training from breaking.

### 7. Prompt masking

If `--mask_prompt_tokens` is enabled, the script uses `DataCollatorForCompletionOnlyLM`.

What this does:

- tokens before the assistant response are masked from the loss
- loss is computed only on the student answer span

That means the model is optimized to generate:

- reasoning
- final answer

and not to waste capacity learning to copy the user prompt.

This is conceptually better for distillation than full-sequence loss, especially when prompts are long MCQs.

Important nuance:

- your code supports masking
- but it is **off by default**

That is fine for flexibility, but from a distillation perspective, enabling it is usually the better choice.

### 8. Training configuration

The `SFTConfig` uses:

- `lr_scheduler_type="cosine"`
- `warmup_ratio=0.03`
- `gradient_checkpointing=True`
- `bf16=True`
- epoch-based saving
- optional epoch-based evaluation

This is a standard and reasonable setup for LoRA fine-tuning.

Good design choices here:

- cosine decay is a safe default for adapter tuning
- warmup helps stabilize early training
- gradient accumulation increases effective batch size
- checkpointing makes long-sequence training feasible
- validation support is built in if `--val_data` is provided

The training objective is plain supervised next-token prediction over teacher traces. That keeps the method simple, reproducible, and easy to defend in a report.

### 9. Saving strategy

After training, the script saves:

- the LoRA adapter
- the tokenizer
- optionally a merged full model

This is practical because:

- adapters are lightweight and fast to store
- merged weights are easier for inference systems that do not handle PEFT well

That design connects nicely with [inference_eval.py](inference_eval.py), which can either:

- load a merged model directly
- or merge the adapter into the base model first

## Why This Strategy Fits The Assignment

Your implementation is well aligned with the assignment constraints:

### 1. It handles cross-family distillation cleanly

Since Qwen and Llama do not share token vocabularies, text-based supervision is the simplest reliable bridge. The teacher teaches through language, not through aligned logits.

### 2. It supports multilingual transfer

Teacher reasoning is explicitly requested in the question language, so the student sees:

- multilingual inputs
- multilingual reasoning
- language-matched answers

This is much better than translating everything into English and hoping transfer happens automatically.

### 3. It is compute-aware

Using LoRA/QLoRA instead of full fine-tuning is a good engineering decision for this assignment. It lets you focus your compute budget on sequence length and data scale rather than on storing full gradients for the entire model.

### 4. It is robust to imperfect teacher generations

The data generation stage retries malformed outputs and can filter wrong ones. This makes the training corpus more trustworthy.

## Function-By-Function Reading Guide

If you want to read the training code quickly, this is the best order:

1. `parse_args()`  
   Understand the training knobs.

2. `build_training_text(...)`  
   This is the heart of the distillation format.

3. `prepare_dataset(...)`  
   This shows how JSONL becomes trainer-ready text.

4. `get_lora_config(...)` and `load_student(...)`  
   These explain the PEFT setup.

5. `main()`  
   This wires together dataset loading, masking, trainer config, training, and saving.

## Strengths Of Your Current Design

The strongest parts of your Part B solution are:

- **correct problem framing**: you chose text distillation instead of forcing incompatible logits KD
- **good data hygiene**: parsed/correctness metadata is preserved
- **multilingual awareness**: language-specific teacher reasoning is enforced
- **PEFT practicality**: LoRA/QLoRA makes training feasible
- **template consistency**: train and inference formats are aligned
- **evaluation discipline**: you report language-wise accuracy, which the assignment explicitly asks for

## Current Limitations

These are not mistakes, but they are the main limitations of the current approach:

### 1. Training is still just SFT on teacher traces

The script does not implement:

- online distillation
- confidence weighting
- answer-only auxiliary loss
- contrastive preference learning
- self-consistency distillation

So the method is strong and clean, but still a fairly standard offline KD pipeline.

### 2. `--teacher_model` is unused in Part B training

This means the training script itself does not yet exploit the assignment's allowance to use the teacher during training.

### 3. Prompt masking is optional, not default

For long MCQs, unmasked loss can spend capacity modeling prompt tokens rather than the reasoning trace.

### 4. Validation data is separate, but not subject-balanced

[make_val_set.py](make_val_set.py) deduplicates against train questions, which is excellent, but it samples held-out rows with random shuffling rather than the subject-balancing logic used in training-data generation.

### 5. There is no explicit language or subject weighting during training

The dataset may be balanced at creation time, but training still treats all examples equally. Some languages or subjects may remain harder and could benefit from oversampling or weighted loss.

## What You Can Do Better To Improve Model Performance

Below are the most useful improvements, ordered roughly from highest practical value to more experimental ideas.

### 1. Turn on prompt masking by default

Best low-effort improvement.

Why it should help:

- reduces wasted loss on user prompt tokens
- focuses optimization on reasoning and answer generation
- usually helps instruction-following style SFT

What to change:

- train with `--mask_prompt_tokens`
- or make it the default in `train_distill.py`

### 2. Always train on a filtered high-quality subset first

A good strategy is:

1. first train on `teacher_correct == True`
2. then optionally continue on a larger mixed set

Why:

- early training benefits from very clean supervision
- later training can use extra diversity

This is often better than training on noisy and clean data mixed from the start.

### 3. Add a short answer-focused second stage

Right now the model learns reasoning and answer together. A useful improvement is:

- Stage 1: train on full reasoning traces
- Stage 2: continue training on shorter outputs where only the final answer line is supervised

Why:

- long CoT helps reasoning behavior
- short answer tuning sharpens classification accuracy
- it reduces cases where the model produces plausible reasoning but ends with the wrong option letter

### 4. Use validation-driven hyperparameter search

You already created [make_val_set.py](make_val_set.py), which is excellent. Use it more aggressively.

The first hyperparameters worth sweeping are:

- learning rate: `1e-4`, `2e-4`, `3e-4`
- epochs: `2`, `3`, `4`
- LoRA rank: `8`, `16`, `32`
- max length: `1024` vs `1536` vs `2048`
- filtered vs unfiltered train data
- masking on vs off

This will probably improve performance more than adding fancy algorithms blindly.

### 5. Reweight hard languages

Even with language-balanced sampling, model difficulty is not language-balanced.

If Tamil, Kannada, or Bengali lag behind, try:

- oversampling those languages during training
- training separate adapters per student and per balancing scheme
- using a curriculum where non-English languages are upweighted late in training

This matters because multilingual reasoning quality usually drops faster in low-resource scripts than in English.

### 6. Reweight hard subjects

Your subject-balanced sampler is already a very good start. The next step is:

- measure accuracy by subject on the validation set
- oversample weak categories such as law, physics, math, or medicine

This is especially useful because MMLU-style benchmarks often have very uneven subject difficulty.

### 7. Add teacher self-consistency before saving traces

Instead of one teacher generation per question, try:

- sample 2-4 teacher outputs for a smaller subset
- keep examples where final answers agree
- prefer the most consistent trace

Why:

- teacher agreement is a cheap proxy for confidence
- self-consistent traces are often cleaner than single-shot outputs

This is one of the best ways to improve data quality if teacher inference budget allows it.

### 8. Use confidence-weighted training

You do not currently score teacher confidence. A simple approximation is:

- weight correct, well-formatted, concise traces more
- weight long or weakly parsed traces less

Possible proxies:

- parsed answer success
- teacher correctness
- trace length
- self-consistency agreement

This can outperform strict keep/drop filtering because it preserves more data without trusting all rows equally.

### 9. Mix in gold-answer-only supervision from the original dataset

The assignment allows using the full training set in addition to teacher outputs.

A strong hybrid strategy is:

- use teacher traces for reasoning imitation
- use original dataset rows for short answer-only supervision

Why:

- teacher traces teach process
- gold labels teach exact classification
- mixing both can improve final answer accuracy

This is one of the most important upgrades if you want to push performance beyond pure offline CoT imitation.

### 10. Add online KD or bootstrapped distillation

This is a bigger upgrade, but it would make your method more novel.

Examples:

- periodically re-query the teacher on student mistakes
- train student, evaluate on val, collect failure cases, then generate new teacher traces only for those cases
- use the teacher only on hard multilingual or hard-subject examples

That would let the training pipeline use teacher budget more intelligently than one-shot offline generation.

### 11. Compare cross-family vs in-family settings explicitly

The report asks you to discuss:

- Qwen teacher -> Qwen student
- Qwen teacher -> Llama student

To make that discussion stronger, compare:

- same data
- same hyperparameters where possible
- same train/val split

Then analyze:

- which model copies reasoning style better
- which model gets better final answer accuracy
- whether cross-family transfer hurts more in non-English languages

This may not directly improve the model, but it will greatly improve the quality of your final report.

## Best Practical Recipe I Would Try Next

If the goal is simply to improve scores with minimal engineering risk, this is the sequence I would try:

1. generate a clean teacher dataset with incorrect and unparsed rows removed
2. train with `--mask_prompt_tokens`
3. use `--filter_correct_only`
4. tune LR, epochs, and LoRA rank on `val.jsonl`
5. run a second short stage on answer-only supervision
6. oversample the weakest languages based on validation metrics

That path is realistic, easy to justify, and likely to produce measurable gains.

## Suggested Report Positioning

If you are writing the final report, a good short description of your method is:

> We use offline multilingual chain-of-thought distillation. The teacher generates language-matched reasoning traces in a structured format, and the student is fine-tuned with LoRA/QLoRA using supervised next-token prediction over the teacher response. This avoids vocabulary-alignment issues in cross-family distillation and keeps training feasible under limited compute.

That description matches your code honestly and presents the method well.

## Final Take

Your Part B code is built on a solid idea: **distill reasoning as text, not logits**. That is the right design choice for multilingual cross-architecture distillation in this assignment.

The current implementation is already strong in terms of engineering judgment:

- clean formatting
- PEFT-aware training
- multilingual supervision
- train/val separation
- language-wise evaluation

The biggest remaining gains will likely come from:

- cleaner teacher data
- prompt-masked training
- stronger validation-based tuning
- hybrid supervision that combines reasoning traces with gold-answer learning

If you improve those four areas, your model should become both more accurate and easier to defend in the report.
