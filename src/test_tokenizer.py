from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
template = "<|im_start|>assistant\n"
template_ids = tokenizer.encode(template, add_special_tokens=False)
print("Template ids:", template_ids)

messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
full_ids = tokenizer.encode(text, add_special_tokens=False)
print("Full ids:", full_ids)

def _find_sublist(haystack: list, needle: list) -> int:
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1

pos = _find_sublist(full_ids, template_ids)
print("Found at:", pos)
