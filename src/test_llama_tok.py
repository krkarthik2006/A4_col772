import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
template_ids = tokenizer.encode(template, add_special_tokens=False)
print("Template ids:", template_ids)

messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "<reasoning>\nHi\n</reasoning>\n#### ANSWER: A"}
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
if pos == -1:
    # See where the header id is
    print("Tokens around where it should be:")
    try:
        idx = full_ids.index(128006) # start header id
        idx2 = full_ids.index(128006, idx+1)
        print("Assistant start:", full_ids[idx2:idx2+10])
    except:
        pass
