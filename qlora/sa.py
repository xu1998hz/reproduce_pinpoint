import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import math

# TODO: I think the scoe is not from comet, just calculated from error types

path = "/ocean/projects/cis230075p/gzhu/output/checkpoint-300"
device = 'cuda:0'

# load the feedback model
feedback_tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
if feedback_tokenizer.pad_token is None:
    print('error! ')

feedback_model = AutoModelForCausalLM.from_pretrained(path, device_map=device, torch_dtype=torch.float16)

# load regular mistral 7b instruct model
base_name = 'mistralai/Mistral-7B-Instruct-v0.2'
base_tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(base_name, device_map=device, torch_dtype=torch.float16)

def apply_template(template, input):
    # FIXME
    system_prompt, input = '', ''
    prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{input} [/INST] "
    return prompt


def generate(input, model, tokenizer, mode, config):
    assert mode in ['feedback', 'base']
    model.eval()
    prompt = apply_template(mode, input)
    batch = tokenizer(prompt, return_tensors='pt', padding=False).to(device)
    input_ids = batch['input_ids'].squeeze(1)
    attention_mask = batch['attention_mask'].squeeze(1)

    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=512, 
        pad_token_id=tokenizer.eos_token_id)
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return out

def get_score(src, candidate):
    # use feedback model to get score
    return 1

# init settings
src = 'abc'
print(f'source is: {src}')
mt = generate(src, base_model, base_tokenizer, 'greedy')
temp = 0.9
n = 4
decay = 0.1

for i in range(n):
    input = src + mt
    # different prompt for diff model
    f = generate(input, feedback_model, feedback_tokenizer, 'greedy')
    input = src + mt + f
    c = generate(input, base_model, base_tokenizer, 'greedy')
    p = min(1, math.exp((get_score(src, c) - get_score(src, mt)) / (n * temp)))
    if random.random() < p:
        mt = c
    temp = max(0, temp - temp * decay)
print(f'final output is: {mt}')

