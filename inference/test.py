import transformers
from mt_metrics_eval import data as mt_data
from mt_metric import get_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import math
from prepare_mt_eval import prepare
import argparse
from tqdm import tqdm
import json
import re
from manifest import Manifest

argparse = argparse.ArgumentParser()
argparse.add_argument('--wmt', default='wmt22')
argparse.add_argument('--lang', default='zh-en')
argparse.add_argument('--batch_size', default=1)
args = argparse.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

weight_path = '/home/guangleizhu/reproduce_pinpoint/finetune/ft_out/mistral_ft_test/checkpoint-66'

# load the fine tuned model
tokenizer = AutoTokenizer.from_pretrained(weight_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(weight_path, device_map=device, torch_dtype=torch.float32)


src_lang, target_lang = 'Chinese', 'English'
evs = mt_data.EvalSet(args.wmt, args.lang)
src = evs.src
for _ in range(50):
    i = random.randint(0, len(src))
    # print(src[i])
    prompt = f"Translate the following {src_lang} source into {target_lang} translation. Give only one clean {target_lang} translation without any explanation. {src_lang} source: {src[i]} {target_lang} translation:"
    # print(prompt)
    batch = tokenizer(prompt, return_tensors='pt', padding=False).to(device)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    model.eval()
    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=512, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        # temperature=0
    )
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(out)
    input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    out = out.replace(input, '').strip()
    # print(out)