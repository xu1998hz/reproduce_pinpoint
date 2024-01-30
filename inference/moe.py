# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# 4 gpus 

# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto')

# text = "Hello my name is"
# inputs = tokenizer(text, return_tensors="pt").to('cuda')

# outputs = model.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

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



def main(args):
    assert args.lang in ['en-de', 'zh-en']

    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    
    
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    del data['seg']
    del data['sys']
    print(data.keys())

    if args.lang == 'en-de':
        src_lang, target_lang = 'English', 'German'
    elif args.lang == 'zh-en':
        src_lang, target_lang = 'Chinese', 'English'

    mt_out = []
    print(len(data['src']))
    # for i in tqdm(range(0, len(data['src']), args.batch_size), desc='Running correction'):
    for i in tqdm(range(0, 10, args.batch_size), desc='Running correction'):
        src = data['src'][i: i + args.batch_size]
        prompts = []
        for s in src:
            print(s)
            messages = []
            src_prompt = f"Translate the following {src_lang} source into {target_lang} translation. Give only one clean {target_lang} translation without any explanation. {src_lang} source: {s} {target_lang} translation:"
            messages.append({"role": "user", "content": src_prompt})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            prompts.append(prompt)
        prompts = tokenizer(prompts, truncation=True, padding=True, max_length=512, return_tensors="pt").to('cuda')
        # print(src)
        # out = model.run(prompts, max_new_tokens=512, do_sample=False)
        out = model.generate(**prompts, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        for i in range(len(out)):
            tmp = tokenizer.decode(out[i], skip_special_tokens=True)
            print(tmp)
        # print(out)
        # print('-' * 60)
        # mt_out.append(out)
        mt_out.extend(out)
    data['out'] = mt_out
    with open(args.out_path, 'w') as f:
        json.dump(data, f)
    

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--wmt', default='wmt22')
    argparse.add_argument('--lang', default='zh-en')
    argparse.add_argument('--data_path', default='/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/mt_out/comet_scores_zh-en_wmt_test_wmt23_llama2.json')
    argparse.add_argument('--out_path', default='mixtral_zh-en-wmt23.json')
    # FIXME: batch not working rn
    argparse.add_argument('--batch_size', default=8)
    argparse.add_argument('--max_length', default=720)
    args = argparse.parse_args()
    print(args)

    main(args)
