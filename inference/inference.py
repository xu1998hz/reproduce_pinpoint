from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from preprocess import TSVDataSet
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import time
import csv
import pickle
import argparse
import os
from pathlib import Path
from typing import Dict, TypeVar, Iterable, List
# import tensor_parallel as tp # do not seem to work

# TODO: add generation config

def eval(model, tokenizer, task, dataloader, out_path, device='cuda:0', debug=False):
    model.eval()
    out_data = []
    with torch.no_grad():
        for batch in tqdm(dataloader, mininterval=20):
            prompt, ref = batch
            batch = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)
            input_ids = batch['input_ids'].squeeze(1)
            print(input_ids.size())
            attention_mask = batch['attention_mask'].squeeze(1)
            # change pad token to eos to supress warning
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)
            out = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            if task == 'mt':
                out_data += [{'src': prompt_ele, 'ref': ref_ele, 'mt': out_ele} for prompt_ele, ref_ele, out_ele in zip(prompt, ref, out)]
            elif task == 'qa':
                out_data += [{'q': prompt_ele, 'ref': ref_ele, 'a': out_ele} for prompt_ele, ref_ele, out_ele in zip(prompt, ref, out)] 
            elif task == 'summ':
                out_data += [{'src': prompt_ele, 'ref': ref_ele, 'sum': out_ele} for prompt_ele, ref_ele, out_ele in zip(prompt, ref, out)]
            else:
                raise NotImplementedError

    with open(out_path, 'wb') as f:
        pickle.dump(out_data, f)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_base", type=str, help="llama or mistral")
    argparser.add_argument("--model_size", type=int, default=7, help="7, 13 or 70b for llama and 7b for mistral")
    argparser.add_argument("--data_file", type=str, default="data/mt_test_data/zh-en_wmt_test_wmt22.tsv")
    argparser.add_argument("--out_path", type=str, default="out/zh-en_wmt_test_wmt22_out.pkl")
    argparser.add_argument("--precision", type=str, default="fp16", help="fp16 or fp32")
    argparser.add_argument("--language", type=str, default="zh-en", help="zh-en or en-de")
    argparser.add_argument("--batch_size", type=int, default="1")
    argparser.add_argument("--task", type=str, help="mt, qa or summ")
    argparser.add_argument("--debug", type=str, default=False)
    args = argparser.parse_args()

    args.debug = args.debug == 'True'
    if args.debug:
        print('!!debug mode!!')

    # v100 does not support bf16
    if args.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # print('loading data from', args.data_file)
    # print('saving to', out_path)
    nrows = None
    if args.debug:
        nrows = 50
    # nrows = 50 # for debug
    dataset = TSVDataSet(args.data_file, args.task, args.language, args.model_base, nrows)
    print(dataset.data)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False)

    if args.model_base == 'llama':
        model = f"meta-llama/Llama-2-{args.model_size}b-chat-hf"
    elif args.model_base == 'mistral':
        model = f"mistralai/Mistral-7B-Instruct-v0.2"
    else:
        raise NotImplementedError
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # time model loading time, taking way too long
    t0 = time.time()
    # model = None
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=dtype, # fp16 for training, fp32 for inference
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    
    t1 = time.time()
    print(f"Loading model takes {(t1 - t0) / 60} minutes")

    eval(model, tokenizer, args.task, dataloader, args.out_path, device, args.debug)

  