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
# import tensor_parallel as tp

# TODO: add generation config

def eval(model, tokenizer, task, dataloader, out_path, device='cuda:0', debug=False):
    model.eval()
    out_data = []
    with torch.no_grad():
        for batch in tqdm(dataloader, mininterval=20):
            prompt, ref = batch
            batch = tokenizer(prompt, return_tensors='pt', padding=False).to(device)
            input_ids = batch['input_ids'].squeeze(1)
            attention_mask = batch['attention_mask'].squeeze(1)
            outputs = model.generate(input_ids, attention_mask=attention_mask)
            out = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if task == 'mt':
                out_data.append({'src': prompt[0], 'ref': ref[0], 'mt': out})
            elif task == 'qa':
                out_data.append({'q': prompt[0], 'ref': ref[0], 'a': out})
            elif task == 'summ':
                out_data.append({'src': prompt[0], 'ref': ref[0], 'sum': out})
            else:
                raise NotImplementedError

    with open(out_path, 'wb') as f:
        pickle.dump(out_data, f)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=int, default=7, help="llama model size 7, 13 or 70b")
    argparser.add_argument("--data_file", type=str, default="mt_test_data/zh-en_wmt_test_wmt22.tsv")
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

    device = 'cuda:0'

    # print('loading data from', args.data_file)
    # print('saving to', out_path)
    nrows = None
    if args.debug:
        nrows = 10
    dataset = TSVDataSet(args.data_file, args.task, args.language, nrows=nrows)
    print(dataset.data)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False)

    model = f"meta-llama/Llama-2-{args.model}b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model)
    # # for batch inferencing
    # tokenizer.pad_token = "[PAD]"
    # tokenizer.padding_side = "left"

    # time model loading time, taking way too long
    t0 = time.time()
    # model = None
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=dtype, # fp16 for training, fp32 for inference
        device_map='auto'
    )
    t1 = time.time()
    print(f"Loading model takes {(t1 - t0) / 60} minutes")

    eval(model, tokenizer, args.task, dataloader, args.out_path, device, args.debug)

  