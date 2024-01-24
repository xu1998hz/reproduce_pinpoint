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


# !! How to run !!
# "python sa.py > out.txt"
# remember to change the path to the model weight

# current observation
# 1. the mistral model is able to make some corrections (sometimes from major -> minor errors)
# 2. feedback model is very unstable, gives different errors for the same input
# 3. output format seems fine by few examples, can add more icl examples

# TODO:
# 1. add config for the model
# 2. load data in batch

def parse_feedback(text):
    return text
    # l = text.split('\n')
    # for line in l:


def feedback_generate(source, candidate, model, tokenizer, config, device, save=False):
    # TODO: add other languages
    prompt = f"You are evaluating a Chinese-to-English Machine translation task. The source is '{source}'. The model generated translation is '{candidate}'. Please identify all errors in the translation, up to a maximum of five. For each error, please give me the corresponding error location, error type and major/minor label for each error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."

    batch = tokenizer(prompt, return_tensors='pt', padding=False).to(device)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    model.eval()

    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=512, 
        # pad_token_id=tokenizer.eos_token_id
    )
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if save:
        print(out)
        print('=' * 60)
    out = out.replace(prompt, '')
    score = get_score(out)
    return score[2], out

def base_generate(src, model, tokenizer, config, device, candidate=None, feedback=None, save=False, icl=True):
    messages = []
    if icl and feedback:
        instances = [{'src': '如果逾期，逾期记录将记录到个人信用报告中，可能会对日后买车、购房等经济生活造成不良影响。',
            'mt': 'If overdue, overdue records will be recorded in your personal credit report, which may negatively impact future car purchases, real estate transactions, and other economic living situations.',
            'feedback': "Your translation contains 1 errors.\nError Location 1: 'overdue', Error Type: Accuracy/Mistranslation, Severity: Major",
            'revised': 'If overdue, overdue records will be recorded in your personal credit report, which may negatively impact future car purchases, real estate transactions, and other economic living situations.'
        }]
    else:
        instances = []
    instances.append({'src': src, 'mt': candidate, 'feedback': feedback, 'revised': None})

    for i in instances:
        src_prompt = f"Translate the following Chinese into English. Give only one clean English translation without any explanation. Chinese:\n{i['src']}\nEnglish:"
        messages.append({"role": "user", "content": src_prompt})
        if i['feedback']:
            feedback_prompt = f"{feedback} Please revise the translation according to feedback. Provide a clean English translation without any explanation. Chinese:\n{src}\nEnglish:"
            messages.extend([
                {"role": "assistant", "content": candidate},
                {"role": "user", "content": feedback_prompt},
            ])
        # only True for icl examples
        if i['revised']:
            messages.append({"role": "assistant", "content": candidate})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    batch = tokenizer(prompt, return_tensors='pt', padding=False).to(device)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    model.eval()

    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=512, 
        pad_token_id=tokenizer.eos_token_id
    )
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    out = out.replace(input, '')
    out = out[1:]
    if feedback:
        print(src, '|', candidate, '|', feedback)
        print('after correction'.center(60, '-'))
        print(out)
    return out


# init settings
def main(args):
    weight_path = args.model_addr
    device = 'cuda:0'

    # load the feedback model
    feedback_tokenizer = AutoTokenizer.from_pretrained(weight_path, use_fast=True)
    feedback_model = AutoModelForCausalLM.from_pretrained(weight_path, device_map=device, torch_dtype=torch.float16)

    # load mistral 7b instruct model
    if args.model == 'mistral':
        base_name = 'mistralai/Mistral-7B-Instruct-v0.2'
        base_tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(base_name, device_map=device, torch_dtype=torch.float16)
    else:
        raise NotImplementedError

    with open(args.data_path, 'r') as f:
        data = json.load(f)
    final_mt = []
    # for i in tqdm(range(len(data['src']))):
    for _ in tqdm(range(5)):
        i = random.randint(0, len(data['src']) - 1)
        print('=' * 60)
        example = data['src'][i]
        mt = data['out'][i]
        # generate initial candidate
        score, f = feedback_generate(example, mt, feedback_model, feedback_tokenizer, None, device)
        f = parse_feedback(f)
        if score == 0:
            print('no error detected, skip')
            final_mt.append(mt)
            continue
        new_candidate = base_generate(example, base_model, base_tokenizer, None, device, mt, f)
        final_mt.append(new_candidate)
    print(final_mt)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--wmt', default='wmt22')
    argparse.add_argument('--lang', default='en-de')
    argparse.add_argument('--model', default='mistral')
    argparse.add_argument('--model_addr', default='/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/finetune/ft_out/en-de/checkpoint-770')
    argparse.add_argument('--data_path', default='/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/comet_scores_en-de_wmt_test_wmt22_mistral.json')
    argparse.add_argument('--out_path', default='/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/test.json')
    # FIXME: batch not working rn
    argparse.add_argument('--batch_size', default=1)
    argparse.add_argument('--max_length', default=720)
    args = argparse.parse_args()
    print(args)
    main(args)