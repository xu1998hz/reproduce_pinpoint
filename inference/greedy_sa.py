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
    l = text.split('\n')
    out = ''
    for line in l[1:]:
        try:
            match = re.search(r'Error Location \d: \'(.*)\', Error Type: (.*), Severity: (Major|Minor)', line)
            location = match.group(1)
            error_type = match.group(2).lower()
            severity = match.group(3).lower()
            s = f"'{location}' is a {severity} {error_type} error."
            out += s
        except:
            continue
    return out

# add language
def feedback_generate(source, candidate, lang, model, tokenizer, config, device, save=False):
    # TODO: add other languages
    if lang == 'en-de':
        prompt = f"You are evaluating a English-to-German Machine translation task. The source is '{source}'. The model generated translation is '{candidate}'. Please identify all errors in the translation, up to a maximum of five. For each error, please give me the corresponding error location, error type and major/minor label for each error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."
    elif lang == 'zh-en':
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
    out = out.replace(prompt, '')
    score = get_score(out)
    return score[2], out

def base_generate(src, lang, model, tokenizer, config, device, candidate=None, feedback=None, save=False, icl=True):
    messages = []
    if lang == 'en-de':
        src_lang, target_lang = 'English', 'German'
    elif lang == 'zh-en':
        src_lang, target_lang = 'Chinese', 'English'
    if icl:
        if lang == 'en-de':
            instances = [{
                'src': "O'Brien told the woman to leave the park and stop harassing her, at which point the woman unleashed a can of mace at the couple.",
                'mt': "O'Brien sagte der Frau, sie solle den Park verlassen und aufhören, sie zu belästigen, woraufhin die Frau eine Dose Keule auf das Paar losließ.",
                'feedback': "'eine Dose Keule' is a major accuracy/mistranslation error.",
                'revised': "O'Brien forderte die Frau auf, den Park zu verlassen und sie nicht weiter zu belästigen, worauf die Frau das Paar mit Pfefferspray attackierte."
            }]
        # O'Brien forderte die Frau auf, den Park zu verlassen und sie in Ruhe zu lassen, woraufhin die Frau das Paar mit Pfefferspray besprühte.
        elif lang == 'zh-en':
            instances = [{
                'src': '如果逾期，逾期记录将记录到个人信用报告中，可能会对日后买车、购房等经济生活造成不良影响。',
                'mt': 'If overdue, overdue records will be recorded in your personal credit report, which may negatively impact future car purchases, real estate transactions, and other economic living situations.',
                'feedback': "'overdue' is a major accuracy/mistranslation.",
                'revised': 'If it is overdue, overdue records will be recorded in your personal credit report, which may negatively impact future car purchases, real estate transactions, and other economic living situations.'
            }]
    else:
        instances = []
    instances.append({'src': src, 'mt': candidate, 'feedback': feedback, 'revised': None})

    for i in instances:
        src_prompt = f"Translate the following {src_lang} source into {target_lang} translation. Give only one clean {target_lang} translation without any explanation. {src_lang} source: {i['src']} {target_lang} translation:"
        messages.append({"role": "user", "content": src_prompt})
        if i['feedback']:
            feedback_prompt = f"{i['feedback']} Please revise the {target_lang} translation according to my feedback. Provide a clean {target_lang} translation without any explanation. {target_lang} translation:"
            messages.extend([
                {"role": "assistant", "content": i['mt']},
                {"role": "user", "content": feedback_prompt},
            ])
        # only True for icl examples
        if i['revised']:
            messages.append({"role": "assistant", "content": i['revised']})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    batch = tokenizer(prompt, return_tensors='pt', padding=False).to(device)
    print(prompt)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    model.eval()
    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=512, 
        pad_token_id=tokenizer.eos_token_id,
        temperature=0
    )
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    out = out.replace(input, '').strip()

    if feedback:
        print(src, '|', candidate, '|', feedback)
        print('after correction'.center(60, '-'))
        print(out)
    return out


# init settings
def feedback(args):
    assert args.lang in ['en-de', 'zh-en']
    weight_path = args.model_addr
    device = 'cuda:0'

    # load the feedback model
    feedback_tokenizer = AutoTokenizer.from_pretrained(weight_path, use_fast=True)
    feedback_model = AutoModelForCausalLM.from_pretrained(weight_path, device_map=device, torch_dtype=torch.float32)

    with open(args.data_path, 'r') as f:
        data = json.load(f)
    f_lst = []
    c = 0
    for i in tqdm(range(len(data['src'])), desc='Getting feedback'):
    # for i in tqdm(range(10), desc='Getting feedback'):
        # i = random.randint(0, len(data['src']) - 1)
        example = data['src'][i]
        mt = data['out'][i]
        # generate initial candidate
        score, f = feedback_generate(example, mt, args.lang, feedback_model, feedback_tokenizer, None, device)
        f = parse_feedback(f)
        if score == 0:
            c += 1
            f_lst.append(None)
        else:
            f_lst.append(f)
    print(f'number of no feedback: {c} ({c / len(data["src"])})')
    return f_lst   


"""Load feedback from a local file and use those feedback for correction step"""
def load_feedback(feedback_addr):
    feedback_ls = ''.join(open(feedback_addr, 'r').readlines()).split('[SPECIAL_TOK_WENDA]')
    feedback_ls = [parse_feedback(f) for f in feedback_ls]
    return feedback_ls

def correction(feedback, args):
    assert args.lang in ['en-de', 'zh-en']
    device = 'cuda:0'

    # load mistral 7b instruct model
    if args.model == 'mistral':
        base_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    else:
        base_name = 'NousResearch/Llama-2-7b-chat-hf'
    print(f"loading {base_name}")

    base_tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float32).to(device)

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    final_mt = []
    for i in tqdm(range(len(data['src'])), desc='Running correction'):
    # for i in tqdm(range(10), desc='Running correction'):
        example = data['src'][i]
        mt = data['out'][i]
        f = feedback[i]
        if f is None:
            final_mt.append(mt)
        else:
            print('=' * 60)
            new_candidate = base_generate(example, args.lang, base_model, base_tokenizer, None, device, mt, f)
            final_mt.append(new_candidate)
    data['out'] = final_mt
    with open(args.out_path, 'w') as f:
        json.dump(data, f)
    

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--wmt', default='wmt22')
    argparse.add_argument('--lang', default='zh-en')
    argparse.add_argument('--model', default='mistral')
    argparse.add_argument('--model_addr', default='/mnt/taurus/home/guangleizhu/reproduce_pinpoint/finetune/ft_out/zh-en/checkpoint-760/')
    argparse.add_argument('--data_path', default='out/mt_out/comet_scores_zh-en_wmt_test_wmt22_mistral.json')
    argparse.add_argument('--out_path', default='out/mt_out/correction_zh-en_wmt_test_wmt22_mistral.json')
    # FIXME: batch not working rn
    argparse.add_argument('--batch_size', default=1)
    argparse.add_argument('--max_length', default=720)
    argparse.add_argument('--feedback_addr', default='out/comet_scores_zh-en_wmt_test_wmt22_mistral.txt')
    argparse.add_argument('--feedback_type', help='improve, score, binary, mqm')
    args = argparse.parse_args()
    print(args)

    if args.feedback_addr:
        f_ls = load_feedback(args.feedback_addr)
    else:
        f_ls = feedback(args)

    correction(f_ls, args)
