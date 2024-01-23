from mt_metrics_eval import data as mt_data
import click
import torch
import json
import argparse
from prepare_mt_eval import prepare
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    GenerationConfig
)
from tqdm import tqdm
import re


def get_score(text):
    score = 0
    major_pattern = re.compile(r'Severity: Major')
    minor_pattern = re.compile(r'Severity: Minor')

    # Find all matches in the text
    major_matches = major_pattern.findall(text)
    minor_matches = minor_pattern.findall(text)
    score -= len(major_matches) * 5
    score -= len(minor_matches) * 1

    return major_matches, minor_matches, score


def store_corr_eval(evs, mt_scores_dict, mode, wmt, lang):
    # process the ground truth human ratings
    mqm_scores = evs.Scores(mode, 'mqm')
    qm_human = mqm_scores.copy()
    qm_no_human = set(mqm_scores) - set(evs.all_refs)

    # print(f'number of mqm scores: {len(mqm_scores)}')
    # print(mqm_scores.keys())
    # print(mt_scores_dict.keys())
    # print(len(evs.all_refs))
    # print(evs.all_refs.keys())
    # print(evs.std_ref)
        # if wmt != 'wmt21.tedtalks':
    #     print('1')
    #     qm_human.update(evs.std_ref)

    if mode == 'sys':
        # compute system-level scores (overwrite) otherwise seg scores are available already
        mt_scores_dict = {mt_name: [sum(scores)/len(scores)] for mt_name, scores in mt_scores_dict.items()}
    mqm_bp = evs.Correlation(mqm_scores, mt_scores_dict, qm_human)
    mqm_bp_no = evs.Correlation(mqm_scores, mt_scores_dict, qm_no_human)

    if mode == 'seg':
        print("Kendall seg_system_human: ", mqm_bp.Kendall()[0])
        print("Kendall seg_system: ", mqm_bp_no.Kendall()[0])
        print("Pearson seg_system_human: ", mqm_bp.Pearson()[0])
        print("Pearson seg_system: ", mqm_bp_no.Pearson()[0])
        print("Spearman seg_system_human: ", mqm_bp.Spearman()[0])
        print("Spearman seg_system: ", mqm_bp_no.Spearman()[0])
    elif mode == 'sys':
        print("sys_system_human: ", mqm_bp.Pearson()[0])
        print("sys_system: ", mqm_bp_no.Pearson()[0])
        print("Pearson sys_system_human: ", mqm_bp.Pearson()[0])
        print("Pearson sys_system: ", mqm_bp_no.Pearson()[0])
        print("Spearman sys_system_human: ", mqm_bp.Spearman()[0])
        print("Spearman sys_system: ", mqm_bp_no.Spearman()[0])
    else:
        print('Please choose between seg and sys!')
        exit(1)

def main(args):
    evs = mt_data.EvalSet(args.wmt, args.lang)
    mt_outs_dict, refs, src = evs.sys_outputs, evs.all_refs[evs.std_ref], evs.src
    print(args.load_file, args.seg)
    if not args.load_file:
        lst = list(mt_outs_dict.keys())
        if args.seg != -1:
            # assume 5 segments, set lst to the ith segment
            lst = lst[int(args.seg) * 4: (int(args.seg) + 1) * 4]
            print(len(lst))
            print(lst)
        for key in lst:
            print(f'current running on mt system: {key}')
            # key = 'refA'
            data_dict = {'src': src, 'mt': mt_outs_dict[key]}
            mt_scores_dict = {key: []}
            # load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_addr, use_fast=True)

            args.device = 'cuda:0'
            dataloader = prepare(data_dict, args.lang, tokenizer, args.batch_size, args.max_length, shuffle=False)

            model = AutoModelForCausalLM.from_pretrained(args.model_addr, device_map=args.device, torch_dtype=torch.float16)
            model.eval()
            for batch in tqdm(dataloader):
                batch = batch.to(args.device)
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                # [batch_size, max_length]
                
                outputs = model.generate(
                    input_ids, 
                    attention_mask=attention_mask, 
                    max_new_tokens=512,
                    # num_return_sequences=1
                    # pad_token_id=tokenizer.eos_token_id
                )
                for i in range(len(outputs)):
                    input = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    # print(input)
                    out = tokenizer.decode(outputs[i], skip_special_tokens=True)
                    out = out.replace(input, '')
                    # print(out)
                    # print('=' * 60)
                    mt_scores_dict[key].append(out)
                # print(i)
            with open(f'../out/{args.lang}_{args.wmt}_{key}_raw.json', 'w') as f:
                json.dump(mt_scores_dict, f)
    else:
        mt_scores_dict = {}
        zero, c = 0, 0
        print(mt_outs_dict.keys())
        for key in mt_outs_dict.keys():
            mt_scores_dict[key] = []
            with open(f'../out/{args.lang}_{args.wmt}_{key}_raw.json', 'r') as f:
                score_dict = json.load(f)
                for line in score_dict[key]:
                    # print(line)
                    score = get_score(line)
                    if score[2] == 0:
                        zero += 1
                    c += 1
                    # print(score)
                    # print('=' * 60)
                    mt_scores_dict[key].append(score[2])
        print(f'{c} datapoints, {zero} of them have zero score: {zero/c} are zero scores')

        store_corr_eval(evs, mt_scores_dict, 'seg', args.wmt, args.lang)
        store_corr_eval(evs, mt_scores_dict, 'sys', args.wmt, args.lang)

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--wmt', default='wmt22')
    argparse.add_argument('--lang', default='en-de')
    argparse.add_argument('--load_file', action='store_true')
    # TODO: tmp work around, --seg 0 - 4
    argparse.add_argument('--seg', default=-1)
    # TODO: change this later, zh-en is trained on lab server 
    argparse.add_argument('--model_addr', default='/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/finetune/ft_out/en-de/checkpoint-770')
    argparse.add_argument('--batch_size', default=1)
    argparse.add_argument('--max_length', default=720)
    args = argparse.parse_args()
    print(args)
    main(args)