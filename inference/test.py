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

# # python3 inference/inference_regression.py -wmt wmt21.news -lang zh-en -model_addr epoch0_best_3616_zhen_pos_neg_sep20.ckpt
# def baselines_cl_eval(mt_outs_dict, refs, emb_type, model, batch_size, tokenizer):
#     with torch.no_grad():
#         # load tokenizer and models, already specified addr for tokenizer
#         mt_scores_dict = {}
#         # generate src embeddings
#         for mt_name, mt_outs in mt_outs_dict.items():
#             mt_scores_dict[mt_name] = []
#             cur_data_dict = {'pivot': refs, 'mt': mt_outs}
#             cur_data_loader = preprocess_data(cur_data_dict, tokenizer, exp_config.max_length, batch_size, shuffle=False, sampler=False, mode='test')
#             for batch in cur_data_loader:
#                 # generate a batch of ref, mt embeddings
#                 score = model(batch, emb_type).squeeze(1).tolist()
#                 mt_scores_dict[mt_name].extend(score)
#         return mt_scores_dict

# def store_corr_eval(evs, mt_scores_dict, mode, wmt, lang):
#     # process the ground truth human ratings
#     mqm_scores = evs.Scores(mode, 'mqm')
#     qm_no_human = set(mqm_scores) - set(evs.all_refs)
#     qm_human = qm_no_human.copy()
#     if wmt != 'wmt21.tedtalks':
#         qm_human.update(human_mapping_dict[wmt][lang])

#     if mode == 'sys':
#         # compute system-level scores (overwrite) otherwise seg scores are available already
#         mt_scores_dict = {mt_name: [sum(scores)/len(scores)] for mt_name, scores in mt_scores_dict.items()}
#     mqm_bp = evs.Correlation(mqm_scores, mt_scores_dict, qm_human)
#     mqm_bp_no = evs.Correlation(mqm_scores, mt_scores_dict, qm_no_human)

#     if mode == 'seg':
#         print("seg_system_human: ", mqm_bp.Kendall()[0])
#         print("seg_system: ", mqm_bp_no.Kendall()[0])
#     elif mode == 'sys':
#         print("sys_system_human: ", mqm_bp.Pearson()[0])
#         print("sys_system: ", mqm_bp_no.Pearson()[0])
#     else:
#         print('Please choose between seg and sys!')
#         exit(1)

def main(args):
    evs = mt_data.EvalSet(args.wmt, args.lang)
    mt_outs_dict, refs, src = evs.sys_outputs, evs.all_refs[evs.std_ref], evs.src
    key = 'refA'
    # data_dict = {'src': src[:5], 'mt': mt_outs_dict[key][:5]}
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
        
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=512,
            # num_return_sequences=1
            # pad_token_id=tokenizer.eos_token_id
        )
        for i in range(len(outputs)):
            input = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            out = tokenizer.decode(outputs[i], skip_special_tokens=True)
            out = out.replace(input, '')
            print(out)
            print('=' * 60)
            mt_scores_dict[key].append(out)
        # print(i)
    with open(f'../out/{args.lang}_{args.wmt}_raw.json', 'w') as f:
        json.dump(mt_scores_dict, f)
    # store_corr_eval(evs, mt_scores_dict, 'seg', wmt, lang)
    # store_corr_eval(evs, mt_scores_dict, 'sys', wmt, lang)

if __name__ == "__main__":
    # TODO: add generation config
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--wmt', default='wmt22')
    argparse.add_argument('--lang', default='zh-en')
    argparse.add_argument('--load_file', default=None)
    # TODO: change this later, zh-en is trained on lab server 
    argparse.add_argument('--model_addr', default='/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/finetune/ft_out/zh-en/checkpoint-107')
    argparse.add_argument('--batch_size', default=1)
    argparse.add_argument('--max_length', default=720)
    args = argparse.parse_args()
    print(args)
    main(args)