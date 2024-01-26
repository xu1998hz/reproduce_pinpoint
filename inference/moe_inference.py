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
    client_connection = "http://128.111.28.82:5000"  # or the public url of the server

    model = Manifest(
        client_name = "huggingface",
        client_connection = client_connection,
    )
    
    with open(args.data_path, 'r') as f:
        data = json.load(f)
    del data['seg']
    del data['sys']
    print(data.keys())

    if args.lang == 'en-de':
        src_lang, target_lang = 'English', 'German'
        icl = True
        icl_examples = [
            {'src': 'We can stand on the Earth and look up at the night sky and see stars with our bare eyes.', 
            'mt': 'Wir können auf der Erde stehen und in den Nachthimmel schauen und die Sterne mit unseren bloßen Augen sehen.'},
            {'src': "O'Brien told the woman to leave the park and stop harassing her, at which point the woman unleashed a can of mace at the couple.",
            'mt': "O'Brien sagte der Frau, sie solle den Park verlassen und aufhören, sie zu belästigen, woraufhin die Frau eine Dose Keule auf das Paar losließ."}
        ]
    elif args.lang == 'zh-en':
        src_lang, target_lang = 'Chinese', 'English'
        icl = False
        icl_examples = []
        

    mt_out = []
    print(len(data['src']))
    for i in tqdm(range(0, len(data['src']), args.batch_size), desc='Running correction'):
    # for i in tqdm(range(0, 5, args.batch_size), desc='Running correction'):
        src = data['src'][i: i + args.batch_size]
        prompts = []
        for s in src:
            messages = []
            if icl:
                for example in icl_examples:
                    messages.append({"role": "user", "content": f"Translate the following {src_lang} source into {target_lang} translation. Give only one clean {target_lang} translation without any explanation. {src_lang} source: {example['src']} {target_lang} translation:"})
                    messages.append({"role": "assistant", "content": example['mt']})

            src_prompt = f"Translate the following {src_lang} source into {target_lang} translation. Give only one clean {target_lang} translation without any explanation. {src_lang} source: {s} {target_lang} translation:"
            messages.append({"role": "user", "content": src_prompt})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            # print(prompt)
            # print('-' * 60)
            prompts.append(prompt)
        # print(src)
        out = model.run(prompts, max_new_tokens=512, do_sample=False)
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
    argparse.add_argument('--lang', default='en-de')
    argparse.add_argument('--data_path', default='/home/guangleizhu/reproduce_pinpoint/out/mt_out/comet_scores_en-de_wmt_test_wmt22_llama2.json')
    argparse.add_argument('--out_path', default='test.json')
    # FIXME: batch not working rn
    argparse.add_argument('--batch_size', default=2)
    argparse.add_argument('--max_length', default=720)
    args = argparse.parse_args()
    print(args)

    main(args)
