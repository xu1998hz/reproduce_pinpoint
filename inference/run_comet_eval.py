"""
CUDA_VISIBLE_DEVICES=2 python3 inference/run_comet_eval.py --model Unbabel/wmt22-comet-da --data out/zh-en_wmt_test_wmt22_out_mistral.pkl --out_path out/comet_scores_zh-en_wmt_test_wmt22_mistral.json
"""


import argparse 
from comet import download_model, load_from_checkpoint
import pickle
import json

argparser = argparse.ArgumentParser()
# argparser.add_argument("--model", type=str, default="Unbabel/wmt22-comet-da")
argparser.add_argument("--lang_dir", type=str, help="zh-en or en-de")
argparser.add_argument("--wmt", type=str, help="wmt22 or wmt23")
argparser.add_argument("--model_type", type=str, help="llama2 or mistral")
args = argparser.parse_args()

data = pickle.load(open(f'out/{args.lang_dir}_wmt_test_{args.wmt}_out_{args.model_type}.pkl', 'rb'))
final_ls = []
src_ls, out_ls = [], []
for ele in data:
    if args.lang_dir == 'zh-en':
        src = ele['src'].split('Chinese:\n')[-1].split('\n')[0].strip()
    elif args.lang_dir == 'en-de':
        src = ele['src'].split('English:\n')[-1].split('\n')[0].strip()
    else:
        print("lang dir is not supported!")
        exit(1)

    mt = ele['mt'].split('[/INST]')[-1].strip()
    src_ls+=[src]
    out_ls+=[mt]
    final_ls+=[{'src': src, 'mt': mt, 'ref': ele['ref']}]

model_path = download_model('Unbabel/wmt22-comet-da')
# Load the model checkpoint:
model = load_from_checkpoint(model_path)
model_output = model.predict(final_ls, batch_size=8, gpus=1)
print(f'comet score: {model_output.system_score}')
cur_dict = {}
cur_dict['sys'] = model_output.system_score
cur_dict['seg'] = model_output.scores
cur_dict['src'] = src_ls
cur_dict['out'] = out_ls

with open(f'out/comet_scores_{args.lang_dir}_wmt_test_{args.wmt}_{args.model_type}.json', 'w') as fp:
    json.dump(cur_dict, fp)