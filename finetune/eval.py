import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    GenerationConfig
)
import random
from trl import SFTTrainer

# model = "meta-llama/Llama-2-7b-hf"
# path = "output/7b"
path = "/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/finetune/ft_out/checkpoint-535"
device = 'cuda:0'

tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
if tokenizer.pad_token is None:
    print('error! ')
# print special toekns
print(tokenizer.special_tokens_map)
# tokenizer.add_special_tokens({"pad_token": "<pad>"})

model = AutoModelForCausalLM.from_pretrained(path, device_map=device, torch_dtype=torch.float16)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map={"": 0}, torch_dtype=torch.float16)

f = '../data/mqm_newstest2021_zhen_parsed.json'

def generate():
    model.eval()
    raw_dataset = load_dataset(
        'json',
        data_files=[f],
        field='instances',
        split="train",
        use_auth_token=None,
    )
    for _ in range(5):
        index = random.randint(0, len(raw_dataset))
        input = raw_dataset[index]['input']
        batch = tokenizer(input, return_tensors='pt', padding=False).to(device)
        input_ids = batch['input_ids'].squeeze(1)
        attention_mask = batch['attention_mask'].squeeze(1)

        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=512,
            num_return_sequences=1
            # pad_token_id=tokenizer.eos_token_id
        )
        for i in range(len(outputs)):
            out = tokenizer.decode(outputs[i], skip_special_tokens=True)
            # remove the input part
            tmp = out.replace(input, '')
            print(out)
            print('=' * 60)
            # print(tmp)
            # print('=' * 60)


generate()

# prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{input} [/INST] "
# input = f"You are evaluating a Chinese-to-English Machine translation task. The source is '当地时间25日下午，一架小型飞机在德国北威州韦瑟尔县撞上一座多层住宅楼的顶层，事故共造成三人死亡。'. The model generated translation is 'On the afternoon of the 25th local time, a small plane crashed into the top floor of a multi-storey residential building in Wesel County, NRW, Germany, killing a total of three people.'. Please identify all errors in the translation, up to a maximum of five. For each error, please give me the corresponding error location, error type and major/minor label for each error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."