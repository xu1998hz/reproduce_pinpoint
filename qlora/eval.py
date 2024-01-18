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

from trl import SFTTrainer

# model = "meta-llama/Llama-2-7b-hf"
# path = "output/7b"
path = "/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/qlora/output/Llama-2-7b-hf_lr_4e-05_eps_5_bs_1_ga32_seq_512_rank_64_alpha_16_dropout_0.05_fp16_True_4bit_True_fa_True"
device = 'cuda:0'

tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
if tokenizer.pad_token is None:
    print('error! ')
# tokenizer.add_special_tokens({"pad_token": "<pad>"})

model = AutoModelForCausalLM.from_pretrained(path, device_map=device, torch_dtype=torch.float16)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map={"": 0}, torch_dtype=torch.float16)

def generate(instruction):
    model.eval()
    system_prompt = f"You are a machine translation feedback model, designed to pinpoint error locations, identify error types, and assess their severity in translations."
    input = f"You are evaluating a Chinese-to-English Machine translation task. The source is '当地时间25日下午，一架小型飞机在德国北威州韦瑟尔县撞上一座多层住宅楼的顶层，事故共造成三人死亡。'. The model generated translation is 'On the afternoon of the 25th local time, a small plane crashed into the top floor of a multi-storey residential building in Wesel County, NRW, Germany, killing a total of three people.'. Please identify all errors in the translation, up to a maximum of five. For each error, please give me the corresponding error location, error type and major/minor label for each error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."
    prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{input} [/INST] "
    batch = tokenizer(prompt, return_tensors='pt', padding=False).to(device)
    input_ids = batch['input_ids'].squeeze(1)
    attention_mask = batch['attention_mask'].squeeze(1)

    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=128,
        num_return_sequences=2
        # pad_token_id=tokenizer.eos_token_id
    )
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(out)


generate(1)