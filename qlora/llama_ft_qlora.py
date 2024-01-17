from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer
import os
from transformers import LlamaForCausalLM
import torch
import json
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
import copy
from typing import Dict, Sequence
import transformers
from torch.utils.data import Dataset
from dataclasses import dataclass
import os
import wandb
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
import random
from trl import SFTTrainer


IGNORE_INDEX = -100

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    # learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    learning_rate: Optional[float] = field(default=1e-6, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    use_fp16: Optional[bool] = field(default=False, metadata={"help": "Use fp16 precision for training"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    # trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    flash_atten: Optional[bool] = field(default=False, metadata={"help": "use flash attention or not"}) 
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    # hf_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=3, metadata={"help": "Limits total number of checkpoints."})
    # gradient_checkpointing: Optional[bool] = field(
    #     default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    # )
    # gradient_checkpointing_kwargs: Optional[dict] = field(
    #     default=None,
    #     metadata={
    #         "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
    #     },
    # )
    # hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    torch_dtype = torch.float16
    if script_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, 
        llm_int8_has_fp16_weight=True,
        # bnb_8bit_compute_dtype=torch_dtype,
    )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    device_map = (
        {"": Accelerator().local_process_index}
    )
    print(device_map)
else:
    print("ERROR! Loading bnb fails")
    device_map = None
    quantization_config = None
    torch_dtype = None


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 63.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        print(f'num new token added: {num_new_tokens}')
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # NOTE: dim should be 1 instead of -1?
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=False)
# Create a new token and add it to the tokenizer
# set pad side to left TODO: double check if this is correct
tokenizer.padding_side = "left"
print(f"tokenizer padding side: {tokenizer.padding_side}")

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    # torch_dtype=torch_dtype,
    # trust_remote_code=script_args.trust_remote_code,
    # token=script_args.hf_token,
)

# tokenizer.add_special_tokens({"pad_token": "<pad>"})
# model.resize_token_embeddings(len(tokenizer))

if tokenizer.pad_token is None:
    print("add pad token and resize embedding: True")
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=
        {
            "pad_token": "<pad>",
        },
        tokenizer=tokenizer,
        model=model,
    )
else:
    print("add pad token and resize embedding: False")

# print("==Model loaded==")
# print(f"usage {model.get_memory_footprint() / 1024 ** 3} GB")

# Step 2: Load the dataset
# dataset = load_dataset(script_args.dataset_name, split="train")
# print('using wrong dataset')

server = 'psc'
if server == 'aries':
    os.environ['TRANSFORMERS_CACHE'] = '/mnt/data6/guangleizhu'
    os.environ['HF_HOME'] = '/mnt/data6/guangleizhu'
    f = "/home/guangleizhu/reproduce_pinpoint/finetune/mqm_newstest2021_zhen_parsed.json"
    output_dir = "/home/guangleizhu/reproduce_pinpoint/finetune/ft_out"
elif server == 'psc':
    os.environ['TRANSFORMERS_CACHE'] = '/ocean/projects/cis230075p/gzhu/hf_cache'
    os.environ['HF_HOME'] = '/ocean/projects/cis230075p/gzhu/hf_cache'
    f = "/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/data/mqm_newstest2021_zhen_parsed.json"
    # output_dir = "/ocean/projects/cis230075p/gzhu/ft_out"

extensions = "json"
raw_dataset = load_dataset(
    extensions,
    data_files=[f],
    field="instances",
    split="train",
)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, eval_dataset=None):
        super(SupervisedDataset, self).__init__()
        if eval_dataset is None:
            dataset = raw_dataset
        else:
            dataset = eval_dataset
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in dataset
        ]
        data_dict = preprocess(dataset["input"], targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, eval_dataset=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer)
    if eval_dataset:
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, eval_dataset=eval_dataset)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator
    )


def preprocess(sources, targets, tokenizer):
    """Preprocess sources and targets for supervised fine-tuning.
    Args:
        sources: a list of strings
        targets: a list of strings
        tokenizer: a tokenizer
    Returns:
        a dictionary of input_ids and labels
    """
    # remove pairs where at least one record is None
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

data_module = make_supervised_data_module(tokenizer=tokenizer)

print(tokenizer.special_tokens_map)
# print first few examples
# for _ in range(1):
#     # use datacollator to collate the data
#     i = random.randint(0, len(data_module["train_dataset"]))
#     example = data_module["data_collator"]([data_module["train_dataset"][i]])
#     print(example)
#     # decode
#     print(tokenizer.decode(example["input_ids"][0]))
#     print('=' * 20)
# exit(0)


# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    # hub_model_id=script_args.hub_model_id,
    # gradient_checkpointing=script_args.gradient_checkpointing,
    ddp_find_unused_parameters=False,
    # optim="adafactor",
    fp16=True, # for mixed precision training
    seed=42,
)


# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("==Using PEFT==")
    print(peft_config)
else:
    peft_config = None


# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=script_args.seq_length,
    train_dataset=data_module["train_dataset"],
    # train_dataset=dataset,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
    packing=True
)

# flash attention ver 1
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir + '/7b')
# save the tokenizer
tokenizer.save_pretrained(script_args.output_dir + '/7b')


# model.save_pretrained("output/adapter_test", save_adapter=True, save_config=True)

# base_model = AutoPeftModelForCausalLM.from_pretrained(script_args.model_name, device_map = "auto", torch_dtype = torch.float16)
# # model_to_merge = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_model).to(“cuda”), lora_adapter)

# merged_model = base_model.merge_and_unload()
# merged_model.save_pretrained("output/7b")

# # Load fine-tuned weights
# model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map = "auto", torch_dtype = torch.bfloat16)
# # Merge the LoRA layers with the base model
# model = model.merge_and_unload()

# # Save fine-tuned model at a new location
# output_merged_dir = "results/news_classification_llama2_7b/final_merged_checkpoint"
# os.makedirs(output_merged_dir, exist_ok = True)
# model.save_pretrained(output_merged_dir, safe_serialization = True)

# # Save tokenizer for easy inference
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained(output_merged_dir)

