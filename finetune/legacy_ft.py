# pip install git+https://github.com/huggingface/transformers transformers-4.28.0.dev0

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
import click
import os
import wandb
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
import random

# set hf cache dir
server = 'psc'
if server == 'aries':
    os.environ['TRANSFORMERS_CACHE'] = '/mnt/data6/guangleizhu'
    os.environ['HF_HOME'] = '/mnt/data6/guangleizhu'
    f = "/home/guangleizhu/reproduce_pinpoint/finetune/mqm_newstest2021_zhen_parsed.json"
    output_dir = "/home/guangleizhu/reproduce_pinpoint/finetune/ft_out"
elif server == 'psc':
    os.environ['TRANSFORMERS_CACHE'] = '/ocean/projects/cis230075p/gzhu/hf_cache'
    os.environ['HF_HOME'] = '/ocean/projects/cis230075p/gzhu/hf_cache'
    f = "/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/finetune/mqm_newstest2021_zhen_parsed.json"
    output_dir = "/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/finetune/ft_out"

random.seed(42) 
os.environ["WANDB__SERVICE_WAIT"] = "300"

"""deepspeed --num_gpus 8 code/finetune_llama.py"""

KEY_TYPE = "type"
KEY_INSTANCES = "instances"
ds_config = "config/ds_config_zero3.json"
do_train = True
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
# max_length = 720
max_length = 512
# f = "data/eval_mt_russian_llama.json"
# output_dir = "/share/edc/home/wendaxu/finetune_llama_ref_russian_may_28"
padding_strategy = "right"
model_name = "meta-llama/Llama-2-7b-chat-hf"
num_epoch = 1

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
    """Tokenize a list of strings with padding."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
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


# sanity check over the fields of json file
with open(f) as fin:
    json_data = json.load(fin)
    if KEY_TYPE not in json_data.keys():
        raise ValueError(
            f'"{KEY_TYPE}" field must be specified for data, e.g.'
            "{\n"
            f'   "{KEY_TYPE}: "text2text",\n'
            f'   "{KEY_INSTANCES}": [\n'
            '       { "text": "Sentence 1: This is a sentence." }\n'
            '       { "text": "Sentence 2: This is another sentence." }\n'
            f"   ]\n"
            "}"
        )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# Load the dataset using the HuggingFace dataset library
extensions = "json"
raw_dataset = load_dataset(
    extensions,
    data_files=[f],
    field=KEY_INSTANCES,
    split="train",
)


print("Start loading in tokenizers")
# config = AutoConfig.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
    # model_max_length=max_length,
    # padding_side=padding_strategy,
    use_fast=False,
)
# tokenizer.add_special_tokens({"pad_token": "<pad>"})


print(f"Start loading in model {model_name}")

# model = LlamaForCausalLM.from_pretrained(model_name)
# dtype = torch.float16
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=dtype, # fp16 for training, fp32 for inference
        # device_map='auto'
    )
# model.resize_token_embeddings(len(tokenizer))


# for check ram usage
# estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1)

# # TODO: double check this
if tokenizer.pad_token is None:
    print("add pad token and resize embedding: True")
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=
        {
            # "eos_token": DEFAULT_EOS_TOKEN,
            # "bos_token": DEFAULT_BOS_TOKEN,
            # "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        },
        tokenizer=tokenizer,
        model=model,
    )
else:
    print("add pad token and resize embedding: False")
    
print(tokenizer.special_tokens_map)

print("Loaded in model and tokenizers")

print("Start making data module")


data_module = make_supervised_data_module(tokenizer=tokenizer)
for _ in range(2):
    # use datacollator to collate the data
    i = random.randint(0, len(data_module["train_dataset"]))
    example = data_module["data_collator"]([data_module["train_dataset"][i]])
    print(example)
    # decode
    print(tokenizer.decode(example["input_ids"][0]))

    # print the input_ids
    # print(data_module["train_dataset"][1])
    # decode
    # print(tokenizer.decode(data_module["train_dataset"][i]["input_ids"]))
    print('=' * 20)

# exit(0)

# class CustomTrainer(Trainer):
#     # redefine the compute_loss function with ce loss
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         loss = F.cross_entropy(logits.view(-1, self.model.config.vocab_size), labels.view(-1), ignore_index=IGNORE_INDEX)

#         return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="no",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=1e-5,
    weight_decay=0,
    num_train_epochs=num_epoch,
    warmup_ratio=0,
    logging_strategy="steps",
    logging_first_step=True,
    save_strategy="epoch",
    save_total_limit=3,
    seed=42,
    run_name="wandb",
    load_best_model_at_end=False,
    greater_is_better=False,
    deepspeed=ds_config,
    log_on_each_node=False,
    logging_steps=1,
    fp16=True,
    lr_scheduler_type="cosine",
)  # tf32=True -> only for A100

print("Start the trainer")

if do_train:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_module["data_collator"],
        # compute_metrics=data_module.compute_metrics,
        preprocess_logits_for_metrics=None
    )

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    # trainer.save_state()