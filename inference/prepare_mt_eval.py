from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import DataCollatorWithPadding

def prepare(data_dict, lang, tokenizer, batch_size, max_length, shuffle=False):
    # no need to shuffle for eval!
    ds = Dataset.from_dict({"src": data_dict['src'], 'mt': data_dict['mt']})
    print(ds)
    # exit()
    def preprocess_function(examples):
        model_inputs = {}
        # pivot examples added into dataloader, one pivot per instance
        lst = []
        for i in range(len(examples['src'])):
            if lang == 'zh-en':
                prompt = f"You are evaluating a Chinese-to-English Machine translation task. The source is '{examples['src'][i]}'. The model generated translation is '{examples['mt'][i]}'. Please identify all errors in the translation, up to a maximum of five. For each error, please give me the corresponding error location, error type and major/minor label for each error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."
            else:
                raise NotImplementedError
            lst.append(prompt)
        assert len(lst) == len(examples['src'])
        input = tokenizer(lst, max_length=max_length, padding='max_length', truncation=True)
        # model_inputs['input_ids'], model_inputs['attn_masks'] = input["input_ids"], input['attention_mask']
        return input

    processed_datasets = ds.map(
        preprocess_function,
        batched=True,
        num_proc=None,
        remove_columns=ds.column_names,
        desc="Running tokenizer on dataset",
    )
    print(processed_datasets)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='max_length',
        max_length=max_length,
        pad_to_multiple_of=None,
        return_tensors = 'pt'
    )
    dataloader = DataLoader(processed_datasets, batch_size=batch_size, collate_fn=data_collator, shuffle=shuffle)
    return dataloader
