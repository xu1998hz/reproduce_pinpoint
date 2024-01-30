import json
import csv
import re
import pandas as pd
from typing import Dict
from copy import deepcopy


def extract_context(text):
    # Regular expression to find all occurrences of <v>...</v>
    pattern = r"<v>(.*?)</v>"
    return re.findall(pattern, text)

def remove_tags(text):
    # Pattern to match <v> and </v> tags
    pattern = r"</?v>"
    return re.sub(pattern, "", text)

def read_tsv_and_convert_to_json(file_path, language):
    # Read the TSV file
    n = 10
    # df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8', on_bad_lines='skip', nrows=5)
    # df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8', nrows=n)

    # Convert DataFrame to a dictionary
    # data_dict = df.to_dict(orient='records')
    data_dict = []
    with open(file_path) as f:
        f.readline()
        for line in f:
            l = line.split('\t')
            d = dict(
                system=l[0],
                doc=l[1],
                doc_id=l[2],
                seg_id=l[3],
                rater=l[4],
                source=l[5],
                target=l[6],
                category=l[7],
                severity=l[8],
            )
            if language == 'zhen':
                d['severity'] = d['severity'][:-1]
            data_dict.append(d)

    # print(data_dict[:3])

    print(len(data_dict))

    # Initialize an empty dictionary for the JSON data
    json_data = {"type": "text2text"}
    json_data['data'] = {}
    json_data['instances'] = []

    # Process each row in the TSV file
    c = 0
    for row in data_dict:
        # Create the unique key
        unique_key = f"{row['system']}_{row['doc']}_{row['doc_id']}_{row['seg_id']}_{row['rater']}"

        # Check if the key is unique and add the data to the JSON dictionary
        if unique_key not in json_data['data']:
            if language == 'zhen':
                input = f"You are evaluating a Chinese-to-English Machine translation task. The source is '{remove_tags(row['source'])}'. The model generated translation is '{remove_tags(row['target'])}'. Please identify all errors in the translation, up to a maximum of five. For each error, please give me the corresponding error location, error type and major/minor label for each error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."
            if language == 'ende':
                input = f"You are evaluating a English-to-German Machine translation task. The source is '{remove_tags(row['source'])}'. The model generated translation is '{remove_tags(row['target'])}'. Please identify all errors in the translation, up to a maximum of five. For each error, please give me the corresponding error location, error type and major/minor label for each error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."
            # system_prompt = f"You are a machine translation feedback model, designed to pinpoint error locations, identify error types, and assess their severity in translations."
            # template = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{input} [/INST] "
            json_data['data'][unique_key] = {
                'input': input,
                'output': '',
                'num': 0
            }

        if row['category'] == 'No-error':
            continue
        else:
            location = extract_context(row['target'])
            if len(location) > 1:
                # should not happen
                print(row)
            # if not in target, try to find in source (omit or source error)
            if len(location) == 0:
                location = extract_context(row['source'])
            if len(location) != 0 or row['category'] == 'Non-translation!':
                json_data['data'][unique_key]['num'] += 1
                n = json_data['data'][unique_key]['num']
                if row['category'] == 'Non-translation!':
                    s = f"\nError Location {n}: '{remove_tags(row['target'])}', Error Type: {row['category']}, Severity: {row['severity']}"
                else:
                    s = f"\nError Location {n}: '{location[0]}', Error Type: {row['category']}, Severity: {row['severity']}"
                json_data['data'][unique_key]['output'] += s
            else:
                # a couple bad datapoints here
                c += 1
        
    print(f'{c} bad datapoints with parsing error')
    # remove unique key and add count
    for k, v in json_data['data'].items():
        v['output'] = f"Your translation contains {v['num']} errors." + v['output']
        del v['num']
        json_data['instances'].append(v)
    del json_data['data']
    # Return the JSON data
    return json_data

# Example usage
# file_path = '/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/data/mqm_newstest2021_zhen.tsv'
# mode = 'zhen'
mode = 'ende'
file_path = f'/home/guangleizhu/reproduce_pinpoint/data/mqm_newstest2021_{mode}.tsv'
json_output = read_tsv_and_convert_to_json(file_path, mode)

# Optionally, write the JSON data to a file
# 'zhen' -> 'zh-en'
mode = mode[:2] + '-' + mode[2:]
with open(f'/home/guangleizhu/reproduce_pinpoint/data/mqm_newstest2021_{mode}_parsed.json', 'w', encoding='utf-8') as json_file:
    json.dump(json_output, json_file, ensure_ascii=False, indent=4)
