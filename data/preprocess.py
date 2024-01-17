import json
import csv
import re
import pandas as pd

def extract_context(text):
    # Regular expression to find all occurrences of <v>...</v>
    pattern = r"<v>(.*?)</v>"
    return re.findall(pattern, text)

def remove_tags(text):
    # Pattern to match <v> and </v> tags
    pattern = r"</?v>"
    return re.sub(pattern, "", text)

def read_tsv_and_convert_to_json(file_path):
    # Read the TSV file
    n = None
    df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8', nrows=n)

    # Convert DataFrame to a dictionary
    data_dict = df.to_dict(orient='records')
    # reader = csv.DictReader(file, delimiter='\t')

    # Initialize an empty dictionary for the JSON data
    json_data = {"type": "text2text"}
    json_data['data'] = {}
    json_data['instances'] = []

    # Process each row in the TSV file
    c = 0
    for row in data_dict:
        # Create the unique key
        unique_key = f"{row['system']}_{row['doc']}_{row['doc_id']}_{row['seg_id']}_{row['rater']}"

        location = extract_context(row['target'])
        if len(location) > 1:
            c += 1
            continue
            # assert False
        if len(location) == 0:
            location = extract_context(row['source'])
            if len(location) == 0:
                continue

        # Check if the key is unique and add the data to the JSON dictionary
        if unique_key not in json_data['data']:
            system_prompt = f"You are a machine translation feedback model, designed to pinpoint error locations, identify error types, and assess their severity in translations."
            input = f"You are evaluating a Chinese-to-English Machine translation task. The source is '{remove_tags(row['source'])}'. The model generated translation is '{remove_tags(row['target'])}'. Please identify all errors in the translation, up to a maximum of five. For each error, please give me the corresponding error location, error type and major/minor label for each error. Major errors can confuse or mislead the reader due to significant change in meaning, while minor errors don't lead to loss of meaning but will be noticed."
            template = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{input} [/INST] "
            # input = remove_tags(row['source']) + ' ' + remove_tags(row['target'])
            json_data['data'][unique_key] = {
                'input': template,
                'output': '',
                'num': 0
            }
        json_data['data'][unique_key]['num'] += 1
        n = json_data['data'][unique_key]['num']
        s = f"\nError Location {n}: '{location[0]}', Error Type: {row['category']}, Severity: {row['severity']}"
        json_data['data'][unique_key]['output'] += s
        
    print(f'{c} datapoints with parsing error')
    # remove unique key and add count
    for k, v in json_data['data'].items():
        v['output'] = f"Your Translation contains {v['num']} errors:" + v['output']
        del v['num']
        json_data['instances'].append(v)
    del json_data['data']
    # Return the JSON data
    return json_data

# Example usage
file_path = '/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/data/mqm_newstest2021_zhen.tsv'
json_output = read_tsv_and_convert_to_json(file_path)

# Optionally, write the JSON data to a file
with open('/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/data/mqm_newstest2021_zhen_parsed.json', 'w', encoding='utf-8') as json_file:
    json.dump(json_output, json_file, ensure_ascii=False, indent=4)
