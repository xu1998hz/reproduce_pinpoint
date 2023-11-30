import pandas as pd
import argparse
import csv
import re
from preprocess import TSVDataSet
import pickle
import os

def is_english(text):
    try:
        # A simple check for English could be based on ASCII characters
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

def parse_text(text):
    # Split the text into subtexts using '\n'
    subtexts = text.split('\n')

    # Keywords to search for
    keywords = {"translation", "translate", "translates", "text", "chinese", "german", "english"}

    # Iterate through each subtext
    for i, subtext in enumerate(subtexts):
        # Count the number of keywords present in the subtext
        keyword_count = sum(word in subtext.lower().split() for word in keywords)

        # Check if at least two keywords are present
        if keyword_count >= 2:
            # If this is the last subtext, return all the previous subtexts
            if i == len(subtexts) - 1:
                return '\n'.join(subtexts[: i])
            # Return the remaining text after this subtext
            return '\n'.join(subtexts[i + 1:])

    # If no subtext meets the criteria, return the original text
    return text

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_file", type=str, default="out/zh-en_wmt_test_wmt22_out.pkl")
    argparser.add_argument("--task", type=str, help="mt, qa or summ")
    argparser.add_argument("--language", type=str, default="zh-en", help="zh-en or en-de")
    argparser.add_argument("--out_path", type=str, default="out/zh-en_wmt_test_wmt22_out_clean.pkl")
    argparser.add_argument("--debug", type=str, default="False")
    args = argparser.parse_args()
    args.debug = args.debug == "True"

    data = pickle.load(open(args.data_file, 'rb'))
    parsed = []

    if args.debug:
        f = open('out/debug_post.txt', 'w')
    for idx, i in enumerate(data):
        a, b = TSVDataSet(None, args.task, args.language).p_lens
        if args.task == 'mt':
            src = i['src'][a: -b]
            ref = i['ref']
            mt = i['mt']
            mt = mt[len(i['src']) - 14:] # 14 = len(<s></s>) * 2
            if args.debug:
                f.write(str(idx).ljust(60, '=') + '\n')
                f.write(mt + '\n')
            mt = parse_text(mt)
            parsed.append({'src': src, 'ref': ref, 'mt': mt})
        elif args.task == 'qa':
            # do nothing for now
            # TODO: double check
            q = i['q'][a:]
            ref = i['ref']
            a = i['a'][len(i['q']) - 1:]
            if args.debug:
                f.write(str(idx).center(60, '=') + '\n')
                f.write(q + '\n')
                f.write('=' * 60 + '\n')
                f.write(i['a'] + '\n')
                f.write('=' * 60 + '\n')
                f.write(a + '\n')
            parsed.append({'q': q, 'ref': ref, 'a': a})
        elif args.task == 'summ':
            # do nothing for now
            src = i['src']
            sum = i['sum'][len(i['src']) - 1:]
            if args.debug:
                f.write(str(idx).center(60, '=') + '\n')
                f.write(src + '\n')
                f.write('=' * 60 + '\n')
                f.write(i['sum'] + '\n')
                f.write('=' * 60 + '\n')
                f.write(sum + '\n')
            ref = i['ref']
            parsed.append({'src': src, 'ref': ref, 'sum': sum})
        else:
            raise NotImplementedError

    if args.debug:
        f.close()
        print(parsed)
    if not args.debug: # don't want to overwrite during debug
        with open(args.out_path, 'wb') as f:
            pickle.dump(parsed, f)
    
