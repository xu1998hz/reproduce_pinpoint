import json

# def is_english(text):
#     try:
#         # A simple check for English could be based on ASCII characters
#         text.encode('ascii')
#         return 1
#     except UnicodeEncodeError:
#         return 0
    
# def mistral_parse(text, language):
#     if text.find('(Literally:') != -1:
#         text = text[:text.find('(Literally:')]
#     # Split the text into subtexts using '\n' and find the part with the most (none)English characters
#     subtexts = text.split('\n')

#     best_match = subtexts[0]
#     if language == 'zh-en':
#         for s in subtexts:
#             if sum([is_english(i) for i in s]) > sum([is_english(i) for i in best_match]):
#                 best_match = s

#     elif language == 'en-de':
#         for s in subtexts:
#             if len(s) - sum([is_english(i) for i in s]) > len(best_match) - sum([is_english(i) for i in best_match]):
#                 best_match = s
#     return best_match

def parse_text(text):
    # Split the text into subtexts using '\n'
    text = text.split('\n\n')[0].split(' (')[0].strip()
    subtexts = text.split('\n')

    # Keywords to search for
    keywords = {"translation", "translate", "translates", "chinese", "german", "english", "note"}
    l = []

    # Iterate through each subtext
    for i, subtext in enumerate(subtexts):
        # Count the number of keywords present in the subtext
        keyword_count = sum(word in subtext.lower().split() for word in keywords)

        # Check if at least two keywords are present
        if keyword_count < 1:
            l.append(subtext)
      
    if len(l) == 0:
        return text
    else:
        return '\n'.join(l)

lang="en-de"
wmt="wmt22"
f = f'out/mt_out/moe_{lang}_wmt_test_{wmt}.json'
out_f = f.replace('.json', '_cleared.json')

with open(f, 'r') as f:
    data = json.load(f)

new_data = {'src': data['src'], 'out': []}
for i in range(len(data['src'])):
    src = data['src'][i]
    print(src)
    out = data['out'][i]
    out = parse_text(out)
    out = out.strip()
    # out = mistral_parse(out, 'zh-en')
    new_data['out'].append(out)
    print(out)
    print('-' * 60)

with open(out_f, 'w') as f:
    json.dump(new_data, f)

