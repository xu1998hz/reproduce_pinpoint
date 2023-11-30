from rouge import Rouge 
import argparse
import pickle   

argparser = argparse.ArgumentParser()
argparser.add_argument("--data", type=str, default="out/qa_out_clean.pkl")
argparser.add_argument("--out_path", type=str, default="out/rogue_qa.txt")
argparser.add_argument("--task", type=str, default="qa or summ")
argparser.add_argument("--debug", type=bool, default=False)
args = argparser.parse_args()

rouge = Rouge()
data = pickle.load(open(args.data, 'rb'))
scores = []

for i in range(len(data)):
    if args.task == 'qa':
        hypo = data[i]['a']
        ref = data[i]['ref']
        r1, r2 = ref
        s1 = rouge.get_scores(hypo, r1[0])[0]['rouge-l']['f']
        s2 = rouge.get_scores(hypo, r2[0])[0]['rouge-l']['f']
        scores.append(max(s1, s2))
    elif args.task == 'summ':
        hypo = data[i]['sum']
        ref = data[i]['ref']
        s = rouge.get_scores(hypo, ref)[0]['rouge-l']['f']
        scores.append(s)
    else:
        raise NotImplementedError

with open(args.out_path, 'w') as f:
    mean = sum(scores) / len(scores)
    f.write(str(mean) + '\n\n')
    for i in range(len(scores)):
        f.write(f'{i}'.center(60, "=") + '\n')
        # f.write(str(i).ljust(60, '=') + '\n')
        f.write(str(scores[i]) + '\n')
        if args.task == 'qa':
            f.write('a: ' + data[i]['a'])
            f.write('\n')   
            f.write('r1: ' + data[i]['ref'][0][0])
            f.write('\n')   
            f.write('r2: ' + data[i]['ref'][1][0])
            f.write('\n')   
        else:
            f.write('sum: ' + data[i]['sum'])
            f.write('\n')   
            f.write('ref: ' + data[i]['ref'])
            f.write('\n')
