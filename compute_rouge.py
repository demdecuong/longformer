import argparse
import os
from rouge_score import rouge_scorer

def read_data(args):
    truth_files = []
    for r, d, f in os.walk(args.truth):
        for file in f:
            truth_files.append(os.path.join(r, file))
    label = []
    for f in truth_files:
        text = ''
        with open(f,'r') as file:
            label.append(' '.join(file.read().strip().split('\n')))

    # print(label[0])
    summary_files = []
    for r, d, f in os.walk(args.summary):
        for file in f:
            summary_files.append(os.path.join(r, file))
    pred = []
    for f in summary_files:
        text = ''
        with open(f,'r') as file:
            pred.append(' '.join(file.read().strip().split('\n')))
    print(truth_files[0],summary_files[0])
    print(label[0])
    print(pred[0])
    r1 = 0
    r2 = 0
    rl = 0
    for i in range(len(label)):
        scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
        scores = scorer.score(pred[i], label[i])
        r1 += scores['rouge1'].fmeasure
        r2 += scores['rouge2'].fmeasure
        rl += scores['rougeL'].fmeasure
    r1 = r1/len(label)
    r2 = r2/len(label)
    rl = rl/len(label)
    print('rouge-1 : {} , rouge-2 : {} , rouge-l : {}'.format(r1,r2,rl))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--truth', help='folder contains ground truth',default='')
    parser.add_argument('--summary', help='folder contains summary',default='')

    args = parser.parse_args()

    read_data(args)
    # python compute_rouge.py --truth cnndm/result/ground_truth/ --summary cnndm/result/summary/