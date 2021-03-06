import numpy as np
import os

dirname = os.path.dirname(os.path.dirname(__file__))

file1 = 'output/avg_cos_similarity.txt' #baseline
file2 = 'output/bertscore_F1_roberta_noidf.txt' #bert

with open(os.path.join(dirname, file1), encoding="utf8") as f:
    baseline = list(map(float, f.readlines()))
with open(os.path.join(dirname, file2), encoding="utf8") as f:
    bert_similarities = list(map(float, f.readlines()))

if len(baseline) != len(bert_similarities):
    raise ValueError('Lengths of files do not match')

diff_base = 0
diff_bert = 0

a = np.sum(np.array(baseline))
b = np.sum(np.array(bert_similarities))
print('Bert score: ', b)
print('Baseline score: ', a)

for i in range(len(baseline)):
    diff_base += 1 - baseline[i]
    diff_bert += 1 - bert_similarities[i]

if diff_bert < diff_base:
    print('Bert is better!')
elif diff_bert > diff_base:
    print('Bert sucks!')
else:
    print('Tied')
