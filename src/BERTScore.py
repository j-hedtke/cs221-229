import numpy as np
from bert_score import score
import tensorflow as tf
import os
import re

dirname = os.path.dirname(os.path.dirname(__file__))

with open(os.path.join(dirname, 'data/datacleaned_valid.txt'), encoding="utf8") as f:
        phrases = f.read().split('\n')
phrases_list = [list(filter(None, line.strip().split(','))) for line in phrases if line.strip()]

first_sentences = [re.sub(r' +', ' ', x[0]) for x in phrases_list]
second_sentences = [re.sub(r' +', ' ', x[1]) for x in phrases_list]

(P, R, F), hashname = score(first_sentences, second_sentences, lang='en', idf=True, return_hash=True) #model_type='bert-base-uncased',

fout = open(os.path.join(dirname, "output/bertscore_F1_roberta_idf.txt"), "w+")
for val in F.numpy().flatten():
        fout.write(str(val) + '\n')
fout.close()
#print(f'{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}')