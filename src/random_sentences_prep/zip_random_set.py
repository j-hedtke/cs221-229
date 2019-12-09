import numpy as np
import os
import re

dirname = os.path.dirname(os.path.dirname(__file__))

with open(os.path.join(dirname, 'data/datacleaned_valid.txt'), encoding="utf8") as f:
        phrases = f.read().split('\n')
phrases_list = [list(filter(None, line.strip().split(','))) for line in phrases if line.strip()]
first_sentences = [re.sub(r' +', ' ', x[0]) for x in phrases_list]

with open(os.path.join(dirname, 'data/random_sentences_cleaned.txt'), encoding="utf8") as f:
    phrases = f.read().split('\n')
phrases_list = [list(filter(None, line.strip().split('\n'))) for line in phrases if line.strip()]
second_sentences = [re.sub(r' +', ' ', x[0]) for x in phrases_list]

f = open(os.path.join(dirname, "data/random_sentences_valid.txt"), "w+")
for x, y in zip(first_sentences, second_sentences):
    x = str(x)
    x = re.sub("[^a-zA-Z0-9 ]+", "", x)
    y = str(y)
    y = re.sub("[^a-zA-Z0-9 ]+", "", y)
    f.write(str(x) + ',' + str(y) + '\n')
f.close()