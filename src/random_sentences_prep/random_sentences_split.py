import numpy as np
import os
import re

dirname = os.path.dirname(os.path.dirname(__file__))

file = 'data/random_sentences.txt'

with open(os.path.join(dirname, file), encoding="utf8") as f:
    fin = f.readlines()
sentences = str(re.sub(r'\([^)]*\)', '', str(fin)))
sentences = [x.replace('\\n', '').replace(',', '').replace('\'', '').replace('\\', '') for x in map(str.strip, str(sentences).split('. ')) if x]
sentences = [str(line).split('.') for line in sentences]

f = open(os.path.join(dirname, "data/random_sentences_cleaned.txt"), "w+")
for i, line in enumerate(sentences):
    line = str(line)
    line = re.sub("[^a-zA-Z ]+", "", line)
    f.write(str(line) + '\n')
f.close()