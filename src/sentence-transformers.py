from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
import pytorch_transformers
import numpy as np
import os
import re

dirname = os.path.dirname(os.path.dirname(__file__))

with open(os.path.join(dirname, 'data/datacleaned_valid.txt'), encoding="utf8") as f:
        phrases = f.read().split('\n')
phrases_list = [list(filter(None, line.strip().split(','))) for line in phrases if line.strip()]
first_sentences = [re.sub(r' +', ' ', x[0]) for x in phrases_list]
second_sentences = [re.sub(r' +', ' ', x[1]) for x in phrases_list]

model = SentenceTransformer('bert-base-cased')
#model = SentenceTransformer('C:/TMP/cs229proj/models/bert_finetuned/')
'''
word_embedding_model = models.RoBERTa('roberta-large')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
'''
first_encoded = model.encode(first_sentences)
second_encoded = model.encode(second_sentences)

total = 0
fout = open(os.path.join(dirname, "output/sentence-transformers_cosine_bert-base.txt"), "w+")
for a, b in zip(first_encoded, second_encoded):
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    total += cosine
    fout.write(str(cosine) + '\n')
fout.close()

print('total score = ', total)