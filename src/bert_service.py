import numpy as np
from bert_serving.client import BertClient
import math
import re
import os
import subprocess
import argparse

def score(phrases, pos=True):
    def scoring(pair):
        query_vec_1, query_vec_2 = bc.encode(pair)
        cosine = np.dot(query_vec_1, query_vec_2) / (np.linalg.norm(query_vec_1) * np.linalg.norm(query_vec_2))
        return cosine #1 / (1 + math.exp(-100 * (cosine - 0.95)))


    with BertClient(port=5555, port_out=5556, check_version=False) as bc:

        print("Start testing")
        if pos:
            f = open("bert_cos_similarity.txt", "w+")
        else:
            f = open(os.path.join(os.pardir, "output/mix_sent_bert_finetuned_cos_similarity.txt"), "w+")

        for i, p in enumerate(phrases_list):
            score = scoring(p)
            print("Similarity of Pair {}: ".format(i + 1), score)
            f.write(str(score) + '\n')
    f.close()



if __name__ == '__main__':
    '''
    with open(os.path.join(os.pardir, 'data/datacleaned_valid.txt'), encoding="utf8") as fp:
        phrases = fp.read().split('\n')
    phrases_list = [list(filter(None, line.strip().split(','))) for line in phrases if line.strip() and re.search('[a-zA-Z]', line)]
    
    #positive
    score(phrases_list)
    '''
    with open(os.path.join(os.pardir, 'data/mix_random_valid.txt'), encoding="utf8") as fp:
        phrases = fp.read().split('\n')
    phrases_list = [list(filter(None, line.strip().split(','))) for line in phrases if line.strip() and re.search('[a-zA-Z]', line)]


    #negative
    score(phrases_list, pos=False)

