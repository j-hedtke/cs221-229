import numpy as np
from bert_serving.client import BertClient
import math
import re
import os

dirname = os.path.dirname(os.path.dirname(__file__))
with open(os.path.join(dirname, 'data/datacleaned_valid.txt'), encoding="utf8") as fp:
     phrases = fp.read().split('\n')
phrases_list = [list(filter(None, line.strip().split(','))) for line in phrases if line.strip() and re.search('[a-zA-Z]', line)]

def scoring(pair):
    query_vec_1, query_vec_2 = bc.encode(pair)
    cosine = np.dot(query_vec_1, query_vec_2) / (np.linalg.norm(query_vec_1) * np.linalg.norm(query_vec_2))
    return 1 / (1 + math.exp(-100 * (cosine - 0.95)))


with BertClient(port=5555, port_out=5556, check_version=False) as bc:

    print("Start testing")

    for i, p in enumerate(phrases_list):
        print("Similarity of Pair {}: ".format(i + 1), scoring(p))

