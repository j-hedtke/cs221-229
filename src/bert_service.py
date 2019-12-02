import numpy as np
from bert_serving.client import BertClient
import math
import re
import os
import subprocess
import argparse
import csv
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

with open(os.path.join(os.pardir, 'data/datacleaned_valid.txt'), encoding="utf8") as fp:
    phrases = fp.read().split('\n')
phrases_list = [list(filter(None, line.strip().split(','))) for line in phrases if line.strip() and re.search('[a-zA-Z]', line)]
first_of_pair = [pair[0] for pair in phrases_list]
second_of_pair = [pair[1] for pair in phrases_list]

def score(phrases, output_path):
    def scoring(pair):
        query_vec_1, query_vec_2 = bc.encode(pair)
        cosine = np.dot(query_vec_1, query_vec_2) / (np.linalg.norm(query_vec_1) * np.linalg.norm(query_vec_2))
        return cosine #1 / (1 + math.exp(-100 * (cosine - 0.95)))


    with BertClient(port=5555, port_out=5556, check_version=False) as bc:

        print("Start testing")
        """
        if pos:
            f = open("bert_cos_similarity.txt", "w+")
        else:
            f = open("neg_bert_cos_similarity.txt", "w+")
        """
        f = open(output_path, 'w+')
        for i, p in enumerate(phrases_list):
            score = scoring(p)
            print("Similarity of Pair {}: ".format(i + 1), score)
            f.write(str(score) + '\n')
    f.close()


def rank_most_similar(n):
    outfile = '../output/similarity_rankings.tsv'
    output = []
    with BertClient(port=5555, port_out=5556, check_version=False) as bc:
        
        for sen1 in first_of_pair[:5]:
            sen_vec1 = bc.encode([sen1])
            matches = []
            for sen2 in second_of_pair:
                sen_vec2 = bc.encode([sen2])
                matches.append((sen2, cosine_similarity(sen_vec1, sen_vec2)))
            matches.sort(key=lambda x: x[1], reverse=True)
            # take top n
            top_n = []
            for i in range(n):
                top_n.append(matches[i][0])
            output.append([sen1, top_n])

    with open(outfile, 'w+') as f:
       writer = csv.writer(f, delimiter='\t')
       for sen in output:
            writer.writerow(sen)

def compute_embeddings():
    infile = open(os.path.join(os.pardir,
                               'data/eng_news_2016_1M/eng_news_2016_1M-sentences.txt'),
                  'r')
    lines = [line for line in infile]
    embeddings = []

    outfile = open(os.path.join(os.pardir, 'output/eng_news_embeddings.jsonl'),
                   'w')

    #writer = csv.writer(outfile, delimiter=',')
    with BertClient(port=5555, port_out=5556, check_version=False) as bc:
        lines.reverse()
        for i,line in enumerate(lines[:100000]):
            ind = line.split()[0]
            sent = line.split()[1]
            emb = bc.encode([sent])[0]
            #embeddings.append([ind, emb])
            #writer.writerow([ind, emb])
            json.dump({ind: emb.tolist()}, outfile)
            outfile.write('\n')
            if i % 1000 == 0:
                print('{} embeddings finished'.format(i))
    
    bc.close()


def pearson():
    pass

def compute_pca():
    pass

if __name__ == '__main__':
    
    """
    with open(os.path.join(os.pardir, 'data/datacleaned_valid.txt'), encoding="utf8") as fp:
        phrases = fp.read().split('\n')
    phrases_list = [list(filter(None, line.strip().split(','))) for line in phrases if line.strip() and re.search('[a-zA-Z]', line)]
    
    #positive
    score(phrases_list)
    
    with open(os.path.join(os.pardir, 'data/neg_datacleaned_valid.txt'), encoding="utf8") as fp:
        phrases = fp.read().split('\n')
    phrases_list = [list(filter(None, line.strip().split(','))) for line in phrases if line.strip() and re.search('[a-zA-Z]', line)]


    #negative
    score(phrases_list, pos=False)
    """
    
    """
    with open(os.path.join(os.pardir, 'data/random_sentences_valid.txt'), encoding="utf8") as fp:
        phrases = fp.read().split('\n')
    phrases_list = [list(filter(None, line.strip().split(','))) for line in phrases if line.strip() and re.search('[a-zA-Z]', line)]


    #negative with random list
    score(phrases_list, 'negrandom_bert_cos_similarity.txt')
    """

    #rank_most_similar(5)
    compute_embeddings()


