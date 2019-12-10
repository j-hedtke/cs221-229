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
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

with open(os.path.join(os.pardir, 'data/random_sentences_valid.txt'), encoding="utf8") as fp:
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


def rank_most_similar(n, finetuned=None):
    if finetuned:
        outfile = '../output/similarity_rankings_finetuned_' + finetuned + '.tsv'
    else:
        outfile = '../output/similarity_rankings.tsv'
    output = []
    emb_cache_1 = defaultdict(list)
    emb_cache_2 = defaultdict(list)
    with BertClient(port=5555, port_out=5556, check_version=False) as bc:
        
        for sen1 in first_of_pair:
            if emb_cache_1.get(sen1, None) is not None:
                sen_vec1 = emb_cache_1[sen1]
            else:
                sen_vec1 = bc.encode([sen1])
                emb_cache_1[sen1] = sen_vec1
            matches = []
            for sen2 in second_of_pair:
                if emb_cache_2.get(sen2, None) is not None:
                    sen_vec2 = emb_cache_2[sen2]
                else:
                    sen_vec2 = bc.encode([sen2])
                    emb_cache_2[sen2]  = sen_vec2
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

def valid_embeddings():
    outfile1 = '../output/valid_embeddings_first_random_finetuned.txt'
    outfile2 = '../output/valid_embeddings_second_random_finetuned.txt'

    with BertClient(port=5555, port_out=5556, check_version=False) as bc:
        first = bc.encode(first_of_pair)
        second = bc.encode(second_of_pair)

    np.savetxt(outfile1, first)
    np.savetxt(outfile2, second)

    bc.close()

def pearson():
    pass

def compute_pca(var_threshold=False, n=2):
    first_emb = []
    second_emb = []
    output_path = '../output/pca.png'
    in1 = '../output/valid_embeddings_first.txt'
    in2 = '../output/valid_embeddings_second.txt'
    f1 = open(in1, 'r')
    f2 = open(in2, 'r')
    first_emb = np.loadtxt(f1)
    second_emb = np.loadtxt(f2)
    f1.close()
    f2.close()
    x1 = first_emb[0,:].reshape((1, 768))

    x = np.concatenate((x1, second_emb[0:4, :]), axis=0)
    #x2 = first_emb[0,:]
    if not var_threshold:
        pca = PCA(n_components=n)
    else:
        pca = PCA(0.95)
    x = StandardScaler().fit_transform(x)
    prin_components = pca.fit_transform(x)
    
    if not var_threshold:
        fig = plt.figure(figsize= (8,8))
        pc = fig.add_subplot(1,1,1)
        pc.set_xlabel('Principal Component 1', fontsize = 15)
        pc.set_ylabel('Principal Component 2', fontsize = 15)
        plt.scatter(prin_components[:,0], prin_components[:,1])

        plt.savefig(output_path)


    print('Explained variance: {}'.format(pca.explained_variance_ratio_.sum()))
    print('Principal components: {}'.format(prin_components))
    print('N principal components: {}'.format(prin_components.shape[1]))
    print(first_of_pair[0])
    print(second_of_pair[0:4])

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

    #rank_most_similar(5, finetuned='mrpc')
    #rank_most_similar(5)
    #compute_embeddings()
    valid_embeddings()
    #compute_pca(var_threshold=True)
    #compute_pca(n=3)


