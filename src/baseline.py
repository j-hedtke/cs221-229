import os
import pathlib
import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors


OUTPUT_DIR = os.path.join(os.pardir, 'output') 
VALID = os.pardir + '/data/datacleaned_valid.txt'
WORD2VECPATH = pathlib.Path('../data/GoogleNews-vectors-negative300.bin')

WORD2VECPATH = pathlib.Path('../models/GoogleNews-vectors-negative300.bin')

m = KeyedVectors.load_word2vec_format(WORD2VECPATH, binary=True)

def average_word_vec(sentence, m, index2word_set, n_features=300):
    feature_vec = np.zeros((n_features,))
    n_words = 0
    words = sentence.split()
    for word in words:
        if word in index:
            n_words += 1
            feature_vec += m[word]

    if n_words:
        feature_vec /= n_words
        return feature_vec

def compute_word_mover_distance():
    pass

def compute_cosine_sim(sentence_1, sentence_2, m, index2word_set):
    return 1 - spatial.distance.cosine(average_word_vec(sentence_1, m, index2word_set), average_word_vec(sentence_2, m, index2word_set))

def similarity_baselines():
    index2word_set = set(m.index2word)
    valid = np.loadtxt(VALID, dtype=str, delimiter = ',', encoding="utf8")
    res = np.zeros(valid.shape[0])
    for i in range(valid.shape[0]):
        res[i] = compute_cosine_sim(valid[i][0], valid[i][1], m, index2word_set)

    np.savetxt(os.pardir + '/output/avg_cos_similarity.txt', res)

def compute_avg_word_embeddings():
    index2word_set = set(m.index2word)
    valid = np.loadtxt(VALID, dtype=str, delimiter = ',')
    res = np.zeros(valid.shape[0])
    res1 = []
    res2 = []
    for i in range(valid.shape[0]):
        res1.append(average_word_vec(valid[i][0], m, index2word_set))
        res2.append(average_word_vec(valid[i][1], m, index2word_set))

    np.savetxt(os.pardir + '/output/valid_baseline_emb_first.txt', res1)
    np.savetxt(os.pardir + '/output/valid_baseline_emb_second.txt', res2)

if __name__ == '__main__':
    #similarity_baselines()
    compute_avg_word_embeddings()

