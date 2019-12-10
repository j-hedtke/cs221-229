import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors


VALID = os.pardir + '/data/random_sentences_valid.txt'
WORD2VECPATH = pathlib.Path('../models/GoogleNews-vectors-negative300.bin')


def average_word_vec(sentence, m, index2word_set, n_features=300):
    feature_vec = np.zeros((n_features,))
    n_words = 0
    words = sentence.split()
    for word in words:
        if word in index2word_set:
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
    m = KeyedVectors.load_word2vec_format(WORD2VECPATH, binary=True)
    index2word_set = set(m.index2word)
    valid = np.loadtxt(VALID, dtype=str, delimiter = ',', encoding="utf8")
    res = np.zeros(valid.shape[0])
    for i in range(valid.shape[0]):
        res[i] = compute_cosine_sim(valid[i][0], valid[i][1], m, index2word_set)

    np.savetxt(os.pardir + '/output/avg_cos_similarity.txt', res)

def generate_embeddings():
    m = KeyedVectors.load_word2vec_format(WORD2VECPATH, binary=True)
    index2word_set = set(m.index2word)
    valid = np.loadtxt(VALID, dtype=str, delimiter=',', encoding="utf8")
    first = []
    second = []
    for i in range(valid.shape[0]):
        first.append(average_word_vec(valid[i][0], m, index2word_set))
        second.append(average_word_vec(valid[i][1], m, index2word_set))
    np.savetxt(os.pardir + '/output/valid_embeddings_first_random_baseline.txt', first)
    np.savetxt(os.pardir + '/output/valid_embeddings_second_random_baseline.txt', second)

if __name__ == '__main__':
    #similarity_baselines()
    generate_embeddings()