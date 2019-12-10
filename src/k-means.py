import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

embed_baseline1_path = os.pardir + '/output/valid_embeddings_first_baseline.txt'
embed_baseline2_path = os.pardir + '/output/valid_embeddings_second_baseline.txt'
embed_baseline1_random_path = os.pardir + '/output/valid_embeddings_first_random_baseline.txt'
embed_baseline2_random_path = os.pardir + '/output/valid_embeddings_second_random_baseline.txt'
embed_finetuned1_path = os.pardir + '/output/valid_embeddings_first_finetuned.txt'
embed_finetuned2_path = os.pardir + '/output/valid_embeddings_first_finetuned.txt'
embed_base1_path = os.pardir + '/output/valid_embeddings_first_base.txt'
embed_base2_path = os.pardir + '/output/valid_embeddings_second_base.txt'
embed_base_random1_path = os.pardir + '/output/valid_embeddings_first_random_baseline.txt'
embed_base_random2_path = os.pardir + '/output/valid_embeddings_second_random_baseline.txt'
embed_finetuned_random1_path = os.pardir + '/output/valid_embeddings_first_random_finetuned.txt'
embed_finetuned_random2_path = os.pardir + '/output/valid_embeddings_second_random_finetuned.txt'
VALID = os.pardir + '/data/datacleaned_valid.txt'
VALID_RANDOM = os.pardir + '/data/random_sentences_valid.txt'
outfile = os.pardir + '/output/K-means/valid_clusters_'

def load_data(file1, file2):
    embedding = []
    for line in open(file1): # no need to use readlines if you don't want to store them
        line_array = np.array([float(i) for i in line.split(' ')])
        embedding.append(line_array)
    for line in open(file2): # no need to use readlines if you don't want to store them
        line_array = np.array([float(i) for i in line.split(' ')])
        embedding.append(line_array)
    return embedding

def kmeans_calc(embed_finetuned):
    kmeans = KMeans(n_clusters=5, random_state=0).fit(embed_finetuned)
    clusters = kmeans.labels_

    return clusters

if __name__ == '__main__':
    valid = np.loadtxt(VALID, dtype=str, delimiter=',', encoding="utf8")
    first = [val[0] for val in valid]
    second = [val[1] for val in valid]
    valid_sentences = []
    for i in range(valid.shape[0]):
        valid_sentences.append(first[i])
    for i in range(valid.shape[0]):
        valid_sentences.append(second[i])

    embed_baseline = load_data(embed_baseline1_path, embed_baseline2_path)
    #embed_base = load_data(embed_base1_path, embed_base2_path)
    #embed_finetuned = load_data(embed_finetuned1_path, embed_finetuned2_path)
    #embed_base_random = load_data(embed_base_random1_path, embed_base_random2_path)
    #embed_finetuned_random = load_data(embed_finetuned_random1_path, embed_finetuned_random2_path)
    #embed_baseline_random = load_data(embed_baseline1_random_path, embed_baseline2_random_path)
    clusters = np.array(kmeans_calc(embed_baseline))
    num_of_clusters = int(max(clusters) - min(clusters))

    for c in range(num_of_clusters + 1):
        sent_from_c = [sent for cluster, sent in zip(clusters, valid_sentences) if cluster == c]
        if len(sent_from_c) >= 5:
            rand5 = np.random.choice(sent_from_c, 5, replace=False)
        else:
            rand5 = np.random.choice(sent_from_c, len(sent_from_c), replace=False)
        name = outfile + 'baseline_' + 'cluster_' + str(c + 1) + '_out_of_' + str(num_of_clusters + 1) + '.txt'
        f1 = open(name, 'w+', encoding="utf8")
        [f1.write(str(sent) + '\n') for sent in rand5]
    '''
    f1 = open(outfile + 'base_' + str(num_of_clusters + 1) + '_clusters.txt', 'w+', encoding="utf8")
    for sent, cluster in zip(valid_sentences, clusters):
        f1.write(str(sent) + ' , ' + str(cluster) + '\n')
    f1.close()
    '''