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
import argparse
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

EMB_DIM = 768


OUTPUT_DIR = os.path.join(os.pardir, 'output')

MODELS = ['baseline', 'bert_base', 'finetuned_mrpc']

with open(os.path.join(os.pardir, 'data/datacleaned_valid.txt'), 'r') as fp:

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

def rank_most_similar_cached(n, model):
    inf1 = os.path.join(OUTPUT_DIR, 'valid_{}_emb_first.txt'.format(model))
    inf2 = os.path.join(OUTPUT_DIR, 'valid_{}_emb_second.txt'.format(model))
    outfile = os.path.join(OUTPUT_DIR,
                           'similarity_rankings_{}.csv'.format(model))
    emb_1 = np.loadtxt(inf1)
    emb_2 = np.loadtxt(inf2)
    
    output = []
    #import pdb;pdb.set_trace()
    for e1, sen1 in zip(emb_1, first_of_pair):
        matches = []
        for e2, sen2 in zip(emb_2, second_of_pair):
            matches.append((sen2, cosine_similarity(e1.reshape(1,-1),
                                                    e2.reshape(1,-1))))
        matches.sort(key=lambda x: x[1], reverse=True)

        top_n = []
        for i in range(n):
            top_n.append(matches[i][0])
        output.append([sen1] + top_n)
    #import pdb;pdb.set_trace()
    with open(outfile, 'w') as f:
       writer = csv.writer(f)
       for sen in output:
            writer.writerow(sen)

def rank_most_similar(n, finetuned=None):
    if finetuned:
        outfile = '../output/similarity_rankings_finetuned_' + finetuned + '.csv'
    else:
        outfile = '../output/similarity_rankings_bert_base.csv'
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
            output.append([sen1] + top_n)

    with open(outfile, 'w+') as f:
       writer = csv.writer(f)
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


def valid_embeddings(outfile_stem):
    outfile1 = '{}/valid_{}_emb_first.txt'.format(OUTPUT_DIR, outfile_stem)
    outfile2 = '{}/valid_{}_emb_second.txt'.format(OUTPUT_DIR, outfile_stem)


    with BertClient(port=5555, port_out=5556, check_version=False) as bc:
        first = bc.encode(first_of_pair)
        second = bc.encode(second_of_pair)

    np.savetxt(outfile1, first)
    np.savetxt(outfile2, second)

    bc.close()

def pearson():
    pass

def rank_most_similar_inmem(n, emb_1, emb_2):
    output = []
    for e1, sen1 in zip(emb_1, first_of_pair):
        matches = []
        for e2, sen2 in zip(emb_2, second_of_pair):
            matches.append((sen2, cosine_similarity(e1.reshape(1,-1),
                                                    e2.reshape(1,-1))))
        matches.sort(key=lambda x: x[1], reverse=True)

        top_n = []
        for i in range(n):
            top_n.append(matches[i][0])
        output.append([sen1] + top_n)
    return output

def compute_accuracy_inmem(sim_rankings):
    n_correct_top_5 = 0
    n_correct_top = 0
    n_total = 0
    for sen1,sen2,row in zip(first_of_pair, second_of_pair, sim_rankings):
        assert(sen1 == row[0])
        if sen2 == row[1]:
            n_correct_top += 1
        if sen2 in row[1:5]:
            n_correct_top_5 +=1
        n_total += 1

    accuracy_top_5 = n_correct_top_5 / n_total
    accuracy_top = n_correct_top / n_total
    return accuracy_top_5, accuracy_top

def compute_pca_no_plot(n, x):
    pca = PCA(n_components=n)

    x = StandardScaler().fit_transform(x)
    prin_components = pca.fit_transform(x)
    return prin_components, pca.explained_variance_ratio_.sum()

def compute_pca_range(inmodel):
    
    in1 = os.path.join(OUTPUT_DIR, 'valid_{}_emb_first.txt'.format(inmodel))
    in2 = os.path.join(OUTPUT_DIR, 'valid_{}_emb_second.txt'.format(inmodel))
    
    f1 = open(in1, 'r')
    f2 = open(in2, 'r')
    first_emb = np.loadtxt(f1)
    second_emb = np.loadtxt(f2)
    f1.close()
    f2.close()
    epsilon = 0.001
    expl_var_threshold = 0.95
    # ith index holds metrics for (i + 1) number of components
    acc = []
    expl_var = []
    x = np.concatenate((first_emb, second_emb), axis=0)
    for n in range(1, EMB_DIM):
        prin_components, expl_var_scalar = compute_pca_no_plot(n, x)
        if expl_var and (abs(expl_var_scalar - expl_var[-1][1]) < epsilon or
                         expl_var_scalar > expl_var_threshold):
            break
        first, second = np.split(prin_components, 2, axis=0)
        rankings = rank_most_similar_inmem(5, first, second)
        acc_top_5, acc_top = compute_accuracy_inmem(rankings)
        acc.append([n, acc_top_5])
        expl_var.append([n, expl_var_scalar])
    
    return acc, expl_var

def plot_pca_expl_var_custom(filename, model, title):
    in1 = os.path.join(OUTPUT_DIR, filename)
    data = []
    with open(in1, 'r') as f:
        for line in f:
            emb = list(json.loads(line).values())[0]
            data.append(emb)

    acc = []
    expl_var = []
    emb = np.array(data)
    mask = np.random.choice([False, True], emb.shape[0], p = [0.9, 0.1])
    emb_sm = emb[mask]
    epsilon = 0.001
    expl_var_threshold = 0.95
    for n in range(1, EMB_DIM):

        prin_components, expl_var_scalar = compute_pca_no_plot(n, emb_sm)
        if expl_var and (abs(expl_var_scalar - expl_var[-1][1]) < epsilon or
                         expl_var_scalar > expl_var_threshold):
            break

        expl_var.append([n, expl_var_scalar])

    sns.set()

    expl_var_pd = pd.DataFrame(np.array(expl_var),columns =
                                   ['N Components', 'Explained Variance'])
    plt.figure()
    ax2 = sns.lineplot(x='N Components', y='Explained Variance',
                       data=expl_var_pd, ci=None)
    ax2.set_title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, '{}_{}.png'.format(filename, model)))

def plot_pca_metrics():
    sns.set()
    ax1 = None
    ax2 = None
    accuracies = []
    expl_variances = []
    final_accuracies = {}
    final_expl_var = {}
    for model in MODELS:
        acc, expl_var = compute_pca_range(model)
        final_accuracies[model] = acc[-1]
        final_expl_var[model] = expl_var[-1]
        acc_pd  = pd.DataFrame(np.array(acc), columns =
                                   ['N Components', 'Accuracy'])
        accuracies.append(acc_pd.assign(model=model))
        
        expl_var_pd  = pd.DataFrame(np.array(expl_var), columns =
                                   ['N Components', 'Explained Variance'])
        expl_variances.append(expl_var_pd.assign(model=model))

    
    concat_acc = pd.concat(accuracies)
    concat_expl_var = pd.concat(expl_variances)

    with open(os.path.join(OUTPUT_DIR, 'dim_reduced_accuracies.csv'), 'w') as f:
        writer = csv.writer(f)
        for k,v in final_accuracies.items():
            writer.writerow([k] + v)
    f.close()
    with open(os.path.join(OUTPUT_DIR, 'dim_reduced_explained_variances.csv'), 'w') as f:
        writer = csv.writer(f)
        for k,v in final_expl_var.items():
            writer.writerow([k] + v)

    plt.figure()
    ax1 = sns.lineplot(x='N Components', y='Accuracy', data=concat_acc,
                       hue='model', style='model', ci=None)
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_accuracy.png'))

    plt.figure()
    ax2 = sns.lineplot(x='N Components', y='Explained Variance',
                       data=concat_expl_var, hue='model',
                       style='model', ci=None)
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_explained_variance.png'))

def compute_pca(model, n, infile, outfile, var_threshold=False):
    first_emb = []
    second_emb = []
    output_path = os.path.join(OUTPUT_DIR, outfile)
    in1 = os.path.join(OUTPUT_DIR, 'valid_{}_emb_first.txt'.format(infile))
    in2 = os.path.join(OUTPUT_DIR, 'valid_{}_emb_second.txt'.format(infile))
    
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


def compute_accuracy(model=None):
    outfile = os.path.join(OUTPUT_DIR, 'accuracy_{}.txt'.format(model))
    rankings_file = os.path.join(OUTPUT_DIR,
                                 'similarity_rankings_{}.csv'.format(model))
    
    sim_rankings = []
    with open(rankings_file, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            sim_rankings.append(r)
    n_correct_top_5 = 0
    n_correct_top = 0
    n_total = 0
    for sen1,sen2,row in zip(first_of_pair, second_of_pair, sim_rankings):
        assert(sen1 == row[0])
        if sen2 == row[1]:
            n_correct_top += 1
        if sen2 in row[1:5]:
            n_correct_top_5 +=1
        n_total += 1

    accuracy_top_5 = n_correct_top_5 / n_total
    accuracy_top = n_correct_top / n_total
    
    with open(outfile, 'w') as f:
        wr = csv.writer(f)
        wr.writerow([str(accuracy_top_5), str(accuracy_top)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running BERT service')
    parser.add_argument('--valid_embeddings')
    parser.add_argument('--pca', nargs='*', help='model, n, infile_stem, outfile_stem')
    parser.add_argument('--plot_pca', action='store_true')
    parser.add_argument('--plot_pca_custom', nargs=3)
    parser.add_argument('--accuracy')
    parser.add_argument('--rankings', nargs='*')
    args = parser.parse_args()
    if args.valid_embeddings:
        valid_embeddings(args.valid_embeddings)
    if args.pca:
        compute_pca(args.pca[0], args.pca[1], args.pca[2], args.pca[3])
    if args.plot_pca:
        plot_pca_metrics()
    if args.plot_pca_custom:
        if args.plot_pca_custom[2] == 'eng_news':
            title = '7.8k Sentences From Subset of the Leipzig Corpora Collection'
        plot_pca_expl_var_custom(args.plot_pca_custom[0],
                                 args.plot_pca_custom[1], title)
    if args.accuracy:
        compute_accuracy(args.accuracy)
    if args.rankings:
        if args.rankings[0] == 'cached':
            rank_most_similar_cached(5, args.rankings[1])
        elif args.rankings[0] == 'not_cached':
            rank_most_similar(5)

