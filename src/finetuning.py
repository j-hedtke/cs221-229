import datetime
import json
import os
import pprint
import random
import string
import sys
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from bert import modeling, optimization, run_classifier 
from bert import run_classifier_with_tfhub, tokenization

DATA_TRAIN = ['MRPC_train', 'SICK_train']
DATA_TEST = ['MRPC_test']

"""Change to args to run as script"""

MAX_SEQ_LENGTH = 256
TRAIN_BATCH_SIZE = 32
LEARNING_RATE = 5e-5
EPOCHS = 5
USE_GPU = True
PREDICT = False
VOCAB_FILE = None
LOWER_CASE = True


DATA_DIR = {
    'MRPC_train': os.path.join(os.pardir, 'data/MRPC/msr_paraphrase_train.txt')
    'MRPC_test': os.path.join(os.pardir, 'data/MRPC/msr_paraphrase_test')
}


class Example():
    def __init__(self, uid, sent_1, sent_2, label):
        self.uid = uid
        self.sent_1 = sent_1
        self.sent_2 = sent_2
        self.label = label

class PaddingExample():
    """Proxy for None"""

class Features():
    def __init__(self, ids, mask, segment_ids, label_id, is_real_example = True)
        self.input_ids = ids
        self.input_mask = mask 
        self.segment_ids = segment_ids 
        self.label_id = label_id 
        self.is_real_example = is_real_example 



def load_bin_class_data(data_name, cols=None, sep='\t'):
    if data_name == 'MRPC':
        cols = [0,3,4]
    with open(DATA_DIR[data_name], 'r') as f:
        rows = pd.read_csv(f, delimiter=sep, usecols=cols)
    return rows

class BinaryClassData():
    def __init__(self, data_name):
        self.data_name = data_name
        if self.data_name == 'MRPC':
            self.load_examples = self._load_examples_MRPC

    def _load_examples_MRPC(rows, type):\
        examples = []
        for i, row in enumerate(rows):
            if i == 0:
                continue
            uid = '{}.{}'.format(type, i)
            sent_1 = tokenization.convert_to_unicode(row[1])
            sent_2 = tokenization.convert_to_unicode(row[2])
            label = row[0]
            examples.append(Example(uid, sent_1, sent_2, label))
        return examples

def featurize_example(example, tokenizer):

    if isinstance(example, PaddingExample):
        return Features(
            input_ids = [0] * max_seq_length,
            input_mask = [0] * max_seq_length,
            segment_ids = [0] * max_seq_length,
            label_id = 0,
            is_real_example = False)

    tokens_1 = tokenizer.tokenize(example.sent_1)
    tokens_2 = tokenizer.tokenize(example.sent_2)

    _truncate_seq_pair(tokens_1, tokens_2, MAX_SEQ_LENGTH - 3)

    tokens = []
    segment_ids = []
    tokens.append('[CLS'])
    segment_ids.append(0)
    for t in tokens_1:
        tokens.append(t)
        segmen_ids.append(0)
    tokens.append('[SEP]')
    for t in tokens_2:
        tokens.append(t)
        segment_ids.append(1)
    tokens.append('[SEP]')
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return Features(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=example.label_id)

def featurize_example_list(examples, max_seq_length, tokenizer):
    features = []
    for i, example in enumerate(examples):
        if i % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (i,
                                                          len(examples)))
        feature = featurize_example(example, tokenizer)
        features.append(feature)

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
        


def main():
    pass

if __name__ == '__main__':
    print('Loaded modules')

