import numpy as np
import re
import os
import pathlib
import json
import csv
import pandas as pd
from pathlib import Path
import random

JSON_DIR_PATH = os.pardir + '/data/gs_transcripts'
SEGMENTS_OUTPUT_PATH = JSON_DIR_PATH + '/segments.txt'
VALID_SET_PATH = os.path.join(os.pardir, 'data', 'datacleaned_valid.txt')
def clean_txt(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        cleaned = [re.sub(r'[\s\n_-]{2,}', '', l) for l in lines]
        print(cleaned)
    f.close()
    with open(os.path.dirname(file) + 'cleaned_' + os.path.basename(file), 'w') as f:
        f.writelines(cleaned)



def clean_json():
    count = 0
    with open(SEGMENTS_OUTPUT_PATH, 'w') as wf:
        for file in os.scandir(JSON_DIR_PATH):
            with open(file, 'r') as rf:
                try:
                    for trans in json.loads(rf.read()):
                        try:
                            wf.write(trans.get('pretty_transcript') + '\n')
                        except TypeError:
                            continue
                except json.decoder.JSONDecodeError:
                    continue
            count += 1
            print('Wrote {} segments to file'.format(file))
            print('{} files written'.format(count))
            
def shuffle_pairs():
    neg_valid_set_output = Path(VALID_SET_PATH).parents[0]/'neg_datacleaned_valid.txt'
    new_rows = []
    data = pd.read_csv(VALID_SET_PATH, sep=',')
    #import pdb; pdb.set_trace()
    for i, row in enumerate(data.iterrows()):
        new = pd.concat([data.iloc[0:i], data.iloc[i+1:]])
        row = random.randint(0, new.shape[0]-1)
        col = random.randint(0, new.shape[1]-1)
        new_sent = new.iloc[row, col]
        new_rows.append([data.iloc[i, col], new_sent])

    with open (neg_valid_set_output, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

if __name__ == '__main__':
    #clean_txt(os.pardir + '/data/valid.txt')
    shuffle_pairs()
    #clean_json()
