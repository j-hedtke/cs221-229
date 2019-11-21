import numpy as numpy
import re
import os
import pathlib
import json

JSON_DIR_PATH = os.pardir + '/data/gs_transcripts'
SEGMENTS_OUTPUT_PATH = JSON_DIR_PATH + '/segments.txt'
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
if __name__ == '__main__':
    #clean_txt(os.pardir + '/data/valid.txt')
    clean_json()
