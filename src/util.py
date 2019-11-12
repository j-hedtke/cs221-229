import numpy as numpy
import re
import os
import pathlib

def clean_txt(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        cleaned = [re.sub(r'[\s\n_-]{2,}', '', l) for l in lines]
        print(cleaned)
    f.close()
    with open(os.path.dirname(file) + 'cleaned_' + os.path.basename(file), 'w') as f:
        f.writelines(cleaned)

if __name__ == '__main__':
    clean_txt(os.pardir + '/data/valid.txt')