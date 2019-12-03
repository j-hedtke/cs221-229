import os
import subprocess
import argparse

BASE_MODEL_DIR = 'C:/TMP/cs229proj/models/uncased_L-12_H-768_A-12' #os.path.join(os.pardir, 'models', 'uncased_L-12_H-768_A-12')
FINETUNED_DIR = 'C:/TMP/cs229proj/models/bert_finetuned/' #os.path.join(os.pardir, 'models', 'finetuned', 'mrpc/MRPC')
MAX_SEQ_LEN = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running BERT as a service')
    parser.add_argument('--finetuned')
    args = parser.parse_args()
    #import pdb;pdb.set_trace()
    if args.finetuned:
        if args.finetuned == 'mrpc':
            cmd = ['bert-serving-start', '-model_dir={}'.format(BASE_MODEL_DIR),
        '-tuned_model_dir={}'.format(FINETUNED_DIR),
        '-ckpt_name=model.ckpt-343', '-max_seq_len={}'.format(MAX_SEQ_LEN)]
    else:
        cmd = ['bert-serving-start', '-model_dir={}'.format(BASE_MODEL_DIR)]

    p = subprocess.check_call(cmd, timeout=86400)

