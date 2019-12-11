import os
import subprocess
import argparse

BASE_MODEL_DIR = os.path.join(os.pardir, 'models', 'uncased_L-12_H-768_A-12')
FINETUNED_DIR = os.path.join(os.pardir, 'models', 'finetuned', 'mrpc/MRPC')
FINETUNED_DIR_STSB = os.path.join(os.pardir, 'models', 'finetuned', 'sts-b/tf')
MAX_SEQ_LEN = 128
NUM_WORKERS = 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running BERT as a service')
    parser.add_argument('--finetuned')
    args = parser.parse_args()
    #import pdb;pdb.set_trace()
    if args.finetuned:
        if args.finetuned == 'mrpc':
            cmd = ['bert-serving-start', '-model_dir={}'.format(BASE_MODEL_DIR),
        '-tuned_model_dir={}'.format(FINETUNED_DIR),
        '-ckpt_name=model.ckpt-343', '-max_seq_len={}'.format(MAX_SEQ_LEN),
        '-num_worker={}'.format(NUM_WORKERS)]
        if args.finetuned == 'stsb':
            cmd = ['bert-serving-start', '-model_dir={}'.format(BASE_MODEL_DIR),
        '-tuned_model_dir={}'.format(FINETUNED_DIR_STSB),
        '-ckpt_name=fine_tuned_tf.ckpt', '-max_seq_len={}'.format(MAX_SEQ_LEN),
        '-num_worker={}'.format(NUM_WORKERS)]
    else:
        cmd = ['bert-serving-start', '-model_dir={}'.format(BASE_MODEL_DIR),
               '-max_seq_len={}'.format(MAX_SEQ_LEN)]

    p = subprocess.check_call(cmd, timeout=86400)

