import os
import subprocess
#from bert import modeling, optimization, run_classifier 
#from bert import run_classifier_with_tfhub, tokenization

DATA_DIR = os.path.join(os.pardir, 'data')
BERT_BASE_DIR = os.path.join(os.pardir, 'models/uncased_L-12_H-768_A-12')
JOB_BASE_DIR = os.pardir
OUTPUT_DIR = 'gs://bert-simsearch'
#OUTPUT_DIR = '/tmp/mrpc_output/'
EXEC_PATH = '../google_bert/bert/run_classifier.py'

from research import Job


def main():
    command = 'python {} \
  --task_name=MRPC \
  --do_train=true \
  --data_dir={}/MRPC \
  --vocab_file={}/vocab.txt \
  --bert_config_file={}/bert_config.json \
  --init_checkpoint={}/bert_model.ckpt \
  --max_seq_length=256 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir={}'.format(EXEC_PATH, DATA_DIR, BERT_BASE_DIR, BERT_BASE_DIR
                          , BERT_BASE_DIR, OUTPUT_DIR)

    #subprocess.check_call(command, shell=True)

    job = Job('stock-run-classifier-mrpc', JOB_BASE_DIR)

    job.build()
    job.run(command)


if __name__ == '__main__':
    main()
