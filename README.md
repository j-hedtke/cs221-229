from https://github.com/hanxiao/bert-as-service:

Starting the BERT service
After installing the server (in the requirements.txt), you should be able to use bert-serving-start CLI as follows:

bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/

where /tmp/english_L-12_H-768_A-12/ is the path to the pretrained model you want to use

After that you will be able to run calculations using BERT service (bert_service.py in the source code)
The code outputs cosine similarities for pairs of sentences tokenized by the utilized model. 