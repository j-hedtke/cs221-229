from https://github.com/hanxiao/bert-as-service:
(implemented in the bert_service.py in the source code)

Starting the BERT service
After installing the server (in the requirements.txt), you should be able to use bert-serving-start CLI as follows:

bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/

where /tmp/english_L-12_H-768_A-12/ is the path to the pretrained model you want to use

After that you will be able to run calculations using BERT service.
The code outputs cosine similarities for pairs of sentences tokenized by the utilized model. 


from https://github.com/Tiiiger/bert_score:
(implemented in the BERTScore.py in the source code)

for English (en) the default model is roberta-large
A model can be changed using model_type=MODEL_TYPE when calling bert_score.score function.
The outputs of the score function are Tensors of precision, recall, and F1 respectively.
