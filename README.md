This repo contains code and files for both cs229 and cs221. 

Here are the relevant files for each of the projects:

####CS229####
/src
  baseline.py
  bert_finetune_sent_transformers_stsb.py
  bert_finetune_with_pytorch.py
  bert_finetuning_from_google_repo.py
  bert_server.py
  bert_service.py
  BERTScore.py
  compare.py
  compare_scores.py
  fine_tune_sts_b.ipynb
  finetuning_nomod.py
  k-means.py
  run_lm_finetuning.py
  util.py
 
/google_bert/bert/*
/data/*
/output/*

Key Files:
baseline.py
bert_finetune_with_pytorch.py
bert_server.py
bert_service.py
fine_tune_sts_b.ipynb
finetuning_nomod.py

####CS221####

/src
  baseline.py
  bert_finetune_with_pytorch.py
  bert_server.py
  bert_service.py
  compare.py
  compare_scores.py
  fine_tune_sts_b.ipynb
  finetuning_nomod.py
  pleb.ipynb
  util.py

/google_bert/bert/*
/data/*
/output/*


Key Files:
baseline.py
bert_finetune_with_pytorch.py
bert_server.py
bert_service.py
fine_tune_sts_b.ipynb
finetuning_nomod.py
pleb.ipynb

####Instructions####

src/

bert_server.py runs Han Xiao's bert-as-service and can be ran as a script to encode sentence vectors using either a pre-trained or finetuned model.

bert_service.py is the bert_server client and where all of the accuracy and analysis functions are implemented. 

finetuning_nomod.py is a script that submits the google_bert/bert/run_classifier.py script to a Gridspace cluster to finetune BERT on the MRPC dataset

finetuning_with_pytorch.py is the custom module based on Chris McCormick's implementation that finetunes on the STS-B dataset. fine_tune_sts_b.ipynb is the notebook that was used on Google Colab to train on a Colab GPU.

pleb.ipynb is the implementation of approximate nearest neighbors via point location in equal balls.

data/

MRPC/ and sts-b/ are self-explanatory.
eng_news_2016_1M/ contains the large dataset of sentences that were used for the ANN experiments and for one of the PCA figures.

data_cleaned_valid.txt is the validation set.
neg_datacleaned_valid.txt contains negative examples of the validation set

output/

accuracy_* contain the accuracies for for each of the models 
dim_reduced* contain the metrics for the PCA experiments
eng_news_embeddings.jsonl_finetuned_mrpc.png, pca_accuracy.png, and pca_explained_variance contain the figures for the pca experiments
