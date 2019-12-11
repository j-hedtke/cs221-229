from pytorch_transformers.convert_pytorch_checkpoint_to_tf import convert_pytorch_checkpoint_to_tf
from transformers import BertModel
import os

model = BertModel.from_pretrained(os.pardir + '/models/finetuned/sts-b/pytorch/')
convert_pytorch_checkpoint_to_tf(model, os.pardir + '/models/finetuned/sts-b/tf/', "fine_tuned_tf")
'''
main([
'--model_name', '/home/pbakhtin/models/bert/scibert_fine_tuned_pytorch/',
'--pytorch_model_path', '/home/pbakhtin/models/bert/scibert_fine_tuned_pytorch/pytorch_model.bin',
'--tf_cache_dir', '/home/pbakhtin/models/bert/scibert_fine_tuned_pytorch/tf',
'--cache_dir', '/home/pbakhtin/models/bert/scibert_fine_tuned_pytorch/'
])
'''