import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
#from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification, BertForNextSentencePrediction
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertForMaskedLM
from tqdm import tqdm, trange
import io
import os
import numpy as np
import os

dirname = os.path.dirname(os.path.dirname(__file__))

model_save_path = os.path.join(dirname, 'output/finetuned_stsb_bert_from_pytorch')
train_data_path = os.path.join(dirname, 'data/sts-b/sts-train.csv')
test_data_path = os.path.join(dirname, 'data/sts-b/sts-test.csv')

MAX_LEN = 256
batch_size = 32

learning_rate = 3e-5
epochs = 3

torch.cuda.empty_cache()

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#model.cuda()

scores_train =[]
first_sent_train = []
second_sent_train = []

scores_test =[]
first_sent_test = []
second_sent_test = []

sent_pairs = []

with open(train_data_path, encoding='utf-8') as fin:
    train_data = fin.read().split('\n')
train_data = [line for line in train_data if line.strip()]
for line in train_data:
    pair = []
    line1 = line.split('\t')
    if float(line1[4]) <= 2.5:
        scores_train.append(0)
    else:
        scores_train.append(1)
    first_sent_train.append(line1[5])
    second_sent_train.append(line1[6])
    pair.append(str(line1[5]))
    pair.append(str(line1[6]))
    sent_pairs.append(pair)


with open(test_data_path, encoding='utf-8') as fin:
    test_data = fin.read().split('\n')
test_data = [line for line in test_data if line.strip()]
for line in test_data:
    line1 = line.split('\t')
    if float(line1[4]) <= 2.5:
        scores_test.append(0)
    else:
        scores_test.append(1)
    first_sent_test.append(line1[5])
    second_sent_test.append(line1[6])

pairs_train = []
pairs_test = []
segment_ids_train = []
segment_ids_test = []
tokenized_pairs_train = []
tokenized_pairs_test = []

for sent1, sent2 in zip(first_sent_train, second_sent_train):
    token1 = tokenizer.tokenize(sent1)
    token2 = tokenizer.tokenize(sent2)
    pair_tokens = []
    pair_segment_ids = []
    pair_tokens.append("[CLS] ")
    pair_segment_ids.append(0)
    for t in token1:
        pair_tokens.append(t)
        pair_segment_ids.append(0)
    pair_tokens.append('[SEP]')
    for t in token2:
        pair_tokens.append(t)
        pair_segment_ids.append(1)
    pair_tokens.append('[SEP]')
    pair_segment_ids.append(1)
    tokenized_pairs_train.append(pair_tokens)
    segment_ids_train.append(pair_segment_ids)

for sent1, sent2 in zip(first_sent_test, second_sent_test):
    token1 = tokenizer.tokenize(sent1)
    token2 = tokenizer.tokenize(sent2)
    pair_tokens = []
    pair_segment_ids = []
    pair_tokens.append("[CLS] ")
    pair_segment_ids.append(0)
    for t in token1:
        pair_tokens.append(t)
        pair_segment_ids.append(0)
    pair_tokens.append('[SEP]')
    for t in token2:
        pair_tokens.append(t)
        pair_segment_ids.append(1)
    pair_tokens.append('[SEP]')
    pair_segment_ids.append(1)
    tokenized_pairs_test.append(pair_tokens)
    segment_ids_test.append(pair_segment_ids)

print("the first tokenized pair:")
print(tokenized_pairs_train[0])
print("the first segment ids:")
print(segment_ids_train[0])

input_ids_train = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_pairs_train]
input_ids_train = pad_sequences(input_ids_train, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
input_ids_test = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_pairs_test]
input_ids_test = pad_sequences(input_ids_test, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
segment_ids_train = pad_sequences(segment_ids_train, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
segment_ids_test = pad_sequences(segment_ids_test, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

#encoded = [tokenizer.encode(s, add_special_tokens=True) for s in sent_pairs]
#input_ids2 = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in sent_pairs]).unsqueeze(0)

attention_masks_train = []
attention_masks_test = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids_train:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks_train.append(seq_mask)
for seq in input_ids_test:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks_test.append(seq_mask)

# Convert all of our data into torch tensors, the required datatype for our model

train_inputs = torch.tensor(input_ids_train).to(torch.int64)
validation_inputs = torch.tensor(input_ids_test).to(torch.int64)
train_labels = torch.tensor(scores_train).float()
validation_labels = torch.tensor(scores_test).float()
train_masks = torch.tensor(attention_masks_train).to(torch.int64)
validation_masks = torch.tensor(attention_masks_test).to(torch.int64)
segment_ids_train = torch.tensor(segment_ids_train).to(torch.int64)
segment_ids_test = torch.tensor(segment_ids_test).to(torch.int64)

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels, segment_ids_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, segment_ids_test)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

#BertPreTrainedModel = BertModel.from_pretrained('bert-base-uncased')

class BertSimilarity(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSimilarity, self).__init__(config)
        self.bert = BertModel(config)
        self.linear = torch.nn.Linear(config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        linear_output = self.linear(pooled_output)
        output = self.sigmoid(linear_output)

        return output

model = BertSimilarity.from_pretrained('bert-base-uncased')
model = model.cuda()

# Set our model to training mode (as opposed to evaluation mode)
model.train()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
    ]
optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate, warmup=.1)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# Store our loss and accuracy for plotting
train_loss_set = []

# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):

    # Training

    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_segment_ids = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        probs = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_segment_ids)
        loss_func = torch.nn.BCELoss()
        batch_loss = loss_func(probs, b_labels)

        train_loss_set.append(batch_loss)
        # Backward pass
        batch_loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update tracking variables
        tr_loss += batch_loss
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    # Validation

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_segment_ids = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            sigmoid = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_segment_ids)

        # Move logits and labels to CPU
        sigmoid = sigmoid.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(sigmoid, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

print("Saving to output folder")
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
