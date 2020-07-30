from nltk.corpus import indian
import nltk
import torch
from torch import *
from torch import nn, tensor
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time, gc, os, typing
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, AdamW
from torchtext import datasets
from collections import Counter
from sklearn.model_selection import train_test_split
import conllu
from conllu import parse_incr
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type=str, required=True, metavar='', help='Path to training file')
parser.add_argument('--valid_data_path', type=str, required=True, metavar='', help='Path to validation file')
parser.add_argument('--base_model_type', type=str, metavar='', required=True, help='Type of transformer model. Currently supports BERT and DistilBERT.')
parser.add_argument('--language', type=str, required=True, help="Multingual DistilBERT or Hindi distilBERT")
parser.add_argument('--freeze', type=bool, metavar='', help='Freeze the base model if True, finetune if False', default=True)
parser.add_argument('--batch_size', type=int, metavar='', default=32)
parser.add_argument('--epochs', type=int, metavar='', help='Epochs to train', default=5)
parser.add_argument('--lr', type=float, metavar='', help='Learning rate', default=0.001)
args = parser.parse_args()

def parse_ud_data(path:str)->tuple:
    '''
    Takes in the path of dataset, parses the CONLL-U format and returns 3 lists.
    Returns
    -all_words: list of lists, where each list has words from one example
    -all_tags: list of lists, each list is the tag sequence for the example
    -tags_list: list for creating tag2idx mapping.
    
    '''
    data_file = open(path,'r',encoding='utf-8')
    tl = []
    for tokenlist in parse_incr(data_file):
        tl.append(tokenlist)
        
    all_words = []
    all_tags = []
    tags_list = []
    for tokenlist in tl:
        words = []
        tags = []
        for i in range(len(tokenlist)):
            words.append(tokenlist[i]['form'])
            tags.append(tokenlist[i]['upos'])
            tags_list.append(tokenlist[i]['upos'])

        assert len(words) == len(tags)
        all_words.append(words)
        all_tags.append(tags)
    
    return all_words, all_tags, tags_list

def create_tag2idx(tags):
    '''
    Creates a vocabulary for labels/tags.
    '''
    tag_counter = Counter(tags)
    tag_vocab = sorted(tag_counter, key=tag_counter.get, reverse=True)
    #print(f"raw-vocab: {len(tag_vocab)}")
    
    tag_vocab.insert(0, '[PAD]')
    tag_vocab.insert(1, '[UNK]')
    tag_vocab.append("[CLS]")
    tag_vocab.append("[SEP]")
    
    #print(f"vocab-length: {len(tag_vocab)}")
    tag2idx = {tag:idx for idx, tag in enumerate(tag_vocab)}
    print(f"tag2idx-length: {len(tag2idx)}")
    idx2tag = {v:k for k,v in tag2idx.items()}
    
    return tag2idx, idx2tag


class POSDataset:
    
    def __init__(self, tokenizer, data, batch_size, tag2idx):
        
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        
        # divide the data into batches
        # list of lists where each list contains batch_size number of examples
        data = [data[i: i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        self.tag2idx = tag2idx
    
    def __len__(self):
        return len(self.data)
    
   
    def __iter__(self):
        
        max_seq_length = 300
        
        # iterate through batches within the data
        for i, batch in enumerate(self.data):
            
            # holder lists for batches
            _input_ids = []
            _input_mask = []
            _label_ids = []
            _label_mask = []
            _valid_ids = []
            _segment_ids = []
            texts = []
            
            # iterate through each example within a batch
            for i in range(len(batch)):
                word_list = batch.iloc[i].words
                label_list = batch.iloc[i].tags
                #text = batch.iloc[i].sentence
                tokens = []
                labels = []
                valid_positions = []
                label_mask = []
                
                # iterate through words of the example
                for i, word in enumerate(word_list):
                    
                    
                    # tokenize the word. Here its possible that the tokenization
                    # returns multiple tokens due to sub words. For example
                    # the word "mister" might get split into ["mis", "##ter"]
                    token = self.tokenizer.tokenize(word)
                    
                    # add all the tokens to the token list.
                    tokens.extend(token)
                    
                    # Extract the label for the token. If the token splits into
                    # subwords, we'll only consider the hidden_states for the first token
                    # and not the "##ter" subword. This is done by maintaining 
                    # a binary array/list of valid_positions.
                    corresponding_label = label_list[i]
                    
                    for j in range(len(token)):
                        if j == 0:
                            labels.append(corresponding_label)
                            valid_positions.append(1)
                            label_mask.append(1)
                        else:
                            valid_positions.append(0)
                            
                # Create a fresh list of tokens that will finally form the input_ids for
                # our model. We also need to prepend "[CLS]" token in the beginning 
                # and [SEP] token at the end of our sequence.
                input_tokens = []
                
                # Not required for distilbert. Used for other models in sentence-pair tasks
                # like QA etc.
                segment_ids = []
                
                # Converting the labels/tags into their IDs.
                label_ids = []
                
            
                input_tokens.append("[CLS]")
                segment_ids.append(0)
                valid_positions.insert(0,1)
                
                # label_mask is also a binary list that maintains 1 for labels and 0 for padding indices
                # Not used in our case. For loss calculation we use ingore_index parameter, which 
                # works fine.
                label_mask.insert(0,1)
                label_ids.append(self.tag2idx["[CLS]"])
                
                
                # Transfer the tokens collected above into input_tokens after adding special tokens.
                
                # example: 
                # tokens = ["Win", "##ter", "is", "com", "##ing"]
                # valid = [1,0,1,1,0]
                # labels = [A,              , B,   C]
                # the condition prevents the iteration going to the last token, i.e ##ing
                # because it would be out of index as there are no labels for subword elements.
                
                for i, token in enumerate(tokens):
                    input_tokens.append(token)
                    segment_ids.append(0)
                    
                    if len(labels) > i:
                        label_ids.append(self.tag2idx[labels[i]])
                    
                input_tokens.append("[SEP]")
                segment_ids.append(0)
                valid_positions.append(1)
                label_mask.append(1)
                label_ids.append(self.tag2idx["[SEP]"])
                
                # Convert the input_tokens into respective ids.
                input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
                input_mask = [1] * len(input_ids)
                label_mask = [1] * len(input_ids)
                
                # sanity check
                # The length of label_ids should equal the number of valid positions.
                assert sum(valid_positions) == len(label_ids)
                
                
                # padding
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    label_ids.append(0)
                    valid_positions.append(0)
                    label_mask.append(0)
                while len(label_ids) < max_seq_length:
                    label_ids.append(0)
                
                
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(label_ids) == max_seq_length
                assert len(valid_positions) == max_seq_length
                assert len(label_mask) == max_seq_length     
                
                
                # stack the examples in the list
                _input_ids.append(input_ids)
                _input_mask.append(input_mask)
                _label_ids.append(label_ids)
                _label_mask.append(label_mask)
                _valid_ids.append(valid_positions)
                _segment_ids.append(segment_ids)
                #texts.append(text)
           
            yield { 
                'input_ids':torch.tensor(_input_ids, dtype=torch.long),
                'input_mask':torch.tensor(_input_mask, dtype=torch.long),
                'label_ids':torch.tensor(_label_ids, dtype=torch.long),
                'label_mask':torch.tensor(_label_mask, dtype=torch.long),
                'valid_ids':torch.tensor(_valid_ids, dtype=torch.long),
                'segment_ids':torch.tensor(_segment_ids, dtype=torch.long)
                #'text':texts
            }
            

            
class POS(nn.Module):
    
    def __init__(self, num_labels, base_model, base_model_type, freeze, device):
        
        super().__init__()
        self.freeze = freeze
        self.device = device
        self.num_labels = num_labels
        self.base_model = base_model
        self.base_model_type = base_model_type
        self.fc1 = nn.Linear(768, 100)
        self.fc2 = nn.Linear(100, num_labels)
        
    def forward(self, input_ids, input_mask, valid_ids, segment_ids):
        
       
            
        if self.freeze == True:
            with torch.no_grad():
                
                if self.base_model_type == 'bert':
                    sequence_output, pooled_output = self.base_model(input_ids=input_ids, 
                                                                     attention_mask=input_mask, 
                                                                     token_type_ids=segment_ids)
                
                elif self.base_model_type == 'distilbert':
                    sequence_output = self.base_model(input_ids=input_ids, attention_mask=input_mask)[0]
        
        else:
            if self.base_model_type == 'bert':
                sequence_output, pooled_output = self.base_model(input_ids=input_ids, 
                                                                 attention_mask=input_mask, 
                                                                 token_type_ids=segment_ids)
            
            elif self.base_model_type == 'distilbert':
                    sequence_output = self.base_model(input_ids=input_ids, attention_mask=input_mask)[0]
        
            
            
        batch_size, max_len, feature_dim = sequence_output.shape
        
        valid_output = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32, device=self.device)
        
        for i in range(batch_size):
            m = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    m += 1
                    valid_output[i][m] = sequence_output[i][j]
        
        sequence_output = F.dropout(valid_output, p=0.3)
        logits = self.fc2(self.fc1(sequence_output))
        # [bs, seq_len, num_labels]
        
        logits = logits.view(-1, self.num_labels)
        # [N, num_labels]
        
        return logits      
    
    
    
def categorical_accuracy(preds, y, tag_pad_idx=0):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # preds = [N, num_labels]
    # y = [N]
    # N = bs * seq_len
    
    # Gets the index of maximum values across the first dimension.
    max_preds = preds.argmax(dim = 1, keepdim = True)
    # [N, 1]
    
    # Gets the index of elements from the ground-truth that are not 0. 
    # Basically index positions which are not padded.
    non_pad_elements = (y != tag_pad_idx).nonzero()
    # [num_nonzero, 1]
    
    # Use non_pad_elements to index the predictions and ground truth. tensor of bools.
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    # [num_nonzero, 1]
    
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])



def seqeval_metrics(preds, y, idx2tag, tag_pad_idx=0):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # preds = [N, num_labels]
    # y = [N]
    # N = bs * seq_len
    
    # Gets the index of maximum values across the first dimension.
    max_preds = preds.argmax(dim = 1, keepdim = True)
    # [N, 1]
    
    # Gets the index of elements from the ground-truth that are not 0. 
    # Basically index positions which are not padded.
    non_pad_elements = (y != tag_pad_idx).nonzero()
    # [num_nonzero, 1]
    
    nonzero_preds = max_preds[non_pad_elements].squeeze(1).tolist()
    nonzero_y = y[non_pad_elements].tolist()
    
    y_true = [[idx2tag[l[0]]] for l in nonzero_y]
    y_pred = [[idx2tag[l[0]]] for l in nonzero_preds]
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return accuracy, f1
    
     
    
def train(model, optimizer, train_dataset):
    
    print("Starting Training")
    train_loss = 0.
    train_acc = 0.
    model.train()
    
    for bi, batch in enumerate(train_dataset):

        if bi % 50 == 0:
            print(f"Starting batch: {bi}")

        input_ids = batch['input_ids'].to(device)
        input_mask = batch['input_mask'].to(device)
        label_ids = batch['label_ids'].to(device)
        valid_ids = batch['valid_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        
        preds = model(input_ids, input_mask, valid_ids, segment_ids)
        loss = F.cross_entropy(preds, label_ids.view(-1), ignore_index=0)
        train_acc += categorical_accuracy(preds, label_ids.view(-1)).item()
        #train_acc += accuracy_score(label_ids.view(-1).tolist(), preds.tolist())
        
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return train_loss/len(train_dataset), train_acc/len(train_dataset)


def validate(model, valid_dataset, idx2tag):
    
    print("Starting validation")
    valid_loss = 0.
    valid_acc = 0.
    model.eval()
    valid_f1 = 0.
    
    for bi, batch in enumerate(valid_dataset):

        if bi % 50 == 0:
            print(f"Starting batch: {bi}")

        input_ids = batch['input_ids'].to(device)
        input_mask = batch['input_mask'].to(device)
        label_ids = batch['label_ids'].to(device)
        valid_ids = batch['valid_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        
        with torch.no_grad():
            
            preds = model(input_ids, input_mask, valid_ids, segment_ids)
            loss = F.cross_entropy(preds, label_ids.view(-1), ignore_index=0)
        
            valid_loss += loss.item()
            #valid_acc += categorical_accuracy(preds, label_ids.view(-1)).item()
            acc, f1  = seqeval_metrics(preds, label_ids.view(-1), idx2tag)
            valid_acc += acc
            valid_f1 += f1
            
    valid_loss = valid_loss / len(valid_dataset)
    valid_acc = valid_acc/len(valid_dataset)
    valid_f1 = valid_f1/len(valid_dataset)
    
    return valid_loss, valid_acc, valid_f1


    
def epoch_time(start_time, end_time):
    '''
    Helper function to record epoch time.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
    

if __name__ == '__main__':

    train_words, train_tags, ud_train_tags = parse_ud_data(args.train_data_path)
    dev_words, dev_tags, ud_dev_tags = parse_ud_data(args.valid_data_path)

    ud_train_df = pd.DataFrame({'words':train_words, 'tags':train_tags}) 
    ud_dev_df = pd.DataFrame({'words':dev_words, 'tags':dev_tags})

    ud_tag2idx, ud_idx2tag = create_tag2idx(ud_train_tags)

    device = torch.device('cuda')
    base_model_type = args.base_model_type
    print("Model type: ", base_model_type)
    print("Language : ", args.language)
    if base_model_type == 'distilbert':
        if args.language == 'hi':
            tokenizer = AutoTokenizer.from_pretrained('hi-lm-distilbert/')
            base_model = AutoModel.from_pretrained('hi-lm-distilbert/').to(device)
        else:
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
            base_model = AutoModel.from_pretrained('distilbert-base-multilingual-cased').to(device) 
    elif base_model_type == 'bert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        base_model = AutoModel.from_pretrained('bert-base-multilingual-cased')
    
    train_dataset = POSDataset(tokenizer, ud_train_df, args.batch_size, ud_tag2idx)
    valid_dataset = POSDataset(tokenizer, ud_dev_df, args.batch_size, ud_tag2idx)

    if args.freeze == True:
        print("Freezing the base model")
        for param in base_model.parameters():
            param.requires_grad = False
    else:
        print("Finetuning")

    model = POS(len(ud_tag2idx), base_model, args.base_model_type, args.freeze, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs = args.epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        start_time = time.time()
        
        train_loss, train_acc = train(model, optimizer, train_dataset)
        valid_loss, valid_acc, valid_f1 = validate(model, valid_dataset, ud_idx2tag)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f"Epoch train loss : {train_loss}| Time: {epoch_mins}m {epoch_secs}s")
        print(f"Epoch valid loss: {valid_loss}")
        print(f"Epoch train accuracy: {train_acc}")
        print(f"Epoch valid accuracy: {valid_acc}")
        print(f"Epoch F1 score: {valid_f1}")
        print("====================================================================================")