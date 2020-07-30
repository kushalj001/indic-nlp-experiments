import transformers
import tokenizers
import torch
from torch import nn, tensor
import numpy as np
import pandas as pd
import typing, os, string, gc, time
import nltk
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW
import argparse
from torch import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type=str, required=True, metavar='', help='Path to training file')
parser.add_argument('--valid_data_path', type=str, required=True, metavar='', help='Path to validation file')
parser.add_argument('--base_model_type', type=str, metavar='', required=True, help='Type of transformer model. Currently supports BERT and DistilBERT.')
parser.add_argument('--language', type=str, required=True, help="Multingual DistilBERT or Hindi distilBERT")
parser.add_argument('--freeze', type=bool, metavar='', help='Freeze the base model if True, finetune if False', default=True)
parser.add_argument('--batch_size', type=int, metavar='', default=32)
parser.add_argument('--epochs', type=int, metavar='', help='Epochs to train', default=5)
parser.add_argument('--lr', type=float, metavar='', help='Learning rate', default=3e-4)
args = parser.parse_args()


def process_dataframe(path):
    '''
    Gets the data from path, converts labels into numerical features and 
    returns a dataframe for the data.
    '''

    df = pd.read_csv(path, sep='\t', encoding='utf-8', header=None)
    df.columns = ['label_text', 'text']
    df.label_text = pd.Categorical(df.label_text)
    df['label'] = df.label_text.cat.codes
    print(f"Number of examples: {len(df)}")
    return df


class BBCHindiDataset:
    '''
    Class to convert raw BBC Hindi dataset into features that can be 
    fed to our model. 
    1. Divides the data into batches.
    2. Tokenizes the text and returns input_ids, token_type_ids and masks.
    3. Acts as a dataloader itself.
    '''
    def __init__(self, data, tokenizer, batch_size):
        
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        data = [data[i: i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        
        
        for batch in self.data:
            
            batch = batch.dropna()
            texts = list(batch.text)
            labels = list(batch.label)
            
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            input_ids = encoded_input['input_ids']
            attention_mask = encoded_input['attention_mask']
            label = torch.tensor(labels, dtype=torch.long)
            token_type_ids = torch.zeros_like(input_ids)
            
            yield {
                'input_ids':input_ids,
                'attention_mask':attention_mask,
                'token_type_ids':token_type_ids,
                'label':label
            }
        


class TextClassifier(nn.Module):
    
    def __init__(self, base_model, base_model_type, freeze):
        
        super().__init__()
        
        self.freeze = freeze
        self.base_model_type = base_model_type
        self.base_model = base_model
        self.fc1 = nn.Linear(768, 100)
        self.fc2 = nn.Linear(100, 14)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        

        if self.freeze == True:

            with torch.no_grad():
        
                if self.base_model_type == 'bert':
                    sequence_output, pooled_output = self.base_model(input_ids=input_ids, 
                                                                        attention_mask=attention_mask, 
                                                                        token_type_ids=token_type_ids)
                
                elif self.base_model_type == 'distilbert':
                    sequence_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        
        else:
            
            if self.base_model_type == 'bert':
                sequence_output, pooled_output = self.base_model(input_ids=input_ids, 
                                                                    attention_mask=attention_mask, 
                                                                    token_type_ids=token_type_ids)
            
            elif self.base_model_type == 'distilbert':
                    sequence_output = self.base_model(input_ids=input_ids, attention_mask=input_mask)[0]
        
        # sequence_output = [batch_size, seq_len, 768]
        sequence_output = F.dropout(sequence_output, p=0.2)

        mean_output = sequence_output.mean(dim=1)
        # [bs, 768]
        
        out = self.fc2(self.fc1(mean_output))
        # out = [bs, 14]
        
        return out


def train(model, optimizer, train_dataset):
    
    print("Starting Training")
    train_loss = 0.
    train_acc = 0.
    model.train()
    
    for bi, batch in enumerate(train_dataset):

        if bi % 20 == 0:
            print(f"Starting batch: {bi}")

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        preds = model(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(preds, labels)
        
        train_loss += loss.item()
        train_acc += (torch.argmax(preds,dim=1)==labels).float().mean().item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return train_loss/len(train_dataset), train_acc/len(train_dataset)
    
    

def validate(model, valid_dataset):
    
    print("Starting validation")
    valid_loss = 0.
    valid_acc = 0.
    model.eval()
    
    for bi, batch in enumerate(valid_dataset):

        if bi % 20 == 0:
            print(f"Starting batch: {bi}")

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        with torch.no_grad():
            
            preds = model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(preds, labels)
        
            valid_loss += loss.item()
            valid_acc += (torch.argmax(preds,dim=1)==labels).float().mean().item()
        
        
    return valid_loss/len(valid_dataset), valid_acc/len(valid_dataset)
    

def epoch_time(start_time, end_time):
    '''
    Helper function to record epoch time.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    train_df = process_dataframe(args.train_data_path)
    valid_df = process_dataframe(args.valid_data_path)
    device = torch.device('cuda')
    print("Model type: ", args.base_model_type)
    print("Language: ", args.language)

    if args.base_model_type == 'distilbert':
        if args.language == 'hi':
            tokenizer = AutoTokenizer.from_pretrained('hi-lm-distilbert/')
            base_model = AutoModel.from_pretrained('hi-lm-distilbert/').to(device)
        elif args.language == 'multilingual':
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
            base_model = AutoModel.from_pretrained('distilbert-base-multilingual-cased').to(device)
    
    elif args.base_model_type == 'bert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        base_model = AutoModel.from_pretrained('bert-base-multilingual-cased').to(device)

    if args.freeze == True:
        print("Freezing the base model")
        for param in base_model.parameters():
            param.requires_grad = False
    else:
        print("Finetuning")

    train_dataset = BBCHindiDataset(train_df, tokenizer, args.batch_size)
    valid_dataset = BBCHindiDataset(valid_df, tokenizer, args.batch_size)
    model =TextClassifier(base_model, args.base_model_type, args.freeze).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    epochs = args.epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        start_time = time.time()
        
        train_loss, train_acc = train(model, optimizer, train_dataset)
        valid_loss, valid_acc = validate(model, valid_dataset)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f"Epoch train loss : {train_loss}| Time: {epoch_mins}m {epoch_secs}s")
        print(f"Epoch valid loss: {valid_loss}")
        print(f"Epoch train accuracy: {train_acc}")
        print(f"Epoch valid accuracy: {valid_acc}")
        print("====================================================================================")


    