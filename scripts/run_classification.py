import transformers
import tokenizers
import torch
from torch import nn, tensor
import numpy as np
import pandas as pd
import typing, os, string, gc, time
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import argparse
from torch import *
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type=str, required=True, metavar='', help='Path to training file')
parser.add_argument('--valid_data_path', type=str, required=True, metavar='', help='Path to validation file')
parser.add_argument('--base_model_type', type=str, metavar='', required=True, help='Type of transformer model. Currently supports BERT and DistilBERT.')
parser.add_argument('--language', type=str, required=True, help="Multingual DistilBERT or Hindi distilBERT")
parser.add_argument('--head_type', type=str, help="Head to train on top of frozen model. Current choices: linear, multilinear, lstm, bilstm, transformer.", default='linear')
parser.add_argument('--freeze', type=eval, metavar='', help='Freeze the base model if True, finetune if False', default='True')
parser.add_argument('--batch_size', type=int, metavar='', default=32)
parser.add_argument('--epochs', type=int, metavar='', help='Epochs to train', default=5)
parser.add_argument('--lr', type=float, metavar='', help='Learning rate', default=3e-4)
parser.add_argument('--lstm_layers', type=int, help='Number of LSTM layers in head if head type is lstm or bilstm.', default=1)
parser.add_argument('--adam_epsilon', type=float, metavar='', help='Adam epsilon', default=1e-6)
parser.add_argument('--weight_decay', type=float, metavar='', help='Adam weight decay', default=0.01)
parser.add_argument('--warmup_proportion', type=float, metavar='', help='Proportion of training to perform linear learning rate warmup', default=0.1)

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
    
    def __init__(self, base_model, base_model_type, head_type, freeze, model_dim, num_lstm_layers, classifier_dim, num_labels):
        
        super().__init__()
        
        self.freeze = freeze
        self.base_model_type = base_model_type
        self.head_type = head_type
        self.base_model = base_model

        self.linear_head = nn.Linear(in_features=model_dim, out_features=classifier_dim)
        
        self.multilinear_head = nn.Sequential(nn.Linear(in_features=model_dim, out_features=256),
                                              nn.Dropout(0.2),
                                              nn.Linear(in_features=256, out_features=classifier_dim),
                                              nn.Dropout(0.2)
                                             )
        
        
        self.lstm_head = nn.Sequential(nn.LSTM(input_size=model_dim, hidden_size=model_dim, 
                                               num_layers=num_lstm_layers, batch_first=True),
                                       nn.Dropout(0.2),
                                       nn.Linear(in_features=model_dim, out_features=classifier_dim),
                                       nn.Dropout(0.2)
                                      )
        
        self.bilstm_head = nn.Sequential(nn.LSTM(input_size=model_dim, hidden_size=model_dim, num_layers=num_lstm_layers,
                                                 bidirectional=True, batch_first=True, dropout=0.2),
                                         nn.Dropout(0.2),
                                         nn.Linear(in_features=model_dim*2, out_features=classifer_dim),
                                         nn.Dropout(0.2)
                                        )
        
        
        self.transformer_head = nn.Sequential(nn.TransformerEncoderLayer(d_model=model_dim, nhead=8, 
                                                                         dim_feedforward=3072),
                                              nn.Linear(in_features=model_dim, out_features=classifier_dim),
                                              nn.Dropout(0.2)
                                             )
        
        self.classifier = nn.Linear(in_features=classifier_dim, out_features=num_labels)
        
        
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        

        if self.freeze == True:

            with torch.no_grad():
        
                if self.base_model_type == 'bert':
                    sequence_output, pooled_output = self.base_model(input_ids=input_ids, 
                                                                        attention_mask=attention_mask, 
                                                                        token_type_ids=token_type_ids)
                
                elif self.base_model_type == 'distilbert':
                    sequence_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)[0]

            sequence_output = F.dropout(sequence_output, p=0.2)

            if self.head_type == 'linear':
                mean_out = sequence_output.mean(dim=1)
                head_out = self.linear_head(mean_out)
            
            elif self.head_type == 'multilinear':
                mean_out = sequence_output.mean(dim=1)
                head_out = self.multilinear_head(mean_out)

            elif self.head_type == 'lstm':
                head_out = self.lstm_head(sequence_output)
                head_out = head_out.mean(dim=1)
            
            elif self.head_type == 'bilstm':
                head_out = self.bilstm_head(sequence_output)
                head_out = head_out.mean(dim=1)
            
            elif self.head_type == 'transformer':
                head_out = self.transformer_head(sequence_output)
                head_out = head_out.mean(dim=1)
            
            out = self.classifier(head_out)
        
        else:
            
            if self.base_model_type == 'bert':
                sequence_output, pooled_output = self.base_model(input_ids=input_ids, 
                                                                    attention_mask=attention_mask, 
                                                                    token_type_ids=token_type_ids)
            
            elif self.base_model_type == 'distilbert':
                sequence_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        
            # sequence_output = [batch_size, seq_len, 768]
            sequence_output = F.dropout(sequence_output, p=0.2)

            mean_output = sequence_output.mean(dim=1)
            # [bs, 768]
        
            out = self.classifier(self.linear_head(mean_output))
            # out = [bs, 14]
        
        return out


def train(model, optimizer, scheduler, train_dataset):
    
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
        scheduler.step()
        optimizer.zero_grad()
    
    wandb.log({
        "Train Accuracy": train_acc/len(train_dataset),
        "Train Loss": train_loss/len(train_dataset)
    })
        
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
    
    wandb.log({
        "Valid Accuracy": valid_acc/len(valid_dataset),
        "Valid Loss": valid_loss/len(valid_dataset)
    })    
           
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

    wandb.init(entity='kushalj', project='indic-nlp')
    config = wandb.config
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.epochs = args.epochs
    config.weight_decay = args.weight_decay
    config.adam_epsilon = args.adam_epsilon
    config.warmup_proportion = args.warmup_proportion
    config.num_lstm_layers = args.num_lstm_layers
    config.head_type = args.head_type

    torch.manual_seed(42)

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

    
    

    train_dataset = BBCHindiDataset(train_df, tokenizer, args.batch_size)
    valid_dataset = BBCHindiDataset(valid_df, tokenizer, args.batch_size)
    
    model_dim = base_model.get_input_embeddings().embedding_dim

    model = TextClassifier(base_model=base_model, 
                           base_model_type=args.base_model_type, 
                           head_type=args.head_type,
                           freeze=args.freeze,
                           model_dim=model_dim,
                           num_lstm_layers=args.num_lstm_layers,
                           classifier_dim=128,
                           num_labels=14).to(device)

    if args.freeze == 'True':
        print("Freezing the base model")
        for param in base_model.parameters():
            param.requires_grad = False
        num_train_optimization_steps = len(train_dataset) * config.epochs
        warmup_steps = int(config.warmup_proportion * num_train_optimization_steps)
        optimizer = AdamW(model.parameters(), lr=config.lr, eps=config.adam_epsilon, weight_decay=args.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

    else:
        print("Finetuning")
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias','LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = len(train_dataset) * config.epochs
        warmup_steps = int(config.warmup_proportion * num_train_optimization_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)



    
    wandb.watch(model, log='all')

    epochs = args.epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        start_time = time.time()
        
        train_loss, train_acc = train(model, optimizer, scheduler, train_dataset)
        valid_loss, valid_acc = validate(model, valid_dataset)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f"Epoch train loss : {train_loss}| Time: {epoch_mins}m {epoch_secs}s")
        print(f"Epoch valid loss: {valid_loss}")
        print(f"Epoch train accuracy: {train_acc}")
        print(f"Epoch valid accuracy: {valid_acc}")
        print("====================================================================================")


    