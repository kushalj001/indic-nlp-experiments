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
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type=str, required=True, metavar='', help='Path to training file')
parser.add_argument('--valid_data_path', type=str, required=True, metavar='', help='Path to validation file')
parser.add_argument('--test_data_path', type=str, help='Path to test file.', default='None')
parser.add_argument('--base_model_type', type=str, metavar='', required=True, help='Type of transformer model. Currently supports BERT and DistilBERT.')
parser.add_argument('--language', type=str, required=True, help="Multingual DistilBERT or Hindi distilBERT")
parser.add_argument('--cuda_device', type=int, required=True, help='GPU ID in a multi-GPU setup.')
parser.add_argument('--head_type', type=str, help="Head to train on top of frozen model. Current choices: linear, multilinear, lstm, bilstm, transformer.", default='linear')
parser.add_argument('--output_strategy', type=str, help='Operation to be performed on the last hidden states before classification.', default='pool')
parser.add_argument('--dropout', type=float, help='Dropout between various layers.', default=0.2)
parser.add_argument('--freeze', type=eval, metavar='', help='Freeze the base model if True, finetune if False', default='True')
parser.add_argument('--batch_size', type=int, metavar='', default=32)
parser.add_argument('--epochs', type=int, metavar='', help='Epochs to train', default=16)
parser.add_argument('--lr', type=float, metavar='', help='Learning rate', default=3e-4)
parser.add_argument('--num_lstm_layers', type=int, help='Number of LSTM layers in head if head type is lstm or bilstm.', default=2)
parser.add_argument('--adam_epsilon', type=float, metavar='', help='Adam epsilon', default=1e-6)
parser.add_argument('--weight_decay', type=float, metavar='', help='Adam weight decay', default=0.01)
parser.add_argument('--warmup_proportion', type=float, metavar='', help='Proportion of training to perform linear learning rate warmup', default=0.1)
parser.add_argument('--multilingual', type=eval, default='False')
args = parser.parse_args()


def process_dataframe(path, language):
    '''
    Gets the data from path, converts labels into numerical features and 
    returns a dataframe for the data.
    '''
    if language == 'hi':
        df = pd.read_csv(path, sep='\t', encoding='utf-8', header=None)
    elif language == 'bn' or language == 'te':
        df = pd.read_csv(path, encoding='utf-8', header=None)

    df.columns = ['label_text', 'text']
    df.label_text = pd.Categorical(df.label_text)
    df['label'] = df.label_text.cat.codes
    print(f"Number of examples: {len(df)}")
    return df


class TextClassificationDataset:
    '''
    Class to convert raw indic languages dataset into features that can be 
    fed to our model. Tested for hi and bn.
    1. Divides the data into batches.
    2. Tokenizes the text and returns input_ids, token_type_ids and masks.
    3. Acts as a dataloader itself.
    '''
    def __init__(self, data, tokenizer, batch_size):
        
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        data = [data[i: i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        self.max_length = 512 if args.base_model_type != 'xlmroberta' else 510

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        
        
        for batch in self.data:
            
            batch = batch.dropna()
            texts = list(batch.text)
            labels = list(batch.label)
            
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
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
    
    def __init__(self, base_model, base_model_type, head_type, output_strategy, freeze, model_dim, num_lstm_layers, classifier_dim, num_labels, dropout):
        
        super().__init__()
        
        self.freeze = freeze
        self.base_model_type = base_model_type
        self.head_type = head_type
        self.base_model = base_model
        self.output_strategy = output_strategy
        
        self.lstm = nn.LSTM(input_size=model_dim, hidden_size=model_dim, 
                            num_layers=num_lstm_layers, batch_first=True, dropout=dropout)

        self.bilstm = nn.LSTM(input_size=model_dim, hidden_size=model_dim, num_layers=num_lstm_layers,
                              bidirectional=True, batch_first=True, dropout=dropout)

        self.linear_head = nn.Linear(in_features=model_dim, out_features=classifier_dim)
        
        self.multilinear_head = nn.Sequential(nn.Linear(in_features=model_dim, out_features=256),
                                              nn.Dropout(dropout),
                                              nn.Linear(in_features=256, out_features=classifier_dim),
                                              nn.Dropout(dropout)
                                             )
        
        
        self.lstm_head = nn.Sequential(
                                       nn.Linear(in_features=model_dim, out_features=classifier_dim),
                                       nn.Dropout(dropout)
                                      )
        
        self.bilstm_head = nn.Sequential(
                                         nn.Linear(in_features=model_dim*2, out_features=classifier_dim),
                                         nn.Dropout(dropout)
                                        )
        
        
        self.transformer_head = nn.Sequential(nn.TransformerEncoderLayer(d_model=model_dim, nhead=8, 
                                                                         dim_feedforward=3072),
                                              nn.Linear(in_features=model_dim, out_features=classifier_dim),
                                              nn.Dropout(dropout)
                                             )
        
        self.classifier = nn.Linear(in_features=classifier_dim, out_features=num_labels)
        
        
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        

        if self.freeze == True:

            with torch.no_grad():
        
                if self.base_model_type == 'distilbert':
                    sequence_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)[0]
                
                else:
                    sequence_output, pooled_output = self.base_model(input_ids=input_ids, 
                                                                        attention_mask=attention_mask, 
                                                                        token_type_ids=token_type_ids)
                
                
            sequence_output = F.dropout(sequence_output, p=0.1)

            # pooling is the same as filtering CLS states from rest of the hidden states.
            if self.head_type == 'linear':
                if self.output_strategy == 'mean':
                    output = sequence_output.mean(dim=1)
                elif self.output_strategy == 'pool':
                    output = sequence_output[:,0,:]

                head_out = self.linear_head(output)
            
            elif self.head_type == 'multilinear':
                if self.output_strategy == 'mean':
                    output = sequence_output.mean(dim=1)
                elif self.output_strategy == 'pool':
                    output = sequence_output[:,0,:]
                head_out = self.multilinear_head(output)

            elif self.head_type == 'lstm':
                lstm_out, _ = self.lstm(sequence_output)
                head_out = self.lstm_head(lstm_out)
                head_out = head_out.mean(dim=1)
            
            elif self.head_type == 'bilstm':
                bilstm_out, _  = self.bilstm(sequence_output)
                head_out = self.bilstm_head(bilstm_out)
                head_out = head_out.mean(dim=1)
            
            elif self.head_type == 'transformer':
                head_out = self.transformer_head(sequence_output)
                head_out = head_out.mean(dim=1)
            
            out = self.classifier(head_out)
        
        else:
            
            if  self.base_model_type == 'distilbert':
                sequence_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)[0]
            else:
                sequence_output, pooled_output = self.base_model(input_ids=input_ids, 
                                                                    attention_mask=attention_mask, 
                                                                    token_type_ids=token_type_ids)
            
            # sequence_output = [batch_size, seq_len, 768]
            sequence_output = F.dropout(sequence_output, p=0.2)

            mean_output = sequence_output.mean(dim=1)
            # [bs, 768]
        
            out = self.classifier(self.linear_head(mean_output))
            # out = [bs, 14]
        
        return out

def init_weights(m):
    if isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.xavier_uniform_(param.data)
            else:
                torch.nn.init.zeros_(param.data)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)


def train(model, optimizer, scheduler, train_dataset):
    
    print("Starting Training")
    train_loss = 0.
    train_acc = 0.
    model.train()
    
    for bi, batch in enumerate(train_dataset):

        if bi % 100 == 0:
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
    train_df = process_dataframe(args.train_data_path, args.language)
    num_labels = train_df.label.nunique()
    valid_df = process_dataframe(args.valid_data_path, args.language)
    device = torch.device('cuda:'+str(args.cuda_device))
    print("Model type: ", args.base_model_type)
    print("Language: ", args.language)
    print("Head:", args.head_type)
    run = wandb.init(entity='kushalj', project='indic-nlp')
    run.save()
    print(wandb.run.dir)
    config = wandb.config
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.epochs = args.epochs
    config.weight_decay = args.weight_decay
    config.num_lstm_layers = args.num_lstm_layers
    config.head_type = args.head_type
    config.dropout = args.dropout
    config.output_strategy = args.output_strategy
    torch.manual_seed(42)
    
    if args.multilingual == True:
        base_model_name = args.base_model_type + '-base-multilingual-cased'
    else:
        base_model_name = args.language + '-lm-' + args.base_model_type

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name).to(device)
    print(f"Loaded model and tokenizer successfully: {base_model_name}")

    
    # if args.base_model_type == 'distilbert':
    #     if args.language == 'hi':
    #         tokenizer = AutoTokenizer.from_pretrained('hi-lm-distilbert/')
    #         base_model = AutoModel.from_pretrained('hi-lm-distilbert/').to(device)
    #     if args.language == 'bn':
    #         tokenizer = AutoTokenizer.from_pretrained('bn-lm-distilbert')
    #         base_model = AutoModel.from_pretrained('bn-lm-distilbert')
    #     elif args.language == 'multilingual':
    #         tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    #         base_model = AutoModel.from_pretrained('distilbert-base-multilingual-cased').to(device)
    
    # elif args.base_model_type == 'bert':
    #     if args.language == 'hi':
    #         tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    #         base_model = AutoModel.from_pretrained('bert-base-multilingual-cased').to(device)
    #     elif args.language == 'bn':
    #         tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    #         base_model = AutoModel.from_pretrained('bert-base-multilingual-cased').to(device)
    #     elif args.language == 'te':
    #         tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    #         base_model = AutoModel.from_pretrained('bert-base-multilingual-cased').to(device)
    #     elif args.language == 'multilingual':
    #         tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    #         base_model = AutoModel.from_pretrained('bert-base-multilingual-cased').to(device)


    
    

    train_dataset = TextClassificationDataset(train_df, tokenizer, args.batch_size)
    valid_dataset = TextClassificationDataset(valid_df, tokenizer, args.batch_size)
    
    model_dim = base_model.get_input_embeddings().embedding_dim

    model = TextClassifier(base_model=base_model, 
                           base_model_type=args.base_model_type, 
                           head_type=args.head_type,
                           output_strategy= args.output_strategy,
                           freeze=args.freeze,
                           model_dim=model_dim,
                           num_lstm_layers=args.num_lstm_layers,
                           classifier_dim=128,
                           num_labels=num_labels,
                           dropout=args.dropout).to(device)
    
    #model.apply(init_weights)

    if args.freeze == True:
        print("Freezing the base model")
        for param in base_model.parameters():
            param.requires_grad = False
        num_train_optimization_steps = len(train_dataset) * args.epochs
        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
        optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon, weight_decay=args.weight_decay)
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
        num_train_optimization_steps = len(train_dataset) * args.epochs
        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)



    # trainable_params = []
    # for name,param in model.named_parameters():
    #     if param.requires_grad == True:
    #         assert 'base_model' not in name
    #         trainable_params.append(name)
            
    #print(trainable_params)

    wandb.watch(model, log='all')
    epochs = args.epochs
    best_valid_loss = 1000
    patience = 0
    best_valid_acc = -1
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        start_time = time.time()
        
        train_loss, train_acc = train(model, optimizer, scheduler, train_dataset)
        valid_loss, valid_acc = validate(model, valid_dataset)
        
        end_time = time.time()
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
        if valid_loss < best_valid_loss - 0.001:
            best_valid_loss = valid_loss
            patience = 0
        elif patience == 3:
            print("Early stopping....")
            break
        else:
            patience += 1
    
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f"Epoch train loss : {train_loss}| Time: {epoch_mins}m {epoch_secs}s")
        print(f"Epoch valid loss: {valid_loss}")
        print(f"Epoch train accuracy: {train_acc}")
        print(f"Epoch valid accuracy: {valid_acc}")
        print("====================================================================================")

    summary = {'Valid Accuracy': best_valid_acc, 'Valid Loss': best_valid_loss}
    wandb.log(summary)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'valid_loss': valid_loss,
                'valid_accuracy': valid_acc
            }, os.path.join(wandb.run.dir, 'acc-{}-epoch-{}.pth'.format(round(valid_acc,3), epoch)))

    if args.test_data_path != 'None':
        print("Running evaluation on test data...")
        test_df = process_dataframe(args.test_data_path, args.language)
        test_dataset = TextClassificationDataset(test_df, tokenizer, args.batch_size)
        test_loss, test_acc = validate(model, test_dataset)
        wandb.log({"Test Accuracy": test_acc, "Test Loss": test_loss})
        print(f"Test Accuracy: {test_acc}")
        print(f"Test Loss: {test_loss}")


    