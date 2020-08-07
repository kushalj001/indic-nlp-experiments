# indic-nlp-experiments


## TODO
- [ ] Update README with latest commands. The scripts after today's changes won't work.
- [x] Test on linux by installing the requirements in a virtual env. The above scripts are tested on Windows with python 3.6.3
- [ ] Run experiments for Bengali DistilBERT (Kushal).
- [ ] Run finetuning experiments for all the models on GPU.
- [x] Include hyperparameter search using Weights and Biases.
- [ ] Run sweeps and update results.
- [ ] Use huggingface's squad script for Hindi.



Notebooks are just added for convenience.
The scripts currently support 3 models:
* Hindi DistilBERT (Panini)
* Multilingual DistilBERT (Huggingface)
* Multilingual BERT (Huggingface)

The data folder contains data for POS-Tagging and Classification for Hindi Language.
The scripts can be run from command line as follows.

### POS-Tagging multilingual DistilBERT  
```
python run_pos.py \ 
--train_data_path data/hi_hdtb-ud-train.conllu \
--valid_data_path data/hi_hdtb-ud-dev.conllu \
--base_model_type distilbert \
--language multilingual
```


### POS-Tagging Hindi DistilBERT (Panini)  
```
python run_pos.py \ 
--train_data_path data/hi_hdtb-ud-train.conllu \
--valid_data_path data/hi_hdtb-ud-dev.conllu \
--base_model_type distilbert \
--language hi
```

### POS-Tagging Hindi BERT with other options

```
python run_pos.py \ 
--train_data_path data/hi_hdtb-ud-train.conllu \
--valid_data_path data/hi_hdtb-ud-dev.conllu \
--base_model_type distilbert \
--language multilingual \
--freeze True \      # Finetunes the whole model if False. Choose a lower learning rate accordingly. Default:True
--batch_size 32 \
--epochs 5 \
--lr 0.001 \
```

### Classification DistilBERT Hindi (Panini)
The API for classification scripts is similar to POS tagging. 
```
python run_classification.py \ 
--train_data_path data/hindi-train.csv \
--valid_data_path data/hindi-test.csv \
--base_model_type distilbert \
--language hi
```


