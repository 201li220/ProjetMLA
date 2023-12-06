from Class_Fine_Tuning_V1 import *

import os
import numpy as np
import json

from modified_transformers import BertForSequenceClassification,BertConfig,AutoTokenizer
from datasets import load_dataset
from torch.optim import AdamW
import torch
torch.cuda.empty_cache()


dataset_cola = [load_dataset('glue', 'cola'),2,'dataset_cola']
dataset_sst2 = [load_dataset('glue', 'sst2'),2,'dataset_sst2']
dataset_mrpc = [load_dataset('glue', 'stsb'),5,'dataset_mrpc']
dataset_qqp = [load_dataset('glue', 'mnli'),3,'dataset_qqp']

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


DATASET = [dataset_cola,dataset_sst2,dataset_mrpc,dataset_qqp]

config = BertConfig(hidden_size = 512,
                max_position_embeddings=512,
                num_attention_heads=8,
                num_hidden_layers = 4)

model = BertForSequenceClassification(config)
for dataset in DATASET:
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    FineTuning = Fine_Tuning(model, tokenizer, dataset[0])
    FineTuning.prepared_data(batch_size=32)
    FineTuning.initialisation(optimizer,epoch=1)
    
    FineTuning.evaluation()
    FineTuning.train_accuracy.append(FineTuning.test_accuracy[0])
    FineTuning.val_accuracy.append(FineTuning.test_accuracy[0])
    
    FineTuning.training()

    FineTuning.evaluation()

    data = [np.array(FineTuning.train_accuracy),
            np.array(FineTuning.val_accuracy),
            np.array(FineTuning.test_accuracy),
            np.array(FineTuning.train_losses),
            np.array(FineTuning.val_losses)
           ]
    file = '/admin/ProjetMLA/Code_Ugo/Resultats/Run_2'
    name_file = dataset[2]+'_data'
    path = os.path.join(file,name_file)
    np.save(path,np.array(data))
        
    name_file = dataset[2]+'_model.safetensors'
    path = os.path.join('/admin/ProjetMLA/Code_Ugo/Saved_Models/Ugo_models/Run_2',name_file)
    model.save_pretrained(path)
    
    torch.cuda.empty_cache()
    
    config.num_labels = dataset[1]
    model = BertForSequenceClassification(config)
