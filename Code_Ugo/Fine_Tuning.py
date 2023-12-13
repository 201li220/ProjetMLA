from Class_Fine_Tuning_V1 import *

import os
import numpy as np
import json

from transformers import BertForSequenceClassification,BertConfig,AutoTokenizer
from datasets import load_dataset
from torch.optim import AdamW
import torch
torch.cuda.empty_cache()


learning_rate = [1e-5, 3e-5, 1e-4, 3e-3]
epoch = [20]


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

config = BertConfig(hidden_size = 1024,
                    max_position_embeddings=512,
                    num_attention_heads=16,
                    num_hidden_layers = 24)

model = BertForSequenceClassification(config)
for lr in learning_rate:
    for epochs in epoch:
        optimizer = AdamW(model.parameters(), lr=lr)
        
        FineTuning = Fine_Tuning(model, tokenizer, dataset_sst2)
        FineTuning.load_data(task = 'sst2',batch_size=8)
        FineTuning.initialisation(optimizer,epoch=epochs)
        
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
        name_file = 'learning_rate_{}_epoch_{}'.format(lr,epochs)+'_data'
        path = os.path.join(file,name_file)
        np.save(path,data)
            
        name_file = 'learning_rate_{}_epoch_{}'.format(lr,epochs)+'_model.safetensors'
        path = os.path.join('/admin/ProjetMLA/Code_Ugo/Saved_Models/Ugo_models/Run_2',name_file)
        model.save_pretrained(path)
        
        torch.cuda.empty_cache()
        
        model = BertForSequenceClassification(config)
