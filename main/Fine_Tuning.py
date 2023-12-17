from Class_Fine_Tuning_V1 import *

import os
import numpy as np
import json
import sys
sys.path.append('/admin/ProjetMLA/Code_Nicolas/modified_transformers')

from modified_transformers import BertForSequenceClassification,BertConfig,AutoTokenizer
from datasets import load_dataset
from torch.optim import AdamW
import torch
torch.cuda.empty_cache()


GLUE_TASK = ['cola','sst2','mrpc','qqp']

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


config = BertConfig(hidden_size = 768,
                    max_position_embeddings=512,
                    num_attention_heads=12,
                    num_hidden_layers = 12,
                    adapter = True,
                    adapter_size = 64,
                    frozen_mode = True
                   )

model = BertForSequenceClassification(config)
for name, param in model.named_parameters():
    print(f'{name}: requires_grad={param.requires_grad}')
for task in GLUE_TASK:
    
    optimizer = AdamW(model.parameters(), lr=3e-5)
    
    FineTuning = Fine_Tuning(model, tokenizer)
    FineTuning.load_data(task=task,batch_size=32)
    FineTuning.initialisation(optimizer,epoch=15)
    
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
    file = '/admin/ProjetMLA/Code_Nicolas/Resultats/Run_stsb'
    name_file = task+'_data'
    path = os.path.join(file,name_file)
    np.save(path,data)
        
    name_file = task+'_model.safetensors'
    path = os.path.join('/admin/ProjetMLA/Code_Nicolas/Saved_Models/Run_stsb',name_file)
    model.save_pretrained(path)
    
    torch.cuda.empty_cache()
    
    model = BertForSequenceClassification(config)