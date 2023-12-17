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


GLUE_TASK = ['cola']
"""
Layers = [
    'classifier.weight', 'classifier.bias',  # Trained_Layers_1: Adds Classifier Weights and Bias
    'pooler.dense.weight', 'pooler.dense.bias',  # Trained_Layers_2: Adds Pooler Weights and Bias layers
    'encoder.layer.11.output.LayerNorm.weight', 'encoder.layer.11.output.LayerNorm.bias',  # Trained_Layers_3: Adds Encoder norm layers
    'encoder.layer.11.output.dense.weight', 'encoder.layer.11.output.dense.bias',  # Trained_Layers_4: Adds Dense encoder Weights and Bias layers
    'encoder.layer.11.attention.output.LayerNorm.weight', 'encoder.layer.11.attention.output.LayerNorm.bias',  # Trained_Layers_5: Adds Attention norm layers
    'encoder.layer.11.attention.output.dense.weight', 'encoder.layer.11.attention.output.dense.bias',  # Trained_Layers_6: Adds Attention dense Weights and Bias
    # Additional layers for Trained_Layers_7 and Trained_Layers_8
    'encoder.layer.10.output.LayerNorm.weight', 'encoder.layer.10.output.LayerNorm.bias',  # Trained_Layers_7: Adds Encoder Layer 1 norm layers
    'encoder.layer.10.output.dense.weight', 'encoder.layer.10.output.dense.bias',  # Trained_Layers_7: Adds Encoder Layer 1 dense Weights and Bias layers
    'encoder.layer.10.attention.output.LayerNorm.weight', 'encoder.layer.10.attention.output.LayerNorm.bias',  # Trained_Layers_7: Adds Encoder Layer 1 Attention norm layers
    'encoder.layer.10.attention.output.dense.weight', 'encoder.layer.10.attention.output.dense.bias',  # Trained_Layers_7: Adds Encoder Layer 1 Attention dense Weights and Bias
    'encoder.layer.9.output.LayerNorm.weight', 'encoder.layer.9.output.LayerNorm.bias',  # Trained_Layers_8: Adds Encoder Layer 2 norm layers
    'encoder.layer.9.output.dense.weight', 'encoder.layer.9.output.dense.bias',  # Trained_Layers_8: Adds Encoder Layer 2 dense Weights and Bias layers
    'encoder.layer.9.attention.output.LayerNorm.weight', 'encoder.layer.9.attention.output.LayerNorm.bias',  # Trained_Layers_8: Adds Encoder Layer 2 Attention norm layers
    'encoder.layer.9.attention.output.dense.weight', 'encoder.layer.9.attention.output.dense.bias',  # Trained_Layers_8: Adds Encoder Layer 2 Attention dense Weights and Bias
]

for i in range(len(Layers)):
    if i > 1:
        Layers[i] = "bert."+Layers[i]
    print(Layers[i])
print("\n Layers : ")
"""

Layers = ['bert.embeddings.word_embeddings.weight', 'classifier.bias', 'classifier.weight', 'bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'bert.encoder.layer.11.output.LayerNorm.bias', 'bert.encoder.layer.11.output.LayerNorm.weight', 'bert.encoder.layer.11.output.dense.bias', 'bert.encoder.layer.11.output.dense.weight', 'bert.encoder.layer.11.intermediate.dense.bias', 'bert.encoder.layer.11.intermediate.dense.weight', 'bert.encoder.layer.11.attention.output.LayerNorm.bias', 'bert.encoder.layer.11.attention.output.LayerNorm.weight', 'bert.encoder.layer.11.attention.output.dense.bias', 'bert.encoder.layer.11.attention.output.dense.weight', 'bert.encoder.layer.11.attention.self.value.bias', 'bert.encoder.layer.11.attention.self.value.weight', 'bert.encoder.layer.11.attention.self.key.bias', 'bert.encoder.layer.11.attention.self.key.weight', 'bert.encoder.layer.11.attention.self.query.bias', 'bert.encoder.layer.11.attention.self.query.weight', 'bert.encoder.layer.10.output.LayerNorm.bias', 'bert.encoder.layer.10.output.LayerNorm.weight', 'bert.encoder.layer.10.output.dense.bias', 'bert.encoder.layer.10.output.dense.weight', 'bert.encoder.layer.10.intermediate.dense.bias', 'bert.encoder.layer.10.intermediate.dense.weight', 'bert.encoder.layer.10.attention.output.LayerNorm.bias', 'bert.encoder.layer.10.attention.output.LayerNorm.weight', 'bert.encoder.layer.10.attention.output.dense.bias', 'bert.encoder.layer.10.attention.output.dense.weight', 'bert.encoder.layer.10.attention.self.value.bias', 'bert.encoder.layer.10.attention.self.value.weight', 'bert.encoder.layer.10.attention.self.key.bias', 'bert.encoder.layer.10.attention.self.key.weight', 'bert.encoder.layer.10.attention.self.query.bias', 'bert.encoder.layer.10.attention.self.query.weight', 'bert.encoder.layer.9.output.LayerNorm.bias', 'bert.encoder.layer.9.output.LayerNorm.weight', 'bert.encoder.layer.9.output.dense.bias', 'bert.encoder.layer.9.output.dense.weight', 'bert.encoder.layer.9.intermediate.dense.bias', 'bert.encoder.layer.9.intermediate.dense.weight', 'bert.encoder.layer.9.attention.output.LayerNorm.bias', 'bert.encoder.layer.9.attention.output.LayerNorm.weight', 'bert.encoder.layer.9.attention.output.dense.bias', 'bert.encoder.layer.9.attention.output.dense.weight', 'bert.encoder.layer.9.attention.self.value.bias', 'bert.encoder.layer.9.attention.self.value.weight', 'bert.encoder.layer.9.attention.self.key.bias', 'bert.encoder.layer.9.attention.self.key.weight', 'bert.encoder.layer.9.attention.self.query.bias', 'bert.encoder.layer.9.attention.self.query.weight', 'bert.encoder.layer.8.output.LayerNorm.bias', 'bert.encoder.layer.8.output.LayerNorm.weight', 'bert.encoder.layer.8.output.dense.bias', 'bert.encoder.layer.8.output.dense.weight', 'bert.encoder.layer.8.intermediate.dense.bias', 'bert.encoder.layer.8.intermediate.dense.weight', 'bert.encoder.layer.8.attention.output.LayerNorm.bias', 'bert.encoder.layer.8.attention.output.LayerNorm.weight', 'bert.encoder.layer.8.attention.output.dense.bias', 'bert.encoder.layer.8.attention.output.dense.weight', 'bert.encoder.layer.8.attention.self.value.bias', 'bert.encoder.layer.8.attention.self.value.weight', 'bert.encoder.layer.8.attention.self.key.bias', 'bert.encoder.layer.8.attention.self.key.weight', 'bert.encoder.layer.8.attention.self.query.bias', 'bert.encoder.layer.8.attention.self.query.weight', 'bert.encoder.layer.7.output.LayerNorm.bias', 'bert.encoder.layer.7.output.LayerNorm.weight', 'bert.encoder.layer.7.output.dense.bias', 'bert.encoder.layer.7.output.dense.weight', 'bert.encoder.layer.7.intermediate.dense.bias', 'bert.encoder.layer.7.intermediate.dense.weight', 'bert.encoder.layer.7.attention.output.LayerNorm.bias', 'bert.encoder.layer.7.attention.output.LayerNorm.weight', 'bert.encoder.layer.7.attention.output.dense.bias', 'bert.encoder.layer.7.attention.output.dense.weight', 'bert.encoder.layer.7.attention.self.value.bias', 'bert.encoder.layer.7.attention.self.value.weight', 'bert.encoder.layer.7.attention.self.key.bias', 'bert.encoder.layer.7.attention.self.key.weight', 'bert.encoder.layer.7.attention.self.query.bias', 'bert.encoder.layer.7.attention.self.query.weight', 'bert.encoder.layer.6.output.LayerNorm.bias', 'bert.encoder.layer.6.output.LayerNorm.weight', 'bert.encoder.layer.6.output.dense.bias', 'bert.encoder.layer.6.output.dense.weight', 'bert.encoder.layer.6.intermediate.dense.bias', 'bert.encoder.layer.6.intermediate.dense.weight', 'bert.encoder.layer.6.attention.output.LayerNorm.bias', 'bert.encoder.layer.6.attention.output.LayerNorm.weight', 'bert.encoder.layer.6.attention.output.dense.bias', 'bert.encoder.layer.6.attention.output.dense.weight', 'bert.encoder.layer.6.attention.self.value.bias', 'bert.encoder.layer.6.attention.self.value.weight', 'bert.encoder.layer.6.attention.self.key.bias', 'bert.encoder.layer.6.attention.self.key.weight', 'bert.encoder.layer.6.attention.self.query.bias', 'bert.encoder.layer.6.attention.self.query.weight', 'bert.encoder.layer.5.output.LayerNorm.bias', 'bert.encoder.layer.5.output.LayerNorm.weight', 'bert.encoder.layer.5.output.dense.bias', 'bert.encoder.layer.5.output.dense.weight', 'bert.encoder.layer.5.intermediate.dense.bias', 'bert.encoder.layer.5.intermediate.dense.weight', 'bert.encoder.layer.5.attention.output.LayerNorm.bias', 'bert.encoder.layer.5.attention.output.LayerNorm.weight', 'bert.encoder.layer.5.attention.output.dense.bias', 'bert.encoder.layer.5.attention.output.dense.weight', 'bert.encoder.layer.5.attention.self.value.bias', 'bert.encoder.layer.5.attention.self.value.weight', 'bert.encoder.layer.5.attention.self.key.bias', 'bert.encoder.layer.5.attention.self.key.weight', 'bert.encoder.layer.5.attention.self.query.bias', 'bert.encoder.layer.5.attention.self.query.weight', 'bert.encoder.layer.4.output.LayerNorm.bias', 'bert.encoder.layer.4.output.LayerNorm.weight', 'bert.encoder.layer.4.output.dense.bias', 'bert.encoder.layer.4.output.dense.weight', 'bert.encoder.layer.4.intermediate.dense.bias', 'bert.encoder.layer.4.intermediate.dense.weight', 'bert.encoder.layer.4.attention.output.LayerNorm.bias', 'bert.encoder.layer.4.attention.output.LayerNorm.weight', 'bert.encoder.layer.4.attention.output.dense.bias', 'bert.encoder.layer.4.attention.output.dense.weight', 'bert.encoder.layer.4.attention.self.value.bias', 'bert.encoder.layer.4.attention.self.value.weight', 'bert.encoder.layer.4.attention.self.key.bias', 'bert.encoder.layer.4.attention.self.key.weight', 'bert.encoder.layer.4.attention.self.query.bias', 'bert.encoder.layer.4.attention.self.query.weight', 'bert.encoder.layer.3.output.LayerNorm.bias', 'bert.encoder.layer.3.output.LayerNorm.weight', 'bert.encoder.layer.3.output.dense.bias', 'bert.encoder.layer.3.output.dense.weight', 'bert.encoder.layer.3.intermediate.dense.bias', 'bert.encoder.layer.3.intermediate.dense.weight', 'bert.encoder.layer.3.attention.output.LayerNorm.bias', 'bert.encoder.layer.3.attention.output.LayerNorm.weight', 'bert.encoder.layer.3.attention.output.dense.bias', 'bert.encoder.layer.3.attention.output.dense.weight', 'bert.encoder.layer.3.attention.self.value.bias', 'bert.encoder.layer.3.attention.self.value.weight', 'bert.encoder.layer.3.attention.self.key.bias', 'bert.encoder.layer.3.attention.self.key.weight', 'bert.encoder.layer.3.attention.self.query.bias', 'bert.encoder.layer.3.attention.self.query.weight', 'bert.encoder.layer.2.output.LayerNorm.bias', 'bert.encoder.layer.2.output.LayerNorm.weight', 'bert.encoder.layer.2.output.dense.bias', 'bert.encoder.layer.2.output.dense.weight', 'bert.encoder.layer.2.intermediate.dense.bias', 'bert.encoder.layer.2.intermediate.dense.weight', 'bert.encoder.layer.2.attention.output.LayerNorm.bias', 'bert.encoder.layer.2.attention.output.LayerNorm.weight', 'bert.encoder.layer.2.attention.output.dense.bias', 'bert.encoder.layer.2.attention.output.dense.weight', 'bert.encoder.layer.2.attention.self.value.bias', 'bert.encoder.layer.2.attention.self.value.weight', 'bert.encoder.layer.2.attention.self.key.bias', 'bert.encoder.layer.2.attention.self.key.weight', 'bert.encoder.layer.2.attention.self.query.bias', 'bert.encoder.layer.2.attention.self.query.weight', 'bert.encoder.layer.1.output.LayerNorm.bias', 'bert.encoder.layer.1.output.LayerNorm.weight', 'bert.encoder.layer.1.output.dense.bias', 'bert.encoder.layer.1.output.dense.weight', 'bert.encoder.layer.1.intermediate.dense.bias', 'bert.encoder.layer.1.intermediate.dense.weight', 'bert.encoder.layer.1.attention.output.LayerNorm.bias', 'bert.encoder.layer.1.attention.output.LayerNorm.weight', 'bert.encoder.layer.1.attention.output.dense.bias', 'bert.encoder.layer.1.attention.output.dense.weight', 'bert.encoder.layer.1.attention.self.value.bias', 'bert.encoder.layer.1.attention.self.value.weight', 'bert.encoder.layer.1.attention.self.key.bias', 'bert.encoder.layer.1.attention.self.key.weight', 'bert.encoder.layer.1.attention.self.query.bias', 'bert.encoder.layer.1.attention.self.query.weight', 'bert.encoder.layer.0.output.LayerNorm.bias', 'bert.encoder.layer.0.output.LayerNorm.weight', 'bert.encoder.layer.0.output.dense.bias', 'bert.encoder.layer.0.output.dense.weight', 'bert.encoder.layer.0.intermediate.dense.bias', 'bert.encoder.layer.0.intermediate.dense.weight', 'bert.encoder.layer.0.attention.output.LayerNorm.bias', 'bert.encoder.layer.0.attention.output.LayerNorm.weight', 'bert.encoder.layer.0.attention.output.dense.bias', 'bert.encoder.layer.0.attention.output.dense.weight', 'bert.encoder.layer.0.attention.self.value.bias', 'bert.encoder.layer.0.attention.self.value.weight', 'bert.encoder.layer.0.attention.self.key.bias', 'bert.encoder.layer.0.attention.self.key.weight', 'bert.encoder.layer.0.attention.self.query.bias', 'bert.encoder.layer.0.attention.self.query.weight', 'bert.embeddings.LayerNorm.bias', 'bert.embeddings.LayerNorm.weight', 'bert.embeddings.token_type_embeddings.weight', 'bert.embeddings.position_embeddings.weight']

Trained_Layers_1 = Layers[:3]  #Adds Classifier       Weights and Bias
Trained_Layers_2 = Layers[:5] #Adds Pooler           Weights and Bias layers
Trained_Layers_3 = Layers[:7] #Adds Encoder          norm layers
Trained_Layers_4 = Layers[:9] #Adds Dense encoder    Weights and Bias layers
Trained_Layers_5 = Layers[:11] #Adds Attention        norm layers
Trained_Layers_6 = Layers[:13] #Adds Attention dense  Weights and Bias
Trained_Layers_7 = Layers[:15+2*16] #Adds encoder 10 9
Trained_Layers_8 = Layers[:15+4*16] #Adds encoder 8 7 
Trained_Layers_9 = Layers[:15+6*16] #Adds encoder 6 5
Trained_Layers_10 = Layers[:15+8*16] #Adds encoder 4 3
Trained_Layers_11 = Layers[:15+10*16]#Adds encoder 2 1

Trained_Layers = [Trained_Layers_1, Trained_Layers_2, Trained_Layers_3,
                  Trained_Layers_4, Trained_Layers_5, Trained_Layers_6,
                 Trained_Layers_7,Trained_Layers_8,Trained_Layers_9,
                 Trained_Layers_10,Trained_Layers_11]
#layer 1 deja faite

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


for i in range(7,len(Trained_Layers)):
    print("Starting TopLayer "+str(i))
    config = BertConfig(hidden_size = 1024,
                        max_position_embeddings=512,
                        num_attention_heads=16,
                        num_hidden_layers = 24,
                        adapter = True,
                        adapter_size = 16,
                        frozen_mode = True
                       )
    
    model = BertForSequenceClassification(config)
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    num_trainable_param = 0
    for param_name in Trained_Layers[i]:
        param = dict(model.named_parameters())[param_name]
        param.requires_grad = True

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters for TopLayer  : {total_trainable_params}")
    
    for task in GLUE_TASK:
        print(f"\n Starting {task} during TopLayer "+str(i))
        optimizer = AdamW(model.parameters(), lr=3e-5)
        
        FineTuning = Fine_Tuning(model, tokenizer)
        FineTuning.load_data(task=task,batch_size=16)
        FineTuning.initialisation(optimizer,epoch=3)
        
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
        file = '/admin/ProjetMLA/Code_Ugo/Resultats/Run_TopLayer'+str(i)
        name_file = task+'_data'
        path = os.path.join(file,name_file)
        np.save(path,data)
            
        name_file = task+'_model.safetensors'
        path = os.path.join('/admin/ProjetMLA/Code_Ugo/Saved_Models/Ugo_models/Run_TopLayer'+str(i),name_file)
        model.save_pretrained(path)
        
        torch.cuda.empty_cache()
        
        model = BertForSequenceClassification(config)