"""
This script fine-tunes a BERT model on various GLUE tasks, including 'cola,' 'sst2,' 'mrpc,' and 'qqp.' 
The script initializes a BERT model with custom configurations and checks whether the correct parameters of the model's layers are frozen. 
It iterates through each GLUE task, loading the corresponding dataset and fine-tuning the model using the AdamW optimizer. 
The script evaluates the model's initial and final performances, appends accuracy metrics, and saves training, validation, and test metrics to NumPy files. 
The trained model parameters are also saved separately for each task. GPU memory is cleared between tasks, 
and the script provides insights into the frozen status of model layers.
"""

# Import necessary classes and modules
from Class_Fine_Tuning_V1 import *
import os
import numpy as np
import json
import sys
sys.path.append('/admin/ProjetMLA/Code_Nicolas/modified_transformers')
from modified_transformers import BertForSequenceClassification, BertConfig, AutoTokenizer
from datasets import load_dataset
from torch.optim import AdamW
import torch
torch.cuda.empty_cache()

# Define GLUE tasks
GLUE_TASK = ['cola', 'sst2', 'mrpc', 'qqp']

# Initialize tokenizer with pre-trained "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Configure BERT model with custom settings using BertConfig
config = BertConfig(
    hidden_size=768,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=12,
    adapter=True,
    adapter_size=64,
    frozen_mode=True
)

# Create a BERT model for sequence classification
model = BertForSequenceClassification(config)

# Check if the correct paramaters of the model's layers are frozen
for name, param in model.named_parameters():
    print(f'{name}: requires_grad={param.requires_grad}')

# Loop through GLUE tasks and fine-tune the BERT model for each task
for task in GLUE_TASK:
    # Initialize AdamW optimizer for model parameters with a learning rate of 3e-5
    optimizer = AdamW(model.parameters(), lr=3e-5)
    
    # Create an instance of the Fine_Tuning class
    FineTuning = Fine_Tuning(model, tokenizer)
    
    # Load data for the specified GLUE task with a batch size of 32
    FineTuning.load_data(task=task, batch_size=32)
    
    # Initialize fine-tuning settings, including optimizer and number of epochs
    FineTuning.initialisation(optimizer, epoch=15)
    
    # Evaluate the initial performance on the test dataset
    FineTuning.evaluation()
    
    # Append initial test accuracy to training and validation accuracy lists
    FineTuning.train_accuracy.append(FineTuning.test_accuracy[0])
    FineTuning.val_accuracy.append(FineTuning.test_accuracy[0])
    
    # Train the model using the fine-tuning process
    FineTuning.training()
    
    # Evaluate the model after training
    FineTuning.evaluation()
    
    # Save training, validation, and test metrics to a NumPy file
    data = [
        np.array(FineTuning.train_accuracy),
        np.array(FineTuning.val_accuracy),
        np.array(FineTuning.test_accuracy),
        np.array(FineTuning.train_losses),
        np.array(FineTuning.val_losses)
    ]
    file = '/admin/ProjetMLA/Code_Nicolas/Resultats/Run_stsb'
    name_file = task+'_data'
    path = os.path.join(file, name_file)
    np.save(path, data)
    
    # Save the model's parameters to a file
    name_file = task+'_model.safetensors'
    path = os.path.join('/admin/ProjetMLA/Code_Nicolas/Saved_Models/Run_stsb', name_file)
    model.save_pretrained(path)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Create a new instance of the BERT model for the next GLUE task
    model = BertForSequenceClassification(config)
    path = os.path.join('/admin/ProjetMLA/Code_Nicolas/Saved_Models/Run_stsb',name_file)
    model.save_pretrained(path)
    
    torch.cuda.empty_cache()
    
    model = BertForSequenceClassification(config)
