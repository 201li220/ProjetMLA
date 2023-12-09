# -*- coding: utf-8 -*-
"""
@author: Yu Jihan
"""

import numpy as np
import torch
from matplotlib.pyplot as plt
from tqdm.auto import tqdm

# HuggingFace ecosystem
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, get_scheduler
from datasets import load_dataset, load_metric

# Folder class
import Finetune

# check for GPU device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device available:', device)

# Define Hyperparameters and Database
GLUE_tasks = ['cola', 'sst2', 'mrpc', 'stsb']
task = 'cola'
checkpoint = 'bert-base-uncased'
batch_size = 32
learning_rate = 3e-5
epoch = 5

# Execute model finetuning
model = Finetune(checkpoint, task, batch_size, learning_rate, epoch)




