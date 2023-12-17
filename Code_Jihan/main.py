# -*- coding: utf-8 -*-
"""
@author: Yu Jihan
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from datasets import load_dataset, load_metric

# Folder class
from Finetune import Fine_Tuning

# Define Hyperparameters and Database
GLUE_tasks = ['cola', 'sst2', 'mrpc', 'stsb']
task = 'cola'
checkpoint = 'bert-base-uncased'
batch_size = 32
learning_rate = 3e-5
epoch = 5

# Execute model finetuning
model = Fine_Tuning(checkpoint, task, batch_size, learning_rate, epoch)
print(model)



