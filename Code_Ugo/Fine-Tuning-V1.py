# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:13:10 2023

@author: Ugo Laziamond
"""
from transformers import BertForSequenceClassification, BertTokenizer,BertModel,BertConfig
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import evaluate

from tqdm.auto import tqdm


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataset = load_dataset('glue', 'cola')

def tokenize_function(features,tokenizer=tokenizer):
    return tokenizer(features['sentence'],padding='max_length',truncation = True)

tokenized_datasets = dataset.map(tokenize_function,batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['sentence'])
tokenized_datasets = tokenized_datasets.remove_columns(['idx'])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataset = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=16)
val_dataset = DataLoader(tokenized_datasets['validation'], shuffle=True, batch_size=16)
test_dataset = DataLoader(tokenized_datasets['test'], shuffle=True, batch_size=16)

config = BertConfig()
model = BertForSequenceClassification(config) 

optimizer = optim.Adam(model.parameters(),lr=0.001)
metric=evaluate.load('accuracy')


epoch = 10
bar_progress = tqdm(range(epoch))

losses=[]
val_accuracy =[]
train_accuracy =[]
for _ in range(epoch):
    model.train()
    l = []
    for batch in train_dataset:
        optimizer.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss
        l.append(loss.item())
        loss.backward()
        optimizer.step()
    losses.append(l.mean())
    print(losses)
    """
    model.eval()
    for batch in train_dataset:
        with torch.no_grad():
            outputs = model(**batch)
        
        predictions = torch.argmax(outputs.logits,dim=-1)
        metric.add_batch(predictions=predictions, references = batch["label"])
    accuracy = metric.compute()
    train_accuracy.append(accuracy)
    for batch in val_dataset:
        with torch.no_grad():
            outputs = model(**batch)
        
        predictions = torch.argmax(outputs.logits,dim=-1)
        metric.add_batch(predictions=predictions, references = batch["label"])
    accuracy = metric.compute()
    val_accuracy.append(accuracy)"""
    bar_progress.update(1)



