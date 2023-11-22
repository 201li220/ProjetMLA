# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:13:10 2023

@author: Ugo Laziamond
"""
from transformers import BertForSequenceClassification, BertTokenizer,BertModel,BertConfig, AutoTokenizer
from datasets import load_dataset

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class Fine_Tuning:
    def __init__(self,model,tokenizer,dataset):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
    
    def prepared_data(self,batch_size=16):
        def tokenize_function(features,tokenizer=self.tokenizer):
            return tokenizer(features['sentence'],padding='max_length',truncation = True)

        tokenized_datasets = self.dataset.map(tokenize_function,batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(['sentence'])
        tokenized_datasets = tokenized_datasets.remove_columns(['idx'])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        self.train_dataset = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=batch_size)
        self.val_dataset = DataLoader(tokenized_datasets['validation'], shuffle=True, batch_size=batch_size)
        self.test_dataset = DataLoader(tokenized_datasets['test'], shuffle=True, batch_size=batch_size)
    
    def initialisation(self,optimizer,epoch=100,using_gpu=False):
        self.epoch = epoch
        self.optimizer = optimizer
        
        if using_gpu:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("gpu")
            self.model.to(device)
        
        self.train_losses=[]
        self.val_losses=[]
        self.val_accuracy =[]
        self.train_accuracy =[]
        self.test_accuracy = []
        
    def training(self,factor=1):
        bar_progress = tqdm(range(self.epoch))
        for i in range(self.epoch):
            self.model.train()
            
            l = []
            i=0
            for batch in self.train_dataset:
                i+=1
                print(i)
                self.optimizer.zero_grad()
                
                outputs = self.model(**batch)
                loss = outputs.loss
                l.append(loss.item())
                loss.backward()
                self.optimizer.step()
            
            l = np.array(l)
            self.train_losses.append(l.mean())
            
            l = []
            for batch in self.val_dataset:
                outputs = self.model(**batch)
                loss = outputs.loss
                l.append(loss.item())
            
            l = np.array(l)
            self.val_losses.append(l.mean())
            
            if i%factor ==0:
                total = 0
                correct = 0
                for batch in self.train_dataset:
                    with torch.no_grad():
                        outputs = self.model(**batch)
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        total += batch["label"].size(0)
                        correct += (predictions==batch["label"]).sum().item()
                self.train_accuracy.append(correct/total)
                
                total = 0
                correct = 0
                for batch in self.val_dataset:
                    with torch.no_grad():
                        outputs = self.model(**batch)
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        total += batch["label"].size(0)
                        correct += (predictions==batch["label"]).sum().item()
                self.val_accuracy.append(correct/total)
            bar_progress.update(1)
    
    def plot_accuracy(self):
        plt.plot(self.train_accuracy,label='Train Accuracy')
        plt.plot(self.val_accuracy,label='Validation Accuracy')
        
        plt.title('Graph of Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
        
    def plot_loss(self):
        plt.plot(self.train_losses,label='Train Loss')
        plt.plot(self.val_losses,label='Validation Loss')
        
        plt.title('Graph of Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        
    def evaluation(self):
        total = 0
        correct = 0
        i = 0
        for batch in self.test_dataset:
            with torch.no_grad():
                print(batch)
                i +=1
                print(i)
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                total += batch["label"].size(0)
                correct += (predictions==batch["label"]).sum().item()
        self.test_accuracy.append(correct/total)
        
        print("The Accuracy on the test dataset :",self.test_accuracy)


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
dataset = load_dataset('glue', 'cola')

config = BertConfig()
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

optimizer = optim.Adam(model.parameters(),lr=0.001)

FineTuning = Fine_Tuning(model, tokenizer, dataset)

FineTuning.prepared_data(batch_size=256)

FineTuning.initialisation(optimizer,epoch=1)

FineTuning.training()

FineTuning.plot_accuracy()

FineTuning.evaluation()




