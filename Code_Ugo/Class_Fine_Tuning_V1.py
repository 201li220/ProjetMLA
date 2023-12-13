"""
Created on Tue Nov 21 18:13:10 2023

@author: Ugo Laziamond
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from datasets import load_dataset
import evaluate

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

torch.cuda.empty_cache()


class Fine_Tuning:
    def __init__(self,model,tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    def load_data(self,task, dataset_size=5000, batch_size=16):

        self.task = task
        
        dataset = load_dataset('glue', self.task)

        task_to_keys = {"cola": ("sentence", None),
                            "mrpc": ("sentence1", "sentence2"),
                            "sst2": ("sentence", None),
                            "stsb": ("sentence1", "sentence2"),
                            "qqp": ("sentence1", "sentence2"),
                           }

        sentence1_key, sentence2_key = task_to_keys[self.task]
        
        def tokenize_function(features, tokenizer=self.tokenizer):
            if sentence2_key is None:
                return tokenizer(features[sentence1_key], padding='max_length',truncation=True,return_tensors='pt')
            return tokenizer(features[sentence1_key], features[sentence2_key], padding='max_length',truncation=True,return_tensors='pt')

        tokenized_datasets = dataset.map(tokenize_function,batched=True)
        
        if sentence2_key is None:
          tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence"])
        else:
          tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets = tokenized_datasets.with_format("torch")
        print(len(tokenized_datasets['train']))
        
        if dataset_size < len(tokenized_datasets['train']):
            reduced_train_dataset = tokenized_datasets['train'].select(range(dataset_size))
        else:
            reduced_train_dataset = tokenized_datasets['train']
        if dataset_size < len(tokenized_datasets['validation']):
            reduced_validation_dataset = tokenized_datasets['validation'].select(range(500))
        else:
            reduced_validation_dataset = tokenized_datasets['validation']
        if dataset_size < len(tokenized_datasets['test']):
            reduced_test_dataset = tokenized_datasets['test'].select(range(500))
        else:
            reduced_test_dataset = tokenized_datasets['test']
        
        A = DataLoader(reduced_train_dataset, shuffle=True, batch_size=batch_size)
        B = DataLoader(reduced_validation_dataset, shuffle=True, batch_size=batch_size)
        C = DataLoader(reduced_test_dataset, shuffle=True, batch_size=batch_size)

        self.train_dataset = [ {k: v.to(self.device) for k, v in batch.items()} for batch in A ]
        self.val_dataset = [ {k: v.to(self.device) for k, v in batch.items()} for batch in B]
        self.test_dataset = [ {k: v.to(self.device) for k, v in batch.items()} for batch in C]
    
    def initialisation(self,optimizer,epoch=100):
        self.epoch = epoch
        self.optimizer = optimizer
        
        self.model.to(self.device)

        self.metric_name = "spearmanr" if self.task == "stsb" else "matthews_correlation" if self.task == "cola" else "f1" if self.task == "mrpc" or self.task=="qqp" else "accuracy"
        self.metric = evaluate.load(self.metric_name)
        
        self.train_losses=[]
        self.val_losses=[]
        self.val_accuracy =[]
        self.train_accuracy =[]
        self.test_accuracy = []
    
    def compute_metrics(self,predictions,labels):
        return self.metric.compute(predictions=predictions, references=labels)[self.metric_name]
    
    def training(self,factor=1):
        bar_progress = tqdm(range(self.epoch))
        for i in range(self.epoch):
            l = []
            a=[]

            for batch in self.train_dataset:
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                l.append(loss.item())
                self.optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(**batch)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    a.append(self.compute_metrics(predictions,batch["labels"]))

            l = np.array(l)
            a = np.array(a)
            self.train_losses.append(l.mean())
            self.train_accuracy.append(a.mean())
            
            l = []
            a = []
            self.model.eval()
            for batch in self.val_dataset:
                with torch.no_grad():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    l.append(loss.item())
                    
                    a.append(self.compute_metrics(predictions,batch["labels"]))
            l = np.array(l)
            self.val_losses.append(l.mean())
            a = np.array(a)
            self.val_accuracy.append(a.mean())
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
        bar_progress = tqdm(range(len(self.test_dataset)))
        self.model.eval()
        a = []

        for batch in self.val_dataset:
            #batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                a.append(self.compute_metrics(predictions,batch["labels"]))
            bar_progress.update(1)
        a = np.array(a)
        self.test_accuracy.append(a.mean())
        
        print("The Accuracy on the test dataset :",self.test_accuracy)


    #Sauvegarde du model à l'endroit 'path'
    def save(self, path):
        torch.save(self.state_dict(), path)
        



        





