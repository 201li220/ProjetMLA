import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

torch.cuda.empty_cache()


class Fine_Tuning:
    def __init__(self,model,tokenizer,dataset):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    def prepared_data(self, dataset_size=4000, batch_size=16):
        def tokenize_function(features, tokenizer=self.tokenizer):
            self.column_names = list(features.keys())
            if len(self.column_names) > 3:
                inputs = {}
                inputs['text'] = features[self.column_names[0]]
                inputs['text_pair'] = features[self.column_names[1]]
            
                tokenized_inputs = tokenizer(
                    **inputs,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            
                return tokenized_inputs
            else:
                tokenized_inputs = tokenizer(
                    features[self.column_names[0]],
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return tokenized_inputs


        tokenized_datasets = self.dataset.map(tokenize_function,batched=True)
        if len(self.column_names)>3:
            tokenized_datasets = tokenized_datasets.remove_columns([self.column_names[0]])
            tokenized_datasets = tokenized_datasets.remove_columns([self.column_names[1]])
        else:
            tokenized_datasets = tokenized_datasets.remove_columns([self.column_names[0]])
        tokenized_datasets = tokenized_datasets.remove_columns(['idx'])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
       
        reduced_train_dataset = tokenized_datasets['train'].select(range(dataset_size))
        reduced_validation_dataset = tokenized_datasets['validation'].select(range(500))
        reduced_test_dataset = tokenized_datasets['test'].select(range(500))
        
        A = DataLoader(reduced_train_dataset, shuffle=True, batch_size=batch_size)
        B = DataLoader(reduced_validation_dataset, shuffle=True, batch_size=batch_size)
        C = DataLoader(reduced_validation_dataset, shuffle=True, batch_size=batch_size)

        self.train_dataset = [ {k: v.to(self.device) for k, v in batch.items()} for batch in A ]
        self.val_dataset = [ {k: v.to(self.device) for k, v in batch.items()} for batch in B]
        self.test_dataset = [ {k: v.to(self.device) for k, v in batch.items()} for batch in C]

        del self.column_names
    
    def initialisation(self,optimizer,epoch=100):
        self.epoch = epoch
        self.optimizer = optimizer
        
        self.model.to(self.device)
        
        self.train_losses=[]
        self.val_losses=[]
        self.val_accuracy =[]
        self.train_accuracy =[]
        self.test_accuracy = []
        
    def training(self,factor=1):
        bar_progress = tqdm(range(self.epoch))
        for i in range(self.epoch):
            l = []
            total = 0
            correct = 0
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
                    total += batch["labels"].size(0)
                    correct += (predictions==batch["labels"]).sum().item()

            l = np.array(l)
            self.train_losses.append(l.mean())
            self.train_accuracy.append(correct/total)
            
            l = []
            total = 0
            correct = 0
            self.model.eval()
            for batch in self.val_dataset:
                with torch.no_grad():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    l.append(loss.item())
                    
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    total += batch["labels"].size(0)
                    correct += (predictions==batch["labels"]).sum().item()
            l = np.array(l)
            self.val_losses.append(l.mean())
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
        bar_progress = tqdm(range(len(self.test_dataset)))
        self.model.eval()
        total = 0
        correct = 0
        for batch in self.test_dataset:
            #batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                total += batch["labels"].size(0)
                correct += (predictions==batch["labels"]).sum().item()
                bar_progress.update(1)
        self.test_accuracy.append(correct/total)
        
        print("The Accuracy on the test dataset :",self.test_accuracy)


    #Sauvegarde du model à l'endroit 'path'
    def save(self, path):
        torch.save(self.state_dict(), path)
        


