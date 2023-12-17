"""
This script defines a Fine_Tuning class for fine-tuning a BERT model on GLUE tasks. 
It initializes the model, tokenizes input sentences, loads datasets, and performs training and evaluation. 
The training loop includes forward propagation, gradient descent, and metric computation. 
The script offers flexibility for various GLUE tasks and allows saving the model's state dictionary to a file.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from datasets import load_dataset  
import evaluate  

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

torch.cuda.empty_cache()  # Clear GPU memory

class Fine_Tuning:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    def load_data(self, task, dataset_size=5000, batch_size=16):
        # Load dataset using Hugging Face datasets library
        dataset = load_dataset('glue', task)

        # Mapping task to corresponding keys in the dataset
        task_to_keys = {"cola": ("sentence", None),
                        "mrpc": ("sentence1", "sentence2"),
                        "sst2": ("sentence", None),
                        "stsb": ("sentence1", "sentence2"),
                        "qqp": ("question1", "question2"),
                       }

        sentence1_key, sentence2_key = task_to_keys[task]
        
        def tokenize_function(features, tokenizer=self.tokenizer):
            # Tokenize input sentences based on the task
            if sentence2_key is None:
                return tokenizer(features[sentence1_key], padding='max_length', truncation=True, return_tensors='pt')
            return tokenizer(features[sentence1_key], features[sentence2_key], padding='max_length', truncation=True, return_tensors='pt')

        # Tokenize the entire dataset
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        
        # Remove unnecessary columns and rename the label column
        if sentence2_key is None:
          tokenized_datasets = tokenized_datasets.remove_columns(["idx", sentence1_key])
        else:
          tokenized_datasets = tokenized_datasets.remove_columns(["idx", sentence1_key, sentence2_key])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets = tokenized_datasets.with_format("torch")
        print(len(tokenized_datasets['train']))
        
        # Select a subset of the dataset based on the specified dataset_size
        reduced_train_dataset = tokenized_datasets['train'].select(range(min(dataset_size, len(tokenized_datasets['train']))))
        reduced_validation_dataset = tokenized_datasets['validation'].select(range(min(500, dataset_size)))
        reduced_test_dataset = tokenized_datasets['test'].select(range(min(500, dataset_size)))
        
        # Create DataLoaders for training, validation, and test datasets
        A = DataLoader(reduced_train_dataset, shuffle=True, batch_size=batch_size)
        B = DataLoader(reduced_validation_dataset, shuffle=True, batch_size=batch_size)
        C = DataLoader(reduced_test_dataset, shuffle=True, batch_size=batch_size)

        # Move datasets to the specified device (GPU or CPU)
        self.train_dataset = [{k: v.to(self.device) for k, v in batch.items()} for batch in A]
        self.val_dataset = [{k: v.to(self.device) for k, v in batch.items()} for batch in B]
        self.test_dataset = [{k: v.to(self.device) for k, v in batch.items()} for batch in C]
    
    def initialisation(self, optimizer, epoch=100):
        # Initialize hyperparameters and metrics
        self.epoch = epoch
        self.optimizer = optimizer
        self.model.to(self.device)

        # Choose metric based on the task
        self.metric_name = "spearmanr" if self.task == "stsb" else "f1" if (self.task == "mrpc" or self.task == "qqp") else "accuracy"
        self.metric = evaluate.load(self.metric_name)
        
        # Initialize lists to store training and validation metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracy = []
        self.train_accuracy = []
        self.test_accuracy = []
    
    def compute_metrics(self, predictions, labels):
        # Compute metrics based on the chosen metric name
        return self.metric.compute(predictions=predictions, references=labels)[self.metric_name]
    
    def training(self, factor=1):
        # Training loop
        bar_progress = tqdm(range(self.epoch * (len(self.train_dataset) + len(self.val_dataset))))
        for i in range(self.epoch):
            l = []
            a = []

            for batch in self.train_dataset:
                #Forward Propagation and Gradient Descent
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                l.append(loss.item())
                self.optimizer.step()

                self.model.eval()
                 #Evaluate the cost function on the validation dataset
                with torch.no_grad():
                    outputs = self.model(**batch)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    a.append(self.compute_metrics(predictions, batch["labels"]))
                bar_progress.update(1)
            l = np.array(l)
            a = np.array(a)
            self.train_losses.append(l.mean())
            self.train_accuracy.append(a.mean())
            
            l = []
            a = []
            self.model.eval()
            #Compute the accuracy on the validation dataset
            for batch in self.val_dataset:
                with torch.no_grad():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    l.append(loss.item())
                    
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    a.append(self.compute_metrics(predictions, batch["labels"]))
                bar_progress.update(1)
            l = np.array(l)
            self.val_losses.append(l.mean())
            a = np.array(a)
            self.val_accuracy.append(a.mean())
    
    def plot_accuracy(self):
        # Plot training and validation accuracy
        plt.plot(self.train_accuracy, label='Train Accuracy')
        plt.plot(self.val_accuracy, label='Validation Accuracy')
        
        plt.title('Graph of Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
        
    def plot_loss(self):
        # Plot training and validation loss
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        
        plt.title('Graph of Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        
    def evaluation(self):
        # Evaluate the model on the test dataset
        bar_progress = tqdm(range(len(self.val_dataset)))
        self.model.eval()
        a = []

        for batch in self.val_dataset:
            with torch.no_grad():
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                a.append(self.compute_metrics(predictions, batch["labels"]))
            bar_progress.update(1)
        a = np.array(a)
        self.test_accuracy.append(a.mean())
        
        print("The Accuracy on the test dataset:", self.test_accuracy)

    def save(self, path):
        # Save the model's state dictionary to a file
        torch.save(self.state_dict(), path)

        



        


