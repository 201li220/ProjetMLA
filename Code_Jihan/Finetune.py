# -*- coding: utf-8 -*-
"""
@author: Yu Jihan
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AdamW
from datasets import load_dataset, load_metric

torch.cuda.empty_cache()

class Fine_Tuning:
    def __init__(self, checkpoint, task, batch_size, learning_rate, epoch):

        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.task = task
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def Tune(self):
        def tokenize_function(self, examples, raw_dataset):
            task_to_keys = {
                "cola": ("sentence", None),
                "mrpc": ("sentence1", "sentence2"),
                "sst2": ("sentence", None),
                "stsb": ("sentence1", "sentence2"),
                }
            sentence1_key, sentence2_key = task_to_keys[self.task]
            if sentence2_key is None:
                print(f"Sentence: {raw_dataset['train'][0][sentence1_key]}")
            else:
                print(f"Sentence 1: {raw_dataset['train'][0][sentence1_key]}")
                print(f"Sentence 2: {raw_dataset['train'][0][sentence2_key]}")

            if sentence2_key is None:
                return tokenizer(examples[sentence1_key], truncation=True)
            return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

        raw_dataset = load_dataset('glue', self.task)
        raw_metric = load_metric('glue', self.task)

        # Load the tokenizerz
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=True)

        # Dynamic padding
        tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets = tokenized_datasets.with_format("torch")

        # Data collator for dynamic padding as per batch
        data_collator = DataCollatorWithPadding(tokenizer)
        train_dataloader = DataLoader(tokenized_datasets["train"], 
                                      batch_size=self.batch_size, 
                                      shuffle=True, 
                                      collate_fn=data_collator
                                    )
        validation_dataloader = DataLoader(tokenized_datasets["validation"],
                                       batch_size=self.batch_size,
                                       collate_fn=data_collator
                                    )
        test_dataloader = DataLoader(tokenized_datasets["test"],
                                        batch_size=self.batch_size,
                                        collate_fn=data_collator
                                    )

        # Tunning
        num_labels = 1 if self.task=="stsb" else 2

        # cache a pre-trained BERT model for two-class classification
        model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=num_labels)

        # Metric
        def compute_metrics(self, eval_pred):
            predictions, labels = eval_pred
            if self.task != "stsb":
                predictions = np.argmax(predictions, axis=1)
            else:
                predictions = predictions[:, 0]
            return raw_metric.compute(predictions=predictions, references=labels)

        metric_name = "spearmanr" if self.task == "stsb" else "matthews_correlation" if self.task == "cola" else "accuracy"
        model_name = self.checkpoint.split("/")[-1]

        args = TrainingArguments(
            f"{model_name}-finetuned-{self.task}",
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epoch,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            push_to_hub=True,
        )

        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        print(trainer.train())
        print(trainer.evaluate())

    if __name__ == "__main__":
        print('rien')