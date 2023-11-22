# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:47:48 2023

@author: Ugo Laziamond
"""

from transformers import BertModel, BertConfig
from torch import nn



class Adapter(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(Adapter,self).__init__()
        self.FeedForward_down_layer = nn.Linear(input_size, hidden_size)
        self.FeedForward_up_layer = nn.Linear(hidden_size,input_size)
        self.Nonlinearity_layer = nn.GELU()
        
    def forward(self,x):
        residual_inputs = x
        x = self.FeedForward_down_layer(x)
        x = self.Nonlinearity_layer(x)
        x = self.FeedForward_up_layer(x)
        return x+residual_inputs

class BERTAdapter(nn.Module):
    def __init__(self,config,hidden_size):
        super(BERTAdapter,self).__init__()
        self.BERT = BertModel(config)
        self.embedding = self.BERT.embeddings
        self.transformer = self.BERT.encoder.layer
        self.pooler = self.BERT.pooler
        self.A1 = Adapter(768,hidden_size)
        self.A2 = Adapter(768,hidden_size)
    
    def forward(self,x):
        outputs_embedding = self.embedding(x)
        for layer in self.transformer:
            inputs = outputs_embedding
            #outputs_attention = layer.attention(outputs_embedding)
            sub_layer = layer.attention
            outputs_self = sub_layer.self(outputs_embedding)
            outputs_lin = sub_layer.output.dense(outputs_self[0])
            outputs_A1 = self.A1.forward(outputs_lin)
            outputs_A1 = outputs_A1+inputs
            outputs_dropout = sub_layer.output.dropout(outputs_A1)
            outputs_attention =  sub_layer.output.LayerNorm(outputs_dropout+outputs_self)
            inputs = outputs_attention
            outputs_intermediate = layer.intermediate(outputs_attention)
            outputs_output = layer.output.dense(outputs_intermediate)
            outputs_A2 = self.A2.forward(outputs_output)
            outputs_A2 = outputs_A2+inputs
            outputs_dropout = layer.output.dropout(outputs_A2)
            outputs_output = layer.output.LayerNorm(outputs_dropout+outputs_attention)
        outputs_pooler = self.pooler(outputs_output)
        return outputs_pooler
