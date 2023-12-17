# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:47:48 2023

@author: Elsa Laziamond
"""

from transformers import BertForSequenceClassification, BertTokenizer,BertModel,BertConfig
from torch import nn

"""Adapter module using bottleneck architecture"""

class Adapter(nn.Module):
    def __init__(self,
                 input_size, 
                 hidden_size):
                     
        """Adapter config:
        Args :
            input_size : Size of the input
            hidden_size : Size of the Adapter
        """
        
        super(Adapter,self).__init__()
        self.FeedForward_down_layer = nn.Linear(input_size, hidden_size)
        self.FeedForward_up_layer = nn.Linear(hidden_size,input_size)
        self.Nonlinearity_layer = nn.GELU

        self._initialize_weights()

    def _initialize_weights(self):

        """Near zero initialization of the feedforward layers 
        resulting in a near identity initialization of the adapter module
        """
        
        nn.init.normal_(self.FeedForward_down_layer.weight, mean = 0, std = 1e-2)
        nn.init.normal_(self.FeedForward_up_layer.weight, mean = 0, std = 1e-2)
        
    def forward(self,x):
        inputs = x
        x = self.FeedForward_down_layer(x)
        x = self.Nonlinearity_layer(x)
        x = self.FeedForward_up_layer(x)
        return x+inputs
