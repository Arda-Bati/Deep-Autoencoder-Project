import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class AE(nn.Module):
    
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers = 1, 
                 tied = True,
                 layer_normalization = False):
        
        super(AE, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tied = tied
        self.layer_normalization = layer_normalization
        
        self.W1 = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.W2 = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.b = nn.Parameter(torch.randn(hidden_size) * 0.01)
        self.c = nn.Parameter(torch.randn(input_size) * 0.01)
        self.slope = nn.Parameter(torch.randn(1))
        
        self.normed = nn.LayerNorm(hidden_size)
    
    def forward(self, batch):
        
        for i in range(self.num_layers):
            batch = F.linear(batch, self.W1, bias = self.b)
            if self.layer_normalization:
                batch = self.normed(batch)
            batch = F.prelu(batch, self.slope)
            # batch = torch.sigmoid(batch)
            if self.tied:
                batch = F.linear(batch, weight = self.W1.t(), bias = self.c)
            else:
                batch = F.linear(batch, weight = self.W2, bias = self.c)
        
        return batch