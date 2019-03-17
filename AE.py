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
                 symmetric = False,
                 layer_normalization = True):
        
        super(AE, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tied = tied
        self.symmetric = symmetric
        self.layer_normalization = layer_normalization
        
        self.W1 = nn.Parameter(torch.randn(hidden_size, input_size, num_layers) * 0.1)
        self.W2 = nn.Parameter(torch.randn(input_size, hidden_size, num_layers) * 0.1)
        self.b = nn.Parameter(torch.randn(hidden_size, num_layers) * 0.01)
        self.c = nn.Parameter(torch.randn(input_size, num_layers) * 0.01)
        
        self.slope = nn.Parameter(torch.tensor(1/3))
        self.normed = nn.LayerNorm(hidden_size)
    
    def forward(self, batch):
        
        if self.tied:  
            for i in range(self.num_layers):
#             if self.tied:
#                 W1 = self.W1[:, :, 0]
#                 W2 = self.W2[:, :, 0]
#             else:
#                 W1 = self.W1[: ,:, i]
#                 W2 = self.W2[:, :, i]
              
                batch = F.linear(batch, weight = self.W1[:, :, 0], bias = self.b[:, i])
                if self.layer_normalization:
                    batch = self.normed(batch)
                batch = F.prelu(batch, self.slope)
                # batch = torch.sigmoid(batch)
                if self.symmetric:
                    batch = F.linear(batch, weight = self.W1[:, :, 0].t(), bias = self.c[:, i])
                else:
                    batch = F.linear(batch, weight = self.W2[:, :, 0], bias = self.c[:, i])
        else:
            for i in range(self.num_layers):
                batch = F.linear(batch, weight = self.W1[:, :, i], bias = self.b[:, i])
                if self.layer_normalization:
                    batch = self.normed(batch)
                batch = F.prelu(batch, self.slope)
                # batch = torch.sigmoid(batch)
                if self.symmetric:
                    batch = F.linear(batch, weight = self.W1[:, :, i].t(), bias = self.c[:, i])
                else:
                    batch = F.linear(batch, weight = self.W2[:, :, i], bias = self.c[:, i])
        
        return batch