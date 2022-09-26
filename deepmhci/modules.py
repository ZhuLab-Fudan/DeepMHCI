from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import deepmhci
import math
from deepmhci.init import truncated_normal_

class IConv(nn.Module):

    def __init__(self, out_channels, kernel_size, mhc_len=34, stride=1, **kwargs):
        super(IConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, kernel_size, mhc_len))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.stride, self.kernel_size = stride, kernel_size
        self.reset_parameters()

    def forward(self, peptide_x, mhc_x, **kwargs):
        w = F.relu(torch.tensordot(mhc_x, self.weight, dims=((1,), (-1,))).permute(2, 0, 3, 1))    
        outputs = torch.stack([(w * peptide_x[:, i: i+self.kernel_size]).sum((-2, -1)) + self.bias[:, None]
                               for i in range(0, peptide_x.shape[1]-self.kernel_size+1, self.stride)])
        return outputs.permute(2, 1, 0)

    def reset_parameters(self):
        truncated_normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

class SLinear(nn.Module):

    def __init__(self, 
                 pep_len, 
                 in_channel, 
                 out_channel, 
                 bias : bool = True
                 ):
        super(SLinear, self).__init__()
        self.pep_len = pep_len
        self.bias_bool = bias
        self.weight = nn.Parameter(torch.Tensor(pep_len, in_channel, out_channel))
        if bias:    
            self.bias = nn.Parameter(torch.Tensor(pep_len, out_channel))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, peptide_x):
        peptide_x = torch.tensordot(peptide_x, self.weight, dims=((-1,), (1,)))
        peptide_x = peptide_x[:,torch.arange(self.pep_len).unsqueeze(-1), torch.arange(self.pep_len).unsqueeze(-1),:]
        if self.bias_bool:
            return peptide_x.squeeze(2) + self.bias
        else:
            return peptide_x.squeeze(2)
        
    def reset_parameters(self):
        truncated_normal_(self.weight, std=0.02)
        if self.bias_bool:
            nn.init.zeros_(self.bias)

class SGLU(nn.Module):

    def __init__(self, 
                 pep_len, 
                 in_channel, 
                 hidden_layer_size,
                 dropout_rate = None,
                 SLinear = SLinear
                ):
        super(SGLU, self).__init__()
        self.pep_len = pep_len
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = SLinear(pep_len, in_channel, hidden_layer_size)
        self.gated_layer = SLinear(pep_len, in_channel, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
            
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x)) 
        return torch.mul(activation, gated), gated
    
