import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepmhci

from deepmhci.data_utils import ACIDS
from deepmhci.init import truncated_normal_
from deepmhci.modules import IConv, SGLU

class Network(nn.Module):
    """

    """
    def __init__(self, *, emb_size, vocab_size=len(ACIDS), padding_idx=0, peptide_pad=3, mhc_len=34, **kwargs):
        super(Network, self).__init__()
        self.emb_size = emb_size
        self.peptide_emb = nn.Embedding(vocab_size, emb_size)
        self.mhc_emb = nn.Embedding(vocab_size, emb_size)
        self.peptide_pad, self.padding_idx, self.mhc_len = peptide_pad, padding_idx, mhc_len

    def forward(self, peptide_x, mhc_x, *args, **kwargs):
        masks = peptide_x[:, self.peptide_pad: peptide_x.shape[1] - self.peptide_pad] != self.padding_idx
        return self.peptide_emb(peptide_x), self.mhc_emb(mhc_x), masks

    def reset_parameters(self):
        nn.init.uniform_(self.peptide_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.mhc_emb.weight, -0.1, 0.1)

class DeepMHCI(Network):
    def __init__(self, *, 
                 conv_num, 
                 conv_size, 
                 conv_off, 
                 linear_size,
                 glu_para,
                 dropout=0.5, **kwargs):
        super(DeepMHCI, self).__init__(**kwargs)
        self.glu = SGLU(**dict(glu_para)) 
        self.conv1 = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv_bn1 = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)
        self.conv2 = nn.ModuleList(IConv(cn, cs, self.mhc_len) for cn, cs in zip(conv_num, conv_size))
        self.conv_bn2 = nn.ModuleList(nn.BatchNorm1d(cn) for cn in conv_num)
        self.conv_off = conv_off
        self.dropout = nn.Dropout(dropout)
        linear_size = [sum(conv_num)] + linear_size
        self.linear = nn.ModuleList([nn.Conv1d(in_s, out_s, 1)
                                     for in_s, out_s in zip(linear_size[:-1], linear_size[1:])])
        self.linear_bn = nn.ModuleList([nn.BatchNorm1d(out_s) for out_s in linear_size[1:]])
        self.output = nn.Conv1d(linear_size[-1], 1, 1)
        self.reset_parameters()
        
    def forward(self, peptide_x, mhc_x, **kwargs):
        peptide_x, mhc_x, masks = super(DeepMHCI, self).forward(peptide_x, mhc_x)
        peptide_x_gated, _ = self.glu(peptide_x)
        
        conv_out1 = torch.cat([conv_bn(F.relu(conv(peptide_x[:, off: peptide_x.shape[1] - off], mhc_x)))
                              for conv, conv_bn, off in zip(self.conv1, self.conv_bn1, self.conv_off)], dim=1)        
        conv_out2 = torch.cat([conv_bn(F.relu(conv(peptide_x_gated[:, off: peptide_x_gated.shape[1] - off], mhc_x)))
                              for conv, conv_bn, off in zip(self.conv2, self.conv_bn2, self.conv_off)], dim=1)
        conv_out_add = F.gelu(conv_out1 + conv_out2)
        conv_out_add = self.dropout(conv_out_add)
        
        for linear, linear_bn in zip(self.linear, self.linear_bn):
            conv_out_add = linear_bn(F.relu(linear(conv_out_add)))  # B, l_s[0], L  
        masks = masks[:, None, -conv_out_add.shape[2]:]  # B, 1, L
        pool_out, _ = conv_out_add.masked_fill(~masks, -np.inf).max(dim=2, keepdim=True)
        return torch.sigmoid(self.output(pool_out).flatten())

    def reset_parameters(self):
        super(DeepMHCI, self).reset_parameters()
        for conv, conv_bn in zip(self.conv1, self.conv_bn1):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)
        for conv, conv_bn in zip(self.conv2, self.conv_bn2):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)
        for linear, linear_bn in zip(self.linear, self.linear_bn):
            truncated_normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)
            linear_bn.reset_parameters()
            nn.init.normal_(linear_bn.weight.data, mean=1.0, std=0.002)
        truncated_normal_(self.output.weight, std=0.1)
        nn.init.zeros_(self.output.bias)
