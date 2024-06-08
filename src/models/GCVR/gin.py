import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, GINConv


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(GConv, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))


    def forward(self, x, edge_index, batch, perturb=None):
        z = x
        zs = []
        for layer in range(self.num_layers):
            z = self.layers[layer](z, edge_index)
            z = self.batch_norms[layer](z)

            if layer == 0 and perturb is not None:
                z += perturb

            if layer == self.num_layers -1:
                z = F.dropout(z, self.dropout, training=self.training)
            else:
                z = F.dropout(F.relu(z), self.dropout, training=self.training)

            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g

class MLP(nn.Module):
    def __init__(self, nhid, nclass, norm_type='batch'):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nhid, nhid)
        self.layer2 = nn.Linear(nhid, nclass)

        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(nhid)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(nhid)
        else:
            raise ValueError
        self.act_fn = nn.ReLU()

    def forward(self, x):

        x = self.layer1(x)
        x = self.norm(x)
        x = self.act_fn(x)
        x = self.layer2(x)

        return x