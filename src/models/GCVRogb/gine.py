import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, GINEConv
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder


def make_gine_conv(input_dim, out_dim):
    return GINEConv(nn.Sequential(nn.Linear(input_dim, out_dim*2), torch.nn.BatchNorm1d(2*out_dim), nn.ReLU(), nn.Linear(out_dim*2, out_dim)))
    
class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(GConv, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)

        for i in range(num_layers):
            self.layers.append(make_gine_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))


    def forward(self, x, edge_index, edge_attr, batch, perturb=None):
        # z = x
        z = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        zs = []
        # for layer in range(self.num_layers):
        #     z = self.layers[layer](z, edge_index)
        #     z = self.batch_norms[layer](z)

        #     if layer == 0 and perturb is not None:
        #         z += perturb

        #     if layer == self.num_layers -1:
        #         z = F.dropout(z, self.dropout, training=self.training)
        #     else:
        #         z = F.dropout(F.relu(z), self.dropout, training=self.training)
        #     zs.append(z)
        
        for layer, (conv, bn) in enumerate(zip(self.layers, self.batch_norms)):
            z = conv(z, edge_index, edge_attr)
            z = bn(z)

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