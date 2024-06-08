import torch as th
import torch.nn as nn
import torch.nn.functional as F
from GCL.augmentors import Compose
from models.GCVR.gin import GConv, MLP
from utils.losses import normalized_mse_loss
import GCL.losses as L

import numpy as np

class Encoder(nn.Module):
    def __init__(self, augmentor, contrast_model, cf, num_choices=1):
        super(Encoder, self).__init__()
        self.aug1, self.aug2 = augmentor
        self.num_augmentors = len(self.aug2)
        self.num_choices = num_choices
        self.device = cf.device
        self.two_aug = cf.two_aug
        self.contrast_model = contrast_model
        self.decoder_layer = cf.decoder_layer
        self.disen_mode = cf.disen_mode
        self.recs_mode = cf.recs_mode
        self.device = cf.device
        self.lam_c = cf.lam_c
        self.lam_d = cf.lam_d
        self.lam_p = cf.lam_p
        self.add_proj = cf.add_proj
        self.adv = True if cf.lam_p > 0 else False

        self.encoder = GConv(input_dim=cf.feat_dim, hidden_dim=cf.n_hidden, num_layers=cf.n_layer, dropout= cf.dropout)

        project_dim = cf.n_hidden * cf.n_layer

        if self.decoder_layer == 1:
            self.essen_layer = nn.Linear(project_dim, project_dim)
            self.aug_layer = nn.Linear(project_dim, project_dim)
        elif self.decoder_layer == 2:
            self.essen_layer = MLP(project_dim, project_dim, norm_type=cf.norm_type)
            self.aug_layer = MLP(project_dim, project_dim, norm_type=cf.norm_type)
        else:
            raise ValueError

        if cf.add_proj:
            self.proj_layer = MLP(project_dim, project_dim, norm_type='batch')

        if self.recs_mode == 'cat':
            self.resc_layer = MLP(project_dim*2, project_dim, norm_type=cf.norm_type)
        elif self.recs_mode == 'dot':
            self.resc_layer = MLP(project_dim, project_dim, norm_type=cf.norm_type)
        else:
            raise ValueError

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                th.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def cal_loss(self, adv_g, g1, g2):

        if self.add_proj:
            adv_g, g1, g2 = [self.proj_layer(g) for g in [adv_g, g1, g2]]

        ess_g1, ess_g2 = [self.essen_layer(g) for g in [g1, g2]]
        aug_g1, aug_g2 = [self.aug_layer(g) for g in [g1, g2]]

        closs = self.contrast_model(g1=ess_g1, g2=ess_g2)

        assert self.lam_c > 0
        if self.lam_d > 0.0:
            dloss = self.reconstruct_loss(g1, ess_g1, aug_g1, g2, ess_g2, aug_g2)
            loss = self.lam_c*closs + self.lam_d * dloss
        else:
            dloss = th.tensor(-1)
            loss = self.lam_c*closs

        if self.adv:
            adv_g = self.essen_layer(adv_g)
            ploss1 = self.contrast_model(g1=ess_g1, g2=adv_g)
            ploss2 = self.contrast_model(g1=ess_g2, g2=adv_g)
            ploss = ploss1 + ploss2
            loss += self.lam_p * ploss
        else:
            ploss = th.tensor(-1)

        return loss, closs, dloss, ploss

    def forward(self, x, edge_index, batch, perturb=None):

        if self.two_aug:
            perm1 = th.randperm(self.num_augmentors)
            idx1 = perm1[:self.num_choices]
            aug_select1 = Compose([self.aug1[id] for id in idx1])
            x1, edge_index1, edge_weight1 = aug_select1(x, edge_index)
        else:
            x1, edge_index1, edge_weight1 = self.aug1(x, edge_index)

        perm2 = th.randperm(self.num_augmentors)
        idx2 = perm2[:self.num_choices]
        aug_select2 = Compose([self.aug2[id] for id in idx2])
        x2, edge_index2, edge_weight2 = aug_select2(x, edge_index)

        z, g = self.encoder(x, edge_index, batch, perturb)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)

        return g, g1, g2

    def reconstruct_loss(self, g1, ess_g1, aug_g1, g2, ess_g2, aug_g2):

        if self.recs_mode == 'cat':
            com_g1 = self.resc_layer(th.cat([ess_g1, aug_g1], dim=1))
            com_g2 = self.resc_layer(th.cat([ess_g2, aug_g2], dim=1))
            cross_g1 = self.resc_layer(th.cat([ess_g2, aug_g1], dim=1))
            cross_g2 = self.resc_layer(th.cat([ess_g1, aug_g2], dim=1))
        elif self.recs_mode == 'dot':
            com_g1 = self.resc_layer(ess_g1 * aug_g1)
            com_g2 = self.resc_layer(ess_g2 * aug_g2)
            cross_g1 = self.resc_layer(ess_g2 * aug_g1)
            cross_g2 = self.resc_layer(ess_g1 * aug_g2)
        else:
            raise ValueError

        if self.disen_mode == 'mse':
            loss1 = F.mse_loss(cross_g1, g1)
            loss2 = F.mse_loss(cross_g2, g2)
            loss3 = F.mse_loss(com_g1, g1)
            loss4 = F.mse_loss(com_g2, g2)
        elif self.disen_mode == 'mse_norm':
            loss1 = normalized_mse_loss(cross_g1, g1)
            loss2 = normalized_mse_loss(cross_g2, g2)
            loss3 = normalized_mse_loss(com_g1, g1)
            loss4 = normalized_mse_loss(com_g2, g2)
        else:
            raise NotImplementedError

        loss = loss1 + loss2 + loss3 + loss4

        return loss