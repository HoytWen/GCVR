import torch as th
import torch.nn as nn
import torch.nn.functional as F
from GCL.augmentors import Compose
from utils.losses import normalized_mse_loss, normalized_l1_loss

class Encoder(nn.Module):
    def __init__(self, encoder, augmentor, contrast_model, cf, num_choices=1):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.aug1, self.aug2 = augmentor
        self.num_augmentors = len(self.aug2)
        self.num_choices = num_choices
        self.device = cf.device
        self.two_aug = cf.two_aug
        self.contrast_model = contrast_model

        self.disen = True if cf.lam_d > 0 else False

        self.aug_pred = True if cf.lam_a > 0 else False
        if self.aug_pred:
            self.aug_loss_func = th.nn.CrossEntropyLoss()

        self.adv = True if cf.lam_p > 0 else False
        self.cf = cf

        project_dim = cf.n_hidden * cf.n_layer
        self.essen_layer = th.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

        self.aug_layer = th.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

        if cf.add_proj:
            self.ori_layer = th.nn.Sequential(
                nn.Linear(project_dim, project_dim),
                nn.BatchNorm1d(project_dim),
                nn.ReLU(inplace=True),
                nn.Linear(project_dim, project_dim))

        if cf.recs_mode == 'dot':
            self.resc_layer = th.nn.Sequential(
                nn.Linear(project_dim, project_dim),
                nn.BatchNorm1d(project_dim),
                nn.ReLU(inplace=True),
                nn.Linear(project_dim, project_dim))
        elif cf.recs_mode == 'cat':
            self.resc_layer = th.nn.Sequential(
                nn.Linear(project_dim * 2, project_dim),
                nn.BatchNorm1d(project_dim),
                nn.ReLU(inplace=True),
                nn.Linear(project_dim, project_dim))
        else:
            raise NotImplementedError

        if self.aug_pred:
            self.aug_pred_layer = th.nn.Sequential(
                nn.Linear(project_dim, project_dim//2),
                nn.BatchNorm1d(project_dim//2),
                nn.ReLU(inplace=True),
                nn.Linear(project_dim//2, self.num_augmentors+1))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                th.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def cal_loss(self, adv_g, g1, g2, aug_label):

        cf = self.cf

        if cf.add_proj:
            adv_g, g1, g2 = [self.ori_layer(g) for g in [adv_g, g1, g2]]

        # ess_g1, ess_g2 = [self.essen_layer(g) for g in [g1, g2]]
        adv_g, ess_g1, ess_g2 = [self.essen_layer(g) for g in [adv_g, g1, g2]]
        aug_g1, aug_g2 = [self.aug_layer(g) for g in [g1, g2]]

        if cf.recs_mode == 'dot':
            com_g1 = self.resc_layer(ess_g1 * aug_g1)
            com_g2 = self.resc_layer(ess_g2 * aug_g2)
            cross_g1 = self.resc_layer(ess_g2 * aug_g1)
            cross_g2 = self.resc_layer(ess_g1 * aug_g2)
        elif cf.recs_mode == 'cat':
            com_g1 = self.resc_layer(th.cat([ess_g1, aug_g1], dim=1))
            com_g2 = self.resc_layer(th.cat([ess_g2, aug_g2], dim=1))
            cross_g1 = self.resc_layer(th.cat([ess_g2, aug_g1], dim=1))
            cross_g2 = self.resc_layer(th.cat([ess_g1, aug_g2], dim=1))
        else:
            raise NotImplementedError

        # closs1 = self.contrast_model(g1=com_g1, g2=g1)
        # closs2 = self.contrast_model(g1=com_g2, g2=g2)
        # closs3 = self.contrast_model(g1=ess_g1, g2=ess_g2)
        # closs = closs1 + closs2 + cf.lam_c * closs3

        closs = self.contrast_model(g1=ess_g1, g2=ess_g2)

        if self.adv:
            closs4 = self.contrast_model(g1=ess_g1, g2=adv_g)
            closs5 = self.contrast_model(g1=ess_g2, g2=adv_g)
            ploss = closs4 + closs5
            closs = closs + cf.lam_p * ploss
        else:
            ploss = th.tensor(-1)

        if self.disen:
            if cf.disen_mode == 'cst':
                dloss1 = self.contrast_model(g1=cross_g1, g2=g1)
                dloss2 = self.contrast_model(g1=cross_g2, g2=g2)
            elif cf.disen_mode == 'mse':
                dloss1 = F.mse_loss(cross_g1, g1)
                dloss2 = F.mse_loss(cross_g2, g2)
            elif cf.disen_mode == 'l1':
                dloss1 = F.l1_loss(cross_g1, g1)
                dloss2 = F.l1_loss(cross_g2, g2)
            elif cf.disen_mode == 'mse_norm':
                dloss1 = normalized_mse_loss(cross_g1, g1)
                dloss2 = normalized_mse_loss(cross_g2, g2)
            elif cf.disen_mode == 'l1_norm':
                dloss1 = normalized_l1_loss(cross_g1, g1)
                dloss2 = normalized_l1_loss(cross_g2, g2)
            else:
                raise NotImplementedError

            dloss = dloss1 + dloss2
            loss = closs + cf.lam_d * dloss
        else:
            dloss = th.tensor(-1)
            loss = closs

        if self.aug_pred:
            aug_rep = th.cat([self.aug_pred_layer(g) for g in [aug_g1, aug_g2]], dim=0)
            aloss = self.aug_loss_func(aug_rep, aug_label)
            loss = loss + cf.lam_a * aloss
        else:
            aloss = th.tensor(-1)
            loss = loss

        with th.no_grad():
            test_loss1 = self.contrast_model(g1=ess_g1, g2=ess_g2)
            test_loss2 = self.contrast_model(g1=aug_g1, g2=aug_g2)


        return loss, closs, dloss, aloss, ploss, test_loss1, test_loss2

    def forward(self, x, edge_index, edge_attr, batch, perturb=None, is_ft=False):

        if not is_ft:
            if self.two_aug:
                perm1 = th.randperm(self.num_augmentors)
                idx1 = perm1[:self.num_choices]
                aug_select1 = Compose([self.aug1[id] for id in idx1])
                x1, edge_index1, edge_weight1 = aug_select1(x, edge_index, edge_attr)
            else:
                x1, edge_index1, edge_weight1 = self.aug1(x, edge_index, edge_attr)
                idx1 = th.zeros([1]).long()

            perm2 = th.randperm(self.num_augmentors)
            idx2 = perm2[:self.num_choices]
            aug_select2 = Compose([self.aug2[id] for id in idx2])
            x2, edge_index2, edge_weight2 = aug_select2(x, edge_index, edge_attr)
            if not self.two_aug:
                idx2 = idx2 + 1

            z, g = self.encoder(x, edge_index, edge_attr, batch, perturb)
            z1, g1 = self.encoder(x1, edge_index1, edge_weight1, batch)
            z2, g2 = self.encoder(x2, edge_index2, edge_weight2, batch)
            
            anchor_aug_label = (th.ones((g.shape[0],))*(idx1)).long()
            pos_aug_label = (th.ones((g.shape[0],))*(idx2)).long()
            aug_label = th.cat([anchor_aug_label, pos_aug_label], dim=0).to(self.device)

            return g, g1, g2, aug_label
        
        else:
            z, g = self.encoder(x, edge_index, edge_attr, batch, perturb)
            return g




class Estimator(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(Estimator, self).__init__()
        project_dim = hidden_dim * num_layers

        self.mu_estimator = th.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ELU(),
            nn.Linear(project_dim, project_dim))

        self.logvar_estimator = th.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ELU(),
            nn.Linear(project_dim, project_dim),
            nn.Tanh()
        )

        self.resc_layer = th.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ELU(),
            nn.Linear(project_dim, project_dim))

    def reparameterize(self, mu, log_var):
        std = th.exp(log_var / 2)
        eps = th.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.resc_layer(z)
        return th.sigmoid(h)

    def forward(self, x):

        mu = self.mu_estimator(x)
        logvar = self.logvar_estimator(x)
        z = self.reparameterize(mu, logvar)
        x_resc = self.decode(z)

        return x_resc, mu, logvar


class FF(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, hidden_dim)
    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)