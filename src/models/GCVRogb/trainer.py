from time import time
import dgl
import torch
import torch as th
import numpy as np
from utils.evaluation import save_results
from utils.util_funcs import print_log
from tensorboardX import SummaryWriter

from torch_geometric.data import DataLoader
import torch.nn.functional as F
# import GCL.augmentors as A
import GCL.losses as L
from GCL.models import DualBranchContrast
from sklearn.model_selection import StratifiedKFold
from ogb.graphproppred import Evaluator

from models.GCVRogb.encoder import Encoder
from models.GCVRogb.gine import GConv
from models.GCVRogb.finetune import FineTuner
from utils.scheduler import CosineDecayScheduler
from utils.Evaluator_unsup import SVMEvaluator, OGBEvaluator
from augmentors import *

class GCVRogb_Trainer():
    def __init__(self, data, cf):
        self.__dict__.update(cf.__dict__)
        self.writer = SummaryWriter(cf.log_path)
        self.dataloader = DataLoader(data, batch_size=cf.step_batch_size, shuffle=True)
        self.cf = cf
        self.adv = True if cf.lam_p > 0 else False

        split_idx = data.get_idx_split()
        self.train_loader = DataLoader(data[split_idx["train"]], batch_size=128, shuffle=True)
        self.valid_loader = DataLoader(data[split_idx["valid"]], batch_size=128, shuffle=True)
        self.test_loader = DataLoader(data[split_idx["test"]], batch_size=128, shuffle=True)
        self.evaluator = Evaluator(cf.dataset)


        # train_label = []
        # for step, data in enumerate(self.train_loader):
        #     label = data.y
        #     train_label.append(label)
        #
        # train_label = th.cat(train_label)
        # print(train_label.sum())
        # print(train_label.sum()/len(train_label))
        #
        # test_label = []
        # for step, data in enumerate(self.test_loader):
        #     label = data.y
        #     test_label.append(label)
        #
        # test_label = th.cat(test_label)
        # print(test_label.sum())
        # print(test_label.sum() / len(test_label))
        # print(ssss)


        # Select split_ratio training data
        if cf.split_ratio < 1:
            perm = torch.randperm(split_idx['train'].size(0))
            k = int(len(split_idx['train']) * cf.split_ratio)
            idx = perm[:k]
            selected_idx = split_idx['train'][idx]
            self.part_train_loader = DataLoader(data[selected_idx], batch_size=128, shuffle=True)

        if cf.two_aug:
            aug1 = [Identity(),
                    RWSampling(num_seeds=1000, walk_length=10),
                    NodeDropping(pn=0.1),
                    FeatureMasking(pf=0.1),
                    EdgeRemoving(pe=0.1)]
            aug2 = [Identity(),
                    RWSampling(num_seeds=1000, walk_length=10),
                    NodeDropping(pn=0.1),
                    FeatureMasking(pf=0.1),
                    EdgeRemoving(pe=0.1)]

        else:
            aug1 = Identity()
            aug2 = [RWSampling(num_seeds=1000, walk_length=10),
                    NodeDropping(pn=0.1),
                    FeatureMasking(pf=0.1),
                    EdgeRemoving(pe=0.1)]

        model = GConv(input_dim=cf.feat_dim, hidden_dim=cf.n_hidden, num_layers=cf.n_layer, dropout= cf.dropout).to(cf.device)
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=cf.tau), mode='G2G', intraview_negs=cf.intra_negative).to(cf.device)
        self.encoder_model = Encoder(encoder=model, augmentor=(aug1, aug2), contrast_model=self.contrast_model, cf=cf).to(cf.device)

        if cf.use_scheduler:
            self.optimizer = th.optim.Adam(self.encoder_model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
            self.scheduler = CosineDecayScheduler(max_val=cf.lr, warmup_steps=cf.epochs//10, total_steps=cf.epochs)
        else:
            self.optimizer = th.optim.Adam(self.encoder_model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)

    def run(self):
        print('Start training...')
        all_results = []
        best_epoch = 0
        best_val, best_test = 0, 0
        for epoch in range(1, self.epochs+1):
            t0 = time()
            loss, closs, dloss, aloss, ploss, tloss1, tloss2 = self.train(epoch)
            print_log({'Epoch': epoch, 'Time': time() - t0, 'loss': loss, 'con_loss': closs, 'pert_loss': ploss,
                       'disen_loss': dloss, 'aug_loss': aloss, 'test_loss1':tloss1, 'test_loss2': tloss2})


        # ======= Finetune =======
        if self.cf.split_ratio < 1:
            finetuner = FineTuner(
                self.encoder_model, self.evaluator, self.part_train_loader, 
                self.valid_loader, self.test_loader, self.cf)
        else:
            finetuner = FineTuner(
                self.encoder_model, self.evaluator, self.train_loader, 
                self.valid_loader, self.test_loader, self.cf)

        for epoch in range(1, self.cf.ft_epochs+1):
            train_results = finetuner.finetune()
            print_log(train_results)
            if epoch % self.test_freq == 0:
                train_auc = finetuner.evaluate('train')
                val_auc = finetuner.evaluate('val')
                test_auc = finetuner.evaluate('test')
                if val_auc > best_val:
                    best_epoch = epoch
                    best_test = test_auc
                    best_val = val_auc
                all_results.append(test_auc)
                print('Evaluation!')
                print_log({
                    'Best Epoch': best_epoch,
                    'Best AUC': best_test,
                    'train_auc': train_auc,
                    'val_auc': val_auc,
                    'test_auc': test_auc,
                })

        if len(all_results) > 0:
            print(f'The Best Result is: {best_test:.4f}')
        else:
            result = self.test()
            print('Evaluation!')
            best_epoch = epoch
            # best_result = result['accuracy']
            # print_log({'MaF1': result['macro_f1'], 'MiF1': result['micro_f1'], 'Best_Acc': best_result, 'Best Epoch': best_epoch})
            # all_results.append(result['accuracy'])
            train_auc = finetuner.evaluate('train')
            val_auc = finetuner.evaluate('val')
            test_auc = finetuner.evaluate('test')
            best_test = test_auc
            print_log({
                    'Best Epoch': best_epoch,
                    'Best AUC': best_test,
                    'train_auc': result['train_auc'],
                    'val_auc': result['val_auc'],
                    'test_auc': result['test_auc'],
                })

        # assert best_result == np.max(all_results)
        # res = {'test_acc': f'{np.max(all_results):.4f}', 'val_acc': f'{np.max(all_results):.4f}', 'best_epoch': f'{best_epoch}'}
        res = {'test_auc': f'{best_test:.4f}', 'val_auc': f'{best_val:.4f}', 'best_epoch': f'{best_epoch}'}
        save_results(self.cf, res)

        return self.encoder_model

    def train(self, epoch):

        epoch_loss = 0
        epoch_closs = 0
        epoch_dloss = 0
        epoch_aloss = 0
        epoch_ploss = 0
        epoch_tloss1 = 0
        epoch_tloss2 = 0

        self.encoder_model.train()

        if self.use_scheduler:
            lr = self.scheduler.get(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        for step, data in enumerate(self.dataloader):
            data = data.to(self.device)

            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = th.ones((num_nodes, 1), dtype=th.float32, device=self.device)
            if self.adv:
                perturb_shape = (data.x.shape[0], self.n_hidden)
                perturb = torch.FloatTensor(*perturb_shape).uniform_(-self.delta, self.delta).to(self.device)
                perturb.requires_grad_()
                for i in range(self.perturb_step):
                    if i == 0:
                        adv_g, g1, g2, aug_label = self.encoder_model(data.x, data.edge_index, data.edge_attr, data.batch, perturb)
                        loss, closs, dloss, aloss, ploss, test_loss1, test_loss2 = self.encoder_model.cal_loss(adv_g, g1, g2, aug_label)
                        loss /= self.perturb_step
                    else:
                        loss.backward()
                        perturb_data = perturb.detach() + self.perturb_step_size * torch.sign(perturb.grad.detach())
                        perturb.data = perturb_data.data
                        perturb.grad[:] = 0

                        adv_g, g1, g2, aug_label = self.encoder_model(data.x, data.edge_index, data.edge_attr, data.batch, perturb)
                        loss, closs, dloss, aloss, ploss, test_loss1, test_loss2 = self.encoder_model.cal_loss(adv_g, g1, g2, aug_label)
                        loss /= self.perturb_step

            else:
                adv_g, g1, g2, aug_label = self.encoder_model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss, closs, dloss, aloss, ploss, test_loss1, test_loss2 = self.encoder_model.cal_loss(adv_g, g1, g2,
                                                                                                aug_label)


            loss.backward()
            if (step+1) % self.accumulation_step == 0:  ## What if the total step can not divided by accumulation_step
                if self.clip > 0:
                    th.nn.utils.clip_grad_norm_(self.encoder_model.parameters(), self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()


            epoch_loss += loss.item() * self.perturb_step if self.adv else loss.item()
            epoch_closs += closs.item()
            epoch_dloss += dloss.item()
            epoch_aloss += aloss.item()
            epoch_ploss += ploss.item()

            epoch_tloss1 += test_loss1.item()
            epoch_tloss2 += test_loss2.item()


        return epoch_loss, epoch_closs, epoch_dloss, epoch_aloss, epoch_ploss, epoch_tloss1, epoch_tloss2

    @th.no_grad()
    def test(self,):
        self.encoder_model.eval()
        if self.cf.split_ratio < 1.:
            train_x, train_y = self.get_embedding(self.part_train_loader)
        else:
            train_x, train_y = self.get_embedding(self.train_loader)
        val_x, val_y = self.get_embedding(self.valid_loader)
        test_x, test_y = self.get_embedding(self.test_loader)
        if self.cf.dataset == 'ogbg-molhiv':
            ee = OGBEvaluator(self.evaluator, base_classifier='svm')
        else:
            ee = OGBEvaluator(self.evaluator, base_classifier='lr')
        train_score, val_score, test_score = ee.evaluate(
            train_x, train_y, val_x, val_y, test_x, test_y)

        results = {
            'train_auc': train_score,
            'val_auc': val_score,
            'test_auc': test_score,
        }

        return results

    @th.no_grad()
    def get_embedding(self, dataloader):
        x, y = [], []
        for data in dataloader:
            data = data.to(self.device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = th.ones((num_nodes, 1), dtype=th.float32, device=self.device)

            # Get embedding
            g, _, _, _ = self.encoder_model(data.x, data.edge_index, data.edge_attr, data.batch)
            g = self.encoder_model.essen_layer(g)
        
            x.append(g)
            y.append(data.y)

        x = th.cat(x, dim=0).cpu()
        y = th.cat(y, dim=0).cpu()

        return x, y

