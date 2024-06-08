from time import time
import dgl
import torch
import torch as th
import numpy as np
from utils.evaluation import save_results
from utils.util_funcs import print_log
import wandb

from torch_geometric.data import DataLoader
import torch.nn.functional as F
import GCL.augmentors as A
import GCL.losses as L
from GCL.models import DualBranchContrast
from sklearn.model_selection import StratifiedKFold

from models.GCVR.encoder import Encoder
from models.GCVR.gin import GConv
from utils.losses import normalized_mse_loss, normalized_l1_loss
from utils.scheduler import CosineDecayScheduler
from utils.Evaluator import SVMEvaluator

class GCVR_Trainer():
    def __init__(self, data, cf):
        self.__dict__.update(cf.__dict__)
        # self.writer = SummaryWriter(cf.log_path)
        # wandb.init(project=f"{cf.model}_{cf.dataset}", entity="eric_wen", config=cf.__dict__)
        self.dataloader = DataLoader(data, batch_size=cf.step_batch_size, shuffle=True)
        self.cf = cf
        self.lam_d = cf.lam_d
        self.adv = True if cf.lam_p > 0 else False

        if cf.two_aug:
            aug1 = [A.Identity(),
                    A.RWSampling(num_seeds=1000, walk_length=10),
                                   A.NodeDropping(pn=0.2),
                                   A.FeatureMasking(pf=0.2),
                                   A.EdgeRemoving(pe=0.2)]
            aug2 = [A.Identity(),
                    A.RWSampling(num_seeds=1000, walk_length=10),
                    A.NodeDropping(pn=0.2),
                    A.FeatureMasking(pf=0.2),
                    A.EdgeRemoving(pe=0.2)]

        else:
            aug1 = A.Identity()
            aug2 = [A.RWSampling(num_seeds=1000, walk_length=10),
                                   A.NodeDropping(pn=0.2),
                                   A.FeatureMasking(pf=0.2),
                                   A.EdgeRemoving(pe=0.2)]

        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=cf.tau), mode='G2G', intraview_negs=cf.intra_negative).to(cf.device)
        self.encoder_model = Encoder(augmentor=(aug1, aug2), contrast_model=self.contrast_model, cf=cf).to(cf.device)

        if cf.use_scheduler:
            self.optimizer = th.optim.Adam(self.encoder_model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
            self.scheduler = CosineDecayScheduler(max_val=cf.lr, warmup_steps=cf.epochs//10, total_steps=cf.epochs)
        else:
            self.optimizer = th.optim.Adam(self.encoder_model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)


    def run(self):
        print('Start training...') 
        all_results = []
        best_epoch = 0
        best_result = 0
        for epoch in range(1, self.epochs+1):
            t0 = time()
            loss, closs, dloss, ploss = self.train(epoch)
            print_log({'Epoch': epoch, 'Time': time() - t0, 'loss': loss, 'con_loss': closs, 'pert_loss': ploss,
                       'disen_loss': dloss})

            # wandb.log({"Epoch": epoch, "loss": loss, "CL_inv": tloss1, "CL_aug": tloss2})

            if epoch % self.test_freq == 0:
                result = self.test()

                if result['accuracy'] > best_result:
                    best_epoch = epoch
                    best_result = result['accuracy']

                all_results.append(result['accuracy'])
                print('Evaluation!')
                print_log({'MiF1': result['micro_f1'], 'MaF1': result['macro_f1'], 'Acc': result['accuracy'],
                           'Best ACC': best_result, 'Best Epoch': best_epoch})

        if len(all_results) > 0:
            print(f'The Best Result is: {np.max(all_results):.4f}')
        else:
            result = self.test()
            print('Evaluation!')
            best_epoch = epoch
            best_result = result['accuracy']
            print_log({'MaF1': result['macro_f1'], 'MiF1': result['micro_f1'], 'Best_Acc': best_result, 'Best Epoch': best_epoch})
            all_results.append(result['accuracy'])

        assert best_result == np.max(all_results)
        res = {'test_acc': f'{np.max(all_results):.4f}', 'val_acc': f'{np.max(all_results):.4f}', 'best_epoch': f'{best_epoch}'}
        save_results(self.cf, res)

        return self.encoder_model

    def train(self, epoch):

        epoch_loss = 0
        epoch_closs = 0
        epoch_dloss = 0
        epoch_ploss = 0

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
                        adv_g, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch, perturb)
                        loss, closs, dloss, ploss = self.encoder_model.cal_loss(adv_g, g1, g2)
                        loss /= self.perturb_step
                    else:
                        loss.backward()
                        perturb_data = perturb.detach() + self.perturb_step_size * torch.sign(perturb.grad.detach())
                        perturb.data = perturb_data.data
                        perturb.grad[:] = 0

                        adv_g, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch, perturb)
                        loss, closs, dloss, ploss = self.encoder_model.cal_loss(adv_g, g1, g2)
                        loss /= self.perturb_step

            else:
                adv_g, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch)
                loss, closs, dloss, ploss = self.encoder_model.cal_loss(adv_g, g1, g2)


            loss.backward()
            if (step+1) % self.accumulation_step == 0:
                if self.clip > 0:
                    th.nn.utils.clip_grad_norm_(self.encoder_model.parameters(), self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()


            epoch_loss += loss.item() * self.perturb_step if self.adv else loss.item()
            epoch_closs += closs.item()
            epoch_dloss += dloss.item()
            epoch_ploss += ploss.item()


        return epoch_loss, epoch_closs, epoch_dloss, epoch_ploss

    @th.no_grad()
    def test(self, eval='inv'):
        self.encoder_model.eval()
        x = []
        y = []

        for data in self.dataloader:
            data = data.to(self.device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = th.ones((num_nodes, 1), dtype=th.float32, device=self.device)

            g, _, _ = self.encoder_model(data.x, data.edge_index, data.batch)
            if self.lam_d > 0:
                g = self.encoder_model.essen_layer(g)

            x.append(g)
            y.append(data.y)

        x = th.cat(x, dim=0).cpu().numpy()
        y = th.cat(y, dim=0).cpu().numpy()

        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        accuracies = []
        maf1s = []
        mif1s = []
        for train_idx, test_idx in kf.split(x, y):

            split = {'train': train_idx, 'test': test_idx}
            result = SVMEvaluator(linear=True).evaluate(x, y, split)
            accuracies.append(result["accuracy"])
            maf1s.append(result["macro_f1"])
            mif1s.append(result["micro_f1"])

        results = {'micro_f1': np.mean(mif1s), 'macro_f1': np.mean(maf1s), 'accuracy': np.mean(accuracies)}

        return results
