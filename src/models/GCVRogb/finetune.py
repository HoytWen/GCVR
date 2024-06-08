from time import time
import dgl
import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
from utils.scheduler import CosineDecayScheduler

class FineTuner():

    def __init__(self, encoder_model, evaluator, train_loader, val_loader, test_loader, cf):
        self.__dict__.update(cf.__dict__)
        self.encoder_model = encoder_model
        self.evaluator = evaluator
        self.eval_metric = evaluator.eval_metric
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        base_classifier = SVC(kernel='linear',probability=True)
        self.classifier = make_pipeline(
            StandardScaler(),
            base_classifier,)
        self.mlp = nn.Linear(cf.n_hidden*cf.n_layer, 1)
        self.sigmoid = nn.Sigmoid()
        self.mlp = self.mlp.to(self.device)
        self.sigmoid = self.sigmoid.to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.encoder_model.parameters()) + list(self.mlp.parameters()), 
            lr=cf.lr, weight_decay=cf.weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()

    def finetune(self,):
        self.optimizer.zero_grad()
        train_losses, train_auc = 0, 0
        probs = None
        ys = None
        for step, data in enumerate(self.train_loader):
            data = data.to(self.device)

            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=self.device)  

            # Get Embedding
            g = self.encoder_model(data.x, data.edge_index, data.edge_attr, data.batch, is_ft=True)
            g = self.encoder_model.essen_layer(g)

            # Predict
            logits = self.mlp(g)

            # Backprop
            train_loss = self.criterion(logits, data.y.float())
            train_loss.backward()
            train_losses += train_loss.detach().item()
            if (step+1) % self.accumulation_step == 0:  ## What if the total step can not divided by accumulation_step
                if self.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.encoder_model.parameters(), self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                # Evaluate on train
                with torch.no_grad():
                    train_probs = self.sigmoid(logits)
                    if probs == None:
                        probs = train_probs
                        ys = data.y
                    else:
                        probs = torch.cat((probs, train_probs), axis=0)
                        ys = torch.cat((ys, data.y), axis=0)
            
        with torch.no_grad():
            train_score = self.scorer(ys, probs)
            train_auc = train_score
                    
            
        return {"train_loss": train_losses, "train_auc": train_auc}
                

    def scorer(self, y_true, y_raw):
        input_dict = {"y_true": y_true, "y_pred": y_raw}
        score = self.evaluator.eval(input_dict)[self.eval_metric]
        return score

    @torch.no_grad()
    def evaluate(self, loader_name):
        if loader_name == 'train':
            dataloader = self.train_loader
        elif loader_name == 'val':
            dataloader = self.val_loader
        else:
            dataloader = self.test_loader
        all_probs = None
        ys = None
        for step, data in enumerate(dataloader):
            data = data.to(self.device)
            g = self.encoder_model(data.x, data.edge_index, data.edge_attr, data.batch, is_ft=True)
            g = self.encoder_model.essen_layer(g)
            logits = self.mlp(g)
            probs = self.sigmoid(logits)
            if all_probs == None:
                all_probs = probs
                ys = data.y
            else:
                all_probs = torch.cat((all_probs, probs), axis=0)
                ys = torch.cat((ys, data.y), axis=0)
        
        score = self.scorer(ys.float(), all_probs)
        return score
    
    @torch.no_grad()
    def get_embedding(self, dataloader):
        x, y = [], []
        for data in dataloader:
            data = data.to(self.device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=self.device)

            # Get embedding
            g, _, _, _ = self.encoder_model(data.x, data.edge_index, data.edge_attr, data.batch)
            g = self.encoder_model.essen_layer(g)
        
            x.append(g)
            y.append(data.y)

        x = torch.cat(x, dim=0).cpu()
        y = torch.cat(y, dim=0).cpu()

        return x, y

class MLP(nn.Module):
    """Support Vector Machine"""

    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        z = self.linear(x)
        return z