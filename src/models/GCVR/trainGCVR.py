import os.path as osp
import sys
sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
from models.GCVR.config import GCVRConfig
import argparse
from utils import *

import warnings
warnings.filterwarnings('ignore')


@time_logger
def trainGCVR(args):

    exp_init(args.gpu, args.seed, args.log_on)

    import torch as th
    from torch_geometric.datasets import TUDataset
    from models.GCVR.trainer import GCVR_Trainer

    cf = GCVRConfig(args)
    cf.device = th.device("cuda:0" if args.gpu >= 0 and th.cuda.is_available() else "cpu")
    print(cf)

    data = TUDataset(root='data/', name=cf.dataset)
    cf.feat_dim = max(data.num_features, 1)
    cf.n_class = data.num_classes

    trainer = GCVR_Trainer(data=data, cf=cf)
    trainer.run()

    return cf


if __name__ == '__main__':
    if __name__ == "__main__":
        parser = argparse.ArgumentParser("Training settings")
        parser = GCVRConfig.add_exp_setting_args(parser)
        exp_args = parser.parse_known_args()[0]
        parser = GCVRConfig(exp_args).add_model_specific_args(parser)
        args = parser.parse_args()
        trainGCVR(args)
