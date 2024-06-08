from utils.conf_utils import ModelConfig, SimpleObject
from utils.proj_settings import TEMP_PATH, LOG_PATH

class GCVRConfig(ModelConfig):

    def __init__(self, args):
        super(GCVRConfig, self).__init__('GCVRogb')
        # ! Model settings
        self.dataset = args.dataset
        self.lr = 0.001
        self.dropout = 0.5
        self.n_hidden = 128
        self.weight_decay = 5e-4
        self.mlp_layer = 2
        self.n_layer = 3
        self.batch_size = 1024
        self.tau = 0.2
        self.lam_c = 3.0
        self.lam_d = 5.0
        self.lam_a = 0.0
        self.lam_p = 0.0
        self.clip = 0.0
        self.delta = 8e-3
        self.perturb_step_size = 8e-3
        self.perturb_step = 5
        self.test_freq = 10
        self.disen_mode = 'mse_norm'
        self.recs_mode = 'dot'
        self.add_proj = False
        self.use_scheduler = True
        self.intra_negative = False
        self.two_aug = False
        self.train_percentage = 90
        self.split_ratio = 0.1
        self.ft_epochs = 200
        # self.simsiam = False
        # ! Other settings
        self.__dict__.update(args.__dict__)
        self.parse_intermediate_settings()
        self.mkdir()

    def parse_intermediate_settings(self):

        tmp = SimpleObject()
        if self.batch_size > 512:
            assert self.batch_size % 512 == 0
            tmp.accumulation_step = self.batch_size // 512
            tmp.step_batch_size = 512
        else:
            tmp.accumulation_step = 1
            tmp.step_batch_size = self.batch_size

        self.__dict__.update(tmp.__dict__)
        # self._ignored_settings += list(tmp.__dict__.keys())

    @property
    def model_cf_str(self):
        return f"{self.model}_lr{self.lr}_drop{self.dropout}_nl{self.n_layer}_ml{self.mlp_layer}_bsz{self.batch_size}_nh{self.n_hidden}_tau{self.tau}_ap{int(self.add_proj)}_dmode{self.disen_mode}_lamc{self.lam_c}_lamd{self.lam_d}_lama{self.lam_a}_lamp{self.lam_p}_in{int(self.intra_negative)}_clip{self.clip}_us{int(self.use_scheduler)}_rm{self.recs_mode}_taug{int(self.two_aug)}_pss{self.perturb_step_size}_ps{self.perturb_step}_sr{self.split_ratio}"

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.model}/{self.dataset}/{self.f_prefix}S{self.seed}.ckpt"

    @property
    def log_path(self):
        return f"{LOG_PATH}{self.model}/{self.dataset}/{self.f_prefix}/"

    def add_model_specific_args(self, parser):
        parser.add_argument("--lr", type=float, default=self.lr, help='Learning rate')
        return parser
