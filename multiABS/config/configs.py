import os
from ..utils import get_logger


class Config(object):
    def __init__(self):
        # parameters for compact dataset
        self.n_lda_topics = 10
        self.n_movies = 1000
        self.n_pos_train = 3
        self.n_neg_train = 2
        self.n_pos_test = 10
        self.n_neg_test = 10
        self.max_history = self.n_pos_train + self.n_neg_train
        # seed
        self.random_seed = 2021
        # training parameters
        self.learning_rate = 0.0001
        self.embedding_dim = 4
        self.lr_gamma = 0.95
        self.accumulation_steps = 1
        self.flood_b = 0.
        self.l2_reg = 0.1
        self.l2_reg_linear = 0.1
        self.l2_reg_dnn = 0.1
        self.l2_reg_embedding = 0.1
        self.dnn_dropout = 0.1
        self.dnn_hidden_units = (32, 16, 8)
        # default settings for log and models
        self.logger = get_logger('info', os.path.join('logs', 'ctr'))
        self.log = 'ctr'
        self.load_model = False
        self.FMembed = False
        self.item_embed = False
        self.add_item_scores = False
        self.add_din = False

    def compile(self, params):
        self.learning_rate = params.lr
        self.embedding_dim = params.embedding_dim
        self.lr_gamma = params.lr_gamma
        self.accumulation_steps = params.acc_steps
        self.l2_reg = params.reg
        self.l2_reg_linear = params.reg
        self.l2_reg_dnn = params.reg
        self.l2_reg_embedding = params.reg
        self.dnn_dropout = params.drop
        self.dnn_hidden_units = eval(params.dnn)
        self.logger = get_logger('info', os.path.join('logs', self.model_name + params.log))
        self.log = self.model_name + params.log
        self.load_model = params.load_model
        self.FMembed = params.FMembed
        self.item_embed = params.item_embed
        self.add_item_scores = params.add_item_scores
        self.add_din = params.add_din

    @staticmethod
    def save_model_path(model_name):
        return os.path.join('saved_dict', model_name + '.ckpt')


class FMConfig(Config):
    def __init__(self):
        super(FMConfig, self).__init__()
        # Folder setting
        self.model_name = 'FM'
        self.save_path = self.save_model_path(self.model_name)
        self.train_set = os.path.join('data', 'duration_train_data')
        self.test_set = os.path.join('data', 'duration_test_data')
        # Default training parameters
        self.learning_rate = 0.1
        self.lr_gamma = 0.95
        self.lr_stop = 0.001
        self.batch_size = 256
        self.num_epochs = 100
        self.require_improvement = 5
        self.l2_reg_linear = 0.01
        self.l2_reg_embedding = 0.01
        self.init_std = 0.001
        # top-k metric
        self.top_k = 5


class CTRConfig(Config):
    def __init__(self, arch):
        super(CTRConfig, self).__init__()
        # Folder setting
        self.model_name = arch
        self.save_path = self.save_model_path(self.model_name)
        # Default training parameters
        self.learning_rate = 0.1
        self.lr_gamma = 0.95
        self.lr_stop = 0.0001
        self.batch_size = 256
        self.num_epochs = 100
        self.require_improvement = 5
        self.accumulation_steps = 4
        self.l2_reg = 0.001
        self.l2_reg_linear = 0.001
        self.l2_reg_dnn = 0.001
        self.l2_reg_embedding = 0.001
        self.init_std = 1.0
        self.neg_num = 1
        # Model params setting
        self.dnn_hidden_units = (48, 32, 16, 8)
        self.dnn_dropout = 0
        self.dnn_use_bn = True
        self.activation = 'prelu'
        self.class_list = ['0', '1']
        self.train_set = os.path.join('data', 'ctr_train_data')
        self.test_set = os.path.join('data', 'ctr_test_data')
        # settings for metric top-k
        self.top_k = 5


class RFConfig(Config):
    def __init__(self):
        super(RFConfig, self).__init__()
        self.model_name = 'RF'
        self.rf_params = {
            'n_estimators': [10, 30, 50, 80],
            'max_depth': [3, 5, 10],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 8],
            'random_state': [self.random_seed]
        }
        self.top_k = 5
        self.train_set = os.path.join('data', 'ctr_train_data')
        self.test_set = os.path.join('data', 'ctr_test_data')


class LRConfig(Config):
    def __init__(self):
        super(RFConfig, self).__init__()
        self.model_name = 'LR'
        self.top_k = 5
        self.train_set = os.path.join('data', 'ctr_train_data')
        self.test_set = os.path.join('data', 'ctr_test_data')
