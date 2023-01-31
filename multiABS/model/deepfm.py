import torch
import torch.nn as nn
from .basemodel import BaseModel
from ..inputs import SparseFeat, VarLenSparseFeat, get_varlen_pooling_list, VectorFeat
from .layers import FM, DNN


class DeepFM(BaseModel):
    """DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"""
    def __init__(self, config, embed_features, embedding_dict, device='gpu', task='binary'):
        super(DeepFM, self).__init__(config, embed_features, embedding_dict, device=device, task=task)
        self.config = config
        self.oov_item = list(filter(lambda x: x.name == 'hist', embed_features))[0].padding_id

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), embed_features)) if len(embed_features) else []
        self.vector_feature_columns = list(
            filter(lambda x: isinstance(x, VectorFeat), embed_features)) if len(embed_features) else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), embed_features)) if len(
            embed_features) else []
        n_features = len(embed_features) * config.embedding_dim

        self.transform = nn.Linear(config.n_lda_topics, config.embedding_dim, bias=True).to(device)
        self.fm = FM()

        self.dnn = DNN(n_features, config.dnn_hidden_units, activation=config.activation, 
                       l2_reg=config.l2_reg, dropout_rate=config.dnn_dropout, use_bn=config.dnn_use_bn,
                       init_std=config.init_std, device=device)
        self.dnn_linear = nn.Linear(config.dnn_hidden_units[-1], 1, bias=False).to(device)

        torch.nn.init.xavier_uniform_(self.dnn_linear.weight)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()),
            l2=config.l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=config.l2_reg)

        self.to(device)

    def forward(self, X):
        # [feat_num, batch_size, each_feature_num=1, embedding_dim]
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        varlen_sparse_embedding_list = get_varlen_pooling_list(self.embedding_dict, X, self.feature_index,
                                                               self.varlen_sparse_feature_columns, self.device)
        embedding_list = sparse_embedding_list + varlen_sparse_embedding_list
        if len(self.vector_feature_columns) > 0:
            additional_embedding = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]]
                                    for feat in self.vector_feature_columns]
            additional_embedding = [self.transform(embedding).unsqueeze(dim=1)
                                    for embedding in additional_embedding]
            embedding_list += additional_embedding

        net_input = torch.cat(embedding_list, dim=1)
        dnn_input = torch.flatten(net_input, start_dim=1)
        logit = self.fm(net_input)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        logit += dnn_logit

        y_pred = self.out(logit)
        return y_pred

