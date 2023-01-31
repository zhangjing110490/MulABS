import torch
import torch.nn as nn
from .basemodel import BaseModel
from .layers import InnerProductLayer
from ..inputs import SparseFeat, VarLenSparseFeat, get_varlen_pooling_list, VectorFeat


class FM(BaseModel):
    """implement Factorization Machine to predict the reading time for each reading record"""
    def __init__(self, config, embed_features, embedding_dict, n_labels,
                 device='cpu', task='multiclass'):
        super(FM, self).__init__(config, embed_features, embedding_dict, device=device, task=task)
              
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), embed_features)) if len(embed_features) else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: (isinstance(x, VarLenSparseFeat) and (x.name != 'hist')), embed_features)) \
            if len(embed_features) else []
        self.vector_feature_columns = list(
            filter(lambda x: isinstance(x, VectorFeat), embed_features)) if len(embed_features) else []
        self.innerproduct = InnerProductLayer(device=device)
        
        self.transform = nn.Linear(config.n_lda_topics, config.embedding_dim, bias=True).to(device)
        n_feats = len(self.sparse_feature_columns) + len(self.varlen_sparse_feature_columns) + \
                  len(self.vector_feature_columns)
        self.fc = nn.Linear(1 + n_feats - len(self.vector_feature_columns), n_labels, bias=True).to(device)
            
    def forward(self, X):
        # [feat_num, batch_size, 1, embedding_dim]
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
  
        linear_logit = self.linear_model(X)
        product_logit = self.innerproduct(embedding_list)
        fc_input = torch.cat([linear_logit, product_logit], dim=1)
        logit = self.fc(fc_input)
        # predicted score is discretized into multi-levels
        y_pred = self.out(logit)
        return y_pred
