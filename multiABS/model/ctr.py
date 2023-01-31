import torch
import torch.nn as nn
from .basemodel import BaseModel
from ..inputs import SparseFeat, VarLenSparseFeat, get_varlen_pooling_list, VectorFeat
from .layers import DNN
import torch.nn.functional as F


class CTR(BaseModel):
    def __init__(self, config, embed_features, embedding_dict, device='cpu'):
        super(CTR, self).__init__(config, embed_features, embedding_dict, device=device)
        self.config = config
        self.oov_item = list(filter(lambda x: x.name == 'hist', embed_features))[0].padding_id

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), embed_features)) if len(embed_features) else []
        self.vector_feature_columns = list(
            filter(lambda x: isinstance(x, VectorFeat), embed_features)) if len(embed_features) else []

        if self.config.add_item_scores or self.config.add_din:
            # For DIN or Attention-based scoring (ABS), history feature is not used in main DNN network
            self.varlen_sparse_feature_columns = list(
                filter(lambda x: (isinstance(x, VarLenSparseFeat) and (x.name != 'hist')), embed_features)) \
                if len(embed_features) else []
            if self.config.add_din:
                n_features = (len(embed_features) - 1) * config.embedding_dim + config.n_lda_topics
            else:
                n_features = (len(embed_features) - 2) * config.embedding_dim + config.n_lda_topics
        else:
            self.varlen_sparse_feature_columns = list(
                filter(lambda x: isinstance(x, VarLenSparseFeat), embed_features)) if len(
                embed_features) else []
            n_features = (len(embed_features) - 1) * config.embedding_dim + config.n_lda_topics

        self.dnn = DNN(n_features, config.dnn_hidden_units, activation=config.activation,
                       l2_reg=config.l2_reg, dropout_rate=config.dnn_dropout, use_bn=config.dnn_use_bn,
                       init_std=config.init_std, device=device)
        self.dnn_linear = nn.Linear(config.dnn_hidden_units[-1], 1, bias=False).to(device)

        if self.config.add_item_scores:
            # weights for ABS and DNN parts
            self.w1 = nn.Parameter(torch.ones((1,)))
            self.w2 = nn.Parameter(torch.ones((1,)))

        torch.nn.init.xavier_uniform_(self.dnn_linear.weight)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()),
            l2=config.l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=config.l2_reg)

        self.to(device)

    def forward(self, X):
        # [feat_num, batch_size, 1, embedding_dim]
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long())
            for feat in self.sparse_feature_columns]

        if self.config.add_din:
            hist_emd = self._calculate_attention_history_embedding(self.embedding_dict['item_id'], X, self.feature_index)
            sparse_embedding_list.append(hist_emd)

        varlen_sparse_embedding_list = get_varlen_pooling_list(self.embedding_dict, X, self.feature_index,
                                                               self.varlen_sparse_feature_columns, self.device)

        if len(self.vector_feature_columns) > 0:
            additional_embedding = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]]
                                    for feat in self.vector_feature_columns]
            additional_embedding = torch.cat(additional_embedding, dim=-1)
        embedding_list = sparse_embedding_list + varlen_sparse_embedding_list
        dnn_input = torch.cat(embedding_list, dim=1)
        dnn_input = torch.flatten(dnn_input, start_dim=1)
        dnn_input = torch.cat((dnn_input, additional_embedding), dim=1)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        # Attention-based scoring (ABS)
        if self.config.add_item_scores:
            item_score = self._calculate_history_scores(self.embedding_dict['item_id'], X, self.feature_index)
            y_pred = self.out(self.w1 * dnn_logit + self.w2 * item_score)
        else:
            y_pred = self.out(dnn_logit)

        return y_pred

    def _calculate_history_scores(self, embed, inputs, feature_index):
        """calculate the score based on the similarity between the target item and the history items"""
        # [batch_size, feat_num*embedding_dim]
        start_id = feature_index['hist'][0]
        end_id = feature_index['hist'][1]
        mid_id = start_id + (end_id - start_id) // 2

        # [bs, max_hist_len]
        ratings = inputs[:, mid_id:end_id].unsqueeze(dim=1)
        hist = inputs[:, start_id:mid_id].unsqueeze(dim=1).long()
        # [bs, max_hist_len, hidden]
        hist_emb = embed(inputs[:, start_id:mid_id].long())
        # [bs, 1, hidden]
        item_emb = embed(inputs[:, feature_index['item_id'][0]].long()).unsqueeze(dim=1)
        # attention
        scores = torch.matmul(item_emb, hist_emb.transpose(1, 2))
        scores = scores.masked_fill(hist == self.oov_item, -1e-9)
        scores = torch.mul(F.softmax(scores, dim=-1), ratings)
        total_score = torch.sum(scores, dim=-1)
        return total_score

    def _calculate_attention_history_embedding(self, embed, inputs, feature_index):
        """calculate DIN: Deep Interest Network for Click-Through Rate Prediction"""
        # [batch_size, feat_num*embedding_dim]
        start_id = feature_index['hist'][0]
        end_id = feature_index['hist'][1]
        mid_id = start_id + (end_id - start_id) // 2

        # [bs, max_hist_len]
        hist = inputs[:, start_id:mid_id].unsqueeze(dim=1).long()
        # [bs, max_hist_len, hidden]
        hist_emb = embed(inputs[:, start_id:mid_id].long())
        # [bs, 1, hidden]
        item_emb = embed(inputs[:, feature_index['item_id'][0]].long()).unsqueeze(dim=1)
        # attention
        scores = torch.matmul(item_emb, hist_emb.transpose(1, 2))
        scores = scores.masked_fill(hist == self.oov_item, 0)
        item_embedding = torch.matmul(scores, hist_emb)

        return item_embedding
