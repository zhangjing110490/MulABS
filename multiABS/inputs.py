from collections import namedtuple, OrderedDict
import torch
import torch.nn as nn


class VectorFeat(namedtuple('VectorFeat',
                            ['name', 'embedding_size'])):
    def __new__(cls, name, embedding_size):
        return super(VectorFeat, cls).__new__(cls, name, embedding_size)


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocab_size', 'dtype', 'embedding_name'])):
    def __new__(cls, name, vocab_size, embedding_dim=4, dtype='int32', embedding_name=None):
        if embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, vocab_size, dtype, embedding_name)


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'padding_id', 'combiner', 'length_name'])):
    def __new__(cls, sparsefeat, maxlen, padding_id, combiner='mean', length_name=None):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, padding_id, combiner, length_name)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocab_size(self):
        return self.sparsefeat.vocab_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name


class SequencePoolingLayer(nn.Module):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of two  tensor [seq_value,seq_len]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.

    """

    def __init__(self, mode='mean', supports_masking=False, device='cpu'):

        super(SequencePoolingLayer, self).__init__()
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError('parameter mode should in [sum, mean, max]')
        self.supports_masking = supports_masking
        self.mode = mode
        self.device = device
        self.eps = torch.FloatTensor([1e-8]).to(device)
        self.to(device)

    def _sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        # Returns a mask tensor representing the first N positions of each cell.
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        mask.type(dtype)
        return mask

    def forward(self, seq_value_len_list):
        if self.supports_masking:
            uiseq_embed_list, mask = seq_value_len_list
            mask = mask.float()
            user_behavior_length = torch.sum(mask, dim=-1, keepdim=True)
            mask = mask.unsqueeze(2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list
            mask = self._sequence_mask(user_behavior_length, maxlen=uiseq_embed_list.shape[1],
                                       dtype=torch.float32)
            mask = torch.transpose(mask, 1, 2)

        embedding_size = uiseq_embed_list.shape[-1]

        mask = torch.repeat_interleave(mask, embedding_size, dim=2)

        if self.mode == 'max':
            hist = uiseq_embed_list - (1 - mask) * 1e9
            hist = torch.max(hist, dim=1, keepdim=True)[0]
            return hist
        hist = uiseq_embed_list * mask.float()
        hist = torch.sum(hist, dim=1, keepdim=False)

        if self.mode == 'mean':
            self.eps = self.eps.to(user_behavior_length.device)
            hist = torch.div(hist, user_behavior_length.type(torch.float32) + self.eps)

        hist = torch.unsqueeze(hist, dim=1)
        return hist


def create_embedding_matrix(embed_features, embed_dim, init_std=0.01, sparse=False, device='cpu'):
    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(
            feat.vocab_size, embed_dim, sparse=sparse,
            padding_idx=(feat.padding_id if isinstance(feat, VarLenSparseFeat) else None))
            for feat in embed_features}
    )
    for tensor in embedding_dict.values():
        nn.init.kaiming_normal(tensor.weight, mode='fan_in')

    return embedding_dict.to(device)


def build_input_features(embed_features):
    features = OrderedDict()

    start = 0
    for feat in embed_features:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, VarLenSparseFeat):
            if feat.name == 'hist':
                features[feat_name] = (start, start + 2 * feat.maxlen)
                start += 2 * feat.maxlen
            else:
                features[feat_name] = (start, start + feat.maxlen)
                start += feat.maxlen

            # in case the length is stored
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = (start, start + 1)
                start += 1
        elif isinstance(feat, VectorFeat):
            features[feat_name] = (start, start + feat.embedding_size)
            start += feat.embedding_size
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features


def get_varlen_pooling_list(embedding_dict, features, feature_index, varlen_sparse_feature_columns, device):
    varlen_sparse_embedding_list = []

    for feat in varlen_sparse_feature_columns:
        embed = embedding_dict[feat.embedding_name] if feat.embedding_name != 'hist' else embedding_dict['item_id']
        start_id = feature_index[feat.name][0]
        end_id = feature_index[feat.name][1]
        mid_id = start_id + (end_id - start_id) // 2

        if feat.name == 'hist':
            end_id = mid_id

        seq_emb = embed(features[:, start_id:end_id].long())
        if feat.length_name is None:
            seq_mask = features[:, start_id:end_id].long() != feat.padding_id
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=True, device=device)(
                [seq_emb, seq_mask])
        else:
            seq_length = features[:,
                         feature_index[feat.length_name][0]:feature_index[feat.length_name][1]].long()
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=False, device=device)(
                [seq_emb, seq_length])
        varlen_sparse_embedding_list.append(emb)
    return varlen_sparse_embedding_list