import torch
import torch.nn.functional as F
from keras_preprocessing.sequence import pad_sequences
import random
import pandas as pd
import numpy as np
import pickle
import os
import logging


def split(x):
    key2index = {}
    key_ans = x.split("|")
    for key in key_ans:
        if key not in key2index:
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


def get_varlen_column(df, col, dic, maxlen=None):
    features = df[col].values
    features = list(map(lambda x: x.split('|'), features))
    unique_features = set(sum(features, []))
    unique_ids = range(1, len(unique_features) + 1)
    key2index = {k: v for k, v in zip(unique_features, unique_ids)}
    index2key = {v: k for k, v in zip(unique_features, unique_ids)}
    maxlen = max(list(map(len, features))) if maxlen is None else maxlen
    feature_ids = list(map(lambda x: [key2index[item] for item in x], features))
    new_feature_name = col + '_lst'
    df[new_feature_name] = pad_sequences(feature_ids, maxlen=maxlen, padding='post').tolist()
    dic[new_feature_name] = {'vocab_size': len(unique_features),
                             'max_len': maxlen,
                             'padding_id': 0}


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


def get_logger(LEVEL, log_file=None):
    head = '[%(asctime)-15s] [%(levelname)s] %(message)s'
    if LEVEL == 'info':
        logging.basicConfig(level=logging.INFO, format=head)
    elif LEVEL == 'debug':
        logging.basicConfig(level=logging.DEBUG, format=head)
    logger = logging.getLogger()
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
    return logger


def explode(series):
    series = series.apply(lambda x: [x] if not isinstance(x, list) else x)
    series = pd.DataFrame(list(series)).stack().reset_index(
        level=1, drop=True)
    res = pd.get_dummies(series).reset_index().groupby("index").sum()
    return res


def neg_sample(item_lst, lst, num):
    ans_lst = [x for x in item_lst if x not in lst]
    return random.sample(ans_lst, num)


def load_embedding_dict(dim, arch):
    embedding_dict = None
    fp = os.path.join("saved_dict", "embedding" + str(dim) + ".pkl")
    if os.path.exists(fp) and arch != 'FM':
        with open(fp, 'rb') as f:
            embedding_dict = pickle.load(f)
        print("Embedding_loaded")
    else:
        print('No embedding')
    return embedding_dict


'''
save embeddings as .pkl
'''


def save_embedding_dict(embedding_dict, frac, arch):
    if arch != 'FM':
        return
    fp = os.path.join("saved_dict", "embedding" + str(frac) + ".pkl")
    with open(fp, 'wb') as f:
        pickle.dump(embedding_dict, f, pickle.HIGHEST_PROTOCOL)


def slice_arrays(arrays, start=None, stop=None):
    """Slice an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `slice_arrays(x, indices)`

    Arguments:
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.

    Returns:
        A slice of the array(s).

    Raises:
        ValueError: If the value of start is a list and stop is not None.
    """

    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None '
                         'if the value of start '
                         'is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


# alpha = neg_sample_ratio
def focal_loss(pred, target, weight=None, gamma=1, alpha=0.4, reduction='mean', avg_factor=None):
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def save_pkl(file_, to_path):
    to_path += '.pkl'
    with open(to_path, 'wb') as f:
        pickle.dump(file_, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(from_path):
    from_path += '.pkl'
    if os.path.exists(from_path):
        try:
            with open(from_path, 'rb') as f:
                res = pickle.load(f)
            return res
        except:
            pass
    return None


def read_log(from_path):
    import re
    try:
        log = open(from_path)
        model_outcome = {"epoch": [],
                         "loss": [], "eval_loss": [],
                         "micro_acc": [], "val_micro_acc": [],
                         "acc": [], "val_acc": [],
                         "auc": [], "val_auc": [],
                         "recall": [], "val_recall": []
                         }
        model = mode = frac = None
        if re.match(r'.+(?P<model>CTR|FM)frac(?P<frac>\d+)(?P<mode>train|test)(?P<embed>(FMembed)*)', from_path):
            matched = re.match(r'.+(?P<model>CTR|FM)frac(?P<frac>\d+)(?P<mode>train|test)(?P<embed>(FMembed)*)',
                               from_path)
            model = matched.group("model") + matched.group('embed')
            mode = matched.group("mode")
            frac = matched.group("frac")

        for line in log.readlines():
            if re.match(r'Epoch (?P<epoch>\d+)/.+', line):
                epoch = re.match(r'Epoch (?P<epoch>\d+)/.+', line).group("epoch")
                model_outcome["epoch"].append(epoch)
                continue
            if re.match(
                    r'.+loss:  (?P<loss>\S+) - micro_acc:  (?P<micro_acc>\S+) eval_loss:  (?P<eval_loss>\S+) - val_micro_acc:  (?P<val_micro_acc>\S+).+',
                    line):
                matched = re.match(
                    r'.+loss:  (?P<loss>\S+) - micro_acc:  (?P<micro_acc>\S+) eval_loss:  (?P<eval_loss>\S+) - val_micro_acc:  (?P<val_micro_acc>\S+).+',
                    line)
                loss = matched.group("loss")
                eval_loss = matched.group("eval_loss")
                micro_acc = matched.group("micro_acc")
                val_micro_acc = matched.group("val_micro_acc")
                model_outcome["loss"].append(loss)
                model_outcome["eval_loss"].append(eval_loss)
                model_outcome["micro_acc"].append(micro_acc)
                model_outcome["val_micro_acc"].append(val_micro_acc)
                continue
            if re.match(
                    r".+loss:  (?P<loss>\S+) - auc:  (?P<auc>\S+) - acc:  (?P<acc>\S+) - recall:  (?P<recall>\S+) eval_loss:  (?P<eval_loss>\S+) - val_auc:  (?P<val_auc>\S+) - val_acc:  (?P<val_acc>\S+) - val_recall:  (?P<val_recall>\S+).+",
                    line):
                matched = re.match(
                    r".+loss:  (?P<loss>\S+) - auc:  (?P<auc>\S+) - acc:  (?P<acc>\S+) - recall:  (?P<recall>\S+) eval_loss:  (?P<eval_loss>\S+) - val_auc:  (?P<val_auc>\S+) - val_acc:  (?P<val_acc>\S+) - val_recall:  (?P<val_recall>\S+).+",
                    line)
                loss = matched.group("loss")
                auc = matched.group("auc")
                acc = matched.group("acc")
                recall = matched.group("recall")
                eval_loss = matched.group("eval_loss")
                val_auc = matched.group("val_auc")
                val_acc = matched.group("val_acc")
                val_recall = matched.group("val_recall")
                model_outcome["loss"].append(loss)
                model_outcome["auc"].append(auc)
                model_outcome["acc"].append(acc)
                model_outcome["recall"].append(recall)
                model_outcome["eval_loss"].append(eval_loss)
                model_outcome["val_auc"].append(val_auc)
                model_outcome["val_acc"].append(val_acc)
                model_outcome["val_recall"].append(val_recall)

        return model_outcome
    except:
        print("please enter the right file path")
