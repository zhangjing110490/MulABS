from sklearn.metrics import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..inputs import SparseFeat, VarLenSparseFeat
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


class LR:
    def __init__(self, config, embed_features):
        self.est = LogisticRegression(random_state=config.random_seed)
        self.config = config
        self.embed_features = embed_features

    def fit_eval(self, x, y, x_test, y_test, metrics, oh_encoded=True):
        X, X_test = [], []
        for feature in self.embed_features:
            name = feature.name
            if name == 'hist' or name == 'item_id':
                continue
            if len(x[name].shape) == 1:
                x[name] = np.expand_dims(x[name], axis=1)
                x_test[name] = np.expand_dims(x_test[name], axis=1)
            if oh_encoded:
                if isinstance(feature, SparseFeat):
                    enc = OneHotEncoder()
                    x[name] = enc.fit_transform(x[name]).toarray()
                    x_test[name] = enc.transform(x_test[name]).toarray()
                if isinstance(feature, VarLenSparseFeat):
                    max_len = feature.vocab_size
                    tmp = np.zeros((x[name].shape[0], max_len + 1), dtype=int)
                    x_id = x[name][:, :(x[name].shape[1]) // 2] if name == 'hist' else x[name]
                    for i in range(len(tmp)):
                        tmp[i, x_id[i]] = 1
                    x[name] = tmp

                    tmp = np.zeros((x_test[name].shape[0], max_len + 1), dtype=int)
                    x_id = x_test[name][:, :(x_test[name].shape[1]) // 2] if name == 'hist' else x_test[name]
                    for i in range(len(tmp)):
                        tmp[i, x_id[i]] = 1
                    x_test[name] = tmp

            X.append(x[name])
            X_test.append(x_test[name])
        X = np.concatenate(X, axis=-1)
        X_test = np.concatenate(X_test, axis=-1)

        self.est.fit(X, y)
        self.predict(x_test, X_test, y_test, metrics)

    def predict(self, x_test, x, y, metrics):
        result = {}
        eval_str = ""
        metric_func = self._get_metrics(metrics)
        y_preds = self.est.predict_proba(x)
        y_label = self.est.predict(x)
        pos_idx = 0 if (y_label[0] == 1 and y_preds[0, 0] > 0.5) or \
                       (y_label[0] == 0 and y_preds[0, 0] < 0.5) else 1

        for metric, func in metric_func.items():
            if metric == 'precision-k':
                result[metric] = func(x_test, y, y_preds[:, pos_idx])
            else:
                result[metric] = func(y, y_preds[:, pos_idx])
            eval_str += ' - val_' + metric + ": {0: .4f}".format(result[metric])
        self.config.logger.info(eval_str)

    def _get_metrics(self, metrics):
        metrics_ = {}
        for metric in metrics:
            if metric == "auc":
                metrics_[metric] = roc_auc_score
            if metric == 'prauc':
                metrics_[metric] = lambda y_true, y_pred: auc(
                    precision_recall_curve(y_true, y_pred)[1], precision_recall_curve(y_true, y_pred)[0])
            if metric == "accuracy" or metric == "acc":
                metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                    y_true, np.where(y_pred > 0.5, 1, 0))
            if metric == 'recall':
                metrics_[metric] = lambda y_true, y_pred: recall_score(
                    y_true, np.where(y_pred > 0.5, 1, 0))
            if metric == 'mcc':
                metrics_[metric] = lambda y_true, y_pred: matthews_corrcoef(
                    y_true, np.where(y_pred > 0.5, 1, 0))
            if metric == 'precision':
                metrics_[metric] = lambda y_true, y_pred: precision_score(
                    y_true, np.where(y_pred > 0.5, 1, 0))
            if metric == 'f1':
                metrics_[metric] = lambda y_true, y_pred: f1_score(
                    y_true, np.where(y_pred > 0.5, 1, 0))
            if metric == 'precision-k':
                metrics_[metric] = lambda x, y_true, y_pred: self._precision_recall_k(
                    x, y_true, y_pred)
        return metrics_

    def _precision_recall_k(self, x, y_true, y_pred):
        users = x['user_id']
        K = self.config.top_k
        df = pd.DataFrame({'user': users, 'true': y_true, 'pred': y_pred})
        df = df.sort_values(['user', 'pred'], ascending=False)
        precision_k = 0.
        for uid, preds in tqdm(df.groupby('user')):
            precision_k += precision_score(preds['true'][0:K], np.where(preds['pred'][0:K] > 0.5, 1, 0))
        precision_k /= df['user'].nunique()
        return precision_k