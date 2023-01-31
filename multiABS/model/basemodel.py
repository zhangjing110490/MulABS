import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .layers import PredictionLayer
from ..inputs import create_embedding_matrix, build_input_features, SparseFeat, VarLenSparseFeat, \
    VectorFeat, get_varlen_pooling_list
from ..utils import concat_fun, focal_loss, save_embedding_dict


class Linear(nn.Module):
    def __init__(self, config, embed_features, feature_index, device):
        super(Linear, self).__init__()
        self.config = config
        self.device = device
        self.embed_features = embed_features
        self.feature_index = feature_index
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), embed_features)) if len(embed_features) else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: (isinstance(x, VarLenSparseFeat) and (x.name != 'hist')), embed_features)) \
            if len(embed_features) else []
        linear_embed_features = [x for x in embed_features if (x.name != 'hist' and not isinstance(x, VectorFeat))]
        self.linear_weights = create_embedding_matrix(linear_embed_features,
                                                      1,
                                                      config.init_std,
                                                      sparse=False,
                                                      device=device)
        self.feature_index = feature_index

    def forward(self, X):
        linear_input = [self.linear_weights[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:
                 self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        varlen_embedding_list = get_varlen_pooling_list(self.linear_weights,
                                                        X,
                                                        self.feature_index,
                                                        self.varlen_sparse_feature_columns,
                                                        self.device)
        linear_input += varlen_embedding_list
        return torch.flatten(concat_fun(linear_input), start_dim=1)


class BaseModel(nn.Module):
    """base model for CTR task and FM task"""
    def __init__(self, config, embed_features, embedding_dict, device, task='binary'):
        super(BaseModel, self).__init__()
        self.config = config
        self.train_log_writer = SummaryWriter(log_dir='../logs/fit/' + config.log + '_train')
        self.val_log_writer = SummaryWriter(log_dir='../logs/fit/' + config.log + '_val')
        self.embed_features = embed_features
        self.embed_keys = [x for x in embed_features if (x.name != 'hist' and not isinstance(x, VectorFeat))]
        self.reg_loss = torch.zeros((1,), device=device)
        self.device = device
        if not embedding_dict:
            self.embedding_dict = create_embedding_matrix(self.embed_keys,
                                                          config.embedding_dim,
                                                          config.init_std,
                                                          sparse=False,
                                                          device=device)
        else:
            self.embedding_dict = embedding_dict.to(device)
        self.feature_index = build_input_features(embed_features)
        self.linear_model = Linear(config, embed_features, self.feature_index, device)
        self.regularization_weight = []
        self.add_regularization_weight(self.embedding_dict.parameters(),
                                       l2=config.l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(),
                                       l2=config.l2_reg_linear)
        self.task = task
        self.out = PredictionLayer(task)
        self.to(device)

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(
                            l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def fit(self, x=None, y=None, x_test=None, y_test=None, init_epoch=0):
        self.initial_input = x_test
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
            x_test = [x_test[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:  # one dimension case
                x[i] = np.expand_dims(x[i], axis=1)
                x_test[i] = np.expand_dims(x_test[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1).astype(float)),
            torch.from_numpy(y))

        self.train()
        loss_func = self.loss_func
        optim = self.optim
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.config.lr_gamma)
        train_loader = DataLoader(dataset=train_tensor_data,
                                  shuffle=False,  # turn off shuffle to avoid information leakage
                                  batch_size=self.config.batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // self.config.batch_size + 1
        dev_best_metric = float("-inf")
        last_improve = 0
        # Start Training
        self.config.logger.info("Train on {0} samples, validate on {1} samples, {2} steps per epoch"
                                .format(len(train_tensor_data), len(x_test[0]), steps_per_epoch))
        dev_auc_opt = 0.0

        for epoch in range(init_epoch, self.config.num_epochs):
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            train_result = {}
            model = self.train()
            try:
                with tqdm(enumerate(train_loader)) as t:
                    for batch, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        if self.task == 'multiclass':
                            y = y_train.to(self.device).long()
                        if self.task == 'binary':
                            y = y_train.to(self.device).float()
                        y_pred = model(x).squeeze()

                        loss = loss_func(y_pred, y, reduction='sum')
                        reg_loss = self.get_regularization_loss()

                        loss = loss + reg_loss
                        loss_epoch += loss.item()
                        loss = loss / self.config.accumulation_steps
                        loss.backward()

                        if ((batch + 1) % self.config.accumulation_steps) == 0:
                            # optimizer the net
                            nn.utils.clip_grad_norm_(model.parameters(), 3, norm_type=2)
                            optim.step()  # update parameters of net
                            optim.zero_grad()  # reset gradient
                        for name, metric_fun in self.metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            train_result[name].append(metric_fun(
                                y.reshape(-1, 1).cpu().data.numpy(),
                                y_pred.cpu().data.numpy().astype("float64")))
            except KeyboardInterrupt:
                t.close()
                raise
            if scheduler.get_last_lr()[0] > self.config.lr_stop:
                scheduler.step()
            t.close()

            # Add epoch_logs
            curr_loss = epoch_logs["loss"] = loss_epoch / sample_num
            self.train_log_writer.add_scalar('Loss/train', float(curr_loss), epoch)
            eval_result = self.evaluate(x_test, y_test, self.config.batch_size)

            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch
                self.train_log_writer.add_scalar(name, epoch_logs[name], epoch)
                for name, result in eval_result.items():
                    epoch_logs['val_' + name] = result
                    self.val_log_writer.add_scalar(name, result, epoch)

            epoch_time = int(time.time() - start_time)
            self.config.logger.info('Epoch {0}/{1}'.format(epoch + 1, self.config.num_epochs))

            eval_str = "{0}s - loss: {1: .4f}".format(epoch_time, epoch_logs['loss'])
            for name in self.metrics:
                eval_str += ' - ' + name + ": {0: .4f}".format(epoch_logs[name])
            eval_str += " eval_loss: {0: .4f}".format(epoch_logs['val_loss'])
            for name in self.metrics:
                eval_str += ' - val_' + name + ": {0: .4f}".format(epoch_logs["val_" + name])
            if self.config.model_name != 'FM':
                eval_str += ' - val_' + 'precision_k' + ": {0: .4f}".format(epoch_logs["val_" + 'precision_k'])

            curr_stop_metric = epoch_logs["val_" + self.stop_metric]
            if dev_best_metric < curr_stop_metric:
                dev_best_metric = curr_stop_metric
                dev_auc_opt = curr_stop_metric
                torch.save(model.state_dict(), self.config.save_path)
                save_embedding_dict(model.embedding_dict, self.config.embedding_dim, self.config.model_name)
                last_improve = epoch
                improve = '*'
            else:
                improve = ''
            eval_str += '   ' + improve
            self.config.logger.info(eval_str)
            if epoch - last_improve > self.config.require_improvement:
                self.config.logger.info("LR here {}".format(optim))
                break
        return dev_auc_opt

    def evaluate(self, x, y, batch_size=256):
        pred_ans = self.predict(x, batch_size).squeeze()
        if self.config.model_name == "FM":
            eval_loss = self.loss_func(pred_ans, torch.Tensor(y).to(torch.long).to(self.device))
        else:
            eval_loss = self.loss_func(pred_ans, torch.Tensor(y).to(self.device))
        eval_result = {'loss': eval_loss.item()}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans.detach().cpu().numpy())

        if self.config.model_name != 'FM':
            eval_result['precision_k'] = self.precision_recall_k(y, pred_ans.detach().cpu().numpy())

        return eval_result

    def precision_recall_k(self, y_true, y_pred):
        users = self.initial_input['user_id']
        K = self.config.top_k
        df = pd.DataFrame({'user': users, 'true': y_true, 'pred': y_pred})
        df = df.sort_values(['user', 'pred'], ascending=False)
        precision_k = 0.
        for uid, preds in df.groupby('user'):
            precision_k += precision_score(preds['true'][0:K], np.where(preds['pred'][0:K] > 0.5, 1, 0))
        precision_k /= df['user'].nunique()
        return precision_k

    def compile(self, optimizer, stop_metric, loss=None, metrics=None):
        if stop_metric not in metrics and stop_metric != 'loss':
            raise ValueError(f'Stop metric {stop_metric} should be one of the metrics or loss.')
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)
        self.stop_metric = stop_metric

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=self.config.learning_rate)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters(), lr=self.config.learning_rate)
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters(), lr=self.config.learning_rate)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            elif loss == 'crossentropy':
                loss_func = F.nll_loss
            elif loss == 'focal':
                loss_func = focal_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True,
                  sample_weight=None, labels=None):
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss

                # metrics for multi-classification
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == 'micro_acc':
                    metrics_[metric] = lambda y_true, y_pred: precision_score(
                        y_true, np.argmax(y_pred, axis=1), average="micro")
                if metric == 'macro_acc':
                    metrics_[metric] = lambda y_true, y_pred: precision_score(
                        y_true, np.argmax(y_pred, axis=1), average="macro")
                if metric == 'multi_acc':
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.argmax(y_pred, axis=1))

                # metrics for binary classification
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

                self.metrics_names.append(metric)
        return metrics_

    def predict(self, x, batch_size=256):
        model = self.eval()
        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1).astype(float)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                y_pred = model(x)
                pred_ans.append(y_pred)
        pred_ans = torch.cat(pred_ans, dim=0)
        return pred_ans
