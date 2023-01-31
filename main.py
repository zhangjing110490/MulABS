import torch
import argparse
from multiABS.config import get_config
from multiABS.utils import load_embedding_dict, load_pkl
from multiABS.reader import get_reader
from multiABS.model import get_model
import time
from multiABS.seeding import seed_everything
import datetime


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'{DEVICE} is being used.')
    # Argument parser
    parser = argparse.ArgumentParser(description="Multi-task model params parser")
    parser.add_argument('--arch', default='CTR', type=str, help='model type, CTR, FM, LR, DeepFM, RF are supported')
    parser.add_argument("--embedding_dim", default=8, type=int, help='embedding dimension')
    parser.add_argument('--reg', default=0.1, type=float, help='L2 regularization parameter')
    parser.add_argument('--drop', default=0.1, type=float, help='drop out rate')
    parser.add_argument('--acc_steps', default=1, type=int, help='accumulation steps to update the network')
    parser.add_argument('--lr_gamma', default=0.95, type=float, help='gamma used in LR scheduler.')
    parser.add_argument('--dnn', default='(32, 16, 8)', type=str, help='dimension in each layer of DNN')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument("--log", default='ctr', type=str, help='name of log file')
    parser.add_argument("--load_model", action='store_true', help='whether to load model from previous training')
    parser.add_argument("--item_embed", action='store_false',
                        help='whether to use item ID as an extra feature')
    parser.add_argument('--FMembed', action='store_true',
                        help='whether to use pretrained embeddings from FM.')
    parser.add_argument("--add_item_scores", action='store_true',
                        help='whether to add attention-based scoring (ABS) module')
    parser.add_argument("--add_din", action='store_true',
                        help='whether to use Deep Interest Network (DIN)')

    params = parser.parse_args()

    # try different hyper-parameters for model arch except RF/LR
    num_dim = [4, 6, 8, 10, 12, 14] if not (params.arch == 'RF' or params.arch == 'LR') else [0]
    dropout = [0.1*x for x in range(1, 6)] if not (params.arch == 'RF' or params.arch == 'LR') else [0]

    for dim in num_dim:
        best_auc = 0.0
        best_drop = 0.0
        for drop in dropout:
            params.embedding_dim = dim
            params.drop = drop

            config = get_config(params.arch)
            config.compile(params)
            seed_everything(config.random_seed)

            # logging
            config.logger.info(f"Training with dimension {dim} and dropout {drop}")

            # Reader
            reader = get_reader(config)
            train_set = reader.get_model_set('train')
            test_set = reader.get_model_set('test')
            X_train, y_train = train_set[0], train_set[1]
            X_test, y_test = test_set[0], test_set[1]
            config.logger.info(f"Data loaded from {config.train_set}")
            # #_classes
            n_labels = len(set(y_test)) if params.arch == 'FM' else None
            config.logger.info(f'There are {n_labels} classes in total about rating.')

            # load pretrained embeddings from FM
            if params.FMembed and params.arch != 'FM':
                embedding_dict = load_embedding_dict(params.embedding_dim, params.arch)
                if embedding_dict is not None:
                    config.logger.info(f"Embedding loaded")
                else:
                    raise ValueError('No Embeddings loaded')
            else:
                embedding_dict = None
                config.logger.info(f"No Embedding")

            config.logger.info(f"Training start at {time.asctime(time.localtime(time.time()))}")
            config.logger.info(f"there are {len(reader.embed_features)} features")

            # create model
            model = get_model(config, reader.embed_features, embedding_dict, n_labels, DEVICE)
            start = datetime.datetime.now()

            if params.arch == 'RF' or params.arch == 'LR':
                model.fit_eval(X_train, y_train, X_test, y_test,
                               metrics=['auc', 'precision-k', 'prauc', 'acc', 'recall', 'precision', 'f1'])
            else:
                if params.arch == 'FM':
                    model.compile("adam",
                                  "micro_acc",
                                  "crossentropy",
                                  metrics=["micro_acc", "multi_acc"])
                else:
                    model.compile("adam",
                                  "auc",
                                  "focal",
                                  metrics=['auc', 'prauc', 'acc', 'recall', 'precision', 'f1'])

            if params.arch != 'RF' and params.arch != 'LR':
                best_score = model.fit(X_train, y_train, X_test, y_test)
                if best_auc < best_score:
                    best_auc = best_score
                    best_drop = drop
            end = datetime.datetime.now()
            time_cost = (end - start) / 60
            config.logger.info(f'time consumed is {time_cost} minutes')
            config.logger.info(f"AUC is {best_score}")

        config.logger.info(f"Best AUC for {dim} is {best_auc} with drop {best_drop}")
