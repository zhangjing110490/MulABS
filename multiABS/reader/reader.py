import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import deque
import copy
import os
from ..utils import get_varlen_column, save_pkl, load_pkl
from ..inputs import SparseFeat, VarLenSparseFeat, VectorFeat
from ..nlp_utils import get_topic_from_LDA


class Reader(object):
    def __init__(self, config):
        self.config = config
        self.data_dir = os.path.join('ml-1m')
        # feature dictionary: 'sparse' for binary feature, 'varlen' for multi-level feature,
        # 'vector' for fixed embedding
        self.feat_dic = {'sparse': {}, 'varlen': {}, 'vector': []}
        self.temp_data_dir = "./data/"

    def _load_data(self):
        # user info
        user_header = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
        df_user = pd.read_csv(os.path.join(self.data_dir, 'users.dat'), sep='::', names=user_header, encoding='ISO-8859-1')
        # item and meta data
        item_header = ['item_id', 'title', 'genres']
        df_item = pd.read_csv(os.path.join(self.data_dir, 'movies.dat'), sep='::', names=item_header, encoding='ISO-8859-1')
        df_item.drop(columns=['title'], axis=1, inplace=True)
        # meta data
        meta_header = ['movieId', 'title', 'release_date', 'rating', 'overview']
        df_meta = pd.read_csv(os.path.join(self.data_dir, 'movie_data.csv'))[meta_header]
        df_meta.rename(columns={'movieId': 'item_id', 'rating': 'avg_score'}, inplace=True)
        df_item = df_item.merge(df_meta, on='item_id', how='left')
        # behavior
        bhv_header = ['user_id', 'item_id', 'rating', 'timestamp']
        df_bhv = pd.read_csv(os.path.join(self.data_dir, 'ratings.dat'), sep='::', names=bhv_header)
        # bind data to object
        self.df_user = df_user
        self.df_item = df_item
        self.df_bhv = df_bhv

    def _generate_small_dataset(self):
        def frac_sample(grouped_data, n_samples, item_set=None, ban_items=None):
            if (item_set is not None) and (ban_items is not None):
                keep_flag = grouped_data['item_id'].apply(lambda x: (x in item_set) and (x not in ban_items))
                grouped_data = grouped_data[keep_flag]
            if len(grouped_data) == 0:
                return None
            sampled = grouped_data.sample(n_samples, replace=(grouped_data.shape[0] < n_samples),
                                          random_state=self.config.random_seed)
            return sampled

        # select most frequent movies and remove those with ratings=3
        selected_movies = self.df_bhv['item_id'].value_counts(sort=True, ascending=False) \
                              .index.tolist()[0:self.config.n_movies]
        self.df_item = self.df_item[self.df_item['item_id'].apply(lambda x: x in selected_movies)].reset_index(
            drop=True)
        self.df_bhv = self.df_bhv[self.df_bhv['item_id'].apply(lambda x: x in selected_movies)]
        self.df_bhv = self.df_bhv[self.df_bhv['rating'] != 3]

        # build train dataset, including positive and negative dataset
        pos_bhv = self.df_bhv[self.df_bhv['rating'] > 3]
        neg_bhv = self.df_bhv[self.df_bhv['rating'] < 3]
        pos_train = pos_bhv.groupby("user_id").apply(lambda x: frac_sample(x, self.config.n_pos_train))
        neg_train = neg_bhv.groupby("user_id").apply(lambda x: frac_sample(x, self.config.n_neg_train))

        # build test dataset, including positive and negative dataset
        item_pool = set(pos_train['item_id'].tolist() + neg_train['item_id'].tolist())
        pos_test = pos_bhv.groupby("user_id").apply(lambda x: frac_sample(x, self.config.n_pos_test, item_pool,
                                                                          pos_train[pos_train['user_id'] == x.iloc[0][
                                                                              'user_id']]['item_id'].tolist()))
        neg_test = neg_bhv.groupby("user_id").apply(lambda x: frac_sample(x, self.config.n_neg_test, item_pool,
                                                                          neg_train[neg_train['user_id'] == x.iloc[0][
                                                                              'user_id']]['item_id'].tolist()))
        # combine positive and negative cases
        df_train = pd.concat([pos_train, neg_train], axis=0).reset_index(drop=True)
        df_test = pd.concat([pos_test, neg_test], axis=0).reset_index(drop=True)
        self.df_bhv = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
        self.train_index = len(df_train)

    def _process_features(self):
        """discretize continuous features, use LDA to process text features"""
        # user
        self.df_user['zip_code'] = self.df_user['zip_code'].apply(lambda x: x[:1])
        self.df_user['age'] = pd.cut(self.df_user['age'],
                                     [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                     labels=['0-10', '10-20', '20-30', '30-40',
                                             '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])
        for feat in ['gender', 'age', 'occupation', 'zip_code']:
            self.__register_feature('sparse', feat, self.df_user)
        # item
        self.df_item['avg_score'] = pd.cut(self.df_item['avg_score'],
                                           [0, 4, 5, 6, 7, 8, 10],
                                           labels=['0-4', '4-5', '5-6', '6-7', '7-8', '8-10'])
        self.df_item['release_date'].fillna(value='xx xx 1995', inplace=True)
        self.df_item['release_date'] = self.df_item['release_date'].apply(lambda x: int(str(x).split(' ')[-1]))
        self.df_item['release_date'] = pd.cut(self.df_item['release_date'],
                                              [1800, 1940, 1960, 1980, 1990, 1995, 2000, 2010, 2020, 2030],
                                              labels=['0-40', '40-60', '60-80', '80-90',
                                                      '90-95', '95-00', '00-10', '10-20', '20-30'])
        self.df_item['title'] = self.df_item['title'].apply(lambda x: str(x) + ' ')
        self.df_item['info'] = self.df_item['title'] + self.df_item['overview']
        self.df_item['info'], _ = get_topic_from_LDA(self.df_item['info'],
                                                     model_dir=os.path.join(os.getcwd(), "saved_models"),
                                                     n_topics=self.config.n_lda_topics)

        self.__register_feature('sparse', 'item_id', self.df_item)
        self.__register_feature('varlen', 'genres', self.df_item)
        self.__register_feature('sparse', 'release_date', self.df_item)
        self.__register_feature('vector', 'info', self.df_item)
        self.oov_item = len(self.feat_dic['sparse']['item_id'].classes_)
        # bhv
        self.df_bhv['timestamp'] = self.df_bhv['timestamp'].apply(lambda x:
                                                                  datetime(1970, 1, 1) +
                                                                  timedelta(seconds=x))
        self.df_bhv['item_id'] = self.feat_dic['sparse']['item_id'].transform(self.df_bhv['item_id'])
        self.df = self.df_bhv.merge(self.df_user, on='user_id', how='left').merge(self.df_item, on='item_id',
                                                                                  how='left')
        self.df_train = self.df.iloc[0:self.train_index].reset_index(drop=True)
        self.df_test = self.df.iloc[self.train_index:].reset_index(drop=True)
        print(f"{self.df.shape[0]} data loaded")
        print(f"{self.df_train.shape[0]} training data loaded")
        print(f"{self.df_test.shape[0]} testing data loaded")

    def __register_feature(self, type_, feat, df=None):
        assert type_ in ['sparse', 'varlen', 'vector']
        if type_ == 'sparse':
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])
            self.feat_dic['sparse'][feat] = copy.deepcopy(lbe)
        elif type_ == 'varlen':
            get_varlen_column(df, feat, self.feat_dic['varlen'])
        elif type_ == 'vector':
            self.feat_dic['vector'].append(feat)

    def _get_embed_features(self):
        """bind features with corresponding information like num_classes, embedding dimension..."""
        sparse_embed_features = [SparseFeat(feat, len(self.feat_dic['sparse'][feat].classes_),
                                            self.config.embedding_dim)
                                 for feat in self.feat_dic['sparse'].keys() if feat != 'item_id']

        varlen_keys = list(self.feat_dic['varlen'].keys())
        if not self.config.item_embed:
            varlen_keys.remove('hist')
        varlen_embed_features = [VarLenSparseFeat(SparseFeat(feat,
                                                             vocab_size=self.feat_dic['varlen'][feat]['vocab_size'] + 1,
                                                             embedding_dim=self.config.embedding_dim),
                                                  maxlen=self.feat_dic['varlen'][feat]['max_len'],
                                                  padding_id=self.feat_dic['varlen'][feat]['padding_id'],
                                                  combiner='mean')
                                 for feat in varlen_keys]
        vector_features = [VectorFeat(feat, self.config.n_lda_topics) for feat in self.feat_dic['vector']]

        if self.config.item_embed:
            sparse_embed_features.append(SparseFeat('item_id',
                                                    len(self.feat_dic['sparse']['item_id'].classes_) + 1,
                                                    self.config.embedding_dim))

        print('sparse feature: {0}, var feature: {1}, vector feature: {2}'.format
              (len(sparse_embed_features), len(varlen_embed_features), len(vector_features)))
        embed_features = sparse_embed_features + varlen_embed_features + vector_features
        return embed_features

    def _generate_history_list(self, mode='ordered'):
        """generate reading history based on timestamp for each user"""
        history = []
        self.df_train = self.df_train.sort_values(['user_id', 'timestamp'])
        self.df_test = self.df_test.sort_values(['user_id', 'timestamp'])
        hist_dic = {}
        for uid, hist in tqdm(self.df_train.groupby('user_id')):
            item_list = hist['item_id'].tolist()
            rating_list = hist['rating'].tolist()
            if mode == 'ordered':
                curr = deque(maxlen=self.config.max_history)
                curr.extend([self.oov_item] * self.config.max_history)
                curr_rating = deque(maxlen=self.config.max_history)
                curr_rating.extend([3] * self.config.max_history)
                history.append(list(curr) + list(curr_rating))

                for i in range(0, len(item_list) - 1):
                    curr.append(item_list[i])
                    curr_rating.append(rating_list[i])
                    history.append(list(curr) + list(curr_rating))
                curr.append(item_list[-1])
                curr_rating.append(rating_list[-1])
                hist_dic[uid] = list(curr) + list(curr_rating)
            else:
                raise ValueError('mode {0} is not supported in generating history'.format(mode))

        self.df_train['hist'] = history
        self.df_test['hist'] = self.df_test['user_id'].apply(lambda x: hist_dic[x])
        self.feat_dic['varlen']['hist'] = {'vocab_size': self.oov_item,
                                           'padding_id': self.oov_item,
                                           'max_len': self.config.max_history}


class FMReader(Reader):
    def __init__(self, config):
        super(FMReader, self).__init__(config)
        if load_pkl(os.path.join(self.temp_data_dir, 'fmdata')):
            '''if data file exists, load it'''
            fm_data = load_pkl(os.path.join(self.temp_data_dir, 'fmdata'))
            self.df_train = fm_data['train']
            self.df_test = fm_data['test']
            self.oov_item = fm_data['oov_item']
            self.df_item = fm_data['item_id']
            self.feat_dic = fm_data['feat_dic']
            self.df = fm_data['df']
            print(f"read data from {self.temp_data_dir}")
        else:
            '''otherwise, generate data from scratch'''
            self._load_data()
            self._generate_small_dataset()
            self._process_features()
            self._generate_history_list(mode='ordered')
            save_pkl({'train': self.df_train, 'test': self.df_test, "oov_item": self.oov_item,
                      'item_id': self.df_item, "feat_dic": self.feat_dic, 'df': self.df},
                       os.path.join(self.temp_data_dir, 'fmdata'))
        self.embed_features = self._get_embed_features()

    def get_model_set(self, mode):
        """bind all features into the final train/test set."""
        data_path = self.config.train_set if mode == 'train' else self.config.test_set
        model_set = load_pkl(data_path)
        if model_set:
            return model_set['input'], model_set['output']

        df = self.df_train if mode == 'train' else self.df_test
        model_input = {name: df[name] for name in self.feat_dic['sparse']}
        for name in self.feat_dic['varlen']:
            model_input[name] = np.stack(df[name].values)
        for name in self.feat_dic['vector']:
            model_input[name] = np.stack(df[name].values)

        # add user_id for calculation of precision K
        model_input['user_id'] = df['user_id']

        df['rating'] = df['rating'].apply(lambda x: (x - 1) if x < 3 else (x - 2))
        model_output = df['rating'].values
        save_pkl({'input': model_input, 'output': model_output}, data_path)
        return model_input, model_output


class CTRReader(Reader):
    def __init__(self, config):
        super(CTRReader, self).__init__(config)
        if load_pkl(os.path.join(self.temp_data_dir, 'fmdata')):
            fm_data = load_pkl(os.path.join(self.temp_data_dir, 'fmdata'))
            self.df_train = fm_data['train']
            self.df_test = fm_data['test']
            self.oov_item = fm_data['oov_item']
            self.df_item = fm_data['item_id']
            self.feat_dic = fm_data['feat_dic']
            self.df = fm_data['df']
            print(f"read data from {self.temp_data_dir}")
        else:
            self._load_data()
            self._generate_small_dataset()
            self._process_features()
            self._generate_history_list(mode='ordered')
            save_pkl({'train': self.df_train, 'test': self.df_test, "oov_item": self.oov_item,
                      'item_id': self.df_item, "feat_dic": self.feat_dic, 'df': self.df},
                     os.path.join(self.temp_data_dir, 'fmdata'))
        self.embed_features = self._get_embed_features()

    def get_model_set(self, mode):
        """bind all features into the final train/test set."""
        data_path = self.config.train_set if mode == 'train' else self.config.test_set
        model_set = load_pkl(data_path)
        if model_set:
            return model_set['input'], model_set['output']

        df = self.df_train if mode == 'train' else self.df_test
        model_input = {name: df[name] for name in self.feat_dic['sparse']}
        for name in self.feat_dic['varlen']:
            model_input[name] = np.stack(df[name].values)
        for name in self.feat_dic['vector']:
            model_input[name] = np.stack(df[name].values)

        # add user_id for calculation of precision K
        model_input['user_id'] = df['user_id']

        df['rating'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)
        model_output = df['rating'].values
        save_pkl({'input': model_input, 'output': model_output}, data_path)
        return model_input, model_output
