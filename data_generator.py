import numpy as np
import pandas as pd
import random
import scipy.io
import os
import mat73
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
from sklearn.mixture import GaussianMixture

from copulas.multivariate import VineCopula
from copulas.univariate import GaussianKDE

from myutils import Utils

# to do: leverage the categorical feature in the current datasets
# currently, data generator only supports for generating the binary classification datasets
class DataGenerator():
    def __init__(self, seed:int=42, dataset:str=None, test_size:float=0.3,
                 generate_duplicates=False, n_samples_threshold=1000, show_statistic=False):
        '''
        Only global parameters should be provided in the DataGenerator instantiation

        seed: seed for reproducible experimental results
        dataset: dataset name
        test_size: testing data size
        '''

        self.seed = seed
        self.dataset = dataset
        self.test_size = test_size

        # 当数据量不够时, 是否生成重复样本
        self.generate_duplicates = generate_duplicates
        self.n_samples_threshold = n_samples_threshold

        # 是否展示统计结果
        self.show_statistic = show_statistic

        # myutils function
        self.utils = Utils()

    def generate_realistic_synthetic(self, X, y, realistic_synthetic_mode, alpha:int, percentage:float):
        '''
        Currently, four types of realistic synthetic outliers can be generated:
        1. local outliers: where normal data follows the GMM distribuion, and anomalies follow the GMM distribution with modified covariance
        2. global outliers: where normal data follows the GMM distribuion, and anomalies follow the uniform distribution
        3. dependency outliers: where normal data follows the vine coupula distribution, and anomalies follow the independent distribution captured by GaussianKDE
        4. cluster outliers: where normal data follows the GMM distribuion, and anomalies follow the GMM distribution with modified mean

        :param X: input X
        :param y: input y
        :param realistic_synthetic_mode: the type of generated outliers
        :param alpha: the scaling parameter for controling the generated local and cluster anomalies
        :param percentage: controling the generated global anomalies
        '''

        if realistic_synthetic_mode in ['local', 'cluster', 'dependency', 'global']:
            pass
        else:
            raise NotImplementedError

        # the number of normal data and anomalies
        pts_n = len(np.where(y == 0)[0])
        pts_a = len(np.where(y == 1)[0])

        # only use the normal data to fit the model
        X = X[y == 0]
        y = y[y == 0]

        # generate the synthetic normal data
        if realistic_synthetic_mode in ['local', 'cluster', 'global']:
            # select the best n_components based on the BIC value
            metric_list = []
            n_components_list = list(np.arange(1, 10))

            for n_components in n_components_list:
                gm = GaussianMixture(n_components=n_components, random_state=self.seed).fit(X)
                metric_list.append(gm.bic(X))

            best_n_components = n_components_list[np.argmin(metric_list)]

            # refit based on the best n_components
            gm = GaussianMixture(n_components=best_n_components, random_state=self.seed).fit(X)

            # generate the synthetic normal data
            X_synthetic_normal = gm.sample(pts_n)[0]

        # we found that copula function may occur error in some datasets
        elif realistic_synthetic_mode == 'dependency':
            # sampling the feature since copulas method may spend too long to fit
            if X.shape[1] > 50:
                idx = np.random.choice(np.arange(X.shape[1]), 50, replace=False)
                X = X[:, idx]

            copula = VineCopula('center') # default is the C-vine copula
            copula.fit(pd.DataFrame(X))

            # sample to generate synthetic normal data
            X_synthetic_normal = copula.sample(pts_n).values

        else:
            pass

        # generate the synthetic abnormal data
        if realistic_synthetic_mode == 'local':
            # generate the synthetic anomalies (local outliers)
            gm.covariances_ = alpha * gm.covariances_
            X_synthetic_anomalies = gm.sample(pts_a)[0]

        elif realistic_synthetic_mode == 'cluster':
            # generate the clustering synthetic anomalies
            gm.means_ = alpha * gm.means_
            X_synthetic_anomalies = gm.sample(pts_a)[0]

        elif realistic_synthetic_mode == 'dependency':
            X_synthetic_anomalies = np.zeros((pts_a, X.shape[1]))

            # using the GuassianKDE for generating independent feature
            for i in range(X.shape[1]):
                kde = GaussianKDE()
                kde.fit(X[:, i])
                X_synthetic_anomalies[:, i] = kde.sample(pts_a)

        elif realistic_synthetic_mode == 'global':
            # generate the synthetic anomalies (global outliers)
            X_synthetic_anomalies = []

            for i in range(X_synthetic_normal.shape[1]):
                low = np.min(X_synthetic_normal[:, i]) * (1 + percentage)
                high = np.max(X_synthetic_normal[:, i]) * (1 + percentage)

                X_synthetic_anomalies.append(np.random.uniform(low=low, high=high, size=pts_a))

            X_synthetic_anomalies = np.array(X_synthetic_anomalies).T

        else:
            pass

        X = np.concatenate((X_synthetic_normal, X_synthetic_anomalies), axis=0)
        y = np.append(np.repeat(0, X_synthetic_normal.shape[0]),
                      np.repeat(1, X_synthetic_anomalies.shape[0]))

        return X, y

    def generator(self, la=None, at_least_one_labeled=False,
                  realistic_synthetic_mode=None, alpha:int=5, percentage:float=0.1,):
        '''
        la: labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
        at_least_one_labeled: whether to guarantee at least one labeled anomalies in the training set
        '''

        # set seed for reproducible results
        self.utils.set_seed(self.seed)

        # # transfer different file format to the numpy array
        # if self.dataset in ['annthyroid', 'cardio', 'mammography', 'musk', 'optdigits', 'pendigits',
        #                     'satimage-2', 'speech', 'thyroid', 'vowels', 'cover', 'http', 'letter',
        #                     'mnist', 'satellite', 'shuttle', 'smtp', 'breastw', 'vertebral',
        #                     'wine']:
        #     if self.dataset in ['http', 'smtp']:
        #         data = mat73.loadmat(os.path.join('datasets', self.dataset + '.mat'))
        #     else:
        #         data = scipy.io.loadmat(os.path.join('datasets', self.dataset + '.mat'))
        #     X = data['X']
        #     y = data['y'].squeeze().astype('int64')
        #
        # elif self.dataset in ['Waveform_withoutdupl_v10', 'InternetAds_withoutdupl_norm_19', 'PageBlocks_withoutdupl_09',
        #                       'SpamBase_withoutdupl_40', 'Wilt_withoutdupl_05', 'Cardiotocography_withoutdupl_22',
        #                       'WBC_withoutdupl_v10', 'WDBC_withoutdupl_v10', 'WPBC_withoutdupl_norm',
        #                       'Arrhythmia_withoutdupl_46', 'HeartDisease_withoutdupl_44', 'Hepatitis_withoutdupl_16',
        #                       'Parkinson_withoutdupl_75', 'Pima_withoutdupl_35', 'Stamps_withoutdupl_09']:
        #     data = pd.read_csv(os.path.join('datasets', self.dataset + '.csv'))
        #
        #     data.columns = [_.split("'")[1] for _ in data.columns]
        #     X = data.drop(['outlier', 'id'], axis=1).values
        #     y = [_.split("'")[1] for _ in data['outlier'].values]
        #     y = np.array([0 if _ == 'no' else 1 for _ in y])
        #
        # elif self.dataset in ['ALOI_withoutdupl', 'glass_withoutduplicates_normalized',
        #                       'Ionosphere_withoutdupl_norm', 'Lymphography_withoutdupl_idf']:
        #     data = pd.read_csv(os.path.join('datasets', self.dataset + '.csv'))
        #
        #     X = data.drop(['outlier','id'], axis=1).values
        #     y = np.array([0 if _ == 'no' else 1 for _ in data['outlier'].values])
        #
        # elif self.dataset in ['abalone.diff', 'comm.and.crime.diff', 'concrete.diff', 'fault.diff', 'imgseg.diff',
        #                       'landsat.diff', 'magic.gamma.diff', 'skin.diff', 'yeast.diff']:
        #     data = pd.read_csv(os.path.join('datasets', self.dataset + '.csv'))
        #     X = data.drop(['point.id', 'motherset', 'origin', 'original.label', 'diff.score', 'ground.truth'],
        #                   axis=1).values
        #     y = np.array([0 if _ == 'nominal' else 1 for _ in data['ground.truth'].values])
        #
        # # Credit Card Fraud Detection (CCFD) dataset
        # elif self.dataset == 'CCFD':
        #     data = pd.read_csv(os.path.join('datasets', self.dataset + '.csv'))
        #     X = data.drop(['Time', 'Class'], axis=1)
        #     y = data['Class'].values
        #
        # # Taiwan Bankruptcy Prediction (TBP) dataset
        # elif self.dataset == 'TBP':
        #     data = pd.read_csv(os.path.join('datasets', self.dataset + '.csv'))
        #     X = data.drop(['Flag'], axis=1)
        #     y = data['Flag'].values
        #
        # elif self.dataset in ['amazon', 'yelp', 'imdb'] + \
        #                      ['agnews_' + str(i) for i in range(4)] +\
        #                      ['FashionMNIST_' + str(i) for i in range(10)] +\
        #                      ['CIFAR10_' + str(i) for i in range(10)] +\
        #                      ['SVHN_' + str(i) for i in range(10)]:
        #     data = np.load(os.path.join('datasets_NLP_CV', self.dataset + '.npz'))
        #     X = data['X']
        #     y = data['y']
        #
        # else:
        #     raise NotImplementedError

        data = np.load(os.path.join('datasets', 'new', self.dataset + '.npz'))
        X = data['X']
        y = data['y']

        # to array
        X = np.array(X)
        y = np.array(y)

        # number of labeled anomalies in the original data
        if type(la) == float:
            if at_least_one_labeled:
                n_labeled_anomalies = ceil(sum(y) * (1 - self.test_size) * la)
            else:
                n_labeled_anomalies = int(sum(y) * (1 - self.test_size) * la)
        elif type(la) == int:
            n_labeled_anomalies = la
        else:
            raise NotImplementedError

        ################################
        # 如果数据集大小不足, 生成duplicate样本:
        if len(y) < self.n_samples_threshold and self.generate_duplicates:
            print(f'数据集{self.dataset}正在生成duplicate samples...')
            self.utils.set_seed(self.seed)
            idx_duplicate = np.random.choice(np.arange(len(y)), self.n_samples_threshold, replace=True)
            X = X[idx_duplicate]
            y = y[idx_duplicate]

        # 如果数据集过大, 整体样本抽样
        if len(y) > 10000:
            print(f'数据集{self.dataset}数据量过大, 正在抽样...')
            self.utils.set_seed(self.seed)
            idx_sample = np.random.choice(np.arange(len(y)), 10000, replace=False)
            X = X[idx_sample]
            y = y[idx_sample]

        # whether to generate realistic synthetic outliers
        if realistic_synthetic_mode is not None:
            # we save the generated dependency anomalies, since the Vine Copula could spend too long for generation
            if realistic_synthetic_mode == 'dependency':
                if not os.path.exists('datasets/synthetic'):
                    os.makedirs('datasets/synthetic')

                filepath = 'dependency_anomalies_' + self.dataset + '_' + str(self.seed) + '.npz'
                try:
                    data_dependency = np.load(os.path.join('datasets', 'synthetic', filepath), allow_pickle=True)
                    X = data_dependency['X']; y = data_dependency['y']

                except:
                    raise NotImplementedError
                    # print(f'Generating dependency anomalies...')
                    # X, y = self.generate_realistic_synthetic(X, y,
                    #                                          realistic_synthetic_mode=realistic_synthetic_mode,
                    #                                          alpha=alpha, percentage=percentage)
                    # np.savez_compressed(os.path.join('datasets', 'synthetic', filepath), X=X, y=y)
                    # pass

            else:
                X, y = self.generate_realistic_synthetic(X, y,
                                                         realistic_synthetic_mode=realistic_synthetic_mode,
                                                         alpha=alpha, percentage=percentage)

        ################################
        # show the statistic
        self.utils.data_description(X=X, y=y)

        # spliting the current data to the training set and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=True, stratify=y)

        # minmax scaling
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # idx of normal samples and unlabeled/labeled anomalies
        idx_normal = np.where(y_train == 0)[0]
        idx_anomaly = np.where(y_train == 1)[0]

        assert type(la) == float
        if at_least_one_labeled:
            idx_labeled_anomaly = np.random.choice(idx_anomaly, ceil(la * len(idx_anomaly)), replace=False)
        else:
            idx_labeled_anomaly = np.random.choice(idx_anomaly, int(la * len(idx_anomaly)), replace=False)

        idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
        # unlabel data = normal data + unlabeled anomalies (which is considered as contamination)
        idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)

        del idx_anomaly, idx_unlabeled_anomaly

        # the label of unlabeled data is 0, and that of labeled anomalies is 1
        y_train[idx_unlabeled] = 0
        y_train[idx_labeled_anomaly] = 1

        return {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test,
                'n_labeled_anomalies':n_labeled_anomalies}

