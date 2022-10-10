import pandas as pd
import numpy as np
import random
import os
import sys
from itertools import product
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import rtdl

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Subset, DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F

# import warnings
# warnings.filterwarnings("ignore")

from myutils import Utils
from baseline.ADSD.model import ADSD_MLP, ADSD_AE, ADSD_VAE
from baseline.ADSD.fit import fit

class adsd():
    def __init__(self, seed:int,
                 model_name='ADSD', architecture='ResNet', loss_name='ADSD',
                 epochs:int=20, batch_size:int=256, act_fun=nn.ReLU(),
                 lr:float=1e-3, mom=0.7, weight_decay:float=1e-2,
                 bw_u:int=1.0, bw_a:int=1.0, plot=False):

        self.seed = seed
        self.utils = Utils()

        # whether to use the gpu device
        if architecture == 'FTTransformer':
            self.device = self.utils.get_device(gpu_specific=True)
        else:
            self.device = self.utils.get_device(gpu_specific=False)

        #hyper-parameters
        self.batch_size = batch_size
        self.act_fun = act_fun
        self.lr = lr
        self.mom = mom
        self.weight_decay = weight_decay

        self.epochs = epochs if architecture not in ['ResNet', 'FTTransformer'] else 50
        self.bw_u = bw_u if architecture != 'FTTransformer' else 0.01
        self.bw_a = bw_a if architecture != 'FTTransformer' else 0.01

        # whether to visualize the training process
        self.plot = plot
        # network architecture
        self.architecture = architecture
        # loss
        self.loss_name = loss_name

    def fit2test(self, data):
        # training set
        X_train = data['X_train']
        y_train = data['y_train']

        # validation set (if necessary)
        if self.architecture in ['ResNet', 'FTTransformer']:
            X_val = torch.from_numpy(X_train.copy()).float()
            y_val = torch.from_numpy(y_train.copy()).float()

        # testing set
        X_test = data['X_test']
        y_test = data['y_test']

        input_size = X_train.shape[1] #input size
        X_test_tensor = torch.from_numpy(X_test).float() # testing set

        # resampling
        X_train, y_train = self.utils.sampler(X_train, y_train, self.batch_size)

        X_train_tensor = torch.from_numpy(X_train).float()
        train_tensor = TensorDataset(torch.from_numpy(X_train).float(), torch.tensor(y_train))
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=False, drop_last=True)

        self.utils.set_seed(self.seed)

        if self.architecture == 'MLP':
            model = ADSD_MLP(input_size=input_size, act_fun=self.act_fun)

        elif self.architecture == 'AE':
            model = ADSD_AE(input_size=input_size, act_fun=self.act_fun)

        elif self.architecture == 'VAE':
            assert self.loss_name == 'Gaussian'
            model = ADSD_VAE(input_size=input_size, act_fun=self.act_fun)

        elif self.architecture == 'ResNet':
            model = rtdl.ResNet.make_baseline(
                d_in=X_train.shape[1],
                d_main=128,
                d_hidden=256,
                dropout_first=0.2,
                dropout_second=0.0,
                n_blocks=2,
                d_out=1)
            model.add_module('reg', nn.BatchNorm1d(num_features=1))

        elif self.architecture == 'FTTransformer':
            model = rtdl.FTTransformer.make_default(
                n_num_features=X_train.shape[1],
                cat_cardinalities=None,
                last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
                d_out=1,
            )
            model.add_module('reg', nn.BatchNorm1d(num_features=1))

        else:
            raise NotImplementedError

        model.to(self.device)

        if self.architecture not in ['ResNet', 'FTTransformer']:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom,
                                        weight_decay=self.weight_decay)  # optimizer
            # training
            fit(train_loader=train_loader,
                model=model, architecture=self.architecture, loss_name=self.loss_name,
                optimizer=optimizer, epochs=self.epochs,
                bw_u=self.bw_u, bw_a=self.bw_a,
                device=self.device, plot=self.plot)

        else:
            if self.architecture == 'FTTransformer':
                optimizer = model.make_default_optimizer()
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom,
                                            weight_decay=self.weight_decay)

            # training
            fit(train_loader=train_loader,
                model=model, architecture=self.architecture, loss_name=self.loss_name,
                X_val=X_val, y_val=y_val, es=True,
                optimizer=optimizer, epochs=self.epochs,
                bw_u=self.bw_u, bw_a=self.bw_a,
                device=self.device, plot=self.plot)

        #evaluating in the testing set
        model.eval()
        with torch.no_grad():
            if self.architecture in ['MLP', 'AE']:
                _, score_test = model(X_test_tensor.to(self.device))
            elif self.architecture == 'VAE':
                _, _, _, score_test = model(X_test_tensor.to(self.device))
            elif self.architecture == 'ResNet':
                score_test = model(X_test_tensor.to(self.device)); score_test = score_test.squeeze()
            elif self.architecture == 'FTTransformer':
                score_test = model(X_test_tensor.to(self.device), x_cat=None); score_test = score_test.squeeze()
            else:
                raise NotImplementedError

            # to cpu device
            score_test = score_test.cpu().numpy()

        result = self.utils.metric(y_true=y_test, y_score=score_test)

        return result