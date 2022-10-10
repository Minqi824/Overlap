import os
import random

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from myutils import Utils
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution

import rtdl
import delu
import scipy

# 实例化utils
utils = Utils()

# calculate the GaussianKDE with Pytorch
class GaussianKDE(Distribution):  # 已经检验过与sklearn计算结果一致
    def __init__(self, X, bw, lam=1e-4, device=None):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """
        self.X = X
        self.bw = bw
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dims).to(device),
                                      covariance_matrix=torch.eye(self.dims).to(device))
        self.lam = lam

    def sample(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.
        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.
        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X == None:
            X = self.X

        # 注意此处取log当值接近0时会产生正负无穷的数
        # 利用with autograd.detect_anomaly()检测出算法发散的原因在于torch.log变量值接近0,需要探究接近0的原因
        log_probs = torch.log(
            (self.bw ** (-self.dims) *
             torch.exp(self.mvn.log_prob(
                 (X.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n + self.lam)

        return log_probs

    def log_prob(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.
        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """

        X_chunks = self.X.split(1000)
        Y_chunks = Y.split(1000)

        log_prob = 0

        for x in X_chunks:
            for y in Y_chunks:
                log_prob += self.score_samples(y, x).sum(dim=0)

        return log_prob

def loss_overlap(s_u, s_a, seed, bw_u=None, bw_a=None, x_num=1000,
                 plot=False, pro=False, device=None):
    # set seed
    utils.set_seed(seed)

    # remove duplicates
    # s_u = torch.unique(s_u)
    # s_a = torch.unique(s_a)

    if bw_u is None and bw_a is None:
        d = 1  # one-dimension data
        n_u = s_u.size(0)
        n_a = s_a.size(0)

        # Scott's Rule, which requires the data from the normal distribution.
        # This may be inappropriate when the neural network output can be arbitrary distribution
        # bw_u = n_u ** (-1. / (d + 4))
        # bw_a = n_a ** (-1. / (d + 4))

        # Silverman's Rule
        bw_u = (n_u * (d + 2) / 4.) ** (-1. / (d + 4))
        bw_a = (n_a * (d + 2) / 4.) ** (-1. / (d + 4))

    # reshape
    s_u = s_u.reshape(-1, 1)
    s_a = s_a.reshape(-1, 1)

    # estimate the Gaussian KDE
    kde_u = GaussianKDE(X=s_u, bw=bw_u, device=device)
    kde_a = GaussianKDE(X=s_a, bw=bw_a, device=device)

    # the range of x axis
    xmin = torch.min(torch.min(s_u), torch.min(s_a))
    xmax = torch.max(torch.max(s_u), torch.max(s_a))

    dx = 0.2 * (xmax - xmin)
    xmin -= dx
    xmax += dx
    x = torch.linspace(xmin.detach(), xmax.detach(), x_num).to(device)

    # estimated pdf
    kde_u_x = torch.exp(kde_u.score_samples(x.reshape(-1, 1)))
    kde_a_x = torch.exp(kde_a.score_samples(x.reshape(-1, 1)))

    if plot:
        plt.plot(x, kde_u_x.detach(), color='blue')
        plt.plot(x, kde_a_x.detach(), color='red')
        plt.show()

    if pro:
        # find the intersection point (could be multiple points)
        intersection_points_idx = torch.where(torch.diff(torch.sign(kde_a_x - kde_u_x)))[0].cpu()

        # 修改了求trapz的bug, 之前版本算出来可能会产生negative的area问题
        if intersection_points_idx.size(0) >= 1:
            c = x[np.random.choice(intersection_points_idx.numpy(), 1)]

            idx_u = torch.where(x > c)[0]
            idx_a = torch.where(x < c)[0]

            area_u = torch.trapz(kde_u_x[idx_u], x[idx_u])
            area_a = torch.trapz(kde_a_x[idx_a], x[idx_a])
            area = area_u + area_a

        else: # no intersection point
            # area = torch.tensor(10.0, requires_grad=True)
            # raise NotImplementedError

            area_u = torch.trapz(kde_u_x, x)
            area_a = torch.trapz(kde_a_x, x)
            area = (area_u + area_a) / 2

            print(f'没有交点! 重叠面积: {area.item()}')

    else:
        inters_x = torch.min(kde_u_x, kde_a_x)
        area = torch.trapz(inters_x, x)

    return area

# only for ResNet and FTTransformer
def apply_model(x_num, x_cat=None, model=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f'Looks like you are using a custom model: {type(model)}.'
            ' Then you have to implement this branch first.'
        )

@torch.no_grad()
def evaluate(X, y=None, model=None, batch_size=None):
    model.eval()
    score = []
    for batch in delu.iter_batches(X, batch_size):
        score.append(apply_model(batch, model=model))
    score = torch.cat(score).squeeze(1).cpu().numpy()
    score = scipy.special.expit(score)

    # calculate the metric
    if y is not None:
        target = y.cpu().numpy()
        metric = utils.metric(y_true=target, y_score=score)
    else:
        metric = {'aucroc': None, 'aucpr': None}

    return score, metric['aucpr']

def fit(train_loader, model, architecture, optimizer, epochs,
        X_val=None, y_val=None, es=False,
        print_loss=False, device=None, bw_u=None, bw_a=None, plot=False):
    '''
    bw_u: the bandwidth of unlabeled samples
    bw_a: the bandwidth of labeled anomalies
    '''
    if architecture in ['ResNet', 'FTTransformer']:
        progress = delu.ProgressTracker(patience=10)

    for epoch in range(epochs):
        loss_batch = []
        model.train()
        for i, data in enumerate(train_loader):

            X, y = data
            X = X.to(device); y = y.to(device)
            X = Variable(X); y = Variable(y)

            # clear gradient
            model.zero_grad()

            # loss forward
            if architecture in ['MLP', 'AEpro']:
                _, score = model(X)
            elif architecture == 'ResNet':
                score = model(X); score = score.squeeze()
            elif architecture == 'FTTransformer':
                score = model(x_num=X, x_cat=None); score = score.squeeze()
            else:
                raise NotImplementedError

            # 注意由于batchnorm的存在要一起计算score
            score_u = score[torch.where(y == 0)[0]]
            score_a = score[torch.where(y == 1)[0]]

            loss = loss_overlap(s_u=score_u, s_a=score_a, seed=utils.unique(epoch, i),
                                bw_u=bw_u, bw_a=bw_a, pro=True, plot=plot, device=device)

            loss_batch.append(loss.item())

            # loss backward
            loss.backward()
            # parameter update
            optimizer.step()

            if (i % 50 == 0) & print_loss:
                print('[%d/%d] [%d/%d] Loss: %.4f' % (epoch + 1, epochs, i, len(train_loader), loss))

        if architecture in ['ResNet', 'FTTransformer'] and es:
            _, val_metric = evaluate(X=X_val.to(device), y=y_val.to(device), model=model, batch_size=1)
            print(f'Epoch {epoch:03d} | Validation metric: {val_metric:.4f}', end='')
            progress.update((1) * val_metric)
            if progress.success:
                print(' <<< BEST VALIDATION EPOCH', end='')
            print()
            if progress.fail:
                break
        else:
            print(f'epoch: {epoch}, loss: {np.mean(loss_batch)}')

    # return model


# 目前发现不合适的dx可能会导致积分之和不为1, 下面代码自动寻找最优的dx
# dx_list = torch.linspace(0, 1, 21)
# errors = []
# for _ in dx_list:
#     dx = _ * (xmax - xmin)
#     xxmin = xmin - dx
#     xxmax = xmax + dx
#
#     x = torch.linspace(xxmin.detach(), xxmax.detach(), x_num)
#     kde_u_x = torch.exp(kde_u.score_samples(x.reshape(-1, 1)))
#     kde_a_x = torch.exp(kde_a.score_samples(x.reshape(-1, 1)))
#
#     area_full_u = torch.trapz(kde_u_x, x)
#     area_full_a = torch.trapz(kde_a_x, x)
#
#     errors.append(torch.max(torch.abs(1.0 - area_full_u), torch.abs(1.0 - area_full_a)).item())
#
# dx = dx_list[np.argmin(errors)]
# print(f'Calculated dx: {dx}')
# xmin -= dx
# xmax += dx
#
# x = torch.linspace(xmin.detach(), xmax.detach(), x_num)
# kde_u_x = torch.exp(kde_u.score_samples(x.reshape(-1, 1)))
# kde_a_x = torch.exp(kde_a.score_samples(x.reshape(-1, 1)))

# 求重叠面积(旧版)
# if intersection_points_idx.size(0) == 1:
#     c = x[intersection_points_idx]
#
#     x_u, x_a = x.clone(), x.clone()
#     x_u[x_u < c] = 0; x_a[x_a > c] = 0
#
#     area_u = torch.trapz(kde_u_x, x_u)
#     area_a = torch.trapz(kde_a_x, x_a)
#     area = area_u + area_a
#
# elif intersection_points_idx.size(0) == 0: # no intersection point
#     # raise NotImplementedError
#
# else:
#     c1 = x[intersection_points_idx[0]]
#     c2 = x[intersection_points_idx[-1]]
#
#     assert c1 <= c2
#
#     x_u, x_a = x.clone(), x.clone()
#     x_u[x_u < c1] = 0; x_a[x_a > c2] = 0
#
#     # area_list = []
#     # for idx in intersection_points_idx:
#     #     c = x[idx]
#     #
#     #     x_u, x_a = x.clone(), x.clone()
#     #     x_u[x_u < c] = 0; x_a[x_a > c] = 0
#     #
#     #     area_u = torch.trapz(kde_u_x, x_u)
#     #     area_a = torch.trapz(kde_a_x, x_a)
#     #     area_list.append(area_u + area_a)
#     #
#     # area = torch.mean(torch.stack(area_list))
#
#     area_u = torch.trapz(kde_u_x, x_u)
#     area_a = torch.trapz(kde_a_x, x_a)
#     area = area_u + area_a