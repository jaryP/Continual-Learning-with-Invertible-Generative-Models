import math
from functools import reduce
from operator import mul

from typing import Callable

import torch


# class ActNorm(torch.nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#
#         if isinstance(input_dim, int):
#             input_dim = [input_dim]
#
#         self.input_dim = input_dim
#
#         self.scale = torch.nn.Parameter(torch.zeros(*input_dim), requires_grad=True)
#         self.bias = torch.nn.Parameter(torch.zeros(*input_dim), requires_grad=True)
#         self.register_buffer('initialization_done', torch.tensor(0))
#
#     def reset_parameters(self):
#         self.scale = torch.nn.Parameter(torch.zeros(*self.input_dim), requires_grad=True)
#         self.bias = torch.nn.Parameter(torch.zeros(*self.input_dim), requires_grad=True)
#         self.register_buffer('initialization_done', torch.tensor(0))
#
#     def backward(self, x, y=None):
#
#         if self.initialization_done == 0:
#             # self.initialization_done += 1
#             # # print(self.scale.shape)
#             # self.scale.data = 1/torch.log(x.std(0) + 1e-6)
#             # self.bias.data = -torch.mean(x, 0)
#             # self.bias.data = -(x * torch.exp(self.scale)).mean(dim=0)
#             # # print(self.scale.shape)
#             assert False
#
#         x = x * self.scale.exp() + self.bias
#         # print(x.mean(0), x.std(0))
#         # input()
#         log_det = self.scale.expand_as(x).sum(1) #.sum(1)  #* reduce(mul, self.input_dim, 1)
#
#         return x, log_det
#
#     def forward(self, x, y=None):
#         if self.initialization_done == 0:
#             self.initialization_done += 1
#
#             # self.bias.squeeze().data.copy_(x.transpose(0,1).flatten(1).mean(1)).view_as(self.scale)
#             # self.scale.squeeze().data.copy_(x.transpose(0,1).flatten(1).std(1, False) + 1e-6).view_as(self.bias)
#
#             # print(self.scale.shape)
#             # self.scale.data = -torch.log(x.std(0))
#             self.scale.data = torch.log(x.std(0) + 1e-6)
#
#             # self.bias.data = -torch.mean(x, 0)
#             # self.bias.data = -(x * torch.exp(self.scale)).mean(dim=0)
#             self.bias.data = x.mean(0)
#             # print(self.scale.shape)
#
#         x = (x - self.bias) / self.scale.exp()
#         log_det = - self.scale.expand_as(x).sum(1) #.sum()  #* reduce(mul, self.input_dim, 1)
#
#         return x, log_det
from torch import nn


class ActNorm(nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """
    def __init__(self, param_dim=(1,48,1)):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x, y=None):
        if not self.initialized:
            # per channel mean and variance where x.shape = (B, C, H, W)
            self.bias.squeeze().data.copy_(x.transpose(0,1).flatten(1).mean(1)).view_as(self.scale)
            self.scale.squeeze().data.copy_(x.transpose(0,1).flatten(1).std(1, False) + 1e-6).view_as(self.bias)
            self.initialized += 1

        z = (x - self.bias) / self.scale
        logdet = - self.scale.abs().log().sum() * x.shape[2]
        return z, logdet

    def backward(self, z, y=None):
        return z * self.scale + self.bias, self.scale.abs().log().sum() * z.shape[2]


class SequentialFlow(torch.nn.Sequential):
    def forward(self, x, y=None):
        log_det = 0
        for module in self:
            x, _log_det = module(x, y=y)
            log_det = log_det + _log_det
        return x, log_det

    def backward(self, u, y=None):
        log_det = 0
        for module in reversed(self):
            u, _log_det = module.backward(u, y=y)
            log_det = log_det + _log_det
        return u, log_det

    def forward_steps(self, x, y=None):
        log_det = 0
        xs = [x, 'input']
        for module in self:
            x, _log_det = module(x, y=y)
            # xs.append(x)
            xs.append([x, module.__class__.__name__])
            log_det = log_det + _log_det
        return xs, log_det

    def backward_steps(self, u, y=None):
        log_det = 0
        us = [(u, 'prior')]
        for module in reversed(self):
            u, _log_det = module.backward(u, y=y)
            us.append([u, module.__class__.__name__])
            log_det = log_det + _log_det
        return us, log_det


class SplitGaussianize(torch.nn.Module):
    def __init__(self, input_dim, conditioning_size=0):
        super().__init__()

        self.split_index = input_dim // 2
        self.out_dim = input_dim - self.split_index

        # self._s1 = function(self.split_index + conditioning_size, self.out_dim * 2)

        # self.log_scale_factor = torch.nn.Parameter(torch.zeros(self.out_dim * 2), requires_grad=True)

        self.g = Gaussianize(self.split_index, self.out_dim, cond_size=0)

        # self.f = torch.nn.Linear(self.split_index, self.out_dim * 2)  # computes the parameters of Gaussian
        # self.log_scale_factor = torch.nn.Parameter(torch.zeros(self.out_dim * 2),
                                                   # requires_grad=True)  # learned scale (cf RealNVP sec 4.1 / Glow official code

        # self.f.weight.data.zero_()
        # self.f.bias.data.zero_()

    def forward(self, x, y=None):
        x1, x2 = x[:, :self.split_index], x[:, self.split_index:]
        z2, log_det = self.g(x1, x2, y=y)

        # print(x1.shape, self.f)
        # x = self.f(x1) * self.log_scale_factor.exp()
        # m, log = x.chunk(2, dim=1)
        # # print(m.shape, log.shape)
        #
        # z = (x2 - m) * torch.exp(-log)
        # logdet = - log.sum(1)

        return x1, z2, log_det

    def backward(self, x1, z2, y=None):
        x2, log_det = self.g.backward(x1, z2, y=y)

        # x = self.f(x1) * self.log_scale_factor.exp()
        # m, log = x.chunk(2, dim=1)

        # x2 = m + z * torch.exp(log)
        # logdet = log.sum(1)

        x = torch.cat([x1, x2], dim=1)  # cat along channel dim

        return x, log_det


class Gaussianize(torch.nn.Module):
    def __init__(self, input_dim, out_dim, cond_size=0):
        super().__init__()

        # self.split_index = input_dim // 2
        # self.out_dim = input_dim - self.split_index

        # self._s1 = function(self.split_index + conditioning_size, self.out_dim * 2)

        # self.log_scale_factor = torch.nn.Parameter(torch.zeros(out_dim * 2), requires_grad=True)
        self.f = torch.nn.Linear(input_dim, out_dim * 2)  # computes the parameters of Gaussian
        # self.f = torch.nn.Conv1d(input_dim, out_dim * 2, 3, 1, 1)
        # self.log_scale_factor = torch.nn.Parameter(torch.zeros(input_dim // 2), requires_grad=True)  # learned
        # scale (cf RealNVP sec 4.1 / Glow official code

        # self.f.weight.data.zero_()
        # self.f.bias.data.zero_()

    def backward(self, x1, x2, y=None):
        # if y is not None:
        #     x1 = torch.cat([y, x1], dim=1)
        # else:
        #     x1 = x1

        # x1, x2 = x[:, :self.split_index], x[:, self.split_index:]
        # print(x1.shape, self.f)
        x = self.f(x1) #* self.log_scale_factor.exp()
        m, log = x.chunk(2, dim=1)
        # log *= self.log_scale_factor.exp()
        # print(m.shape, log.shape)

        z = (x2 - m) * torch.exp(-log)
        logdet = - log.sum(1)

        return z, logdet

    def forward(self, x1, x2, y=None):
        # if y is not None:
        #     x1 = torch.cat([y, x1], dim=1)

        x = self.f(x1) #* self.log_scale_factor.exp()
        m, log = x.chunk(2, dim=1)
        # log *= self.log_scale_factor.exp()

        z = m + x2 * torch.exp(log)
        logdet = log.sum(1)

        # x = torch.cat([x1, x2], dim=1)  # cat along channel dim

        return z, logdet


class Permutation(torch.nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        if isinstance(in_ch, tuple):
            in_ch = in_ch[0]

        self.in_ch = in_ch
        self.register_buffer('p', torch.randperm(in_ch))
        self.register_buffer('invp', torch.argsort(self.p))

    def forward(self, x, y=None):
        out = x[:, self.p]
        return out, 0

    def backward(self, x, y=None):
        out = x[:, self.invp]
        return out, 0


class Rescale(torch.nn.Module):
    def __init__(self, dim):
        super(Rescale, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = self.weight * x
        return x


class CouplingLayer(torch.nn.Module):
    def __init__(self, input_dim, conditioning_size=None,
                 function: Callable = None):
        super().__init__()

        if conditioning_size is None:
            conditioning_size = 0

        if isinstance(input_dim, (list, tuple)) and len(input_dim) > 1:
            input_dim = input_dim[0]

        self.split_index = input_dim // 2
        self.out_dim = input_dim - self.split_index

        self._s = function(self.split_index + conditioning_size, self.out_dim * 2)
        # self._t = function(self.split_index + conditioning_size, self.out_dim)

        # for i in range(len(self._t)):
        #     if isinstance(self._t[i], torch.nn.ReLU):
        #         self._t[i] = torch.nn.Tanh()

        self._s[-1].weight.data.zero_()
        self._s[-1].bias.data.zero_()
        # self.rescale = torch.nn.utils.weight_norm(Rescale(self.out_dim))
        self.log_scale_factor = torch.nn.Parameter(torch.zeros(self.out_dim), requires_grad=True)

    def s(self, x, y, f):

        if y is not None:
            _x = torch.cat([y, x], dim=1)
        else:
            _x = x

        _x = self._s(_x)  # * self.log_scale_factor.exp()

        s, t = _x.chunk(2, 1)
        # s = s * self.log_scale_factor.exp()

        # s, t = self._s(x), self._t(x)

        s = torch.sigmoid(s) * self.log_scale_factor.exp()
        # s = self.rescale(torch.sigmoid(s))
        # s = torch.tanh(s) * self.rescale

        return s, t

    def forward(self, x, y=None):
        x1, x2 = x[:, :self.split_index], x[:, self.split_index:]

        s1, t1 = self.s(x1, y, self._s)
        # s1 = s1.mul(-1).exp()

        x2 = (x2 - t1) / s1.exp()
        x = torch.cat((x1, x2), 1)

        log_det = - torch.sum(s1, 1)

        return x, log_det

    def backward(self, x, y=None):
        x1, x2 = x[:, :self.split_index], x[:, self.split_index:]

        s1, t1 = self.s(x1, y, self._s)
        # s1 = s1.exp()

        x2 = x2 * s1.exp() + t1

        x = torch.cat((x1, x2), 1)
        log_det = torch.sum(s1, 1)

        return x, log_det


class BatchNorm(torch.nn.Module):
    # https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.input_size = input_size

        self.log_gamma = torch.nn.Parameter(torch.zeros(self.input_size))
        self.beta = torch.nn.Parameter(torch.zeros(self.input_size))

        self.register_buffer('running_mean', torch.zeros(self.input_size))
        self.register_buffer('running_var', torch.ones(self.input_size))

        self.batch_mean = None
        self.batch_var = None

    def reset_parameters(self):
        self.register_buffer('running_mean', torch.zeros(self.input_size))
        self.register_buffer('running_var', torch.ones(self.input_size))

        self.log_gamma = torch.nn.Parameter(torch.zeros(self.input_size))
        self.beta = torch.nn.Parameter(torch.zeros(self.input_size))

        self.batch_mean = None
        self.batch_var = None

    def forward(self, x, **kwargs):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)

            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        log_det = self.log_gamma - 0.5 * torch.log(var + self.eps)

        return y, log_det.expand_as(x).sum(1)

    def backward(self, x, **kwargs):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_det = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_det.expand_as(x).sum(1)


class RNVPBlock(SequentialFlow):
    def __init__(self, input_dim, coupling_f=None, conditioning_size=0):
        super().__init__(BatchNorm(input_dim),
                         CouplingLayer(input_dim, conditioning_size=conditioning_size, function=coupling_f),
                         Permutation(input_dim))


class RNVPLevel(torch.nn.Module):
    def __init__(self, input_dim, n_blocks, coupling_f=None, conditioning_size=0):
        super().__init__()
        self._blocks = SequentialFlow(*[RNVPBlock(input_dim, coupling_f=coupling_f,
                                                  conditioning_size=conditioning_size) for _ in range(n_blocks)])
        self._split = SplitGaussianize(input_dim, conditioning_size=conditioning_size)

    def forward(self, x, y=None):
        log_det = 0

        x, _log_det = self._blocks.forward(x, y=y)
        log_det += _log_det
        x, z, _log_det = self._split.forward(x, y=y)
        log_det += _log_det

        return x, z, log_det

    def backward(self, x, z, y=None):
        log_det = 0

        x, _log_det = self._split.backward(x, z, y=y)
        log_det += _log_det
        x, _log_det = self._blocks.backward(x, y=y)
        log_det += _log_det

        return x, log_det


class RNVP(torch.nn.Module):
    def __init__(self, n_levels, levels_blocks, input_dim, hidden_size, n_hidden=1,
                 coupling_f=None, conditioning_size=0):

        super().__init__()

        if coupling_f is None:
            def coupling_f(ind, outd):
                hs = hidden_size

                if isinstance(hidden_size, float):
                    hs = int(hidden_size * ind)

                s = [torch.nn.Linear(ind, hs), torch.nn.ReLU()] #, torch.nn.BatchNorm1d(hs)]

                for i in range(n_hidden):
                    s.append(torch.nn.Linear(hs, hs))
                    s.append(torch.nn.ReLU())
                    # s.append(nn.Dropout(0.25))
                    # s.append(torch.nn.BatchNorm1d(hs))

                s.append(torch.nn.Linear(hs, outd))

                net = torch.nn.Sequential(*s)

                return net

        self._levels = torch.nn.ModuleList()
        self.input_dim = input_dim
        self.to_flatten = False

        in_dim = input_dim

        if not isinstance(in_dim, int):
            in_dim = tuple(in_dim)

            if isinstance(input_dim, (tuple, float)):
                in_dim = reduce(mul, in_dim, 1)
                self.to_flatten = True

        self.dims = []

        d = in_dim
        for i in range(n_levels):
            x_dim = d // 2
            z_dim = d - d // 2

            level = RNVPLevel(d, n_blocks=levels_blocks, coupling_f=coupling_f,
                              conditioning_size=conditioning_size)
            d = x_dim

            assert d > 0

            self.dims.append((x_dim, z_dim))

            self._levels.append(level)

        last_dim = in_dim - sum(z for x, z in self.dims)
        # last_z = in_dim - in_dim // 2

        # last_dim = in_dim

        self.dims.append((last_dim, last_dim))
        # print(self.dims)

        self.g = Gaussianize(last_dim, last_dim, conditioning_size)

        # self.norm = ActNorm(input_dim)
        # self.register_buffer('base_dist_mean', torch.zeros(1))
        # self.register_buffer('base_dist_var', torch.ones(1))

    def forward(self, x, y=None):
        log_det = 0
        zs = []

        if self.to_flatten:
            x = torch.flatten(x, 1)

        for m in self._levels:
            x, z, _log_det = m(x, y=y)
            log_det += _log_det
            zs.append(z)

        z, _log_det = self.g(torch.zeros_like(x), x, y)
        zs.append(z)

        log_det += _log_det

        z = torch.cat(zs, 1)

        return z, log_det

    def backward(self, u: torch.Tensor, y: torch.Tensor = None, device: str = 'cpu'):
        # if zs is None:
        #     if y is None:
        #         assert batch_size is not None, 'Must either specify batch_size or pass a batch of z random numbers.'
        #     else:
        #         batch_size = y.shape[0]
        # else:
        #     assert len(zs) == len(self._levels)
        #     batch_size = zs[0].shape[0]
        #
        # if batch_size is None:
        #     batch_size = zs[0][0].shape

        log_det = 0
        # zs = []

        if self.to_flatten:
            u = torch.flatten(u, 1)

        # print(u.shape)
        zs = torch.split(u, [zd for _, zd in self.dims], dim=1)
        # for xd, zd in self.dims:
        #     pass

        # x = torch.zeros_like(zs[-1])
        z = zs[-1]
        x, log_det = self.g.backward(torch.zeros_like(z), z, y)

        for i, m in enumerate(reversed(self._levels)):

            z = zs[-i - 2]

            x, _log_det = m.backward(x, z, y=y)

            log_det += _log_det

        # x, _log_det = self.norm.backward(x)
        # log_det += _log_det

        if self.to_flatten:
            x = x.view((u.size(0),) + self.input_dim)

        # print(x.shape)

        return x, log_det

    # def sample(self, size: int, device: str='cpu'):
    #     return [self.base_dist.sample((size, self.dims[i][1])).squeeze().to(device) for i in range(len(self.dims))]
    #
    #     # return self.base_dist.sample((size, z_dim)).squeeze().to(device)
    # @property
    # def base_dist(self):
    #     return torch.distributions.Normal(self.base_dist_mean, self.base_dist_var)
    # def log_prob(self, zs):
    #     return sum(self.base_dist.log_prob(z).sum(1) for z in zs)
