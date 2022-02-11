from functools import reduce
from operator import mul
from typing import Callable
import torch

from continual_ai.cl_strategies.multi_task.prer.utils import SequentialFlow, \
    EmbeddingPreprocessing


class SplitGaussianize(torch.nn.Module):
    def __init__(self, input_dim, conditioning_size=0):
        super().__init__()

        self.split_index = input_dim // 2
        self.out_dim = input_dim - self.split_index

        self.g = Gaussianize(self.split_index, self.out_dim, cond_size=0)

    def forward(self, x, y=None):
        x1, x2 = x[:, :self.split_index], x[:, self.split_index:]
        z2, log_det = self.g(x1, x2, y=y)

        return x1, z2, log_det

    def backward(self, x1, z2, y=None):
        x2, log_det = self.g.backward(x1, z2, y=y)

        x = torch.cat([x1, x2], dim=1)  # cat along channel dim

        return x, log_det


class Split(torch.nn.Module):
    def __init__(self, input_dim, conditioning_size=0):
        super().__init__()

        self.split_index = input_dim // 2
        self.out_dim = input_dim - self.split_index

    def forward(self, x, y=None):
        x1, x2 = x[:, :self.split_index], x[:, self.split_index:]
        return x1, x2, 0

    def backward(self, x1, z2, y=None):
        return torch.cat([x1, z2], dim=1), 0


class Gaussianize(torch.nn.Module):
    def __init__(self, input_dim, out_dim, cond_size=0):
        super().__init__()

        self.f = torch.nn.Linear(input_dim,
                                 out_dim * 2)  # computes the parameters of Gaussian
        self.f.bias.data.fill_(0)
        torch.nn.init.orthogonal_(self.f.weight.data)

    def forward(self, x1, x2, y=None):
        # return x2, 0
        x = self.f(x1)
        m, log = x.chunk(2, dim=1)

        z = (x2 - m) * torch.exp(-log)
        logdet = - log.sum(1)

        return z, logdet

    def backward(self, x1, x2, y=None):
        # return x2, 0
        x = self.f(x1)
        m, log = x.chunk(2, dim=1)

        z = m + x2 * torch.exp(log)
        logdet = log.sum(1)

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

        self._com = function(self.split_index + conditioning_size, self.out_dim)
        self._s = torch.nn.Linear(self.out_dim, self.out_dim, bias=True)
        self._t = torch.nn.Linear(self.out_dim, self.out_dim, bias=True)

        def init(m):
            if isinstance(m, torch.nn.Linear):
                m.bias.data.fill_(0)
                # torch.nn.init.orthogonal_(m.weight.data)
                m.weight.data.zero_()
            # if isinstance(m, torch.nn.Sequential):
            #     m[-1].data.fill_(0)

        # self._com.apply(init)
        self._s.apply(init)
        self._t.apply(init)

        # torch.nn.init.orthogonal_(self._t.weight.data)
        # self._s.weight.data.fill_(0)
        # torch.nn.init.orthogonal_(self._s.weight.data)

        # self._s.weight.data.zero_()
        # self._s.bias.data.zero_()
        self.log_scale_factor = torch.nn.Parameter(torch.zeros(self.out_dim),
                                                   requires_grad=True)
        # self.rescale = torch.nn.utils.weight_norm(Rescale(self.out_dim))

    def s(self, x, y):

        if y is not None:
            _x = torch.cat([y, x], dim=1)
        else:
            _x = x

        com = self._com(_x) * self.log_scale_factor.exp()
        # s, t = self._s(torch.relu(com)), self._t(torch.relu(com))
        s, t = self._s(com), self._t(com)

        # s, t = _x.chunk(2, 1)

        # s = self.rescale(torch.tanh(s))  # * self.log_scale_factor.exp()
        s = torch.sigmoid(s + 1.5)  # * self.log_scale_factor.exp()

        return s, t

    def backward(self, x, y=None):
        x1, x2 = x[:, :self.split_index], x[:, self.split_index:]

        s1, t1 = self.s(x1, y)

        x2 = (x2 - t1) / s1
        x = torch.cat((x1, x2), 1)

        log_det = - torch.sum(s1.log(), 1)

        return x, log_det

    def forward(self, x, y=None):
        x1, x2 = x[:, :self.split_index], x[:, self.split_index:]

        s1, t1 = self.s(x1, y)

        x2 = x2 * s1 + t1

        x = torch.cat((x1, x2), 1)
        log_det = torch.sum(s1.log(), 1)

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

            self.running_mean.mul_(self.momentum).add_(
                self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(
                self.batch_var.data * (1 - self.momentum))

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


class Actnorm(torch.nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """

    def __init__(self, param_dim=(1, 3, 1, 1), return_logdet=True):
        super().__init__()
        self.return_logdet = return_logdet
        self.scale = torch.nn.Parameter(torch.ones(param_dim))
        self.bias = torch.nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x, **kwargs):
        if not self.initialized:
            # per channel mean and variance where x.shape = (B, C, H, W)
            self.bias.data.copy_(x.mean(0))
            self.scale.data.copy_(x.std(0))
            # self.bias.squeeze().data.copy_(
            #     x.transpose(0, 1).flatten(1).mean(1)).view_as(self.scale)
            # self.scale.squeeze().data.copy_(
            #     x.transpose(0, 1).flatten(1).std(1, False) + 1e-6).view_as(
            #     self.bias)
            self.initialized += 1

        z = (x - self.bias) / self.scale

        if self.return_logdet:
            logdet = - self.scale.abs().log().sum()
            return z, logdet
        else:
            return z

    def backward(self, z, **kwargs):
        return z * self.scale + self.bias, \
               self.scale.abs().log().sum()

class RNVPBlock(SequentialFlow):
    def __init__(self, input_dim, coupling_f=None, conditioning_size=0):
        super().__init__(
            Actnorm(input_dim),
            # BatchNorm(input_dim, momentum=0.9),
            Permutation(input_dim),
            CouplingLayer(input_dim,
                          conditioning_size=conditioning_size,
                          function=coupling_f)
        )


class RNVPLevel(torch.nn.Module):
    def __init__(self, input_dim, n_blocks, coupling_f=None,
                 conditioning_size=0):
        super().__init__()
        self._blocks = SequentialFlow(
            *[RNVPBlock(input_dim, coupling_f=coupling_f,
                        conditioning_size=conditioning_size) for _ in
              range(n_blocks)])
        # self._split = SplitGaussianize(input_dim, conditioning_size=conditioning_size)
        # self._split = Split(input_dim)
        self._split = SplitGaussianize(input_dim)

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
    def __init__(self, n_levels, levels_blocks, input_dim, hidden_size,
                 n_hidden=1,
                 coupling_f=None, conditioning_size=0):

        super().__init__()

        if coupling_f is None:
            def coupling_f(ind, outd):
                hs = hidden_size

                if isinstance(hidden_size, float):
                    hs = int(hidden_size * ind)

                s = [torch.nn.Linear(ind, hs),
                     # Actnorm(hs, return_logdet=False),
                     torch.nn.ReLU()]  # , torch.nn.Dropout(0.2)]

                for i in range(n_hidden):
                    s.append(torch.nn.Linear(hs, hs))
                    # s.append(Actnorm(hs, return_logdet=False))
                    s.append(torch.nn.ReLU())
                    # s.append(torch.nn.Dropout(0.2))

                s.append(torch.nn.Linear(hs, outd))
                # s.append(Actnorm(outd, return_logdet=False))
                s.append(torch.nn.ReLU())

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

        self.dims.append((last_dim, last_dim))
        self.gaussianize = Gaussianize(last_dim, last_dim)
        self.preprocessing = EmbeddingPreprocessing()

        # self.mean = None
        # self.std = None
        self.register_parameter('mean', torch.nn.Parameter(torch.tensor(0),
                                                           requires_grad=False))
        self.register_parameter('std', torch.nn.Parameter(torch.tensor(1),
                                                          requires_grad=False))

    def set_mean_std(self, mean, std):
        self.preprocessing.set(mean, std)
        # self.register_parameter('mean',
        #                         torch.nn.Parameter(mean, requires_grad=False))
        # self.register_parameter('std',
        #                         torch.nn.Parameter(std, requires_grad=False))
        # self.mean = torch.nn.Parameter(torch.tensor(mean), requires_grad=False)
        # self.std = torch.nn.Parameter(torch.tensor(std), requires_grad=False)

    def forward(self, x, y=None):
        log_det = 0
        zs = []

        if self.to_flatten:
            x = torch.flatten(x, 1)

        # if self.mean is not None:
        #     x -= self.mean
        #     x /= self.std
        #
        x = self.preprocessing(x)

        for m in self._levels:
            x, z, _log_det = m(x, y=y)
            log_det += _log_det
            zs.append(z)

        x, ld = self.gaussianize(torch.zeros_like(x), x)
        log_det += ld

        zs.append(x)

        z = torch.cat(zs, 1)

        return z, log_det

    def backward(self, u: torch.Tensor, y: torch.Tensor = None,
                 device: str = 'cpu'):
        if self.to_flatten:
            u = torch.flatten(u, 1)

        zs = torch.split(u, [zd for _, zd in self.dims], dim=1)
        x = zs[-1]
        x, ld = self.gaussianize.backward(torch.zeros_like(x), x)
        log_det = ld

        # log_det = 0

        for i, m in enumerate(reversed(self._levels)):
            z = zs[-i - 2]

            x, _log_det = m.backward(x, z, y=y)

            log_det += _log_det

        # if self.mean is not None:
        #     x *= self.std
        #     x += self.mean
        #
        x = self.preprocessing.backward(x)

        if self.to_flatten:
            x = x.view((u.size(0),) + self.input_dim)

        return x, log_det
