import math
from typing import Callable, List, Tuple, Union

import torch
from torch import nn
from torch.nn import init


class ChannelWiseLinear(nn.Module):
    def __init__(self, inp: tuple, out=None):
        super().__init__()

        channels, w = inp

        if out is None:
            out = w

        t = torch.zeros((channels, w, out))
        b = torch.zeros((1, 1, out))

        self.register_parameter('weight', nn.Parameter(t, requires_grad=True))
        self.register_parameter('bias', nn.Parameter(b, requires_grad=True))

        bound = 1 / math.sqrt(w)
        init.uniform_(self.weight, -bound, bound)

        init.uniform_(self.bias, -1, 1)

    def forward(self, x):
        return torch.einsum('bcw,cwo->bco', [x, self.weight]) + self.bias

    def extra_repr(self):
        return '{}'.format(self.weight.shape)


class ChannelWiseLinearModel(nn.Module):
    def __init__(self, input_dim: tuple, out_dim: int,
                 hidden_dim_channel: int = None, hidden_dim_width: int = None,
                 hidden_layers_channel: int = 1, hidden_layers_width: int = 1):
        super().__init__()

        c, w = input_dim

        if hidden_dim_width is None:
            hidden_dim_width = w * 2
        elif isinstance(hidden_dim_width, float):
            hidden_dim_width = int(w * hidden_dim_width)

        if hidden_dim_channel is None:
            hidden_dim_channel = c * 2
        elif isinstance(hidden_dim_channel, float):
            hidden_dim_channel = int(c * hidden_dim_channel)

        layers = [TransposeClass(-1, -2)]

        b = [ChannelWiseLinear((w, c), hidden_dim_channel), nn.ReLU()]

        for i in range(hidden_layers_channel):
            b.append(ChannelWiseLinear((w, hidden_dim_channel), hidden_dim_channel))
            b.append(nn.ReLU())

        b.append(ChannelWiseLinear((w, hidden_dim_channel), c))
        b = nn.Sequential(*b)

        layers.append(b)
        layers.append(TransposeClass(-1, -2))

        self.net1 = torch.nn.Sequential(*layers)

        # layers = [TransposeClass(-1, -2)]
        # layers = []

        # b = [nn.Conv1d(c, hidden_dim_channel, kernel_size=3, padding=1, stride=1), nn.ReLU()]
        #
        # for i in range(hidden_layers_channel):
        #     b.append(nn.Conv1d(hidden_dim_channel, hidden_dim_channel, kernel_size=3, padding=1, stride=1))
        #     b.append(nn.ReLU())
        #
        # b.append(nn.Conv1d(hidden_dim_channel, c, kernel_size=3, padding=1, stride=1))
        # b = nn.Sequential(*b)

        # self.net1 = torch.nn.Sequential(*b)

        b = [ChannelWiseLinear((c, w), hidden_dim_width), nn.ReLU()]

        for i in range(hidden_layers_width):
            b.append(ChannelWiseLinear((c, hidden_dim_width), hidden_dim_width))
            b.append(nn.ReLU())

        b.append(ChannelWiseLinear((c, hidden_dim_width), out_dim))

        b = nn.Sequential(*b)

        self.net = nn.Sequential(*b)

    def forward(self, x):
        x = self.net1(x)
        x = self.net(x)
        # x = torch.relu(x) #+ x
        return x


class Convolutional1DModel(nn.Module):
    def __init__(self, input_dim: tuple, out_dim: int, hidden_dim: int = None, hidden_layers=1,
                 hidden_layers_channel: int = 1, hidden_layers_width: int = None):
        super().__init__()

        c, w = input_dim

        if hidden_dim is None:
            hidden_dim = c

        if hidden_layers_width:
            hidden_layers_width = w

        assert hidden_dim % c == 0

        layers = [nn.Conv1d(c, hidden_dim, kernel_size=3, padding=1, stride=1, groups=c), nn.ReLU()]

        for i in range(hidden_layers):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=1, groups=1))
            layers.append(nn.ReLU())

        layers.append(nn.Conv1d(hidden_dim, c, kernel_size=3, padding=1, stride=1, groups=c))
        layers.append(nn.ReLU())

        layers.extend([ChannelWiseLinear((c, w), hidden_layers_width), nn.ReLU()])

        for i in range(hidden_layers_channel):
            layers.extend([ChannelWiseLinear((c, hidden_layers_width), hidden_layers_width), nn.ReLU()])

        layers.extend([ChannelWiseLinear((c, hidden_layers_width), out_dim)])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_f(hidden_dim_channel: Union[int, float] = None, hidden_dim_width: Union[int, float] = None,
          hidden_layers_channel: int = 1, hidden_layers_width: int = 1, model='linear'):
    if model == 'linear':
        def f(input_dim: tuple, out_dim: int):
            return ChannelWiseLinearModel(input_dim, out_dim,
                                          hidden_dim_channel, hidden_dim_width,
                                          hidden_layers_channel, hidden_layers_width)
    else:
        def f(input_dim: tuple, out_dim: int):
            return Convolutional1DModel(input_dim, out_dim,
                                        hidden_dim_channel, hidden_layers_channel,
                                        hidden_layers_channel, hidden_layers_width
                                        )

    return f


class TransposeClass(nn.Module):
    def __init__(self, d1, d2):
        super().__init__()
        self.d1 = d1
        self.d2 = d2

    def forward(self, x):
        return torch.transpose(x, self.d1, self.d2)


class InvertibleReshape(nn.Module):
    def __init__(self, input_dim, flat_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.flat_dim = flat_dim

    def forward(self, x, y=None):
        return torch.flatten(x, self.flat_dim)

    def backward(self, x, y=None):
        return x.view(*((x.size(0),) + self.input_dim))


class SequentialFlow(torch.nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

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
    def __init__(self, input_dim, gaussianize_f: Callable):
        super().__init__()
        c, w = input_dim

        if w % 2 == 0:
            o_dim = w
            w = w // 2
        else:
            o_dim = (w // 2) * 2
            w = w - w // 2

        input_dim = (c, w)

        self.g = Gaussianize(input_dim=input_dim, out_dim=o_dim, gaussianize_f=gaussianize_f)

    def forward(self, x, y=None):
        x1, x2 = x.chunk(2, -1)
        z2, log_det = self.g(x1, x2, y=y)

        return x1, z2, log_det

    def backward(self, x1, z2, y=None):
        x2, log_det = self.g.backward(x1, z2, y=y)

        x = torch.cat([x1, x2], dim=-1)

        return x, log_det


class Gaussianize(torch.nn.Module):
    def __init__(self, input_dim, out_dim, gaussianize_f):
        super().__init__()

        self.f = gaussianize_f(input_dim, out_dim)

    def forward(self, x1, x2, y=None):
        x = self.f(x1)  # * self.log_scale_factor.exp()
        m, log = x.chunk(2, dim=-1)

        z = (x2 - m) * torch.exp(-log)

        return z, -log.sum([1, 2])

    def backward(self, x1, x2, y=None):
        x = self.f(x1)  # * self.log_scale_factor.exp()
        m, log = x.chunk(2, dim=-1)

        z = m + x2 * torch.exp(log)

        return z, log.sum([1, 2])


class Permutation(torch.nn.Module):
    def __init__(self, in_dim, dim='channel'):
        super().__init__()

        # if dim == 'channel':
        #     in_dim = in_dim[0]
        # elif dim == 'width':
        in_dim = in_dim[1]
        # else:
        #     assert False

        self.dim = 'width'

        self.register_buffer('p', torch.randperm(in_dim))
        self.register_buffer('invp', torch.argsort(self.p))

    def forward(self, x, y=None):
        if self.dim == 'channel':
            return x[:, self.p], 0
        else:
            return x[:, :, self.p], 0

    def backward(self, x, y=None):
        if self.dim == 'channel':
            return x[:, self.invp], 0
        else:
            return x[:, :, self.invp], 0


class Rescale(torch.nn.Module):
    def __init__(self, dim):
        super(Rescale, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = self.weight * x
        return x


class CouplingLayer(torch.nn.Module):
    def __init__(self, input_dim: tuple, function: Callable, split_dim='channel',
                 conditioning_size: int = 0):
        super().__init__()

        # Cx(H*W)
        assert len(input_dim) == 2
        input_channels, w = input_dim

        w = w / 2
        ow = math.floor(w) * 2
        w = math.ceil(w)

        # ow = w
        # if w % 2 != 0:
        #     ow -= 1
        # ow = ow * 2

        # if w % 2 == 0:
        #     # z_dim = w - w // 2
        #     x_dim = z_dim = w // 2
        # else:
        #     z_dim = math.floor(w / 2)
        #     x_dim = math.ceil(w / 2)

        # if split_dim == 'width':
        self.split_dim = -1
        # elif split_dim == 'channel':
        #     # input_channels = input_channels / 2
        #     # input_channels = math.ceil(input_channels)
        #     self.split_dim = -1
        # else:
        #     assert False

        input_dim = (input_channels, w + conditioning_size)

        self._s = function(input_dim, ow)

        # self.rescale = torch.nn.utils.weight_norm(Rescale((input_channels, 1)))
        # self._s.net[-1].weight.data.zero_()

        # self.forward, self.backward = self.backward, self.forward
        # self._s.net[-1].w.data.zero_()

        # self.register_buffer('mask', mask)
        # self.split_dim = split_dim

        # self.log_scale_factor = torch.nn.Parameter(torch.zeros((input_channels, ow)), requires_grad=True)

    def s(self, x, y):

        if y is not None:
            y = y.unsqueeze(1).expand((-1, x.size(1), -1))
            # print(x.shape, y.shape)
            _x = torch.cat([y, x], dim=-1)
        else:
            _x = x

        _x = self._s(_x)  # * self.log_scale_factor.exp()
        s, t = _x.chunk(2, self.split_dim)

        # s = self._s(_x)
        # t = self._t(_x)
        # s = self.rescale(torch.sigmoid(s + 2.5))
        # t = torch.tanh(t + 2.)
        s = torch.sigmoid(s + 2.5)

        return s, t

    def backward(self, x, y=None):

        x1, x2 = x.chunk(2, self.split_dim)

        s, t = self.s(x1, y)

        x2 = (x2 - t) * s.exp().reciprocal()

        x = torch.cat((x1, x2), dim=self.split_dim)

        return x, - s.sum([1, 2])

    def forward(self, x, y=None):

        x1, x2 = x.chunk(2, self.split_dim)

        s, t = self.s(x1, y)

        x2 = x2 * s.exp() + t

        x = torch.cat((x1, x2), dim=self.split_dim)

        return x, s.sum([1, 2])


class BatchNorm(torch.nn.Module):
    # https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py

    def __init__(self, input_size: tuple, momentum=0.9, eps=1e-5):
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

        return y, log_det.expand_as(x).sum([1, 2])

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

        return x, log_det.expand_as(x).sum([1, 2])


class NFLevel(torch.nn.Module):
    def __init__(self, input_dim: tuple, n_blocks: int, coupling_f: Callable, gaussianize_f: Callable,
                 conditioning_size: int = 0):
        super().__init__()
        levels = list()

        for i in range(n_blocks):
            dim = 'width'
            levels.append(BatchNorm(input_dim))
            levels.append(CouplingLayer(input_dim, conditioning_size=conditioning_size, split_dim=dim,
                                        function=coupling_f))
            levels.append(Permutation(input_dim, dim=dim))

        self._blocks = SequentialFlow(*levels)

        self._split = SplitGaussianize(input_dim, gaussianize_f=gaussianize_f)

    def forward(self, x, y=None):
        x, log_det = self._blocks.forward(x, y=y)

        if self._split is not None:
            x, z, _log_det = self._split.forward(x, y=y)
            log_det += _log_det
            return x, z, log_det

        return x, log_det

    def backward(self, x, z=None, y=None):
        log_det = 0
        if self._split is not None:
            x, _log_det = self._split.backward(x, z, y=y)
            log_det += _log_det

        x, _log_det = self._blocks.backward(x, y=y)
        log_det += _log_det

        return x, log_det


class ChannelWiseNF(torch.nn.Module):
    def __init__(self, n_levels: int, levels_blocks: int, input_dim: tuple, coupling_f: Callable,
                 gaussianize_f: Callable,
                 conditioning_size: int = 0):

        super().__init__()

        self.reshape = None

        if len(input_dim) > 2:
            self.reshape = InvertibleReshape(input_dim, 2)
            input_dim = (input_dim[0], input_dim[1] * input_dim[2])

        self._levels = torch.nn.ModuleList()

        c, w = input_dim

        self.dims = []

        for i in range(n_levels):
            assert w > 3

            if w % 2 == 0:
                x_dim = z_dim = w // 2
            else:
                z_dim = math.floor(w / 2)
                x_dim = math.ceil(w / 2)

            level = NFLevel((c, w), n_blocks=levels_blocks, coupling_f=coupling_f,
                            conditioning_size=conditioning_size, gaussianize_f=gaussianize_f)
            w = x_dim

            self.dims.append(z_dim)

            self._levels.append(level)

        last_dim = input_dim[1] - sum(self.dims)

        self.dims.append(last_dim)

        assert sum(self.dims) == input_dim[1]

        i_dim = (c, last_dim)
        self.g = Gaussianize(i_dim, last_dim * 2, gaussianize_f=gaussianize_f)

    def forward(self, x, y=None):
        zs = []

        if self.reshape is not None:
            x = self.reshape(x)

        log_det = torch.zeros_like(x).sum([1, 2])

        for i, m in enumerate(self._levels):
            x, z, _log_det = m(x, y=y)
            log_det += _log_det

            zs.append(z)

        z, _log_det = self.g(torch.zeros_like(x), x, y)
        zs.append(z)

        log_det += _log_det

        z = torch.cat(zs, -1)

        return z, log_det

    def backward(self, u: torch.Tensor, y: torch.Tensor = None):

        zs = torch.split(u, self.dims, dim=-1)
        z = zs[-1]

        x, log_det = self.g.backward(torch.zeros_like(z), z, y)

        for i, m in enumerate(reversed(self._levels)):
            z = zs[- i - 2]
            x, _log_det = m.backward(x, z, y=y)

        if self.reshape is not None:
            x = self.reshape.backward(x)

        return x, 0


if __name__ == '__main__':
    x = torch.randn((10, 20, 5, 5))

    # cwl = ChannelWiseLinearModel(input_dim=(20, 25), out_dim=13 * 10, hidden_layers=2, hidden_dim=100)
    # print(cwl)
    # r = cwl(torch.randn((10, 20, 25)))
    # print(r.shape)
    f = get_f(model='convolutional')

    print(f((20, 25), 50)(torch.randn((10, 20, 25))).shape)

    # exit()
    # sg = SplitGaussianize(input_dim=(20, 25), gaussianize_f=get_f())
    # print(sg)

    # x, z, _ = sg(torch.randn((10, 20, 25)))

    # g = Gaussianize((20, 25), 50, get_f())
    # z1, _ = g(torch.randn((10, 20, 25)), torch.randn((10, 20, 25)))

    # print(x.shape, z.shape, z1.shape)

    cwnf = ChannelWiseNF(n_levels=1, levels_blocks=2, input_dim=(20, 5, 5), coupling_f=get_f(),
                         gaussianize_f=get_f(model='convolutional'))
    print(cwnf)

    cwnf(torch.randn((10, 20, 25)))

    x, _ = cwnf.backward(torch.randn((10, 20, 25)))

    print(x.shape)

    cwnf = ChannelWiseNF(n_levels=4, levels_blocks=2, input_dim=(20, 6, 6), coupling_f=get_f(),
                         gaussianize_f=get_f())
    # print(cwnf)

    cwnf(torch.randn((10, 20, 36)))

    x, _ = cwnf.backward(torch.randn((10, 20, 36)))

    print(x.shape)

    exit()

    # l = ChannelWiseLinearModel(input_dim=(20, 25), out_dim=25, hidden_layers=2, hidden_dim=100)
    # print(l)
    #
    # l(torch.randn((10, 20, 25)))
    #
    # exit()
    # gcp = GlowLevel(input_dim=(2, 25), coupling_f=ChannelWiseLinearModel, dim='channel', n_blocks=3, split=False)
    #
    # i = torch.randn((10, 2, 25))
    # a, b = gcp(i)

    # print(i[0])
    # print(a[0])
    # exit()
    # def f(inp, out):
    #     return torch.nn.Conv1d(inp, out, 3, 1, padding=1)
    #
    #
    # cl = CouplingLayer(input_dim=(48, 25))
    glow = ChannelWiseNF(n_levels=1, levels_blocks=1, input_dim=(20, 5, 5))
    print(glow)
    u, log_det = glow(x)
    glow.backward(u)
    print(log_det)
    exit()
    # rnvp = ChannelWiseRNVPNoSplit(2, 20, (20, 5, 5), input_dim='channel')
    #
    # print(rnvp)
    #
    # u, log_det = rnvp(x)
    # print(log_det.min(), log_det.max())
    # print(log_det.clamp(min=0.01).min(), log_det.clamp(min=0.01).max())
    # input()
    # print(u.mean(), u.std())
    #
    # rnvp.eval()
    # xh, _ = rnvp.backward(u)
    #
    # print(torch.abs(x - xh).mean())
    # u, log_det = rnvp(x)
    #
    # # print(u)
    # rnvp.eval()
    # x_hat, _ = rnvp.backward(u)
    #
    # diff = x - x_hat
    # diff = diff.abs()
    #
    # print(diff.mean())
    #
    # # # w = torch.randn((10, 48, 25))
    # x = torch.flatten(x, 2)
    # print(x.shape)
    #
    # w1 = torch.randn((48, 25, 100))
    #
    # res = torch.einsum('bcw,cwo->bco', (x, w1))
    #
    # print(res.shape)
    #
    # # x = x.permute(2, 1, 0)
    # torch.bmm(x, w1)
