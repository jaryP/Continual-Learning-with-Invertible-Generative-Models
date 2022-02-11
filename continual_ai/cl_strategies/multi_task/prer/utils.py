import numpy as np
from torch import nn
from torch.distributions import Categorical
from typing import Union
import torch


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


class Prior(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.register_buffer('prior_mean', torch.zeros(input_size))
        self.register_buffer('prior_var', torch.ones(input_size))

    @property
    def distribution(self):
        return torch.distributions.Normal(self.prior_mean, self.prior_var)

    def log_prob(self, x):
        log_prob = self.distribution.log_prob(x)
        return log_prob

    def sample(self, samples=1):
        return self.distribution.rsample((samples,))


class InvertibleNetwork(torch.nn.Module):
    def __init__(self, layers: Union[list, tuple], prior: Prior):
        super().__init__()

        self.prior = prior
        self.flows = SequentialFlow(*layers)

    def forward(self, x, y=None, return_step=False):

        if return_step:
            u, logdet = self.flows.forward_steps(x, y)
        else:
            u, logdet = self.flows.forward(x, y)

        return u, torch.flatten(logdet, 1)

    def backward(self, u, y=None, return_step=False):
        if return_step:
            x, logdet = self.flows.backward_steps(u, y)
        else:
            x, logdet = self.flows.backward(u, y)

        return x, torch.flatten(logdet, 1)

    def sample(self, samples=1, u=None, y=None, return_step=False, return_logdet=False):
        self.eval()
        if u is None:
            u = self.prior.sample(samples)
        z, det = self.backward(u, y=y, return_step=return_step)
        if return_logdet:
            return z, det
        return z

    def logprob(self, x):
        return self.prior.log_prob(x)


class NoneConditioner(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.size = 0

    def forward(self, *args, **kwargs):
        return None


class Conditioner(nn.Module):
    def __init__(self, config: dict, classes):  # , emb_dim: Union[int, Tuple[int, int, int]]):
        super().__init__()

        dimension = config.get('dimension', None)

        self.classes = classes

        self._embeddings = None

        self.type = config['type']

        if self.type == 'one_hot':
            self.c = self.one_hot
            self.size = classes
        elif self.type == 'class_embedding':
            assert dimension is not None
            self._embeddings = torch.nn.Embedding(classes, dimension)
            self.c = self.embedding
            self.size = dimension
        else:
            assert False

    @property
    def embeddings(self):
        if self._embeddings is not None:
            return self._embeddings.weight
        return None

    def one_hot(self, y):
        c = torch.nn.functional.one_hot(y, self.classes)
        return c.float()

    def embedding(self, y):
        c = self._embeddings(y)
        return c

    def forward(self, y):
        return self.c(y)


class EmbeddingPreprocessing(nn.Module):
    def __init__(self, mean=0, std=1):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float))

    def set(self, mean, std):
        # self.mean = torch.tensor(mean, dtype=torch.float)
        # self.std = torch.tensor(std, dtype=torch.float)
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std
        # return (x - self.mean) / (self.std - self.mean)

    def backward(self, x):
        return (x * self.std) + self.mean
        # return (x * (self.std - self.mean)) + self.mean


def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)


def reconstruction_loss(x, x_hat, softclip=6, reduction='mean'):
    log_sigma = ((x - x_hat) ** 2).mean().sqrt().log()
    log_sigma = -softclip + torch.nn.functional.softplus(log_sigma + softclip)

    rec = gaussian_nll(x_hat, log_sigma, x)

    if reduction == 'mean':
        return rec.mean([1, 2, 3]).mean(0)
    return rec.sum()