from functools import reduce
from operator import mul

from torch import nn
from torch.distributions import Categorical
from typing import Union, Tuple
import torch


def get_linear_mask(dimension: int, mask_type: str = 'half'):
    mask_type = mask_type.lower()
    assert mask_type in ['half', 'odd']

    if mask_type == 'odd':
        return torch.arange(0, dimension).float() % 2
    else:
        mask = torch.zeros(dimension)
        mask[:dimension // 2] = 1
        return mask


def get_image_mask(dimension: Tuple[int, int, int], mask_type: str = 'channel'):
    mask_type = mask_type.lower()
    assert mask_type in ['channel', 'pixel']

    channels, height, width = dimension
    if mask_type == 'channel':
        mask = torch.ones(channels, 1, 1)
        mask[:channels // 2] = 0
        return mask
    else:
        mask = torch.ones(height, width)
        for i in range(height):
            for j in range(i % 2, width, 2):
                mask[i, j] = 0
        return mask


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
            u, _log_det = module.inverse(u, y=y)
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
            u, _log_det = module.inverse(u, y=y)
            us.append([u, module.__class__.__name__])
            log_det = log_det + _log_det
        return us, log_det


class Prior(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # self.to_flat = False

        # if isinstance(input_size, tuple):
        #     #     input_size = reduce(mul, input_size, 1)
        #     #
        #     # print(input_size)
        #     input_size = (input_size[0], input_size[1] * input_size[2])

        # if isinstance(input_size, (list, tuple)):
        #     self.register_buffer('prior_mean', torch.zeros(*input_size))
        #     self.register_buffer('prior_var', torch.ones(*input_size))
        # else:

        self.register_buffer('prior_mean', torch.zeros(input_size))
        self.register_buffer('prior_var', torch.ones(input_size))

        # self.input_size = input_size
        # self.register_buffer('prior_mean', torch.zeros(1))
        # self.register_buffer('prior_var', torch.ones(1))

    @property
    def distribution(self):
        return torch.distributions.Normal(self.prior_mean, self.prior_var)

    def log_prob(self, x):
        # if self.to_flat:
        #     x = torch.flatten(x, 1)
        log_prob = self.distribution.log_prob(x)
        return log_prob

    def sample(self, samples=1):
        return self.distribution.rsample((samples, ))


# def get_nf(config: dict, input_dim: Union[int, Tuple[int, int, int]], prior: Prior,
#            conditioning_size: int = 0, clamp: int = 2):
#     blocks = config['blocks']
#     t = config['type']
#     if t == 'linear':
#         nf = get_linear_nf(blocks=blocks, input_dim=input_dim, conditioning_size=conditioning_size, prior=prior,
#                            clamp=clamp)
#     else:
#         # nf = get_conv_nf(blocks=blocks, input_dim=input_dim, conditioning_size=conditioning_size, prior=prior,
#         #                  clamp=clamp)
#         assert False
#     return nf


# def get_linear_nf(blocks: int, input_dim: Union[int, Tuple[int, int, int]], prior: Prior, conditioning_size: int = 0,
#                   clamp: int = 2):
#     def linear_fc(ind, outd):
#         return torch.nn.Sequential(*[torch.nn.Linear(ind, ind * 2),
#                                      torch.nn.ReLU(),
#                                      # torch.nn.Linear(ind * 2, ind * 2),
#                                      # torch.nn.ReLU(),
#                                      torch.nn.Linear(ind * 2, outd),
#                                      ])
#
#     def conv_fc(ind, outd):
#         return torch.nn.Sequential(*[torch.nn.Conv2d(ind, 128, 3, 1, padding=1),
#                                      torch.nn.ReLU(),
#                                      # torch.nn.Linear(ind * 2, ind * 2),
#                                      # torch.nn.ReLU(),
#                                      torch.nn.Conv2d(128, 128, 3, 1, padding=1),
#                                      torch.nn.ReLU(),
#
#                                      torch.nn.Conv2d(128, outd, 1, 1, padding=0),
#                                      ])
#     layers = []
#
#     to_flatten = False
#
#     if isinstance(input_dim, tuple):
#         to_flatten = True
#         # layers.append(InvertibleFlatten(input_dim))
#         flat_dim = reduce(mul, input_dim, 1)
#         f = conv_fc
#     else:
#         flat_dim = input_dim
#         f = linear_fc
#
#     for i in range(blocks):
#         if to_flatten:
#             layers.append(BatchNorm2D(input_dim, momentum=0.9))
#             layers.append(CouplingLayer(input_dim, function=conv_fc, clamp=clamp,
#                                         conditioning_size=conditioning_size))
#             layers.append(Permutation(input_dim))
#         else:
#             layers.append(_BatchNorm(flat_dim, momentum=0.9))
#             layers.append(CouplingLayer(flat_dim, function=linear_fc, clamp=clamp,
#                                         conditioning_size=conditioning_size))
#             # layers.append(BatchNorm(flat_dim, momentum=0.9))
#             layers.append(Permutation(flat_dim))
#
#         # if to_flatten:
#         #     layers.append(InvertibleReshape(input_dim))
#
#     # layers.append(InvertibleFlatten(flat_dim))
#
#     return InvertibleNetwork(layers, prior=prior)
#
#
# def get_conv_nf(blocks: int, input_dim: Tuple[int, int, int], prior: Prior, conditioning_size: int = 0):
#     def conv_fc(ind, outd):
#         return torch.nn.Sequential(*[torch.nn.Conv2d(ind, ind * 2, stride=1, kernel_size=3, padding=1),
#                                      torch.nn.ReLU(),
#                                      torch.nn.Conv2d(ind * 2, outd, stride=1, kernel_size=3, padding=1)])
#
#     layers = []
#
#     # if isinstance(input_dim, tuple):
#     #     layers.append(InvertibleFlatten(input_dim))
#     #     input_dim = reduce(mul, input_dim, 1)
#
#     mask = get_image_mask(input_dim, 'pixel')
#
#     for i in range(blocks):
#
#         layers.append(MaskedCouplingLayer(input_dim, mask, function=conv_fc, clamp=2,
#                                           conditioning_size=conditioning_size))
#
#         if i != blocks - 1:
#             layers.append(BatchNorm(input_dim))
#
#             # layers.append(BatchNorm(input_dim))
#             # layers.append(Permutation(input_dim))
#             layers.append(Invertible1x1Conv(input_dim))
#
#     layers.append(InvertibleFlatten(input_dim))
#
#     return InvertibleNetwork(layers, prior=prior)


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


class Conditioner(nn.Module):
    def __init__(self, config: dict, classes, emb_dim: Union[int, Tuple[int, int, int]]):
        super().__init__()

        conditioning_config = config['conditioning_config']
        dimension = conditioning_config['dimension']

        # self.linear = config['type'] == 'linear'
        self.classes = classes
        self.size = conditioning_config['dimension']
        self.emb_dim = emb_dim

        if isinstance(emb_dim, tuple):
            self.linear = False
        else:
            self.linear = True

        self._embeddings = None

        self.type = conditioning_config['type']

        if self.type == 'one_hot':
            self.c = self.one_hot
            self.size = classes
        elif self.type == 'class_embedding':
            self._embeddings = torch.nn.Embedding(classes, dimension)
            self.c = self.embedding
            self.size = dimension
        elif self.type == 'none':
            self.c = lambda x: None
            self.size = 0
        else:
            assert False

    @property
    def embeddings(self):
        if self._embeddings is not None:
            return self._embeddings.weight
        return None

    def one_hot(self, y):
        c = torch.nn.functional.one_hot(y, self.classes)
        # if not self.linear:
        #     c = c.unsqueeze(-1).unsqueeze(-1).expand((-1, -1, self.emb_dim[-2], self.emb_dim[-1]))
        return c.float()

    def embedding(self, y):
        c = self._embeddings(y)
        if not self.linear:
            assert False
        return c

    def forward(self, y):
        return self.c(y)


# @torch.no_grad()
# def _modify_batch(x, y, decoder, generative_model, labels, conditioner, zero_prob=0.5, return_mask_embeddings=False):
#     conditioner.eval()
#     decoder.eval()
#     conditioner.eval()
#
#     p = 1 - zero_prob
#     binomial = torch.distributions.binomial.Binomial(probs=p)
#
#     batch_size = x.shape[0]
#     mask = binomial.sample((batch_size,)).to(x.device)
#
#     probs = torch.zeros(max(labels) + 1, device=x.device)
#     for i in labels:
#         probs[i] = 1
#
#     m = Categorical(probs)
#     y_sampled = m.sample(torch.Size([batch_size])).to(x.device)
#
#     embs = generative_model.sample(batch_size, y=conditioner(y_sampled))
#
#     z = decoder(embs)
#
#     x_mask = mask.clone()
#     for i in range(len(x.shape[1:])):
#         x_mask.unsqueeze_(-1)
#
#     x = x * x_mask + z * (1 - x_mask)
#     y = y * mask + y_sampled.long() * (1 - mask)
#     y = y.long()
#
#     if return_mask_embeddings:
#         return x, y, mask, embs
#
#     return x, y
#
#
# @torch.no_grad()
# def _generate_batch(size, reconstructor, generative_model, labels, conditioner, device):
#     reconstructor.eval()
#     generative_model.eval()
#     conditioner.eval()
#
#     probs = torch.zeros(max(labels) + 1, device=device)
#     for i in labels:
#         probs[i] = 1
#
#     m = Categorical(probs)
#     y_sampled = m.sample(torch.Size([size]))
#
#     embs = generative_model.sample(size, y=conditioner(y_sampled))
#
#     z = reconstructor(embs)
#     y = y_sampled.long()
#
#     return z, y, embs


def modify_batch(x, y, decoder, generative_model, labels, conditioner, prior, zero_prob=0.5, return_mask_embeddings=False):
    conditioner.eval()
    decoder.eval()
    conditioner.eval()
    prior.eval()
    generative_model.eval()

    # p = 1 - zero_prob
    binomial = torch.distributions.binomial.Binomial(probs=zero_prob)

    batch_size = x.shape[0]
    mask = binomial.sample((batch_size,)).to(x.device)

    probs = torch.zeros(max(labels) + 1, device=x.device)
    for i in labels:
        probs[i] = 1

    m = Categorical(probs)
    y_sampled = m.sample(torch.Size([batch_size])).to(x.device)

    # embs = generative_model.sample(batch_size, y=conditioner(y_sampled))
    u = prior.sample(batch_size)
    embs, _ = generative_model.backward(u, y=conditioner(y_sampled))
    # embs = torch.cat([embs, conditioner(y_sampled)], 1)

    z = decoder(embs)

    assert not torch.isnan(z).any(), 'Old Emb Images NaN'

    x_mask = mask.clone()
    for i in range(len(x.shape[1:])):
        x_mask.unsqueeze_(-1)

    x = x * x_mask + z * (1 - x_mask)
    y = y * mask + y_sampled.long() * (1 - mask)
    y = y.long()

    if return_mask_embeddings:
        return x, y, mask, embs

    return x, y


def generate_batch(size, reconstructor, generative_model, labels, conditioner, prior, device):
    reconstructor.eval()
    generative_model.eval()
    conditioner.eval()
    prior.eval()

    probs = torch.zeros(max(labels) + 1, device=device)
    for i in labels:
        probs[i] = 1

    m = Categorical(probs)
    y_sampled = m.sample(torch.Size([size]))

    # embs = generative_model.sample(size, y=conditioner(y_sampled))
    u = prior.sample(size)
    embs, _ = generative_model.backward(u, y=conditioner(y_sampled))
    # embs, _ = generative_model.backward(size, y=conditioner(y_sampled))

    z = reconstructor(embs)
    y = y_sampled.long()

    return z, y, embs
