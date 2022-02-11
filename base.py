from functools import reduce
from operator import mul

import numpy as np
import torch
from torch import nn

from continual_ai.cl_settings.base import ClassificationTask

from continual_ai.datasets import MNIST, SVHN, CIFAR10, CIFAR100


def get_dataset(dataset_name, device):
    transformer = lambda x: (torch.tensor(np.array(x) / 255.0,
                                                    dtype=torch.float,
                                                    device=device))
    target_transformer = lambda y: (torch.tensor(y, device=device))

    if dataset_name == 'mnist':
        dataset = MNIST(transformer=transformer,
                        target_transformer=target_transformer)

    elif dataset_name == 'cifar10':
        dataset = CIFAR10(transformer=transformer,
                          target_transformer=target_transformer)

    elif dataset_name == 'cifar100':
        dataset = CIFAR100(transformer=transformer,
                           target_transformer=target_transformer)

    # elif dataset == 'kmnist':
    #     dataset = KMNIST(transformer=transformer,
    #                     target_transformer=target_transformer)
    # #
    # elif dataset == 'k49mnist':
    #     dataset = K49MNIST()

    elif dataset_name == 'svhn':
        dataset = SVHN(transformer=transformer,
                       target_transformer=target_transformer)
    # else:
    #     raise ValueError('The dataset parameters can be {}'.format(['mnist', 'cifar10', 'svhn', 'kmist', 'k49mnist']))

    classes = len(dataset.labels)

    sample_shape = dataset.sample_dimension
    image_channels = sample_shape[0]
    image_shape = sample_shape[-1]

    return dataset, transformer, image_channels, image_shape, classes


def classification_score_on_task(encoder: torch.nn.Module, solver: torch.nn.Module,
                                 task: ClassificationTask, evaluate_task_index: int = None):
    with torch.no_grad():
        encoder.eval()
        solver.eval()
        task.set_labels_type('task')

        true_labels = []
        predicted_labels = []

        if evaluate_task_index is None:
            evaluate_task_index = task.index

        for x, y in task:
            true_labels.extend(y)
            emb = encoder(x)
            a = solver(emb, task=evaluate_task_index)
            predicted_labels.extend(torch.nn.functional.softmax(a, dim=1).max(dim=1)[1].tolist())

        eq = np.asarray(true_labels) == np.asarray(predicted_labels)

        score = eq.sum() / len(eq)

    return score


def get_predictions(encoder: torch.nn.Module, solver: torch.nn.Module,
                    task: ClassificationTask, evaluate_task_index: int = None):
    with torch.no_grad():
        encoder.eval()
        solver.eval()
        task.set_labels_type('task')

        true_labels = []
        predicted_labels = []

        if evaluate_task_index is None:
            evaluate_task_index = task.index

        for j, x, y in task:
            true_labels.extend(y.tolist())
            emb = encoder(x)
            a = solver(emb, task=evaluate_task_index)
            predicted_labels.extend(a.max(dim=1)[1].tolist())

    return np.asarray(true_labels), np.asarray(predicted_labels)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view((-1, *self.shape))


class Encoder(nn.Module):
    def __init__(self, dataset, proj_dim=None, cond_size=0):
        super().__init__()
        if dataset == 'mnist':
            if proj_dim is None:
                proj_dim = 100

            encoder = nn.Sequential(
                nn.Conv2d(1, 12, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 24, 4, stride=2, padding=0),
                # nn.ReLU()
            )

            embedding_dim = encoder(torch.rand((1, 1, 28, 28))).shape[1:]
            flat_embedding_dim = reduce(mul, embedding_dim, 1)
            projector = nn.Linear(flat_embedding_dim + cond_size, proj_dim)

            # encoder.add_module('flatten', nn.Flatten()
            #                    )

        elif dataset == 'svhn' or dataset == 'cifar10' or dataset == 'cifar100':
            if proj_dim is None:
                if dataset == 'svhn':
                    proj_dim = 400
                else:
                    proj_dim = 200

            encoder = nn.Sequential(
                nn.Conv2d(3, 24, 3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(24, 24, 3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(24, 48, 3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(48, 48, 4, stride=2, padding=0),
                # nn.ReLU()
            )

            embedding_dim = encoder(torch.rand((1, 3, 32, 32))).shape[1:]
            flat_embedding_dim = reduce(mul, embedding_dim, 1)
            projector = nn.Linear(flat_embedding_dim + cond_size, proj_dim)
            # rec_projector = nn.Linear(flat_embedding_dim + cond_size, proj_dim)

            # encoder.add_module('flatten', nn.Flatten())

        else:
            assert False

        self.encoder = encoder

        # self.class_projector = nn.Linear(flat_embedding_dim + cond_size, proj_dim)
        # self.rec_projector = nn.Linear(flat_embedding_dim + cond_size, proj_dim)

        self.embedding_dim_before_projection = embedding_dim
        self.flat_cnn_size = flat_embedding_dim
        self.embedding_dim = proj_dim

        self.projector = projector

    # def reconstruction(self):
    #     self.projector = self.rec_projector
    #
    # def classification(self):
    #     self.projector = self.class_projector
    def cnn(self, x):
        emb = self.encoder(x)

        return emb

    def flatten_cnn(self, x):
        emb = self.encoder(x)
        emb = torch.flatten(emb, 1)

        return emb

    def forward(self, x, y=None):
        emb = self.flatten_cnn(x)

        if self.projector is not None:
            if y is not None:
                emb = self.projector(torch.cat((emb, y), -1))
            else:
                emb = self.projector(emb)

        return emb


class Decoder(nn.Module):
    def __init__(self, dataset, encoder: Encoder, cond_size=0):
        super().__init__()

        proj_dim = encoder.embedding_dim
        embedding_dim_before_projection = encoder.embedding_dim_before_projection
        flat_embedding_dim_before_projection = reduce(mul, embedding_dim_before_projection, 1)

        if dataset == 'mnist':
            decoder = nn.Sequential(
                nn.ReLU(),
                nn.Linear(proj_dim + cond_size, flat_embedding_dim_before_projection),
                Reshape(embedding_dim_before_projection),
                nn.ReLU(),
                nn.ConvTranspose2d(24, 12, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(12, 1, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )

        elif dataset == 'svhn' or dataset == 'cifar10' or dataset == 'cifar100':

            decoder = nn.Sequential(
                # nn.ReLU(),
                nn.Linear(proj_dim + cond_size, flat_embedding_dim_before_projection),
                Reshape(embedding_dim_before_projection),
                nn.ReLU(),
                nn.ConvTranspose2d(48, 48, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(48, 24, 3, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(24, 24, 3, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(24, 3, 3, stride=1, padding=0),
                nn.Sigmoid(),
            )
        else:
            assert False

        self.decoder = decoder

    def forward(self, emb, y=None):
        if y is not None:
            emb = torch.cat((emb, y), -1)
        x = self.decoder(emb)

        return x
