from functools import reduce
from operator import mul

import numpy as np
import torch
from torch import nn

from continual_ai.cl_settings.base import ClassificationTask

from continual_ai.datasets import MNIST, SVHN, CIFAR10  # , CIFAR10, KMNIST, K49MNIST, SVHN
# from model import Glow


def get_dataset(dataset_name, device):
    # preprocessing = lambda x: (torch.tensor(x[0], dtype=torch.float, device=device) / 255,
    #                            torch.tensor(x[1], dtype=torch.long, device=device))

    # transformer = lambda x: (torch.tensor(x / 255, dtype=torch.float32))

    transformer = lambda x: (torch.tensor(x, dtype=torch.float32, device=device) / 255)
    target_transformer = lambda y: (torch.tensor(y, device=device))

    if dataset_name == 'mnist':
        dataset = MNIST(transformer=transformer,
                        target_transformer=target_transformer)

    elif dataset_name == 'cifar10':
        dataset = CIFAR10(transformer=transformer,
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


def get_model(dataset):
    if dataset == 'mnist':

        encoder = nn.Sequential(
            nn.Conv2d(1, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=0),
            # nn.ReLU(),
            # nn.Conv2d(24, 24, 4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(24, 24, 3, stride=1, padding=1),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(24 * 6 * 6, 100),
            # nn.Tanh(),
        )

        decoder = nn.Sequential(
            # nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(100, 24 * 6 * 6),
            Reshape((24, 6, 6)),
            # nn.ReLU(),
            # nn.ConvTranspose2d(24, 24, 3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(24, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        embedding_dim = encoder(torch.rand((1, 1, 28, 28))).shape[1:]
        if len(embedding_dim) == 1:
            embedding_dim = embedding_dim[0]

    # elif dataset == 'cifar10':
    #     encoder = nn.Sequential(
    #         nn.Conv2d(3, 12, 4, stride=2, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(12, 24, 4, stride=2, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(24, 24, 4, stride=2, padding=1),
    #         # nn.ReLU(),
    #         # nn.Conv2d(24, 24, 3, stride=1, padding=0),
    #         nn.Flatten(),
    #     )
    #
    #     # embedding_dim = tuple(encoder(torch.rand((1, 3, 32, 32))).shape[1:])
    #     # print(embedding_dim)
    #     embedding_dim = encoder(torch.rand((1, 3, 32, 32))).shape[1]
    #
    #     decoder = nn.Sequential(
    #         Reshape((24, 4, 4)),
    #         # nn.ReLU(),
    #         # nn.ConvTranspose2d(24, 24, 3, stride=1, padding=0),
    #         nn.ReLU(),
    #         nn.ConvTranspose2d(24, 24, 4, stride=2, padding=1),
    #         nn.ReLU(),
    #         nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
    #         nn.ReLU(),
    #         nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
    #         nn.Sigmoid(),
    #     )

    # elif dataset == 'kmnist' or dataset == 'k49mnist':
    #
    #     encoder = nn.Sequential(
    #         nn.Conv2d(3, 24, 3, stride=1, padding=0),
    #         nn.ReLU(),
    #         nn.Conv2d(24, 24, 3, stride=1, padding=0),
    #         nn.ReLU(),
    #         nn.Conv2d(24, 48, 3, stride=1, padding=0),
    #         nn.ReLU(),
    #         nn.Conv2d(48, 48, 4, stride=2, padding=0),
    #         nn.ReLU(),
    #         nn.Conv2d(48, 48, 4, stride=2, padding=0),
    #     )
    #
    #     decoder = nn.Sequential(
    #         nn.ReLU(),
    #         nn.ConvTranspose2d(48, 48, 4, stride=2, padding=0),
    #         nn.ReLU(),
    #         nn.ConvTranspose2d(48, 48, 4, stride=2, padding=0),
    #         nn.ReLU(),
    #         nn.ConvTranspose2d(48, 24, 3, stride=1, padding=0),
    #         nn.ReLU(),
    #         nn.ConvTranspose2d(24, 24, 3, stride=1, padding=0),
    #         nn.ReLU(),
    #         nn.ConvTranspose2d(24, 3, 3, stride=1, padding=0),
    #         nn.Sigmoid(),
    #     )
    #
    #     embedding_dim = encoder(torch.rand((1, 1, 28, 28))).shape[1:]

    elif dataset == 'svhn' or dataset == 'cifar10':
        encoder = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(48, 48, 4, stride=2, padding=0),
            # nn.ReLU(),
            # nn.Conv2d(24, 24, 4, stride=2, padding=0),
        )

        tuple_dim = encoder(torch.rand((1, 3, 32, 32))).shape[1:]
        flat_tuple_dim = reduce(mul, tuple_dim, 1)
        # embedding_dim = flat_tuple_dim
        embedding_dim = 100
        # print(tuple_dim)

        encoder.add_module('flatten', nn.Flatten())
        encoder.add_module('linear', nn.Linear(flat_tuple_dim, embedding_dim))

        decoder = nn.Sequential(
            nn.Linear(embedding_dim, flat_tuple_dim),
            Reshape(tuple_dim),
            nn.ReLU(),
            # nn.ConvTranspose2d(48, 48, 4, stride=2, padding=0),
            # nn.ReLU(),
            nn.ConvTranspose2d(48, 48, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 24, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 3, stride=1, padding=0),
            nn.Sigmoid(),
        )

        # embedding_dim = encoder(torch.rand((1, 3, 32, 32))).shape[1:]

    else:
        raise ValueError('The dataset parameters can be {}'.format(['mnist', 'cifar10', 'kmnist', 'k49mnist']))

    return encoder, decoder, embedding_dim

