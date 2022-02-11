import logging
import numpy as np

from typing import Union

import torch

import torch.nn.functional as F
# from continual_ai.continual_learning_strategies.base import NaiveMethod, Container
# from continual_ai.base import ExperimentConfig
# from continual_ai.utils import Sampler
from continual_ai.cl_strategies import NaiveMethod, Container
from continual_ai.iterators import Sampler
from continual_ai.utils import ExperimentConfig


class EmbeddingRegularization(NaiveMethod):
    # TODO: implementare SIT
    # TODO: implementare pesi associati alle immagini
    """
    @article{POMPONI2020,
    title = "Efficient continual learning in neural networks with embedding regularization",
    journal = "Neurocomputing",
    year = "2020",
    issn = "0925-2312",
    doi = "https://doi.org/10.1016/j.neucom.2020.01.093",
    url = "http://www.sciencedirect.com/science/article/pii/S092523122030151X",
    author = "Jary Pomponi and Simone Scardapane and Vincenzo Lomonaco and Aurelio Uncini",
    keywords = "Continual learning, Catastrophic forgetting, Embedding, Regularization, Trainable activation functions",
    }
    """
    def __init__(self, config: ExperimentConfig, logger: logging.Logger=None,
                 random_state: Union[np.random.RandomState, int] = None, **kwargs):

        NaiveMethod.__init__(self)
        config = config.cl_technique_config

        self.memorized_task_size = config.get('task_memory_size', 300)
        self.sample_size = min(config.get('sample_size', 100), self.memorized_task_size)
        self.importance = config.get('penalty_importance', 1)
        self.distance = config.get('distance', 'cosine')
        # self.supervised = config.get('supervised', True)
        self.normalize = config.get('normalize', False)
        self.batch_size = config.get('batch_size', 25)

        if random_state is None or isinstance(random_state, int):
            self.RandomState = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.RandomState = random_state
        else:
            raise ValueError("random_state can be None, Int or numpy RandomState object, an {} was give"
                             .format(type(random_state)))

        if logger is not None:
            logger.info('ER parameters:')
            logger.info(F'\tMemorized task size: {self.memorized_task_size}')
            logger.info(F'\tSample size: {self.sample_size}')
            logger.info(F'\tPenalty importance: {self.importance}')
            logger.info(F'\tDistance: {self.distance}')
            logger.info(F'\tNormalize: {self.normalize}')

        self.task_memory = []

    def on_task_ends(self, container: Container, *args, **kwargs):

        task = container.current_task

        task.train()

        _, images, _ = task.sample(size=self.memorized_task_size)

        container.encoder.eval()

        embs = container.encoder(images)
        if self.normalize:
            embs = F.normalize(embs, p=2, dim=1)

        m = list(zip(images.detach(), embs.detach()))

        self.task_memory.append(m)

    def before_gradient_calculation(self, container: Container, *args, **kwargs):

        if len(self.task_memory) > 0:

            to_back = []
            loss = 0

            for t in self.task_memory:

                b = Sampler(t, dimension=self.sample_size, replace=False, return_indexes=False)()
                image, embeddings = zip(*b)

                image = torch.stack(image)
                embeddings = torch.stack(embeddings)
                # print(image.shape)

                new_embedding = container.encoder(image)

                if self.normalize:
                    new_embedding = F.normalize(new_embedding, p=2, dim=1)

                if self.distance == 'euclidean':
                    dist = (embeddings - new_embedding).norm(p=None, dim=1)
                elif self.distance == 'cosine':
                    cosine = torch.nn.functional.cosine_similarity(embeddings, new_embedding, dim=1)
                    dist = 1 - cosine
                else:
                    assert False

                # to_back.append(dist)

                loss += dist.mean() * self.importance

            # to_back = torch.cat(to_back)
            # loss *= self.importance
            # container.current_loss += torch.mul(to_back.mean(), self.importance)
            container.current_loss += loss

# class _EmbeddingReguralization(NaiveMethod):
#     def __init__(self, model: torch.nn.Module, **kwargs):
#         NaiveMethod.__init__(self)
#         self.model = model
#
#         self.memorized_task_size = kwargs.get('memorized_task_size', 300)
#         self.sample_size = min(kwargs.get('sample_size', 100), self.memorized_task_size)
#         self.importance = kwargs.get('penalty_importance', 10)
#         self.distance = kwargs.get('distance', 'cosine')
#         self.supervised = kwargs.get('supervised', True)
#         self.normalize = kwargs.get('normalize', True)
#         self.batch_size = kwargs.get('batch_size', 25)
#
#         self.task_memory = []
#         self.device = self.model.device
#
#     def on_task_ends(self, *args, **kwargs):
#
#         task = kwargs['task']
#
#         task.train()
#         images, _ = task.sample(size=self.memorized_task_size)
#         images = torch.Tensor(images)
#
#         self.model.eval()
#
#         images = images.to(self.device)
#
#         m = WeightedMemory()
#
#         with torch.no_grad():
#
#             for i in images:
#                 output = self.model.embedding(i.unsqueeze(0))
#
#                 if self.normalize:
#                     output = F.normalize(output, p=2, dim=1)
#
#                 embeddings = output.cpu()
#
#                 m.add((i.cpu(), embeddings.cpu()))
#
#             self.task_memory.append(m)
#
#         self.model.train()
#
#     def before_gradient_calculation(self, *args, **kwargs):
#
#         if len(self.task_memory) > 0:
#             self.model.eval()
#
#             to_back = []
#             for t in self.task_memory:
#                 m = t.sample(size=self.sample_size)
#                 for image, embedding in m(batch_size=self.batch_size, shuffle=True):
#
#                     image = torch.stack(image, 0).to(self.device)
#                     embedding = torch.cat(embedding, 0)
#
#                     new_embedding = self.model.embedding(image)
#
#                     if self.normalize:
#                         new_embedding = F.normalize(new_embedding, p=2, dim=1)
#
#                     new_embedding = new_embedding.cpu()
#                     if self.distance == 'euclidean':
#                         dist = (embedding - new_embedding).norm(p=None, dim=1)
#                     elif self.distance == 'cosine':
#                         cosine = torch.nn.functional.cosine_similarity(embedding, new_embedding)
#                         dist = 1 - cosine
#
#                     to_back.append(dist)
#
#             to_back = torch.cat(to_back).to(self.device)
#             # torch.mul(to_back.mean(), self.importance).backward()
#
#             # print('prima', kwargs['loss'])
#             kwargs['loss'] += torch.mul(to_back.mean(), self.importance)
#             # print('dopo', kwargs['loss'])
#
#             self.model.train()
