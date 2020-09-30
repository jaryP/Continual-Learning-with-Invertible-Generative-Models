import logging

import itertools
from copy import deepcopy

import torch

# from continual_ai.continual_learning_strategies import NaiveMethod, Container
# from continual_ai.base import ExperimentConfig
from continual_ai.cl_strategies import NaiveMethod, Container
from continual_ai.utils import ExperimentConfig


class ElasticWeightConsolidation(NaiveMethod):
    """
    @article{kirkpatrick2017overcoming,
      title={Overcoming catastrophic forgetting in neural networks},
      author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
      journal={Proceedings of the national academy of sciences},
      volume={114},
      number={13},
      pages={3521--3526},
      year={2017},
      publisher={National Acad Sciences}
    }
    """

    def __init__(self, config: ExperimentConfig, logger: logging.Logger = None, **kwargs):
        super().__init__()
        self.config = config.cl_technique_config

        self.sample_size = self.config.get('sample_size', 200)
        self.importance = self.config.get('penalty_importance', 1e3)
        # self.batch_size = self.config.get('batch_size', config.train_config['batch_size'])

        if logger is not None:
            logger.info('EWC parameters:')
            logger.info(F'\tTask Sample size: {self.sample_size}')
            logger.info(F'\tPenalty importance: {self.importance}')
            # logger.info(F'\tBatch size: {self.batch_size}')

        self.memory = list()
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

    def on_task_ends(self, container: Container, *args, **kwargs):

        task = container.current_task
        # task = kwargs['task']

        # task_m = task.sample(size=self.sample_size, as_memory=True)

        final_w = {n: deepcopy(p.data) for n, p in itertools.chain(container.encoder.named_parameters())
                   if p.requires_grad and p.grad is not None}

        container.encoder.train()
        container.solver.train()
        task.train()

        container.encoder.zero_grad()
        container.solver.zero_grad()

        _s = 0

        # TODO: utilizzare il reale sample_size (come GEM)

        loss = 0
        for i, (_, image, label) in enumerate(task):
            emb = container.encoder(image)
            o = container.solver(emb, task=task.index)

            loss += self.loss(o, label)

            _s += label.shape[0]
            if _s >= self.sample_size:
                break

        loss.backward()

        f_matrix = {}
        for n, p in itertools.chain(container.encoder.named_parameters()):
            if p.requires_grad and p.grad is not None:
                f_matrix[n] = (deepcopy(p.grad.data) ** 2) / _s

        self.memory.append((final_w, f_matrix))

    def before_gradient_calculation(self, container: Container, *args, **kwargs):

        if len(self.memory) > 0:

            penalty = 0
            p = {n: deepcopy(p.data) for n, p in itertools.chain(container.encoder.named_parameters())
                 if p.requires_grad and p.grad is not None}

            for w, f in self.memory:
                for n in w.keys():
                    _loss = f[n] * (p[n] - w[n]) ** 2
                    penalty += _loss.sum()

            container.current_loss += penalty * self.importance


class OnlineElasticWeightConsolidation(NaiveMethod):
    # TODO: Da provare
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config.cl_technique_config

        self.task_size = self.config.get('task_size', 200)
        self.importance = self.config.get('penalty_importance', 1e3)
        self.batch_size = self.config.get('batch_size', config.train_config['batch_size'])
        # self.num_batches = self.config.get('num_batches', config.train_config['batch_size'])

        # self.model = model
        self.memory = list()
        self.loss = torch.nn.CrossEntropyLoss()

        self.f = {}
        self.final_w = {}

    def on_task_ends(self, container: Container, *args, **kwargs):

        task = container.current_task
        # task = kwargs['task']

        task.test()
        # task_m = task.sample(size=self.sample_size, as_memory=True)

        self.final_w = {n: deepcopy(p.data) for n, p in itertools.chain(container.encoder.named_parameters())
                        if p.requires_grad and p.grad is not None}

        container.encoder.train()
        container.solver.train()

        container.encoder.zero_grad()
        container.solver.zero_grad()

        _s = 0
        for i, (image, label) in enumerate(task):
            emb = container.encoder(image)
            o = container.solver(emb, task=task.index)
            self.loss(o, label).backward()

            _s += label.shape[0]
            if _s >= self.task_size:
                break

        # f_matrix = {}
        # for n, p in itertools.chain(container.encoder.named_parameters()):
        #     if p.requires_grad and p.grad is not None:
        #         f_matrix[n] = (deepcopy(p.grad.data) ** 2) / _s

        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.f[n] = self.f.get(n, 0) + (deepcopy(p.grad.data) ** 2) / _s

        # self.memory.append((final_w, f_matrix))

    def before_gradient_calculation(self, container: Container, *args, **kwargs):

        if container.current_task.index > 0:

            penalty = 0

            p = {n: deepcopy(p.data) for n, p in itertools.chain(container.encoder.named_parameters())
                 if p.requires_grad and p.grad is not None}

            for n in p.keys():
                _loss = self.f[n] * (p[n] - self.final_w[n]) ** 2
                penalty += _loss.sum()

            container.current_loss += penalty * self.importance


# class _OnlineElasticWeightConsolidation(NaiveMethod):
#     def __init__(self, model: torch.nn.Module, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.sample_size = kwargs.get('sample_size', 200)
#         self.importance = kwargs.get('penalty_importance', 1e3)
#         self.batch_size = kwargs.get('batch_size', 1)
#
#         self.model = model
#         self.memory = list()
#         self.loss = torch.nn.CrossEntropyLoss()
#
#         self.f = {}
#         self.final_w = {}
#
#     def on_task_ends(self, *args, **kwargs):
#         task = kwargs['task']
#
#         task.train()
#         task_m = task.sample(size=self.sample_size)
#
#         self.final_w = {n: deepcopy(p.data) for n, p in self.model.named_parameters()
#                         if p.requires_grad and p.grad is not None}
#
#         self.model.eval()
#         self.model.zero_grad()
#
#         for image, label in task_m(self.batch_size):
#             image = torch.FloatTensor(image).to(self.model.device)
#             label = torch.LongTensor(label).to(self.model.device)
#             o = self.model(image)
#             self.loss(o, label).backward()
#
#         for n, p in self.model.named_parameters():
#             if p.requires_grad and p.grad is not None:
#                 self.f[n] = self.f.get(n, 0) + (deepcopy(p.grad.data) ** 2) / len(task_m)
#
#     def before_gradient_calculation(self, *args, **kwargs):
#
#         if len(self.memory) > 0:
#             loss = kwargs['loss']
#
#             penalty = 0
#
#             p = {n: deepcopy(p.data) for n, p in self.model.named_parameters()
#                  if p.requires_grad and p.grad is not None}
#
#             for n in p.keys():
#                 curret_w = p[n]
#                 f = self.f[n]
#                 last_w = self.final_w[n]
#                 _loss = f[n] * (curret_w[n] - last_w[n]) ** 2
#                 penalty += _loss.sum()
#
#             loss += penalty * self.importance
