import logging

import itertools
from copy import deepcopy

import torch
from continual_ai.cl_strategies import NaiveMethod, Container
from continual_ai.utils import ExperimentConfig


class ElasticWeightConsolidation(NaiveMethod):
    """
    @article{kirkpatrick2017overcoming,
      title={Overcoming catastrophic forgetting in neural networks},
      author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume
              and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago
              and Grabska-Barwinska, Agnieszka and others},
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

        self.importance = self.config.get('importance', 1e3)

        if logger is not None:
            logger.info('EWC parameters:')
            logger.info(F'\tPenalty importance: {self.importance}')

        self.memory = list()
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

    def on_task_ends(self, container: Container, *args, **kwargs):

        task = container.current_task

        final_w = {n: deepcopy(p.data) for n, p in itertools.chain(container.encoder.named_parameters())
                   if p.requires_grad and p.grad is not None}

        container.encoder.train()
        container.solver.train()
        task.train()

        container.encoder.zero_grad()
        container.solver.zero_grad()

        _s = 0
        cumloss = 0

        for i, (_, image, label) in enumerate(task):
            emb = container.encoder(image)
            o = container.solver(emb, task=task.index)
            # cumloss += self.loss(o, label)
            self.loss(o, label).backward()
            _s += image.size(0)

        # cumloss = cumloss / _s
        # cumloss.backward()

        f_matrix = {}
        for n, p in itertools.chain(container.encoder.named_parameters()):
            if p.requires_grad and p.grad is not None:
                f_matrix[n] = (deepcopy(p.grad.data) ** 2) / _s

        self.memory.append((final_w, f_matrix))

    def before_gradient_calculation(self, container: Container, *args, **kwargs):

        if len(self.memory) > 0:

            penalty = 0
            p = {n: p for n, p in container.encoder.named_parameters()
                 if p.requires_grad and p.grad is not None}
            
            for w, f in self.memory:
                for n in w.keys():
                    _loss = f[n] * (p[n] - w[n]) ** 2
                    penalty += _loss.sum() * self.importance

            container.current_loss += penalty


class OnlineElasticWeightConsolidation(NaiveMethod):
    # TODO: Da provare
    def __init__(self, config: ExperimentConfig, **kwargs):
        super().__init__()
        self.config = config.cl_technique_config

        # self.task_size = self.config.get('task_size', 200)
        self.importance = self.config.get('penalty_importance', 1e3)
        # self.batch_size = self.config.get('batch_size', config.train_config['batch_size'])
        # self.num_batches = self.config.get('num_batches', config.train_config['batch_size'])

        # self.model = model
        self.memory = list()
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

        self.f = {}
        self.final_w = {}

    def on_task_ends(self, container: Container, *args, **kwargs):

        task = container.current_task
        # task = kwargs['task']

        task.train()
        # task_m = task.sample(size=self.sample_size, as_memory=True)

        self.final_w = {n: deepcopy(p.data) for n, p in container.encoder.named_parameters()
                        if p.requires_grad and p.grad is not None}

        container.encoder.train()
        container.solver.train()

        container.encoder.zero_grad()
        container.solver.zero_grad()

        _s = 0
        cumloss = 0

        for i, (_, image, label) in enumerate(task):
            # _, images, labels = task.sample(size=self.task_siz)
            emb = container.encoder(image)
            o = container.solver(emb, task=task.index)
            cumloss += self.loss(o, label)
            _s += image.size(0)
        # self.loss(o, labels).backward()
        # cumloss = cumloss / _s
        cumloss.backward()

        # for i, (image, label) in enumerate(task):
        #     emb = container.encoder(image)
        #     o = container.solver(emb, task=task.index)
        #     self.loss(o, label).backward()
        #
        #     _s += label.shape[0]
        #     if _s >= self.task_size:
        #         break

        # f_matrix = {}
        # for n, p in itertools.chain(container.encoder.named_parameters()):
        #     if p.requires_grad and p.grad is not None:
        #         f_matrix[n] = (deepcopy(p.grad.data) ** 2) / _s

        for n, p in container.encoder.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.f[n] = self.f.get(n, 0) + ((deepcopy(p.grad.data) ** 2) / _s)

        # self.memory.append((final_w, f_matrix))

    def before_gradient_calculation(self, container: Container, *args, **kwargs):

        if container.current_task.index > 0:

            penalty = 0

            p = {n: deepcopy(p.data) for n, p in container.encoder.named_parameters()
                 if p.requires_grad and p.grad is not None}

            for n in p.keys():
                _loss = self.f[n] * (p[n] - self.final_w[n]) ** 2
                penalty += _loss.sum()

            container.current_loss += (penalty * self.importance)