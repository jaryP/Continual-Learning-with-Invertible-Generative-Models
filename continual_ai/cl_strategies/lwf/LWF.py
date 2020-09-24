from collections import defaultdict

import numpy as np
import torch
from typing import Union

import logging

from continual_ai.cl_strategies import NaiveMethod, Container
from .utils import KnowledgeDistillationLoss
from continual_ai.utils import ExperimentConfig


class LearningWithoutForgetting(NaiveMethod):
    """
    @article{li2017learning,
      title={Learning without forgetting},
      author={Li, Zhizhong and Hoiem, Derek},
      journal={IEEE transactions on pattern analysis and machine intelligence},
      volume={40},
      number={12},
      pages={2935--2947},
      year={2017},
      publisher={IEEE}
    }
    """
    
    def __init__(self, config: ExperimentConfig, logger: logging.Logger = None,
                 random_state: Union[np.random.RandomState, int] = None,
                 **kwargs):

        super(LearningWithoutForgetting, self).__init__()

        lwf_config = config.cl_technique_config

        self.alpha = lwf_config.get('alpha', 1)

        assert 0 <= self.alpha <= 1

        temperature = lwf_config.get('temperature', 1)

        if random_state is None or isinstance(random_state, int):
            self.RandomState = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.RandomState = random_state
        else:
            raise ValueError("random_state can be None, Int or numpy RandomState object, an {} was give"
                             .format(type(random_state)))

        if logger is not None:
            logger.info('LWF parameters:')
            # logger.info(F'\tMemory size: {self.memory_size}')
            logger.info(F'\tAlpha: {self.alpha}')
            logger.info(F'\tTemperature: {temperature}')

        self.task_memory = defaultdict(dict)
        self.loss_f = KnowledgeDistillationLoss(temperature)

    # def on_task_starts(self, container: Container, *args, **kwargs):
    #     for i in range(container.current_task.index):
    #         container.optimizer.add_param_group({'params': container.solver.trainable_parameters(i)})
    # def on_task_ends(self, container: Container, *args, **kwargs):
    #
    #     task = container.current_task
    #
    #     task.train()
    #     images, labels = task.sample(size=self.memory_size)
    #-r cd ..
    
    #     container.encoder.eval()
    #     container.solver.eval()
    #
    #     predictions = container.solver(container.encoder(images))
    #
    #     a = list(zip(images.detach(), predictions.detach()))
    #
    #     self.task_memory.append(a)
    
    def on_task_starts(self, container: Container, *args, **kwargs):

        self.task_memory = defaultdict(dict)
        task = container.current_task

        if task.index > 0:
            task.train()

            with torch.no_grad():
                for t in range(task.index):
                    for indexes, x, y in task:
                        pred = container.solver(container.encoder(x), task=t).detach()
                        for i, p in zip(indexes, pred):
                            self.task_memory[t][i] = p

        # task.train()
        # images, labels = task.sample(size=self.memory_size)
        #
        # container.encoder.eval()
        # container.solver.eval()
        #
        # predictions = container.solver(container.encoder(images))
        #
        # a = list(zip(images.detach(), predictions.detach()))
        #
        # self.task_memory.append(a)

    def before_gradient_calculation(self, container: Container, *args, **kwargs):

        if len(self.task_memory) > 0:

            indexes, x, y = container.current_batch
            # a = self.task_memory.values()
            tot_loss = 0

            for task, v in self.task_memory.items():
                new_preds = container.solver(container.encoder(x), task=task)

                old_preds = torch.zeros_like(new_preds)

                for i, idx in enumerate(indexes):
                    old_preds[i] = v[idx]

                tot_loss += self.loss_f(new_preds, old_preds)

            # for i, t in enumerate(self.task_memory):
            #
            #     images, predictions = zip(*t)
            #
            #     images = torch.stack(images)
            #     predictions = torch.stack(predictions)
            #
            #     container.encoder.eval()
            #     container.solver.eval()
            #
            #     new_predictions = container.solver(container.encoder(images), task=i)

            # print(tot_loss)
            # container.current_loss *= (1 - self.alpha)
            container.current_loss *= (1 - self.alpha)
            container.current_loss += self.alpha * tot_loss
