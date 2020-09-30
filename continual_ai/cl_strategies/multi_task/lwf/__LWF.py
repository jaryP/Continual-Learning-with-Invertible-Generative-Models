import numpy as np
import torch
from typing import Union

import logging

from continual_ai.cl_strategies import NaiveMethod, Container
from .utils import KnowledgeDistillationLoss
from continual_ai.utils import ExperimentConfig


class LearningWithoutForgetting(NaiveMethod):
    #TODO: citazione
    """
    """

    def __init__(self, config: ExperimentConfig, logger: logging.Logger = None,
                 random_state: Union[np.random.RandomState, int] = None,
                 **kwargs):

        super().__init__()

        lwf_config = config.cl_technique_config

        self.memory_size = lwf_config.get('memory_size', 100)
        self.alpha = lwf_config.get('alpha', 0.5)
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
            logger.info(F'\tMemory size: {self.memory_size}')
            logger.info(F'\tAlpha: {self.alpha}')
            logger.info(F'\tTemperature: {temperature}')

        self.task_memory = []
        self.loss_f = KnowledgeDistillationLoss(temperature)

    # def on_task_starts(self, container: Container, *args, **kwargs):
    #     for i in range(container.current_task.index):
    #         container.optimizer.add_param_group({'params': container.solver.trainable_parameters(i)})

    def on_task_ends(self, container: Container, *args, **kwargs):

        task = container.current_task

        task.train()
        images, labels = task.sample(size=self.memory_size)

        container.encoder.eval()
        container.solver.eval()

        predictions = container.solver(container.encoder(images))

        a = list(zip(images.detach(), predictions.detach()))

        self.task_memory.append(a)

    def after_back_propagation(self, container: Container, *args, **kwargs):

        if len(self.task_memory) > 0:

            for i, t in enumerate(self.task_memory):

                images, predictions = zip(*t)

                images = torch.stack(images)
                predictions = torch.stack(predictions)

                container.encoder.eval()
                container.solver.eval()

                new_predictions = container.solver(container.encoder(images), task=i)

                loss = self.loss_f(new_predictions, predictions)

                container.current_loss *= (1 - self.alpha)
                container.current_loss += self.alpha * loss

