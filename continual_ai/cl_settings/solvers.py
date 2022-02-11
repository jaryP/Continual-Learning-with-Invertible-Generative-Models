__all__ = ['MultiHeadTaskSolver', 'SingleIncrementalTaskSolver']

import itertools
from abc import ABC, abstractmethod

from operator import mul

from functools import reduce

import numpy as np
import torch
from torch import nn


class Solver(ABC):

    @property
    @abstractmethod
    def current_task(self):
        raise NotImplementedError

    @current_task.setter
    @abstractmethod
    def current_task(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def tasks(self):
        raise NotImplementedError

    @abstractmethod
    def trainable_parameters(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def add_task(self, **kwargs):
        raise NotImplementedError


class MultiHeadTaskSolver(nn.Module):

    def __init__(self, input_dim, topology: callable = None):
        super(MultiHeadTaskSolver, self).__init__()

        if topology is None:
            topology = self.base_topology

        # if topology is None:
        #     topology = [input_dim]

        if hasattr(input_dim, '__len__') and len(input_dim) > 1:
            input_dim = reduce(mul, input_dim, 1)
            self.flat_input = True
        else:
            self.flat_input = False

        self._tasks = nn.ModuleList()
        self.input_dim = input_dim

        # if hasattr(input_dim, '__len__') and len(input_dim) > 1:
        #     input_dim = reduce(mul, input_dim, 1)

        self.classification_layer = None

        self.topology = topology
        self._task = 0

    def base_topology(self, ind, outd):
        return torch.nn.Sequential(*[torch.nn.Linear(ind, ind),
                                     torch.nn.Dropout(0.5),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(ind, ind // 4),
                                     torch.nn.Dropout(0.5),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(ind // 4, outd)])

    def add_task(self, output_size):
        self._tasks.append(self.topology(self.input_dim, output_size))

    @property
    def task(self):
        return self._task

    def trainable_parameters(self, t=None, recuse=True):
        if t is None:
            t = self.task
        th = self.heads[t]
        for param in th.parameters(recurse=recuse):
            yield param

    @property
    def heads(self):
        return self._tasks

    @task.setter
    def task(self, value):
        if value > len(self._tasks):
            raise ValueError('ERROR (MODIFICARE)')
        self._task = value

    def forward(self, x, task=None):

        _t = self.task
        if task is not None:
            _t = task

        if self.flat_input:
            x = torch.flatten(x, 1)

        x = self.heads[_t](x)

        return x


class SingleIncrementalTaskSolver(nn.Module):
    #TODO: to finish
    def __init__(self, input_dim, topology: callable = None, flat_input=False, device='cpu'):
        super(SingleIncrementalTaskSolver, self).__init__()

        self.device = device

        if topology is None:
            topology = self.base_topology

        self.flat_input = flat_input

        if hasattr(input_dim, '__len__') and len(input_dim) > 1:
            input_dim = reduce(mul, input_dim, 1)
            self.flat_input = True
        else:
            self.flat_input = False

        self.input_dim = input_dim

        self._task = 0

        self.__net = torch.nn.Sequential()

        self.last_layer_dimension = input_dim // 4

        net = topology(input_dim, self.last_layer_dimension)

        if self.flat_input:
            self.__net.add_module('flatten', nn.Flatten())

        self.__net.add_module('net', net)
        self.__net.to(self.device)

        self.register_buffer('initialization_done', torch.tensor(0))
        self.register_parameter('w', None)
        self.register_parameter('b', None)

    def base_topology(self, ind, outd):
        return torch.nn.Sequential(*[torch.nn.ReLU(),
                                     torch.nn.Linear(ind, ind),
                                     torch.nn.Dropout(0.2),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(ind, ind // 4),
                                     torch.nn.Dropout(0.2),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(ind // 4, outd)])

    def _get_wb(self, in_features, out_features):
        w = torch.empty(out_features, in_features, device=self.device)
        b = torch.empty(out_features, device=self.device)
        torch.nn.init.kaiming_uniform_(w, a=np.sqrt(5))
        bound = 1 / np.sqrt(in_features)
        torch.nn.init.uniform_(b, -bound, bound)
        return w, b

    def add_task(self, output_size):

        if self.initialization_done == 0:
            self.initialization_done += 1
            w, b, = self._get_wb(self.last_layer_dimension, output_size)
            w = torch.nn.Parameter(w, requires_grad=True)
            b = torch.nn.Parameter(b, requires_grad=True)
        else:
            current_shape = self.w.shape
            w, b, = self._get_wb(self.last_layer_dimension, current_shape[0] + output_size)
            w[:current_shape[0], :current_shape[1]] = self.w.data
            b[:current_shape[0]] = self.b.data
            w = torch.nn.Parameter(w, requires_grad=True)
            b = torch.nn.Parameter(b, requires_grad=True)

        self.register_parameter('w', w)
        self.register_parameter('b', b)

        return self

    @property
    def task(self):
        return self._task

    @property
    def heads(self):
        return self._tasks

    @task.setter
    def task(self, value):
        if value > len(self._tasks):
            raise ValueError('ERROR (MODIFICARE)')
        self._task = value

    def forward(self, x, task=None):

        x = self.__net(x)
        x = torch.relu(x)
        x = torch.nn.functional.linear(x, self.w, self.b)

        return x

    def load_state_dict(self, state_dict, strict=True):
        self.w = torch.nn.Parameter(state_dict['w'], requires_grad=True)
        self.b = torch.nn.Parameter(state_dict['b'], requires_grad=True)
        super(SingleIncrementalTaskSolver, self).load_state_dict(state_dict, strict=True)

    def trainable_parameters(self, t=None, recuse=True):
        for param in itertools.chain(self.__net.parameters(recuse), [self.w, self.b]):
            yield param
