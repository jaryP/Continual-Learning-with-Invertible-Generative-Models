from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from torch.utils.data import BatchSampler, RandomSampler
from continual_ai.base import ClassificationDataset


class ClassificationTask(ClassificationDataset):
    def __init__(self, base_dataset: ClassificationDataset, task_labels: np.ndarray, labels_map: np.ndarray,
                 batch_size=16):

        self.__current_y = 'task'
        self.batch_size = batch_size
        self.__subset = 'train'
        self.index = None
        self.current_batch_size = None
        self.current_sampler = None

        self.dataset_labels = []
        self.task_labels = []

        x_train, old_y_train, y_train = self.extract_labels(base_dataset,
                                                            labels=task_labels, labels_map=labels_map, split='train')
        x_test, old_y_test, y_test = self.extract_labels(base_dataset,
                                                         labels=task_labels, labels_map=labels_map, split='test')
        x_dev, old_y_dev, y_dev = self.extract_labels(base_dataset,
                                                      labels=task_labels, labels_map=labels_map, split='dev')

        super(ClassificationTask, self).__init__(x_train=x_train, y_train=y_train, x_dev=x_dev, y_dev=y_dev,
                                                 x_test=x_test, y_test=y_test, transformer=None,
                                                 target_transformer=None)

        self.final_transformer = base_dataset.transformer
        self.final_target_transformer = base_dataset.target_transformer

        self._old_y_train = old_y_train
        self._old_y_test = old_y_test
        self._old_y_dev = old_y_dev

        self.t2d = dict(zip(y_train, old_y_train))
        self.d2t = dict(zip(old_y_train, y_train))

        for i, j in self.t2d.items():
            self.dataset_labels.append(j)
            self.task_labels.append(i)

    def extract_labels(self, dataset: ClassificationDataset, labels: np.ndarray, labels_map: np.ndarray, split='train'):
        assert split in ['train', 'test', 'dev']
        getattr(dataset, split)()

        if dataset.x is None:
            return dataset.x, dataset.y, dataset.y

        idx = np.arange(len(dataset))
        x, y = dataset.data

        idx_c = idx[np.in1d(y, labels)]

        x = x[idx_c]
        y = y[idx_c]

        new_y = labels_map[y]

        return x, y, new_y

    def set_labels_type(self, s):
        assert s in ['task', 'dataset']
        self.__current_y = s

    def _y(self):
        if self.__current_y == 'task':
            y = super(ClassificationTask, self)._y()
        else:
            y = getattr(self, F'_old_y_{self._split}')

        return y

    def __len__(self):
        return super(ClassificationTask, self).__len__()  # // self.batch_size + 1

    def __call__(self, batch_size=None, sampler=None):
        self.current_batch_size = batch_size
        # if current_sampler is None:
        #     self.current_sampler = RandomSampler(self)
        return self

    def __iter__(self):

        batch = self.batch_size
        sampler = RandomSampler(self)

        if self.current_batch_size is not None:
            batch = self.current_batch_size
            self.current_batch_size = None

        if self.current_sampler is not None:
            sampler = self.current_sampler
            self.current_sampler = None

        for idx in BatchSampler(batch_size=batch, sampler=sampler, drop_last=False):
            x, y, idxs = [], [], []
            for i in idx:
                _i, _x, _y = self[i]
                x.append(_x)
                y.append(_y)
                idxs.append(_i)

            if self.final_transformer is not None:
                x = self.final_transformer(x)

            if self.final_target_transformer is not None:
                y = self.final_target_transformer(y)

            yield idxs, x, y

    def sample(self, size):
        for idx in BatchSampler(batch_size=size, sampler=RandomSampler(self), drop_last=False):
            x, y, idxs = [], [], []
            for i in idx:
                _i, _x, _y = self[i]
                x.append(_x)
                y.append(_y)
                idxs.append(_i)

            if self.final_transformer is not None:
                x = self.final_transformer(x)

            if self.final_target_transformer is not None:
                y = self.final_target_transformer(y)

            return idxs, x, y


class TasksContainer(ABC):
    def __init__(self, dataset: ClassificationDataset, labels_per_task: int, batch_size: int,
                 shuffle_labels: bool = False,
                 random_state: Union[np.random.RandomState, int] = None):

        self._tasks = list()
        self._current_task = 0
        self.current_batch_size = None

        if random_state is None or isinstance(random_state, int):
            self.RandomState = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.RandomState = random_state

        self.generate_tasks(dataset, labels_per_task=labels_per_task, batch_size=batch_size,
                            shuffle_labels=shuffle_labels)

        for i, t in enumerate(self._tasks):
            setattr(t, 'index', i)

    def __len__(self):
        return len(self._tasks)

    @abstractmethod
    def generate_tasks(self, dataset: ClassificationDataset, labels_per_task: int, batch_size: int,
                       shuffle_labels: bool = False):
        raise NotImplementedError

    def add_task(self, task):
        self._tasks.append(task)

    @property
    def task(self):
        return self._tasks[self._current_task]

    @task.setter
    def task(self, v: int):
        if v > len(self):
            raise ValueError('ERROR (MODIFICARE)')
        self._current_task = v

    def __getitem__(self, i: int):
        if i > len(self):
            raise ValueError('ERROR (MODIFICARE)')
        return self._tasks[i]

    def __iter__(self):
        for t in self._tasks:
            yield t