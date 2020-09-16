from abc import ABC, abstractmethod
from os import makedirs
from os.path import join, dirname, exists
from typing import Callable, Union, Tuple
import numpy as np

from continual_ai.utils import split_dataset


class ClassificationDataset(object):
    def __init__(self, x_train, y_train, x_dev, y_dev, x_test, y_test, transformer: Callable = None,
                 target_transformer: Callable = None):

        super(ClassificationDataset, self).__init__()
        self._x_train, self._y_train, self._x_dev, self._y_dev, self._x_test, self._y_test = \
            x_train, y_train, x_dev, y_dev, x_test, y_test

        self.transformer = transformer  # if transformer is not None else lambda x: x
        self.target_transformer = target_transformer  # if target_transformer is not None else lambda x: x

        self._split = 'train'

        self.sample_dimension = self._x_train[0]
        # self.classes = len(set(self._y_train))

        labels = set(y_train)
        labels = list(labels)
        labels.sort()

        self.labels = np.asarray(labels, dtype=int)

    def __setattr__(self, attr, val):
        if attr == 'split':
            raise ValueError('{} can be set using the functions train(), test() and dev()'.
                             format(attr, self.__class__.__name__))

        super(ClassificationDataset, self).__setattr__(attr, val)

    def train(self):
        self._split = 'train'

    def test(self):
        self._split = 'test'

    def dev(self):
        self._split = 'dev'

    def _x(self):
        return getattr(self, F'_x_{self._split}')

    def _y(self):
        return getattr(self, F'_y_{self._split}')

    @property
    def current_split(self):
        return self._split

    @property
    def x(self):
        x = self._x()
        if x is None:
            return None
        if self.transformer is not None:
            x = self.transformer(x)
        return x

    @property
    def y(self):
        y = self._y()
        if y is None:
            return None
        if self.target_transformer is not None:
            y = self.target_transformer(y)
        return y

    def __getitem__(self, item):
        return item, self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)

    @property
    def data(self):
        return self._x(), self._y()

    def split_dataset(self, test_split: float = 0.2, dev_split: float = 0.0, balance_labels: bool = False,
                      random_state: Union[np.random.RandomState, int] = None):

        x = []
        y = []

        for i in ['trai n', 'test', 'dev']:
            if not getattr(self, '_x_{}'.format(i)) is None:
                x.append(getattr(self, '_x_{}'.format(i)))
                y.append(getattr(self, '_y_{}'.format(i)))

        x = np.concatenate(x, 0)
        y = np.concatenate(y, 0)

        (self._x_train, self._y_train, _), (self._x_dev, self._y_dev, _), (self._x_test, self._y_test, _) = \
            split_dataset(x, y, test_split=test_split, dev_split=dev_split, random_state=random_state,
                          balance_labels=balance_labels)

    # def pre_process(self, f=None):
    #     for i in 'train', ''
    #     if f is not None and hasattr(f, '__call__'):
    #         self._x, self._y = f(self._x, self._y)


class AbstractBaseDataset(ClassificationDataset, ABC):
    def __init__(self, name: str = 'dataset', transformer: Callable =None, target_transformer: Callable = None,
                 download_if_missing: bool = True, data_folder: str = None):

        if data_folder is None:
            data_folder = join(dirname(__file__), 'datasets', name)

        self.data_folder = data_folder
        self._split = 'train'
        self.name = name

        if not exists(self.data_folder):
            if not download_if_missing:
                raise IOError("Data not found and `download_if_missing` is False")
            else:
                if not exists(self.data_folder):
                    makedirs(self.data_folder)

                self.download_dataset()

        (x_train, y_train), (x_dev, y_dev), (x_test, y_test) = self.load_dataset()

        super().__init__(x_train, y_train, x_dev, y_dev, x_test, y_test, transformer=transformer,
                         target_transformer=target_transformer)

        # super(ClassificationDataset).__init__(_x_train, _y_train, _x_dev, _y_dev, _x_test, _y_test)
        # super(ClassificationDataset, self).__init__(x_train, y_train, x_dev, y_dev, x_test, y_test)

    @abstractmethod
    def load_dataset(self) -> Tuple[tuple, tuple, tuple]:
        raise NotImplementedError

    @abstractmethod
    def download_dataset(self):
        raise NotImplementedError