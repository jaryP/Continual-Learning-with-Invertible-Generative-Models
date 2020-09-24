__all__ = ['split_dataset', 'ExperimentConfig', 'DatasetIterator']

from collections import defaultdict
import yaml
from typing import Union, Tuple
import numpy as np


def _get_balanced_index(y: np.ndarray, test_split: float, dev_split: float = 0,
                        random_state: np.random.RandomState = None) -> Tuple[list, list, list]:
    train = []
    test = []
    dev = []

    d = defaultdict(list)

    for i, _y in enumerate(y):
        d[_y].append(i)

    for index_list in d.values():

        if random_state is not None:
            random_state.shuffle(index_list)
        else:
            np.random.shuffle(index_list)

        ln = len(index_list)

        _test_split = int(ln * test_split)
        _dev_split = int(ln * dev_split)

        # print(test_split, dev_split)

        train.extend(index_list[_test_split + _dev_split:])
        test.extend(index_list[:_test_split])
        dev.extend(index_list[_test_split:_test_split + _dev_split])

    assert sum(map(len, [train, test, dev])) == len(y)
    # print(len(train), len(test), len(dev))

    return train, dev, test


def _get_split_index(y: np.ndarray, test_split: float, dev_split: float = 0,
                     random_state: np.random.RandomState = None) -> Tuple[list, list, list]:
    index_list = np.arange(len(y))

    if random_state is not None:
        random_state.shuffle(index_list)
    else:
        np.random.shuffle(index_list)

    ln = len(index_list)

    _test_split = int(ln * test_split)
    _dev_split = int(ln * dev_split)

    train = index_list[_test_split + _dev_split:]
    test = index_list[:_test_split]
    dev = index_list[_test_split:_test_split + _dev_split]

    return train, dev, test


def split_dataset(x: np.ndarray, y: np.ndarray, test_split: float, dev_split: float = 0,
                  balance_labels: bool = True, old_y: np.ndarray = None,
                  random_state: np.random.RandomState = None) -> Tuple[tuple, tuple, tuple]:

    if balance_labels:
        train, dev, test = _get_balanced_index(y, test_split, dev_split, random_state)
    else:
        train, dev, test = _get_split_index(y, test_split, dev_split, random_state)

    train = (x[train], y[train], old_y[train] if old_y is not None else None)
    test = (x[test], y[test], old_y[test] if old_y is not None else None)
    dev = (x[dev], y[dev], old_y[dev] if old_y is not None else None)

    return train, dev, test


class ExperimentConfig:
    train_config = dict(experiments=1, cuda=True, batch_size=64, epochs=10, load=True, save=True,
                        save_path="results", warm_up_epochs=0)

    cl_technique_config = dict(name='naive')
    cl_config = dict(cl_problem='sit', label_per_task=2, shuffle_labels=False, dataset='mnist')

    def __init__(self, path):
        experiment_config = yaml.load(open(path), Loader=yaml.FullLoader)

        for d in experiment_config.keys():
            d = d.lower()
            if hasattr(self, d):
                for k, v in experiment_config[d].items():
                    k = k.lower()
                    if isinstance(v, str):
                        v = v.lower()
                    getattr(self, d)[k] = v

    def __str__(self):
        return '#' * 100 + '\n' + \
               F'Train config: {self.train_config}\n' + \
               F'Cl config: {self.cl_config}\n' + \
               F'Cl Technique config: {self.cl_technique_config}\n' + '#' * 100

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            fields = ['train_config', 'cl_technique_config', 'cl_config']
            for field in fields:
                if not getattr(self, field) == getattr(other, field):
                    return False
            return True
        else:
            raise TypeError('Comparing object is not of the same type.')


class DatasetIterator:
    def __init__(self, dataset, batch_size: int = 1,
                 random_state: Union[np.random.RandomState, int] = None,
                 preprocessing=None):

        self.batch_size = batch_size

        self.dataset = dataset

        self.current_batch_size = None
        self.current_shuffle = True
        self.current_preprocessing = None

        if hasattr(preprocessing, '__call__'):
            self.preprocessing = preprocessing
        else:
            # TODO: inserire warning
            self.preprocessing = None

        if random_state is None or isinstance(random_state, int):
            self.RandomState = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.RandomState = random_state
        else:
            raise ValueError("random_state can be None, Int or numpy RandomState object, an {} was give"
                             .format(type(random_state)))

    def __call__(self, batch_size: int = None, shuffle: bool = None, preprocessing=None, **kwargs):
        self.current_batch_size = batch_size
        self.current_shuffle = shuffle
        self.current_preprocessing = preprocessing
        return self

    def __iter__(self):
        bs = self.batch_size if self.current_batch_size is None else self.current_batch_size
        batch_pre = self.current_preprocessing \
            if (self.current_preprocessing is not None and hasattr(self.current_preprocessing, '__call__')) \
            else self.preprocessing

        idx = np.arange(len(self.dataset))

        self.RandomState.shuffle(idx)

        self.current_shuffle = None
        self.current_preprocessing = None
        self.current_batch_size = None

        start = 0

        while True:

            b = [self.dataset[i] for i in idx[start: start + bs]]

            # for i in idx[start: start + bs]:
            # _x, _y = self.dataset[i]
            # b.append(self.dataset[i])
            # x.append(_x)
            # y.append(_y)

            if len(b) == 0:
                return

            start = start + bs

            if batch_pre is not None:
                b = zip(b)

                xb = self.preprocessing(b)
            else:
                xb = list(zip(b))

            yield xb

    def sample(self, size):
        b = next(iter(self.__call__(batch_size=size)))
        return b
