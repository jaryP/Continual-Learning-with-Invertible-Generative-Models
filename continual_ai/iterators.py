from abc import ABC
from typing import Union
import numpy as np


class Iterator(ABC):
    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class LinearIterator(Iterator):
    def __init__(self, to_iter):
        assert hasattr(to_iter, '__getitem__') and hasattr(to_iter, '__len__')
        self._toiter = to_iter
        self._ln = len(to_iter)
        self._current_iter = None

    @property
    def idx_iter(self):
        return self._current_iter

    # @idx_iter.setter
    # def idx_iter(self, v):
    #     self._current_iter = iter(v)

    def set_iter(self, v):
        self._current_iter = iter(v)

    def __next__(self):
        for i in self.idx_iter:
            return self._toiter[i]
        raise StopIteration

    def __iter__(self):
        self.set_iter(np.arange(self._ln))
        # self.idx_iter = np.arange(self._ln)
        return self


class RandomIterator(LinearIterator):
    def __init__(self, to_iter, random_state: Union[int, np.random.RandomState] = None):
        super().__init__(to_iter)
        assert hasattr(to_iter, '__getitem__') and hasattr(to_iter, '__len__')

        if random_state is None or isinstance(random_state, int):
            self.RandomState = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.RandomState = random_state
        else:
            raise ValueError("random_state can be None, Int or numpy RandomState object, an {} was give"
                             .format(type(random_state)))

    def __iter__(self):
        idx = np.arange(self._ln)
        self.RandomState.shuffle(idx)
        # self.idx_iter = idx
        self.set_iter(idx)
        return self


class LinearBatchIterator(LinearIterator):
    def __init__(self, to_iter, batch_size: int = 1):
        super().__init__(to_iter)
        assert hasattr(to_iter, '__getitem__') and hasattr(to_iter, '__len__')

        self._bs = batch_size

    def __next__(self):
        batch = []

        for i in self.idx_iter:
            batch.append(self._toiter[i])
            if len(batch) == self._bs:
                return batch

        if len(batch) > 0:
            return batch
        else:
            raise StopIteration

    def __len__(self):
        return self._ln // self._bs


class RandomBatchIterator(LinearBatchIterator):
    def __init__(self, to_iter, batch_size: int = 1, random_state: Union[np.random.RandomState, int] = None):
        super().__init__(to_iter, batch_size)

        assert hasattr(to_iter, '__getitem__') and hasattr(to_iter, '__len__')

        if random_state is None or isinstance(random_state, int):
            self.RandomState = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.RandomState = random_state
        else:
            raise ValueError("random_state can be None, Int or numpy RandomState object, an {} was give"
                             .format(type(random_state)))

        self._toiter = to_iter

    def __iter__(self):
        idx = np.arange(self._ln)
        self.RandomState.shuffle(idx)
        # self.idx_iter = idx
        self.set_iter(idx)
        return self


class Sampler:
    def __init__(self, x, dimension: int = 1, replace: bool = False, return_indexes: bool = False,
                 weights: Union[np.ndarray, list, tuple] = None,
                 random_state: Union[np.random.RandomState, int] = None):

        self._x = x
        self.replace = False
        self.dimension = dimension
        self.current_dimension = dimension
        self.replace = replace
        self.return_indexes = return_indexes

        if random_state is None or isinstance(random_state, int):
            self.RandomState = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.RandomState = random_state
        else:
            raise ValueError("random_state can be None, Int or numpy RandomState object, an {} was give"
                             .format(type(random_state)))

        self._w = weights
        if self._w is not None and np.sum(self._w) != 1:
            sm = np.sum(self._w)
            self._w = [w / sm for w in self._w]

    # def __iter__(self):
    #     raise NotImplementedError
    #     # return self
    #
    # def __next__(self):
    #     # idx = np.random.choice(len(self._x))
    #     # b = [self._x[i] for i in idx]
    #     # return b
    #     # raise StopIteration
    #     raise NotImplementedError

    def __call__(self, dimension: int = None):
        idx = self.RandomState.choice(len(self._x), dimension if dimension is not None else self.dimension,
                                      replace=self.replace, p=self._w)

        b = [self._x[i] for i in idx]
        if self.return_indexes:
            return b, idx
        return b
