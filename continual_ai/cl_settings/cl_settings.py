from typing import Union
import numpy as np
from continual_ai.base import AbstractBaseDataset
from .base import TasksContainer, ClassificationTask


class MultiTask(TasksContainer):
    def __init__(self, dataset: AbstractBaseDataset, labels_per_task: int, batch_size: int,
                 shuffle_labels: bool = False,
                 random_state: Union[np.random.RandomState, int] = None):

        super(MultiTask, self).__init__(dataset=dataset, labels_per_task=labels_per_task, shuffle_labels=shuffle_labels,
                                        random_state=random_state, batch_size=batch_size)

    def generate_tasks(self, dataset, labels_per_task: int, batch_size: int, shuffle_labels: bool = True):

        labels = dataset.labels

        if shuffle_labels:
            self.RandomState.shuffle(dataset.labels)

        labels_sets = [list(labels[i:i + labels_per_task]) for i in range(0, len(labels), labels_per_task)]

        if len(labels_sets[-1]) == 1:
            labels_sets[-2].append(labels_sets[-1][0])
            labels_sets = labels_sets[:-1]

        labels_sets = np.asarray(labels_sets)

        labels_map = np.zeros(len(labels), dtype=int)

        for i in labels_sets:
            for j in range(len(i)):
                labels_map[i[j]] = j

        for i, ls in enumerate(labels_sets):
            self._tasks.append(ClassificationTask(batch_size=batch_size, base_dataset=dataset,
                                                  labels_map=labels_map, task_labels=ls))


class SingleIncrementalTask(TasksContainer):
    def __init__(self, dataset: AbstractBaseDataset, labels_per_task: int, batch_size: int,
                 shuffle_labels: bool = False, random_state: Union[np.random.RandomState, int] = None):

        super(SingleIncrementalTask, self).__init__(dataset=dataset, labels_per_task=labels_per_task,
                                                    shuffle_labels=shuffle_labels,
                                                    random_state=random_state, batch_size=batch_size)

    def generate_tasks(self, dataset, labels_per_task: int, batch_size: int, shuffle_labels: bool = False):

        labels = dataset.labels

        if shuffle_labels:
            self.RandomState.shuffle(dataset.labels)

        labels_sets = [list(labels[i:i + labels_per_task]) for i in range(0, len(labels), labels_per_task)]

        if len(labels_sets[-1]) == 1:
            labels_sets[-2].append(labels_sets[-1][0])
            labels_sets = labels_sets[:-1]

        labels_sets = np.asarray(labels_sets)

        labels_map = np.zeros(len(labels), dtype=int)

        offset = 0
        for i in labels_sets:
            for j in range(len(i)):
                labels_map[i[j]] = j + offset
            offset += len(i)

        for i, ls in enumerate(labels_sets):
            self._tasks.append(ClassificationTask(batch_size=batch_size, base_dataset=dataset,
                                                  labels_map=labels_map, task_labels=ls))
