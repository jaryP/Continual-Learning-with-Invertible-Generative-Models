import codecs
import gzip
import pickle
import tarfile
from os.path import join
from typing import Callable
from urllib.request import urlretrieve
import numpy as np
import torch
from torch.utils.data import DataLoader
# from classification.base1 import AbstractBaseDataset
# from base import AbstractBaseDataset
import scipy.io as sio

# TODO: implementare train/test di default per ogni dataset
# TODO: implementare split random per ogni dataset
from continual_ai.base import AbstractBaseDataset


class MNIST(AbstractBaseDataset):
    url = {'train': {'images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                     'labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'},
           'test': {'images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                    'labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'}}

    def __init__(self, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, target_transformer: Callable = None, ):

        self._get_int = lambda x: int(codecs.encode(x, 'hex'), 16)

        self.file_names = [url.rpartition('/')[2] for url in self.url]

        super().__init__(name='MNIST', download_if_missing=download_if_missing, data_folder=data_folder,
                         transformer=transformer, target_transformer=target_transformer)

    def _load_image(self, data):
        length = self._get_int(data[4:8])
        num_rows = self._get_int(data[8:12])
        num_cols = self._get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        parsed = parsed.reshape((length, num_rows, num_cols))

        return parsed

    def _load_label(self, data):
        length = self._get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        parsed = parsed.reshape((length, -1))
        parsed = np.squeeze(parsed)
        return parsed

    def load_dataset(self):

        x_train, y_train = None, None
        x_dev, y_dev = None, None
        x_test, y_test = None, None

        for i in 'train', 'test':
            for j in 'images', 'labels':
                v = self.url[i]

                with gzip.GzipFile(join(self.data_folder, v[j].rpartition('/')[2])) as zip_f:
                    data = zip_f.read()

                    if j == 'images':
                        if i == 'train':
                            x_train = self._load_image(data)[:, None]
                        else:
                            x_test = self._load_image(data)[:, None]

                    else:
                        if i == 'train':
                            y_train = self._load_label(data)
                        else:
                            y_test = self._load_label(data)

        return (x_train, y_train), (x_dev, y_dev), (x_test, y_test)

    def download_dataset(self):
        for _, type in self.url.items():
            for _, url in type.items():
                urlretrieve(url, join(self.data_folder, url.rpartition('/')[2]))


class CIFAR10(AbstractBaseDataset):

    files = {'train' : ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'],
             'test': ['test_batch']}

    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    def __init__(self, download_if_missing=True, data_folder=None, fine_labels=False,
                 transformer: Callable = None, target_transformer: Callable = None):

        # self.url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

        # self.file_name = self.url.rpartition('/')[2]
        self.fine_labels = fine_labels

        # AbstractDataset.__init__(self, name='CIFAR10', download_if_missing=download_if_missing, data_folder=data_folder)
        super(CIFAR10, self).__init__(name='CIFAR10', download_if_missing=download_if_missing, data_folder=data_folder,
                                      transformer=transformer, target_transformer=target_transformer)

    # def __len__(self):
    #     return len(self._x)
    #
    # def __getitem__(self, item):
    #     return self._x[item], self._y[item]

    def load_dataset(self):

        x_train, y_train = [], []
        x_dev, y_dev = None, None
        x_test, y_test = [], []

        with tarfile.open(join(self.data_folder, self.url.rpartition('/')[2]), 'r') as tar:
            for item in tar:
                name = item.name.rpartition('/')[-1]
                if 'batch' in name:
                    contentfobj = tar.extractfile(item)
                    x, y = None, None
                    if contentfobj is not None:
                        entry = pickle.load(contentfobj, encoding='latin1')
                        if 'data' in entry:
                            x = entry['data']
                            if self.fine_labels:
                                y = entry['fine_labels']
                            else:
                                y = entry['labels']

                    if name in self.files['train']:
                        x_train.append(x)
                        y_train.append(y)
                    elif name in self.files['test']:
                        x_test.append(x)
                        y_test.append(y)

        x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32)
        x_test = np.concatenate(x_test).reshape(-1, 3, 32, 32)

        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)

        return (x_train, y_train), (x_dev, y_dev), (x_test, y_test)

    def download_dataset(self):
        urlretrieve(self.url, join(self.data_folder, self.url.rpartition('/')[2]))


# class CIFAR100(AbstractDataset):
#     def __init__(self, download_if_missing=True, data_folder=None, fine_labels=True):
#
#         self.url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
#
#         self.file_name = self.url.rpartition('/')[2]
#         self.fine_labels = fine_labels
#
#         AbstractDataset.__init__(self, name='CIFAR100', download_if_missing=download_if_missing,
#                                  data_folder=data_folder)
#
#     # def __len__(self):
#     #     return len(self._x)
#     #
#     # def __getitem__(self, item):
#     #     return self._x[item], self._y[item]
#
#     def load_dataset(self):
#
#         x, y = [], []
#
#         with tarfile.open(join(self.data_folder, self.file_name), 'r') as tar:
#             for item in tar:
#                 name = item.name.rpartition('/')[-1]
#                 if 'train' in name or 'test' in name:
#                     contentfobj = tar.extractfile(item)
#                     if contentfobj is not None:
#                         entry = pickle.load(contentfobj, encoding='latin1')
#                         if 'data' in entry:
#                             x.append(entry['data'])
#                             if self.fine_labels:
#                                 y.append((entry['fine_labels']))
#                             else:
#                                 y.append(entry['coarse_labels'])
#
#         self._y = np.concatenate(y)
#         self._x = np.concatenate(x).reshape(-1, 3, 32, 32)
#
#     def download_dataset(self):
#         urlretrieve(self.url, join(self.data_folder, self.file_name))
#
#
# class KMNIST(MNIST):
#
#     def __init__(self, download_if_missing: bool = True, data_folder: str = None):
#         self._get_int = lambda x: int(codecs.encode(x, 'hex'), 16)
#
#         self.url = \
#             [
#                 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
#                 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
#                 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
#                 'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz',
#             ]
#
#         self.file_names = [url.rpartition('/')[2] for url in self.url]
#
#         AbstractDataset.__init__(self, name='KMNIST',
#                                  download_if_missing=download_if_missing, data_folder=data_folder)
#
#
# class K49MNIST(MNIST):
#
#     def __init__(self, download_if_missing: bool = True, data_folder: str = None):
#
#         self._get_int = lambda x: int(codecs.encode(x, 'hex'), 16)
#
#         self.url = \
#             [
#                 'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
#                 'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
#                 'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
#                 'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz',
#             ]
#
#         self.file_names = [url.rpartition('/')[2] for url in self.url]
#
#         AbstractDataset.__init__(self, name='K49MNIST',
#                                  download_if_missing=download_if_missing, data_folder=data_folder)
#
#     def load_dataset(self):
#
#         X = []
#         Y = []
#
#         for url, filename in zip(self.url, self.file_names):
#             file_path = join(self.data_folder, filename)
#
#             data = np.load(file_path)
#             data = data[data.files[0]]
#
#             if 'imgs' in filename:
#                 X.extend(data)
#             else:
#                 Y.extend(data)
#
#         self._y = np.asarray(Y)
#         self._x = np.stack(X)[:, None]
#

class SVHN(AbstractBaseDataset):
    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"], }

    # url = {'train': {'images': "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
    #                  'labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'},
    #        'test': {'images': "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
    #                 'labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'}}

    # 'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
    #           "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, download_if_missing: bool = True, data_folder: str = None,
                 transformer: Callable = None, target_transformer: Callable = None, ):

        super().__init__(name='SVHN', download_if_missing=download_if_missing, data_folder=data_folder,
                         transformer=transformer, target_transformer=target_transformer)

    def load_dataset(self):

        x_train, y_train = None, None
        x_dev, y_dev = None, None
        x_test, y_test = None, None

        for t, (url, name, _) in self.split_list.items():
            file_path = join(self.data_folder, name)
            loaded_mat = sio.loadmat(file_path)

            x = loaded_mat['X']
            y = loaded_mat['y'].astype(np.int64).squeeze()

            np.place(y, y == 10, 0)
            x = np.transpose(x, (3, 2, 0, 1))

            if t == 'train':
                x_train, y_train = x, y
            else:
                x_test, y_test = x, y

        return (x_train, y_train), (x_dev, y_dev), (x_test, y_test)

    def download_dataset(self):
        for t, (url, name, _) in self.split_list.items():
            urlretrieve(url, join(self.data_folder, name))

# if __name__ == '__main__':
#
#     transformer = lambda x: x * -1
#
#     d = MNIST(transformer=transformer)
#     d.test()
#     print(len(d))
#
#     d.split_dataset(test_split=0.5)
#
#     d.train()
#     print(len(d))
#
#     d.test()
#     print(len(d))
#
#     # di = DatasetIterator(d, batch_size=32, preprocessing=lambda x, y: (torch.tensor(x), torch.tensor(y)))
#     di = DataLoader(d, batch_size=12, shuffle=True)
#
#     for i, x, y in di:
#         print(i)
