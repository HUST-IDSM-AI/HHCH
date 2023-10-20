import pickle
import time

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import torchvision.datasets as dsets
from PIL import Image

from dataLoader.pre_process import ResizeImage, PlaceCrop


class MyTestDataset(Dataset):
    def __init__(self, data, labels, transform, dataset='cifar10'):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index):
        if self.dataset == 'cifar10':
            time_start = time.time()
            pilImg = Image.fromarray(self.data[index])

            image = self.transform(pilImg)
            time_end = time.time()
            print("time resize {}".format(time_end - time_start))
            return (image, self.labels[index])
        else:
            return (self.transform(self.data[index]), self.labels[index])

    def __len__(self):
        return len(self.data)


class MyTrainDataset(Dataset):
    def __init__(self, data, labels, transform_train, transform_test):
        self.data = data
        self.labels = labels
        self.transform_train = transform_train
        self.transform_test = transform_test

    def __getitem__(self, index):
        pilImg = Image.fromarray(self.data[index])
        imgi = self.transform_train(pilImg)
        imgj = self.transform_train(pilImg)
        val_img = self.transform_test(pilImg)
        return ((imgi, imgj, val_img), self.labels[index], index)

    def __len__(self):
        return len(self.data)


def get_cifar(root):
    # Dataset
    train_dataset = dsets.CIFAR10(root=root,
                                  train=True,
                                  download=False)

    test_dataset = dsets.CIFAR10(root=root,
                                 train=False
                                 )

    database_dataset = dsets.CIFAR10(root=root,
                                     train=True
                                     )

    # train with 5000 images
    X = train_dataset.data
    L = np.array(train_dataset.targets)

    first = True
    for label in range(10):
        index = np.where(L == label)[0]
        N = index.shape[0]
        prem = np.random.permutation(N)
        index = index[prem]

        data = X[index[0:500]]
        labels = L[index[0: 500]]
        if first:
            Y_train = labels
            X_train = data
        else:
            Y_train = np.concatenate((Y_train, labels))
            X_train = np.concatenate((X_train, data))
        first = False

    Y_train = np.eye(10)[Y_train]

    idxs = list(range(len(test_dataset.data)))
    np.random.shuffle(idxs)
    test_data = np.array(test_dataset.data)
    test_tragets = np.array(test_dataset.targets)

    X_val = test_data[idxs[:5000]]
    Y_val = np.eye(10)[test_tragets[idxs[:5000]]]

    X_test = test_data[idxs[5000:]]
    Y_test = np.eye(10)[test_tragets[idxs[5000:]]]

    X_database = database_dataset.data
    Y_database = np.eye(10)[database_dataset.targets]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_database, Y_database


def get_cifar_2(root):
    # Dataset
    train_dataset = dsets.CIFAR10(root=root,
                                  train=True,
                                  download=False)

    test_dataset = dsets.CIFAR10(root=root,
                                 train=False
                                 )

    database_dataset = dsets.CIFAR10(root=root,
                                     train=True
                                     )

    # train with 5000 images
    X = train_dataset.data
    L = np.array(train_dataset.targets)

    first = True
    for label in range(10):
        index = np.where(L == label)[0]
        N = index.shape[0]
        prem = np.random.permutation(N)
        index = index[prem]

        data = X[index[0:1000]]
        labels = L[index[0: 1000]]
        if first:
            Y_train = labels
            X_train = data
        else:
            Y_train = np.concatenate((Y_train, labels))
            X_train = np.concatenate((X_train, data))
        first = False

    Y_train = np.eye(10)[Y_train]

    idxs = list(range(len(test_dataset.data)))
    np.random.shuffle(idxs)
    test_data = np.array(test_dataset.data)
    test_tragets = np.array(test_dataset.targets)

    X_val = test_data[idxs[:1000]]
    Y_val = np.eye(10)[test_tragets[idxs[:1000]]]

    X_test = test_data[idxs[1000:]]
    Y_test = np.eye(10)[test_tragets[idxs[1000:]]]

    X_database = database_dataset.data
    Y_database = np.eye(10)[database_dataset.targets]

    X_database = np.concatenate((X_database, X_test))
    Y_database = np.concatenate((Y_database, Y_test))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_database, Y_database



