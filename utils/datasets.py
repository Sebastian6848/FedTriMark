# -*- coding: UTF-8 -*-

import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, MNIST, ImageFolder, CIFAR100
import numpy as np
import os
import sys
from PIL import Image

'''
1. 从给定的完整数据集（dataset）中，根据指定的索引（idxs）提取一个子集。
2. 这个子集可以用于训练或测试，特别是在联邦学习等场景中，每个客户端可能只使用数据集的一部分。
'''
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
'''
1. 下载数据集 和 测试集
2. 对数据集进行数据增强，增加数据的多样性，防止模型过拟合
3. 对测试集进行简单的预处理即可
4. 数据集用于训练模型，测试集用于测试模型图像分类的性能
'''
def get_full_dataset(dataset_name, img_size=(32, 32)):
    if dataset_name == 'mnist':
        train_dataset = MNIST('./data/mnist/', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Resize(img_size),
                                  transforms.Normalize((0.1307,), (0.3081,))  # 标准化：将图像的像素值减去均值 0.1307，然后除以标准差 0.3081，单通道，使模型更快收敛
                              ]))
        test_dataset = MNIST('./data/mnist/', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Resize(img_size),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    elif dataset_name == 'cifar10':
        train_dataset = CIFAR10('./data/cifar10/', train=True, download=True,   # 训练集：使用了数据增强技术（如填充、随机裁剪、随机翻转），以增加数据的多样性，防止模型过拟合。
                                transform=transforms.Compose([
                                    transforms.ToTensor(),  # 将图像从 PIL 格式或 NumPy 数组转换为 PyTorch 张量（Tensor），并将像素值从 [0, 255] 缩放到 [0, 1]
                                    transforms.Pad(4, padding_mode="reflect"),  # 对图像进行填充，每边填充 4 个像素，填充模式为 reflect（反射填充）
                                    transforms.RandomCrop(img_size),  # 随机裁剪图像到指定大小 img_size
                                    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像，增加数据多样性
                                    transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.244, 0.262))  # 对图像进行标准化，减去均值 (0.491, 0.482, 0.447)，除以标准差 (0.247, 0.244, 0.262)
                                ]))
        test_dataset = CIFAR10('./data/cifar10/', train=False, download=True,  # 测试集：仅做了必要的预处理
                               transform=transforms.Compose([
                                   transforms.ToTensor(),  # 将图像转换为 PyTorch 张量，并将像素值缩放到 [0, 1]
                                   transforms.Resize(img_size),  # 将图像调整为指定大小 img_size
                                   transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.244, 0.262))
                               ]))
    elif dataset_name == 'cifar100':
        train_dataset = CIFAR100('./data/cifar100/', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Pad(4, padding_mode="reflect"),
                                    transforms.RandomCrop(img_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.244, 0.262))
                                ]))
        test_dataset = CIFAR100('./data/cifar100/', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Resize(img_size),
                                   transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.244, 0.262))
                               ]))
    else:
        exit("Unknown Dataset")
    return train_dataset, test_dataset

"""
将数据集平均分成 num_clients 份，每份随机分配给一个客户端。
每个客户端的数据分布与完整数据集相同。
"""
def iid_split(dataset, num_clients):
    """
    Split I.I.D client data
    :param dataset:
    :param num_clients:
    :return: dict of image indexes
    """

    dataset_len = len(dataset)
    num_items = dataset_len // num_clients
    dict_clients = dict()
    all_idxs = [i for i in range(dataset_len)]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
    return dict_clients

"""
每个类别的样本按比例分配给客户端，比例由狄利克雷分布决定。
某些客户端可能更集中于某些类别，导致数据分布不均匀。
"""
def dniid_split(dataset, num_clients, param=0.8):
    """
    Using Dirichlet distribution to sample non I.I.D client data
    :param dataset:
    :param num_clients:
    :param param: parameter used in Dirichlet distribution
    :return: dict of image indexes
    """
    dataset_len = len(dataset)
    dataset_y = np.array(dataset.targets)
    labels = set(dataset_y)
    sorted_idxs = dict()
    for label in labels:
        sorted_idxs[label] = []

    # sort indexes by labels
    for i in range(dataset_len):
        sorted_idxs[dataset_y[i]].append(i)

    for label in labels:
        sorted_idxs[label] = np.array(sorted_idxs[label])

    # initialize the clients' dataset dict
    dict_clients = dict()
    for i in range(num_clients):
        dict_clients[i] = None
    # split the dataset separately
    for label in labels:
        idxs = sorted_idxs[label]
        sample_split = np.random.dirichlet(np.array(num_clients * [param]))
        accum = 0.0
        num_of_current_class = idxs.shape[0]
        for i in range(num_clients):
            client_idxs = idxs[int(accum * num_of_current_class):
                               min(dataset_len, int((accum + sample_split[i]) * num_of_current_class))]
            if dict_clients[i] is None:
                dict_clients[i] = client_idxs
            else:
                dict_clients[i] = np.concatenate((dict_clients[i], client_idxs))
            accum += sample_split[i]
    return dict_clients

"""
将数据集按类别排序后，切成多个小块（分片），每个客户端随机分配到若干小块。
某些客户端可能只包含少数类别的样本，导致数据分布极端不均匀。
"""
def pniid_split(dataset, num_clients, num_of_shards_each_clients=2):
    """
    Simulate pathological non I.I.D distribution
    :param dataset:
    :param num_clients:
    :param num_of_shards_each_clients:
    :return:
    """
    dataset_len = len(dataset)
    dataset_y = np.array(dataset.targets)

    sorted_idxs = np.argsort(dataset_y)

    size_of_each_shards = dataset_len // (num_clients * num_of_shards_each_clients)
    per = np.random.permutation(num_clients * num_of_shards_each_clients)
    dict_clients = dict()
    for i in range(num_clients):
        idxs = np.array([])
        for j in range(num_of_shards_each_clients):
            idxs = np.concatenate((idxs, sorted_idxs[per[num_of_shards_each_clients * i + j] * size_of_each_shards:
                                   min(dataset_len, (per[num_of_shards_each_clients * i + j] + 1) * size_of_each_shards)]))
        dict_clients[i] = idxs
    return dict_clients
