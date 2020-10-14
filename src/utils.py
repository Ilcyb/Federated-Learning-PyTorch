#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
import torchvision.utils as vutils
import math
import numpy as np


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
        else:
            data_dir = './data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups

# 根据label将数据集整体划分为一个dict[key=label,value=dataset]
def get_dataset_split_by_label(args):
    if args.dataset == 'cifar':
        pass

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
        else:
            data_dir = './data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        digit_indexs = {}
        for digit in range(10):
            idx = train_dataset.train_labels == digit
            digit_indexs[digit] = [i for i in range(len(idx)) if idx[i]==True]
        
        return train_dataset, test_dataset, digit_indexs

# 将给定的多个数据集合成一个数据集
def merge_multidatasets(*datasets):
    total_dataset = datasets[0]
    total_data = torch.cat([dataset.train_data for dataset in datasets] ,0)
    total_label = torch.cat([dataset.train_labels for dataset in datasets], 0)
    total_dataset.train_labels = total_label
    total_dataset.train_data = total_data
    return total_dataset

# 根据传入的label分组返回对应的datasets分组
def get_dataset_ganattack(args, label_split):
    train_dataset, test_dataset, label_indexs = get_dataset_split_by_label(args)
    local_data_fraction = len(label_indexs)/args.num_users # 为什么本地数据集对总数据集的占比是（分类数量/总人数）而不是（1/总人数）？因为数据集大小已经被分类数量给稀释了一次了
    # idx_groups [[labels1的数据的idxs],[labels2的数据的idxs],...]
    idx_groups = []
    if local_data_fraction>1:
        local_data_fraction = 1
    for labels in label_split:
        indexs = []
        for label in labels:
            indexs +=  list(np.random.choice(label_indexs[label], (int)(len(label_indexs[label])*local_data_fraction), replace=False))
        idx_groups.append(indexs)
    return train_dataset, test_dataset, idx_groups

# 根据传入的label分组返回随机选取的数据
def get_dataset_idxgroup_ganattack(args, label_split, label_indexs):
    local_data_fraction = len(label_indexs)/args.num_users # 为什么本地数据集对总数据集的占比是（分类数量/总人数）而不是（1/总人数）？因为数据集大小已经被分类数量给稀释了一次了
    # idx_groups [[labels1的数据的idxs],[labels2的数据的idxs],...]
    idx_groups = []
    if local_data_fraction>1:
        local_data_fraction = 1
    for labels in label_split:
        indexs = []
        for label in labels:
            indexs +=  list(np.random.choice(label_indexs[label], (int)(len(label_indexs[label])*local_data_fraction), replace=False))
        idx_groups.append(indexs)
    return idx_groups

# # C(n, m)
# def cob(n, m):
#     return math.factorial(n)//(math.factorial(n-m)*math.factorial(m)) 

# # 将给定的组件组合成required_nums个不同的组合
# def combination(components, required_nums):
#     max_combination_nums = 0
#     components_len = len(components)
#     for i in range(1, components_len+1):
        
    

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
