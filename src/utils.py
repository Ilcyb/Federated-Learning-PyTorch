#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
from numpy.lib.type_check import imag
import torch
from torch.utils import data
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import attface_iid
from sampling import svhn_iid
import torchvision.utils as vutils
import math
import numpy as np
import pathlib
import os
import imageio
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

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
            # Sample IID user data from cifar
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from cifar
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset in ('mnist', 'fmnist'):
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

    elif args.dataset == 'attface':
        data_dir = './data/attface/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # TODO
    elif args.dataset == 'svhn':
        data_dir = './data/svhn'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        train_dataset = datasets.SVHN(data_dir, split='train', download=True,
                                       transform=apply_transform)

        test_dataset = datasets.SVHN(data_dir, split='test', download=True,
                                      transform=apply_transform)
        if args.iid:
            # Sample IID user data from svhn
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            raise NotImplementedError()

    return train_dataset, test_dataset, user_groups

# 根据label将数据集整体划分为一个dict[key=label,value=dataset]
def get_dataset_split_by_label(args):

    train_dataset = None
    test_dataset = None

    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
        label_indexs = {}
        for i in range(10):
            label_indexs[i] = []
        for i in range(len(train_dataset.targets)):
            label_indexs[train_dataset.targets[i]].append(i)
        return train_dataset, test_dataset, label_indexs

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

        label_indexs = {}
        for label in range(10):
            idx = train_dataset.targets == label
            label_indexs[label] = [i for i in range(len(idx)) if idx[i]==True]
        
    return train_dataset, test_dataset, label_indexs

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

def get_experiment_result_location(model:str, dataset:str, label:int, hyperparams:dict, mode:str, experiment_name=None):
    name_prefix = ''
    for name, value in hyperparams.items():
        if name_prefix!='':
            name_prefix+='_'
        name_prefix+='{}_{}'.format(name, value)
    if experiment_name != None:
        name_prefix+=experiment_name
    current_folder_path = pathlib.Path().absolute()
    if mode == 'debug':
        prefix = os.path.join(current_folder_path, os.path.join(
            'debug_save', os.path.join(model, os.path.join(dataset, os.path.join(str(label), name_prefix)))))
    elif mode == 'production':
        prefix = os.path.join(current_folder_path, os.path.join(
            'save', os.path.join(model, os.path.join(dataset, os.path.join(str(label), name_prefix)))))
    i=1
    while True:
        path = os.path.join(prefix, 'training_'+str(i))
        if os.path.isdir(path):
            i+=1
            continue
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'fake_images'))
        return path

def save_grid(images, path):
    size = math.ceil((int)(math.sqrt(len(images))))
    grid = vutils.make_grid(images, size)
    vutils.save_image(grid, os.path.join(path, 'fake_images_grid.png'))

def generate_gif_from_file(image_folder, gif_path, duration=0.1):
    image_paths = []
    files = os.listdir(image_folder)
    for file in files:
        image_path = os.path.join(image_folder, file)
        image_paths.append(image_path)

    frames = []
    for image_path in image_paths:
        frames.append(imageio.imread(image_path))
    imageio.mimsave(gif_path, frames, 'GIF', duration=duration)

def generate_gif_from_list(image_list, gif_path, duration=0.1):
    frames = []
    for image in image_list:
        frames.append(image)
    imageio.mimsave(gif_path, frames, 'GIF', duration=duration)

def plot_loss_acc(loss, acc, path):
    x = [i+1 for i in range(len(loss))]
    plt.plot(x, loss, 'r--', label='loss')
    plt.title('Loss')
    plt.xlabel('iter')
    plt.ylabel('loss value')
    plt.xticks(range(len(x)))
    x_major_locator=MultipleLocator((int)(len(loss)/8))
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()
    plt.savefig(os.path.join(path, 'loss.png'))

    plt.clf()

    plt.plot(x, [a*100 for a in acc], 'g--', label='acc')
    plt.title('Accuracy')
    plt.xlabel('iter')
    plt.ylabel('accuracy rate(%)')
    plt.xticks(range(len(x)))
    x_major_locator=MultipleLocator((int)(len(acc)/8))
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()
    plt.savefig(os.path.join(path, 'acc.png'))
    
    plt.clf()
    
def compute_avgpsnr(generated_image, batch_images, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    psnrs = np.zeros(len(batch_images))
    for i in range(len(batch_images)):
        mse = np.mean((np.array(generated_image, dtype=np.float32) - np.array(batch_images[i], dtype=np.float32)) ** 2)
        if mse == 0:
            psnrs[i] = 100
        psnrs[i] = 20 * np.log10(max_value / (np.sqrt(mse)))
    return psnrs.mean()

def plot_avg_psnr(avg_psnrs, path):
    x = [i+1 for i in range(len(avg_psnrs))]
    plt.plot(x, avg_psnrs, 'r', label='AVG PSNR')
    plt.title('AVG PSNR')
    plt.xlabel('iter')
    plt.ylabel('psnr value')
    plt.xticks(range(len(x)))
    x_major_locator=MultipleLocator((int)(len(avg_psnrs)/8))
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()
    plt.savefig(os.path.join(path, 'avg_psnr.png'))