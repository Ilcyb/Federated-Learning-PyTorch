#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
from numpy.core.fromnumeric import shape
from numpy.lib.type_check import imag
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class AdversaryDataset(Dataset):
    def __init__(self, datas, targets) -> None:
        self.data = datas
        self.targets = targets

    def __getitem__(self, index: int):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self) -> int:
        return len(self.data)

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # idx 是分给这个用户的data的索引列表 list类型
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()

                # if self.args.verbose and (batch_idx % 10 == 0):
                    # print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     global_round, iter, batch_idx * len(images),
                    #     len(self.trainloader.dataset),
                    #     100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


class AdversaryUpdate(LocalUpdate):
    def __init__(self, args, dataset, idxs, logger, 
    adversary_gan_update=None, discriminator_model=None):
        self.args = args
        self.logger = logger
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        self.adversary_gan_update = adversary_gan_update
        self.discriminator_model = discriminator_model
        # self.train_dataset = dataset
        self.idxs = idxs
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_generator(self):
        self.adversary_gan_update.update_discriminator(self.discriminator_model)
        self.adversary_gan_update.trainD()

    def update_weights(self, model, global_round):
        # 服务器不针对常规数据集进行训练
        # model_state_dict, epoch_loss = super(AdversaryUpdate, self).update_weights(model, global_round)
        # model.load_state_dict(model_state_dict)
        # 服务器攻击需要
        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        model.train()

        for _ in range(self.args.local_ep):
            for _, (images, labels) in enumerate(self.adversary_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()

                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()

        return model.state_dict()

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def weight_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class AdversaryGanUpdate():
    def __init__(self, discriminator_model, generator_model, args, 
    logger, want_label_index, false_label_index=10) -> None:
        self.args = args
        self.logger = logger
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        self.modelD = discriminator_model.to(self.device)
        self.modelG = generator_model.to(self.device)
        self.modelG.apply(weight_init)
        self.want_label_index = want_label_index
        self.false_label_index = false_label_index
        self.batch_size = args.local_gan_bs
        # size of generate input
        self.nz = 100
        self.lr = args.local_gan_lr
        self.beta1 = 0.5

    def update_discriminator(self, discriminator_model):
        self.modelD = discriminator_model.to(self.device)

    def trainD(self):
        # if self.args.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(self.modelG.parameters(), lr=self.args.lr,
        #                                 momentum=0.5)
        optimizer = torch.optim.Adam(self.modelG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        for _ in range(self.args.local_gan_epoch):
            noise = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)
            fake = self.modelG(noise)
            self.modelG.zero_grad()
            wanted_labels = torch.full((self.batch_size,), self.want_label_index, dtype=torch.long, device=self.device)
            output = self.modelD(fake)
            loss = self.criterion(output, wanted_labels)
            loss.backward()
            optimizer.step()

    def generate_fake_datas(self, num):
        noise = torch.randn(num, self.nz, 1, 1, device=self.device)
        # fake_tensors  type:float tensor  shape:[num*1*28*28]
        fake_tensors = self.modelG(noise).to('cpu')
        fake_images = []
        fake_labels = []
        for i in range(len(fake_tensors)):
            fake_images.append(fake_tensors[i].detach())
            fake_labels.append(self.false_label_index)
        return ['fake - 10'], fake_images, fake_labels

class AdversaryGanUpdateMnist(AdversaryGanUpdate):
    def __init__(self, discriminator_model, generator_model, args, 
    logger, want_label_index, false_label_index=10) -> None:
        super(AdversaryGanUpdateMnist, self).__init__(discriminator_model, generator_model,
        args, logger, want_label_index, false_label_index)

class AdversaryGanUpdateCifar(AdversaryGanUpdate):
    def __init__(self, discriminator_model, generator_model, args, 
    logger, want_label_index, false_label_index=10) -> None:
        super(AdversaryGanUpdateCifar, self).__init__(discriminator_model, generator_model,
        args, logger, want_label_index, false_label_index)

class AdversaryGanUpdateSVHN(AdversaryGanUpdate):
    def __init__(self, discriminator_model, generator_model, args, 
    logger, want_label_index, false_label_index=10) -> None:
        super(AdversaryGanUpdateCifar, self).__init__(discriminator_model, generator_model,
        args, logger, want_label_index, false_label_index)