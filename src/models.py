#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.modules.activation import LogSoftmax


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

class DCGANDiscriminator_mnist(nn.Module):
    def __init__(self, args) -> None:
        super(DCGANDiscriminator_mnist, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.Tanh(),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(32, 64, 5),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 200),
            nn.Tanh(),
            nn.Linear(200, 11),
            nn.LogSoftmax()
        )
    
    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 256)
        return self.fc(x)

class DCGANGenerator_mnist(nn.Module):
    def __init__(self, args) -> None:
        super(DCGANGenerator_mnist, self).__init__()
        self.main = nn.Sequential(
            # 100*1

            nn.ConvTranspose2d(100, 256, 4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 256*4*4

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128*8*8

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64*16*16

            # 按照论文里的结构得到的是1*32*32的输出，但mnist需要的是1*28*28
            # nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            # nn.Tanh()
            # # 1*32*32

            # 修改结构
            nn.ConvTranspose2d(64, 1, 4, 2, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class DCGANDiscriminator_cifar10(nn.Module):
    def __init__(self, args) -> None:
        super(DCGANDiscriminator_cifar10, self).__init__()
        self.conv = nn.Sequential(
            # 3*32*32

            nn.Conv2d(3, 32, 5),
            # 32*28*28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 32*14*14
            
            nn.Conv2d(32, 64, 3),
            # 64*12*12
            nn.ReLU(),

            nn.Conv2d(64, 128, 3),
            # 128*10*10
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 128*5*5
        )

        self.fc = nn.Sequential(
            nn.Linear(128*5*5, 1024),
            nn.ReLU(),
            nn.Linear(1024, 200),
            nn.ReLU(),
            nn.Linear(200, 11),
            nn.LogSoftmax()
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128*5*5)
        return self.fc(x)


class DCGANGenerator_cifar10(nn.Module):
    def __init__(self, args) -> None:
        super(DCGANGenerator_cifar10, self).__init__()
        self.main = nn.Sequential(
            # 100*1

            nn.ConvTranspose2d(100, 256, 4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 256*4*4

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128*8*8

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64*16*16

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # 3*32*32
        )

    def forward(self, input):
        return self.main(input)

class DCGANDiscriminator_ATTFace(nn.Module):
    pass

class DCGANGenerator_ATTFace(nn.Module):
    pass