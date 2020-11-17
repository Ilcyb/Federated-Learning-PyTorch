#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from options import args_parser
from update import LocalUpdate, test_inference, AdversaryGanUpdateMnist, AdversaryGanUpdateCifar, AdversaryUpdate, AdversaryGanUpdateSVHN
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, DCGANDiscriminator_mnist, DCGANGenerator_mnist, DCGANDiscriminator_cifar10, DCGANGenerator_cifar10, DCGANDiscriminator_SVHN, DCGANGenerator_SVHN
from utils import get_dataset, average_weights, exp_details, get_dataset_ganattack, get_dataset_split_by_label, \
                  get_dataset_idxgroup_ganattack, get_experiment_result_location, save_grid, generate_gif_from_file, \
                  generate_gif_from_list,plot_loss_acc, compute_avgpsnr, plot_avg_psnr

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device('cuda:{}'.format(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    if args.model == 'dcgan':
        # train_dataset, test_dataset, user_groups = get_dataset(args)
        _, _, user_groups = get_dataset(args)
        train_dataset, test_dataset, label_indexs = get_dataset_split_by_label(args)
    else:
        train_dataset, test_dataset, user_groups = get_dataset(args)
    global_model = None

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            # global_model = DCGANDiscriminator_cifar10(args=args)
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)

    elif args.model == 'dcgan':
        # deep convolutional generative adversarial networks
        if args.dataset == 'mnist':
            global_model = DCGANDiscriminator_mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = DCGANDiscriminator_cifar10(args=args)
        elif args.dataset == 'svhn':
            global_model = DCGANDiscriminator_SVHN(args=args)
        else:
            # TODO add datasets support
            exit('Error: unrecognized dataset')
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    fake_images = []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    avg_psnrs = []
    print_every = 2
    val_loss_pre, counter = 0, 0
    save_location = get_experiment_result_location(args.model, args.dataset,
                                                   args.wanted_label_index,
                                                   {'ganlr': args.local_gan_lr, 
                                                   'ganepoch': args.local_gan_epoch, 
                                                   'optimizer': args.optimizer,
                                                   'localepoch':args.local_ep},
                                                   args.mode,
                                                   args.experiment_name)

    # adversary model
    if args.model == 'dcgan' and args.dataset == 'mnist':
        generator_model = DCGANGenerator_mnist(args=args)
        adversary_gan_update = AdversaryGanUpdateMnist(copy.deepcopy(global_model), generator_model,
                                                       args, logger, args.wanted_label_index, false_label_index=10)
    elif args.model == 'dcgan' and args.dataset == 'cifar':
        generator_model = DCGANGenerator_cifar10(args=args)
        adversary_gan_update = AdversaryGanUpdateCifar(copy.deepcopy(global_model), generator_model,
                                                       args, logger, args.wanted_label_index, false_label_index=10)
    elif args.model == 'dcgan' and args.dataset == 'svhn':
        generator_model = DCGANGenerator_SVHN(args=args)
        adversary_gan_update = AdversaryGanUpdateCifar(copy.deepcopy(global_model), generator_model,
                                                       args, logger, args.wanted_label_index, false_label_index=10)

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        label_split = [[i] for i in range(10)]
        idx_group = get_dataset_idxgroup_ganattack(args, label_split, label_indexs)
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        # 从总用户中随机抽取需要的用户
        idxs_users = np.random.choice(
            range(0, args.num_users), m, replace=False)

        # FIXME 当前non-iid实现下 label的划分组合个数只能写死 所以参加训练的用户数也只能为这个写死的固定值
        assert len(label_split) == max(int(args.frac * args.num_users), 1)

        data_idx = 0
        for idx in idxs_users:
            # TODO 不应该每一轮都新建一个Update类
            global_model_copy = copy.deepcopy(global_model)
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=idx_group[data_idx], logger=logger)
            w, loss = local_model.update_weights(
                model=global_model_copy, global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            data_idx += 1

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        # print('test idx:{}'.format(idx))
        for c in range(args.num_users):
            # FIXME
            # 这里的user_groups[idx]是否应该是user_groups[c]?
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # 服务器进行攻击
    if args.model == 'dcgan':
        global_model_copy = copy.deepcopy(global_model)
        server_adversary = AdversaryUpdate(args=args, dataset=train_dataset,
                                            idxs=[], logger=logger,
                                            adversary_gan_update=adversary_gan_update,
                                            discriminator_model=global_model_copy)
        want_targets = (train_dataset.targets == args.wanted_label_index)
        want_targets = [i for i in range(len(want_targets)) if want_targets[i]==True]
        for epoch in range(args.epochs):
            server_adversary.train_generator()
            w = server_adversary.update_weights(global_model_copy, epoch)
            global_model_copy.load_state_dict(w)
            server_adversary.update_discriminator(global_model_copy)
            randz = torch.randn(1, 100, 1, 1, device=device)
            generated_fake_image = generator_model(randz).to('cpu').detach()
            vutils.save_image(
                generated_fake_image, os.path.join(save_location, os.path.join('fake_images', 'epoch_{}.png'.format(epoch))))
            fake_images.append(generated_fake_image[0])

            # 随机抽取图片计算 AVG PSNR
            random_image_idxs = np.random.choice(want_targets, 10, replace=False)
            batch_images = []
            for idx in random_image_idxs:
                batch_images.append(train_dataset.data[idx])
            avg_psnr = compute_avgpsnr(generated_fake_image, batch_images)
            avg_psnrs.append(avg_psnr)

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    plot_loss_acc(train_loss, train_accuracy, save_location)
    plot_avg_psnr(avg_psnrs, save_location)
    generate_gif_from_file(os.path.join(save_location, 'fake_images'), os.path.join(save_location, 'training.gif'))
    print('fake images shape:{}'.format(fake_images[0].shape))
    save_grid(fake_images, save_location)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    # file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
