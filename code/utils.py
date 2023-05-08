"""
Dataset MNIST, FMNIST, CIFAR, 
divide into iid and non-iid
author : sourasb@cs.umu.se
"""
import torch as th
from torchvision import datasets, transforms
from data_div import iid_mnist, iid_cifar, iid_fmnist, non_iid_fmnist, non_iid_mnist, non_iid_cifar, non_iid_mnist_uneq


def get_data(args):
    if args.dataset == 'mnist':
        data_dir = '../data/mnist/'

        TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.13,), (0.3,))])

        train = datasets.MNIST(data_dir, train=True, download=True, transform=TRANSFORM)
        test = datasets.MNIST(data_dir, train=False, download=True, transform=TRANSFORM)

        if args.iid == 0:
            user_groups = iid_mnist(train, args.num_users)

        elif args.iid == 1:
            user_groups = non_iid_mnist(train, args.num_users)

        elif args.iid == 2:
            user_groups = non_iid_mnist_uneq(train, args.num_users)

    elif args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train = datasets.CIFAR10(data_dir, train=True, download=True,
                                 transform=apply_transform)

        test = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=apply_transform)

        if args.iid == 0:
            user_groups = iid_cifar(train, args.num_users)

        elif args.iid == 1:
            user_groups = non_iid_cifar(train, args.num_users)

        # elif args.iid == 2:
        # user_groups = non_iid_mnist_uneq(train, args.num_users)
    elif args.dataset == 'fmnist':
        data_dir = '../data/fmnist/'

        TRANSFORM = transforms.Compose([
            transforms.ToTensor()
        ])

        train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=TRANSFORM)
        test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=TRANSFORM)

        if args.iid == 0:
            user_groups = iid_fmnist(train, args.num_users)

        elif args.iid == 1:
            user_groups = non_iid_fmnist(train, args.num_users)

    return train, test, user_groups


def details(args):
    print("----------Experiment---------")
    print(args)
    print(f' Model : {args.model}')
    print(f'')
    print(f' Optimizer : {args.optimizer}')
    print(f' learning rate : {args.lr}')
    print(f' Global Rounds : {args.epoch}')

    print(' Federated settings')
    if args.iid == 0:
        print('Dataset : IID')
    elif args.iid == 1:
        print('Dataset : non-IID')
    elif args.iid == 2:
        print('Dataset : non-IID_random')
    else:
        print("wrong dataset type")
    if args.user_type == 0:
        print(f'Fraction of participating users: {args.frac}')
    else:
        print(' Adaptive participation')

    '''
    Method 
    '''

    if args.method == 'fedavg':
        print("method :", args.method)
    elif args.method == 'fedprox':
        print("method :", args.method)
    else:
        print("unsuitable method")

    '''
    Proximal term
    '''

    print("Proximal term constant", args.mu)




    print(f'Local batch size: {args.l_batch}')
    print(f'Local epochs: {args.l_epoch}')
    print(f'straggler_frac: {args.straggler_frac}')
