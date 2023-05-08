'''
Input arguments:
'''
import argparse


def argument():
    parser = argparse.ArgumentParser()

    # main arguments

    parser.add_argument("--dataset", type=str, default='mnist', help=" MNIST ")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--gpu", default=None, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument("--optimizer", type=str, default='sgd', help="type of optimizer")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--stop", type=int, default=10, help="rounds of early stopping")
    parser.add_argument("--verbose", type=int, default=1, help="verbose")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--mu', type=float, default=0.01, help="proximal term constant")

    # arguments for federated settings

    parser.add_argument("--straggler_frac", type=float, default=0.5, help="amount of straggler present")
    parser.add_argument("--adaptiveness", type=int, default=5, help="adaptiveness added")

    parser.add_argument("--epoch", type=int, default=10, help="number of training rounds")
    parser.add_argument("--num_users", type=int, default=100, help="number of training clients")
    parser.add_argument("--iid", type=int, default=0, help="dataset type iid or non-iid")
    parser.add_argument("--l_epoch", type=int, default=10, help="number of local rounds")
    parser.add_argument("--l_batch", type=int, default=128, help="local batch size")
    parser.add_argument("--n_equal", type=int, default=0, help='data distribution (use 0 for equal division)')
    parser.add_argument("--user_type", type=int, default=0, help='random : random uscriterion = nn.CrossEntropyLoss()er selection \n \
                                                                        FLANP: Additive user selection \n \
                                                                        MOOACS: Multi objective based selection')
    parser.add_argument("--method", type=str, default='fedavg', help='fedavg, fedprox')
    parser.add_argument('--frac', type=float, default=0.2, help='the fraction of participating devices')
    parser.add_argument("--rm_straggler", type=int, default=0, help='0: without removing stragglers \n \
                                                                    1: remove stragglers')
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                            use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                            of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                            mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                            strided convolutions")
    args = parser.parse_args()
    return args
