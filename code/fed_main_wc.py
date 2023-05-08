import torch
import time
import pickle
from tqdm import tqdm
import numpy as np
import os
from tensorboardX import SummaryWriter
import copy
from model import cnn_mnist, cnn_fmnist, cnn_cifar, MLP
from model_update import LocalUpdate, test_inference
from arguments import argument
from aggregation import fed_avg
from utils import details, get_data
# from optimization import moo_ACS
import random
from pytorchtools import EarlyStopping


def main():
    start_time = time.time()

    path_project = os.path.abspath('..')

    logger = SummaryWriter('../logs')

    args = argument()
    # print(args)
    details(args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train, test, user_groups = get_data(args)
    # print(train)
    # print(test)
    # len_dict = {key: len(value) for key, value in user_groups.item()}
    # print(len(user_groups[99]))

    # build neural network model
    # 1. build convolutional neural network
    # 2. build multilayer perceptron

    if args.model == 'cnn':
        # print("here")
        if args.dataset == 'mnist':
            global_model = cnn_mnist(args=args)
            # print(global_model)
        elif args.dataset == 'fmnist':
            global_model = cnn_mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = cnn_cifar(args=args)
            # print(global_model)
    elif args.model == 'mlp':
        img_size = train[0][0].shape
        len_in = 1

        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.n_classes)

    else:
        exit("Error: unrecognized model")

    global_model.to(device)
    global_model.train()

    # copy weights

    global_weights = global_model.state_dict()

    # print(global_weights)

    # training
    train_loss, train_accuracy = [], []
    val_accuracy, val_loss = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    early_stopping = EarlyStopping(patience=args.stop, verbose=True)
    idxs_users_wc = random.sample(range(1, 10000), 100)
    print("idxs_users_wc",idxs_users_wc)
    max_time = np.array([])
    if args.user_type == 0:
        print("data division is random")
        for epoch in range(args.epoch):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch + 1} |\n')

            global_model.train()

            # Random client selection
            if args.rm_straggler == 0:
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                idxs_users_wc = random.sample(range(1, 10000), m)
                print("idxs_users_wc", idxs_users_wc)

            elif args.rm_straggler == 1:
                m = max(int(args.frac * args.num_users * (1 - args.straggler_frac)), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                print("at rm_straggler")
                print("straggler free nodes = ",m)
                print("straggler free nodes = ", m)

            else:
                print("Wrong entry")
                # straggler percentage 50%

            idxs_epoch = [args.l_epoch] * (int((1 - args.straggler_frac) * args.num_users) + 1)
            # print("len idxs_epoch", len(idxs_epoch))
            # print("idxs_epoch = ",idxs_epoch)
            for i in range(int(args.straggler_frac * args.num_users)):
                idxs_epoch.append(random.randint(1, int(0.5 * args.l_epoch)))
            random.shuffle(idxs_epoch)
            if len(idxs_epoch) % 10 != 0:
                idxs_epoch.pop()
            # print("idxs_epoch", idxs_epoch)
            # print("len_epoch:", len(idxs_epoch))

            # print("idxs_epoch1", idxs_epoch1)
            # idxs_epoch = np.array(idxs_epoch1)
            # idxs_epoch = np.full(shape=int(args.num_users/2), fill_value=args.l_epoch, dtype=np.int)
            # idxs_epoch = idxs_epoch.astype(np.int32)
            # print("idxs_epoch = ",idxs_epoch)
            # stag_epoch = np.random.randint(low=1, high=int(args.l_epoch), size=int(args.num_users)),
            # stag_epoch = stag_epoch.tolist()
            # print("stag_epoch = ", stag_epoch)
            # idxs_epoch = np.concatenate((idxs_epoch, stag_epoch))
            # np.random.shuffle(idxs_epoch)
            # print("idxs_epoch = ",idxs_epoch)

            # print(m)
            # print(idxs_users)
            participants_list = []
            for idx in idxs_users:
                participants_list.append([idx, idxs_epoch[idx]])

                # print("idxs:", user_groups[idx])
                # print("idxe: ", idxs_epoch[idx])
                if args.rm_straggler == 0:
                    local_model = LocalUpdate(args=args, dataset=train, idxs=user_groups[idx], idxe=10,
                                              logger=logger)
                elif args.rm_straggler == 1:
                    local_model = LocalUpdate(args=args, dataset=train, idxs=user_groups[idx], idxe=args.l_epoch,
                                              logger=logger)
                else:
                    print("wrong entry")

                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))


            max_delay = np.amax(idxs_users_wc)
            print("max_delay", max_delay)
            max_time = np.append(max_time, max_delay)
            print("max_time", max_time)
            # print(local_losses)
            # update global weights
            print("participants list", participants_list)
            global_weights = fed_avg(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc_val, list_loss_val = [], []
            global_model.eval()
            for c in range(args.num_users):
                # print("c:", c)
                local_model = LocalUpdate(args=args, dataset=train,
                                          idxs=user_groups[c], idxe=idxs_epoch[c], logger=logger)
                acc, loss = local_model.inference(model=global_model)
                list_acc_val.append(acc)
                list_loss_val.append(loss)
            val_accuracy.append(sum(list_acc_val) / len(list_acc_val))
            val_loss.append(sum(list_loss_val) / len(list_loss_val))

            # Early stopping

            # print("loss:", train_loss)
            # print global training loss after every 'i' rounds
            if (epoch + 1) % print_every == 0:
                print(f' \nAvg validation Stats after {epoch + 1} global rounds:')
                print(f'Validation Loss : {val_loss[-1]}')
                print('Validation Accuracy: {:.2f}% \n'.format(100 * val_accuracy[-1]))
            early_stopping(val_loss[-1], global_model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        global_model.load_state_dict(torch.load('checkpoint.pt'))

        # Test inference after completion of training
        test_acc, test_loss, f1, p, r = test_inference(args, global_model, test)
        val_accuracy_100 = [i * 100 for i in val_accuracy]
        print(f' \n Results after {args.epoch} global rounds of training:')
        print(" Train loss: ", train_loss)
        print(" Validation loss: ", val_loss)
        print(" Validation accuracy: ", val_accuracy_100)
        print(" Test Accuracy: {:.2f}%".format(100 * test_acc))
        print(" Test loss: {:.2f}".format(test_loss))

        # Saving the objects train_loss and train_accuracy:
        file_name = path_project + '/save/Fed-MOODS-Random_{}_{}_method[{}]_straggler[{}]_iid[{}]_LE[{}]_B[{}]_GE[{}]_adaptive[{}].pkl'. \
            format(args.dataset, args.model, args.method, args.straggler_frac, args.iid,
                   args.l_epoch, args.l_batch, args.epoch, args.adaptiveness)

        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, val_loss, val_accuracy_100, [100 * test_acc, test_loss], [f1, p, r],
                         max_time],
                        f)

        '''with open(file_name, 'wb') as f:
            pickle.dump([train_loss], f)
            pickle.dump([val_loss, val_accuracy_100], f)
            pickle.dump(100 * test_acc, f)
            pickle.dump(test_loss, f)
            pickle.dump(time.time() - start_time, f)'''

        print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    # selection based computational speed

    elif args.user_type == 1:
        flag = 0
        idxs_users = [14, 63, 64, 40, 51, 32, 71, 21, 46, 62, 9, 8, 36, 52, 54, \
                      12, 77, 44, 90, 2, 38, 5, 93, 60, 20, 23, 95, 87, 31, 67, 88, \
                      65, 73, 13, 43, 11, 17, 25, 76, 26, 15, 47, 53, 80, 94, 19, 56, \
                      42, 34, 48, 72, 16, 50, 6, 49, 28, 30, 4, 33, 24, 78, 39, 57, 35, \
                      69, 70, 82, 66, 79, 45, 68, 86, 37, 89, 91, 92, 3, 98, 61, 55, 41, \
                      97, 27, 74, 85, 18, 0, 84, 1, 10, 7, 83, 96, 22, 29, 99, 58, 59, 75, 81]

        idxs_users_wc.sort()
        print(len(idxs_users_wc))
        print("idxs_users_wc", idxs_users_wc)


        idxs_users[::-1]
        print("len_idxs_users",len(idxs_users))
        # idxs_epoch = [args.l_epoch] * int(args.num_users / 2)
        # print("idxs_epoch = ",idxs_epoch)
        # for i in range(int(args.num_users / 2)):
        #     idxs_epoch.append(random.randint(1, args.l_epoch))
        # idxs_epoch.sort(reverse=True)
        # print("idxs_epoch :",idxs_epoch)
        # straggler percentage 50%

        '''idxs_epoch = [args.l_epoch] * (int((1 - args.straggler_frac) * args.num_users) + 1)
        print("len idxs_epoch", len(idxs_epoch))
        print("idxs_epoch = ",idxs_epoch)
        
        for i in range(int(args.straggler_frac * args.num_users)):
            idxs_epoch.append(random.randint(1, int(0.5 * args.l_epoch)))
        if len(idxs_epoch) % 10 != 0:
            idxs_epoch.pop()
        idxs_epoch[::-1]'''

        idxs_epoch = [args.l_epoch] * args.num_users
        print("len idxs_epoch", len(idxs_epoch))
        print("idxs_epoch = ", idxs_epoch)
        print('idxs_users', idxs_users)

        epoch_list = [None] * 100
        max_time = np.array([])
        for i in range(len(epoch_list)):
            epoch_list.append(args.l_epoch)
        for epoch in range(args.epoch):
            #time_taken = []
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch + 1} |\n')

            global_model.train()

            # selection based computational speed
            idxs_users = np.array(idxs_users)
            # print(idxs_users)
            if flag == 0:
                idxs1 = idxs_users[0: args.adaptiveness]
                flag += args.adaptiveness
            else:
                if flag + args.adaptiveness <= len(idxs_users):
                    flag += args.adaptiveness
                    idxs1 = idxs_users[0:flag]
                    print("idxs1 :", len(idxs1))

                else:
                    idxs1 = idxs_users[0:len(idxs_users)]
                    print("idxs1 :", len(idxs1))

            # print(" len idxs1", len(idxs1))
            for idx in idxs1:
                print(idxs_epoch[idx])


                local_model = LocalUpdate(args=args, dataset=train, idxs=user_groups[idx], idxe=idxs_epoch[idx],
                                          logger=logger)
                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
            # print(local_losses)
            # update global weights

            global_weights = fed_avg(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            #wall clock time simulation
            time_taken = np.array(idxs_users_wc[0:len(idxs1)])
            print("time_taken", time_taken)
            max_delay = np.amax(time_taken)
            print("max_delay", max_delay)
            max_time = np.append(max_time, max_delay)
            print("max_time", max_time)
            # Calculate avg training accuracy over all users at every epoch
            list_acc_val, list_loss_val = [], []
            global_model.eval()
            for c in range(args.num_users):
                # print("c:", c)
                local_model = LocalUpdate(args=args, dataset=train,
                                          idxs=user_groups[c], idxe=idxs_epoch[c], logger=logger)
                acc, loss = local_model.inference(model=global_model)
                list_acc_val.append(acc)
                list_loss_val.append(loss)
            val_accuracy.append(sum(list_acc_val) / len(list_acc_val))
            val_loss.append(sum(list_loss_val) / len(list_loss_val))
            # print("loss:", train_loss)
            # print global training loss after every 'i' rounds
            if (epoch + 1) % print_every == 0:
                print(f' \nAvg validation Stats after {epoch + 1} global rounds:')
                print(f'Validation Loss : {val_loss[-1]}')
                print('Validation Accuracy: {:.2f}% \n'.format(100 * val_accuracy[-1]))
            early_stopping(val_loss[-1], global_model)
            print('time:', time.time()-start_time)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        global_model.load_state_dict(torch.load('checkpoint.pt'))

        # Test inference after completion of training
        test_acc, test_loss, f1, p, r = test_inference(args, global_model, test)
        val_accuracy_100 = [i * 100 for i in val_accuracy]
        print(f' \n Results after {args.epoch} global rounds of training:')
        # print(" Avg Train Accuracy: {:.2f}%".format(100 * sum(train_accuracy) / len(train_accuracy)))

        print(f' \n Results after {args.epoch} global rounds of training:')
        print(" Train loss: ", train_loss)
        print(" Validation loss: ", val_loss)
        print(" Validation accuracy: ", val_accuracy_100)
        print(" Test Accuracy: {:.2f}%".format(100 * test_acc))
        print(" Test loss: {:.2f}".format(test_loss))

        # Saving the objects train_loss and train_accuracy:
        file_name = path_project + '/save/Fed-MOODS-WC_{}_{}_method[{}]_straggler[{}]_iid[{}]_LE[{}]_B[{}]_GE[{}]_adaptive[{}].pkl'. \
            format(args.dataset, args.model, args.method, args.straggler_frac, args.iid,
                   args.l_epoch, args.l_batch, args.epoch, args.adaptiveness)

        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, val_loss, val_accuracy_100, [100 * test_acc, test_loss], [f1, p, r],
                         max_time],
                        f)
        # pickle.dump([val_loss, val_accuracy_100], f)
        # pickle.dump(100 * test_acc, f)
        # pickle.dump(test_loss, f)
        # pickle.dump(time.time() - start_time, f)

    #print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    '''elif args.user_type == 2:

        for epoch in range(args.epoch):
            if epoch % 10 == 0:
                idxs1 = []
                idxs_users = moo_ACS()

            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch + 1} |\n')

            global_model.train()

            # selection based on multi-objective optimization
            for ix in range(len(idxs_users)):
                idxs1.append(idxs_users[ix])
                print(" len idxs1", len(idxs1))
                for idx in idxs1:
                    local_model = LocalUpdate(args=args, dataset=train, idxs=user_groups[idx], logger=logger)
                    w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))
                print(local_losses)
                # update global weights
                global_weights = fed_avg(local_weights)

                # update global weights
                global_model.load_state_dict(global_weights)

                loss_avg = sum(local_losses) / len(local_losses)
                train_loss.append(loss_avg)

                # Calculate avg training accuracy over all users at every epoch
                list_acc_val, list_loss_val = [], []
                global_model.eval()
                for c in range(args.num_users):
                    print("c:", c)
                    local_model = LocalUpdate(args=args, dataset=train,
                                              idxs=user_groups[idx], logger=logger)
                    acc, loss = local_model.inference(model=global_model)
                    list_acc_val.append(acc)
                    list_loss_val.append(loss)
                val_accuracy.append(sum(list_acc_val) / len(list_acc_val))
                val_loss.append(sum(list_loss_val) / len(list_loss_val))
                print("loss:", train_loss)
                # print global training loss after every 'i' rounds
                if (epoch + 1) % print_every == 0:
                    print(f' \nAvg validation Stats after {epoch + 1} global rounds:')
                    print(f'Validation Loss : {val_loss[-1]}')
                    print('Validation Accuracy: {:.2f}% \n'.format(100 * val_accuracy[-1]))

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test)
        print(f' \n Results after {args.epoch} global rounds of training:')
        # print(" Avg Train Accuracy: {:.2f}%".format(100 * sum(train_accuracy) / len(train_accuracy)))
        print(" Test Accuracy: {:.2f}%".format(100 * test_acc))
        print(" Test loss: {:.2f}".format(test_loss))
        # Saving the objects train_loss and train_accuracy:
        file_name = '/home/soura/WASP-Phd/straggler-mitigation-fl/save/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
            format(args.dataset, args.model, args.epoch, args.frac, args.iid,
                   args.l_epoch, args.l_batch)

        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy], f)
            pickle.dump(100 * test_acc, f)
            pickle.dump(test_loss, f)
            pickle.dump(time.time() - start_time, f)
        print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))'''


if __name__ == "__main__":
    main()

