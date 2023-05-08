import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
from sklearn import metrics


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


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, idxe, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.idxe = idxe
        # method = method
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (70, 20, 10)
        idxs_train = idxs[:int(0.7 * len(idxs))]
        idxs_val = idxs[int(0.7 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.l_batch, shuffle=True)
        # print(trainloader)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val) / 10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)), shuffle=False)

        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        global_model = copy.deepcopy(model)
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr) #,
                                      #  momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        else:
            print("Error in optimizer")

        # select proximal criteria
        if self.args.method == 'fedprox':
            proximal_criterion = nn.MSELoss(reduction='mean')

        for iter in range(self.idxe):
            batch_loss = []
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            #print(device)
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(device), labels.to(device)
                # print("batch_idx",batch_idx)
                # print("images :",images)
                # print("labels :",labels)
                # clear the gradients
                model.zero_grad()
                # make a forword pass
                log_probs = model(images)

                # calculate the loss + proximal term
                #_, pred = torch.max(log_probs, 1)
                pred = model(images)

                if self.args.method == 'fedprox':
                    proximal_term = 0.0

                    # iterate through the current and global model parameters
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        # update proximal term
                        proximal_term += (w - w_t).norm(2)
                    loss = self.criterion(log_probs, labels) + (self.args.mu / 2) * proximal_term
                else:
                    loss = self.criterion(pred, labels.reshape(-1))
                # do a backwards pass
                loss.backward()
                # perform a optimization step
                optimizer.step()

                # if self.args.verbose and (batch_idx % 10 == 0):
                # print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                # global_round, iter, batch_idx * len(images),
                # len(self.trainloader.dataset),
                # 100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            loss_epoch = sum(epoch_loss) / len(epoch_loss)
        # print("epoch_loss:", loss_epoch)

        return model.state_dict(), loss_epoch

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss = []
        total, correct = 0.0, 0.0
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #print(device)
        batch_index = 0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(device), labels.to(device)
            batch_index += 1
            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss.append(batch_loss.item())

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            # print("pred_labels.dtype",pred_labels.dtype)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        val_loss = sum(loss) / len(loss)
        return accuracy, val_loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)
    batch_index = 0
    loss = []
    predict = []
    actual = []
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        batch_index += 1
        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
       #  print(batch_loss.item())
        loss.append(batch_loss.item())
        # print("test loss:",loss)
        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        predict.append(pred_labels.to("cpu").tolist())
        actual.append(labels.to("cpu").tolist())
    flat_predict = [item for sublist in predict for item in sublist]
    flat_actual = [item for sublist in actual for item in sublist]
    # print("predict",flat_predict)
    # print("actual",flat_actual)
    precision = metrics.precision_score(flat_actual, flat_predict, average='macro')
    print("precision:", precision)
    recall = metrics.recall_score(flat_actual, flat_predict, average='macro')
    print("recall", recall)
    accuracy = metrics.accuracy_score(flat_actual, flat_predict)
    print("accuracy", accuracy)
    f1_score = metrics.f1_score(flat_actual, flat_predict, average='macro')
    print("F1 score", f1_score)
    loss = np.array(loss)
    print("correct :", correct)
    print("total :", total)
    test_loss = sum(loss) / len(loss)
    print("test loss now:", test_loss)
    print("bnatch index:", len(loss))
    # print("batch idx:",batch_idx)

    # print('| test loss : {} |  : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    # global_round, iter, batch_idx * len(images),
    # len(self.trainloader.dataset),
    # 100. * batch_idx / len(self.trainloader), loss.item()))
    # accuracy = correct / total
    return accuracy, test_loss, f1_score, precision, recall
