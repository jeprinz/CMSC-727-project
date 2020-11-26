# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

import argparse
import optuna
from Net import Net


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def objective(trial, args):
    '''
    Optimize the hyperpatameters
    :param trial:
    :param args: arg parser with epochs, use_rprop
    :return:
    '''
    learning_rate = trial.suggest_uniform('learning_rate', 0.001, 0.5)
    momentum = trial.suggest_uniform('momentum', 0.1, 0.9)
    batch_size = int(trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64, 128]))

    valid_accuracy, _, _ = run_model(epochs=args.epochs, batch_size=batch_size, use_rprop=args.use_rprop,
                         learning_rate=learning_rate, momentum=momentum)
    return -1 * valid_accuracy


def run_model(epochs, batch_size, use_rprop, learning_rate, momentum):
    '''
    Function to run (train and test) the model once
    :param epochs: number of training epochs
    :param batch_size:
    :param use_rprop: True if using rprop optimizer, False if using SGD optimizer
    :param learning_rate:
    :param momentum:
    :return:
    '''
    # load data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    validset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    num_train = len(trainset)
    valid_size = 0.1
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              sampler=train_sampler, num_workers=0)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                              sampler=valid_sampler, num_workers=0)

    #set num_workers to 0 if you get a BrokenPipeError
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)


    # set up the model and optimizer
    net = Net()

    criterion = nn.CrossEntropyLoss()
    if(use_rprop):
        optimizer = optim.Rprop(net.parameters(), lr=learning_rate) #(default params: lr = 0.01, etas = (0.5,1.2), step_sizes(1e-06,50))
    else:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # train the model
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        total_train = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            total_train += labels.size(0)


        # test the model ont he validation set
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the ', total, ' validation images: %d %%' % (
                100 * correct / total))
        valid_accuracy = 100 * correct / total

    print('Finished Training')
    print("train size: ", total_train)

    return valid_accuracy, net, testloader


def test_model(testloader, net):
    # test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the ', total, ' test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


# parse command line args
parser = argparse.ArgumentParser(description='Collect hyperparameters.')
parser.add_argument('--epochs', type=int, help='number of epochs for training')
parser.add_argument('--num_trials', type=int,
                        help='The number of times Optuna will train the model. Higher means better optimization, but longer training time')
parser.add_argument('--batch_size', type=int, nargs='?', default=16, help='number of samples per training batch')
parser.add_argument('--learning_rate', type=float, nargs='?', default=0.1, help='the learning rate between 0 and 1')
parser.add_argument('--momentum', type=float, nargs='?', default=0, help='the momentum to use betweeon 0 and 1')
parser.add_argument('--use_rprop', type=bool, help='True if using rprop, False if using sgd')
args = parser.parse_args()


if args.num_trials > 0:
    # create study and optimize
    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, args), n_trials=args.num_trials)
    print("The best parameters are: \n", study.best_params)
else:
    # only train the model on the specified params and test it
    _, net, testloader = run_model(args.epochs, args.batch_size, args.use_rprop, args.learning_rate, args.momentum)
    test_model(testloader, net)

