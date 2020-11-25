# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import argparse
import optuna



def objective(trial):
    # parse command line args
    parser = argparse.ArgumentParser(description='Collect hyperparameters.')
    parser.add_argument('--epochs', type=int, help='number of epochs for training')
   # parser.add_argument('--batch_size', type=int, help='number of samples per training batch')
    parser.add_argument('--use_rprop', type=bool, help='True if using rprop, False if using sgd')
    args = parser.parse_args()

    learning_rate = trial.suggest_uniform('learning_rate', 0.001, 0.5)
    momentum = trial.suggest_uniform('momentum', 0.1, 0.9)
    batch_size = int(trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64, 128]))

    accuracy = run_model(epochs=args.epochs, batch_size=batch_size, use_rprop=args.use_rprop,
                         learning_rate=learning_rate, momentum=momentum)
    return -1 * accuracy


def run_model(epochs, batch_size, use_rprop, learning_rate, momentum):
    # load data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    #set num_workers to 0 if you get a BrokenPipeError
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # define the model
    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(6,16,5)
            self.fc1 = nn.Linear(16*5*5,120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self,x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # train the model
    net = Net()

    criterion = nn.CrossEntropyLoss()
    if(use_rprop):
        optimizer = optim.Rprop(net.parameters()) #(default params: lr = 0.01, etas = (0.5,1.2), step_sizes(1e-06,50))
    else:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
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

    print('Finished Training')


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

    print('Accuracy of the network on the 10000 test images: %d %%' % (
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

    return 100 * correct / total  # return the overall accuracy


study = optuna.create_study()
study.optimize(objective, n_trials=10)

print(study.best_params)  # E.g. {'x': 2.002108042}
