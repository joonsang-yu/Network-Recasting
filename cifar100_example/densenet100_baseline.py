# Import torch and model

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import sys
sys.path.append("../common")

from model_generator import ModelGenerator
from net import Net

# Set hyper params

batch_size = 64
num_epoch  = 300

lr = 0.1
gamma = 0.2             # learning rate decay
weight_decay = 0.0001

## for SGD
opt_momentum = 0.9
opt_nesterov = True

dropout_on = False
batchnorm_on = True 

scheduler_step_size = [150, 225]

pretrained_model = './cifar100_densenet100_pretrained.pth'

# Load dataset

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

transform_train = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, 4),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# Define densenet 100

model_gen = ModelGenerator(dropout = dropout_on, batchnorm = batchnorm_on)

model_gen.CifarDensenetConfig(k = 12, num_layers = 100, cifar = 100)
model = model_gen.GetCifarDensenet()

net = Net(model)
net.Gpu()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.GetTotalParams(), lr=lr, weight_decay=weight_decay, momentum=opt_momentum, nesterov=opt_nesterov )
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = scheduler_step_size, gamma = gamma)

# Training (back-propagation)

for epoch in range(num_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    scheduler.step()
    net.TrainMode()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #scheduler.step()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss = (running_loss * i + loss.cpu().data.numpy()) / (i+1)

    correct = 0
    total = 0
    net.TestMode()
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
        
    print('%d epoch end, loss: %3.6f, Test Acc: %4.2f %%' %(epoch + 1, running_loss, 100 * correct / total))
    
print('\nTraining is finished!')

# Validation 

correct = 0
total = 0
net.TestMode()
for data in testloader:
    images, labels = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the 10000 test images: %4.2f %%' % (100 * correct / total))

torch.save(net.GetStateDict(), pretrained_model)
