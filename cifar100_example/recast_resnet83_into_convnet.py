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

dropout_on = True
batchnorm_on = True 

scheduler_step_size = 20

# for recasting

lr_recasting = 0.001
num_epoch_recasting = 60

# for fine tune

lr_fine_tune = 0.001
num_epoch_fine_tune = 100


model_gen = ModelGenerator(dropout = dropout_on, batchnorm = batchnorm_on)

model_gen.CifarResnetConfig(num_layers = 83, block_type = 'Bottleneck', cifar = 100)

# Recasting block
# 0: conv layer, 1-27: Residual block
recasting_block_indices = range(1, 28)
target_block_type = 'ConvBlock'

# Compression rate
# the number of filters decreased to [compression_rate]

compression_ratio = 1

## file path
pretrained_model = './cifar100_resnet83_pretrained.pth'
compressed_model = './cifar100_resnet83_to_convenet.pth'


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


# Load pre-trained model (teacher network)

model = model_gen.GetCifarResnet()
teacher = Net(model)

state = torch.load(pretrained_model)
teacher.LoadFromStateDict(state)

teacher.Gpu()

correct = 0
total = 0
teacher.TestMode()
for data in testloader:
    images, labels = data
    outputs = teacher(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the 10000 test images: %4.2f %%' % (100 * correct / total))


# Define student network

model = model_gen.GetCifarResnet()
student = Net(model)

state = torch.load(pretrained_model)
student.LoadFromStateDict(state)

student.Gpu()


correct = 0
total = 0
student.TestMode()
for data in testloader:
    images, labels = data
    outputs = student(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the 10000 test images: %4.2f %%' % (100 * correct / total))

# Sequential recasting

# define MSE loss
MSE = nn.MSELoss()

teacher.TestMode()

for block_idx in recasting_block_indices:
    
    ################################################    Recasting process ######################################################
    # current block recasting
    
    config = student.GetBlockConfig(block_idx)
    
    config[2] = round(config[2] * compression_ratio)    # apply compression ratio
    
    # Handling corner case: bottleneck block recasting
    if len(config) == 5:                         
        is_bottleneck = True
        mid_feature = config[4]
        # We reduce the output dimension of bottleneck block.
        # output dimension of new block is the same with output dimension of 3x3 conv in bottleneck block
        config[4] = round(mid_feature * compression_ratio)
    else :
        is_bottleneck = False
        
    new_block = model_gen.GenNewBlock([target_block_type, config])
    source_block_type = config[0]
    
    student.Recasting(block_idx, new_block)
    
    
    # next block recasting
    
    config = student.GetBlockConfig(block_idx + 1)
    
    config[1] = round(config[1] * compression_ratio)    # apply compression ratio
    
    # Handling corner case: bottleneck block recasting
    if is_bottleneck == True:                         
        # Change next input dim to output dim of target block
        config[1] = round(mid_feature * compression_ratio)
    
    new_block = model_gen.GenNewBlock([config[0], config])
    student.Recasting(block_idx + 1, new_block)
    
    ################################################    Recasting process end ##################################################
    
    student.Gpu()
    
    params = student.GetCurrParams(block_idx)
    
    optimizer = optim.Adam(params, lr = lr_recasting)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = scheduler_step_size)
    
    print('\nBlock %d recasting is done (%s -> %s).' %(block_idx, source_block_type, target_block_type))
    print('Training start\n')
    for epoch in range(num_epoch_recasting):  # loop over the dataset multiple times
        
        running_loss = 0.0
        scheduler.step()
        
        student.TrainMode()
            
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            corrects = teacher(inputs, next_block= block_idx + 1)
            outputs = student(inputs, next_block = block_idx + 1)

            targets = Variable(corrects.data.clone())
            
            loss = MSE(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss = (running_loss * i + loss.cpu().data.numpy()) / (i+1)

        
        correct = 0
        total = 0
        student.TestMode()
        for data in testloader:
            images, labels = data
            outputs = student(Variable(images.cuda()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()
        
        test_acc = 100 * correct / total
        

        print('(%d/%d) epoch end, loss: %3.6f, Test Acc: %4.2f %%' %(epoch + 1, num_epoch_recasting, running_loss, test_acc))
    
    
print('\nSequential recasting is finished')


# Fine-tuning (KD + Cross-entropy)


import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# define loss functions
MSE = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

# pruning ratio for every layer    
optimizer = optim.Adam(student.GetTotalParams(), lr = lr_fine_tune)
scheduler = lr_scheduler.StepLR(optimizer, step_size = scheduler_step_size)
teacher.TestMode()
student.Gpu()

print('Fine tuning start\n')

for epoch in range(num_epoch_fine_tune):  # loop over the dataset multiple times

    running_loss = 0.0
    scheduler.step()
    student.TrainMode()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        corrects = teacher(inputs)
        outputs = student(inputs)

        targets = Variable(corrects.data.clone())
        loss_KD = MSE(outputs, targets)
        loss_CE = criterion(outputs, labels)
        
        loss = loss_KD + loss_CE
        
        loss.backward()
        optimizer.step()

        running_loss = (running_loss * i + loss.cpu().data.numpy()) / (i+1)

    correct = 0
    total = 0
    student.TestMode()
    for data in testloader:
        images, labels = data
        outputs = student(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

    print('(%d/%d) epoch end, loss: %3.6f, Test Acc: %4.2f %%' %(epoch + 1, num_epoch_fine_tune, running_loss, 100 * correct / total))
    
print('\nFine tuning is finished')


print('Teacher network architecture \n')
teacher.PrintBlockDetail()

print('Student network architecture \n')
student.PrintBlockDetail()
