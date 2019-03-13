# ## Import torch and model

import os

if 'IMAGENET' in os.environ:
    IMAGENET_PATH = os.environ['IMAGENET']
else :
    print('Please set environment variable IMAGENET to your ilsvrc2012 path')

import warnings
warnings.filterwarnings('ignore')


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


# Multi gpu mode
from DataParallel_KD import DataParallel_KD


## Set hyper params

##################### Hyper parameters #############################

num_workers = 32

batch_size = 128

lr_recasting = 0.0005
lr_fine_tune = 0.0001

num_epoch_recasting = 10
num_epoch_fine_tune = 20

scheduler_step_size = 7

gamma = 0.1


model_gen = ModelGenerator(dropout = False, batchnorm = True)
model_gen.ImagenetResnetConfig(num_layers = 50, block_type = 'Bottleneck')

# Recasting block
# 0: conv layer, 1-16: Residual block
recasting_block_indices = range(1, 12)
target_block_type = 'ConvBlock'

# Compression rate
# the number of filters decreased to [compression_rate]

compression_ratio = 1

## file path
pretrained_model = './imagenet_resnet50_pretrained.pth'
compressed_model = './imagenet_resnet50_to_mixed_arch.pth'


## Load dataset

traindir = os.path.join(IMAGENET_PATH, 'train')
valdir = os.path.join(IMAGENET_PATH, 'val')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

trainset = torchvision.datasets.ImageFolder(root=traindir, transform=transform_train)
testset = torchvision.datasets.ImageFolder(root=valdir, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

num_train_classes = len([name for name in os.listdir(traindir)])
num_valid_classes = len([name for name in os.listdir(valdir)])

print("num_train_classes = '{}'".format(num_train_classes))
print("num_valid_classes = '{}'".format(num_valid_classes))


## Load pre-trained model

model = model_gen.GetImagenetResnet()
teacher = Net(model)

state = torch.load(pretrained_model)
teacher.LoadFromStateDict(state)

teacher.Gpu()


correct = 0
correct_top5 = 0
correct_tmp = 0

total = 0


for data in testloader:
    images, labels = data
    
    
    outputs = teacher(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    
    _, predicted_top5 = torch.topk(outputs.data, 5)
    predicted_top5 = predicted_top5.t()
    predicted_top5_mat = predicted_top5.eq(labels.cuda().view(1,-1).expand_as(predicted_top5))

    del outputs
    for k in (1,5):
        correct_tmp = predicted_top5_mat[:k].view(-1).float().sum(0, keepdim=True)
    
    correct_top5 += correct_tmp[0]

    print ('.', end=' ')

print ('\n')
print('Top1 Acc: %4.2f %%, Top5 Acc: %4.2f %%' %(100 * correct / total, 100 * correct_top5 / total))


## Load model for transformation

model = model_gen.GetImagenetResnet()
student = Net(model)

state = torch.load(pretrained_model)
student.LoadFromStateDict(state)

student.Gpu()

correct = 0
correct_top5 = 0
correct_tmp = 0

total = 0

for data in testloader:
    images, labels = data

    outputs = student(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    
    _, predicted_top5 = torch.topk(outputs.data, 5)
    predicted_top5 = predicted_top5.t()
    predicted_top5_mat = predicted_top5.eq(labels.cuda().view(1,-1).expand_as(predicted_top5))

    del outputs
    for k in (1,5):
        correct_tmp = predicted_top5_mat[:k].view(-1).float().sum(0, keepdim=True)
    
    correct_top5 += correct_tmp[0]

    print ('.', end=' ')

print ('\n')
print('Top1 Acc: %4.2f %%, Top5 Acc: %4.2f %%' %(100 * correct / total, 100 * correct_top5 / total))


## Sequential recasting

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
            targets = Variable(corrects.data.cpu().clone())
            del corrects
            
            outputs = student(inputs, next_block = block_idx + 1)
            
            loss = MSE(outputs, targets.cuda())
            loss.backward()
            optimizer.step()

            del outputs
            
            running_loss = (running_loss * i + loss.cpu().data.numpy()) / (i+1)
            
        
    correct = 0
    correct_top5 = 0
    correct_tmp = 0

    total = 0

    student.TestMode()

    for data in testloader:
        images, labels = data

        outputs = student(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

        _, predicted_top5 = torch.topk(outputs.data, 5)
        predicted_top5 = predicted_top5.t()
        predicted_top5_mat = predicted_top5.eq(labels.cuda().view(1,-1).expand_as(predicted_top5))

        del outputs
        for k in (1,5):
            correct_tmp = predicted_top5_mat[:k].view(-1).float().sum(0, keepdim=True)

        correct_top5 += correct_tmp[0]

    print('Top1 Acc: %4.2f %%, Top5 Acc: %4.2f %%' %(100 * correct / total, 100 * correct_top5 / total))
    print ('\n')
        

print('Recasting is finished')


## Fine-tuning (KD + Cross-entropy)

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

print('Fine tuning start')

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
        targets = Variable(corrects.data.cpu().clone())
        del corrects
            
        outputs = student(inputs)

        loss_KD = MSE(outputs, targets.cuda())
        loss_CE = criterion(outputs, labels)
        
        loss = loss_KD + loss_CE
        
        loss.backward()
        optimizer.step()

        del outputs
        
        running_loss = (running_loss * i + loss.cpu().data.numpy()) / (i+1)
        
    
    
    if epoch % 5 == 0 :
        correct = 0
        correct_top5 = 0
        correct_tmp = 0

        total = 0

        for data in testloader:
            images, labels = data

            outputs = student(Variable(images.cuda()))
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()

            _, predicted_top5 = torch.topk(outputs.data, 5)
            predicted_top5 = predicted_top5.t()
            predicted_top5_mat = predicted_top5.eq(labels.cuda().view(1,-1).expand_as(predicted_top5))

            for k in (1,5):
                correct_tmp = predicted_top5_mat[:k].view(-1).float().sum(0, keepdim=True)

            correct_top5 += correct_tmp[0]

            del outputs
            
        print ('\n')
        print('Top1 Acc: %4.2f %%, Top5 Acc: %4.2f %%' %(100 * correct / total, 100 * correct_top5 / total))
        
    else :
        print ('.', end=' ')
    
print('Fine tuning is finished')

