import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as pretrained_models


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import sys
sys.path.append("../common")

from model_generator import ModelGenerator
from net import Net
import loader_util as util

# Hyper parameter
batch_size = 32

# Model save path
model_path = './imagenet_densenet121_pretrained.pth' 

# ILSVRC2012 dataset path
traindir = os.path.join('/home/data/ILSVRC2012/images', 'train')
valdir = os.path.join('/home/data/ILSVRC2012/images', 'val')

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

trainset = datasets.ImageFolder(root=traindir, transform=transform_train)
testset = datasets.ImageFolder(root=valdir, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

num_train_classes = len([name for name in os.listdir(traindir)])
num_valid_classes = len([name for name in os.listdir(valdir)])

print("num_train_classes = '{}'".format(num_train_classes))
print("num_valid_classes = '{}'".format(num_valid_classes))

densenet = pretrained_models.densenet121(num_classes=1000, pretrained='imagenet')
print('Pre-trained densenet121 is loaded')

# Parsing

net = []

features = densenet.features

b = []
b = util.ConvBlockLoader(features.conv0, features.norm0, features.relu0)

net.append(b)

net.append(features.pool0)

b = util.DenseBlockLoader(features.denseblock1, features.transition1.norm, features.transition1.relu)
net.append(b)

b = util.TransitionLoader(features.transition1)
net.append(b)

net.append(features.transition1.pool)

b = util.DenseBlockLoader(features.denseblock2, features.transition2.norm, features.transition2.relu)
net.append(b)

b = util.TransitionLoader(features.transition2)
net.append(b)

net.append(features.transition2.pool)

b = util.DenseBlockLoader(features.denseblock3, features.transition3.norm, features.transition3.relu)
net.append(b)

b = util.TransitionLoader(features.transition3)
net.append(b)

net.append(features.transition2.pool)

b = util.DenseBlockLoader(features.denseblock4, features.norm5)
net.append(b)

net.append(nn.AvgPool2d(kernel_size=7, stride=7, padding=0, ceil_mode=False, count_include_pad=True))

net.append('Flatten')

b = util.FCBlockLoader(densenet.classifier, option = 'FCOnly')
net.append(b)


# Model save

model_gen = ModelGenerator(dropout = False, batchnorm = True)
model_gen.ImagenetDensenetConfig(num_layers = 121)

model = model_gen.GetImagenetDensenet()
densenet121 = Net(model)

densenet121.LoadFromTorchvision(net)

torch.save(densenet121.GetStateDict(), './imagenet_densenet121_pretrained.pth')

print('Model saved')

# Model reload

model_gen = ModelGenerator(dropout = False, batchnorm = True)
model_gen.ImagenetDensenetConfig(num_layers = 121)

model = model_gen.GetImagenetDensenet()
densenet121 = Net(model)

state = torch.load('./imagenet_densenet121_pretrained.pth')
densenet121.LoadFromStateDict(state)

print('Reload model for the verification')

# Test

correct = 0
correct_top5 = 0
correct_tmp = 0

total = 0

densenet121.TestMode()
densenet121.Gpu()

print('Inference for test set')

for i, data in enumerate(testloader, 0):
    images, labels = data

    outputs = densenet121(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()
    
    _, predicted_top5 = torch.topk(outputs.data, 5)
    predicted_top5 = predicted_top5.t()
    predicted_top5_mat = predicted_top5.eq(labels.cuda().view(1,-1).expand_as(predicted_top5))

    for k in (1,5):
        correct_tmp = predicted_top5_mat[:k].view(-1).float().sum(0, keepdim=True)
    
    correct_top5 += correct_tmp[0]

    if i % 50 == 0:
        print ('.', end=' ')

print ('\n')
print('Top1 Acc: %4.2f %%, Top5 Acc: %4.2f %%' %(100 * correct / total, 100 * correct_top5 / total))
