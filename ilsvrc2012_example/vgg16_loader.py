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
model_path = './imagenet_vgg16_pretrained.pth' 

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

vggnet = pretrained_models.vgg16_bn(num_classes=1000, pretrained='imagenet')
print('Pre-trained resnet50 is loaded')


# Parsing


net = []

for x in vggnet.features:
    
    if isinstance(x, nn.Conv2d):
        b = []
        b.append(x)
        b.append('ConvBlock')
        net.append(b)
        
    elif isinstance(x, nn.BatchNorm2d) :
        net[-1].insert(-1, x)
        
    elif isinstance(x, nn.ReLU) :
        net[-1].insert(-1, x)
        
    elif isinstance(x, nn.Dropout2d) :
        net[-1].insert(-1, x)
        
    else :
        net.append(x)
    
net.append('Flatten')

for x in vggnet.classifier:
    if isinstance(x, nn.Linear):
        b = []
        b.append(x)
        b.append('FCBlock')
        net.append(b)
        
    elif isinstance(x, nn.BatchNorm1d) :
        net[-1].insert(-1, x)
        
    elif isinstance(x, nn.ReLU) :
        net[-1].insert(-1, x)
        
#    elif isinstance(x, nn.Dropout) :
#        net[-1].insert(-1, x)
        
    else :
        net.append(x)


# Model save

model_gen = ModelGenerator(dropout = False, batchnorm = True)
model_gen.ImagenetVgg16Config(dropout = False, batchnorm =True)

model = model_gen.GetImagenetVgg16()
vgg16_ = Net(model)

vgg16_.LoadFromTorchvision(net)

torch.save(vgg16_.GetStateDict(), './imagenet_vgg16_pretrained.pth')

print('Model saved')

# Model reload

model_gen = ModelGenerator(dropout = False, batchnorm = True)
model_gen.ImagenetVgg16Config(dropout = False, batchnorm =True)

model = model_gen.GetImagenetVgg16()
vgg16_ = Net(model)

state = torch.load('./imagenet_vgg16_pretrained.pth')
vgg16_.LoadFromStateDict(state)

print('Reload model for the verification')

# Test

correct = 0
correct_top5 = 0
correct_tmp = 0

total = 0

vgg16_.TestMode()
vgg16_.Gpu()

print('Inference for test set')

for i, data in enumerate(testloader, 0):
    images, labels = data

    outputs = vgg16_(Variable(images.cuda()))
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
        print ('.', end='', flush=True)

print ('\n')
print('Top1 Acc: %4.2f %%, Top5 Acc: %4.2f %%' %(100 * correct / total, 100 * correct_top5 / total))
