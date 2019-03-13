import torch.nn as nn

def FCBlockLoader(fc, bn = None, relu = None, dropout = None, option = 'Normal'):
    
    block = []
    
    block.append(fc)
    
    if option == 'FCOnly':
        block.append('FCBlock')
        return block
    
    if bn != None :
        block.append(bn)
    if relu != None :
        block.append(relu)
    if dropout != None :
        block.append(dropout)
    
    block.append('FCBlock')
    
    return block

def ConvBlockLoader(conv, bn = None, relu = None, dropout = None, option = 'Normal'):
    
    block = []
    
    block.append(conv)
    
    if option == 'ConvOnly':
        block.append('ConvBlock')
        return block
    
    if bn != None :
        block.append(bn)
    if relu != None :
        block.append(relu)
    if dropout != None :
        block.append(dropout)
    
    block.append('ConvBlock')
    
    return block

def BasicBlockLoader(basicblock):
    block = []

    block.append(basicblock.conv1)
    block.append(basicblock.bn1)
    block.append(basicblock.relu)
    
    block.append(basicblock.conv2)
    block.append(basicblock.bn2)
    
    if basicblock.downsample != None :
        sub_block = []
        for x in basicblock.downsample:
            sub_block.append(x)
        block.append(sub_block)
        
    block.append(basicblock.relu)
    block.append('ResidualBlock')
    return block

def BottleneckLoader(bottleneck):
    block = []
    
    block.append(bottleneck.conv1)
    block.append(bottleneck.bn1)
    block.append(bottleneck.relu)
    
    block.append(bottleneck.conv2)
    block.append(bottleneck.bn2)
    block.append(bottleneck.relu)
    
    block.append(bottleneck.conv3)
    block.append(bottleneck.bn3)
    
    if bottleneck.downsample != None :
        sub_block = []
        for x in bottleneck.downsample:
            sub_block.append(x)
        block.append(sub_block)
    
    block.append(bottleneck.relu)
    block.append('ResidualBlock')
    return block

def DenseBlockLoader(dense, bn, relu = None):
    
    num_conv = len(dense)
    block = []
    for sub_block in dense :
        sub_b = []
        for m in sub_block :
            sub_b.append(m)
        block.append(sub_b)
    
    block.append(bn)
    
    if relu == None :
        relu = nn.ReLU(inplace=True)
        
    block.append(relu)
    
    block.append('DenseBlock')
    return block

def TransitionLoader(tran):
    b = []
    b.append(tran.conv)
    
    b.append('ConvBlock')
        
    return b
