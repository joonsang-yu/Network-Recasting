from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
       
        # Block container
        self.blocks = model

        # Count the number of weighted blocks
        count = 0 
        for b in self.blocks:
            if isinstance(b, list):
                count += 1

        # Total number of weighted blocks
        self.num_weighted_blocks = count

    def ForwardConvBlock(self, op, x):

        # Convolution block structure
        # Dropout also can be included
        # We also consider FC block as special type of convolution block.
        # Conv: [Conv, BN, ReLU]
        # Some convolution block has only convolutional layer ([Conv])

        for m in op:
            x = m(x)

        return x

    def ForwardResidualBlock(self, op, x):

        # Residual block structures
        # BasicBlock: [Conv, BN, ReLU, Conv, BN, [Conv(downsample), BN], ReLU]
        # Bottleneck: [Conv, BN, ReLU, Conv, BN, ReLU, Conv, BN, [Conv(downsample), BN], ReLU]
        
        residual = x
      
        relu = op[-1]
        op = op[:-1]

        if len(op) == 6 or len(op) == 9:    # 6: BasicBlock, 9: Bottleneck
            downsample = True               
            down_op = op[-1]                # Detach downsample layer
            op = op[:-1]
        else :
            downsample = False
        
        for m in op :
            x = m(x)
        
        if downsample == True :             # Downsample layer operation
            for r in down_op :
                residual = r(residual)
        
        return relu(x + residual)                 # Summation with residual
    
    def ForwardDenseBlock(self, op, x):
        
        # Dense block structure
        # Dense-BC: [[BN, ReLU, Conv, BN, ReLU, Conv] * N, BN, ReLU]

        bn = op[-2]
        relu = op[-1]
        op = op[:-2]

        for sub_block in op:                # Dense block operation
            prev = x
            for m in sub_block:
                x = m(x)
                
            x = torch.cat((prev, x), 1)     # Channel-wise concatenation
       
        x = bn(x)
        return relu(x)

    def forward(self, x, next_block = None):
    
        if next_block is None or next_block >= self.num_weighted_blocks :
            next_block = self.num_weighted_blocks - 1

        i = 0
        weighted_block_idx = 0
        while weighted_block_idx <= next_block :

            op = self.blocks[i]
            i += 1
            
            # Flatten
            if op == 'Flatten' :
                _, d, h, w = x.size()
                x = x.view(-1, d * h * w)

            # Block is removed
            elif op == 'Removed' : 
                x = x                       
                weighted_block_idx += 1

            # Block operation
            elif isinstance(op, list) :                          
                if op[-1] == 'ConvBlock' or op[-1] == 'FCBlock':
                    x = self.ForwardConvBlock(op[:-1], x)
                elif op[-1] == 'ResidualBlock':
                    x = self.ForwardResidualBlock(op[:-1], x)
                elif op[-1] == 'DenseBlock':
                    x = self.ForwardDenseBlock(op[:-1], x)
                weighted_block_idx += 1

            # Non weighted block operation (ex. maxpooling)
            else :
                x = op(x)
            
        return x
        
    def SetMode(self, start, end, mode=True):
        
        # Set layer mode (train, test)
        # Both start and end block are included


        # Sync start index

        i = 0
        weighted_block_idx = 0

        while True:
            if isinstance(self.blocks[i], list) :
                if weighted_block_idx == start:
                    break;
                else :
                    weighted_block_idx += 1
            i += 1

        # Set mode
        while weighted_block_idx <= end:
            b = self.blocks[i]
            i += 1
            if isinstance(b, list) :
                for l in b[:-1] :
                    if isinstance(l, list) :
                        for s in l:
                            s.train(mode)
                    else :
                        l.train(mode)

                weighted_block_idx += 1

        return self
    
    def TrainMode(self):
        return self.SetMode(mode = True, start = 0, end = self.num_weighted_blocks - 1)
    
    def TestMode(self):
        return self.SetMode(mode = False, start = 0, end = self.num_weighted_blocks - 1)
    
    def PartialTrainMode(self, index):
        self.TestMode()
        return self.SetMode(mode = True, start = index, end = index + 1)
    
    def GetParams(self, start, end):

        # Return all parameters from blocks
        # The parameters of BN is also included

        params = []
       
        # Error case
        if (end > self.num_weighted_blocks):
            return params

        # Sync start index
        i = 0
        weighted_block_idx = 0

        while True:
            if isinstance(self.blocks[i], list) :
                if weighted_block_idx == start:
                    break;
                else :
                    weighted_block_idx += 1
            i += 1

        # Get parameters from each blocks
        while weighted_block_idx <= end:
            b = self.blocks[i]
            i += 1
            if isinstance(b, list) :
                weighted_block_idx += 1
                for l in b[:-1] :
                    if isinstance(l, list) :
                        for s in l :
                            for param in s.parameters():
                                params.append(param)
                    else :
                        for param in l.parameters():
                            params.append(param)
            
        return params

    def GetCurrParams(self, index):
        return self.GetParams(0, index + 1)
    
    def GetTotalParams(self):
        return self.GetParams(0, self.num_weighted_blocks - 1)

    def GetParamsReplicas(self):

        # return parameter replicas for multi gpu

        params = []
       
        i = 0
        weighted_block_idx = 0

        while weighted_block_idx < self.num_weighted_blocks:
            b = self.blocks[i]
            i += 1
            if isinstance(b, list) :
                weighted_block_idx += 1
                for l in b[:-1] :
                    if isinstance(l, list) :
                        for s in l :
                            for param in s.parameters():
                                params.append(param)
                    else :
                        for param in l.parameters():
                            params.append(param)
        return params
    
    def SetParamsReplicas(self, params):
        
        # Set parameters for multi gpu

        i = 0
        weighted_block_idx = 0

        params = list(params)
        while weighted_block_idx < self.num_weighted_blocks:
            b = self.blocks[i]
            i += 1
            if isinstance(b, list) :
                weighted_block_idx += 1
                for l in b[:-1] :
                    if isinstance(l, list) :
                        for s in l :
                            for key, _ in s._parameters.items():
                                if s._parameters[key] is not None:
                                    s._parameters[key] = params.pop(0)
                                
                    else :
                        for key, _ in l._parameters.items():
                            if l._parameters[key] is not None:
                                l._parameters[key] = params.pop(0)
         
        return self

    def GetBuffersReplicas(self):

        # Return buffers for multi gpu

        buffers = []
        
        i = 0
        weighted_block_idx = 0

        while weighted_block_idx < self.num_weighted_blocks:
            b = self.blocks[i]
            i += 1
            if isinstance(b, list) :
                weighted_block_idx += 1
                for l in b[:-1] :
                    if isinstance(l, list) :
                        for s in l :
                            for buf in s._all_buffers():
                                buffers.append(buf)
                    else :
                        for buf in l._all_buffers():
                            buffers.append(buf)
            
        return buffers
    
    def SetBuffersReplicas(self, buffers):

        # Set buffer values for multi gpu

        i = 0
        weighted_block_idx = 0

        buffers = list(buffers)
        while weighted_block_idx < self.num_weighted_blocks:
            b = self.blocks[i]
            i += 1
        
            if isinstance(b, list) :
                weighted_block_idx += 1
                for l in b[:-1] :
                    if isinstance(l, list) :
                        for s in l :
                            for key, _ in s._buffers.items():
                                if s._buffers[key] is not None:
                                    s._buffers[key] = buffers.pop(0)
                    else :
                        for key, _ in l._buffers.items():
                            if l._buffers[key] is not None:
                                l._buffers[key] = buffers.pop(0)
         
        return self
    
    def Cpu(self):

        # Move tensor to CPU

        i = 0
        weighted_block_idx = 0

        while weighted_block_idx < self.num_weighted_blocks:
            b = self.blocks[i]
            i += 1
            if isinstance(b, list) :
                weighted_block_idx += 1
                for l in b[:-1] :
                    if isinstance(l, list) :
                        for s in l :
                            s.cpu()
                    else :
                        l.cpu()
        return
    
    def Gpu(self, gpuid = 0):

        # Move tensor to CPU

        i = 0
        weighted_block_idx = 0

        while weighted_block_idx < self.num_weighted_blocks:
            b = self.blocks[i]
            i += 1
            if isinstance(b, list) :
                weighted_block_idx += 1
                for l in b[:-1] :
                    if isinstance(l, list) :
                        for s in l :
                            s.cuda(gpuid)
                    else :
                        l.cuda(gpuid)
        return
        
    def GetStateDict(self):

        # Get state for the save model 

        states = []
        i = 0
        weighted_block_idx = 0
        
        while weighted_block_idx < self.num_weighted_blocks:
            b = self.blocks[i]
            i += 1
            if isinstance(b, list) :
                weighted_block_idx += 1
                for l in b[:-1] :
                    if isinstance(l, list) :
                        for s in l :
                            states.append(s.state_dict())
                    else :
                        states.append(l.state_dict())
        
        return states 
    
    def LoadFromStateDict(self, state):

        # Load parameters from saved model

        i = 0
        weighted_block_idx = 0
        
        while weighted_block_idx < self.num_weighted_blocks:
            b = self.blocks[i]
            i += 1
            if isinstance(b, list) :
                weighted_block_idx += 1
                for l in b[:-1] :
                    if isinstance(l, list) :
                        for s in l :
                            s.load_state_dict(state.pop(0))
                    else :
                        l.load_state_dict(state.pop(0))
                
        return self
                    
    def GetBlockConfig(self, source_block_idx):
        
        # Return block configuration of the source block

        i = 0
        weighted_block_idx = 0

        while True:
            if isinstance(self.blocks[i], list) :
                if weighted_block_idx == source_block_idx:
                    break;
                else :
                    weighted_block_idx += 1
            i += 1

        block = self.blocks[i]
        block_type = block[-1]
       
        config = []
        config.append(block_type)

        if block_type == 'FCBlock':
            m = block[0]
            config.append(m.in_features)
            config.append(m.out_features)

            if isinstance(block[1], nn.BatchNorm1d):
                config.append(True)
            else:
                config.append(False)

            if isinstance(block[-1], nn.Dropout):
                config.append(True)
            else:
                config.append(False)

            config.append(m.bias is not None)

            if len(block) == 2 :
                config.append('FCOnly')
            else:
                config.append('Normal')

        elif block_type == 'ConvBlock':
            m = block[0]
            config.append(m.in_channels)
            config.append(m.out_channels)
            config.append(m.kernel_size)
            config.append(m.stride)
            config.append(m.padding)

            if isinstance(block[1], nn.BatchNorm2d):
                config.append(True)
            else:
                config.append(False)

            if isinstance(block[-2], nn.Dropout2d):
                config.append(True)
            else:
                config.append(False)

            config.append(m.bias is not None)

            if len(block) == 2 :
                config.append('ConvOnly')
            else:
                config.append('Normal')

        elif block_type == 'ResidualBlock':
            # BasicBlock: [Conv, BN, ReLU, Conv, BN, [Conv(downsample), BN], ReLU]
            # Bottleneck: [Conv, BN, ReLU, Conv, BN, ReLU, Conv, BN, [Conv(downsample), BN], ReLU]

            # Basic block case
            if len(block) < 9 :
                m = block[0]
                config.append(m.in_channels)
                config.append(m.out_channels)
                config.append(m.stride)
            
            # Bottleneck case
            else :
                m1 = block[0]
                m2 = block[3]
                m3 = block[6]
                config.append(m1.in_channels)
                config.append(m3.out_channels)
                config.append(m2.stride)
                # Mid channels for the bottleneck
                config.append(m1.out_channels)

        elif block_type == 'DenseBlock':
            # Dense-BC: [[BN, ReLU, Conv, BN, ReLU, Conv] * N, BN, ReLU]
            m1 = block[0]
            m2 = block[-3]

            # Get dimension from BN
            config.append(m1[0].num_features)
            config.append(m2.num_features)
            config.append(m1[3].num_features)
            # Dense output: [in_features, out_features, growth_rate]

        return config

    def Recasting(self, source_block_idx, target_block):

        i = 0
        weighted_block_idx = 0

        while True:
            if isinstance(self.blocks[i], list) :
                if weighted_block_idx == source_block_idx:
                    break;
                else :
                    weighted_block_idx += 1
            i += 1
        
        if weighted_block_idx >= self.num_weighted_blocks :
            print('Out of Range! Source block index is higher than the number of blocks')
            return self

        temp = self.blocks[i]
        self.blocks[i] = target_block
    
        return self
   
    def PrintBlocksSummary(self):
        print('# Input # \n')
        for b in self.blocks:
            if isinstance(b, list):
                print(b[-1])
            else :
                print(b)
        print('# Logit # \n')

    def PrintBlocksDetail(self):
        print('# Input # \n')
        for b in self.blocks:
            if isinstance(b, list):
                for sub_b in b:
	
                    if isinstance(sub_b, list):
                        for sub_sub_b in sub_b:
                            if isinstance(sub_sub_b, nn.Conv2d):
                                print('- - Conv2d, in_channels: %d, out_channels: %d kernel size: %d, stride: %d' 
					%(sub_sub_b.in_channels, sub_sub_b.out_channels, sub_sub_b.kernel_size[0], sub_sub_b.stride[0]))
                            elif isinstance(sub_sub_b, nn.BatchNorm2d):
                                print('- - BatchNorm2d')
                            elif isinstance(sub_sub_b, nn.ReLU):
                                print('- - ReLU')
                    elif isinstance(sub_b, nn.Linear):
                        print('- Linear, in_features: %d, out_features: %d' 
				%(sub_b.in_features, sub_b.out_features))
                    elif isinstance(sub_b, nn.Conv2d):
                        print('- Conv2d, in_channels: %d, out_channels: %d kernel size: %d, stride: %d' 
				%(sub_b.in_channels, sub_b.out_channels, sub_b.kernel_size[0], sub_b.stride[0]))
                    elif isinstance(sub_b, nn.BatchNorm1d):
                        print('- BatchNorm1d')
                    elif isinstance(sub_b, nn.BatchNorm2d):
                        print('- BatchNorm2d')
                    elif isinstance(sub_b, nn.ReLU):
                        print('- ReLU')
                print(b[-1], '\n')
            else :
	        
                if isinstance(b, nn.AvgPool2d):
                    print('AvgPool2d, kernel size: %d, stride: %d' %(b.kernel_size, b.stride))
                elif isinstance(b, nn.MaxPool2d):
                    print('MaxPool2d, kernel size: %d, stride: %d' %(b.kernel_size, b.stride))
                else :
                    print(b)
                print('')
	
        print('# Logit # \n')


    def LoadFromTorchvision(self, model):

        self.blocks = model

        # Count the number of weighted blocks
        count = 0 
        for b in self.blocks:
            if isinstance(b, list):
                count += 1

        # Total number of weighted blocks
        self.num_weighted_blocks = count
