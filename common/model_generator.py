import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ModelGenerator:
    def __init__(self, dropout = True, batchnorm = False):
        
        self.cifar10_cnn = self.Cifar10CnnConfig(dropout, batchnorm)
        self.cifar_resnet = 0
        self.cifar_wrn = 0
        self.cifar_densenet = 0
        self.cifar_vgg16 = 0
        
        self.imagenet_alexnet = self.ImagenetAlexnetConfig(dropout, batchnorm)
        self.imagenet_vgg16 = self.ImagenetVgg16Config(dropout, batchnorm)
        self.imagenet_resnet = 0
        self.imagenet_densenet = 0
       
    def Generator (self, config):

        blocks = []

        for idx, module in enumerate(config, 0):
            module_type = module[0]

            # Maxpooling layer 
            if module_type == 'Max2d':
                block = nn.MaxPool2d(kernel_size = module[1],
                                 stride = module[2],
                                 padding = module[3])
                
            # Averagepooling layer
            elif module_type == 'Avg2d':
                block = nn.AvgPool2d(kernel_size = module[1],
                                 stride = module[2],
                                 padding = module[3])

            # Flatten layer (Conv (2d) -> Flatten (2d->1d) -> Linear (1d))
            elif module_type == 'Flatten':
                block = 'Flatten'

            # FC Block
            # It can contain both BN and ReLU as well as Linear module.  
            # Expected output structure: [Linear, BN, ReLU] or [Linear]
            elif module_type == 'FCBlock' :
                block = []

                modules = module[1]

                for m in modules :
                    if m[0] == 'FC':
                        b = nn.Linear(in_features = m[1],
                                      out_features = m[2],
                                      bias = m[3])
                        nn.init.xavier_uniform(b.weight.data)
                        block.append(b)
                    elif m[0] == 'BN1d':
                        b = nn.BatchNorm1d(num_features = m[1])
                        block.append(b)
                    elif m[0] == 'ReLU':
                        b = nn.ReLU(inplace=True)
                        block.append(b)
                    elif m[0] == 'Dropout1d':
                        b = nn.Dropout(p = m[1])
                        block.append(b)

                block.append('FCBlock')

            # Conv Block
            # It can contain both BN and ReLU as well as Conv2d module.  
            # Expected output structure: [Conv2d, BN, ReLU] or [Conv2d]
            elif module_type == 'ConvBlock' :
                block = []

                modules = module[1]

                for m in modules :
                    if m[0] == 'Conv2d':
                        b = nn.Conv2d(in_channels = m[1],
                                      out_channels = m[2],
                                      kernel_size = m[3],
                                      stride = m[4],
                                      padding = m[5],
                                      bias = m[6])
                        nn.init.xavier_uniform(b.weight.data)
                        block.append(b)
                    elif m[0] == 'BN2d':
                        b = nn.BatchNorm2d(num_features = m[1])
                        block.append(b)
                    elif m[0] == 'ReLU':
                        b = nn.ReLU(inplace=True)
                        block.append(b)
                    elif m[0] == 'Dropout2d':
                        b = nn.Dropout2d(p = m[1])
                        block.append(b)

                block.append('ConvBlock')
            
            # Residual Block
            # Downsample block may appear, but most block does not have downsample block.
            # BasicBlock: [Conv, BN, ReLU, Conv, BN, [Conv(downsample)], ReLU]
            # Bottleneck: [Conv, BN, ReLU, Conv, BN, ReLU, Conv, BN, [Conv(downsample)], ReLU]
            elif module_type == 'BasicBlock' or module_type == 'Bottleneck':
                block = []
               
                modules = module[1]

                relu = nn.ReLU(inplace=True)
                modules = modules[:-1]
            
                if len(modules) == 6 or len(modules) == 9 :
                    downsample = True
                    sub_module = modules[-1]
                    modules = modules[:-1]
                else :
                    downsample = False
                
                for m in modules :
                    if m[0] == 'Conv2d_res' :
                        b = nn.Conv2d(in_channels = m[1],
                                      out_channels = m[2],
                                      kernel_size = m[3],
                                      stride = m[4],
                                      padding = m[5],
                                      bias = False)
                        nn.init.xavier_uniform(b.weight.data)
                        block.append(b)
                        
                    elif m[0] == 'BN2d' :
                        b = nn.BatchNorm2d(num_features = m[1])
                        block.append(b)
                    
                    elif m[0] == 'ReLU' :
                        b = nn.ReLU(inplace=True)
                        block.append(b)
                    
                if downsample == True :
                    sub_block = []
                    
                    for m in sub_module :
                        if m[0] == 'Conv2d_res' :
                            b = nn.Conv2d(in_channels = m[1],
                                          out_channels = m[2],
                                          kernel_size = m[3],
                                          stride = m[4],
                                          padding = m[5],
                                          bias = False)
                            nn.init.xavier_uniform(b.weight.data)
                            sub_block.append(b)
                            
                        if m[0] == 'BN2d' :
                            b = nn.BatchNorm2d(num_features = m[1])
                            sub_block.append(b)
                
                    block.append(sub_block)
              
                if relu != 0 :
                    block.append(relu)

                block.append('ResidualBlock')

            # Dense Block 
            # Dense-BC: [[BN, ReLU, Conv, BN, ReLU, Conv] * N, BN, ReLU]
                
            elif module_type == 'DenseBlock':
                block = []

                modules = module[1]

                m = modules[-2]
                bn = nn.BatchNorm2d(num_features = m[1])
                relu = nn.ReLU(inplace=True)
                modules = modules[:-2]

                for sub_m in modules :
                    sub_b = []
                    for m in sub_m:

                        if m[0] == 'Conv2d_dense' :
                            b = nn.Conv2d(in_channels = m[1],
                                          out_channels = m[2],
                                          kernel_size = m[3],
                                          stride = m[4],
                                          padding = m[5],
                                          bias = False)
                            nn.init.xavier_uniform(b.weight.data)
                            sub_b.append(b)

                        elif m[0] == 'BN2d' :
                            b = nn.BatchNorm2d(num_features = m[1])
                            sub_b.append(b)

                        elif m[0] == 'ReLU' :
                            b = nn.ReLU(inplace=True)
                            sub_b.append(b)

                    block.append(sub_b)

                block.append(bn)
                block.append(relu)
                block.append('DenseBlock')

            blocks.append(block)
        return blocks

    def FCBlock (self, in_feature, out_feature, batchnorm = True, dropout = False, bias = True, option = 'Normal'):
        block = []

        block.append(('FC', in_feature, out_feature, bias))

        if option == 'FCOnly':
            return ('FCBlock', block)

        if batchnorm == True:
            block.append(('BN1d', out_feature))
        block.append(('ReLU',False))
        if dropout == True :
            block.append(('Dropout1d', 0.2))

        return ('FCBlock', block)

    def ConvBlock (self, in_feature, out_feature, kernel_size, stride, padding, batchnorm = True, dropout = False, bias = True, option = 'Normal'):
        block = []

        block.append(('Conv2d', in_feature, out_feature, kernel_size, stride, padding, bias))


        if option == 'ConvOnly':
            return ('ConvBlock', block)

        if batchnorm == True:
            block.append(('BN2d', out_feature))
        block.append(('ReLU',False))
        if dropout == True :
            block.append(('Dropout2d', 0.2))

        return ('ConvBlock', block)

    def BasicBlock (self, in_feature, out_feature, stride):
        block = []
        
        block.append(('Conv2d_res', in_feature, out_feature, 3, stride, 1))
        block.append(('BN2d', out_feature))
        block.append(('ReLU',False))
        block.append(('Conv2d_res', out_feature, out_feature, 3, 1, 1))
        block.append(('BN2d', out_feature))
        
        if in_feature != out_feature :
            sub_block = []
            
            sub_block.append(('Conv2d_res', in_feature, out_feature, 1, stride, 0))
            sub_block.append(('BN2d', out_feature))
            
            block.append(sub_block)
        
        block.append(('ReLU',False))
        return ('BasicBlock', block)

    
    def Bottleneck (self, in_feature, mid_feature, out_feature, stride):
        
        block = []
        
        block.append(('Conv2d_res', in_feature, mid_feature, 1, 1, 0))
        block.append(('BN2d', mid_feature))
        block.append(('ReLU',False))
        block.append(('Conv2d_res', mid_feature, mid_feature, 3, stride, 1))
        block.append(('BN2d', mid_feature))
        block.append(('ReLU',False))
        block.append(('Conv2d_res', mid_feature, out_feature, 1, 1, 0))
        block.append(('BN2d', out_feature))
        
        if in_feature != out_feature :
            sub_block = []
            
            sub_block.append(('Conv2d_res', in_feature, out_feature, 1, stride, 0))
            sub_block.append(('BN2d', out_feature))
            
            block.append(sub_block)
                
        block.append(('ReLU',False))
        return ('Bottleneck', block)    
        
    def DenseBlock (self, in_feature, out_feature, growth_rate, num_conv):
        
        block = []
        
        for i in range(num_conv):
            subblock = []
            subblock.append(('BN2d', in_feature + growth_rate * i))
            subblock.append(('ReLU', False))
            subblock.append(('Conv2d_dense', in_feature + growth_rate * i, 4 * growth_rate, 1, 1, 0))
            subblock.append(('BN2d', 4 * growth_rate))
            subblock.append(('ReLU', False))
            subblock.append(('Conv2d_dense', 4 * growth_rate, growth_rate, 3, 1, 1))
            block.append(subblock)
            
        block.append(('BN2d', out_feature))
        block.append(('ReLU',False))
        return ('DenseBlock', block)
    ###################
    ##   CIFAR10     ##
    ###################
    def Cifar10CnnConfig(self, dropout, batchnorm):
        config = []
      
        config.append(self.ConvBlock(3, 32, 3, 1, 0, batchnorm, dropout))
        config.append(self.ConvBlock(32, 32, 3, 1, 0, batchnorm, dropout))
        
        config.append(('Max2d', 2, 2, 0))
        
        config.append(self.ConvBlock(32, 64, 3, 1, 0, batchnorm, dropout))
        config.append(self.ConvBlock(64, 64, 3, 1, 0, batchnorm, dropout))
        
        config.append(('Max2d', 2, 2, 0))
        
        config.append(('Flatten',False))
        
        config.append(self.FCBlock(64 * 5 * 5, 512, batchnorm, dropout))
        config.append(self.FCBlock(512, 10, option = 'FCOnly'))
        
        return config
    
    def CifarResnetConfig(self, num_layers, block_type = 'BasicBlock', cifar = 10):
        config = []
        
        num_fc = 1
        
        config.append(self.ConvBlock(3, 16, 3, 1, 1, bias = False, option = 'ConvOnly'))
        
        if block_type == 'BasicBlock' :
            N = int((num_layers - 2) / 6)
            config.append(self.BasicBlock(16, 16, 1))
            for i in range(N-1):
                config.append(self.BasicBlock(16, 16, 1))
        
            config.append(self.BasicBlock(16, 32, 2))
            for i in range(N-1):
                config.append(self.BasicBlock(32, 32, 1))
                
            config.append(self.BasicBlock(32, 64, 2))
            for i in range(N-1):
                config.append(self.BasicBlock(64, 64, 1))
            
            num_fc = 64
                
        elif block_type == 'Bottleneck' :
            N = int((num_layers - 2) / 9)
            config.append(self.Bottleneck(16, 16, 16*4, 1))
            for i in range(N-1):
                config.append(self.Bottleneck(16*4, 16, 16*4, 1))

            config.append(self.Bottleneck(16*4, 32, 32*4, 2))
            for i in range(N-1):
                config.append(self.Bottleneck(32*4, 32, 32*4, 1))

            config.append(self.Bottleneck(32*4, 64, 64*4, 2))
            for i in range(N-1):
                config.append(self.Bottleneck(64*4, 64, 64*4, 1))
                
            num_fc = int(64 * 4)
        
        config.append(('Avg2d', 8, 8, 0))
        config.append(('Flatten',False))
        config.append(self.FCBlock(num_fc * 1 * 1, cifar, option = 'FCOnly'))
        
        self.cifar_resnet = config

    def CifarWrnConfig(self, k, num_layers, cifar = 10):
        config = []
        
        
        config.append(self.ConvBlock(3, 16, 3, 1, 1, bias = False, option = 'ConvOnly'))
        
        N = int((num_layers - 4) / 6)
        config.append(self.BasicBlock(16, 16 * k, 1))
        for i in range(N-1):
            config.append(self.BasicBlock(16 * k, 16 * k, 1))

        config.append(self.BasicBlock(16 * k, 32 * k, 2))
        for i in range(N-1):
            config.append(self.BasicBlock(32 * k, 32 * k, 1))

        config.append(self.BasicBlock(32 * k, 64 * k, 2))
        for i in range(N-1):
            config.append(self.BasicBlock(64 * k, 64 * k, 1))

        num_fc = 64 * k
        
        config.append(('Avg2d', 8, 8, 0))
        config.append(('Flatten',False))
        config.append(self.FCBlock(num_fc * 1 * 1, cifar, option = 'FCOnly'))
        
        self.cifar_wrn = config        
        
    
    def CifarDensenetConfig (self, k, num_layers, cifar = 10):
        
        import math
        config = []
        N = int((num_layers - 4) / 6)
        reduction = 0.5
        
        config.append(self.ConvBlock(3, k * 2, 3, 1, 1, option = 'ConvOnly'))
       
        k1 = k * 2
        config.append(self.DenseBlock(k1, k1 + k*N, k, N))
        k2 = math.floor((k1 + k*N) * reduction)

        config.append(self.ConvBlock(k1 + k*N, k2, 1, 1, 0, option = 'ConvOnly'))
        config.append(('Avg2d', 2, 2, 0))
       
        config.append(self.DenseBlock(k2, k2 + k*N, k, N))
        k3 = math.floor((k2+ k*N) * reduction)   

        config.append(self.ConvBlock(k2 + k*N, k3, 1, 1, 0, option = 'ConvOnly'))
        config.append(('Avg2d', 2, 2, 0))
        
        config.append(self.DenseBlock(k3, k3 + k*N, k, N))
        
        config.append(('Avg2d', 8, 8, 0))
        
        config.append(('Flatten',False))
        
        config.append(self.FCBlock((k3 + k*N) * 1 * 1, cifar, option = 'FCOnly'))
        
        self.cifar_densenet = config

    def CifarVgg16Config(self, cifar = 10):
        config = []
        
        # Conv2d -> in_channels, out_channels, kernel_size, stride, padding
        # Max2d  -> kernel_size, stride, padding
        Conv_dropout = False
        
        config.append(self.ConvBlock(3, 64, 3, 1, 1, True, False))
        config.append(self.ConvBlock(64, 64, 3, 1, 1, True, False))

        config.append(('Max2d', 2, 2, 0))
        
        config.append(self.ConvBlock(64, 128, 3, 1, 1, True, False))
        config.append(self.ConvBlock(128, 128, 3, 1, 1, True, False))

        config.append(('Max2d', 2, 2, 0))
        
        config.append(self.ConvBlock(128, 256, 3, 1, 1, True, False))
        config.append(self.ConvBlock(256, 256, 3, 1, 1, True, False))
        config.append(self.ConvBlock(256, 256, 3, 1, 1, True, False))
        
        config.append(('Max2d', 2, 2, 0))
        
        config.append(self.ConvBlock(256, 512, 3, 1, 1, True, False))
        config.append(self.ConvBlock(512, 512, 3, 1, 1, True, False))
        config.append(self.ConvBlock(512, 512, 3, 1, 1, True, False))
        
        config.append(('Max2d', 2, 2, 0))
        
        config.append(self.ConvBlock(512, 512, 3, 1, 1, True, False))
        config.append(self.ConvBlock(512, 512, 3, 1, 1, True, False))
        config.append(self.ConvBlock(512, 512, 3, 1, 1, True, False))

        config.append(('Max2d', 2, 2, 0))
        
        config.append(('Flatten',False))
        
        config.append(self.FCBlock(512 * 1 * 1, 512, True, False))
        config.append(self.FCBlock(512, cifar, option = 'FCOnly'))
                
        self.cifar_vgg16 = config
         
    ###################
    ##   Imagenet    ##
    ###################
    def ImagenetVgg16Config(self, dropout, batchnorm):
        config = []
        
        config.append(self.ConvBlock(3, 64, 3, 1, 1, True, False, bias = True))
        config.append(self.ConvBlock(64, 64, 3, 1, 1, True, False, bias = True))
        
        config.append(('Max2d', 2, 2, 0))
        
        config.append(self.ConvBlock(64, 128, 3, 1, 1, True, False, bias = True))
        config.append(self.ConvBlock(128, 128, 3, 1, 1, True, False, bias = True))

        config.append(('Max2d', 2, 2, 0))
        
        config.append(self.ConvBlock(128, 256, 3, 1, 1, True, False, bias = True))
        config.append(self.ConvBlock(256, 256, 3, 1, 1, True, False, bias = True))
        config.append(self.ConvBlock(256, 256, 3, 1, 1, True, False, bias = True))
        
        config.append(('Max2d', 2, 2, 0))
        
        config.append(self.ConvBlock(256, 512, 3, 1, 1, True, False, bias = True))
        config.append(self.ConvBlock(512, 512, 3, 1, 1, True, False, bias = True))
        config.append(self.ConvBlock(512, 512, 3, 1, 1, True, False, bias = True))
        
        config.append(('Max2d', 2, 2, 0))
        
        config.append(self.ConvBlock(512, 512, 3, 1, 1, True, False, bias = True))
        config.append(self.ConvBlock(512, 512, 3, 1, 1, True, False, bias = True))
        config.append(self.ConvBlock(512, 512, 3, 1, 1, True, False, bias = True))

        config.append(('Max2d', 2, 2, 0))
        
        config.append(('Flatten',False))
        
        config.append(self.FCBlock(512 * 7 * 7, 4096, False, dropout, bias = True))
        config.append(self.FCBlock(4096, 4096, False, dropout, bias = True))
        config.append(self.FCBlock(4096, 1000, bias = True, option = 'FCOnly'))
               
        return config

    
    def ImagenetAlexnetConfig(self, dropout, batchnorm):
        config = []
        
        config.append(self.ConvBlock(3, 64, 11, 4, 2, False, False))
        config.append(('Max2d', 3, 2, 0))
        
        config.append(self.ConvBlock(64, 192, 5, 1, 2, False, False))
        config.append(('Max2d', 3, 2, 0))
        
        config.append(self.ConvBlock(192, 384, 3, 1, 1, False, False))
        config.append(self.ConvBlock(384, 256, 3, 1, 1, False, False))
        config.append(self.ConvBlock(256, 256, 3, 1, 1, False, False))
        config.append(('Max2d', 3, 2, 0))
        
        config.append(('Flatten',False))
        
        config.append(self.FCBlock(256 * 6 * 6, 4096, False, False))
        config.append(self.FCBlock(4096, 4096, False, False))
        config.append(self.FCBlock(4096, 1000, option = 'FCOnly'))
               
        return config

    def ImagenetResnetConfig(self, num_layers, block_type = 'BasicBlock'):
        config = []
        
        num_fc = 1
        
        if num_layers == 18 :
            N = [2, 2, 2, 2]
        elif num_layers == 34 :
            N = [3, 4, 6, 3]
        elif num_layers == 50 :
            N = [3, 4, 6, 3]
        
        config.append(self.ConvBlock(3, 64, 7, 2, 3, True, False, bias = False))
        config.append(('Max2d', 3, 2, 1))
        
        if block_type == 'BasicBlock' :
            config.append(self.BasicBlock(64, 64, 1))
            for i in range(N[0]-1):
                config.append(self.BasicBlock(64, 64, 1))
        
            config.append(self.BasicBlock(64, 128, 2))
            for i in range(N[1]-1):
                config.append(self.BasicBlock(128, 128, 1))
                
            config.append(self.BasicBlock(128, 256, 2))
            for i in range(N[2]-1):
                config.append(self.BasicBlock(256, 256, 1))
                
            config.append(self.BasicBlock(256, 512, 2))
            for i in range(N[3]-1):
                config.append(self.BasicBlock(512, 512, 1))
            
            num_fc = 512
                
        elif block_type == 'Bottleneck' :
            config.append(self.Bottleneck(64, 64, 64 * 4, 1))
            for i in range(N[0]-1):
                config.append(self.Bottleneck(64 * 4, 64, 64 * 4, 1))
        
            config.append(self.Bottleneck(64 * 4, 128, 128 * 4, 2))
            for i in range(N[1]-1):
                config.append(self.Bottleneck(128 * 4, 128, 128 * 4, 1))
                
            config.append(self.Bottleneck(128 * 4, 256, 256 * 4, 2))
            for i in range(N[2]-1):
                config.append(self.Bottleneck(256 * 4, 256, 256 * 4, 1))
                
            config.append(self.Bottleneck(256 * 4, 512, 512 * 4, 2))
            for i in range(N[3]-1):
                config.append(self.Bottleneck(512 * 4, 512, 512 * 4, 1))
                
            num_fc = int(512 * 4)
        
        config.append(('Avg2d', 7, 7, 0))
        config.append(('Flatten',False))
        config.append(self.FCBlock(num_fc * 1 * 1, 1000, option = 'FCOnly'))
       
        self.imagenet_resnet = config
    

    def ImagenetDensenetConfig(self, num_layers):
        
        import math
        
        k = 32
        if num_layers == 121 :
            N = [6, 12, 24, 16]
        elif num_layers == 169 :
            N = [6, 12, 32, 32]
        elif num_layers == 201 :
            N = [6, 12, 48, 32]
        
        config = []
        reduction = 0.5
        
        config.append(self.ConvBlock(3, k * 2, 7, 2, 3, True, False, bias = False))
        config.append(('Max2d', 3, 2, 1))
        
        k1 = k * 2
        config.append(self.DenseBlock(k1, k1 + k*N[0], k, N[0]))
        
        k2 = math.floor((k1 + k * N[0]) * reduction)
        config.append(self.ConvBlock(k1 + k*N[0], k2, 1, 1, 0, bias = False, option = 'ConvOnly'))
        config.append(('Avg2d', 2, 2, 0))
       
        config.append(self.DenseBlock(k2, k2 + k*N[1], k, N[1]))
        
        k3 = math.floor((k2+ k*N[1]) * reduction)   
        config.append(self.ConvBlock(k2 + k*N[1], k3, 1, 1, 0, bias = False, option = 'ConvOnly'))
        config.append(('Avg2d', 2, 2, 0))
        
        config.append(self.DenseBlock(k3, k3 + k*N[2], k, N[2]))
        
        k4 = math.floor((k3+ k*N[2]) * reduction)  
        config.append(self.ConvBlock(k3 + k*N[2], k4, 1, 1, 0, bias = False, option = 'ConvOnly'))
        config.append(('Avg2d', 2, 2, 0))

        config.append(self.DenseBlock(k4, k4 + k*N[3], k, N[3]))
        
        config.append(('Avg2d', 7, 7, 0))
        
        config.append(('Flatten',False))
        
        config.append(self.FCBlock(k4 + k*N[3] * 1 * 1, 1000, option = 'FCOnly'))
        
        self.imagenet_densenet = config
        
####################################################################################################
#                                    Block generation part                                         #
####################################################################################################

    def GenNewFC(self, in_feature, out_feature, batchnorm, dropout, bias, option = 'Normal'):
        # Generate new fully connected block for the recasting
        config = self.FCBlock(in_feature,
                              out_feature,
                              batchnorm,
                              dropout,
                              bias,
                              option)
        block = self.Generator([config])
        return block[0]

    def GenNewConv(self, in_feature, out_feature, kernel_size, stride, padding, batchnorm, dropout, bias, option = 'Normal'):
        # Generate new convolution block for the recasting
        config = self.ConvBlock(in_feature, 
                                out_feature, 
                                kernel_size, 
                                stride, 
                                padding,
                                batchnorm,
                                dropout,
                                bias,
                                option)
        block = self.Generator([config])
        return block[0]

    def GenNewBasic(self, in_feature, out_feature, stride):
        # Generate new residual (basic) block for the recasting
        config = self.BasicBlock(in_feature, out_feature, stride)
        block = self.Generator([config])
        return block[0]

    def GenNewBottle(self, in_feature, mid_feature, out_feature, stride):
        # Generate new residual (bottleneck) block for the recasting
        config = self.Bottleneck(in_feature, mid_feature, out_feature, stride)
        block = self.Generator([config])
        return block[0]

    def GenNewDense(self, in_feature, out_feature, growth_rate):
        # Generate new dense block for the recasting
        num_conv = (out_feature - in_feature) / groth_rate
        config = self.DenseBlock(in_feature, out_feature, growth_rate, num_conv)
        block = self.Generator([config])
        return block[0]

    def GenNewBlock(self, info):
        block_type = info[0]
        config = info[1]
       
        # initialize configuration

        c = dict()
        c['in'] = 0             # in_features
        c['out'] = 0            # out_features
        c['k'] = 3              # kernel_size
        c['st'] = 1             # stride
        c['pad'] = 1            # padding
        c['bn'] = True          # batch normalization 
        c['dropout'] = False    # dropout 
        c['bias'] = False       # bias 
        c['g'] = 16             # growth_rate
        c['mid'] = 0            # mid_features - for the bottleneck block
        c['opt'] = 'Normal'     # option

        # Load configuration from given information

        if config[0] == 'FCBlock':
            c['in'] = config[1]
            c['out'] = config[2]
            c['bn'] = config[3]
            c['dropout'] = config[4]
            c['bias'] = config[5]
            c['opt'] = config[6]
        elif config[0] == 'ConvBlock':
            c['in'] = config[1]
            c['out'] = config[2]
            c['k'] = config[3]
            c['st'] = config[4]
            c['pad'] = config[5]
            c['bn'] = config[6]
            c['dropout'] = config[7]
            c['bias'] = config[8]
            c['opt'] = config[9]
        elif config[0] == 'ResidualBlock':
            c['in'] = config[1]
            c['out'] = config[2]
            c['st'] = config[3]
            # Set number of output features to mid features (output of 3x3 conv)
            if len(config) == 5 :
                c['out'] = config[4]
        elif config[0] == 'DenseBlock':
            c['in'] = config[1]
            c['out'] = config[2]
            c['g'] == config[3]


        # Generate and return new block

        if block_type == 'FCBlock':
            return self.GenNewFC(c['in'], c['out'], c['bn'], c['dropout'], c['bias'], c['opt'])
        elif block_type == 'ConvBlock':
            return self.GenNewConv(c['in'], c['out'], c['k'], c['st'], c['pad'], c['bn'], c['dropout'], c['bias'], c['opt'])
        elif block_type == 'ResidualBlock':
            # Corner case: Bottleneck -> Bottleneck (to rebuilt next block)
            if config[0] == 'ResidualBlock' and len(config) == 5 :
                b = self.GenNewBottle(c['in'], config[4], config[2], c['st'])
            else :
                # Normal case: any -> BasicBlock
                b = self.GenNewBasic(c['in'], c['out'], c['st'])
            return b
        elif block_type == 'DenseBlock':
            return self.GenNewDense(c['in'], c['out'], c['g'])
####################################################################################################
    def GetCifar10Cnn(self):
        return self.Generator(self.cifar10_cnn)
    def GetCifarResnet(self):
        return self.Generator(self.cifar_resnet)
    def GetCifarWrn(self):
        return self.Generator(self.cifar_wrn)
    def GetCifarDensenet(self):
        return self.Generator(self.cifar_densenet)
    def GetCifarVgg16(self):
        return self.Generator(self.cifar_vgg16)

    def GetImagenetAlexnet(self):
        return self.Generator(self.imagenet_alexnet)
    def GetImagenetVgg16(self):
        return self.Generator(self.imagenet_vgg16)
    def GetImagenetResnet(self):
        return self.Generator(self.imagenet_resnet)
    def GetImagenetDensenet(self):
        return self.Generator(self.imagenet_densenet)
    
