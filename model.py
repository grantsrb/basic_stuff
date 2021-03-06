import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
Simple, sequential convolutional net.
'''

class ConvNet(nn.Module):

    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, input_shape, output_dim, h_size=288, bnorm=True):
        super(ConvModel, self).__init__()

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.h_size = h_size
        self.bnorm = bnorm

        # Conv Layers
        self.convs = nn.ModuleList([])
        shape = input_shape.copy()

        ksize=3; stride=1; padding=1; out_depth=16
        self.convs.append(self.conv_block(input_shape[-3],out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=24
        self.convs.append(self.conv_block(in_depth,out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=32
        self.convs.append(self.conv_block(in_depth,out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=48
        self.convs.append(self.conv_block(in_depth,out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=64
        self.convs.append(self.conv_block(in_depth,out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)
        
        self.features = nn.Sequential(*self.convs)

        # FC Layers
        self.fcs = nn.ModuleList([])
        self.flat_size = int(np.prod(shape))
        print("Flat Features Size:", self.flat_size)
        self.fcs.append(self.dense_block(self.flat_size, self.h_size, bnorm=bnorm))
        self.fcs.append(self.dense_block(self.h_size, self.h_size, bnorm=bnorm))
        self.fcs.append(self.dense_block(self.h_size, self.output_dim, bnorm=False))
        self.classifier = nn.Sequential(*self.fcs)

    def get_new_shape(self, shape, depth, ksize, padding, stride):
        new_shape = [depth]
        for i in range(2):
            new_shape.append(self.new_size(shape[i+1], ksize, padding, stride))
        return new_shape
        
    def new_size(self, shape, ksize, padding, stride):
        return (shape - ksize + 2*padding)//stride + 1

    def forward(self, x):
        feats = self.features(x)
        feats = feats.view(feats.shape[0], -1)
        logits = self.classifier(feats)
        return logits

    def conv_block(self, chan_in, chan_out, ksize=3, stride=1, padding=1, activation="lerelu", max_pool=False, bnorm=True):
        block = []
        block.append(nn.Conv2d(chan_in, chan_out, ksize, stride=stride, padding=padding))
        if activation is not None: activation=activation.lower()
        if "relu" in activation:
            block.append(nn.ReLU())
        elif "elu" in activation:
            block.append(nn.ELU())
        elif "tanh" in activation:
            block.append(nn.Tanh())
        elif "lerelu" in activation:
            block.append(nn.LeakyReLU(negative_slope=.05))
        elif "selu" in activation:
            block.append(nn.SELU())
        if max_pool:
            block.append(nn.MaxPool2d(2, 2))
        if bnorm:
            block.append(nn.BatchNorm2d(chan_out))
        return nn.Sequential(*block)

    def dense_block(self, chan_in, chan_out, activation="relu", bnorm=True):
        block = []
        block.append(nn.Linear(chan_in, chan_out))
        if activation is not None: activation=activation.lower()
        if "relu" in activation:
            block.append(nn.ReLU())
        elif "elu" in activation:
            block.append(nn.ELU())
        elif "tanh" in activation:
            block.append(nn.Tanh())
        elif "lerelu" in activation:
            block.append(nn.LeakyReLU())
        elif "selu" in activation:
            block.append(nn.SELU())
        if bnorm:
            block.append(nn.BatchNorm1d(chan_out))
        return nn.Sequential(*block)

    def add_noise(self, x, mean=0.0, std=0.01):
        """
        Adds a normal distribution over the entries in a matrix.
        """
        means = torch.zeros(*x.size()).float()
        if mean != 0.0:
            means = means + mean
        noise = self.cuda_if(torch.normal(means,std=std))
        if type(x) == type(Variable()):
            noise = Variable(noise)
        return x+noise

    def multiply_noise(self, x, mean=1, std=0.01):
        """
        Multiplies a normal distribution over the entries in a matrix.
        """
        means = torch.zeros(*x.size()).float()
        if mean != 0:
            means = means + mean
        noise = self.cuda_if(torch.normal(means,std=std))
        if type(x) == type(Variable()):
            noise = Variable(noise)
        return x*noise

    def req_grads(self, yes):
        """
        An on-off switch for the requires_grad parameter for each internal Parameter.

        yes - Boolean denoting whether gradients should be calculated.
        """
        for param in self.parameters():
            param.requires_grad = yes

