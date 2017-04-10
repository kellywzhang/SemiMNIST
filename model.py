import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import to_gpu

class ConvNet(nn.Module):
    def __init__(self, layers, noise):
        super(ConvNet, self).__init__()
        self.layer_modules, self.batch_norms, self.act_modules = self.parse_layers(layers)
        # noise is list of values for standard deviation of noise wanted for each layer of convolutional portion of model
        self.noise = noise

        # add modules to convnet object
        for i in range(len(self.layer_modules)):
            self.add_module("layer"+str(i), self.layer_modules[i])
            if self.batch_norms[i] is not None:
                self.add_module("bn"+str(i), self.batch_norms[i])

        self.fc1 = nn.Linear(self.flat_count, 100)
        self.fc1bn = nn.BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True)
        self.fc2 = nn.Linear(100, 10)
        self.fc2bn = nn.BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True)

        self.init_weights()

    def forward(self, x, corrupted, cuda=False):
        # keep track of "lateral" outputs for ladder network
        laterals = []
        # Loop through all convlutional/pooling layers
        for m in range(len(self.layer_modules)):
            # add noise
            if corrupted and self.noise[m] > 0:
                x = x + to_gpu(Variable(torch.normal(means=torch.zeros(x.size()), std=torch.zeros(x.size()).fill_(self.noise[m]))), cuda)
            # convolutional or pooling layer
            x = self.layer_modules[m](x)
            # batch normalization
            if self.batch_norms[m] is not None:
                x = self.batch_norms[m](x)
            laterals.append(x)
            # activation
            if self.act_modules[m] is not None:
                x = self.act_modules[m](x)
        # Flatten for fully connect layers
        x = x.view(-1, self.flat_count)
        x = self.fc1(x)
        x = self.fc1bn(x)
        laterals.append(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.fc2bn(x)
        laterals.append(x)
        return x, laterals

    def init_weights(self):
        # initializes all weights to uniform distribution between -0.1 and 0.1
        # all biases initialized to 0
        init_range = 0.1
        for mod in self.layer_modules:
            try:
                mod.weight.data.uniform_(-init_range, init_range)
                mod.bias.data.fill_(0)
            except:
                pass
        self.fc1.weight.data.uniform_(-init_range, init_range)
        self.fc2.weight.data.uniform_(-init_range, init_range)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)

    def parse_layers(self, layers):
        # Takes layers argument and creates corresponding model
        # layer types: convf (padding), convv (no padding), maxpool
        # arguments: layer-type, filters, kernel, batch-normalization bool, activation

        # e.g. layers = [["convf", 32, 5, True, "lrelu"], ["maxpool", 0, 2, True, "lrelu"],
        #         ["convf", 64, 3, True, "lrelu"], ["convf", 64, 3, True, "lrelu"],
        #         ["maxpool", 0, 2, False, ""], ["convf", 128, 3, True, "lrelu"]]

        self.layers = layers
        channels = [1]
        layer_modules = {}
        act_modules = {}
        bn_modules = {}
        input_dim = [28]
        filter_count = []

        for l, layer in enumerate(layers):
            layer_type, filters, kernel, bn, act = layer

            # Find layer type and create convolutional / pooling objects
            if layer_type in ["convf", "convv"]:
                if layer_type == "convv":
                    mod = nn.Conv2d(channels[-1], filters, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True)
                    input_dim.append(input_dim[-1]-kernel+1)
                else:
                    mod = nn.Conv2d(channels[-1], filters, kernel, stride=1, padding=kernel-1, dilation=1, groups=1, bias=True)
                    input_dim.append(input_dim[-1]-kernel+1+2*(kernel-1))
                channels.append(filters)
                filter_count.append(filters)
            elif layer_type in ["maxpool"]:
                mod = nn.MaxPool2d(kernel, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
                input_dim.append(input_dim[-1]/kernel)
            else:
                raise ValueError("Invalid layer type")
            layer_modules[l] = mod

            # Batcn normalization module if True
            if bn:
                bnmod = nn.BatchNorm2d(filter_count[-1], eps=1e-05, momentum=0.1, affine=True)
            else:
                bnmod = None
            bn_modules[l] = bnmod

            # Activation function
            if act == "relu":
                actmod = F.relu
            elif act == "lrelu":
                actmod = F.leaky_relu
            else:
                actmod = None
            act_modules[l] = actmod

        # Set the total number of parameters after convolution layers (for fc layers)
        self.flat_count = int(filter_count[-1]*input_dim[-1]*input_dim[-1])
        return layer_modules, bn_modules, act_modules
