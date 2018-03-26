import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 

class Discriminator(nn.Module):
    self, args, src_dict, dst_dict, use_cuda = True

    def __init__(self, args, src_dict, dst_dict, use_cuda = True):
        super(Discriminator, self).__init__()

        self.src_dict_size = len(src_dict)
        self.trg_dict_size = len(dst_dict)
        self.pad_idx = dst_dict.pad()
        self.use_cuda = use_cuda

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = self.embed_dim * 2,
                      out_channels = 64,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 64,
                      out_channels = 20,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2))

        self.fc_in = nn.Linear(1280, 20)
        self.fc_out = nn.Linear(20,2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):

        out = self.conv1(input)
        out = self.conv2(out)
        out = out.view(out.size(0), 1280)
        out = F.relu(self.fc_in(out))
        out = self.fc_out(out)
        out = self.sigmoid(out)

        return out

