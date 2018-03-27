import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import generator

class Discriminator(nn.Module):
    self, args, src_dict, dst_dict, use_cuda = True

    def __init__(self, args, src_dict, dst_dict, use_cuda = True):
        super(Discriminator, self).__init__()

        self.src_dict_size = len(src_dict)
        self.trg_dict_size = len(dst_dict)
        self.pad_idx = dst_dict.pad()
        self.use_cuda = use_cuda

        self.embed_src_tokens = generator.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
        self.embed_trg_tokens = generator.Embedding(len(dst_dict), args.encoder_embed_dim, dst_dict.pad())

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
        self.fc_out = nn.Linear(20,1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, src_sentence, trg_sentence, pad_idx):
        padded_src_sentence = self.pad_sentences(src_sentence, pad_idx)
        padded_trg_sentence = self.pad_sentences(trg_sentence, pad_idx)
        padded_src_embed = self.embed_src_tokens(padded_src_sentence)
        padded_trg_embed = self.embed_trg_tokens(padded_trg_sentence)

        # build 2D-image like tensor
        src_temp = torch.stack([padded_src_embed] * 50, dim=0)
        trg_temp = torch.stack([padded_trg_embed] * 50, dim=0)
        disc_input = torch.cat([src_temp, trg_temp], dim=1)
        out = self.conv1(input)
        out = self.conv2(out)
        out = out.view(out.size(0), 1280)
        out = F.relu(self.fc_in(out))
        out = self.fc_out(out)
        out = self.sigmoid(out)

        return out

    def pad_sentences(sentences, pad_idx, size=50):
        res = sentences[0].new(len(sentences), size).fill_(pad_idx)
        for i, v in enumerate(sentences):
            res[i][:, len(v)] = v
        return res