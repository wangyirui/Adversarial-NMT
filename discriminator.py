import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import generator

class Discriminator(nn.Module):
    def __init__(self, args, src_dict, dst_dict, use_cuda = True):
        super(Discriminator, self).__init__()

        self.src_dict_size = len(src_dict)
        self.trg_dict_size = len(dst_dict)
        self.pad_idx = dst_dict.pad()
        self.pad_dim = args.pad_dim
        self.use_cuda = use_cuda

        self.embed_src_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim)
        self.embed_trg_tokens = nn.Embedding(len(dst_dict), args.decoder_embed_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = args.encoder_embed_dim * 2,
                      out_channels = 512,
                      kernel_size = 3,
                      stride = 1),
            # nn.BatchNorm2d(512),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 512,
                      out_channels = 256,
                      kernel_size = 3,
                      stride = 1),
            # nn.BatchNorm2d(256),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2))
        self.fc_in = nn.Linear(256*121, 20)
        self.fc_out= nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, src_sentence, trg_sentence, pad_idx):
        padded_src_sentence = self.pad_sentences(src_sentence, pad_idx, self.pad_dim)
        padded_trg_sentence = self.pad_sentences(trg_sentence, pad_idx, self.pad_dim)
        padded_src_embed = self.embed_src_tokens(padded_src_sentence)
        padded_trg_embed = self.embed_trg_tokens(padded_trg_sentence)

        # build 2D-image like tensor
        src_temp = torch.stack([padded_src_embed] * self.pad_dim, dim=1)
        trg_temp = torch.stack([padded_trg_embed] * self.pad_dim, dim=1)
        trg_temp = trg_temp.transpose(1,2)
        input = torch.cat([src_temp, trg_temp], dim=-1)
        # Convolution require shape (N,C,H,W)
        input = input.permute(0, 3, 1, 2)

        out = self.conv1(input)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc_in(out)
        out = self.fc_out(out)
        out = self.sigmoid(out)

        return out

    def pad_sentences(self, sentences, pad_idx, size=50):
        res = sentences[0].new(len(sentences), size).fill_(pad_idx)
        for i, v in enumerate(sentences):
            try:
                res[i][:len(v)] = v
            except:
                print("Error!")
        return res