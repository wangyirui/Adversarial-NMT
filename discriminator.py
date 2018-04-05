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
        self.fixed_max_len = args.fixed_max_len
        self.use_cuda = use_cuda

        self.embed_src_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim)
        self.embed_trg_tokens = nn.Embedding(len(dst_dict), args.decoder_embed_dim)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 1,
                      out_channels = 20,
                      kernel_size = args.decoder_embed_dim,
                      stride = 1),
            nn.BatchNorm1d(20)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 20,
                      out_channels = 40,
                      kernel_size = 3,
                      stride = 1),
            nn.BatchNorm2d(40)
        )

        self.fc_in = nn.Linear(40*11*11, 20)
        self.fc_out= nn.Linear(20, 1)


    def forward(self, src_sentence, trg_sentence, pad_idx):
        padded_src_sentence = self.pad_sentences(src_sentence, pad_idx, self.fixed_max_len)
        padded_trg_sentence = self.pad_sentences(trg_sentence, pad_idx, self.fixed_max_len)
        padded_src_embed = self.embed_src_tokens(padded_src_sentence)
        padded_trg_embed = self.embed_trg_tokens(padded_trg_sentence)

        bsz = padded_src_embed.size(0)
        out1 = []
        out2 = []
        for i in range(bsz):
            x1 = self.conv1(padded_src_embed[i].unsqueeze(1))
            x1 = F.relu(x1)
            out1.append(x1)
            x2 = self.conv1(padded_trg_embed[i].unsqueeze(1))
            x2 = F.relu(x2)
            out2.append(x2)
        out1 = torch.stack(out1, dim=0).squeeze(-1)
        out1 = torch.stack([out1]*self.fixed_max_len, dim=1)
        out2 = torch.stack(out2, dim=0).squeeze(-1)
        out2 = torch.stack([out2]*self.fixed_max_len, dim=2)
        # use plus operation to merge two results
        out = torch.add(out1, out2) # (bsz, len, len, channel)

        out = out.permute(0, 3, 1, 2)
        out = F.max_pool2d(out, kernel_size=2, stride=2)

        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = out.view(out.size(0), -1)
        out = self.fc_in(out)
        out = self.fc_out(out)
        out = F.sigmoid(out)


        return out

    def pad_sentences(self, sentences, pad_idx, size=50):
        res = sentences[0].new(len(sentences), size).fill_(pad_idx)
        for i, v in enumerate(sentences):
            res[i][:len(v)] = v
        return res