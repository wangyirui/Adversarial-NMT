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

        self.kernel_sizes = [i for i in range(1, args.fixed_max_len, 4)]
        self.num_filters = [100 + i * 10 for i in range(1, args.fixed_max_len, 4)]
        # self.kernel_sizes = [1,5,10,15]
        # self.num_filters = [100, 150, 200, 250]

        self.embed_src_tokens = Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
        self.embed_trg_tokens = Embedding(len(dst_dict), args.decoder_embed_dim, dst_dict.pad())

        self.conv2d = ConvlutionLayer(args, self.kernel_sizes, self.num_filters)
        self.highway = HighwayMLP(sum(self.num_filters), nn.functional.relu, nn.functional.sigmoid)
        self.dropout_in = nn.Dropout2d(p=0.2)
        self.dropout_out = nn.Dropout2d(p=0.5)
        self.fc = Linear(2 * sum(self.num_filters), 1)

    def forward(self, src_sentence, trg_sentence, pad_idx):
        padded_src_embed = self.embed_src_tokens(src_sentence)
        padded_src_embed = self.dropout_in(padded_src_embed)
        padded_src_embed = padded_src_embed.unsqueeze(1)

        padded_trg_embed = self.embed_src_tokens(trg_sentence)
        padded_trg_embed = self.dropout_in(padded_trg_embed)
        padded_trg_embed = padded_trg_embed.unsqueeze(1)

        batch_size = padded_src_embed.size(0)
        src_conv_out = self.conv2d(padded_src_embed).view(batch_size, -1)
        trg_conv_out = self.conv2d(padded_trg_embed).view(batch_size, -1)

        src_highway_out = self.dropout_out(self.highway(src_conv_out))
        trg_highway_out = self.dropout_out(self.highway(trg_conv_out))

        concat_out = torch.cat([src_highway_out, trg_highway_out], dim=1)
        scores = F.sigmoid(self.fc(concat_out))

        return scores

    def pad_sentences(self, sentences, pad_idx, size=50):
        res = sentences[0].new(len(sentences), size).fill_(pad_idx)
        for i, v in enumerate(sentences):
            res[i][:len(v)] = v
        return res


class ConvlutionLayer(nn.Module):
    def __init__(self, args, kernel_sizes, num_filters):
        super(ConvlutionLayer, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.max_len = args.fixed_max_len

        self.conv2d = nn.ModuleList([
            nn.Sequential(
                Conv2d(in_channels=1,
                       out_channels=num_filters[i],
                       kernel_size=(kernel_sizes[i], args.decoder_embed_dim),
                       stride=1,
                       padding=0),
                nn.BatchNorm2d(num_filters[i]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(self.max_len - kernel_sizes[i] + 1, 1))
            )
            for i in range(len(self.kernel_sizes))
        ])

    def forward(self, input):
        out = []
        for i, conv2d in enumerate(self.conv2d):
            x = conv2d(input)
            x = x.permute(0, 2, 3, 1)
            out.append(x)

        out = torch.cat(out, dim=3)
        return out

class HighwayMLP(nn.Module):
    def __init__(self, input_size, activation_function=nn.functional.relu, gate_activation=nn.functional.softmax):
        super(HighwayMLP, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = Linear(input_size, input_size)
        self.gate_layer = Linear(input_size, input_size)

    def forward(self, x):
        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))
        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)
        out = torch.add(multiplyed_gate_and_normal, multiplyed_gate_and_input)

        return out


def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            # param.data.uniform_(-0.1, 0.1)
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    nn.init.kaiming_uniform_(m.weight.data)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m