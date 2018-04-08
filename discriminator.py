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

        self.embed_src_tokens = Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
        self.embed_trg_tokens = Embedding(len(dst_dict), args.decoder_embed_dim, dst_dict.pad())

        self.conv2d = ConvlutionLayer(args, self.kernel_sizes, self.num_filters)
        self.highway = HighwayMLP(sum(self.num_filters), nn.functional.relu, nn.functional.sigmoid)
        self.fc = Linear(2 * sum(self.num_filters), 2)

    def forward(self, src_sentence, trg_sentence, pad_idx):
        padded_src_sentence = src_sentence
        padded_trg_sentence = trg_sentence
        # padded_src_sentence = self.pad_sentences(src_sentence, pad_idx, self.fixed_max_len)
        # padded_trg_sentence = self.pad_sentences(trg_sentence, pad_idx, self.fixed_max_len)
        padded_src_embed = self.embed_src_tokens(padded_src_sentence)
        padded_trg_embed = self.embed_trg_tokens(padded_trg_sentence)
        padded_src_embed = padded_src_embed.unsqueeze(1)
        padded_trg_embed = padded_trg_embed.unsqueeze(1)

        # padded_src_input = torch.stack([padded_src_embed]*self.fixed_max_len, dim=2)
        # padded_trg_input = torch.stack([padded_trg_embed]*self.fixed_max_len, dim=3)
        # padded_input = torch.cat([padded_src_input, padded_trg_input], dim=1)

        src_conv_out = self.conv2d(padded_src_embed)
        trg_conv_out = self.conv2d(padded_trg_embed)
        batch_size = padded_src_embed.size(0)
        src_out_flat = src_conv_out.view(batch_size, -1)
        trg_out_flat = trg_conv_out.view(batch_size, -1)
        src_highway_out = self.highway(src_out_flat)
        trg_highway_out = self.highway(trg_out_flat)
        concat_out = torch.cat([src_highway_out, trg_highway_out], dim=1)
        scores = self.fc(concat_out)
        scores = F.softmax(scores, dim=1)

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
                Conv2d(in_channels=1 if i == 0 else num_filters[i - 1],
                       out_channels=num_filters[i],
                       kernel_size=self.kernel_sizes[i],
                       stride=1,
                       padding=0),
                nn.BatchNorm2d(num_filters[i]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(self.max_len - kernel_sizes[i] + 1, 1))
            )
            for i in range(len(self.kernel_sizes))
        ])

    def forward(self, input):
        x = input
        out = []
        for i, conv2d in enumerate(self.conv2d):
            x = conv2d(x)
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
            nn.init.xavier_uniform(param.data)
        elif 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    nn.init.xavier_uniform(m.weight.data)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m