import torch
import torch.nn as nn
import torch.nn.functional as F 


class Discriminator(nn.Module):
    def __init__(self, args, src_dict, dst_dict, use_cuda = True):
        super(Discriminator, self).__init__()

        self.src_dict_size = len(src_dict)
        self.trg_dict_size = len(dst_dict)
        self.pad_idx = dst_dict.pad()
        self.fixed_max_len = args.fixed_max_len
        self.use_cuda = use_cuda


        self.embed_src_tokens = Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
        self.embed_trg_tokens = Embedding(len(dst_dict), args.decoder_embed_dim, dst_dict.pad())


        self.conv1 = nn.Sequential(
            Conv2d(in_channels=1,
                   out_channels=32,
                   kernel_size=(3, 1000),
                   stride=(1, 1000),
                   padding=1)
        )

        self.conv2 = nn.Sequential(
            Conv2d(in_channels=64,
                   out_channels=128,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            Conv2d(in_channels=128,
                   out_channels=256,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            Linear(256 * 12 * 12, 20),
            #nn.ReLU(),
            # Linear(20, 20),
            # nn.ReLU(),
            nn.Dropout(),
            Linear(20, 2),
        )

    def forward(self, src_sentence, trg_sentence):
        batch_size = src_sentence.size(0)

        src_out = self.embed_src_tokens(src_sentence)
        src_out = src_out.unsqueeze(1)
        trg_out = self.embed_src_tokens(trg_sentence)
        trg_out = trg_out.unsqueeze(1)

        out1 = self.conv1(src_out)
        out1 = out1.squeeze(3)
        out2 = self.conv1(trg_out)
        out2 = out2.squeeze(3)

        out1 = torch.stack([out1] * out2.size(2), dim=3)
        out2 = torch.stack([out2] * out1.size(2), dim=2)
        out = torch.cat([out1, out2], dim=1)

        out = self.conv2(out)
        out = self.conv3(out)
        out = out.contiguous().view(out.size(0), -1)

        out = self.classifier(out)

        return out

def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            # param.data.uniform_(-0.1, 0.1)
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
    return m

def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            # param.data.uniform_(-0.1, 0.1)
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    nn.init.kaiming_uniform_(m.weight.data)
    if bias:
        nn.init.constant_(m.bias.data, 0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m