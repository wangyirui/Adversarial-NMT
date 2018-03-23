import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 

class Discriminator(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, word_emb_size, 
        src_vocab, trg_vocab, use_cuda=False):
        super(Discriminator, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.word_emb_size = word_emb_size
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.use_cuda = use_cuda
        
        self.embedding1 = nn.Embedding(src_vocab_size, word_emb_size)
        self.embedding2 = nn.Embedding(trg_vocab_size, word_emb_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = word_emb_size*2,
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

        self.mlp = nn.Linear(1280, 20)
        self.ll = nn.Linear(20,2)
        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax() 

    def forward(self, src_batch, trg_batch, is_train=False):
        # src_batch : (src_seq_len, batch_size)
        src_embedded = self.embedding1(src_batch) #(s,b,e)
        trg_embedded = self.embedding2(trg_batch)

        # 
        src_padded = np.zeros((35,src_batch.size(-1), self.word_emb_size))
        trg_padded = np.zeros((35, src_batch.size(-1), self.word_emb_size))
        src_padded[:src_embedded.size(0), :src_embedded.size(1), :src_embedded.size(2)] = src_embedded.data
        trg_padded[:trg_embedded.size(0), :trg_embedded.size(1), :trg_embedded.size(2)] = trg_embedded.data

        src_padded = np.transpose(np.expand_dims(src_padded,2),(1,3,2,0)) #(b,c,h,w)
        src_padded = np.concatenate([src_padded]*35, axis=2)
        trg_padded = np.transpose(np.expand_dims(trg_padded,2),(1,3,0,2)) 
        trg_padded = np.concatenate([trg_padded]*35, axis=3)

        input = Variable(torch.from_numpy(np.concatenate([src_padded,trg_padded],axis=1)).float())

        if self.use_cuda:
            input = input.cuda()

        out = self.conv1(input)
        out = self.conv2(out)
        out = out.view(out.size(0), 1280)
        out = F.relu(self.mlp(out))
        out = self.ll(out)
        out = self.sigmoid(out)
        #out = self.softmax(out)

        return out

