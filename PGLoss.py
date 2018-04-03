import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math


class PGLoss(torch.nn.Module):
    def __init__(self, ignore_index=None, size_average=False, reduce=True):
        super(PGLoss, self).__init__()
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, logprobs, label, reward, use_cuda):
        bsz, seqlen, _ = logprobs.size()
        loss = 0
        for i in range(bsz):
            trg_label = label[i,:]
            row_idx = torch.LongTensor(range(seqlen))
            if use_cuda:
                row_idx = row_idx.cuda()
            if self.ignore_index != None:
                logprobs[:, :, self.ignore_index] = 0
            trg_log_prob = logprobs[i, :, :][row_idx, trg_label]
            trg_log_prob *= reward[i]

            loss += -torch.sum(trg_log_prob)

        if self.size_average:
            loss /= bsz


        # for i in range(seqlen):
        #     # TODO: should h be detached from graph (.detach())?
        #     for j in range(bsz):
        #         loss += -logprobs[j][label.data[j][i]] * reward[j]  # log(P(y_t|Y_1:Y_{t-1})) * Q
        #
        return loss
