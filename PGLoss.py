import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math


class PGLoss(torch.nn.Module):
    def __init__(self):
        super(PGLoss, self).__init__()
        self.loss = 0.0

    def forward(self, logprobs, label, reward):
        bsz, seqlen, _ = logprobs.size()

        for i in range(bsz):
            trg_label = label[i]
            trg_log_prob = logprobs[i][trg_label]
            trg_log_prob *= reward[i]
            self.loss += trg_log_prob

        return self.loss / bsz