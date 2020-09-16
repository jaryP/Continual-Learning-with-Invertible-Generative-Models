import torch
from torch import nn

from continual_ai.cl_strategies.prer.maf import IAF, MAF


class StudentTeacher(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, conditioning_size=None, activation='relu',
                 input_order='sequential', batch_norm=True):
        super().__init__()
        self.student = IAF(n_blocks, input_size, hidden_size, n_hidden, conditioning_size,
                           activation, input_order, batch_norm)
        self.teacher = MAF(n_blocks, input_size, hidden_size, n_hidden, conditioning_size,
                           activation, input_order, batch_norm)

    def forward(self, x, y=None):
        return self.teacher(x, y)

    def backward(self, x, y=None):
        return self.student(x, y)
