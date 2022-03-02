import os
import torch
from torch.autograd import Variable


def to_numpy(var, use_cuda=False):
    return var.cpu().data.numpy() if use_cuda else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, use_cuda=False):
    FLOAT = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(FLOAT)

def soft_update(target, source, oft_replace):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - oft_replace) + param.data * oft_replace
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)