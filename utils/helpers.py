# -*- coding: utf-8 -*-
#
# References:
# - https://github.com/LiyingCV/Long-Range-Grouping-Transformer

import os
import numpy as np
import torch

def reduce_value(value):
    world_size = int(os.environ['WORLD_SIZE'])
    if world_size < 2:  # for single-GPU training
        return value
    with torch.no_grad():
        torch.distributed.all_reduce(value)  
        value /= world_size  
    return value

def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x

def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or \
       type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

