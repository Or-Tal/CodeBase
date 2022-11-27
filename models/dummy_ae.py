# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.
# Author: Or Tal.
import logging

import torch
import torch.nn as nn
from munch import DefaultMunch
import torch.nn.functional as F


class EncoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, act=nn.ReLU):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=(kernel_size - 1)//2)
        self.c2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=2, padding=(kernel_size - 1)//2)
        self.act = act()

    def forward(self, x):
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, act=nn.ReLU):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.c2 = nn.ConvTranspose1d(out_ch, out_ch, kernel_size + 1, stride=2, padding=(kernel_size - 1) // 2)
        self.act = act()

    def forward(self, x):
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        return x


class DummyAE(nn.Module):

    @staticmethod
    def construct_dummy_from_dict(kwargs: dict):
        args = DefaultMunch.fromDict(kwargs)
        return DummyAE(args)

    def __init__(self, model_args):
        super().__init__()
        out_ch = model_args.initial_hidden_channels
        in_ch = model_args.in_channels

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for kernel_size in model_args.kernels:
            self.encoders.append(EncoderBlock(in_ch, out_ch, kernel_size))
            self.decoders.insert(0, DecoderBlock(out_ch, in_ch, kernel_size))
            in_ch = out_ch
            out_ch *= 2

    def pad_n_sum(self, x, skip):
        if x.shape[-1] < skip.shape[-1]:
            return x + skip[..., :x.shape[-1]]
        elif x.shape[-1] > skip.shape[-1]:
            skip = F.pad(skip, (0, x.shape[-1] - skip.shape[-1]), 'constant', 0)
        return x + skip

    def forward(self, x):
        skips = []
        for i, en in enumerate(self.encoders):
            if i != 0:
                skips.append(x)
            x = en(x)
        for i, de in enumerate(self.decoders):
            if i != 0:
                x = x + skips.pop()
            x = de(x)
        return x



