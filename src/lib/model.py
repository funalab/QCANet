# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_L2(nn.Module):
    def __init__(
            self,
            ndim=3,
            n_class=2,
            init_channel=2,
            kernel_size=3,
            pool_size=2,
            ap_factor=2,
            gpu=-1,
            loss_func='nn.CrossEntropyLoss'
        ):
        super(Model_L2, self).__init__()
        self.gpu = gpu
        self.pool_size = pool_size
        self.phase = 'train'
        self.loss_func = eval(loss_func)()

        self.c0=nn.Conv3d(1, init_channel, kernel_size, 1, int(kernel_size/2))
        self.c1=nn.Conv3d(init_channel, int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))

        self.c2=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))
        self.c3=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))

        self.c4=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))
        self.c5=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))

        self.dc0=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), self.pool_size, self.pool_size, 0)
        self.dc1=nn.Conv3d(int(init_channel * (ap_factor ** 2) + init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))
        self.dc2=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))

        self.dc3=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), self.pool_size, self.pool_size, 0)
        self.dc4=nn.Conv3d(int(init_channel * (ap_factor ** 1) + init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))
        self.dc5=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))

        self.dc6=nn.Conv3d(int(init_channel * (ap_factor ** 1)), n_class, 1, 1)

        self.bnc0=nn.BatchNorm3d(init_channel)
        self.bnc1=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))

        self.bnc2=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))
        self.bnc3=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))

        self.bnc4=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bnc5=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))

        self.bndc1=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bndc2=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bndc4=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))
        self.bndc5=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))

        self.pool = nn.MaxPool3d(pool_size, pool_size)

    def _calc(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        syn0 = F.relu(self.bnc1(self.c1(e0)))
        del e0
        e1 = self.pool(syn0)
        e2 = F.relu(self.bnc2(self.c2(e1)))
        syn1 = F.relu(self.bnc3(self.c3(e2)))
        del e1, e2
        e3 = self.pool(syn1)
        e4 = F.relu(self.bnc4(self.c4(e3)))
        e5 = F.relu(self.bnc5(self.c5(e4)))
        del e3, e4
        d0 = torch.cat([self.dc0(e5), syn1], dim=1)
        del e5, syn1
        d1 = F.relu(self.bndc1(self.dc1(d0)))
        d2 = F.relu(self.bndc2(self.dc2(d1)))
        del d0, d1
        d3 = torch.cat([self.dc3(d2), syn0], dim=1)
        del d2, syn0
        d4 = F.relu(self.bndc4(self.dc4(d3)))
        d5 = F.relu(self.bndc5(self.dc5(d4)))
        del d3, d4
        d6 = self.dc6(d5)
        del d5
        return d6

    def forward(self, x, t=None, seg=True):
        h = self._calc(x)
        if seg:
            pred = F.softmax(h, dim=1)
            del h
            return pred.data
        else:
            loss = self.loss_func(h, t.long())
            pred = F.softmax(h, dim=1)
            del h
            return loss, pred.data


class Model_L3(nn.Module):
    def __init__(
            self,
            ndim=3,
            n_class=2,
            init_channel=2,
            kernel_size=3,
            pool_size=2,
            ap_factor=2,
            gpu=-1,
            loss_func='nn.CrossEntropyLoss'
        ):
        super(Model_L3, self).__init__()
        self.gpu = gpu
        self.pool_size = pool_size
        self.phase = 'train'
        self.loss_func = eval(loss_func)()

        self.c0=nn.Conv3d(1, init_channel, kernel_size, 1, int(kernel_size/2))
        self.c1=nn.Conv3d(init_channel, int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))

        self.c2=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))
        self.c3=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))

        self.c4=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))
        self.c5=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))

        self.c6=nn.Conv3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))
        self.c7=nn.Conv3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2))

        self.dc0=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 4)), self.pool_size, self.pool_size, 0)
        self.dc1=nn.Conv3d(int(init_channel * (ap_factor ** 3) + init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))
        self.dc2=nn.Conv3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))

        self.dc3=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), self.pool_size, self.pool_size, 0)
        self.dc4=nn.Conv3d(int(init_channel * (ap_factor ** 2) + init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))
        self.dc5=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))

        self.dc6=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), self.pool_size, self.pool_size, 0)
        self.dc7=nn.Conv3d(int(init_channel * (ap_factor ** 1) + init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))
        self.dc8=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))

        self.dc9=nn.Conv3d(int(init_channel * (ap_factor ** 1)), n_class, 1, 1)

        self.bnc0=nn.BatchNorm3d(init_channel)
        self.bnc1=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))

        self.bnc2=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))
        self.bnc3=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))

        self.bnc4=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bnc5=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))

        self.bnc6=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))
        self.bnc7=nn.BatchNorm3d(int(init_channel * (ap_factor ** 4)))

        self.bndc1=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))
        self.bndc2=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))
        self.bndc4=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bndc5=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bndc7=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))
        self.bndc8=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))

        self.pool = nn.MaxPool3d(pool_size, pool_size)

    def _calc(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        syn0 = F.relu(self.bnc1(self.c1(e0)))
        del e0
        e1 = self.pool(syn0)
        e2 = F.relu(self.bnc2(self.c2(e1)))
        syn1 = F.relu(self.bnc3(self.c3(e2)))
        del e1, e2
        e3 = self.pool(syn1)
        e4 = F.relu(self.bnc4(self.c4(e3)))
        syn2 = F.relu(self.bnc5(self.c5(e4)))
        del e3, e4
        e5 = self.pool(syn2)
        e6 = F.relu(self.bnc6(self.c6(e5)))
        e7 = F.relu(self.bnc7(self.c7(e6)))
        del e5, e6
        d0 = torch.cat([self.dc0(e7), syn2], dim=1)
        del e7, syn2
        d1 = F.relu(self.bndc1(self.dc1(d0)))
        d2 = F.relu(self.bndc2(self.dc2(d1)))
        del d0, d1
        d3 = torch.cat([self.dc3(d2), syn1], dim=1)
        del d2, syn1
        d4 = F.relu(self.bndc4(self.dc4(d3)))
        d5 = F.relu(self.bndc5(self.dc5(d4)))
        del d3, d4
        d6 = torch.cat([self.dc6(d5), syn0], dim=1)
        del d5, syn0
        d7 = F.relu(self.bndc7(self.dc7(d6)))
        d8 = F.relu(self.bndc8(self.dc8(d7)))
        del d6, d7
        d9 = self.dc9(d8)
        del d8
        return d9

    def forward(self, x, t=None, seg=True):
        h = self._calc(x)
        if seg:
            pred = F.softmax(h, dim=1)
            del h
            return pred.data
        else:
            loss = self.loss_func(h, t.long())
            pred = F.softmax(h, dim=1)
            del h
            return loss, pred.data


class Model_L4(nn.Module):
    def __init__(
            self,
            ndim=3,
            n_class=2,
            init_channel=2,
            kernel_size=3,
            pool_size=2,
            ap_factor=2,
            gpu=-1,
            loss_func='nn.CrossEntropyLoss'
        ):
        super(Model_L4, self).__init__()
        self.gpu = gpu
        self.pool_size = pool_size
        self.phase = 'train'
        self.loss_func = eval(loss_func)()

        self.c0=nn.Conv3d(1, init_channel, kernel_size, 1, int(kernel_size/2))
        self.c1=nn.Conv3d(init_channel, int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))
        self.c2=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))
        self.c3=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))

        self.c4=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))
        self.c5=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))

        self.c6=nn.Conv3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))
        self.c7=nn.Conv3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2))

        self.c8=nn.Conv3d(int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2))
        self.c9=nn.Conv3d(int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 5)), kernel_size, 1, int(kernel_size/2))

        self.dc0=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 5)), int(init_channel * (ap_factor ** 5)), self.pool_size, self.pool_size, 0)
        self.dc1=nn.Conv3d(int(init_channel * (ap_factor ** 4) + init_channel * (ap_factor ** 5)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2))
        self.dc2=nn.Conv3d(int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2))

        self.dc3=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 4)), self.pool_size, self.pool_size, 0)
        self.dc4=nn.Conv3d(int(init_channel * (ap_factor ** 3) + init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))
        self.dc5=nn.Conv3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2))

        self.dc6=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), self.pool_size, self.pool_size, 0)
        self.dc7=nn.Conv3d(int(init_channel * (ap_factor ** 2) + init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))
        self.dc8=nn.Conv3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2))

        self.dc9=nn.ConvTranspose3d(int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), self.pool_size, self.pool_size, 0)
        self.dc10=nn.Conv3d(int(init_channel * (ap_factor ** 1) + init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))
        self.dc11=nn.Conv3d(int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2))

        self.dc12=nn.Conv3d(int(init_channel * (ap_factor ** 1)), n_class, 1, 1)

        self.bnc0=nn.BatchNorm3d(init_channel)
        self.bnc1=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))

        self.bnc2=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))
        self.bnc3=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))

        self.bnc4=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bnc5=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))

        self.bnc6=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))
        self.bnc7=nn.BatchNorm3d(int(init_channel * (ap_factor ** 4)))

        self.bnc8=nn.BatchNorm3d(int(init_channel * (ap_factor ** 4)))
        self.bnc9=nn.BatchNorm3d(int(init_channel * (ap_factor ** 5)))
        self.bndc1=nn.BatchNorm3d(int(init_channel * (ap_factor ** 4)))
        self.bndc2=nn.BatchNorm3d(int(init_channel * (ap_factor ** 4)))
        self.bndc4=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))
        self.bndc5=nn.BatchNorm3d(int(init_channel * (ap_factor ** 3)))
        self.bndc7=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bndc8=nn.BatchNorm3d(int(init_channel * (ap_factor ** 2)))
        self.bndc10=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))
        self.bndc11=nn.BatchNorm3d(int(init_channel * (ap_factor ** 1)))

        self.pool = nn.MaxPool3d(pool_size, pool_size)

    def _calc(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        syn0 = F.relu(self.bnc1(self.c1(e0)))
        del e0
        e1 = self.pool(syn0)
        e2 = F.relu(self.bnc2(self.c2(e1)))
        syn1 = F.relu(self.bnc3(self.c3(e2)))
        del e1, e2
        e3 = self.pool(syn1)
        e4 = F.relu(self.bnc4(self.c4(e3)))
        syn2 = F.relu(self.bnc5(self.c5(e4)))
        del e3, e4
        e5 = self.pool(syn2)
        e6 = F.relu(self.bnc6(self.c6(e5)))
        syn3 = F.relu(self.bnc7(self.c7(e6)))
        del e5, e6
        e7 = self.pool(syn3)
        e8 = F.relu(self.bnc8(self.c8(e7)))
        e9 = F.relu(self.bnc9(self.c9(e8)))
        del e7, e8
        d0 = torch.cat([self.dc0(e9), syn3], dim=1)
        del e9, syn3
        d1 = F.relu(self.bndc1(self.dc1(d0)))
        d2 = F.relu(self.bndc2(self.dc2(d1)))
        del d0, d1
        d3 = torch.cat([self.dc3(d2), syn2], dim=1)
        del d2, syn2
        d4 = F.relu(self.bndc4(self.dc4(d3)))
        d5 = F.relu(self.bndc5(self.dc5(d4)))
        del d3, d4
        d6 = torch.cat([self.dc6(d5), syn1], dim=1)
        del d5, syn1
        d7 = F.relu(self.bndc7(self.dc7(d6)))
        d8 = F.relu(self.bndc8(self.dc8(d7)))
        del d6, d7
        d9 = torch.cat([self.dc9(d8), syn0], dim=1)
        del d8, syn0
        d10 = F.relu(self.bndc10(self.dc10(d9)))
        d11 = F.relu(self.bndc11(self.dc11(d10)))
        del d9, d10

        d12 = self.dc12(d11)
        del d11
        return d12

    def forward(self, x, t=None, seg=True):
        h = self._calc(x)
        if seg:
            pred = F.softmax(h, dim=1)
            del h
            return pred.data
        else:
            loss = self.loss_func(h, t.long())
            pred = F.softmax(h, dim=1)
            del h
            return loss, pred.data
