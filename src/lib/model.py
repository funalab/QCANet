# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, Variable
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from src.lib.loss import softmax_dice_loss

class Model_L2(Chain):
    def __init__(
            self,
            ndim=3,
            n_class=2,
            init_channel=2,
            kernel_size=3,
            pool_size=2,
            ap_factor=2,
            gpu=-1,
            class_weight=np.array([1, 1]).astype(np.float32),
            loss_func='F.softmax_cross_entropy'
        ):
        self.gpu = gpu
        self.pool_size = pool_size
        if gpu >= 0:
            self.class_weight = cuda.to_gpu(np.array(class_weight).astype(np.float32))
        else:
            self.class_weight = np.array(class_weight).astype(np.float32)
        self.train = True
        self.loss_func = loss_func
        initializer = chainer.initializers.HeNormal()
        super(Model_L2, self).__init__(

            c0=L.ConvolutionND(ndim, 1, init_channel, kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            c1=L.ConvolutionND(ndim, init_channel, int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            c2=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            c3=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            c4=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            c5=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            dc0=L.DeconvolutionND(ndim, int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), self.pool_size, self.pool_size, 0, initialW=initializer, initial_bias=None),
            dc1=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2) + init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            dc2=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            dc3=L.DeconvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), self.pool_size, self.pool_size, 0, initialW=initializer, initial_bias=None),
            dc4=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1) + init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            dc5=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            dc6=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), n_class, 1, 1, initialW=initializer, initial_bias=None),

            bnc0=L.BatchNormalization(init_channel),
            bnc1=L.BatchNormalization(int(init_channel * (ap_factor ** 1))),

            bnc2=L.BatchNormalization(int(init_channel * (ap_factor ** 1))),
            bnc3=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),

            bnc4=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),
            bnc5=L.BatchNormalization(int(init_channel * (ap_factor ** 3))),

            bndc1=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),
            bndc2=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),
            bndc4=L.BatchNormalization(int(init_channel * (ap_factor ** 1))),
            bndc5=L.BatchNormalization(int(init_channel * (ap_factor ** 1)))
        )

    def _calc(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        syn0 = F.relu(self.bnc1(self.c1(e0)))
        del e0
        e1 = F.max_pooling_nd(syn0, self.pool_size, self.pool_size)
        e2 = F.relu(self.bnc2(self.c2(e1)))
        syn1 = F.relu(self.bnc3(self.c3(e2)))
        del e1, e2
        e3 = F.max_pooling_nd(syn1, self.pool_size, self.pool_size)
        e4 = F.relu(self.bnc4(self.c4(e3)))
        e5 = F.relu(self.bnc5(self.c5(e4)))
        del e3, e4
        d0 = F.concat([self.dc0(e5), syn1])
        del e5, syn1
        d1 = F.relu(self.bndc1(self.dc1(d0)))
        d2 = F.relu(self.bndc2(self.dc2(d1)))
        del d0, d1
        d3 = F.concat([self.dc3(d2), syn0])
        del d2, syn0
        d4 = F.relu(self.bndc4(self.dc4(d3)))
        d5 = F.relu(self.bndc5(self.dc5(d4)))
        del d3, d4
        d6 = self.dc6(d5)
        del d5
        return d6

    def __call__(self, x, t=None, seg=True):
        h = self._calc(x)
        if seg:
            pred = F.softmax(h)
            del h
            return pred.data
        else:
            #loss = eval(self.loss_func)(h, t, class_weight=self.class_weight)
            loss = eval(self.loss_func)(h, t)
            pred = F.softmax(h)
            del h
            return loss, pred.data


class Model_L3(Chain):
    def __init__(
            self,
            ndim=3,
            n_class=2,
            init_channel=2,
            kernel_size=3,
            pool_size=2,
            ap_factor=2,
            gpu=-1,
            class_weight=np.array([1, 1]).astype(np.float32),
            loss_func='F.softmax_cross_entropy'
        ):
        self.gpu = gpu
        self.pool_size = pool_size
        if gpu >= 0:
            self.class_weight = cuda.to_gpu(np.array(class_weight).astype(np.float32))
        else:
            self.class_weight = np.array(class_weight).astype(np.float32)
        self.train = True
        self.loss_func = loss_func
        initializer = chainer.initializers.HeNormal()
        super(Model_L3, self).__init__(

            c0=L.ConvolutionND(ndim, 1, init_channel, kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            c1=L.ConvolutionND(ndim, init_channel, int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            c2=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            c3=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            c4=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            c5=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            c6=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            c7=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            dc0=L.DeconvolutionND(ndim, int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 4)), self.pool_size, self.pool_size, 0, initialW=initializer, initial_bias=None),
            dc1=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 3) + init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            dc2=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            dc3=L.DeconvolutionND(ndim, int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), self.pool_size, self.pool_size, 0, initialW=initializer, initial_bias=None),
            dc4=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2) + init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            dc5=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            dc6=L.DeconvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), self.pool_size, self.pool_size, 0, initialW=initializer, initial_bias=None),
            dc7=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1) + init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            dc8=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            dc9=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), n_class, 1, 1, initialW=initializer, initial_bias=None),

            bnc0=L.BatchNormalization(init_channel),
            bnc1=L.BatchNormalization(int(init_channel * (ap_factor ** 1))),

            bnc2=L.BatchNormalization(int(init_channel * (ap_factor ** 1))),
            bnc3=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),

            bnc4=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),
            bnc5=L.BatchNormalization(int(init_channel * (ap_factor ** 3))),

            bnc6=L.BatchNormalization(int(init_channel * (ap_factor ** 3))),
            bnc7=L.BatchNormalization(int(init_channel * (ap_factor ** 4))),

            bndc1=L.BatchNormalization(int(init_channel * (ap_factor ** 3))),
            bndc2=L.BatchNormalization(int(init_channel * (ap_factor ** 3))),
            bndc4=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),
            bndc5=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),
            bndc7=L.BatchNormalization(int(init_channel * (ap_factor ** 1))),
            bndc8=L.BatchNormalization(int(init_channel * (ap_factor ** 1)))
        )

    def _calc(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        syn0 = F.relu(self.bnc1(self.c1(e0)))
        del e0
        e1 = F.max_pooling_nd(syn0, self.pool_size, self.pool_size)
        e2 = F.relu(self.bnc2(self.c2(e1)))
        syn1 = F.relu(self.bnc3(self.c3(e2)))
        del e1, e2
        e3 = F.max_pooling_nd(syn1, self.pool_size, self.pool_size)
        e4 = F.relu(self.bnc4(self.c4(e3)))
        syn2 = F.relu(self.bnc5(self.c5(e4)))
        del e3, e4
        e5 = F.max_pooling_nd(syn2, self.pool_size, self.pool_size)
        e6 = F.relu(self.bnc6(self.c6(e5)))
        e7 = F.relu(self.bnc7(self.c7(e6)))
        del e5, e6
        d0 = F.concat([self.dc0(e7), syn2])
        del e7, syn2
        d1 = F.relu(self.bndc1(self.dc1(d0)))
        d2 = F.relu(self.bndc2(self.dc2(d1)))
        del d0, d1
        d3 = F.concat([self.dc3(d2), syn1])
        del d2, syn1
        d4 = F.relu(self.bndc4(self.dc4(d3)))
        d5 = F.relu(self.bndc5(self.dc5(d4)))
        del d3, d4
        d6 = F.concat([self.dc6(d5), syn0])
        del d5, syn0
        d7 = F.relu(self.bndc7(self.dc7(d6)))
        d8 = F.relu(self.bndc8(self.dc8(d7)))
        del d6, d7
        d9 = self.dc9(d8)
        del d8
        return d9

    def __call__(self, x, t=None, seg=True):
        h = self._calc(x)
        if seg:
            pred = F.softmax(h)
            del h
            return pred.data
        else:
            #loss = eval(self.loss_func)(h, t, class_weight=self.class_weight)
            loss = eval(self.loss_func)(h, t)
            pred = F.softmax(h)
            del h
            return loss, pred.data


class Model_L4(Chain):
    def __init__(
            self,
            ndim=3,
            n_class=2,
            init_channel=2,
            kernel_size=3,
            pool_size=2,
            ap_factor=2,
            gpu=-1,
            class_weight=np.array([1, 1]).astype(np.float32),
            loss_func='F.softmax_cross_entropy'
        ):
        self.gpu = gpu
        self.pool_size = pool_size
        if gpu >= 0:
            self.class_weight = cuda.to_gpu(np.array(class_weight).astype(np.float32))
        else:
            self.class_weight = np.array(class_weight).astype(np.float32)
        self.train = True
        self.loss_func = loss_func
        initializer = chainer.initializers.HeNormal()
        super(Model_L4, self).__init__(

            c0=L.ConvolutionND(ndim, 1, init_channel, kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            c1=L.ConvolutionND(ndim, init_channel, int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            c2=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            c3=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            c4=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            c5=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            c6=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            c7=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            c8=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            c9=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 5)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            dc0=L.DeconvolutionND(ndim, int(init_channel * (ap_factor ** 5)), int(init_channel * (ap_factor ** 5)), self.pool_size, self.pool_size, 0, initialW=initializer, initial_bias=None),
            dc1=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 4) + init_channel * (ap_factor ** 5)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            dc2=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 4)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            dc3=L.DeconvolutionND(ndim, int(init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 4)), self.pool_size, self.pool_size, 0, initialW=initializer, initial_bias=None),
            dc4=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 3) + init_channel * (ap_factor ** 4)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            dc5=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            dc6=L.DeconvolutionND(ndim, int(init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 3)), self.pool_size, self.pool_size, 0, initialW=initializer, initial_bias=None),
            dc7=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2) + init_channel * (ap_factor ** 3)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            dc8=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            dc9=L.DeconvolutionND(ndim, int(init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 2)), self.pool_size, self.pool_size, 0, initialW=initializer, initial_bias=None),
            dc10=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1) + init_channel * (ap_factor ** 2)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),
            dc11=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), int(init_channel * (ap_factor ** 1)), kernel_size, 1, int(kernel_size/2), initialW=initializer, initial_bias=None),

            dc12=L.ConvolutionND(ndim, int(init_channel * (ap_factor ** 1)), n_class, 1, 1, initialW=initializer, initial_bias=None),

            bnc0=L.BatchNormalization(init_channel),
            bnc1=L.BatchNormalization(int(init_channel * (ap_factor ** 1))),

            bnc2=L.BatchNormalization(int(init_channel * (ap_factor ** 1))),
            bnc3=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),

            bnc4=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),
            bnc5=L.BatchNormalization(int(init_channel * (ap_factor ** 3))),

            bnc6=L.BatchNormalization(int(init_channel * (ap_factor ** 3))),
            bnc7=L.BatchNormalization(int(init_channel * (ap_factor ** 4))),

            bnc8=L.BatchNormalization(int(init_channel * (ap_factor ** 4))),
            bnc9=L.BatchNormalization(int(init_channel * (ap_factor ** 5))),
            bndc1=L.BatchNormalization(int(init_channel * (ap_factor ** 4))),
            bndc2=L.BatchNormalization(int(init_channel * (ap_factor ** 4))),
            bndc4=L.BatchNormalization(int(init_channel * (ap_factor ** 3))),
            bndc5=L.BatchNormalization(int(init_channel * (ap_factor ** 3))),
            bndc7=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),
            bndc8=L.BatchNormalization(int(init_channel * (ap_factor ** 2))),
            bndc10=L.BatchNormalization(int(init_channel * (ap_factor ** 1))),
            bndc11=L.BatchNormalization(int(init_channel * (ap_factor ** 1)))

        )

    def _calc(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        syn0 = F.relu(self.bnc1(self.c1(e0)))
        del e0
        e1 = F.max_pooling_nd(syn0, self.pool_size, self.pool_size)
        e2 = F.relu(self.bnc2(self.c2(e1)))
        syn1 = F.relu(self.bnc3(self.c3(e2)))
        del e1, e2
        e3 = F.max_pooling_nd(syn1, self.pool_size, self.pool_size)
        e4 = F.relu(self.bnc4(self.c4(e3)))
        syn2 = F.relu(self.bnc5(self.c5(e4)))
        del e3, e4
        e5 = F.max_pooling_nd(syn2, self.pool_size, self.pool_size)
        e6 = F.relu(self.bnc6(self.c6(e5)))
        syn3 = F.relu(self.bnc7(self.c7(e6)))
        del e5, e6
        e7 = F.max_pooling_nd(syn3, self.pool_size, self.pool_size)
        e8 = F.relu(self.bnc8(self.c8(e7)))
        e9 = F.relu(self.bnc9(self.c9(e8)))
        del e7, e8
        d0 = F.concat([self.dc0(e9), syn3])
        del e9, syn3
        d1 = F.relu(self.bndc1(self.dc1(d0)))
        d2 = F.relu(self.bndc2(self.dc2(d1)))
        del d0, d1
        d3 = F.concat([self.dc3(d2), syn2])
        del d2, syn2
        d4 = F.relu(self.bndc4(self.dc4(d3)))
        d5 = F.relu(self.bndc5(self.dc5(d4)))
        del d3, d4
        d6 = F.concat([self.dc6(d5), syn1])
        del d5, syn1
        d7 = F.relu(self.bndc7(self.dc7(d6)))
        d8 = F.relu(self.bndc8(self.dc8(d7)))
        del d6, d7
        d9 = F.concat([self.dc9(d8), syn0])
        del d8, syn0
        d10 = F.relu(self.bndc10(self.dc10(d9)))
        d11 = F.relu(self.bndc11(self.dc11(d10)))
        del d9, d10

        d12 = self.dc12(d11)
        del d11
        return d12

    def __call__(self, x, t=None, seg=True):
        h = self._calc(x)
        if seg:
            pred = F.softmax(h)
            del h
            return pred.data
        else:
            #loss = eval(self.loss_func)(h, t, class_weight=self.class_weight)
            loss = eval(self.loss_func)(h, t)
            pred = F.softmax(h)
            del h
            return loss, pred.data
