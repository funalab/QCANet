import numpy as np
import chainer
from chainer import cuda, function, utils
from chainer.functions.activation import softmax


class DiceLoss(function.Function):
    def __init__(self, eps=0.0):
        # avoid zero division error
        self.eps = eps

    def check_type_forward(self, in_types):
        utils.type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        utils.type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype == np.int32,
            t_type.ndim == x_type.ndim - 1,

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[1:],
        )

    @staticmethod
    def _check_input_values(x, t):
        if not (((0 <= t) &
                 (t < x.shape[1]))).all():
            msg = ('Each label `t` need to satisfy '
                   '`0 <= t < x.shape[1]`')
            raise ValueError(msg)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        if chainer.is_debug():
            self._check_input_values(x, t)
        # one-hot encoding of ground truth
        t = encode_one_hot_vector(t, x.shape[1])
        # compute intersection and union
        if len(x.shape) == 4:
            self.intersect = xp.sum((x * t), axis=(0, 2, 3))
            self.union = xp.sum((x * x), axis=(0, 2, 3)) + xp.sum((t * t), axis=(0, 2, 3))
        elif len(x.shape) == 5:
            self.intersect = xp.sum((x * t), axis=(0, 2, 3, 4))
            self.union = xp.sum((x * x), axis=(0, 2, 3, 4)) + xp.sum((t * t), axis=(0, 2, 3, 4))
        else:
            raise ValueError('mismatch len(x.shape) : {}'.format(len(x.shape)))
        # compute dice loss
        dice = (2. * self.intersect + self.eps) / (self.union + self.eps)
        return utils.force_array(xp.mean(1. - dice), dtype=xp.float32),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        nb_class = x.shape[1]
        t = encode_one_hot_vector(t, nb_class)

        gx = xp.zeros_like(x)
        gloss = grad_outputs[0]
        for i in range(nb_class):
            x_i = x[:, i]
            t_i = t[:, i]
            intersect = self.intersect[i]
            union = self.union[i]

            numerator = xp.multiply(union + self.eps, t_i) - xp.multiply(2. * intersect + self.eps, x_i)
            denominator = xp.power(union + self.eps, 2)
            dDice = 2 * xp.divide(numerator, denominator).astype(xp.float32)
            gx[:, i] = dDice

        gx *= gloss / nb_class
        return -gx.astype(xp.float32), None


def _encode_one_hot_vector_core(x, nb_class):
    xp = cuda.get_array_module(x)
    batch, h, w, d = x.shape

    res = xp.zeros((batch, nb_class, h, w, d), dtype=xp.float32)
    x = x.reshape(batch, -1)
    for i in range(batch):
        y = xp.identity(nb_class, dtype=xp.float32)[x[i]]
        res[i] = xp.swapaxes(y, 0, 1).reshape((nb_class, h, w, d))
    return res

def encode_one_hot_vector(x, nb_class):
    if isinstance(x, cuda.ndarray):
        with x.device:
            return _encode_one_hot_vector_core(x, nb_class)
    else:
        return _encode_one_hot_vector_core(x, nb_class)

def dice_loss(x, t, eps=0.0):
    return DiceLoss(eps)(x, t)

def softmax_dice_loss(x, t, eps=1e-7):
    return DiceLoss(eps)(softmax.softmax(x, axis=1), t)
