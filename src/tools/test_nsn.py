# -*- coding: utf-8 -*-

import csv
import sys
import time
import random
import copy
import math
import os
import numpy as np
import os.path as pt
from skimage import io
from skimage import transform as tr
from argparse import ArgumentParser
from chainer import cuda

sys.path.append(os.getcwd())
from src.lib.model import Model_L2, Model_L3, Model_L4
from src.lib.utils import mirror_extension_image


class TestNSN():
    def __init__(
        self,
        model=None,
        patchsize=128,
        stride=64,
        resolution=(1.0, 1.0, 2.18),
        scaling=True,
        opbase=None,
        gpu=False,
        ndim=3
        ):
        self.model = model
        self.patchsize = (patchsize, patchsize, patchsize)
        self.stride = (stride, stride, stride)
        self.resolution = resolution
        self.scaling = scaling
        self.opbase = opbase
        self.gpu = gpu
        self.psep = '/'
        self.ndim = ndim

    def NuclearSegmentation(self, image_path):
        segbase = 'SegmentationImages'
        if not (pt.exists(self.opbase + self.psep + segbase)):
            os.mkdir(self.opbase + self.psep + segbase)

        image = io.imread(image_path)
        im_size = image.shape
        if self.ndim == 2:
            ip_size = (int(image.shape[0] * self.resolution[1]), int(image.shape[1] * self.resolution[0]))
            sh = [int(self.stride[0]/2), int(self.stride[1]/2)]
        elif self.ndim == 3:
            ip_size = (int(image.shape[0] * self.resolution[2]), int(image.shape[1] * self.resolution[1]), int(image.shape[2] * self.resolution[0]))
            sh = [int(self.stride[0]/2), int(self.stride[1]/2), int(self.stride[2]/2)]
        print('ip_size: {}'.format(ip_size))
        image = tr.resize(image, ip_size, order = 1, preserve_range = True)
        im_size_ip = image.shape
        print('ip_size: {}'.format(im_size_ip))
        #pre_psize = int(self.patchsize - (self.stride/2))

        # Scaling
        if self.scaling:
            image = image.astype(np.float32)
            #image = image / image.max()
            image = (image - image.min()) / (image.max() - image.min())



        ''' calculation for pad size'''
        if np.min(self.patchsize) > np.max(np.array(im_size) + np.array(sh)*2):
            if self.ndim == 2:
                pad_size = [self.patchsize[0], self.patchsize[1]]
            elif self.ndim == 3:
                pad_size = [self.patchsize[0], self.patchsize[1], self.patchsize[2]]
        else:
            pad_size = []
            for axis in range(len(im_size_ip)):
                if (im_size_ip[axis] + 2*sh[axis] - self.patchsize[axis]) % self.stride[axis] == 0:
                    stride_num = int((im_size_ip[axis] + 2*sh[axis] - self.patchsize[axis]) / self.stride[axis])
                else:
                    stride_num = int((im_size_ip[axis] + 2*sh[axis] - self.patchsize[axis]) / self.stride[axis]) + 1
                pad_size.append(int(self.stride[axis] * stride_num + self.patchsize[axis]))

        pre_img = np.zeros(pad_size)

        if self.ndim == 2:
            image = mirror_extension_image(image=image, ndim=self.ndim, length=int(np.max(self.patchsize)))[self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1]]
            for y in range(0, pad_size[0]-self.stride[0], self.stride[0]):
                for x in range(0, pad_size[1]-self.stride[1], self.stride[1]):
                    x_patch = image[y:y+self.patchsize[0], x:x+self.patchsize[1]]
                    x_patch = np.expand_dims(np.expand_dims(x_patch.astype(np.float32), axis=0), axis=0)
                    if self.gpu >= 0:
                        x_patch = cuda.to_gpu(x_patch)
                    s_output = self.model(x=x_patch, t=None, seg=True)
                    if self.gpu >= 0:
                        s_output = cuda.to_cpu(s_output)
                    pred = copy.deepcopy((0 < (s_output[0][1] - s_output[0][0])) * 255)
                    # Add segmentation image
                    pre_img[y:y+self.stride[0], x:x+self.stride[1]] += pred[sh[0]:-sh[0], sh[1]:-sh[1]]
            seg_img = (pre_img > 0) * 255
            seg_img = seg_img[:im_size_ip[0], :im_size_ip[1]]

        elif self.ndim == 3:
            image = mirror_extension_image(image=image, ndim=self.ndim, length=int(np.max(self.patchsize)))[self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1], self.patchsize[2]-sh[2]:self.patchsize[2]-sh[2]+pad_size[2]]
            for z in range(0, pad_size[0]-self.stride[0], self.stride[0]):
                for y in range(0, pad_size[1]-self.stride[1], self.stride[1]):
                    for x in range(0, pad_size[2]-self.stride[2], self.stride[2]):
                        x_patch = image[z:z+self.patchsize[0], y:y+self.patchsize[1], x:x+self.patchsize[2]]
                        x_patch = np.expand_dims(np.expand_dims(x_patch.astype(np.float32), axis=0), axis=0)
                        if self.gpu >= 0:
                            x_patch = cuda.to_gpu(x_patch)
                        s_output = self.model(x=x_patch, t=None, seg=True)
                        if self.gpu >= 0:
                            s_output = cuda.to_cpu(s_output)
                        pred = copy.deepcopy((0 < (s_output[0][1] - s_output[0][0])) * 255)
                        # Add segmentation image
                        pre_img[z:z+self.stride[0], y:y+self.stride[1], x:x+self.stride[2]] += pred[sh[0]:-sh[0], sh[1]:-sh[1], sh[2]:-sh[2]]
            seg_img = (pre_img > 0) * 255
            seg_img = seg_img[:im_size_ip[0], :im_size_ip[1], :im_size_ip[2]]
        seg_img = (tr.resize(seg_img, im_size, order = 1, preserve_range = True) > 0) * 255
        filename = self.opbase + self.psep + segbase + self.psep + os.path.basename(image_path)
        #filename = self.opbase + self.psep + segbase + self.psep + 'segimg_t{0:03d}.tif'.format(int(image_path[image_path.rfind('/')+1:image_path.rfind('.')]))
        io.imsave(filename, seg_img.astype(np.uint8))

        return seg_img.astype(np.uint16)


if __name__ == '__main__':

    start_time = time.time()
    ap = ArgumentParser(description='python test_nsn.py')
    ap.add_argument('--indir', '-i', nargs='?', default='../images/example_input/', help='Specify input image')
    ap.add_argument('--outdir', '-o', nargs='?', default='result_test_nsn', help='Specify output files directory for create detection image')
    ap.add_argument('--model', '-m', nargs='?', default='../models/p128/learned_nsn.model', help='Specify loading file path of learned NSN Model')
    ap.add_argument('--gpu', '-g', type=int, default=-1, help='Specify GPU ID (negative value indicates CPU)')
    ap.add_argument('--patchsize', '-p', type=int, default=128, help='Specify patch size')
    ap.add_argument('--stride', '-s', type=int, default=64, help='Specify stride size')
    ap.add_argument('--scaling', action='store_true', help='Specify Image-wise Scaling Flag')
    ap.add_argument('--resolution_x', '-x', type=float, default=1.0, help='Specify microscope resolution of x axis (default=1.0)')
    ap.add_argument('--resolution_y', '-y', type=float, default=1.0, help='Specify microscope resolution of y axis (default=1.0)')
    ap.add_argument('--resolution_z', '-z', type=float, default=2.18, help='Specify microscope resolution of z axis (default=2.18)')

    args = ap.parse_args()
    argvs = sys.argv
    util = Utils()
    psep = '/'

    opbase = util.createOpbase(args.outdir)
    patchsize = args.patchsize
    stride = args.stride
    print('Patch Size: {}'.format(patchsize))
    print('Stride Size: {}'.format(stride))
    with open(opbase + psep + 'result.txt', 'w') as f:
        f.write('python ' + ' '.join(argvs) + '\n')
        f.write('[Properties of parameter]\n')
        f.write('Output Directory: {}\n'.format(opbase))
        f.write('Patch Size: {}\n'.format(patchsize))
        f.write('Stride Size: {}\n'.format(stride))

    # Create Model
    class_weight = np.array([1, 1]).astype(np.float32)
    if args.gpu >= 0:
        class_weight = cuda.to_gpu(class_weight)
    # Adam
    # nsn = Model_L2(class_weight=class_weight, n_class=2, init_channel=16,
    #                kernel_size=3, pool_size=2, ap_factor=2, gpu=args.gpu)
    # SGD
    nsn = Model_L2(class_weight=class_weight, n_class=2, init_channel=16,
                   kernel_size=3, pool_size=2, ap_factor=2, gpu=args.gpu)

    # Load Model
    if not args.model == '0':
        util.loadModel(args.model, nsn)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        nsn.to_gpu()

    # Segmentation Phase
    test_nsn = TestNSN(model=nsn, patchsize=patchsize, stride=stride,
                       resolution=(args.resolution_x, args.resolution_y, args.resolution_z),
                       scaling=args.scaling, opbase=opbase, gpu=args.gpu)
    dlist = os.listdir(args.indir)
    for l in dlist:
        test_nsn.NuclearSegmentation(args.indir + psep + l)

    end_time = time.time()
    etime = end_time - start_time
    print('Elapsed time is (sec) {}'.format(etime))
    with open(opbase + psep + 'result.txt', 'a') as f:
        f.write('======================================\n')
        f.write('Elapsed time is (sec) {} \n'.format(etime))
    print('NSN Test Completed Process!')
