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
from lib.model import Model_L2, Model_L3, Model_L4
from lib.utils import Utils
from lib.trainer import NSNTrainer


class TestNSN():
    def __init__(self, model=None, patchsize=128, stride=64, resolution=(1.0, 1.0, 2.18),
                 scaling=True, opbase=None, gpu=False):
        self.model = model
        self.patchsize = patchsize
        self.stride = stride
        self.resolution = resolution
        self.scaling = scaling
        self.opbase = opbase
        self.gpu = gpu
        self.psep = '/'

    def NuclearSegmentation(self, image_path):
        segbase = 'SegmentationImages'
        if not (pt.exists(self.opbase + self.psep + segbase)):
            os.mkdir(self.opbase + self.psep + segbase)

        util = Utils(self.patchsize)
        image = io.imread(image_path)
        im_size = image.shape
        ip_size = (int(image.shape[0] * self.resolution[2]), int(image.shape[1] * self.resolution[1]), int(image.shape[2] * self.resolution[0]))
        image = tr.resize(image, ip_size, order = 1, preserve_range = True)
        im_size_ip = image.shape
        #pre_psize = int(self.patchsize - (self.stride/2))

        # Scaling
        if self.scaling:
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min())

        # Extension Image            
        if self.patchsize > np.max(ip_size):
            pad_size = self.patchsize
        else:
            if (np.max(ip_size) - self.patchsize) % self.stride == 0:
                stride_num = (np.max(ip_size) - self.patchsize) / self.stride
            else:
                stride_num = (np.max(ip_size) - self.patchsize) / self.stride + 1
            pad_size = self.stride * stride_num + self.patchsize
        image = util.mirrorExtensionImage(image=image, length=int(self.patchsize))[0:pad_size, 0:pad_size, 0:pad_size]
        pre_img = np.zeros((image.shape))

        for z in range(0, pad_size-self.stride, self.stride):
            for y in range(0, pad_size-self.stride, self.stride):
                for x in range(0, pad_size-self.stride, self.stride):
                    x_patch = image[z:z+self.patchsize, y:y+self.patchsize, x:x+self.patchsize]
                    x_patch = x_patch.reshape(1, 1, self.patchsize, self.patchsize, self.patchsize).astype(np.float32)
                    if self.gpu >= 0:
                        x_patch = cuda.to_gpu(x_patch)
                    s_output = self.model(x_patch, seg=True)
                    if self.gpu >= 0:
                        s_output = cuda.to_cpu(s_output)
                    pred = copy.deepcopy((0 < (s_output[0][1] - s_output[0][0])) * 255)
                    # Add segmentation image
                    pre_img[z:z+self.patchsize, y:y+self.patchsize, x:x+self.patchsize] += pred
        seg_img = (pre_img > 0) * 255
        seg_img = seg_img[0:im_size_ip[0], 0:im_size_ip[1], 0:im_size_ip[2]]
        seg_img = (tr.resize(seg_img, im_size, order = 1, preserve_range = True) > 0) * 255
        filename = self.opbase + self.psep + segbase + self.psep + 'segimg_{}.tif'.format(image_path[image_path.rfind('/')+1:image_path.rfind('.')])
        io.imsave(filename, seg_img.astype(np.uint8))

        return seg_img
        

if __name__ == '__main__':
    
    start_time = time.time()
    ap = ArgumentParser(description='python test_nsn.py')
    ap.add_argument('--indir', '-i', nargs='?', default='../images/example_input/', help='Specify input image')
    ap.add_argument('--outdir', '-o', nargs='?', default='result_test_nsn', help='Specify output files directory for create detection image')
    ap.add_argument('--model', '-m', nargs='?', default='../models/p128/learned_nsn.npz', help='Specify loading file path of learned NSN Model')
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
