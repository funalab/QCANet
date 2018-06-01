# -*- coding: utf-8 -*-

from chainer import cuda

import csv
import sys
import time
import random
import copy
import math
import os
import numpy as np
from skimage import io
from skimage import transform as tr
from argparse import ArgumentParser

from lib.model import Model_L2, Model_L3, Model_L4
from lib.utils import Utils
from lib.trainer import NDNTrainer


def main():
    start_time = time.time()
    ap = ArgumentParser(description='python train_ndn.py')
    ap.add_argument('--indir', '-i', nargs='?', default='../example_datasets/', help='Specify input files directory learning data')
    ap.add_argument('--outdir', '-o', nargs='?', default='result_train_ndn', help='Specify output files directory for create segmentation image and save model file')
    ap.add_argument('--gpu', '-g', type=int, default=-1, help='Specify GPU ID (negative value indicates CPU)')
    ap.add_argument('--patchsize', '-p', type=int, default=128, help='Specify one side voxel size of ROI')
    ap.add_argument('--paddingsize', type=int, default=128, help='Specify image size after padding')
    ap.add_argument('--epoch', '-e', type=int, default=10, help='Specify number of sweeps over the dataset to train')
    ap.add_argument('--resolution_x', '-x', type=float, default=1.0, help='Specify microscope resolution of x axis (default=1.0)')
    ap.add_argument('--resolution_y', '-y', type=float, default=1.0, help='Specify microscope resolution of y axis (default=1.0)')
    ap.add_argument('--resolution_z', '-z', type=float, default=2.18, help='Specify microscope resolution of z axis (default=2.18)')
    ap.add_argument('--batchsize', '-b', type=int, default=1, help='Specify mini batch size')
    ap.add_argument('--crossvalidation', '-c', type=int, default=2, help='Specify k-fold cross validation')
    ap.add_argument('--normalization', '-n', action='store_true', help='Specify mean normalization method flag')
    ap.add_argument('--augmentation', '-a', action='store_true', help='Specify data augmentation flag (flip)')
    ap.add_argument('--classweight', '-w', action='store_true', help='Specify Softmax_Corss_Entropy with Class Weight Flag')
    ap.add_argument('--scaling', '-s', action='store_true', help='Specify Image-wise Scaling Image Flag')
    ap.add_argument('--opt_method', nargs='?', choices=['Adam', 'SGD'], help='Specify Choices Optimizer Method (Adam or SGD)')

    args = ap.parse_args()
    argvs = sys.argv
    patchsize = args.patchsize
    util = Utils(patchsize, args.batchsize)
    opbase = util.createOpbase(args.outdir)
    psep = '/'

    # Load Images
    images = {}
    xlist, ylist = [], []
    dlist = os.listdir(args.indir)
    try:
        dlist.pop(dlist.index('.DS_Store'))
    except:
        pass
    dlist = np.sort(dlist)
    for dl in dlist:
        xlist.append('ori' + dl)
        ylist.append('ans' + dl)
        images['ori' + dl] = io.imread(args.indir + psep + dl + psep + 'image.tif')
        images['ans' + dl] = io.imread(args.indir + psep + dl + psep + 'detection_gt.tif')

    # Pre-Processing 1 (Interpolation)
    for dl in dlist:
        ip_size = (int(images['ori' + dl].shape[0] * args.resolution_z), int(images['ori' + dl].shape[1] * args.resolution_y), int(images['ori' + dl].shape[2] * args.resolution_x))
        images['ori' + dl] = tr.resize(images['ori' + dl], ip_size, order = 1, preserve_range = True)
        images['ans' + dl] = tr.resize(images['ans' + dl], ip_size, order = 1, preserve_range = True)
        
    # Pre-Processing 2 (Dataset Augmentation)
    orilist = copy.deepcopy(ylist)
    if args.augmentation:
        expxlist, expylist = [], []
        lz, ly, lx = images[xlist[0]].shape
        for t in range(len(xlist)):
            aug_ori = util.dataAugmentation(images[xlist[t]], rot=False, flip=True)
            aug_ans = util.dataAugmentation(images[ylist[t]], rot=False, flip=True)
            for i in range(len(aug_ori)):
                ori = aug_ori[i]
                ans = aug_ans[i]
                listname_ori = xlist[t] + '_r' + str(i)
                listname_ans = ylist[t] + '_r' + str(i)
                images[listname_ori] = copy.deepcopy(ori.astype(np.float32))
                images[listname_ans] = copy.deepcopy(ans.astype(np.float32))
                xlist.append(listname_ori)
                ylist.append(listname_ans)
                expxlist.append(listname_ori)
                expylist.append(listname_ans)
        print('Augmentation Complete!')

    # Pre-Processing 3 (Image Scaling)
    if args.scaling:
        for t in xlist:
            smax = np.max(images[t])
            smin = np.min(images[t])
            images[t] = images[t].astype(np.float32)
            images[t] = (images[t] - smin) / (smax - smin)

    # GT Processing
    for t in ylist:
        images[t] = (images[t] > 0) * 1
        images[t] = images[t].astype(np.int32)

    # Pre-Processing 4 (Image Padding)
    lz, ly, lx = images[ylist[0]].shape
    for t in xlist:
        images[t] = util.mirrorExtensionImage(images[t], int(patchsize/2))[0:args.paddingsize, 0:args.paddingsize, 0:args.paddingsize]
    for t in ylist:
        images[t] = util.mirrorExtensionImage(images[t], int(patchsize/2))[0:args.paddingsize, 0:args.paddingsize, 0:args.paddingsize]

    # StrageIndex
    # [t, x, y, z]
    # t is index of ylist, otherwise start point of coordinates, stride is patchsize
    idx, exidx = [], []
    for t in range(len(ylist)):
        for z in range(0, lz, patchsize):
            for y in range(0, ly, patchsize):
                for x in range(0, lx, patchsize):
                    if t < len(orilist):  # original image index
                        idx.append([t, x, y, z])
                    else:  # expansion image index
                        exidx.append([t, x, y, z])
    print('Strage Index Complete!')

    # Validation of Training and Test Datasets
    N_all = len(idx)
    random.shuffle(idx)
    crossIdx = []
    for k in range(args.crossvalidation):
        crossIdx.append(idx[int(N_all * k / args.crossvalidation) : int(N_all * (k+1) / args.crossvalidation)])


    # Number of Original Data Size
    N_train = len(crossIdx[0] * (args.crossvalidation-1))
    N_test = len(crossIdx[0])
        
    # Output Data Properties
    print('[Training Data Properties]')
    print('number of Train Data : ' + str(N_train))
    print('[Test Data Properties]')
    print('number of Test Data : ' + str(N_test))
    with open(opbase + psep + 'result.txt', 'w') as f:
        f.write('python ' + ' '.join(argvs) + '\n')
        f.write('[Hyperparameter of Learning Properties]\n')
        f.write('Patch Size: {}×{}×{}\n'.format(patchsize, patchsize, patchsize))
        f.write('Output Directory: {}\n'.format(opbase))
        if args.scaling:
            f.write('Pre-Processing (Scaling Image): True\n')
        else:
            f.write('Pre-Processing (Scaling Image): False\n')
        if args.normalization:
            f.write('Pre-Processing (Normalization Image): True\n')
        else:
            f.write('Pre-Processing (Normalization Image): False\n')
        if args.augmentation:
            f.write('Pre-Processing (Augmentation Image): True\n')
        else:
            f.write('Pre-Processing (Augmentation Image): False\n')
        if args.classweight:
            f.write('Class Weight: True\n')
        else:
            f.write('Class Weight: False\n')
        f.write('GPU: {}\n'.format(args.gpu))
        f.write('[Dataset Properties]\n')
        f.write('number of Original Train Data : {}\n'.format(N_train))
        f.write('number of Original Test Data : {}\n'.format(N_test))
        f.write('x resolution : {}\n'.format(args.resolution_x))
        f.write('y resolution : {}\n'.format(args.resolution_y))
        f.write('z resolution : {}\n'.format(args.resolution_z))
    with open(opbase + psep + 'TestResult.csv', 'w') as f:
        c = csv.writer(f)
        c.writerow(['Epoch', 'Accuracy', 'Recall', 'Precision', 'Specificity', 'F-measure', 'IoU'])
    with open(opbase + psep + 'TrainResult.csv', 'w') as f:
        c = csv.writer(f)
        c.writerow(['Epoch', 'Accuracy', 'Recall', 'Precision', 'Specificity', 'F-measure', 'IoU'])

    bestScore = []
    for k in range(args.crossvalidation):
        print('==========================================')
        print('Cross Validation: ' + str(k + 1))

        testIdx = crossIdx[k]
        trainIdx = []
        for kk in range(k, k+args.crossvalidation-1):
            ni = kk - args.crossvalidation + 1
            trainIdx += crossIdx[ni]
            if args.augmentation:
                for kkk in range(len(crossIdx[ni])):  # Add Augmentation Data
                    trainIdx += [exidx[crossIdx[ni][kkk][0] * 3]]
                    trainIdx += [exidx[crossIdx[ni][kkk][0] * 3 + 1]]
                    trainIdx += [exidx[crossIdx[ni][kkk][0] * 3 + 2]]

        # Pre-Processing 5 (Mean Normalization)
        if args.normalization:
            sumImg = 0
            for t in trainIdx:
                images[xlist[t[0]]] = images[xlist[t[0]]].astype(np.float32)
                sumImg += images[xlist[t[0]]]
            mean_image = copy.deepcopy(sumImg / len(trainIdx))
            print('Normalization method : Mean Norm')
            io.imsave(opbase + psep + 'meanImg' + str(k+1) + '.tif', mean_image)
        else:
            mean_image = None
            print('Normalization method : Not Adapted')

        with open (opbase + psep + 'result.txt', 'a') as f:
            f.write('=========================================\n')
            f.write('Cross Validation: {}\n'.format(k + 1))
            f.write('Number of Train Data: {}\n'.format(len(trainIdx)))
            f.write('Number of Test Data: {}\n'.format(len(testIdx)))
            f.write('Validation Data: {}\n'.format(testIdx))
        print('Number of Train Data: ' + str(len(trainIdx)))
        print('Number of Test Data: ' + str(len(testIdx)))

        if args.classweight:
            obj, bg = 0, 0
            for n in range(len(trainIdx)):
                y_patch = images[ylist[trainIdx[n][0]]][ trainIdx[n][3]:trainIdx[n][3]+patchsize, trainIdx[n][2]:trainIdx[n][2]+patchsize, trainIdx[n][1]:trainIdx[n][1]+patchsize ]
                bg += np.unique(y_patch, return_counts=True)[1][0]
                obj += np.unique(y_patch, return_counts=True)[1][1]
            bg_w  = float(bg + obj) / bg
            obj_w = float(bg + obj) / obj
            print('bg_w: ' + str(bg_w))
            print('obj_w: ' + str(obj_w))
            class_weight = np.array([bg_w, obj_w]).astype(np.float32)                
        else:
            class_weight = np.array([1, 1]).astype(np.float32)

        if args.gpu >= 0:
            class_weight = cuda.to_gpu(class_weight)

        # Create Model
        if args.opt_method == 'SGD':
            ndn = Model_L3(class_weight=class_weight, n_class=2, init_channel=16,
                           kernel_size=3, pool_size=2, ap_factor=2, gpu=args.gpu)
        elif args.opt_method == 'Adam':
            ndn = Model_L4(class_weight=class_weight, n_class=2, init_channel=12,
                           kernel_size=5, pool_size=2, ap_factor=2, gpu=args.gpu)
        else:
            print('Warning!! Specify Choices Optimizer Method (Adam or SGD)')
            sys.exit()

        if args.gpu >= 0:
            cuda.get_device(args.gpu).use()  # Make a specified GPU current
            ndn.to_gpu()  # Copy the SegmentNucleus model to the GPU

        stime = time.time()
        trainer = NDNTrainer(images=images, model=ndn, epoch=args.epoch, patchsize=patchsize,
                             batchsize=args.batchsize, gpu=args.gpu, opbase=opbase,
                             mean_image=mean_image, opt_method=args.opt_method)
        train_eval, test_eval, bs = trainer.NDNTraining(trainIdx, testIdx, xlist, ylist, k)
        bestScore.append(bs)
        etime = time.time()
        print(etime - stime)

    AveRec, AvePre, AveFme = 0, 0, 0
    for k in range(args.crossvalidation):
        AveRec += bestScore[k][0]
        AvePre += bestScore[k][1]
        AveFme += bestScore[k][2]
    with open(opbase + psep + 'result.txt', 'a') as f:
        f.write('======================================\n')
        f.write('Average Best Recall={}\n'.format(AveRec / args.crossvalidation))
        f.write('Average Best Precision={}\n'.format(AvePre / args.crossvalidation))
        f.write('Average Best Fmeasure={}\n'.format(AveFme / args.crossvalidation))

    end_time = time.time()
    process_time = end_time - start_time
    print('Elapsed time is (sec) {}'.format(process_time))
    with open(opbase + psep + 'result.txt', 'a') as f:
        f.write('======================================\n')
        f.write('Elapsed time is (sec) {} \n'.format(process_time))
    print('NDN Training Completed Process!')
        
if __name__ == '__main__':
    main()
