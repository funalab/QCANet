# -*- coding: utf-8 -*-

import chainer
from chainer import cuda

import csv
import sys
import time
import random
import copy
import math
import os
import numpy as np
import argparse
import configparser

sys.path.append(os.getcwd())
from src.lib.trainer import NSNTrainer, NDNTrainer
from src.lib.utils import createOpbase
from src.lib.utils import create_dataset_parser, create_model_parser, create_runtime_parser
from src.lib.utils import print_args
from src.lib.utils import get_dataset, get_model

def main():

    """ Implementation of Quantitative Criteria Acquisition Network based on Chainer """
    start_time = time.time()

    ''' ConfigParser '''
    conf_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    conf_parser.add_argument("-c", "--conf_file",
                             help="Specify config file", metavar="FILE_PATH")
    args, remaining_argv = conf_parser.parse_known_args()

    dataset_conf_dict, model_conf_dict, runtime_conf_dict = {}, {}, {}

    if args.conf_file is not None:
        config = configparser.ConfigParser()
        config.read([args.conf_file])
        dataset_conf_dict = dict(config.items("Dataset"))
        model_conf_dict = dict(config.items("Model"))
        runtime_conf_dict = dict(config.items("Runtime"))


    ''' Parameters '''
    dataset_parser, dataset_args, remaining_argv = \
        create_dataset_parser(remaining_argv, **dataset_conf_dict)
    model_parser, model_args, remaining_argv = \
        create_model_parser(remaining_argv, **model_conf_dict)
    runtime_parser, runtime_args, remaining_argv = \
        create_runtime_parser(remaining_argv, **runtime_conf_dict)

    ap = argparse.ArgumentParser(
        description='Learning Nuclear Segmentation Network',
        parents=[conf_parser, dataset_parser, model_parser, runtime_parser])
    ap.add_argument('--gpu_id', type=int, help='GPU option')
    args = ap.parse_args()

    patchsize = args.patch_size
    opbase = createOpbase(args.save_dir)
    psep = '/'
    print_args(dataset_args, model_args, runtime_args)
    #with open(opbase + psep + 'result.txt', 'w') as f:
    #    f.write(print_args(dataset_args, model_args, runtime_args))


    ''' Dataset '''
    print('Loading datasets...')
    train_dataset, validation_dataset = get_dataset(args)
    print('-- train_dataset.size = {}\n-- validation_dataset.size = {}'.format(
        len(train_dataset), len(validation_dataset)))


    ''' Iterator '''
    train_iterator = chainer.iterators.SerialIterator(
        train_dataset, int(args.batchsize), repeat=True, shuffle=True
    )
    validation_iterator = chainer.iterators.SerialIterator(
        validation_dataset, int(args.batchsize), repeat=True, shuffle=False
    )

    ''' Model '''
    print('Initializing models...')
    model = get_model(args)
    if args.init_model is not None:
        print('Load model from', args.init_model)
        try:
            chainer.serializers.load_npz(args.init_model, model)
        except:
            chainer.serializers.load_hdf5(args.init_model, model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the SegmentNucleus model to the GPU


    ''' Training Phase '''
    with open(opbase + psep + 'TestResult.csv', 'w') as f:
        c = csv.writer(f)
        c.writerow(['Epoch', 'Accuracy', 'Recall', 'Precision', 'Specificity', 'F-measure', 'IoU'])
    with open(opbase + psep + 'TrainResult.csv', 'w') as f:
        c = csv.writer(f)
        c.writerow(['Epoch', 'Accuracy', 'Recall', 'Precision', 'Specificity', 'F-measure', 'IoU'])

    mean_image = None
    if args.model == 'NSN' or '3DUNet':
        trainer = NSNTrainer(
            model=model,
            epoch=args.epoch,
            patchsize=eval(args.patch_size),
            batchsize=args.batchsize,
            gpu=args.gpu,
            opbase=opbase,
            mean_image=mean_image,
            opt_method=args.optimizer
        )
    elif args.model == 'NDN':
        trainer = NDNTrainer(
            model=model,
            epoch=args.epoch,
            patchsize=eval(args.patch_size),
            batchsize=args.batchsize,
            gpu=args.gpu,
            opbase=opbase,
            mean_image=mean_image,
            opt_method=args.optimizer,
            delv=3,
            r_thr=10
        )
    train_eval, test_eval, best_score = trainer.training((train_iterator, validation_iterator))



    # images = {}
    # xlist, ylist = [], []
    # dlist = os.listdir(args.indir)
    # try:
    #     dlist.pop(dlist.index('.DS_Store'))
    # except:
    #     pass
    # dlist = np.sort(dlist)
    # for dl in dlist:
    #     xlist.append('ori' + dl)
    #     ylist.append('ans' + dl)
    #     images['ori' + dl] = io.imread(args.indir + psep + dl + psep + 'image.tif')
    #     images['ans' + dl] = io.imread(args.indir + psep + dl + psep + 'segmentation_gt.tif')
    #
    # # Pre-Processing 1 (Interpolation)
    # for dl in dlist:
    #     ip_size = (int(images['ori' + dl].shape[0] * args.resolution_z), int(images['ori' + dl].shape[1] * args.resolution_y), int(images['ori' + dl].shape[2] * args.resolution_x))
    #     images['ori' + dl] = tr.resize(images['ori' + dl], ip_size, order = 1, preserve_range = True)
    #     images['ans' + dl] = tr.resize(images['ans' + dl], ip_size, order = 1, preserve_range = True)
    #
    # # Pre-Processing 2 (Dataset Augmentation)
    # orilist = copy.deepcopy(ylist)
    # if args.augmentation:
    #     expxlist, expylist = [], []
    #     lz, ly, lx = images[xlist[0]].shape
    #     for t in range(len(xlist)):
    #         aug_ori = util.dataAugmentation(images[xlist[t]], rot=False, flip=True)
    #         aug_ans = util.dataAugmentation(images[ylist[t]], rot=False, flip=True)
    #         for i in range(len(aug_ori)):
    #             ori = aug_ori[i]
    #             ans = aug_ans[i]
    #             listname_ori = xlist[t] + '_r' + str(i)
    #             listname_ans = ylist[t] + '_r' + str(i)
    #             images[listname_ori] = copy.deepcopy(ori.astype(np.float32))
    #             images[listname_ans] = copy.deepcopy(ans.astype(np.float32))
    #             xlist.append(listname_ori)
    #             ylist.append(listname_ans)
    #             expxlist.append(listname_ori)
    #             expylist.append(listname_ans)
    #     print('Augmentation Complete!')
    #
    # # Pre-Processing 3 (Image Scaling)
    # if args.scaling:
    #     for t in xlist:
    #         smax = np.max(images[t])
    #         smin = np.min(images[t])
    #         images[t] = images[t].astype(np.float32)
    #         images[t] = (images[t] - smin) / (smax - smin)
    #
    # # GT Processing
    # for t in ylist:
    #     images[t] = (images[t] > 0) * 1
    #     images[t] = images[t].astype(np.int32)
    #
    # # Pre-Processing 4 (Image Padding)
    # lz, ly, lx = images[ylist[0]].shape
    # for t in xlist:
    #     images[t] = util.mirrorExtensionImage(images[t], int(patchsize/2))[0:args.paddingsize, 0:args.paddingsize, 0:args.paddingsize]
    # for t in ylist:
    #     images[t] = util.mirrorExtensionImage(images[t], int(patchsize/2))[0:args.paddingsize, 0:args.paddingsize, 0:args.paddingsize]
    #
    # # StrageIndex
    # # [t, x, y, z]
    # # t is index of ylist, otherwise start point of coordinates, stride is patchsize
    # idx, exidx = [], []
    # for t in range(len(ylist)):
    #     for z in range(0, lz, patchsize):
    #         for y in range(0, ly, patchsize):
    #             for x in range(0, lx, patchsize):
    #                 if t < len(orilist):  # original image index
    #                     idx.append([t, x, y, z])
    #                 else:  # expansion image index
    #                     exidx.append([t, x, y, z])
    # print('Strage Index Complete!')
    #
    # # Validation of Training and Test Datasets
    # N_all = len(idx)
    # random.shuffle(idx)
    # crossIdx = []
    # for k in range(args.crossvalidation):
    #     crossIdx.append(idx[int(N_all * k / args.crossvalidation) : int(N_all * (k+1) / args.crossvalidation)])


    # Number of Original Data Size
    # N_train = len(crossIdx[0] * (args.crossvalidation-1))
    # N_test = len(crossIdx[0])

    # Output Data Properties
    # print('[Training Data Properties]')
    # print('number of Train Data : ' + str(N_train))
    # print('[Test Data Properties]')
    # print('number of Test Data : ' + str(N_test))
    # with open(opbase + psep + 'result.txt', 'w') as f:
    #     f.write('python ' + ' '.join(argvs) + '\n')
    #     f.write('[Hyperparameter of Learning Properties]\n')
    #     f.write('Patch Size: {}×{}×{}\n'.format(patchsize, patchsize, patchsize))
    #     f.write('Output Directory: {}\n'.format(opbase))
    #     if args.scaling:
    #         f.write('Pre-Processing (Scaling Image): True\n')
    #     else:
    #         f.write('Pre-Processing (Scaling Image): False\n')
    #     if args.normalization:
    #         f.write('Pre-Processing (Normalization Image): True\n')
    #     else:
    #         f.write('Pre-Processing (Normalization Image): False\n')
    #     if args.augmentation:
    #         f.write('Pre-Processing (Augmentation Image): True\n')
    #     else:
    #         f.write('Pre-Processing (Augmentation Image): False\n')
    #     if args.classweight:
    #         f.write('Class Weight: True\n')
    #     else:
    #         f.write('Class Weight: False\n')
    #     f.write('GPU: {}\n'.format(args.gpu))
    #     f.write('[Dataset Properties]\n')
    #     f.write('number of Original Train Data : {}\n'.format(N_train))
    #     f.write('number of Original Test Data : {}\n'.format(N_test))
    #     f.write('x resolution : {}\n'.format(args.resolution_x))
    #     f.write('y resolution : {}\n'.format(args.resolution_y))
    #     f.write('z resolution : {}\n'.format(args.resolution_z))
    # with open(opbase + psep + 'TestResult.csv', 'w') as f:
    #     c = csv.writer(f)
    #     c.writerow(['Epoch', 'Accuracy', 'Recall', 'Precision', 'Specificity', 'F-measure', 'IoU'])
    # with open(opbase + psep + 'TrainResult.csv', 'w') as f:
    #     c = csv.writer(f)
    #     c.writerow(['Epoch', 'Accuracy', 'Recall', 'Precision', 'Specificity', 'F-measure', 'IoU'])

    # bestScore = []
    # for k in range(args.crossvalidation):
    #     print('==========================================')
    #     print('Cross Validation: ' + str(k + 1))
    #
    #     testIdx = crossIdx[k]
    #     trainIdx = []
    #     for kk in range(k, k+args.crossvalidation-1):
    #         ni = kk - args.crossvalidation + 1
    #         trainIdx += crossIdx[ni]
    #         if args.augmentation:
    #             for kkk in range(len(crossIdx[ni])):  # Add Augmentation Data
    #                 trainIdx += [exidx[crossIdx[ni][kkk][0] * 3]]
    #                 trainIdx += [exidx[crossIdx[ni][kkk][0] * 3 + 1]]
    #                 trainIdx += [exidx[crossIdx[ni][kkk][0] * 3 + 2]]
    #
    #     # Pre-Processing 5 (Mean Normalization)
    #     if args.normalization:
    #         sumImg = 0
    #         for t in trainIdx:
    #             images[xlist[t[0]]] = images[xlist[t[0]]].astype(np.float32)
    #             sumImg += images[xlist[t[0]]]
    #         mean_image = copy.deepcopy(sumImg / len(trainIdx))
    #         print('Normalization method : Mean Norm')
    #         io.imsave(opbase + psep + 'meanImg' + str(k+1) + '.tif', mean_image)
    #     else:
    #         mean_image = None
    #         print('Normalization method : Not Adapted')
    #
    #     with open (opbase + psep + 'result.txt', 'a') as f:
    #         f.write('=========================================\n')
    #         f.write('Cross Validation: {}\n'.format(k + 1))
    #         f.write('Number of Train Data: {}\n'.format(len(trainIdx)))
    #         f.write('Number of Test Data: {}\n'.format(len(testIdx)))
    #         f.write('Validation Data: {}\n'.format(testIdx))
    #     print('Number of Train Data: ' + str(len(trainIdx)))
    #     print('Number of Test Data: ' + str(len(testIdx)))
    #
    #     if args.classweight:
    #         obj, bg = 0, 0
    #         for n in range(len(trainIdx)):
    #             y_patch = images[ylist[trainIdx[n][0]]][ trainIdx[n][3]:trainIdx[n][3]+patchsize, trainIdx[n][2]:trainIdx[n][2]+patchsize, trainIdx[n][1]:trainIdx[n][1]+patchsize ]
    #             bg += np.unique(y_patch, return_counts=True)[1][0]
    #             obj += np.unique(y_patch, return_counts=True)[1][1]
    #         bg_w  = float(bg + obj) / bg
    #         obj_w = float(bg + obj) / obj
    #         print('bg_w: ' + str(bg_w))
    #         print('obj_w: ' + str(obj_w))
    #         class_weight = np.array([bg_w, obj_w]).astype(np.float32)
    #     else:
    #         class_weight = np.array([1, 1]).astype(np.float32)
    #
    #     if args.gpu >= 0:
    #         class_weight = cuda.to_gpu(class_weight)
    #
    #     # Create Model
    #     if args.opt_method == 'SGD':
    #         nsn = Model_L2(class_weight=class_weight, n_class=2, init_channel=16,
    #                        kernel_size=3, pool_size=2, ap_factor=2, gpu=args.gpu)
    #     elif args.opt_method == 'Adam':
    #         nsn = Model_L2(class_weight=class_weight, n_class=2, init_channel=16,
    #                        kernel_size=3, pool_size=2, ap_factor=2, gpu=args.gpu)
    #     else:
    #         print('Warning!! Specify Choices Optimizer Method (Adam or SGD)')
    #         sys.exit()
    #
    #     if args.gpu >= 0:
    #         cuda.get_device(args.gpu).use()  # Make a specified GPU current
    #         nsn.to_gpu()  # Copy the SegmentNucleus model to the GPU
    #
    #     stime = time.time()
    #     trainer = NSNTrainer(images=images, model=nsn, epoch=args.epoch, patchsize=patchsize,
    #                          batchsize=args.batchsize, gpu=args.gpu, opbase=opbase,
    #                          mean_image=mean_image, opt_method=args.opt_method)
    #     train_eval, test_eval, bs = trainer.NSNTraining(trainIdx, testIdx, xlist, ylist, k)
    #     bestScore.append(bs)
    #     etime = time.time()
    #     print(etime - stime)
    #
    # AveAcc, AveRec, AvePre, AveSpe, AveFme, AveIoU = 0, 0, 0, 0, 0, 0
    # for k in range(args.crossvalidation):
    #     AveAcc += bestScore[k][0]
    #     AveRec += bestScore[k][1]
    #     AvePre += bestScore[k][2]
    #     AveSpe += bestScore[k][3]
    #     AveFme += bestScore[k][4]
    #     AveIoU += bestScore[k][5]
    # with open(opbase + psep + 'result.txt', 'a') as f:
    #     f.write('======================================\n')
    #     f.write('Average Best Accuracy={}\n'.format(AveAcc / args.crossvalidation))
    #     f.write('Average Best Recall={}\n'.format(AveRec / args.crossvalidation))
    #     f.write('Average Best Precision={}\n'.format(AvePre / args.crossvalidation))
    #     f.write('Average Best Specificity={}\n'.format(AveSpe / args.crossvalidation))
    #     f.write('Average Best Fmeasure={}\n'.format(AveFme / args.crossvalidation))
    #     f.write('Average Best IoU={}\n'.format(AveIoU / args.crossvalidation))

    end_time = time.time()
    process_time = end_time - start_time
    print('Elapsed time is (sec) {}'.format(process_time))
    with open(opbase + psep + 'result.txt', 'a') as f:
        f.write('======================================\n')
        f.write('Elapsed time is (sec) {} \n'.format(process_time))

if __name__ == '__main__':
    main()
