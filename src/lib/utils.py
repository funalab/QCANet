# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import copy
import time
import argparse
import skimage.io as io
from os import path as pt
from distutils.util import strtobool
from chainer import serializers

from src.lib.dataset import PreprocessedDataset
from src.lib.model import Model_L2, Model_L3, Model_L4

def get_dataset(args):
    train_dataset = PreprocessedDataset(
        root_path=args.root_path,
        split_list=args.split_list_train,
        train=True,
        model=args.model,
        arr_type=args.input_format,
        normalization=args.normalization,
        augmentation=args.augmentation,
        scaling=args.scaling,
        resolution=eval(args.resolution),
        crop_size=eval(args.patchsize)
    )
    validation_dataset = PreprocessedDataset(
        root_path=args.root_path,
        split_list=args.split_list_validation,
        train=False,
        model=args.model,
        arr_type=args.input_format,
        normalization=args.normalization,
        augmentation=False,
        scaling=args.scaling,
        resolution=eval(args.resolution),
        crop_size=eval(args.patchsize)
    )
    return train_dataset, validation_dataset


def get_model(args):
    if args.model == 'NSN':
        model = Model_L2(
            class_weight=args.class_weight,
            n_class=args.ch_out,
            init_channel=args.ch_base,
            kernel_size=3,
            pool_size=2,
            ap_factor=2,
            gpu=args.gpu
        )
    elif args.model == 'NDN':
        model = Model_L4(
            class_weight=args.class_weight,
            n_class=args.ch_out,
            init_channel=args.ch_base,
            kernel_size=3,
            pool_size=2,
            ap_factor=2,
            gpu=args.gpu
        )

    return model


def create_dataset_parser(remaining_argv, **conf_dict):
    input_formats = ['tif', 'npz', 'png', 'jpg']
    image_dtypes = ['uint8', 'uint16', 'int32', 'float32']

    parser = argparse.ArgumentParser(description='Dataset Parameters', add_help=False)
    parser.set_defaults(**conf_dict)
    parser.add_argument('--root_path', type=str,
                        help='/path/to/dataset')
    parser.add_argument('--split_list', type=str,
                        help='/path/to/split_list.txt (list of {train,validation} files)')
    parser.add_argument('--input_format', choices=input_formats,
                        help='Input format {"numpy"(2D,3D), "image"(2D), "nii"(3D)}')
    parser.add_argument('--image_dtype', choices=image_dtypes,
                        help="Data type of input image array ('uint8', 'int32', 'float32' etc)")
    parser.add_argument('--resolution',
                        help='Specify microscope resolution of x-, y-, z-axis [um/pixel] (default = (1.0, 1.0, 2.18)')
    args, remaining_argv = parser.parse_known_args(remaining_argv)

    return parser, args, remaining_argv


def create_model_parser(remaining_argv, **conf_dict):
    model_list = ['NSN', 'NDN']

    parser = argparse.ArgumentParser(description='Model Parameters', add_help=False)
    parser.set_defaults(**conf_dict)
    parser.add_argument('--model', choices=model_list,
                        help='Model name {"NSN", "NDN"}')
    parser.add_argument('--ndim', type=int,
                        help='Dimensions of input / convolution kernel')
    parser.add_argument('--lossfun', type=str,
                        help='Specify Loss function')
    parser.add_argument('--init_model',
                        help='Initialize the segmentor from given file')
    parser.add_argument('--ch_in', type=int,
                        help='Number of channels for input (image)')
    parser.add_argument('--ch_base', type=int,
                        help='Number of base channels (to control total memory and segmentor performance)')
    parser.add_argument('--ch_out', type=int,
                        help='Number of channels for output (label)')
    args, remaining_argv = parser.parse_known_args(remaining_argv)

    return parser, args, remaining_argv


def create_runtime_parser(remaining_argv, **conf_dict):
    optimizer_list = ['SGD', 'MomentumSGD', 'Adam']

    parser = argparse.ArgumentParser(description='Runtime Parameters', add_help=False)
    parser.set_defaults(**conf_dict)
    parser.add_argument('--save_dir', type=str,
                        help='Root directory which trained files are saved')
    parser.add_argument('--batchsize', '-B', type=int,
                        help='Learning minibatch size')
    parser.add_argument('--val_batchsize', '-b', type=int,
                        help='Validation minibatch size')
    parser.add_argument('--epoch', '-E', type=int,
                        help='Specify number of sweeps over the dataset to train')
    parser.add_argument('--optimizer', choices=optimizer_list,
                        help='Optimizer name {"MomentumSGD", "SGD", "Adam"}')
    parser.add_argument('--init_lr', type=float,
                        help='Initial learning rate for discriminator ("alpha" in case of Adam)')
    parser.add_argument('--momentum', type=float,
                        help='Momentum (used in MomentumSGD)')
    parser.add_argument('--lr_reduction_ratio', type=float,
                        help='Learning rate reduction ratio')
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay for optimizer scheduling')
    parser.add_argument('--gpu', type=int,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--patch_size',
                        help='Specify one side voxel size of ROI')
    parser.add_argument('--padding_size',
                        help='Specify image size after padding')
    parser.add_argument('--normalization', type=strtobool,
                        help='If True, mean normalization')
    parser.add_argument('--augmentation', type=strtobool,
                        help='If True, data augmentation (rotation)')
    parser.add_argument('--class_weight', type=strtobool,
                        help='If True, use class weight with softmax corss entropy')
    parser.add_argument('--scaling', type=strtobool,
                        help='If True, image-wise scaling image')
    args, remaining_argv = parser.parse_known_args(remaining_argv)

    return parser, args, remaining_argv


def print_args(dataset_args, model_args, runtime_args):
    """ Export config file
    Args:
        dataset_args    : Argument Namespace object for loading dataset
        model_args        : Argument Namespace object for model parameters
        runtime_args    : Argument Namespace object for runtime parameters
    """
    dataset_dict = {k: v for k, v in vars(dataset_args).items() if v is not None}
    model_dict = {k: v for k, v in vars(model_args).items() if v is not None}
    runtime_dict = {k: v for k, v in vars(runtime_args).items() if v is not None}
    print('============================')
    print('[Dataset]')
    for k, v in dataset_dict.items():
        print('%s = %s' % (k, v))
    print('\n[Model]')
    for k, v in model_dict.items():
        print('%s = %s' % (k, v))
    print('\n[Runtime]')
    for k, v in runtime_dict.items():
        print('%s = %s' % (k, v))
    print('============================\n')


def loadImages(path):
    imagePathes = map(lambda a:os.path.join(path,a),os.listdir(path))
    try:
        imagePathes.pop(imagePathes.index(path + '/.DS_Store'))
    except:
        pass
    imagePathes = np.sort(imagePathes) # list on Linux
    images = np.array(map(lambda x: io.imread(x), imagePathes))
    return images


def oneSideExtensionImage(images, patchsize):
    lz, ly, lx = images.shape
    if lx % patchsize != 0:
        sx = lx + patchsize - 1
    else:
        sx = lx
    if ly % patchsize != 0:
        sy = ly + patchsize - 1
    else:
        sy = ly
    if lz % patchsize != 0:
        sz = lz + patchsize - 1
    else:
        sz = lz
    exbox = np.zeros((sz, sy, sx))
    exbox += images.min()
    exbox[0:lz, 0:ly, 0:lx] = images
    return copy.deepcopy(exbox)


def patch_crop(x_data, y_data, idx, n, patchsize):
    x_patch = copy.deepcopy( np.array( x_data[ idx[n][2]:idx[n][2]+patchsize, idx[n][1]:idx[n][1]+patchsize, idx[n][0]:idx[n][0]+patchsize ] ).reshape(1, patchsize, patchsize, patchsize).astype(np.float32) )   # np.shape(idx_O[n][0]) [0] : x座標, n : 何番目の座標か
    y_patch = copy.deepcopy( np.array(y_data[ idx[n][2] ][ idx[n][1] ][ idx[n][0] ]).reshape(1).astype(np.int32) )
    return x_patch, y_patch


def crossSplit(objData, bgData, objLabel, bgLabel, k_cross, n):
    objx, bgx, objy, bgy = [], [], [], []
    N = len(objData)
    for i in range(k_cross):
        objx.append(objData[i*N/k_cross:(i+1)*N/k_cross])
        objy.append(objLabel[i*N/k_cross:(i+1)*N/k_cross])
        bgx.append(bgData[i*N/k_cross:(i+1)*N/k_cross])
        bgy.append(bgLabel[i*N/k_cross:(i+1)*N/k_cross])
    x_train, y_train, x_test, y_test = [], [], [], []
    for i in range(k_cross):
        if i == n:
            x_test.append(copy.deepcopy(objx[i]))
            x_test.append(copy.deepcopy(bgx[i]))
            y_test.append(copy.deepcopy(objy[i]))
            y_test.append(copy.deepcopy(bgy[i]))
        else:
            x_train.append(copy.deepcopy(objx[i]))
            x_train.append(copy.deepcopy(bgx[i]))
            y_train.append(copy.deepcopy(objy[i]))
            y_train.append(copy.deepcopy(bgy[i]))
    x_train = np.array(x_train).reshape(2*N*(k_cross-1)/k_cross, args.batchsize, patchsize, patchsize, patchsize)
    y_train = np.array(y_train).reshape(2*N*(k_cross-1)/k_cross, args.batchsize)
    x_test = np.array(x_test).reshape(2*N/k_cross, args.batchsize, patchsize, patchsize, patchsize)
    y_test = np.array(y_test).reshape(2*N/k_cross, args.batchsize)
    return copy.deepcopy(x_train), copy.deepcopy(x_test), copy.deepcopy(y_train), copy.deepcopy(y_test)


# Rotation & Flip for Data Augmentation (fix z-axis)
def dataAugmentation(image, rot=True, flip=True):
    lz, ly, lx = image.shape
    if rot and flip:
        flip = np.zeros((lz, ly, lx))
        rot90 = np.zeros((lz, lx, ly))
        rot90_f = np.zeros((lz, lx, ly))
        rot180 = np.zeros((lz, ly, lx))
        rot180_f = np.zeros((lz, ly, lx))
        rot270 = np.zeros((lz, lx, ly))
        rot270_f = np.zeros((lz, lx, ly))
        for z in range(lz):
            flip[z] = np.flip(image[z], 1)
            rot90[z] = np.rot90(image[z])
            rot90_f[z] = np.rot90(flip[z])
            rot180[z] = np.rot90(rot90[z])
            rot180_f[z] = np.rot90(rot90_f[z])
            rot270[z] = np.rot90(rot180[z])
            rot270_f[z] = np.rot90(rot180_f[z])
        aug_images = [flip, rot90, rot90_f, rot180, rot180_f, rot270, rot270_f]
    elif flip:
        flip_v = np.zeros((lz, ly, lx))
        flip_h = np.zeros((lz, ly, lx))
        flip_vh = np.zeros((lz, ly, lx))
        for z in range(lz):
            flip_v[z] = np.flip(image[z], 0)
            flip_h[z] = np.flip(image[z], 1)
            flip_vh[z] = np.flip(flip_h[z], 0)
        aug_images = [flip_v, flip_h, flip_vh]
    elif rot:
        rot90 = np.zeros((lz, lx, ly))
        rot180 = np.zeros((lz, ly, lx))
        rot270 = np.zeros((lz, lx, ly))
        for z in range(lz):
            rot90[z] = np.rot90(image[z])
            rot180[z] = np.rot90(rot90[z])
            rot270[z] = np.rot90(rot180[z])
        aug_images = [rot90, rot180, rot270]
    else:
        print('No Augmentation!')
        aug_images = []
    return aug_images


# Create Opbase for Output Directory
def createOpbase(opbase):
    if (opbase[len(opbase) - 1] == '/'):
        opbase = opbase[:len(opbase) - 1]
    if not (opbase[0] == '/'):
        if (opbase.find('./') == -1):
            opbase = './' + opbase
    t = time.ctime().split(' ')
    if t.count('') == 1:
        t.pop(t.index(''))
    opbase = opbase + '_' + t[1] + t[2] + t[0] + '_' + t[4] + '_' + t[3].split(':')[0] + t[3].split(':')[1] + t[3].split(':')[2]
    if not (pt.exists(opbase)):
        os.mkdir(opbase)
        print('Output Directory not exist! Create...')
    print('Output Directory: {}'.format(opbase))
    return opbase


def loadModel(model_path, model, opbase):
    try:
        serializers.load_hdf5(model_path, model)
        print('Loading Model: {}'.format(model_path))
        with open(os.path.join(opbase, 'result.txt'), 'a') as f:
            f.write('Loading Model: {}\n'.format(model_path))
    except:
        print('Not Found: {}'.format(model_path))
        print('Usage : Input File Path of Model (ex ./hoge.model)')
        sys.exit()


# Oneside Mirroring Padding in Image-wise Processing
def mirrorExtensionImage(image, length=10):
    lz, ly, lx = image.shape
    exbox = np.pad(image, pad_width=length, mode='reflect')
    return copy.deepcopy(exbox[length:lz+length*2, length:ly+length*2, length:lx+length*2])


def splitImage(image, stride):
    lz, ly, lx = np.shape(image)
    num_split = int(((lx - stride) / stride) * ((ly - stride) / stride) * ((lz - stride) / stride))
    s_image = np.zeros((num_split, self.patchsize, self.patchsize, self.patchsize))
    num = 0
    for z in range(0, lz-stride, stride):
        for y in range(0, ly-stride, stride):
            for x in range(0, lx-stride, stride):
                s_image[num] = image[z:z+self.patchsize, y:y+self.patchsize, x:x+self.patchsize]
                num += 1
    return copy.deepcopy(s_image)
