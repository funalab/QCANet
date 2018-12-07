# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import copy
import time
import skimage.io as io
from os import path as pt
from chainer import serializers

class Utils():

    def __init__(self, patchsize=128, batchsize=1):
        self.patchsize = patchsize
        self.batchsize = batchsize
        self.psep = '/'

        
    def loadImages(self, path):
        imagePathes = map(lambda a:os.path.join(path,a),os.listdir(path))
        try:
            imagePathes.pop(imagePathes.index(path + '/.DS_Store'))
        except:
            pass
        imagePathes = np.sort(imagePathes) # list on Linux
        images = np.array(map(lambda x: io.imread(x), imagePathes))
        return images

    
    def oneSideExtensionImage(self, images, patchsize):
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

    
    def patch_crop(self, x_data, y_data, idx, n, patchsize):
        x_patch = copy.deepcopy( np.array( x_data[ idx[n][2]:idx[n][2]+patchsize, idx[n][1]:idx[n][1]+patchsize, idx[n][0]:idx[n][0]+patchsize ] ).reshape(1, patchsize, patchsize, patchsize).astype(np.float32) )   # np.shape(idx_O[n][0]) [0] : x座標, n : 何番目の座標か
        y_patch = copy.deepcopy( np.array(y_data[ idx[n][2] ][ idx[n][1] ][ idx[n][0] ]).reshape(1).astype(np.int32) )
        return x_patch, y_patch

    
    def crossSplit(self, objData, bgData, objLabel, bgLabel, k_cross, n):
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
    def dataAugmentation(self, image, rot=True, flip=True):
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
    def createOpbase(self, opbase):
        if (opbase[len(opbase) - 1] == self.psep):
            opbase = opbase[:len(opbase) - 1]
        if not (opbase[0] == self.psep):
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
        self.opbase = opbase
        return opbase
        

    def loadModel(self, model_path, model):
        try:
            serializers.load_npz(model_path, model)
            print('Loading Model: {}'.format(model_path))
            with open(self.opbase + self.psep + 'result.txt', 'a') as f:
                f.write('Loading Model: {}\n'.format(model_path))
        except:
            serializers.load_hdf5(model_path, model)
            print('Loading Model: {}'.format(model_path))
            with open(self.opbase + self.psep + 'result.txt', 'a') as f:
                f.write('Loading Model: {}\n'.format(model_path))


    # Oneside Mirroring Padding in Image-wise Processing
    def mirrorExtensionImage(self, image, length=10):
        lz, ly, lx = image.shape
        exbox = np.pad(image, pad_width=length, mode='reflect')
        return copy.deepcopy(exbox[length:lz+length*2, length:ly+length*2, length:lx+length*2])


    def splitImage(self, image, stride):
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
