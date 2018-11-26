# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
from skimage import io
import chainer
from skimage import io
from skimage import transform as tr

def read_img(path, arr_type='npz'):
    """ read image array from path
    Args:
        path (str)          : path to directory which images are stored.
        arr_type (str)      : type of reading file {'npz','jpg','png','tif'}
    Returns:
        image (np.ndarray)  : image array
    """
    if arr_type == 'npz':
        image = np.load(path)['arr_0']
    elif arr_type in ('png', 'jpg'):
        image = imread(path, mode='L')
    elif arr_type == 'tif':
        image = io.imread(path)
    else:
        raise ValueError('invalid --input_type : {}'.format(arr_type))

    return image.astype(np.int32)


def crop_pair_2d(
        image1,
        image2=None,
        crop_size=(512, 512),
        nonzero_image1_thr=0.0,
        nonzero_image2_thr=0.0,
        nb_crop=1,
        augmentation=True
):
    """ 2d {image, label} patches are cropped from array.
    Args:
        image1 (np.ndarray)         : Input 2d image array from 1st domain
        image2 (np.ndarray)         : Input 2d image array from 2nd domain
        crop_size ((int, int))      : Crop patch from array
        nonzero_image1_thr (float)  : Crop if nonzero pixel ratio is higher than threshold
        nonzero_image2_thr (float)  : Crop if nonzero pixel ratio is higher than threshold
        nb_crop (int)               : Number of cropping patches at once
    Returns:
        if nb_crop = 1:
            cropped_image1 (np.ndarray)  : cropped 2d image array
            cropped_image2 (np.ndarray)  : cropped 2d label array
        if nb_crop > 1:
            cropped_images1 (list)       : cropped 2d image arrays
            cropped_images2 (list)       : cropped 2d label arrays
    """
    y_len, x_len = image1.shape
    assert y_len >= crop_size[0]
    assert x_len >= crop_size[1]
    cropped_images1 = []
    cropped_images2 = []

    while 1:
        # get cropping position (image)
        top = random.randint(0, x_len-crop_size[1]-1) if x_len > crop_size[1] else 0
        left = random.randint(0, y_len-crop_size[0]-1) if y_len > crop_size[0] else 0
        bottom = top + crop_size[1]
        right = left + crop_size[0]

        # crop {image_A, image_B}
        cropped_image1 = image1[left:right, top:bottom]
        cropped_image2 = image2[left:right, top:bottom]
        # get nonzero ratio
        nonzero_image1_ratio = np.nonzero(cropped_image1)[0].size / cropped_image1.size
        nonzero_image2_ratio = np.nonzero(cropped_image2)[0].size / cropped_image2.size

        # rotate {image_A, image_B}
        if augmentation:
            aug_flag = random.randint(0, 3)
            cropped_image1 = np.rot90(cropped_image1, k=aug_flag)
            cropped_image2 = np.rot90(cropped_image2, k=aug_flag)

        # break loop
        if (nonzero_image1_ratio >= nonzero_image1_thr) \
                and (nonzero_image2_ratio >= nonzero_image2_thr):
            if nb_crop == 1:
                return cropped_image1, cropped_image2
            elif nb_crop > 1:
                cropped_images1.append(cropped_image1)
                cropped_images2.append(cropped_image2)
                if len(cropped_images1) == nb_crop:
                    return np.array(cropped_images1), np.array(cropped_images2)
            else:
                raise ValueError('invalid value nb_crop :', nb_crop)


def crop_pair_3d(
        image1,
        image2,
        crop_size=(128, 128, 128),
        #nonzero_image1_thr=0.001,
        nonzero_image1_thr=0.0,
        #nonzero_image2_thr=0.001,
        nonzero_image2_thr=0.0,
        nb_crop=1,
        augmentation=True
    ):
    """ 3d {image, label} patches are cropped from array.
    Args:
        image1 (np.ndarray)                  : Input 3d image array from 1st domain
        image2 (np.ndarray)                  : Input 3d label array from 2nd domain
        crop_size ((int, int, int))         : Crop image patch from array randomly
        nonzero_image1_thr (float)           : Crop if nonzero pixel ratio is higher than threshold
        nonzero_image2_thr (float)           : Crop if nonzero pixel ratio is higher than threshold
        nb_crop (int)                       : Number of cropping patches at once
    Returns:
        if nb_crop == 1:
            cropped_image1 (np.ndarray)  : cropped 3d image array
            cropped_image2 (np.ndarray)  : cropped 3d label array
        if nb_crop > 1:
            cropped_images1 (list)       : cropped 3d image arrays
            cropped_images2 (list)       : cropped 3d label arrays
    """
    z_len, y_len, x_len = image1.shape
    #_, x_len, y_len, z_len = image1.shape
    assert x_len >= crop_size[0]
    assert y_len >= crop_size[1]
    assert z_len >= crop_size[2]
    cropped_images1 = []
    cropped_images2 = []

    while 1:
        # get cropping position (image)
        top = random.randint(0, x_len-crop_size[0]-1) if x_len > crop_size[0] else 0
        left = random.randint(0, y_len-crop_size[1]-1) if y_len > crop_size[1] else 0
        front = random.randint(0, z_len-crop_size[2]-1) if z_len > crop_size[2] else 0
        bottom = top + crop_size[0]
        right = left + crop_size[1]
        rear = front + crop_size[2]

        # crop image
        cropped_image1 = image1[front:rear, left:right, top:bottom]
        cropped_image2 = image2[front:rear, left:right, top:bottom]
        # get nonzero ratio
        nonzero_image1_ratio = np.nonzero(cropped_image1)[0].size / float(cropped_image1.size)
        nonzero_image2_ratio = np.nonzero(cropped_image2)[0].size / float(cropped_image2.size)

        # rotate {image_A, image_B}
        if augmentation:
            aug_flag = random.randint(0, 3)
            for z in range(cropped_image1.shape[0]):
                cropped_image1[z] = np.rot90(cropped_image1[z], k=aug_flag)
                cropped_image2[z] = np.rot90(cropped_image2[z], k=aug_flag)

        # break loop
        if (nonzero_image1_ratio >= nonzero_image1_thr) \
                and (nonzero_image2_ratio >= nonzero_image2_thr):
            if nb_crop == 1:
                return cropped_image1, cropped_image2
            elif nb_crop > 1:
                cropped_images1.append(cropped_image1)
                cropped_images2.append(cropped_image2)
                if len(cropped_images1) == nb_crop:
                    return np.array(cropped_images1), np.array(cropped_images2)
            else:
                raise ValueError('invalid value nb_crop :', nb_crop)


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(
        self,
        root_path,
        split_list,
        train=True,
        model='NSN',
        arr_type='npz',
        normalization=False,
        augmentation=True,
        scaling=True,
        resolution=eval('(1.0, 1.0, 2.18)'),
        crop_size=eval('(96, 96, 96)'),
        ndim=3
        ):
        self.root_path = root_path
        self.split_list = split_list
        self.model = model
        self.arr_type = arr_type
        self.normalization = normalization
        self.augmentation = augmentation
        self.scaling = scaling
        self.resolution = resolution
        self.crop_size = crop_size
        self.train = train
        self.ndim = ndim

        with open(split_list, 'r') as f:
            self.img_path = f.read().split()

    def __len__(self):
        return len(self.img_path)

    def _get_image_2d(self, i):
        image = read_img(os.path.join(self.root_path, 'images_raw', self.img_path[i]), self.arr_type)
        ip_size = (int(image.shape[0] * self.resolution[1]), int(image.shape[1] * self.resolution[1]))
        image = tr.resize(image.astype(np.float32), ip_size, order=1, preserve_range=True)
        pad_size = np.max(np.array(self.crop_size) - np.array(ip_size))
        if pad_size > 0:
            image = np.pad(image, pad_width=pad_size, mode='reflect')
        if self.scaling:
            image = (image - image.min()) / (image.max() - image.min())
        return image.astype(np.float32)

    def _get_label_2d(self, i):
        if self.model == 'NSN' or self.model == '3DUNet':
            label = read_img(os.path.join(self.root_path, 'images_nsn', self.img_path[i]), self.arr_type)
        elif self.model == 'NDN':
            label = read_img(os.path.join(self.root_path, 'images_ndn', self.img_path[i]), self.arr_type)
        else:
            print('Warning: select model')
            sys.exit()
        ip_size = (int(label.shape[0] * self.resolution[1]), int(label.shape[1] * self.resolution[1]))
        label = (tr.resize(label, ip_size, order=1, preserve_range=True) > 0) * 1
        pad_size = np.max(np.array(self.crop_size) - np.array(ip_size))
        if pad_size > 0:
            label = np.pad(label, pad_width=pad_size, mode='reflect')
        return label.astype(np.int32)

    def _get_image_3d(self, i):
        image = read_img(os.path.join(self.root_path, 'images_raw', self.img_path[i]), self.arr_type)
        ip_size = (int(image.shape[0] * self.resolution[2]), int(image.shape[1] * self.resolution[1]), int(image.shape[2] * self.resolution[0]))
        image = tr.resize(image.astype(np.float32), ip_size, order=1, preserve_range=True)
        pad_size = np.max(np.array(self.crop_size) - np.array(ip_size))
        if pad_size > 0:
            image = np.pad(image, pad_width=pad_size, mode='reflect')
        if self.scaling:
            image = (image - image.min()) / (image.max() - image.min())
        return image.astype(np.float32)

    def _get_label_3d(self, i):
        if self.model == 'NSN' or self.model == '3DUNet':
            label = read_img(os.path.join(self.root_path, 'images_nsn', self.img_path[i]), self.arr_type)
        elif self.model == 'NDN':
            label = read_img(os.path.join(self.root_path, 'images_ndn', self.img_path[i]), self.arr_type)
        else:
            print('Warning: select model')
            sys.exit()
        ip_size = (int(label.shape[0] * self.resolution[2]), int(label.shape[1] * self.resolution[1]), int(label.shape[2] * self.resolution[0]))
        label = (tr.resize(label, ip_size, order=1, preserve_range=True) > 0) * 1
        pad_size = np.max(np.array(self.crop_size) - np.array(ip_size))
        if pad_size > 0:
            label = np.pad(label, pad_width=pad_size, mode='reflect')
        return label.astype(np.int32)

    def get_example(self, i):
        if self.ndim == 2:
            image = self._get_image_2d(i)
            label = self._get_label_2d(i)
        elif self.ndim == 3:
            image = self._get_image_3d(i)
            label = self._get_label_3d(i)
        else:
            print('Error: ndim must be 2 or 3 dimensions')
        if self.train:
            if self.ndim == 2:
                x, t = crop_pair_2d(image, label, crop_size=self.crop_size)
            elif self.ndim == 3:
                x, t = crop_pair_3d(image, label, crop_size=self.crop_size)
            return np.expand_dims(np.expand_dims(x.astype(np.float32), axis=0), axis=0), np.expand_dims(t.astype(np.int32), axis=0)
        else:
            return np.expand_dims(np.expand_dims(image.astype(np.float32), axis=0), axis=0), np.expand_dims(label.astype(np.int32), axis=0)
