# -*- coding: utf-8 -*-

import os
import numpy as np
from skimage import io
import chainer


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
        import skimage.io as io
        image = io.imread(path)
    else:
        raise ValueError('invalid --input_type : {}'.format(arr_type))

    return image.astype(np.int32)


def crop_pair_3d(
        image1,
        image2,
        crop_size=(96, 96, 96),
        nonzero_image1_thr=0.1,
        nonzero_image2_thr=0.1,
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
        cropped_image1 = image1[top:bottom, left:right, front:rear]
        cropped_image2 = image2[top:bottom, left:right, front:rear]
        # get nonzero ratio
        nonzero_image1_ratio = np.nonzero(cropped_image1)[0].size / cropped_image1.size
        nonzero_image2_ratio = np.nonzero(cropped_image2)[0].size / cropped_image2.size

        # rotate {image_A, image_B}
        if augmentation:
            aug_flag = random.randint(0, 3)
            image1_aug = np.zeros((lz, ly, lx))
            for z in range(z_len):
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
        ):
        self.root_path = root_path
        self.split_list = split_list
        self.model = model
        self.arr_type = arr_type
        self.normalization = normalization
        self.augmentation = augmentation
        self.scaling = scaling
        with open(split_list, 'r') as f:
            self.img_path = f.read().split()

    def __len__(self):
        return len(self.img_path)

    def _get_image(self, i):
        image = read_img(os.path.join(self.root_path, 'images_raw', self.img_path[i]), self.arr_type)
        if self.scaling:
            return image.astype(np.float32) / image.max()
        else:
            return image.astype(np.float32)

    def _get_label(self, i):
        if self.model == 'NSN':
            label = read_img(os.path.join(self.root_path, 'images_nsn', self.img_path[i]), self.arr_type)
        elif self.model == 'NDN':
            label = read_img(os.path.join(self.root_path, 'images_ndn', self.img_path[i]), self.arr_type)
        else:
            print('Warning: select model')
            sys.exit()
        return label / label.max()

    def get_example(self, i):
        x, t = crop_pair_3d(self._get_image(i), self._get_label(i), crop_size=self)
        return np.expand_dims(x.astype(np.float32), axis=0), np.expand_dims(y.astype(np.int32), axis=0)
