# -*- coding: utf-8 -*-

import csv
import sys
import time
import random
import copy
import math
import os
import numpy as np
import configparser
from argparse import ArgumentParser
from os import path as pt
from numpy.core.records import array
import skimage.io as io
from skimage import morphology
from skimage.morphology import watershed
from scipy import ndimage
import torch

sys.path.append(os.getcwd())
from src.lib.trainer import NSNTrainer, NDNTrainer
from src.lib.utils import createOpbase
from src.lib.utils import create_dataset_parser, create_model_parser, create_runtime_parser
from src.lib.utils import print_args
from src.lib.utils import get_model
from src.tools.test_nsn import TestNSN
from src.tools.test_ndn import TestNDN
from src.lib.model import Model_L2, Model_L3, Model_L4

def main():

    start_time = time.time()
    ap = ArgumentParser(description='python qca_net.py')
    ap.add_argument('--indir', '-i', nargs='?', default='images/example_input', help='Specify input files directory : Phase contrast cell images in gray scale')
    ap.add_argument('--outdir', '-o', nargs='?', default='results/result_qcanet', help='Specify output files directory for create segmentation, labeling & classification images')
    ap.add_argument('--model_nsn', '-ms', nargs='?', default='models/learned_nsn.npz', help='Specify loading file path of Learned Segmentation Model')
    ap.add_argument('--model_ndn', '-md', nargs='?', default='models/learned_ndn.npz', help='Specify loading file path of Learned Detection Model')
    ap.add_argument('--gpu', '-g', type=int, default=-1, help='Specify GPU ID (negative value indicates CPU)')
    ap.add_argument('--patchsize_seg', '-ps', type=int, default=96, help='Specify pixel size of Segmentation Patch')
    ap.add_argument('--patchsize_det', '-pd', type=int, default=96, help='Specify pixel size of Detection Patch')
    ap.add_argument('--stride_seg', '-ss', type=int, default=48, help='Specify pixel size of Segmentation Stride')
    ap.add_argument('--stride_det', '-sd', type=int, default=48, help='Specify pixel size of Detection Stride')
    ap.add_argument('--delete', '-d', type=int, default=0, help='Specify Pixel Size of Delete Region for Cell Detection Model')
    ap.add_argument('--scaling_seg', action='store_true', help='Specify Image-wise Scaling Flag in Detection Phase')
    ap.add_argument('--scaling_det', action='store_true', help='Specify Image-wise Scaling Flag in Classification Phase')
    ap.add_argument('--resolution_x', '-x', type=float, default=1.0, help='Specify microscope resolution of x axis (default=1.0)')
    ap.add_argument('--resolution_y', '-y', type=float, default=1.0, help='Specify microscope resolution of y axis (default=1.0)')
    ap.add_argument('--resolution_z', '-z', type=float, default=2.18, help='Specify microscope resolution of z axis (default=2.18)')
    ap.add_argument('--ndim', type=int, default=3,
                        help='Dimensions of input / convolution kernel')
    ap.add_argument('--lossfun', type=str, default='softmax_dice_loss',
                        help='Specify Loss function')
    ap.add_argument('--ch_base', type=int, default=16,
                        help='Number of base channels (to control total memory and segmentor performance)')
    # ap.add_argument('--ch_base_ndn', type=int, default=12,
    #                     help='Number of base channels (to control total memory and segmentor performance)')
    ap.add_argument('--ch_out', type=int, default=2,
                        help='Number of channels for output (label)')
    ap.add_argument('--model', default='NSN',
                        help='Specify class weight with softmax corss entropy')



    args = ap.parse_args()
    argvs = sys.argv
    psep = '/'

    opbase = createOpbase(args.outdir)
    wsbase = 'WatershedSegmentationImages'
    if not (pt.exists(opbase + psep + wsbase)):
        os.mkdir(opbase + psep + wsbase)


    print('Patch Size of Segmentation: {}'.format(args.patchsize_seg))
    print('Patch Size of Detection: {}'.format(args.patchsize_det))
    print('Delete Voxel Size of Detection Region: {}'.format(args.delete))
    print('Scaling Image in Segmentation Phase: {}'.format(args.scaling_seg))
    print('Scaling Image in Detection Phase: {}'.format(args.scaling_det))
    with open(opbase + psep + 'result.txt', 'w') as f:
        f.write('python ' + ' '.join(argvs) + '\n')
        f.write('[Properties of parameter]\n')
        f.write('Output Directory: {}\n'.format(opbase))
        f.write('Patch Size of Segmentation: {}\n'.format(args.patchsize_seg))
        f.write('Patch Size of Detection: {}\n'.format(args.patchsize_det))
        f.write('Delete Pixel Size of Detection Region: {}\n'.format(args.delete))
        f.write('Scaling Image in Segmentation Phase: {}\n'.format(args.scaling_seg))
        f.write('Scaling Image in Detection Phase: {}\n'.format(args.scaling_det))


    # Create Model
    print('Initializing models...')
    if args.model_nsn is not None:
        print('Load NSN from', args.model_nsn)
        nsn = torch.load(args.model_nsn)
    else:
        raise ValueError('Specified model path')

    if args.model_ndn is not None:
        print('Load NDN from', args.model_ndn)
        ndn = torch.load(args.model_ndn)
    else:
        raise ValueError('Specified model path')

    nsn = nsn.to(args.device)
    ndn = ndn.to(args.device)

    dlist = os.listdir(args.indir)
    with open(opbase + psep + 'result.txt', 'a') as f:
        try:
            dlist.pop(dlist.index('.DS_Store'))
        except:
            pass
        dlist = np.sort(dlist)
        test_nsn = TestNSN(
            model=nsn,
            patchsize=args.patchsize_seg,
            stride=args.stride_seg,
            resolution=(args.resolution_x, args.resolution_y, args.resolution_z),
            scaling=args.scaling_seg,
            opbase=opbase,
            gpu=args.gpu,
            ndim=args.ndim
            )
        test_ndn = TestNDN(
            model=ndn,
            patchsize=args.patchsize_det,
            stride=args.stride_det,
            resolution=(args.resolution_x, args.resolution_y, args.resolution_z),
            scaling=args.scaling_det,
            delv=args.delete,
            opbase=opbase,
            gpu=args.gpu,
            ndim=args.ndim
            )
        for dl in dlist:
            image_path = args.indir + psep + dl
            print('[{}]'.format(image_path))
            f.write('[{}]\n'.format(image_path))

            ### Segmentation Phase ###
            seg_img = test_nsn.NuclearSegmentation(image_path)

            ### Detection Phase ###
            det_img = test_ndn.NuclearDetection(image_path)

            ### Post-Processing ###
            if det_img.sum() > 0:
                distance = ndimage.distance_transform_edt(seg_img)
                wsimage = watershed(-distance, det_img, mask=seg_img)
            else:
                wsimage = morphology.label(seg_img, neighbors=4)
            labels = np.unique(wsimage)
            wsimage = np.searchsorted(labels, wsimage)
            filename = os.path.join(opbase, wsbase, os.path.basename(image_path)[:os.path.basename(image_path).rfind('.')] + '.tif')
            # filename = opbase + psep + wsbase + psep + 'ws_t{0:03d}.tif'.format(int(image_path[image_path.rfind('/')+1:image_path.rfind('.')]))
            io.imsave(filename, wsimage.astype(np.uint16))

            f.write('Number of Nuclei: {}\n'.format(wsimage.max()))
            volumes = np.unique(wsimage, return_counts=True)[1][1:]
            f.write('Mean Volume of Nuclei: {}\n'.format(volumes.mean()))
            f.write('Volume of Nuclei: {}\n'.format(volumes))

    end_time = time.time()
    etime = end_time - start_time
    with open(opbase + psep + 'result.txt', 'a') as f:
        f.write('======================================\n')
        f.write('Elapsed time is (sec) {} \n'.format(etime))
    print('Elapsed time is (sec) {}'.format(etime))
    print('QCANet Completed Process!')

if __name__ == '__main__':
    main()
