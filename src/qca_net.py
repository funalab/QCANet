# -*- coding: utf-8 -*-

import chainer
from chainer import cuda, serializers

import sys
import time
import os
import os.path as pt
import numpy as np
from scipy import ndimage
from skimage import io
from skimage import morphology
from skimage.morphology import watershed
from argparse import ArgumentParser
from lib.model import Model_L2, Model_L3, Model_L4
from lib.utils import Utils
from test_nsn import TestNSN
from test_ndn import TestNDN


def main():
    
    start_time = time.time()
    ap = ArgumentParser(description='python qca_net.py')
    ap.add_argument('--indir', '-i', nargs='?', default='../images/example_input', help='Specify input files directory : Phase contrast cell images in gray scale')
    ap.add_argument('--outdir', '-o', nargs='?', default='result_qca_net', help='Specify output files directory for create segmentation, labeling & classification images')
    ap.add_argument('--model_nsn', '-ms', nargs='?', default='../models/p128/learned_nsn.model', help='Specify loading file path of Learned Segmentation Model')
    ap.add_argument('--model_ndn', '-md', nargs='?', default='../models/p128/learned_ndn.model', help='Specify loading file path of Learned Detection Model')
    ap.add_argument('--gpu', '-g', type=int, default=-1, help='Specify GPU ID (negative value indicates CPU)')
    ap.add_argument('--patchsize_seg', '-ps', type=int, default=128, help='Specify pixel size of Segmentation Patch')
    ap.add_argument('--patchsize_det', '-pd', type=int, default=128, help='Specify pixel size of Detection Patch')
    ap.add_argument('--stride_seg', '-ss', type=int, default=64, help='Specify pixel size of Segmentation Stride')
    ap.add_argument('--stride_det', '-sd', type=int, default=64, help='Specify pixel size of Detection Stride')
    ap.add_argument('--delete', '-d', type=int, default=0, help='Specify Pixel Size of Delete Region for Cell Detection Model')
    ap.add_argument('--scaling_seg', action='store_true', help='Specify Image-wise Scaling Flag in Detection Phase')
    ap.add_argument('--scaling_det', action='store_true', help='Specify Image-wise Scaling Flag in Classification Phase')
    ap.add_argument('--resolution_x', '-x', type=float, default=1.0, help='Specify microscope resolution of x axis (default=1.0)')
    ap.add_argument('--resolution_y', '-y', type=float, default=1.0, help='Specify microscope resolution of y axis (default=1.0)')
    ap.add_argument('--resolution_z', '-z', type=float, default=2.18, help='Specify microscope resolution of z axis (default=2.18)')

    args = ap.parse_args()
    argvs = sys.argv
    util = Utils()
    psep = '/'
    
    opbase = util.createOpbase(args.outdir)
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
    class_weight = np.array([1, 1]).astype(np.float32)
    if args.gpu >= 0:
        class_weight = cuda.to_gpu(class_weight)
    # NSN_SGD
    nsn = Model_L2(class_weight=class_weight, n_class=2, init_channel=16,
                   kernel_size=3, pool_size=2, ap_factor=2, gpu=args.gpu)
    # NDN_Adam
    ndn = Model_L4(class_weight=class_weight, n_class=2, init_channel=12,
                   kernel_size=5, pool_size=2, ap_factor=2, gpu=args.gpu)
    # Def-NDN_Adam
    # ndn = Model_L3(class_weight=class_weight, n_class=2, init_channel=8,
    #                kernel_size=3, pool_size=2, ap_factor=2, gpu=args.gpu)

    # Load Model
    if not args.model_nsn == '0':
        util.loadModel(args.model_nsn, nsn)
    if not args.model_ndn == '0':
        util.loadModel(args.model_ndn, ndn)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()  # Make a specified GPU current
        nsn.to_gpu()  # Copy the SegmentNucleus model to the GPU
        ndn.to_gpu()
        
    dlist = os.listdir(args.indir)
    with open(opbase + psep + 'result.txt', 'a') as f:
        try:
            dlist.pop(dlist.index('.DS_Store'))
        except:
            pass
        dlist = np.sort(dlist)
        test_nsn = TestNSN(model=nsn, patchsize=args.patchsize_seg, stride=args.stride_seg,
                           resolution=(args.resolution_x, args.resolution_y, args.resolution_z),
                           scaling=args.scaling_seg, opbase=opbase, gpu=args.gpu)
        test_ndn = TestNDN(model=ndn, patchsize=args.patchsize_det, stride=args.stride_det,
                           resolution=(args.resolution_x, args.resolution_y, args.resolution_z),
                           scaling=args.scaling_det, delv=args.delete,
                           opbase=opbase, gpu=args.gpu)
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
            filename = opbase + psep + wsbase + psep + 'ws_{}.tif'.format(image_path[image_path.rfind('/')+1:image_path.rfind('.')])
            io.imsave(filename, wsimage.astype(np.uint8))

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
    print('QCA Net Completed Process!')

if __name__ == '__main__':        
    main()
