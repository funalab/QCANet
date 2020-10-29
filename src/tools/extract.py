# -*- coding: utf-8 -*-

import csv
import sys
import time
import random
import copy
import math
import os
sys.path.append(os.getcwd())
import os.path as pt
import numpy as np
from skimage import io
from skimage import morphology
from skimage import measure
from skimage.morphology import watershed
from scipy import ndimage
from argparse import ArgumentParser
from src.lib.utils import createOpbase
from src.tools.graph_draw import GraphDraw

def main():
    starttime = time.time()
    ap = ArgumentParser(description='python extract.py')
    ap.add_argument('--indir', '-i', nargs='?', default='images/example_output', help='Specify input files directory SegmentationImages')
    ap.add_argument('--outdir', '-o', nargs='?', default='results/result_extract', help='Specify output files directory for create segmentation image and save model file')
    ap.add_argument('--roi', '-r', type=int, default=0, help='Specify ROI GT')
    ap.add_argument('--labeling4', action='store_true', help='Specify Labeling Flag (Gray Scale Image)')
    ap.add_argument('--labeling8', action='store_true', help='Specify Labeling Flag (Gray Scale Image)')
    ap.add_argument('--time_slice', '-t', type=int, default=10, help='Specify time slice of time-series data (min)')

    args = ap.parse_args()
    opbase = args.outdir
    argvs = sys.argv
    psep = '/'
    opbase = createOpbase(args.outdir)

    with open(opbase + psep + 'result.txt', 'w') as f:
        f.write('python ' + ' '.join(argvs) + '\n')

    # Each Criteria
    cnt_num = []
    sum_vol, sum_area = [], []
    mean_vol, mean_area, std_vol, std_area = [], [], [], []
    cent_x, cent_y, cent_z = [], [], []

    with open(opbase + psep + 'criteria.csv', 'w') as f:
        c = csv.writer(f)
        c.writerow(['time point', 'Count', 'Sum Volume', 'Mean Volume', 'SD Volume', 'Sum Area', 'Mean Area', 'SD Area', 'Centroid_x', 'Centroid_y', 'Centroid_z'])

    kernel = np.ones((3,3,3),np.uint8)
    dlist = os.listdir(args.indir)
    try:
        dlist.pop(dlist.index('.DS_Store'))
    except:
        pass
    for l in range(len(dlist)):
        img = io.imread(os.path.join(args.indir, dlist[l]))
        # img = io.imread(os.path.join(args.indir, 'ws_t{0:03d}.tif'.format(l+1)))
        # img = io.imread(os.path.join(args.indir, 'segimg_{0:03d}.tif'.format(l+1)))
        # img = io.imread(os.path.join(args.indir, 'segimg_t{}.tif'.format(l+1)))
        if args.labeling4:
            img = morphology.label(img, neighbors=4)
        elif args.labeling8:
            img = morphology.label(img, neighbors=8)
        else:
            pass

        # Number
        nuclei_num = np.max(img)
        cnt_num.append(nuclei_num)
        # Volume
        nuclei_vol = np.unique(img, return_counts=True)[1][1:]
        sum_vol.append(np.sum(nuclei_vol))
        mean_vol.append(np.mean(nuclei_vol))
        std_vol.append(np.std(nuclei_vol))
        # Surface Area
        img_area = np.zeros((np.shape(img)))
        for n in range(1, nuclei_num+1):
            img_bin = np.array((img < n+1) * (img > n-1)).astype(np.uint8)
            img_ero = img_bin - morphology.erosion(img_bin, selem=kernel)
            img_area += img_ero * n
        nuclei_area = np.unique(img_area, return_counts=True)[1][1:]
        sum_area.append(np.sum(nuclei_area))
        mean_area.append(np.mean(nuclei_area))
        std_area.append(np.std(nuclei_area))
        # Centroid Coodinates
        props = measure.regionprops(img)
        coordinates = [[float(p.centroid[0]), float(p.centroid[1]), float(p.centroid[2])] for p in props]
        x, y, z = [], [], []
        for p in range(len(coordinates)):
            x.append(coordinates[p][2])
            y.append(coordinates[p][1])
            z.append(coordinates[p][0])
        cent_x.append(x)
        cent_y.append(y)
        cent_z.append(z)

        with open(opbase + psep + 'criteria.csv', 'a') as f:
            c = csv.writer(f)
            c.writerow([l + 1, nuclei_num, np.sum(nuclei_vol), np.mean(nuclei_vol), np.std(nuclei_vol), np.sum(nuclei_area), np.mean(nuclei_area), np.std(nuclei_area), x, y, z])

        with open(opbase + psep + 'result.txt', 'a') as f:
            f.write('=============Time Point : ' + str(l + 1) + '=============\n')
            f.write('Count : {}\n'.format(nuclei_num))
            f.write('Sum Volume : {}\n'.format(np.sum(nuclei_vol)))
            f.write('Mean Volume : {}\n'.format(np.mean(nuclei_vol)))
            f.write('Standard Deviation Volume : {}\n'.format(np.std(nuclei_vol)))
            f.write('Sum Area : {}\n'.format(np.sum(nuclei_area)))
            f.write('Mean Area : {}\n'.format(np.mean(nuclei_area)))
            f.write('Standard Deviation Area : {}\n'.format(np.std(nuclei_area)))
            f.write('Centroid : {}\n'.format(coordinates))

    # Time Scale
    # dt = args.time_slice / float(60 * 24)
    # Time = [dt*x for x in range(len(cnt_num))]
    # figbase = opbase + psep + 'figs_criteria'
    # os.mkdir(figbase)
    # gd = GraphDraw(figbase, args.roi)
    # gd.graph_draw_number(Time, cnt_num)
    # gd.graph_draw_volume(Time, sum_vol, mean_vol, std_vol)
    # gd.graph_draw_surface(Time, sum_area, mean_area, std_area)
    # gd.graph_draw_centroid(cent_x, cent_y, cent_z)


if __name__ == '__main__':
    main()
