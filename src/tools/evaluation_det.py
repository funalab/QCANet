# -*- coding: utf-8 -*-

'''
# Input
Input (center images) : center detection images of gray scale
'''

import cv2
import csv
import sys
import time
import random
import copy
import math
import os
import os.path as pt
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import morphology
from skimage.morphology import watershed
from scipy import ndimage
from argparse import ArgumentParser


plt.style.use('ggplot')
starttime = time.time()

#===============Argument Parser=====================

ap = ArgumentParser(description='python analysis_withWS.py')
ap.add_argument('--input', '-i', nargs='?', default='../SegmentNuclei/CenterImages', help='Specify input files directory CenterImages (gray scale)')
ap.add_argument('--outdir', '-o', nargs='?', default='EvalDecResult', help='Specify output files directory for create segmentation image and save model file')
ap.add_argument('--radius', '-r', type=int, default=10, help='GT radius')
ap.add_argument('--roi',  type=int, default=2, help='Specify ROI GT')
ap.add_argument('--labeling4', action='store_true', help='Specify Labeling Flag (Gray Scale Image)')
ap.add_argument('--labeling8', action='store_true', help='Specify Labeling Flag (Gray Scale Image)')
ap.add_argument('--delete', '-dc', type=int, default=0, help='Specify Delete voxel size less than of center estimation area')

args = ap.parse_args()
opbase = args.outdir
argvs = sys.argv
sys.setrecursionlimit(200000) # 10000 is an example, try with different values

#===============File Write Parameter===============

# File Format
fform = 'tif'
# Path Separator
psep = '/'
if (args.outdir[len(opbase) - 1] == psep):
    opbase = opbase[:len(opbase) - 1]
if not (args.outdir[0] == psep):
    if (args.outdir.find('./') == -1):
        opbase = './' + opbase
# Create Opbase
t = time.ctime().split(' ')
if t.count('') == 1:
    t.pop(t.index(''))
opbase = opbase + '_' + t[1] + t[2] + t[0] + '_' + t[4] + '_' + t[3].split(':')[0] + t[3].split(':')[1] + t[3].split(':')[2]
# Output Directory の有無を Check
if not (pt.exists(opbase)):
    os.mkdir(opbase)
    print 'Output Directory not exist! Create...'
print 'Output Directory:', opbase


# Read Image
# imagePathes = map(lambda a:os.path.join(args.input,a),os.listdir(args.input))
# try:
#     imagePathes.pop(imagePathes.index(args.input + '.DS_Store'))
# except:
#     pass
# imagePathes = np.sort(imagePathes)
# images = np.array(map(lambda x: skimage.imread(x), imagePathes)).astype(np.uint8)


# Read GT
filename = 'GT/10minGroundTruth/CSVfile/test' + str(args.roi) + '.csv'
f = open(filename, 'r')
dataReader = csv.reader(f)
l = []
for i in dataReader:
    l.append(i)
GTCount = []
GTcenter, tmp = [], []
tp = 1
for i in range(len(l)):
    if l[i][3] == str(tp+1):
        tp += 1
        GTcenter.append(tmp)
        tmp = []
    tmp.append([float(l[i][0]), float(l[i][1]), float(l[i][2])])
GTcenter.append(tmp)
f.close()

thr = args.radius

numGT = len(l)
numPR = 0
num_delv = args.delete


filename = opbase + psep + 'result.txt'
f = open(filename, 'w')
f.write('python ' + ' '.join(argvs) + '\n')
f.write('===========================================' + '\n')
f.write('ROI: {}'.format(args.roi)  + '\n')
f.write('radius of threshold: {}'.format(args.radius)  + '\n')
f.write('number of delete voxel size: {}'.format(args.delete)  + '\n')
f.close()


PRcenter = []
filename = opbase + psep + 'center.csv'
f = open(filename, 'w')
c = csv.writer(f)
# Read Image Center

dlist = os.listdir(args.input)
try:
    dlist.pop(dlist.index('.DS_Store'))
except:
    pass

for num in range(len(dlist)):
    image = io.imread(os.path.join(args.input, 'labimg_t{}.tif'.format(num+1)))
    z, y, x = np.shape(image)
    if args.labeling4:
        marker = morphology.label(image, neighbors=4)
    elif args.labeling8:
        marker = morphology.label(image, neighbors=8)
    else:
        marker = copy.deepcopy(image)
    mask_size = np.unique(marker, return_counts=True)[1] < (num_delv+1)
    remove_voxel = mask_size[marker]
    marker[remove_voxel] = 0
    label = np.unique(marker)
    marker = np.searchsorted(label, marker)
    numPR += np.max(marker)
    tmp = np.zeros((np.max(marker), 3))
    count = np.zeros(np.max(marker)).reshape(np.max(marker), 1)
    for i in range(z):
        for j in range(y):
            for k in range(x):
                if marker[i][j][k] > 0:
                    tmp[marker[i][j][k]-1][0] += k
                    tmp[marker[i][j][k]-1][1] += j
                    tmp[marker[i][j][k]-1][2] += i
                    count[marker[i][j][k]-1] += 1
    tmp /= count
    c.writerow(tmp)
    PRcenter.append(tmp)
f.close()

filename = opbase + psep + 'result.txt'
f = open(filename, 'a')
f.write('number of GT: {}'.format(numGT)  + '\n')
f.write('number of Predict: {}'.format(numPR)  + '\n')
f.write('===========================================' + '\n')
f.close()


def search_list(node, used, idx):
    if len(node) == idx:
        return 0
    else:
        tmp = []
        for i in range(len(node[idx])):
            if used[node[idx][i]] == 0:
                used[node[idx][i]] += 1
                tmp.append(search_list(node, used, idx+1) + 1)
                used[node[idx][i]] -= 1
            else:
                tmp.append(search_list(node, used, idx+1))
        return np.max(tmp)


TP = 0
for tp in range(len(GTcenter)):
    pare = []
    for gn in range(len(GTcenter[tp])):
        tmp = 0
        chaild = []
        for pn in range(len(PRcenter[tp])):
            if np.sum((GTcenter[tp][gn] - PRcenter[tp][pn])**2) < thr**2:
                chaild.append(pn)
                tmp += 1
        if tmp > 0:
            pare.append(chaild)
    used = np.zeros(len(PRcenter[tp]))
    TP += search_list(pare, used, 0)

FP = numPR - TP
FN = numGT - TP
Sen = TP / float(TP + FN)
Pre = TP / float(TP + FP)
Fmeasure = 2 * Pre * Sen / (Pre + Sen)
print 'Sensitivity: ' + str(Sen)
print 'Precision: ' + str(Pre)
print 'F-measure: ' + str(Fmeasure)

filename = opbase + psep + 'result.txt'
f = open(filename, 'a')
f.write('Sensitivity: {}'.format(Sen) + '\n')
f.write('Precision: {}'.format(Pre) + '\n')
f.write('F-measure: {}'.format(Fmeasure) + '\n')
f.close()
