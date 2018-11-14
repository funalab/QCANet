# -*- coding: utf-8 -*-

import numpy as np
import copy
import skimage.io as io
import os
import sys
from skimage import morphology
from skimage.morphology import watershed
from skimage import transform
from scipy import ndimage
from argparse import ArgumentParser
sys.path.append(os.getcwd())

from src.lib.utils import createOpbase


class Evaluator():
    def __init__(self):
        pass


    def mucov(self, y, y_ans):
        sum_iou = 0
        for i in range(1, y.max()+1):
            mask_y = np.array((y == i) * 1).astype(np.uint8)
            best_iou = 0
            for j in range(1, y_ans.max()+1):
                mask_y_ans = np.array((y_ans == j) * 1).astype(np.uint8)
                best_iou = np.max([self.iou(mask_y, mask_y_ans), best_iou])
            print('best IoU: {}'.format(best_iou))
            sum_iou += best_iou
        return sum_iou / y.max()


    def iou(self, pred, gt):
        countListPos = copy.deepcopy(pred + gt)
        countListNeg = copy.deepcopy(pred - gt)
        TP = len(np.where(countListPos.reshape(countListPos.size)==2)[0])
        FP = len(np.where(countListNeg.reshape(countListNeg.size)==1)[0])
        FN = len(np.where(countListNeg.reshape(countListNeg.size)==-1)[0])
        try:
            iou = TP / float(TP + FP + FN)
        except:
            iou = 0
        return iou


    def qca_watershed(self, seg, det):
        distance = ndimage.distance_transform_edt(seg)
        markers = morphology.label(det, neighbors=4)
        wsimg = watershed(-distance, markers, mask=seg)
        return wsimg


    def detection_evaluator(self, PR, GT, thr):
        numPR, numGT = len(PR), len(GT)
        pare = []
        for gn in range(numGT):
            tmp = 0
            chaild = []
            for pn in range(numPR):
                if np.sum((GT[gn] - PR[pn])**2) < thr**2:
                    chaild.append(pn)
                    tmp += 1
            if tmp > 0:
                pare.append(chaild)
        used = np.zeros(numPR)
        TP = self._search_list(pare, used, 0)

        evals = {}
        FP = numPR - TP
        FN = numGT - TP
        try:
            evals['recall'] = TP / float(TP + FN)
        except:
            evals['recall'] = 0.0
        try:
            evals['precision'] = TP / float(TP + FP)
        except:
            evals['precision'] = 0.0
        try:
            evals['F-measure'] = 2 * evals['recall'] * evals['precision'] / (evals['recall'] + evals['precision'])
        except:
            evals['F-measure'] = 0.0
        try:
            evals['IoU'] = TP / float(TP + FP + FN)
        except:
            evals['IoU'] = 0.0
        return evals


    def _search_list(self, node, used, idx):
        if len(node) == idx:
            return 0
        else:
            tmp = []
            for i in range(len(node[idx])):
                if used[node[idx][i]] == 0:
                    used[node[idx][i]] += 1
                    tmp.append(self._search_list(node, used, idx+1) + 1)
                    used[node[idx][i]] -= 1
                else:
                    tmp.append(self._search_list(node, used, idx+1))
            return np.max(tmp)



if __name__ == '__main__':
    ap = ArgumentParser(description='python evaluation.py')
    ap.add_argument('--indir', '-i', nargs='?', default='../InstanceSegmentationResult/ROI2_Feb25Sun_2018_001348/WatershedSegmentationImages', help='Specify input files')
    #ap.add_argument('--gtdir', '-g', nargs='?', default='../InstanceSegmentationResult/ROI2_Feb25Sun_2018_001348/WatershedSegmentationImages', help='Specify ground truth files')
    ap.add_argument('--outdir', '-o', nargs='?', default='EvaluationMUCov', help='Specify output files directory for create figures')
    ap.add_argument('--labeling4', action='store_true', help='Specify Labeling Flag (Gray Scale Image)')
    ap.add_argument('--labeling8', action='store_true', help='Specify Labeling Flag (Gray Scale Image)')
    ap.add_argument('--roi', type=int, help='specify ROI number')
    args = ap.parse_args()
    argvs = sys.argv
    psep = '/'
    opbase = createOpbase(args.outdir)

    with open(opbase + psep + 'result.txt', 'w') as f:
        f.write('python ' + ' '.join(argvs) + '\n')

    # t_list = [1, 50, 100, 150, 188, 250, 256, 257, 258, 262, 320, 335, 338, 350, 398, 450, 474, 502]
    # n_list = [2, 2, 2, 2, 3, 4, 5, 7, 7, 8, 9, 13, 14, 16, 20, 32, 40, 50]
    t_list = [1, 51, 101, 151, 201, 251, 301, 351, 401, 451, 501]

    evaluation = Evaluator()
    all_iou, all_mucov = [], []

    with open(opbase + psep + 'result.txt', 'a') as f:
        for t in t_list:
            if args.labeling4:
                seg = io.imread(args.indir + psep + 'segimg_{0:03d}.tif'.format(t))
                #seg = io.imread(args.indir + psep + 'segimg_t{}.tif'.format(t))
                seg = morphology.label(seg, neighbors=4)
            elif args.labeling8:
                seg = io.imread(args.indir + psep + 'segimg_{0:03d}.tif'.format(t))
                #seg = io.imread(args.indir + psep + 'segimg_t{}.tif'.format(t))
                seg = morphology.label(seg, neighbors=8)
            else:
                #seg = io.imread(args.indir + psep + 'ws_{0:03d}.tif'.format(t))
                seg = io.imread(args.indir + psep + 'ws_t{}.tif'.format(t))
            labels = np.unique(seg)
            seg = np.searchsorted(labels, seg)
            seg_bin = np.array((seg > 0) * 1).astype(np.uint8)

            filename = 'datasets/images_qcanet_tif/ROI{}_t{}.tif'.format(args.roi, t)
            gt = io.imread(filename)
            gt_bin = np.array((gt > 0) * 1).astype(np.uint8)

            iou = evaluation.iou(seg_bin, gt_bin)
            mucov = evaluation.mucov(seg, gt)
            all_iou.append(iou)
            all_mucov.append(mucov)
            print('tp{}: IoU={}'.format(t, iou))
            print('tp{}: MUCov={}'.format(t, mucov))
            f.write('tp{}: IoU={}\n'.format(t, iou))
            f.write('tp{}: MUCov={}\n'.format(t, mucov))
        mean_iou = np.mean(all_iou)
        std_iou = np.std(all_iou)
        mean_mucov = np.mean(all_mucov)
        std_mucov = np.std(all_mucov)
        print('Mean IoU: {}'.format(mean_iou))
        print('Std IoU: {}'.format(std_iou))
        print('Mean MUCov: {}'.format(mean_mucov))
        print('Std MUCov: {}'.format(std_mucov))
        f.write('Mean IoU={}\n'.format(mean_iou))
        f.write('Std IoU={}\n'.format(std_iou))
        f.write('Mean MUCov={}\n'.format(mean_mucov))
        f.write('Std MUCov={}\n'.format(std_mucov))
