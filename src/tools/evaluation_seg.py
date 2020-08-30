# -*- coding: utf-8 -*-

import json
import numpy as np
import copy
import skimage.io as io
import os
import sys
from skimage import morphology
from skimage.morphology import watershed
from skimage import transform
from skimage import measure
from scipy import ndimage
from datetime import datetime
import pytz
from argparse import ArgumentParser
sys.path.append(os.getcwd())

class Evaluator():
    def __init__(self):
        pass

    def mucov(self, y, y_ans):
        sum_iou = 0
        label_list_y = np.unique(y)[1:]
        print('candidate label (pre): {}'.format(label_list_y))
        for i in label_list_y:
            y_mask = np.array((y == i) * 1).astype(np.int8)
            rp = measure.regionprops(y_mask)[0]
            bbox = rp.bbox
            y_ans_roi = y_ans[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
            label_list = np.unique(y_ans_roi)[1:]
            best_iou, best_thr = 0, 0
            for j in label_list:
                y_ans_mask = np.array((y_ans == j) * 1).astype(np.int8)
                iou, thr = self.iou(y_mask, y_ans_mask)
                if best_iou <= iou:
                    best_iou = iou
                    best_thr = np.max([thr, best_thr])
            print('c{0:03} best IoU in MUCov: {1}'.format(i, best_iou))
            if best_thr > 0.5:
                sum_iou += best_iou
            else:
                sum_iou += 0.0
        return sum_iou / len(label_list_y)

    def seg(self, y, y_ans):
        sum_iou = 0
        label_list_y_ans = np.unique(y_ans)[1:]
        print('candidate label (gt): {}'.format(label_list_y_ans))
        for i in label_list_y_ans:
            y_ans_mask = np.array((y_ans == i) * 1).astype(np.int8)
            rp = measure.regionprops(y_ans_mask)[0]
            bbox = rp.bbox
            y_roi = y[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
            label_list = np.unique(y_roi)[1:]
            best_iou, best_thr = 0, 0
            for j in label_list:
                y_mask = np.array((y == j) * 1).astype(np.int8)
                iou, thr = self.iou(y_mask, y_ans_mask)
                if best_iou <= iou:
                    best_iou = iou
                    best_thr = np.max([thr, best_thr])
            print('c{0:03} best IoU in SEG: {1}'.format(i, best_iou))
            if best_thr > 0.5:
                sum_iou += best_iou
            else:
                sum_iou += 0.0
        return sum_iou / len(label_list_y_ans)

    def iou(self, pred, gt):
        pred, gt = pred.astype(np.int8), gt.astype(np.int8)
        countListPos = copy.deepcopy(pred + gt)
        countListNeg = copy.deepcopy(pred - gt)
        TP = len(np.where(countListPos.reshape(countListPos.size)==2)[0])
        FP = len(np.where(countListNeg.reshape(countListNeg.size)==1)[0])
        FN = len(np.where(countListNeg.reshape(countListNeg.size)==-1)[0])
        try:
            iou = TP / float(TP + FP + FN)
            thr = TP / float(TP + FN)
        except:
            iou = 0
            thr = 0
        return iou, thr

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
    ap.add_argument('--gtdir', '-g', nargs='?', default='../InstanceSegmentationResult/ROI2_Feb25Sun_2018_001348/WatershedSegmentationImages', help='Specify ground truth files')
    ap.add_argument('--outdir', '-o', nargs='?', default='evaluation_of_instance_segmentation', help='Specify output files directory for create figures')
    ap.add_argument('--labeling4', action='store_true', help='Specify Labeling Flag (Gray Scale Image)')
    ap.add_argument('--labeling8', action='store_true', help='Specify Labeling Flag (Gray Scale Image)')
    args = ap.parse_args()
    argvs = sys.argv
    psep = '/'
    current_datetime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    opbase = args.outdir + '_' + current_datetime
    os.makedirs(opbase, exist_ok=True)

    with open(opbase + psep + 'result.txt', 'w') as f:
        f.write('python ' + ' '.join(argvs) + '\n')

    in_list = np.sort(os.listdir(args.indir))
    gt_list = np.sort(os.listdir(args.gtdir))
    assert len(in_list) == len(gt_list)

    evaluation = Evaluator()
    all_iou, all_mucov, all_seg = [], [], []

    with open(opbase + psep + 'result.txt', 'a') as f:
        for t in range(len(in_list)):
            print('#################')
            print('Input file name: {}'.format(in_list[t]))
            print('GT file name   : {}'.format(gt_list[t]))
            print('#################')
            pre = io.imread(os.path.join(args.indir, in_list[t]))
            if args.labeling4:
                pre = morphology.label(pre, neighbors=4)
            elif args.labeling8:
                pre = morphology.label(pre, neighbors=8)
            else:
                pass
            labels = np.unique(pre)
            pre = np.searchsorted(labels, pre).astype(np.uint16)
            pre_bin = np.array((pre > 0) * 1).astype(np.int8)

            gt = io.imread(os.path.join(args.gtdir, gt_list[t])).astype(np.uint16)
            gt_bin = np.array((gt > 0) * 1).astype(np.int8)

            iou, _ = evaluation.iou(pre_bin, gt_bin)
            mucov = evaluation.mucov(pre, gt)
            seg = evaluation.seg(pre, gt)
            all_iou.append(iou)
            all_mucov.append(mucov)
            all_seg.append(seg)
            print('#################')
            print('file name: {}'.format(in_list[t]))
            print('tp{}: IoU={}'.format(t, iou))
            print('tp{}: MUCov={}'.format(t, mucov))
            print('tp{}: SEG={}'.format(t, seg))
            print('#################')
            print('')
            f.write('file name: {}\n'.format(in_list[t]))
            f.write('tp{}: IoU={}\n'.format(t, iou))
            f.write('tp{}: MUCov={}\n'.format(t, mucov))
            f.write('tp{}: SEG={}\n\n'.format(t, seg))
        mean_iou = np.mean(all_iou)
        std_iou = np.std(all_iou)
        mean_mucov = np.mean(all_mucov)
        std_mucov = np.std(all_mucov)
        mean_seg = np.mean(all_seg)
        std_seg = np.std(all_seg)
        print('')
        print('Mean IoU: {}'.format(mean_iou))
        print('Std IoU: {}'.format(std_iou))
        print('Mean MUCov: {}'.format(mean_mucov))
        print('Std MUCov: {}'.format(std_mucov))
        print('Mean SEG: {}'.format(mean_seg))
        print('Std SEG: {}'.format(std_seg))
        f.write('\n')
        f.write('Mean IoU={}\n'.format(mean_iou))
        f.write('Std IoU={}\n'.format(std_iou))
        f.write('Mean MUCov={}\n'.format(mean_mucov))
        f.write('Std MUCov={}\n'.format(std_mucov))
        f.write('Mean SEG={}\n'.format(mean_seg))
        f.write('Std SEG={}\n'.format(std_seg))
    metrics = {'IoU': mean_iou, 'MUCov': mean_mucov, 'SEG': mean_seg,
               'IoU_std': std_iou, 'MUCov_std': std_mucov, 'SEG_std': std_seg}
    with open(os.path.join(opbase, 'result.json'), 'w') as f:
        json.dump(metrics, f)
