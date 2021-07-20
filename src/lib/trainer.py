# -*- coding: utf-8 -*-

import numpy as np
import copy
import csv
import time
import sys
import skimage.io as io
from skimage import morphology

import chainer
from chainer import Variable, optimizers, cuda, serializers

from src.lib.utils import mirror_extension_image

sys.setrecursionlimit(200000)

class NSNTrainer():

    def __init__(
            self,
            model,
            epoch,
            patchsize,
            batchsize,
            gpu,
            opbase,
            mean_image=None,
            opt_method='Adam',
            ndim=3
    ):
        self.model = model
        self.epoch = epoch
        self.patchsize = patchsize
        self.batchsize = batchsize
        self.gpu = gpu
        self.opbase = opbase
        self.mean_image = mean_image
        self.opt_method = opt_method
        self.criteria = ['Accuracy', 'Recall', 'Precision', 'Specificity', 'F-measure', 'IoU']
        self.ndim = ndim
        self.val_iteration = 1

    def training(self, iterators):
        train_iter, val_iter = iterators

        if self.opt_method == 'Adam':
            #opt_nsn = optimizers.Adam(alpha=0.05767827010227712, beta1=0.9687170166672859,
            #                          beta2=0.9918705323205452, eps=0.03260658847351856)
            opt_nsn = optimizers.Adam()
            opt_nsn.setup(self.model)
            #opt_nsn.add_hook(chainer.optimizer.WeightDecay(0.00000416029939))
            opt_nsn.add_hook(chainer.optimizer.WeightDecay(0.001))

        elif self.opt_method == 'SGD':
            opt_nsn = optimizers.SGD(lr=1.0)
            #opt_nsn = optimizers.SGD(lr=0.01)
            opt_nsn.setup(self.model)
            #opt_nsn.add_hook(chainer.optimizer.WeightDecay(0.001))
            opt_nsn.add_hook(chainer.optimizer.WeightDecay(0.00009))

        elif self.opt_method == 'MomentumSGD':
            opt_nsn = optimizers.MomentumSGD(lr=0.0001, momentum=0.99)
            opt_nsn.setup(self.model)
            #opt_nsn.add_hook(chainer.optimizer.WeightDecay(0.00001))

        train_eval, test_eval = {}, {}
        train_eval['loss'], test_eval['loss'] = [], []
        for cri in self.criteria:
            train_eval[cri] = []
            test_eval[cri] = []
        N_train = train_iter.dataset.__len__()
        N_test = val_iter.dataset.__len__()
        with open(self.opbase + '/result.txt', 'w') as f:
            f.write('N_train: {}\n'.format(N_train))
            f.write('N_test: {}\n'.format(N_test))
        bestAccuracy, bestRecall, bestPrecision, bestSpecificity, bestFmeasure, bestIoU = 0, 0, 0, 0, 0, 0
        bestEpoch = 0

        for epoch in range(1, self.epoch + 1):
            print('[epoch {}]'.format(epoch))
            traeval, train_sum_loss = self._trainer(train_iter, opt_nsn, epoch=epoch)
            train_eval['loss'].append(train_sum_loss / (N_train * self.batchsize))

            if epoch % self.val_iteration == 0:
                teseval, test_sum_loss = self._validater(val_iter, epoch=epoch)
                test_eval['loss'].append(test_sum_loss / (N_test * self.batchsize))
            else:
                teseval = {}
                for cri in self.criteria:
                    teseval[cri] = 0
                    test_eval['loss'].append(0)
                    test_sum_loss = 0

            for cri in self.criteria:
                train_eval[cri].append(traeval[cri])
                test_eval[cri].append(teseval[cri])
            print('train mean loss={}'.format(train_sum_loss / (N_train * self.batchsize)))
            print('train accuracy={}, train recall={}'.format(traeval['Accuracy'], traeval['Recall']))
            print('train precision={}, specificity={}'.format(traeval['Precision'], traeval['Specificity']))
            print('train F-measure={}, IoU={}'.format(traeval['F-measure'], traeval['IoU']))
            print('test mean loss={}'.format(test_sum_loss / (N_test * self.batchsize)))
            print('test accuracy={}, recall={}'.format(teseval['Accuracy'], teseval['Recall']))
            print('test precision={}, specificity={}'.format(teseval['Precision'], teseval['Specificity']))
            print('test F-measure={}, IoU={}'.format(teseval['F-measure'], teseval['IoU']))
            with open(self.opbase + '/result.txt', 'a') as f:
                f.write('========================================\n')
                f.write('[epoch' + str(epoch) + ']\n')
                f.write('train mean loss={}\n'.format(train_sum_loss / (N_train * self.batchsize)))
                f.write('train accuracy={}, train recall={}\n'.format(traeval['Accuracy'], traeval['Recall']))
                f.write('train precision={}, specificity={}\n'.format(traeval['Precision'], traeval['Specificity']))
                f.write('train F-measure={}, IoU={}\n'.format(traeval['F-measure'], traeval['IoU']))
                if epoch % self.val_iteration == 0:
                    f.write('validation mean loss={}\n'.format(test_sum_loss / (N_test * self.batchsize)))
                    f.write('validation accuracy={}, recall={}\n'.format(teseval['Accuracy'], teseval['Recall']))
                    f.write('validation precision={}, specificity={}\n'.format(teseval['Precision'], teseval['Specificity']))
                    f.write('validation F-measure={}, IoU={}\n'.format(teseval['F-measure'], teseval['IoU']))
            with open(self.opbase + '/TrainResult.csv', 'a') as f:
                c = csv.writer(f)
                c.writerow([epoch, traeval['Accuracy'], traeval['Recall'], traeval['Precision'], traeval['Specificity'], traeval['F-measure'], traeval['IoU']])
            with open(self.opbase + '/ValResult.csv', 'a') as f:
                c = csv.writer(f)
                c.writerow([epoch, teseval['Accuracy'], teseval['Recall'], teseval['Precision'], teseval['Specificity'], teseval['F-measure'], teseval['IoU']])

            if epoch == 1:
                pastLoss = train_sum_loss

            if train_sum_loss > pastLoss and self.opt_method == 'SGD':
                learning_rate = opt_nsn.lr * 1.0
                opt_nsn = optimizers.SGD(learning_rate)
                opt_nsn.setup(self.model)
                with open(self.opbase + '/result.txt', 'a') as f:
                    f.write('lr: {}\n'.format(opt_nsn.lr))
                    
            pastLoss = train_sum_loss


            if bestAccuracy <= teseval['Accuracy']:
                bestAccuracy = teseval['Accuracy']
            if bestRecall <= teseval['Recall']:
                bestRecall = teseval['Recall']
                # Save Model
                # model_name = 'NSN_Recall_p' + str(self.patchsize) + '.npz'
                # serializers.save_npz(self.opbase + '/' + model_name, self.model)
            if bestPrecision <= teseval['Precision']:
                bestPrecision = teseval['Precision']
                # Save Model
                # model_name = 'NSN_Precision_p' + str(self.patchsize) + '.npz'
                # serializers.save_npz(self.opbase + '/' + model_name, self.model)
            if bestSpecificity <= teseval['Specificity']:
                bestSpecificity = teseval['Specificity']
            if bestFmeasure <= teseval['F-measure']:
                bestFmeasure = teseval['F-measure']
            if bestIoU <= teseval['IoU']:
                bestIoU = teseval['IoU']
                bestEpoch = epoch
                # Save Model
                if epoch > 0:
                    model_name = 'NSN_IoU_p' + str(self.patchsize) + '.npz'
                    serializers.save_npz(self.opbase + '/' + model_name, self.model)
                else:
                    bestIoU = 0.0

        bestScore = [bestAccuracy, bestRecall, bestPrecision, bestSpecificity, bestFmeasure, bestIoU]
        print('========================================')
        print('Best Epoch : ' + str(bestEpoch))
        print('Best Accuracy : ' + str(bestAccuracy))
        print('Best Recall : ' + str(bestRecall))
        print('Best Precision : ' + str(bestPrecision))
        print('Best Specificity : ' + str(bestSpecificity))
        print('Best F-measure : ' + str(bestFmeasure))
        print('Best IoU : ' + str(bestIoU))
        with open(self.opbase + '/result.txt', 'a') as f:
            f.write('################################################\n')
            f.write('BestAccuracy={}\n'.format(bestAccuracy))
            f.write('BestRecall={}, BestPrecision={}\n'.format(bestRecall, bestPrecision))
            f.write('BestSpecificity={}, BestFmesure={}\n'.format(bestSpecificity, bestFmeasure))
            f.write('BestIoU={}, BestEpoch={}\n'.format(bestIoU, bestEpoch))
            f.write('################################################\n')

        return train_eval, test_eval, bestScore


    def _trainer(self, dataset_iter, opt_nsn, epoch):
        TP, TN, FP, FN = 0, 0, 0, 0
        dataset_iter.reset()
        N = dataset_iter.dataset.__len__()
        sum_loss = 0
        #perm = np.random.permutation(N)
        for num in range(N):
            #n = perm[num]
            if self.mean_image is None:
                x_patch, y_patch = dataset_iter.next()[0]
            else:
                x_patch, y_patch = dataset_iter.next()[0][0] - self.mean_image, dataset_iter.next()[0][1]

            if self.gpu >= 0:
                x_patch = cuda.to_gpu(x_patch)
                y_patch = cuda.to_gpu(y_patch)

            s_loss, s_output = self.model(x=x_patch, t=y_patch, seg=False)
            opt_nsn.target.cleargrads()
            s_loss.backward()
            opt_nsn.update()

            #train_eval['loss'].append(s_loss.data)
            sum_loss += float(cuda.to_cpu(s_loss.data) * self.batchsize)

            if self.gpu >= 0:
                y_patch = cuda.to_cpu(y_patch)
                s_output = cuda.to_cpu(s_output)
            #make pred (0 : background, 1 : object)
            pred = copy.deepcopy((0 < (s_output[0][1] - s_output[0][0])) * 1)
            countListPos = copy.deepcopy(pred.astype(np.int16) + y_patch.astype(np.int16))
            countListNeg = copy.deepcopy(pred.astype(np.int16) - y_patch.astype(np.int16))
            TP += len(np.where(countListPos.reshape(countListPos.size)==2)[0])
            TN += len(np.where(countListPos.reshape(countListPos.size)==0)[0])
            FP += len(np.where(countListNeg.reshape(countListNeg.size)==1)[0])
            FN += len(np.where(countListNeg.reshape(countListNeg.size)==-1)[0])

        evals = self._evaluator(TP, TN, FP, FN)
        return evals, sum_loss


    def _validater(self, dataset_iter, epoch):
        TP, TN, FP, FN = 0, 0, 0, 0
        dataset_iter.reset()
        N = dataset_iter.dataset.__len__()
        sum_loss = 0
        #perm = np.random.permutation(N)
        for num in range(N):
            if self.mean_image is None:
                x_batch, y_batch = dataset_iter.next()[0]
            else:
                x_batch, y_batch = dataset_iter.next()[0][0] - self.mean_image, dataset_iter.next()[0][1]

            if self.ndim == 2:
                im_size = x_batch.shape[2:]
                stride = [int(self.patchsize[0]/2), int(self.patchsize[1]/2)]
                sh = [int(stride[0]/2), int(stride[1]/2)]
            elif self.ndim == 3:
                im_size = x_batch.shape[2:]
                stride = [int(self.patchsize[0]/2), int(self.patchsize[1]/2), int(self.patchsize[2]/2)]
                sh = [int(stride[0]/2), int(stride[1]/2), int(stride[2]/2)]

            ''' calculation for pad size'''
            if np.min(self.patchsize) > np.max(im_size):
                if self.ndim == 2:
                    pad_size = [self.patchsize[0], self.patchsize[1]]
                elif self.ndim == 3:
                    pad_size = [self.patchsize[0], self.patchsize[1], self.patchsize[2]]
            else:
                pad_size = []
                for axis in range(len(im_size)):
                    if (im_size[axis] + 2*sh[axis] - self.patchsize[axis]) % stride[axis] == 0:
                        stride_num = (im_size[axis] + 2*sh[axis] - self.patchsize[axis]) / stride[axis]
                    else:
                        stride_num = (im_size[axis] + 2*sh[axis] - self.patchsize[axis]) / stride[axis] + 1
                    pad_size.append(int(stride[axis] * stride_num + self.patchsize[axis]))

            gt = copy.deepcopy(y_batch)
            pre_img = np.zeros(pad_size)

            if self.ndim == 2:
                x_batch = mirror_extension_image(image=x_batch, ndim=self.ndim, length=int(np.max(self.patchsize)))[:, :, self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1]]
                y_batch = mirror_extension_image(image=y_batch, ndim=self.ndim,  length=int(np.max(self.patchsize)))[:, self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1]]
                for y in range(0, pad_size[0]-self.patchsize[0], stride[0]):
                    for x in range(0, pad_size[1]-self.patchsize[1], stride[1]):
                        x_patch = x_batch[:, :, y:y+self.patchsize[0], x:x+self.patchsize[1]]
                        y_patch = y_batch[:, y:y+self.patchsize[0], x:x+self.patchsize[1]]
                        if self.gpu >= 0:
                            x_patch = cuda.to_gpu(x_patch)
                            y_patch = cuda.to_gpu(y_patch)
                        s_loss, s_output = self.model(x=x_patch, t=y_patch, seg=False)
                        sum_loss += float(cuda.to_cpu(s_loss.data) * self.batchsize)
                        if self.gpu >= 0:
                            s_output = cuda.to_cpu(s_output)
                        pred = copy.deepcopy((0 < (s_output[0][1] - s_output[0][0])) * 255)
                        # Add segmentation image
                        pre_img[y:y+stride[0], x:x+stride[1]] += pred[sh[0]:-sh[0], sh[1]:-sh[1]]
                seg_img = (pre_img > 0) * 1
                seg_img = seg_img[:im_size[0], :im_size[1]]
            elif self.ndim == 3:
                x_batch = mirror_extension_image(image=x_batch, ndim=self.ndim, length=int(np.max(self.patchsize)))[:, :, self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1], self.patchsize[2]-sh[2]:self.patchsize[2]-sh[2]+pad_size[2]]
                y_batch = mirror_extension_image(image=y_batch, ndim=self.ndim, length=int(np.max(self.patchsize)))[:, self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1], self.patchsize[2]-sh[2]:self.patchsize[2]-sh[2]+pad_size[2]]
                for z in range(0, pad_size[0]-self.patchsize[0], stride[0]):
                    for y in range(0, pad_size[1]-self.patchsize[1], stride[1]):
                        for x in range(0, pad_size[2]-self.patchsize[2], stride[2]):
                            x_patch = x_batch[:, :, z:z+self.patchsize[0], y:y+self.patchsize[1], x:x+self.patchsize[2]]
                            y_patch = y_batch[:, z:z+self.patchsize[0], y:y+self.patchsize[1], x:x+self.patchsize[2]]
                            if self.gpu >= 0:
                                x_patch = cuda.to_gpu(x_patch)
                                y_patch = cuda.to_gpu(y_patch)
                            s_loss, s_output = self.model(x=x_patch, t=y_patch, seg=False)
                            sum_loss += float(cuda.to_cpu(s_loss.data) * self.batchsize)
                            if self.gpu >= 0:
                                s_output = cuda.to_cpu(s_output)
                            pred = copy.deepcopy((0 < (s_output[0][1] - s_output[0][0])) * 255)
                            # Add segmentation image
                            pre_img[z:z+stride[0], y:y+stride[1], x:x+stride[2]] += pred[sh[0]:-sh[0], sh[1]:-sh[1], sh[2]:-sh[2]]
                seg_img = (pre_img > 0) * 1
                seg_img = seg_img[:im_size[0], :im_size[1], :im_size[2]]
            gt = gt[0]
            # io.imsave('{}/segimg{}_validation.tif'.format(self.opbase, num), np.array(seg_img * 255).astype(np.uint8))
            # io.imsave('{}/gtimg{}_validation.tif'.format(self.opbase, num), np.array(gt * 255).astype(np.uint8))
            countListPos = copy.deepcopy(seg_img.astype(np.int16) + gt.astype(np.int16))
            countListNeg = copy.deepcopy(seg_img.astype(np.int16) - gt.astype(np.int16))
            TP += len(np.where(countListPos.reshape(countListPos.size)==2)[0])
            TN += len(np.where(countListPos.reshape(countListPos.size)==0)[0])
            FP += len(np.where(countListNeg.reshape(countListNeg.size)==1)[0])
            FN += len(np.where(countListNeg.reshape(countListNeg.size)==-1)[0])

        evals = self._evaluator(TP, TN, FP, FN)
        return evals, sum_loss


    def _evaluator(self, TP, TN, FP, FN):

        evals = {}
        try:
            evals['Accuracy'] = (TP + TN) / float(TP + TN + FP + FN)
        except:
            evals['Accuracy'] = 0.0
        try:
            evals['Recall'] = TP / float(TP + FN)
        except:
            evals['Recall'] = 0.0
        try:
            evals['Precision'] = TP / float(TP + FP)
        except:
            evals['Precision'] = 0.0
        try:
            evals['Specificity'] = TN / float(TN + FP)
        except:
            evals['Specificity'] = 0.0
        try:
            evals['F-measure'] = 2 * evals['Recall'] * evals['Precision'] / (evals['Recall'] + evals['Precision'])
        except:
            evals['F-measure'] = 0.0
        try:
            evals['IoU'] = TP / float(TP + FP + FN)
        except:
            evals['IoU'] = 0.0
        return evals

    

class NDNTrainer():

    def __init__(
            self,
            model,
            epoch,
            patchsize,
            batchsize,
            gpu,
            opbase,
            mean_image=None,
            opt_method='Adam',
            delv=3,
            r_thr=10,
            ndim=3
    ):
        self.model = model
        self.epoch = epoch
        self.patchsize = patchsize
        self.batchsize = batchsize
        self.gpu = gpu
        self.opbase = opbase
        self.mean_image = mean_image
        self.opt_method = opt_method
        self.criteria = ['Recall', 'Precision', 'F-measure', 'IoU']
        self.delv = delv
        self.r_thr = r_thr
        self.ndim = ndim
        self.eval_epoch = 0


    def training(self, iterators):
        train_iter, val_iter = iterators

        if self.opt_method == 'Adam':
            opt_ndn = optimizers.Adam(alpha=0.07984883572883512, beta1=0.9113157387413141,
                                      beta2=0.9931108449092836, eps=0.07309957525741932)
            #opt_ndn = optimizers.Adam()
            opt_ndn.setup(self.model)
            opt_ndn.add_hook(chainer.optimizer.WeightDecay(0.00000570679784139))
            #opt_ndn.add_hook(chainer.optimizer.WeightDecay(0.000001))

        elif self.opt_method == 'SGD':
            opt_ndn = optimizers.SGD(lr=0.66243829123061737)
            opt_ndn.setup(self.model)
            opt_ndn.add_hook(chainer.optimizer.WeightDecay(0.00000226))

        train_eval, test_eval = {}, {}
        train_eval['loss'], test_eval['loss'] = [], []
        for cri in self.criteria:
            train_eval[cri] = []
            test_eval[cri] = []
        N_train = train_iter.dataset.__len__()
        N_test = val_iter.dataset.__len__()
        with open(self.opbase + '/result.txt', 'w') as f:
            f.write('N_train: {}\n'.format(N_train))
            f.write('N_test: {}\n'.format(N_test))
        bestEpoch, bestRecall, bestPrecision, bestFmeasure, bestIoU = 0, 0, 0, 0, 0

        for epoch in range(1, self.epoch + 1):
            print('[epoch {}]'.format(epoch))
            traeval, train_sum_loss = self._trainer(train_iter, opt_ndn, epoch=epoch)
            train_eval['loss'].append(train_sum_loss / (N_train * self.batchsize))
            if epoch >= self.eval_epoch:
                teseval, test_sum_loss = self._validater(val_iter, epoch=epoch)
                test_eval['loss'].append(test_sum_loss / (N_test * self.batchsize))
            else:
                teseval = {}
                for cri in self.criteria:
                    teseval[cri] = 0
                    test_eval['loss'].append(0)
                    test_sum_loss = 0
                

            for cri in self.criteria:
                train_eval[cri].append(traeval[cri])
                test_eval[cri].append(teseval[cri])
            print('train mean loss={}'.format(train_sum_loss / (N_train * self.batchsize)))
            print('train recall={}, presicion={}'.format(traeval['Recall'], traeval['Precision']))
            print('train F-measure={}, IoU={}'.format(traeval['F-measure'], traeval['IoU']))
            print('test mean loss={}'.format(test_sum_loss / (N_test * self.batchsize)))
            print('test recall={}, presicion={}'.format(teseval['Recall'], teseval['Precision']))
            print('test F-measure={}, IoU={}'.format(teseval['F-measure'], teseval['IoU']))
            with open(self.opbase + '/result.txt', 'a') as f:
                f.write('========================================\n')
                f.write('[epoch {}]\n'.format(epoch))
                f.write('train mean loss={}\n'.format(train_sum_loss / (N_train * self.batchsize)))
                f.write('train recall={}, presicion={}\n'.format(traeval['Recall'], traeval['Precision']))
                f.write('train F-measure={}, IoU={}\n'.format(traeval['F-measure'], traeval['IoU']))
                if epoch >= self.eval_epoch:
                    f.write('validation mean loss={}\n'.format(test_sum_loss / (N_test * self.batchsize)))
                    f.write('validation recall={}, presicion={}\n'.format(teseval['Recall'], teseval['Precision']))
                    f.write('validation F-measure={}, IoU={}\n'.format(teseval['F-measure'], teseval['IoU']))
            with open(self.opbase + '/TrainResult.csv', 'a') as f:
                c = csv.writer(f)
                c.writerow([epoch, traeval['Recall'], traeval['Precision'], traeval['F-measure'], traeval['IoU']])
            with open(self.opbase + '/ValResult.csv', 'a') as f:
                c = csv.writer(f)
                c.writerow([epoch, teseval['Recall'], teseval['Precision'], teseval['F-measure'], teseval['IoU']])

            if epoch == 1:
                pastLoss = train_sum_loss

            if train_sum_loss > pastLoss and self.opt_method == 'SGD':
                learning_rate = opt_ndn.lr * 1.0
                opt_ndn = optimizers.SGD(learning_rate)
                opt_ndn.setup(self.model)
                with open(self.opbase + '/result.txt', 'a') as f:
                    f.write('lr: {}\n'.format(opt_ndn.lr))

            pastLoss = train_sum_loss

            if bestRecall <= teseval['Recall']:
                bestRecall = teseval['Recall']
                # Save Model
                # model_name = 'NDN_Recall_p' + str(self.patchsize) + '.npz'
                # serializers.save_npz(self.opbase + '/' + model_name, self.model)
            if bestPrecision <= teseval['Precision']:
                bestPrecision = teseval['Precision']
                # Save Model
                # smodelfile = 'NDN_Precision_p' + str(self.patchsize) + '.npz'
                # serializers.save_npz(self.opbase + '/' + model_name, self.model)
            if bestFmeasure <= teseval['F-measure']:
                bestFmeasure = teseval['F-measure']
                bestEpoch = epoch
                # Save Model
                model_name = 'NDN_Fmeasure_p' + str(self.patchsize) + '.npz'
                serializers.save_npz(self.opbase + '/' + model_name, self.model)
            if bestIoU <= teseval['IoU']:
                bestIoU = teseval['IoU']

        bestScore = [bestRecall, bestPrecision, bestFmeasure, bestIoU]
        print('========================================')
        print('Best Epoch (F-measure) : ' + str(bestEpoch))
        print('Best Recall : ' + str(bestRecall))
        print('Best Precision : ' + str(bestPrecision))
        print('Best F-measure : ' + str(bestFmeasure))
        print('Best IoU : ' + str(bestIoU))
        with open(self.opbase + '/result.txt', 'a') as f:
            f.write('################################################\n')
            f.write('BestEpoch={}\n'.format(bestEpoch))
            f.write('BestRecall={}, BestPrecision={}\n'.format(bestRecall, bestPrecision))
            f.write('BestFmesure={}, BestIoU={}\n'.format(bestFmeasure, bestIoU))
            f.write('################################################\n')

        return train_eval, test_eval, bestScore


    def _trainer(self, dataset_iter, opt_ndn, epoch):
        TP, numPR, numGT = 0, 0, 0
        dataset_iter.reset()
        N = dataset_iter.dataset.__len__()
        sum_loss = 0
        #perm = np.random.permutation(N)
        for num in range(N):
            #n = perm[num]
            if self.mean_image is None:
                x_patch, y_patch = dataset_iter.next()[0]
            else:
                x_patch, y_patch = dataset_iter.next()[0][0] - self.mean_image, dataset_iter.next()[0][1]

            if self.gpu >= 0:
                x_patch = cuda.to_gpu(x_patch)
                y_patch = cuda.to_gpu(y_patch)

            s_loss, s_output = self.model(x=x_patch, t=y_patch, seg=False)
            opt_ndn.target.cleargrads()
            s_loss.backward()
            opt_ndn.update()

            #train_eval['loss'].append(s_loss.data)
            sum_loss += float(cuda.to_cpu(s_loss.data) * self.batchsize)

            if self.gpu >= 0:
                y_patch = cuda.to_cpu(y_patch[0])
                s_output = cuda.to_cpu(s_output)
            #make pred (0 : background, 1 : object)
            pred = copy.deepcopy((0 < (s_output[0][1] - s_output[0][0])) * 1)
            if epoch >= self.eval_epoch:
                #make Centroid Pred (0 : background, 1 : object)
                markers_pr = morphology.label(pred, neighbors=4)
                mask_size = np.unique(markers_pr, return_counts=True)[1] < (self.delv + 1)
                remove_voxel = mask_size[markers_pr]
                markers_pr[remove_voxel] = 0
                labels = np.unique(markers_pr)
                markers_pr = np.searchsorted(labels, markers_pr)
                numPR += np.max(markers_pr)
                center_pr = np.zeros((np.max(markers_pr), self.ndim))
                count_pr = np.zeros(np.max(markers_pr)).reshape(np.max(markers_pr), 1)
                # make Centroid GT
                markers_gt = morphology.label(y_patch, neighbors=4)
                numGT += np.max(markers_gt)
                center_gt = np.zeros((np.max(markers_gt), self.ndim))
                count_gt = np.zeros(np.max(markers_gt)).reshape(np.max(markers_gt), 1)
                if self.ndim == 2:
                    y, x = np.shape(y_patch)
                    for i in range(y):
                        for j in range(x):
                            if markers_pr[i][j] > 0:
                                center_pr[markers_pr[i][j]-1][0] += j
                                center_pr[markers_pr[i][j]-1][1] += i
                                count_pr[markers_pr[i][j]-1] += 1
                            if markers_gt[i][j] > 0:
                                center_gt[markers_gt[i][j]-1][0] += j
                                center_gt[markers_gt[i][j]-1][1] += i
                                count_gt[markers_gt[i][j]-1] += 1
                elif self.ndim == 3:
                    z, y, x = np.shape(y_patch)
                    for i in range(z):
                        for j in range(y):
                            for k in range(x):
                                if markers_pr[i][j][k] > 0:
                                    center_pr[markers_pr[i][j][k]-1][0] += k
                                    center_pr[markers_pr[i][j][k]-1][1] += j
                                    center_pr[markers_pr[i][j][k]-1][2] += i
                                    count_pr[markers_pr[i][j][k]-1] += 1
                                if markers_gt[i][j][k] > 0:
                                    center_gt[markers_gt[i][j][k]-1][0] += k
                                    center_gt[markers_gt[i][j][k]-1][1] += j
                                    center_gt[markers_gt[i][j][k]-1][2] += i
                                    count_gt[markers_gt[i][j][k]-1] += 1
                center_pr /= count_pr
                center_gt /= count_gt
                pare = []
                for gn in range(len(center_gt)):
                    tmp = 0
                    chaild = []
                    for pn in range(len(center_pr)):
                        if np.sum((center_gt[gn] - center_pr[pn])**2) < self.r_thr**2:
                            chaild.append(pn)
                            tmp += 1
                    if tmp > 0:
                        pare.append(chaild)
                used = np.zeros(len(center_pr))
                TP += self._search_list(pare, used, 0)

        evals = self._evaluator(TP, numPR, numGT)
        return evals, sum_loss


    def _validater(self, dataset_iter, epoch):
        TP, numPR, numGT = 0, 0, 0
        dataset_iter.reset()
        N = dataset_iter.dataset.__len__()
        sum_loss = 0
        #perm = np.random.permutation(N)
        for num in range(N):
            if self.mean_image is None:
                x_batch, y_batch = dataset_iter.next()[0]
            else:
                x_batch, y_batch = dataset_iter.next()[0][0] - self.mean_image, dataset_iter.next()[0][1]

            if self.ndim == 2:
                im_size = x_batch.shape[2:]
                stride = [int(self.patchsize[0]/2), int(self.patchsize[1]/2)]
                sh = [int(stride[0]/2, stride[1]/2)]
            elif self.ndim == 3:
                im_size = x_batch.shape[2:]
                stride = [int(self.patchsize[0]/2), int(self.patchsize[1]/2), int(self.patchsize[2]/2)]
                sh = [int(stride[0]/2), int(stride[1]/2), int(stride[2]/2)]

            ''' calculation for pad size'''
            if np.min(self.patchsize) > np.max(im_size):
                if self.ndim == 2:
                    pad_size = [self.patchsize[0], self.patchsize[1]]
                elif self.ndim == 3:
                    pad_size = [self.patchsize[0], self.patchsize[1], self.patchsize[2]]
            else:
                pad_size = []
                for axis in range(len(im_size)):
                    if (im_size[axis] + 2*sh[axis] - self.patchsize[axis]) % stride[axis] == 0:
                        stride_num = (im_size[axis] + 2*sh[axis] - self.patchsize[axis]) / stride[axis]
                    else:
                        stride_num = (im_size[axis] + 2*sh[axis] - self.patchsize[axis]) / stride[axis] + 1
                    pad_size.append(int(stride[axis] * stride_num + self.patchsize[axis]))

            gt = copy.deepcopy(y_batch)

            if self.ndim == 2:
                x_batch = mirror_extension_image(image=x_batch, ndim=self.ndim, length=int(np.max(self.patchsize)))[:, :, self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1]]
                y_batch = mirror_extension_image(image=y_batch, ndim=self.ndim, length=int(np.max(self.patchsize)))[:, self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1]]
                pre_img = np.zeros(pad_size)
                for y in range(0, pad_size[0]-self.patchsize[0], stride[0]):
                    for x in range(0, pad_size[1]-self.patchsize[1], stride[1]):
                        x_patch = x_batch[:, :, y:y+self.patchsize[0], x:x+self.patchsize[1]]
                        y_patch = y_batch[:, y:y+self.patchsize[0], x:x+self.patchsize[1]]
                        if self.gpu >= 0:
                            x_patch = cuda.to_gpu(x_patch)
                            y_patch = cuda.to_gpu(y_patch)
                        s_loss, s_output = self.model(x=x_patch, t=y_patch, seg=False)
                        sum_loss += float(cuda.to_cpu(s_loss.data) * self.batchsize)
                        if self.gpu >= 0:
                            s_output = cuda.to_cpu(s_output)
                        pred = copy.deepcopy((0 < (s_output[0][1] - s_output[0][0])) * 255)
                        # Add segmentation image
                        pre_img[y:y+self.patchsize[0]-stride[0], x:x+self.patchsize[1]-stride[1]] += pred[sh[0]:-sh[0], sh[1]:-sh[1]]
                seg_img = (pre_img > 0) * 1
                seg_img = seg_img[0:im_size[0], 0:im_size[1]]
            elif self.ndim == 3:
                x_batch = mirror_extension_image(image=x_batch, ndim=self.ndim, length=int(np.max(self.patchsize)))[:, :, self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1], self.patchsize[2]-sh[2]:self.patchsize[2]-sh[2]+pad_size[2]]
                y_batch = mirror_extension_image(image=y_batch, ndim=self.ndim, length=int(np.max(self.patchsize)))[:, self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1], self.patchsize[2]-sh[2]:self.patchsize[2]-sh[2]+pad_size[2]]
                pre_img = np.zeros(pad_size)
                for z in range(0, pad_size[0]-self.patchsize[0], stride[0]):
                    for y in range(0, pad_size[1]-self.patchsize[1], stride[1]):
                        for x in range(0, pad_size[2]-self.patchsize[2], stride[2]):
                            x_patch = x_batch[:, :, z:z+self.patchsize[0], y:y+self.patchsize[1], x:x+self.patchsize[2]]
                            y_patch = y_batch[:, z:z+self.patchsize[0], y:y+self.patchsize[1], x:x+self.patchsize[2]]
                            if self.gpu >= 0:
                                x_patch = cuda.to_gpu(x_patch)
                                y_patch = cuda.to_gpu(y_patch)
                            s_loss, s_output = self.model(x=x_patch, t=y_patch, seg=False)
                            sum_loss += float(cuda.to_cpu(s_loss.data) * self.batchsize)
                            if self.gpu >= 0:
                                s_output = cuda.to_cpu(s_output)
                            pred = copy.deepcopy((0 < (s_output[0][1] - s_output[0][0])) * 255)
                            # Add segmentation image
                            pre_img[z:z+self.patchsize[0]-stride[0], y:y+self.patchsize[1]-stride[1], x:x+self.patchsize[2]-stride[2]] += pred[sh[0]:-sh[0], sh[1]:-sh[1], sh[2]:-sh[2]]
                seg_img = (pre_img > 0) * 1
                seg_img = seg_img[0:im_size[0], 0:im_size[1], 0:im_size[2]]
            gt = gt[0]
            # io.imsave('{}/segimg{}_validation.tif'.format(self.opbase, num), np.array(seg_img * 255).astype(np.uint8))
            # io.imsave('{}/gtimg{}_validation.tif'.format(self.opbase, num), np.array(gt * 255).astype(np.uint8))

            if epoch >= self.eval_epoch:
                #make Centroid Pred (0 : background, 1 : object)
                markers_pr = morphology.label(seg_img, neighbors=4)
                mask_size = np.unique(markers_pr, return_counts=True)[1] < (self.delv + 1)
                remove_voxel = mask_size[markers_pr]
                markers_pr[remove_voxel] = 0
                labels = np.unique(markers_pr)
                markers_pr = np.searchsorted(labels, markers_pr)
                numPR += np.max(markers_pr)
                center_pr = np.zeros((np.max(markers_pr), self.ndim))
                count_pr = np.zeros(np.max(markers_pr)).reshape(np.max(markers_pr), 1)
                # make Centroid GT
                markers_gt = morphology.label(gt, neighbors=4)
                numGT += np.max(markers_gt)
                center_gt = np.zeros((np.max(markers_gt), self.ndim))
                count_gt = np.zeros(np.max(markers_gt)).reshape(np.max(markers_gt), 1)
                if self.ndim == 2:
                    y, x = np.shape(gt)
                    for i in range(y):
                        for j in range(x):
                            if markers_pr[i][j] > 0:
                                center_pr[markers_pr[i][j]-1][0] += j
                                center_pr[markers_pr[i][j]-1][1] += i
                                count_pr[markers_pr[i][j]-1] += 1
                            if markers_gt[i][j] > 0:
                                center_gt[markers_gt[i][j]-1][0] += j
                                center_gt[markers_gt[i][j]-1][1] += i
                                count_gt[markers_gt[i][j]-1] += 1
                elif self.ndim == 3:
                    z, y, x = np.shape(gt)
                    for i in range(z):
                        for j in range(y):
                            for k in range(x):
                                if markers_pr[i][j][k] > 0:
                                    center_pr[markers_pr[i][j][k]-1][0] += k
                                    center_pr[markers_pr[i][j][k]-1][1] += j
                                    center_pr[markers_pr[i][j][k]-1][2] += i
                                    count_pr[markers_pr[i][j][k]-1] += 1
                                if markers_gt[i][j][k] > 0:
                                    center_gt[markers_gt[i][j][k]-1][0] += k
                                    center_gt[markers_gt[i][j][k]-1][1] += j
                                    center_gt[markers_gt[i][j][k]-1][2] += i
                                    count_gt[markers_gt[i][j][k]-1] += 1
                center_pr /= count_pr
                center_gt /= count_gt
                pare = []
                for gn in range(len(center_gt)):
                    tmp = 0
                    chaild = []
                    for pn in range(len(center_pr)):
                        if np.sum((center_gt[gn] - center_pr[pn])**2) < self.r_thr**2:
                            chaild.append(pn)
                            tmp += 1
                    if tmp > 0:
                        pare.append(chaild)
                used = np.zeros(len(center_pr))
                TP += self._search_list(pare, used, 0)

        evals = self._evaluator(TP, numPR, numGT)
        return evals, sum_loss


    def _evaluator(self, TP, numPR, numGT):
        evals = {}
        FP = numPR - TP
        FN = numGT - TP
        try:
            evals['Recall'] = TP / float(TP + FN)
        except:
            evals['Recall'] = 0.0
        try:
            evals['Precision'] = TP / float(TP + FP)
        except:
            evals['Precision'] = 0.0
        try:
            evals['F-measure'] = 2 * evals['Recall'] * evals['Precision'] / (evals['Recall'] + evals['Precision'])
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

