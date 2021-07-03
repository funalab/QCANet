# -*- coding: utf-8 -*-

import csv
import sys
import time
import random
import copy
import math
import os
import numpy as np
import argparse
import configparser

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from src.lib.trainer import NSNTrainer, NDNTrainer
from src.lib.utils import createOpbase
from src.lib.utils import create_dataset_parser, create_model_parser, create_runtime_parser
from src.lib.utils import print_args
from src.lib.utils import get_dataset, get_model

seed = 109

def main():

    """ Implementation of Quantitative Criteria Acquisition Network based on PyTorch """
    start_time = time.time()

    ''' ConfigParser '''
    conf_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    conf_parser.add_argument("-c", "--conf_file",
                             help="Specify config file", metavar="FILE_PATH")
    args, remaining_argv = conf_parser.parse_known_args()

    dataset_conf_dict, model_conf_dict, runtime_conf_dict = {}, {}, {}

    if args.conf_file is not None:
        config = configparser.ConfigParser()
        config.read([args.conf_file])
        dataset_conf_dict = dict(config.items("Dataset"))
        model_conf_dict = dict(config.items("Model"))
        runtime_conf_dict = dict(config.items("Runtime"))


    ''' Parameters '''
    dataset_parser, dataset_args, remaining_argv = \
        create_dataset_parser(remaining_argv, **dataset_conf_dict)
    model_parser, model_args, remaining_argv = \
        create_model_parser(remaining_argv, **model_conf_dict)
    runtime_parser, runtime_args, remaining_argv = \
        create_runtime_parser(remaining_argv, **runtime_conf_dict)

    ap = argparse.ArgumentParser(
        description='Learning Nuclear Segmentation Network',
        parents=[conf_parser, dataset_parser, model_parser, runtime_parser])
    ap.add_argument('--gpu_id', type=int, help='GPU option')
    args = ap.parse_args()

    patchsize = args.patch_size
    opbase = createOpbase(args.save_dir)
    psep = '/'
    print_args(dataset_args, model_args, runtime_args)
    #with open(opbase + psep + 'result.txt', 'w') as f:
    #    f.write(print_args(dataset_args, model_args, runtime_args))

    # Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ''' Dataset '''
    print('Loading datasets...')
    train_dataset, validation_dataset = get_dataset(args)
    print('-- train_dataset.size = {}\n-- validation_dataset.size = {}'.format(
        len(train_dataset), len(validation_dataset)))


    ''' Iterator '''
    train_iterator = DataLoader(
        dataset=train_dataset,
        batch_size=int(args.batchsize),
        shuffle=True
    )
    validation_iterator = DataLoader(
        dataset=validation_dataset,
        batch_size=int(args.val_batchsize),
        shuffle=False
    )

    ''' Model '''
    print('Initializing models...')
    model = get_model(args)
    if args.init_model is not None:
        print('Load model from', args.init_model)
        model = torch.load(args.init_model)
    model = model.to(args.gpu)

    ''' Optimizer '''
    # Initialize an optimizer
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=args.init_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
            )
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(
            params=model.parameters(),
            lr=args.init_lr,
            rho=args.momentum,
            weight_decay=args.weight_decay
            )
    elif args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(
            params=model.parameters(),
            lr=args.init_lr,
            weight_decay=args.weight_decay
            )
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=args.init_lr,
            weight_decay=args.weight_decay
            )
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=args.init_lr,
            weight_decay=args.weight_decay
            )

    ''' Training Phase '''
    with open(opbase + psep + 'TestResult.csv', 'w') as f:
        c = csv.writer(f)
        c.writerow(['Epoch', 'Accuracy', 'Recall', 'Precision', 'Specificity', 'F-measure', 'IoU'])
    with open(opbase + psep + 'TrainResult.csv', 'w') as f:
        c = csv.writer(f)
        c.writerow(['Epoch', 'Accuracy', 'Recall', 'Precision', 'Specificity', 'F-measure', 'IoU'])

    mean_image = None
    if args.model == 'NSN' or args.model == '3DUNet':
        trainer = NSNTrainer(
            model=model,
            epoch=args.epoch,
            patchsize=eval(args.patch_size),
            batchsize=args.batchsize,
            gpu=args.gpu,
            opbase=opbase,
            optimizer=optimizer,
            mean_image=mean_image,
            ndim=args.ndim
        )
    elif args.model == 'NDN':
        trainer = NDNTrainer(
            model=model,
            epoch=args.epoch,
            patchsize=eval(args.patch_size),
            batchsize=args.batchsize,
            gpu=args.gpu,
            opbase=opbase,
            optimizer=optimizer,
            mean_image=mean_image,
            delv=3,
            r_thr=10,
            ndim=args.ndim
        )
    train_eval, test_eval, best_score = trainer.training((train_iterator, validation_iterator))

    end_time = time.time()
    process_time = end_time - start_time
    print('Elapsed time is (sec) {}'.format(process_time))
    with open(opbase + psep + 'result.txt', 'a') as f:
        f.write('======================================\n')
        f.write('Elapsed time is (sec) {} \n'.format(process_time))

if __name__ == '__main__':
    main()
