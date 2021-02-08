import csv
import sys
import logging
import argparse
import os
import numpy as np
import rdkit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union
from scipy import sparse
from tqdm import tqdm
from rdkit import RDLogger

from model import TemplateNN
from dataset import FingerprintDataset

def train(args):
    # TODO: add GPU support, add test, display statistics
    # TODO: add early stopping
    # TODO: add checkpointing & resuming from checkpoint
    logging.info(f'Loading templates from file: {args.templates_file}')
    with open(args.data_folder / args.templates_file, 'r') as f:
        templates = f.readlines()
    template_cnt = 0
    for p in templates:
        pa, cnt = p.strip().split(': ')
        if int(cnt) >= args.min_freq:
            template_cnt += 1
    logging.info(f'Total number of template patterns: {template_cnt}')

    model = TemplateNN(
        output_size=template_cnt+1, # to allow predicting None template
        size=args.hidden_size,
        num_layers_body=args.depth,
        input_size=args.fp_size
    )
    criterion = nn.CrossEntropyLoss() # should set reduction = 'sum' & then divide loss at end of epoch
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = FingerprintDataset(
                            args.prodfps_prefix+'_train.npz', 
                            args.labels_prefix+'_train.csv'
                        )
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    valid_dataset = FingerprintDataset(
                            args.prodfps_prefix+'_valid.npz', 
                            args.labels_prefix+'_valid.csv'
                        )
    valid_loader = DataLoader(valid_dataset, batch_size=args.bs_eval, shuffle=False)
    
    for epoch in range(args.epochs):
        running_loss = 0.
        for i, data in tqdm(enumerate(train_loader)):
            inputs, labels, idxs = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.

    logging.info('Finished Training')
    return model

def test(model, args):
    test_dataset = FingerprintDataset(
                            args.prodfps_prefix+'_test.npz', 
                            args.labels_prefix+'_test.csv'
                        )
    test_loader = DataLoader(test_dataset, batch_size=args.bs_eval, shuffle=False)

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            inputs, labels, idxs = data

            outputs = model(inputs)

    logging.info('Finished Testing')

def parse_args():
    parser = argparse.ArgumentParser("train.py")
    # mode & metadata
    parser.add_argument("--expt_name", help="experiment name", type=str, default="")
    # parser.add_argument("--date_trained", help="date trained (DD_MM_YYYY)", 
    #                     type=str, default=date.today().strftime("%d_%m_%Y"))
    # file names
    parser.add_argument("--log_file", help="log_file", type=str, default="train")
    parser.add_argument("--prodfps_file_prefix",
                        help="npz file of product fingerprints",
                        type=str)
    parser.add_argument("--labels_file_prefix",
                        help="npy file of labels",
                        type=str)
    parser.add_argument("--csv_file_prefix",
                        help="csv file of various metadata about the rxn",
                        type=str)
    parser.add_argument("--radius", help="Fingerprint radius", type=int, default=2)
    parser.add_argument("--fp_size", help="Fingerprint size", type=int, default=32681)
    # parser.add_argument("--fp_type", help='Fingerprint type ["count", "bit"]', type=str, default="count")
    # training params
    parser.add_argument("--checkpoint", help="whether to save model checkpoints", action="store_true")
    parser.add_argument("--random_seed", help="random seed", type=int, default=0)
    parser.add_argument("--bs", help="batch size", type=int, default=128)
    parser.add_argument("--bs_eval", help="batch size (valid/test)", type=int, default=256)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=5e-3)
    parser.add_argument("--epochs", help="num. of epochs", type=int, default=30)
    parser.add_argument("--early_stop", help="whether to use early stopping", action="store_true") # type=bool, default=True) 
    parser.add_argument("--early_stop_criteria",
                        help="criteria for early stopping ['loss', 'acc', top1_acc', 'top5_acc', 'top10_acc', 'top50_acc']",
                        type=str, default='top1_acc')
    parser.add_argument("--early_stop_patience",
                        help="num. of epochs tolerated without improvement in criteria before early stop",
                        type=int, default=2)
    parser.add_argument("--early_stop_min_delta",
                        help="min. improvement in criteria needed to not early stop", type=float, default=1e-4)
    # model params
    parser.add_argument("--hidden_size", help="hidden size", type=int, default=512)
    parser.add_argument("--depth", help="depth", type=int, default=5)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
 
    RDLogger.DisableLog("rdApp.warning")
    os.makedirs("./logs", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fh = logging.FileHandler(f"./logs/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    if args.data_folder is None:
        args.data_folder = Path(__file__).resolve().parents[0] / 'data'
    else:
        args.data_folder = Path(args.data_folder)

    if args.labels_file_prefix is None:
        args.labels_file_prefix = f'50k_{args.fp_size}dim_{args.radius}rad_labels'
    if args.prodfps_file_prefix is None:
        args.prodfps_file_prefix = f'50k_{args.fp_size}dim_{args.radius}rad_prod_fps'
    if args.csv_file_prefix is None:
        args.csv_file_prefix = f'50k_{args.fp_size}dim_{args.radius}rad_csv'

    model = train(args)
    test(model, args)