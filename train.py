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
    
    model = TemplateNN(
        output_size=args.template_count,
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
    valid_loader = DataLoader(valid_dataset, batch_size=args.bs, shuffle=False)
    
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
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

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
                        help="csv file of labels",
                        type=str)

    parser.add_argument("--radius", help="Fingerprint radius", type=int, default=2)
    parser.add_argument("--fp_size", help="Fingerprint size", type=int, default=32681)
    # parser.add_argument("--fp_type", help='Fingerprint type ["count", "bit"]', type=str, default="count")
    # training params
    parser.add_argument("--checkpoint", help="whether to save model checkpoints", action="store_true")
    parser.add_argument("--random_seed", help="random seed", type=int, default=0)
    
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
        args.labels_file_prefix = f'50k_{args.fp_size}dim_{args.radius}rad_csv'
    if args.prodfps_file_prefix is None:
        args.prodfps_file_prefix = f'50k_{args.fp_size}dim_{args.radius}rad_prod_fps'

    model = train(args)
    test(model, args)