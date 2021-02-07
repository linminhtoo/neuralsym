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

from model import TemplateNN
from dataset import FingerprintDataset

def train(args):
    # TODO: add GPU support, add test, display statistics
    # TODO: add checkpointing & resuming from checkpoint
    model = TemplateNN(
        output_size=args.template_count,
        size=args.hidden_size,
        num_layers_body=args.depth,
        input_size=args.fp_size
    )
    criterion = nn.CrossEntropyLoss()
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

if __name__ == '__main__':
    model = train(args)
    test(model, args)

    # TODO: parse args