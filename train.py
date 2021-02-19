import csv
import sys
import logging
import time
import traceback
import argparse
import os
import numpy as np
import rdkit
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun

from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union
from scipy import sparse
from tqdm import tqdm
from rdkit import RDLogger

from model import TemplateNN
from dataset import FingerprintDataset

DATA_FOLDER = Path(__file__).resolve().parent / 'data'
CHECKPOINT_FOLDER = Path(__file__).resolve().parent / 'checkpoint'

def seed_everything(seed: Optional[int] = 0) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    logging.info(f"Using seed: {seed}\n")

def train(args):
    # TODO: add in top-K accuracy
    # TODO: resuming from checkpoint
    # TODO: ReduceLROnPlateau
    seed_everything(args.random_seed)

    logging.info(f'Loading templates from file: {args.templates_file}')
    with open(DATA_FOLDER / args.templates_file, 'r') as f:
        templates = f.readlines()
    templates_filtered = []
    for p in templates:
        pa, cnt = p.strip().split(': ')
        if int(cnt) >= args.min_freq:
            templates_filtered.append(pa)
    logging.info(f'Total number of template patterns: {len(templates_filtered)}')

    model = TemplateNN(
        output_size=len(templates_filtered)+1, # TODO: this should just be len(templates_filtered) and then do masking at valid/test time
        size=args.hidden_size,
        num_layers_body=args.depth,
        input_size=args.fp_size
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using {device} device')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_dataset = FingerprintDataset(
                            args.prodfps_prefix+'_train.npz', 
                            args.labels_prefix+'_train.npy'
                        )
    train_size = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    valid_dataset = FingerprintDataset(
                            args.prodfps_prefix+'_valid.npz', 
                            args.labels_prefix+'_valid.npy'
                        )
    valid_size = len(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=args.bs_eval, shuffle=False)
    del train_dataset, valid_dataset

    proposals_data_valid = pd.read_csv(
        DATA_FOLDER / f"{args.csv_prefix}_valid.csv", 
        index_col=None, dtype='str'
    )
    
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []    
    max_valid_acc = float('-inf')
    wait = 0 # early stopping patience counter
    start = time.time()
    for epoch in range(args.epochs):
        train_loss, train_correct, train_seen = 0, 0, 0
        train_loader = tqdm(train_loader, desc='training')
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels, idxs = data
            inputs, labels = inputs.to(device), labels.to(device)

            model.zero_grad()
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.shape, labels.shape, inputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_seen += labels.shape[0]
            outputs = nn.Softmax(dim=1)(outputs)
            batch_preds = torch.topk(outputs, k=1)[1].squeeze(dim=-1)
            # print(batch_preds.shape, labels.shape)
            train_correct += torch.sum(
                torch.eq(
                    batch_preds, labels
                ), dim=0
            ).item()
            train_loader.set_description(f"training: loss={train_loss/train_seen:.4f}, acc={train_correct/train_seen:.4f}")
            train_loader.refresh()
        train_losses.append(train_loss/train_seen)
        train_accs.append(train_correct/train_seen)

        model.eval()
        with torch.no_grad():
            valid_loss, valid_correct, valid_seen = 0, 0, 0
            valid_loader = tqdm(valid_loader, desc='validating')
            for i, data in enumerate(valid_loader):
                inputs, labels, idxs = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                valid_seen += labels.shape[0]
                outputs = nn.Softmax(dim=-1)(outputs) 
                batch_preds = torch.topk(outputs, k=1)[1].squeeze(dim=-1) # TODO: include top-K accuracy
                valid_correct += torch.sum(
                            torch.eq(
                                batch_preds, labels
                            ), dim=0
                        ).item()

                valid_loader.set_description(f"validating: loss={valid_loss/valid_seen:.4f}, acc={valid_correct/valid_seen:.4f}")
                valid_loader.refresh()

                # display some examples + model predictions/labels for monitoring model generalization
                try:
                    for j in range(i * args.bs_eval, (i+1) * args.bs_eval):
                        # peek at a random sample of current batch to monitor training progress
                        if j % (valid_size // 5) == random.randint(0, 3) or j % (valid_size // 8) == random.randint(0, 4): 
                            rxn_idx = random.sample(list(range(args.bs_eval)), k=1)[0]
                            rxn_true_class = labels[rxn_idx]
                            rxn_pred_class = int(batch_preds[rxn_idx].item())
                            rxn_pred_score = outputs[rxn_idx, rxn_pred_class].item()
                            rxn_true_score = outputs[rxn_idx, rxn_true_class].item()

                            # load template database
                            rxn_pred_temp = templates_filtered[rxn_pred_class]
                            rxn_true_temp_idx = int(proposals_data_valid.iloc[idxs[rxn_idx].item(), 4])
                            if rxn_true_temp_idx < len(templates_filtered):
                                rxn_true_temp = templates_filtered[rxn_true_temp_idx]
                            else:
                                rxn_true_temp = 'Template not in training data'
                            rxn_true_prod = proposals_data_valid.iloc[idxs[rxn_idx].item(), 1]
                            rxn_true_prec = proposals_data_valid.iloc[idxs[rxn_idx].item(), 2]

                            # apply template to get predicted precursor
                            # TODO: need to fix this. why template could not be applied even though it's correct
                            rxn = rdchiralReaction(rxn_pred_temp.split('>>')[-1] + '>>' + rxn_pred_temp.split('>>')[0]) # reverse template
                            prod = rdchiralReactants(rxn_true_prod)
                            rxn_pred_prec = rdchiralRun(rxn, prod)

                            logging.info(f'\ncurr product:                          \t\t{rxn_true_prod}')
                            logging.info(f'pred template:                          \t{rxn_pred_temp}')
                            logging.info(f'true template:                          \t{rxn_true_temp}')
                            if rxn_pred_class == len(templates_filtered):
                                logging.info(f'pred precursor (score = {rxn_pred_score:+.4f}):\t\tNULL template')
                            elif len(rxn_pred_prec) == 0:
                                logging.info(f'pred precursor (score = {rxn_pred_score:+.4f}):\t\tTemplate could not be applied')
                            else:
                                logging.info(f'pred precursor (score = {rxn_pred_score:+.4f}):\t\t{rxn_pred_prec}')
                            logging.info(f'true precursor (score = {rxn_true_score:+.4f}):\t\t{rxn_true_prec}')
                            break
                except Exception as e: # do nothing # https://stackoverflow.com/questions/11414894/extract-traceback-info-from-an-exception-object/14564261#14564261
                    tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                    logging.info("".join(tb_str))
                    logging.info('\nIndex out of range (last minibatch)')

        valid_losses.append(valid_loss/valid_seen)
        valid_accs.append(valid_correct/valid_seen)

        if args.checkpoint and valid_accs[-1] > max_valid_acc:
            # checkpoint model
            model_state_dict = model.state_dict()
            checkpoint_dict = {
                "epoch": epoch,
                "state_dict": model_state_dict, "optimizer": optimizer.state_dict(),
                "train_accs": train_accs, "train_losses": train_losses,
                "valid_accs": valid_accs, "valid_losses": valid_losses,
                "max_valid_acc": max_valid_acc
            }
            checkpoint_filename = (
                CHECKPOINT_FOLDER
                / f"{args.expt_name}_{epoch:04d}.pth.tar"
            )
            torch.save(checkpoint_dict, checkpoint_filename)

        if args.early_stop and max_valid_acc - valid_accs[-1] > args.early_stop_min_delta:
            if args.early_stop_patience <= wait:
                message = f"\nEarly stopped at the end of epoch: {epoch}, \
                \ntrain loss: {train_losses[-1]:.4f}, train acc: {train_accs[-1]:.4f}, \
                \nvalid loss: {valid_losses[-1]:.4f}, valid acc: {valid_accs[-1]:.4f} \
                \n"
                logging.info(message)
                break
            else:
                wait += 1
                logging.info(
                    f'\nIncrease in valid acc < early stop min delta {args.early_stop_min_delta}, \
                    \npatience count: {wait} \
                    \n'
                )
        else:
            wait = 0
            max_valid_acc = max(max_valid_acc, valid_accs[-1])

        message = f"\nEnd of epoch: {epoch}, \
            \ntrain loss: {train_losses[-1]:.4f}, train top-1 acc: {train_accs[-1]:.4f}, \
            \nvalid loss: {valid_losses[-1]:.4f}, valid top-1 acc: {valid_accs[-1]:.4f} \
            \n"
        logging.info(message)

    logging.info(f'Finished training, total time (minutes): {(time.time() - start) / 60}')
    return model

def test(model, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logging.info(f'Loading templates from file: {args.templates_file}')
    with open(DATA_FOLDER / args.templates_file, 'r') as f:
        templates = f.readlines()
    templates_filtered = []
    for p in templates:
        pa, cnt = p.strip().split(': ')
        if int(cnt) >= args.min_freq:
            templates_filtered.append(pa)
    logging.info(f'Total number of template patterns: {len(templates_filtered)}')

    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_dataset = FingerprintDataset(
                            args.prodfps_prefix+'_test.npz', 
                            args.labels_prefix+'_test.npy'
                        )
    test_size = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.bs_eval, shuffle=False)
    del test_dataset

    proposals_data_test = pd.read_csv(
        DATA_FOLDER / f"{args.csv_prefix}_test.csv", 
        index_col=None, dtype='str'
    )

    model.eval()
    with torch.no_grad():
        test_loss, test_correct, test_seen = 0, 0, 0
        test_loader = tqdm(test_loader, desc='testing')
        for i, data in enumerate(test_loader):
            inputs, labels, idxs = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            test_seen += labels.shape[0]
            outputs = nn.Softmax(dim=-1)(outputs) 
            batch_preds = torch.topk(outputs, k=1)[1].squeeze(dim=-1) # TODO: include top-K accuracy
            test_correct += torch.sum(
                            torch.eq(
                                torch.topk(batch_preds, k=1)[1].squeeze(dim=-1), labels
                            ), dim=0
                        ).item()

            test_loader.set_description(f"testing: loss={test_loss/test_seen:.4f}, acc={test_correct/test_seen:.4f}")
            test_loader.refresh()

            # display some examples + model predictions/labels for monitoring model generalization
            try:
                for j in range(i * args.bs_eval, (i+1) * args.bs_eval):
                    # peek at a random sample of current batch to monitor training progress
                    if j % (test_size // 5) == random.randint(0, 3) or j % (test_size // 8) == random.randint(0, 4): 
                        rxn_idx = random.sample(list(range(args.bs_eval)), k=1)[0]
                        rxn_true_class = labels[rxn_idx]
                        rxn_pred_class = int(batch_preds[rxn_idx].item())
                        rxn_pred_score = outputs[rxn_idx, rxn_pred_class].item()
                        rxn_true_score = outputs[rxn_idx, rxn_true_class].item()

                        # load template database
                        rxn_pred_temp = templates_filtered[rxn_pred_class]
                        rxn_true_temp_idx = int(proposals_data_test.iloc[idxs[rxn_idx].item(), 4])
                        if rxn_true_temp_idx < len(templates_filtered):
                            rxn_true_temp = templates_filtered[rxn_true_temp_idx]
                        else:
                            rxn_true_temp = 'Template not in training data'
                        rxn_true_prod = proposals_data_test.iloc[idxs[rxn_idx].item(), 1]
                        rxn_true_prec = proposals_data_test.iloc[idxs[rxn_idx].item(), 2]

                        # apply template to get predicted precursor
                        rxn = rdchiralReaction(rxn_pred_temp.split('>>')[-1] + '>>' + rxn_pred_temp.split('>>')[0]) # reverse template
                        prod = rdchiralReactants(rxn_true_prod)
                        rxn_pred_prec = rdchiralRun(rxn, prod)

                        logging.info(f'\ncurr product:                          \t\t{rxn_true_prod}')
                        logging.info(f'pred template:                          \t{rxn_pred_temp}')
                        logging.info(f'true template:                          \t{rxn_true_temp}')
                        if rxn_pred_class == len(templates_filtered):
                            logging.info(f'pred precursor (score = {rxn_pred_score:+.4f}):\t\tNULL template')
                        elif len(rxn_pred_prec) == 0:
                            logging.info(f'pred precursor (score = {rxn_pred_score:+.4f}):\t\tTemplate could not be applied')
                        else:
                            logging.info(f'pred precursor (score = {rxn_pred_score:+.4f}):\t\t{rxn_pred_prec}')
                        logging.info(f'true precursor (score = {rxn_true_score:+.4f}):\t\t{rxn_true_prec}')
                        break
            except Exception as e: # do nothing # https://stackoverflow.com/questions/11414894/extract-traceback-info-from-an-exception-object/14564261#14564261
                tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
                logging.info("".join(tb_str))
                logging.info('\nIndex out of range (last minibatch)')

    message = f" \
    \ntest loss: {test_loss/test_seen:.4f}, test top-1 acc: {test_correct/test_seen:.4f}"
    logging.info(message)
    logging.info('Finished Testing')

def parse_args():
    parser = argparse.ArgumentParser("train.py")
    # mode & metadata
    parser.add_argument("--expt_name", help="experiment name", type=str, default="")
    parser.add_argument("--do_train", help="whether to train", action="store_true")
    parser.add_argument("--do_test", help="whether to test", action="store_true")
    # file names
    parser.add_argument("--log_file", help="log_file", type=str, default="train")
    parser.add_argument("--templates_file", help="templates_file", type=str, default="50k_training_templates")
    parser.add_argument("--prodfps_prefix",
                        help="npz file of product fingerprints",
                        type=str)
    parser.add_argument("--labels_prefix",
                        help="npy file of labels",
                        type=str)
    parser.add_argument("--csv_prefix",
                        help="csv file of various metadata about the rxn",
                        type=str)
    parser.add_argument("--radius", help="Fingerprint radius", type=int, default=2)
    parser.add_argument("--min_freq", help="Min freq of template", type=int, default=1)
    parser.add_argument("--fp_size", help="Fingerprint size", type=int, default=32681)
    # parser.add_argument("--fp_type", help='Fingerprint type ["count", "bit"]', type=str, default="count")
    # training params
    parser.add_argument("--checkpoint", help="whether to save model checkpoints", action="store_true")
    parser.add_argument("--random_seed", help="random seed", type=int, default=0)
    parser.add_argument("--bs", help="batch size", type=int, default=128)
    parser.add_argument("--bs_eval", help="batch size (valid/test)", type=int, default=256)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--epochs", help="num. of epochs", type=int, default=30)
    parser.add_argument("--early_stop", help="whether to use early stopping", action="store_true") # type=bool, default=True) 
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

    if args.labels_prefix is None:
        args.labels_prefix = f'50k_{args.fp_size}dim_{args.radius}rad_labels'
    if args.prodfps_prefix is None:
        args.prodfps_prefix = f'50k_{args.fp_size}dim_{args.radius}rad_prod_fps'
    if args.csv_prefix is None:
        args.csv_prefix = f'50k_{args.fp_size}dim_{args.radius}rad_csv'

    if args.do_train:
        model = train(args)
    else:
        # load model from saved checkpoint
        raise NotImplementedError
    if args.do_test:
        test(model, args)