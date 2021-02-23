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
from collections import defaultdict
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

def infer_all(args):
    seed_everything(args.random_seed)
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

    # load model from checkpoint
    checkpoint = torch.load(
        CHECKPOINT_FOLDER / f"{args.expt_name}.pth.tar",
        map_location=device,
    )
    model = TemplateNN(
        output_size=len(templates_filtered)+1,
        size=args.hidden_size,
        num_layers_body=args.depth,
        input_size=args.fp_size
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    for phase in args.phases:
        dataset = FingerprintDataset(
                        args.prodfps_prefix+f'_{phase}.npz', 
                        args.labels_prefix+f'_{phase}.npy'
                    )
        loader = DataLoader(dataset, batch_size=args.bs, shuffle=False)
        del dataset

        proposals_data = pd.read_csv(
            DATA_FOLDER / f"{args.csv_prefix}_{phase}.csv", 
            index_col=None, dtype='str'
        )

        preds = []
        loader = tqdm(loader, desc=f'Inferring on {phase}')
        model.eval()
        with torch.no_grad():
            for data in loader:
                inputs, labels, idxs = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                outputs = nn.Softmax(dim=1)(outputs)

                preds.append(torch.topk(outputs, k=args.topk, dim=1)[1])
            preds = torch.cat(preds, dim=0).squeeze(dim=-1).cpu().numpy()
        np.save(
            DATA_FOLDER / f"neuralsym_{args.topk}topk_{args.maxk}maxk_preds_{phase}",
            preds
        )
        logging.info(f'Saved preds of {phase} as npy!')
        
def compile_into_csv(args):
    logging.info(f'Loading templates from file: {args.templates_file}')
    with open(DATA_FOLDER / args.templates_file, 'r') as f:
        templates = f.readlines()
    templates_filtered = []
    for p in templates:
        pa, cnt = p.strip().split(': ')
        if int(cnt) >= args.min_freq:
            templates_filtered.append(pa)
    logging.info(f'Total number of template patterns: {len(templates_filtered)}')

    for phase in args.phases:
        # load mapped_rxn_smi & get precursors & prod_smi

        # load predictions npy files
        preds = np.load(DATA_FOLDER / f"neuralsym_{args.topk}topk_{args.maxk}maxk_preds_{phase}")
        
        # match predictions to true_precursors & get rank


        if phase == 'train': # true precursor has been removed from the proposals, so whatever is left are negatives
            proposed_col_names = [f'neg_precursor_{i}' for i in range(1, args.topk + 1)]
        else: # validation/testing, we don't assume true precursor is present & we also do not remove them if present
            proposed_col_names = [f'cand_precursor_{i}' for i in range(1, args.maxk + 1)]
            # logging.info(f'len(proposed_col_names): {len(proposed_col_names)}')
        col_names = ['orig_rxn_smi', 'prod_smi', 'true_precursors', 'rank_of_true_precursor']
        col_names.extend(proposed_col_names)

        with open(
            DATA_FOLDER /
            f"neuralsym_{args.topk}topk_{args.maxk}maxk_{phase}.csv", 'w'
        ) as out_csv:
            writer = csv.writer(out_csv)
            writer.writerow(col_names) # header
            for row in rows:
                writer.writerow(row)
        
        logging.info(f'Saved proposals of {phase} as CSV!')

def parse_args():
    parser = argparse.ArgumentParser("inference.py")
    # file names
    parser.add_argument("--log_file", help="log_file", type=str, default="inference")
    parser.add_argument("--expt_name", help="name of expt to load model from", type=str)
    parser.add_argument("--templates_file", help="templates_file", type=str, default="50k_training_templates")
    parser.add_argument("--prodfps_prefix",
                        help="npz file of product fingerprints",
                        type=str)
    parser.add_argument("--csv_prefix",
                        help="csv file of various metadata about the rxn",
                        type=str)
    # metadata
    parser.add_argument("--phases", help="Phases to do inference on", type=str, 
                        default=['train', 'valid', 'test'], nargs='+')
    parser.add_argument("--min_freq", help="Min freq of template", type=int, default=1)
    parser.add_argument("--fp_size", help="Fingerprint size", type=int, default=32681)
    parser.add_argument("--random_seed", help="random seed", type=int, default=0)
    parser.add_argument("--bs", help="batch size", type=int, default=500)
    parser.add_argument("--topk", help="How many top-k proposals to put in train (not guaranteed)", 
                        type=int, default=200)
    parser.add_argument("--maxk", help="How many top-k proposals to generate and put in valid/test (not guaranteed)", 
                        type=int, default=200)
    # model params
    parser.add_argument("--hidden_size", help="hidden size", type=int, default=300)
    parser.add_argument("--depth", help="depth", type=int, default=0)
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

    if args.prodfps_prefix is None:
        args.prodfps_prefix = f'50k_{args.fp_size}dim_{args.radius}rad_prod_fps'
    if args.csv_prefix is None:
        args.csv_prefix = f'50k_{args.fp_size}dim_{args.radius}rad_csv'

    logging.info(f'{args}')
    infer_all(args)
    compile_into_csv(args)