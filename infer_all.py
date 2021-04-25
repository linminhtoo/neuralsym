import csv
import sys
import logging
import time
import argparse
import os
import pickle
import numpy as np
import rdkit
import random
import multiprocessing
import torch
import torch.nn as nn
import pandas as pd
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun

from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict, Counter
from functools import partial
from scipy import sparse
from tqdm import tqdm
from rdkit import RDLogger
from rdkit import Chem

from model import TemplateNN_Highway, TemplateNN_FC
from dataset import FingerprintDataset

DATA_FOLDER = Path(__file__).resolve().parent / 'data'
CHECKPOINT_FOLDER = Path(__file__).resolve().parent / 'checkpoint'

def infer_all(args):
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

    if args.model == 'Highway':
        model = TemplateNN_Highway(
            output_size=len(templates_filtered),
            size=args.hidden_size,
            num_layers_body=args.depth,
            input_size=args.fp_size
        )
    elif args.model == 'FC':
        model = TemplateNN_FC(
            output_size=len(templates_filtered),
            size=args.hidden_size,
            input_size=args.fp_size
        )
    else:
        raise ValueError('Unrecognized model name')

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    for phase in args.phases:
        dataset = FingerprintDataset(
                        args.prodfps_prefix+f'_{phase}.npz', 
                        args.labels_prefix+f'_{phase}.npy'
                    )
        loader = DataLoader(dataset, batch_size=args.bs, shuffle=False)
        del dataset

        preds = []
        loader = tqdm(loader, desc=f'Inferring on {phase}')
        model.eval()
        with torch.no_grad():
            for data in loader:
                inputs, labels, idxs = data # we don't need labels & idxs
                inputs = inputs.to(device)

                outputs = model(inputs)
                outputs = nn.Softmax(dim=1)(outputs)

                preds.append(torch.topk(outputs, k=args.topk, dim=1)[1])
            preds = torch.cat(preds, dim=0).squeeze(dim=-1).cpu().numpy()
        logging.info(f'preds.shape: {preds.shape}')
        np.save(
            DATA_FOLDER / f"neuralsym_{args.topk}topk_{args.maxk}maxk_preds_{args.seed}_{phase}",
            preds
        )
        logging.info(f'Saved preds of {phase} as npy!')

def gen_precs(templates_filtered, preds, phase_topk, task):
    i, prod_smi_nomap = task
    # generate predictions from templates
    precursors, dup_count = [], 0
    pred_temp_idxs = preds[i]
    for idx in pred_temp_idxs:
        template = templates_filtered[idx]
        
        rxn = rdchiralReaction(template)
        prod = rdchiralReactants(prod_smi_nomap)
        try:
            precs = rdchiralRun(rxn, prod)
            precursors.extend(precs)
        except:
            continue

    # remove duplicate predictions
    seen = []
    for prec in precursors: # canonicalize all predictions
        prec = Chem.MolToSmiles(Chem.MolFromSmiles(prec), True)
        if prec not in seen:
            seen.append(prec)
        else:
            dup_count += 1

    if len(seen) < phase_topk:
        seen.extend(['9999'] * (phase_topk - len(seen)))
    else:
        seen = seen[:phase_topk]
    
    return precursors, seen, dup_count

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
        # load predictions npy files
        preds = np.load(DATA_FOLDER / f"neuralsym_{args.topk}topk_{args.maxk}maxk_preds_{args.seed}_{phase}.npy")

        # load mapped_rxn_smi
        with open(DATA_FOLDER / f'{args.rxn_smi_prefix}_{phase}.pickle', 'rb') as f:
            clean_rxnsmi_phase = pickle.load(f)

        proposals_data = pd.read_csv(
            DATA_FOLDER / f"{args.csv_prefix}_{phase}.csv", 
            index_col=None, dtype='str'
        )

        tasks = []
        for i in range(len(clean_rxnsmi_phase)): # build tasks
            tasks.append((i, proposals_data.iloc[i, 1]))

        proposals_phase = {}
        proposed_precs_phase, prod_smiles_phase, rcts_smiles_phase = [], [], []
        proposed_precs_phase_withdups = [] # true representation of model predictions, for calc_accs() 
        prod_smiles_mapped_phase = [] # helper for analyse_proposed() 
        phase_topk = args.topk if phase == 'train' else args.maxk
        dup_count = 0
        num_cores = len(os.sched_getaffinity(0))
        logging.info(f'Parallelizing over {num_cores} cores')
        pool = multiprocessing.Pool(num_cores)

        gen_precs_partial = partial(gen_precs, templates_filtered, preds, phase_topk)
        for i, result in enumerate(tqdm(pool.imap(gen_precs_partial, tasks), 
                            total=len(clean_rxnsmi_phase), desc='Generating predicted reactants')):
            precursors, seen, this_dup = result
            dup_count += this_dup

            prod_smi = clean_rxnsmi_phase[i].split('>>')[-1]
            prod_smiles_mapped_phase.append(prod_smi)
            
            prod_smi_nomap = proposals_data.iloc[i, 1]
            prod_smiles_phase.append(prod_smi_nomap)

            rcts_smi_nomap = proposals_data.iloc[i, 2]
            rcts_smiles_phase.append(rcts_smi_nomap)

            proposals_phase[prod_smi] = precursors
            proposed_precs_phase.append(seen)
            proposed_precs_phase_withdups.append(precursors)

        with open(DATA_FOLDER / f'precs_{args.seed}_{phase}.pickle', 'wb') as f:
            pickle.dump(proposed_precs_phase_withdups, f)
        with open(DATA_FOLDER / f'seen_{args.seed}_{phase}.pickle', 'wb') as f:
            pickle.dump(proposed_precs_phase, f)
        
        dup_count /= len(clean_rxnsmi_phase)
        logging.info(f'Avg # dups per product: {dup_count}')

        # match predictions to true_precursors & get rank
        logging.info('\nCalculating ranks before removing duplicates')
        _ = calc_accs( 
            [phase],
            clean_rxnsmi_phase,
            rcts_smiles_phase,
            proposed_precs_phase_withdups,
        ) # just to calculate accuracy

        logging.info('\nCalculating ranks after removing duplicates')
        ranks_dict = calc_accs(
                    [phase],
                    clean_rxnsmi_phase,
                    rcts_smiles_phase,
                    proposed_precs_phase
                )
        ranks_phase = ranks_dict[phase]
        if phase == 'train':
            logging.info('\n(For training only) Double checking accuracy after removing ground truth predictions')
            _ = calc_accs(
                    [phase],
                    clean_rxnsmi_phase,
                    rcts_smiles_phase,
                    proposed_precs_phase
                )
        
        analyse_proposed(
            prod_smiles_phase,
            prod_smiles_mapped_phase,
            proposals_phase, # this func needs this to be a dict {mapped_prod_smi: proposals}
        )
        
        combined = {} 
        zipped = []
        for rxn_smi, prod_smi, rcts_smi, rank_of_true_precursor, proposed_rcts_smi in zip(
            clean_rxnsmi_phase,
            prod_smiles_phase,
            rcts_smiles_phase,
            ranks_phase,
            proposed_precs_phase,
        ):
            result = []
            result.extend([rxn_smi, prod_smi, rcts_smi, rank_of_true_precursor])
            result.extend(proposed_rcts_smi)
            zipped.append(result)

        combined[phase] = zipped
        logging.info('Zipped all info for each rxn_smi into a list for dataframe creation!')

        temp_dataframe = pd.DataFrame(
            data={'zipped': combined[phase]}
        )
        phase_dataframe = pd.DataFrame(
            temp_dataframe['zipped'].to_list(),
            index=temp_dataframe.index
        )

        if phase == 'train': # true precursor has been removed from the proposals, so whatever is left are negatives
            proposed_col_names = [f'neg_precursor_{i}' for i in range(1, args.topk + 1)]
        else: # validation/testing, we don't assume true precursor is present & we also do not remove them if present
            proposed_col_names = [f'cand_precursor_{i}' for i in range(1, args.maxk + 1)]
        col_names = ['orig_rxn_smi', 'prod_smi', 'true_precursors', 'rank_of_true_precursor']
        col_names.extend(proposed_col_names)
        phase_dataframe.columns = col_names

        phase_dataframe.to_csv(
            DATA_FOLDER / 
            f'neuralsym_{args.topk}topk_{args.maxk}maxk_noGT_{args.seed}_{phase}.csv',
            index=False
        )
        logging.info(f'Saved proposals of {phase} as CSV!')

def calc_accs( 
            phases : List[str],
            clean_rxnsmi_phase : List[str],
            rcts_smiles_phase : List[str],
            proposed_precs_phase : List[str],
            ) -> Dict[str, List[int]]:
    ranks = {}
    for phase in phases:
        phase_ranks = []
        if phase == 'train':
            for idx in tqdm(range(len(clean_rxnsmi_phase))):
                true_precursors = rcts_smiles_phase[idx]
                all_proposed_precursors = proposed_precs_phase[idx]

                found = False
                for rank, proposal in enumerate(all_proposed_precursors): # ranks are 0-indexed 
                    if true_precursors == proposal:
                        phase_ranks.append(rank)
                        # remove true precursor from proposals 
                        all_proposed_precursors.pop(rank) 
                        all_proposed_precursors.append('9999')
                        found = True
                        break

                if not found:
                    phase_ranks.append(9999)    
        else:
            for idx in tqdm(range(len(clean_rxnsmi_phase))):
                true_precursors = rcts_smiles_phase[idx]
                all_proposed_precursors = proposed_precs_phase[idx]

                found = False
                for rank, proposal in enumerate(all_proposed_precursors): # ranks are 0-indexed  
                    if true_precursors == proposal:
                        phase_ranks.append(rank) 
                        # do not pop true precursor from proposals! 
                        found = True
                        break

                if not found:
                    phase_ranks.append(9999) 
        ranks[phase] = phase_ranks

        logging.info('\n')
        for n in [1, 3, 5, 10, 20, 50, 100, 200]:
            total = float(len(ranks[phase]))
            acc = sum([r+1 <= n for r in ranks[phase]]) / total
            logging.info(f'{phase.title()} Top-{n} accuracy: {acc * 100 : .3f}%')
        logging.info('\n')

    return ranks # dictionary 

def analyse_proposed(
                    prod_smiles_phase : List[str],
                    prod_smiles_mapped_phase : List[str],
                    proposals_phase : Dict[str, List[str]],
                    ): 
    proposed_counter = Counter()
    total_proposed, min_proposed, max_proposed = 0, float('+inf'), float('-inf')
    key_count = 0
    for key, mapped_key in zip(prod_smiles_phase, prod_smiles_mapped_phase): 
        precursors = proposals_phase[mapped_key]
        precursors_count = len(precursors)
        total_proposed += precursors_count
        if precursors_count > max_proposed:
            max_proposed = precursors_count
            prod_smi_max = key
        if precursors_count < min_proposed:
            min_proposed = precursors_count
            prod_smi_min = key
        
        proposed_counter[key] = precursors_count
        key_count += 1
        
    logging.info(f'Average precursors proposed per prod_smi (dups removed): {total_proposed / key_count}')
    logging.info(f'Min precursors: {min_proposed} for {prod_smi_min}')
    logging.info(f'Max precursors: {max_proposed} for {prod_smi_max})')

    logging.info(f'\nMost common 20:')
    for i in proposed_counter.most_common(20):
        logging.info(f'{i}')
    logging.info(f'\nLeast common 20:')
    for i in proposed_counter.most_common()[-20:]:
        logging.info(f'{i}')
    return

def parse_args():
    parser = argparse.ArgumentParser("inference.py")
    # file names
    parser.add_argument("--log_file", help="log_file", type=str, default="inference")
    parser.add_argument("--expt_name", help="name of expt to load model from", type=str)
    parser.add_argument("--rxn_smi_prefix", help="rxn_smi file", 
                        type=str, default="50k_clean_rxnsmi_noreagent_allmapped_canon")
    parser.add_argument("--templates_file", help="templates_file", 
                        type=str, default="50k_training_templates")
    parser.add_argument("--prodfps_prefix",
                        help="npz file of product fingerprints",
                        type=str)
    parser.add_argument("--labels_prefix",
                        help="npy file of labels",
                        type=str)
    parser.add_argument("--csv_prefix",
                        help="csv file of various metadata about the rxn",
                        type=str)
    # metadata
    parser.add_argument("--seed", help="Seed used for model training", type=int, default=1337)
    parser.add_argument("--model", help="['Highway', 'FC']", type=str, default='Highway')
    parser.add_argument("--phases", help="Phases to do inference on", type=str, 
                        default=['train', 'valid', 'test'], nargs='+')
    parser.add_argument("--min_freq", help="Min freq of template", type=int, default=1)
    parser.add_argument("--radius", help="Fingerprint radius", type=int, default=2)
    parser.add_argument("--fp_size", help="Fingerprint size", type=int, default=32681)
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

    if args.labels_prefix is None:
        args.labels_prefix = f'50k_1000000dim_{args.radius}rad_to_{args.fp_size}_labels'
    if args.prodfps_prefix is None:
        args.prodfps_prefix = f'50k_1000000dim_{args.radius}rad_to_{args.fp_size}_prod_fps'
    if args.csv_prefix is None:
        args.csv_prefix = f'50k_1000000dim_{args.radius}rad_to_{args.fp_size}_csv'

    logging.info(f'{args}')
    if not (DATA_FOLDER / f"neuralsym_{args.topk}topk_{args.maxk}maxk_preds_train.npy").exists():
        infer_all(args) # <10 sec to infer on train + valid + test on 1x RTX2080
    compile_into_csv(args) # this is slow, needs ~1.5h on 8 cores (parallelized)