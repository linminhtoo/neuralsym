import csv
import pickle 
import sys
import logging
import argparse
import os
import numpy as np
import rdkit
import scipy
import multiprocessing

from functools import partial
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union
from scipy import sparse
from tqdm import tqdm

from rdkit import RDLogger
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from rdchiral.template_extractor import extract_from_reaction

sparse_fp = scipy.sparse.csr_matrix

def mol_smi_to_count_fp(
    mol_smi: str, radius: int = 2, fp_size: int = 32681, dtype: str = "int32"
) -> scipy.sparse.csr_matrix:
    fp_gen = GetMorganGenerator(
        radius=radius, useCountSimulation=True, includeChirality=True, fpSize=fp_size
    )
    mol = Chem.MolFromSmiles(mol_smi)
    uint_count_fp = fp_gen.GetCountFingerprint(mol)
    count_fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(uint_count_fp, count_fp)
    return sparse.csr_matrix(count_fp, dtype=dtype)

# use fp_size = 32681 and radius = 2 for now
# TODO: fold to 1,000,000-dim --> log(x+1) --> variance threshold into 32681
def gen_prod_fps(args):
    # parallelizing makes it very slow for some reason
    for phase in ['train', 'valid', 'test']:
        logging.info(f'Processing {phase}')

        with open(args.data_folder / f'{args.rxnsmi_file_prefix}_{phase}.pickle', 'rb') as f:
            clean_rxnsmi_phase = pickle.load(f)

        phase_prod_smi_nomap = []
        phase_rxn_prod_fps = []
        for rxn_id, rxn_smi in enumerate(tqdm(clean_rxnsmi_phase, desc='Processing rxn_smi')):
            prod_smi_map = rxn_smi.split('>>')[-1]
            prod_mol = Chem.MolFromSmiles(prod_smi_map)
            [atom.ClearProp('molAtomMapNumber') for atom in prod_mol.GetAtoms()]
            prod_smi_nomap = Chem.MolToSmiles(prod_mol, True)
            # Sometimes stereochem takes another canonicalization... (just in case)
            prod_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(prod_smi_nomap), True)
            phase_prod_smi_nomap.append(prod_smi_nomap)

            prod_fp = mol_smi_to_count_fp(prod_smi_nomap, args.radius, args.fp_size)
            phase_rxn_prod_fps.append(prod_fp)

        # these are the input data into the network
        phase_rxn_prod_fps = sparse.vstack(phase_rxn_prod_fps)
        sparse.save_npz(
            args.data_folder / f"{args.output_file_prefix}_prod_fps_{phase}.npz",
            phase_rxn_prod_fps
        )

        with open(args.data_folder / f"{args.output_file_prefix}_prod_smis_nomap_{phase}.smi", 'wb') as f:
            pickle.dump(phase_prod_smi_nomap, f, protocol=4)

def get_tpl(task):
    idx, react, prod = task
    reaction = {'_id': idx, 'reactants': react, 'products': prod}
    template = extract_from_reaction(reaction)
    # https://github.com/connorcoley/rdchiral/blob/master/rdchiral/template_extractor.py
    return idx, template

def cano_smarts(smarts):
    tmp = Chem.MolFromSmarts(smarts)
    if tmp is None:
        return None, smarts
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    cano = Chem.MolToSmarts(tmp)
    if '[[se]]' in cano:  # strange parse error
        cano = smarts
    return cano

def get_train_templates(args):
    '''
    For the expansion rules, a more general rule definition was employed. Here, only
    the reaction centre was extracted. Rules occurring at least three times
    were kept. The two sets encompass 17,134 and 301,671 rules, and cover
    52% and 79% of all chemical reactions from 2015 and after, respectively.
    '''
    logging.info('Extracting templates from training data')
    phase = 'train'
    with open(args.data_folder / f'{args.rxnsmi_file_prefix}_{phase}.pickle', 'rb') as f:
        clean_rxnsmi_phase = pickle.load(f)

    templates = {}
    rxns = []
    for idx, rxn_smi in enumerate(clean_rxnsmi_phase):
        r = rxn_smi.split('>>')[0]
        p = rxn_smi.split('>>')[-1]
        rxns.append((idx, r, p))
    logging.info(f'Total training rxns: {len(rxns)}')

    num_cores = len(os.sched_getaffinity(0))
    logging.info(f'Parallelizing over {num_cores} cores')
    pool = multiprocessing.Pool(num_cores)
    for result in tqdm(pool.imap_unordered(get_tpl, rxns),
                        total=len(rxns)):
        idx, template = result
        if 'reaction_smarts' not in template:
            continue # no template could be extracted

        # canonicalize template
        p_temp = cano_smarts(template['products']) # reaction_smarts
        r_temp = cano_smarts(template['reactants'])
        cano_temp = r_temp + '>>' + p_temp
        # cano_temp = cano_smarts(template['reaction_smarts'])

        if cano_temp not in templates:
            templates[cano_temp] = 1
        else:
            templates[cano_temp] += 1

    templates = sorted(templates.items(), key=lambda x: x[1], reverse=True)
    templates = ['{}: {}\n'.format(p[0], p[1]) for p in templates]
    with open(args.data_folder / args.templates_file, 'w') as f:
        f.writelines(templates)

def get_template_idx(temps_filtered, task):
    # extract template for this rxn_smi, and match it to template dictionary from training data
    rxn_idx, rxn_smi = task
    r = rxn_smi.split('>>')[0]
    p = rxn_smi.split('>>')[-1]
    rxn = (rxn_idx, r, p)
    rxn_idx, rxn_template = get_tpl(rxn)
    p_temp = cano_smarts(rxn_template['products']) # reaction_smarts
    r_temp = cano_smarts(rxn_template['reactants'])
    cano_temp = r_temp + '>>' + p_temp

    # rxn_mol = Chem.MolFromSmiles(rxn_smi)
    # [a.SetAtomMapNum(0) for a in rxn_mol.GetAtoms()]

    for temp_idx, extr_temp in enumerate(temps_filtered):
        if extr_temp == cano_temp:
            return rxn_idx, temp_idx # template_idx = label

    # for temp_idx, pattern in enumerate(temps_filtered):
    #     pattern_mol = Chem.MolFromSmarts(pattern)
    #     if pattern_mol is None:
    #         logging.info('error: pattern_mol is None')
    #     try:
    #         matches = rxn_mol.GetSubstructMatches(
    #                         pattern_mol,
    #                         useChirality=True
    #                     ) # TODO: should this be False? (in RetroXpert it's False)
    #     except:
    #         continue
    #     else:
    #         if len(matches) > 0:
    #             return rxn_idx, temp_idx # template_idx = label
    # logging.info(rxn_smi)
    return rxn_idx, -1 # no template matching

def match_templates(args):
    logging.info(f'Loading templates from file: {args.templates_file}')
    with open(args.data_folder / args.templates_file, 'r') as f:
        templates = f.readlines()
    temps_filtered = []
    for p in templates:
        pa, cnt = p.strip().split(': ')
        if int(cnt) >= args.min_freq:
            temps_filtered.append(pa)
    logging.info(f'Total number of template patterns: {len(temps_filtered)}')

    logging.info('Matching against extracted templates')
    for phase in ['train', 'valid', 'test']:
        logging.info(f'Processing {phase}')
        with open(args.data_folder / f"{args.output_file_prefix}_prod_smis_nomap_{phase}.smi", 'rb') as f:
            phase_prod_smi_nomap = pickle.load(f)
        with open(args.data_folder / f'{args.rxnsmi_file_prefix}_{phase}.pickle', 'rb') as f:
            clean_rxnsmi_phase = pickle.load(f)
        
        tasks = [
            (idx, rxn_smi) for idx, rxn_smi in enumerate(clean_rxnsmi_phase)
        ]

        num_cores = len(os.sched_getaffinity(0))
        logging.info(f'Parallelizing over {num_cores} cores')
        pool = multiprocessing.Pool(num_cores)

        # make CSV file to save labels (template_idx) & rxn data for monitoring training
        col_names = ['rxn_idx', 'prod_smi', 'rcts_smi', 'temp_idx', 'template']
        rows = []
        labels = []
        found = 0
        get_template_partial = partial(get_template_idx, temps_filtered)
        for result in tqdm(pool.imap_unordered(get_template_partial, tasks),
                       total=len(tasks)):
            rxn_idx, template_idx = result

            rcts_smi_map = clean_rxnsmi_phase[rxn_idx].split('>>')[0]
            rcts_mol = Chem.MolFromSmiles(rcts_smi_map)
            [atom.ClearProp('molAtomMapNumber') for atom in rcts_mol.GetAtoms()]
            rcts_smi_nomap = Chem.MolToSmiles(rcts_mol, True)
            # Sometimes stereochem takes another canonicalization...
            rcts_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(rcts_smi_nomap), True)

            template = temps_filtered[template_idx] if template_idx != -1 else ''
            rows.append([
                rxn_idx,
                phase_prod_smi_nomap[rxn_idx],
                rcts_smi_nomap,
                template,
                template_idx,
            ])
            labels.append(template_idx)
            found += (template_idx != -1)
        
        logging.info(f'Template coverage: {found / len(tasks) * 100:.2f}%')
        labels = np.array(labels)
        np.save(
            args.data_folder / f"{args.output_file_prefix}_labels_{phase}",
            labels
        )
        with open(
            args.data_folder /
            f"{args.output_file_prefix}_csv_{phase}.csv", 'w'
        ) as out_csv:
            writer = csv.writer(out_csv)
            writer.writerow(col_names) # header
            for row in rows:
                writer.writerow(row)

''' 
(reference from RetroXpert)
pattern_feat feature dim:  646
# ave center per mol: 36
'''

def parse_args():
    parser = argparse.ArgumentParser("prepare_data.py")
    # file names
    parser.add_argument("--log_file", help="log_file", type=str, default="prepare_data")
    parser.add_argument("--data_folder", help="Path to data folder (do not change)", type=str,
                        default=None) 
    parser.add_argument("--rxnsmi_file_prefix", help="Prefix of the 3 pickle files containing the train/valid/test reaction SMILES strings (do not change)", type=str,
                        default='50k_clean_rxnsmi_noreagent_allmapped_canon') 
    parser.add_argument("--output_file_prefix", help="Prefix of output files", 
                        type=str)
    parser.add_argument("--templates_file", help="Filename of templates extracted from training data", 
                        type=str, default='50k_training_templates')
    # parser.add_argument("--parallelize", help="Whether to parallelize over all available cores", action="store_true")
    
    parser.add_argument("--min_freq", help="Minimum frequency of template in training data to be retained", type=int, default=1)
    parser.add_argument("--radius", help="Fingerprint radius", type=int, default=2)
    parser.add_argument("--fp_size", help="Fingerprint size", type=int, default=32681)
    # parser.add_argument("--fp_type", help='Fingerprint type ["count", "bit"]', type=str, default="count")
    return parser.parse_args()

if __name__ == '__main__':
    # NOTE: rmbr to use allmapped_canon_{phase}.pickle
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

    if args.output_file_prefix is None:
        args.output_file_prefix = f'50k_{args.fp_size}dim_{args.radius}rad'

    logging.info(args)

    if not (args.data_folder / f"{args.output_file_prefix}_prod_fps_valid.npz").exists():
        # ~2 min on 40k train prod_smi on 16 cores
        gen_prod_fps(args)
    if not (args.data_folder / args.templates_file).exists():
        # ~40 sec on 40k train rxn_smi on 16 cores
        get_train_templates(args)
    if True: # not (args.data_folder / f"{args.output_file_prefix}_csv_train.csv").exists():
        # ~3 min on 40k train rxn_smi on 16 cores
        match_templates(args)
    
    logging.info('Done!')
