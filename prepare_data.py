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
    for phase in ['train', 'valid', 'test']:
        logging.info(f'Processing {phase}')

        with open(args.data_folder / f'{args.rxnsmi_file_prefix}_{phase}.pickle', 'rb') as handle:
            clean_rxnsmi_phase = pickle.load(handle)

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

        with open(args.data_folder / f"{args.output_file_prefix}_prod_smis_{phase}.smi") as f:
            pickle.dump(phase_prod_smi_nomap, f, protocol=4)

def get_tpl(task):
    idx, react, prod = task
    reaction = {'_id': idx, 'reactants': react, 'products': prod}
    template = extract_from_reaction(reaction, super_general=True)
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
    phase == 'train'
    with open(args.data_folder / f'{args.rxnsmi_file_prefix}_{phase}.pickle', 'rb') as handle:
        clean_rxnsmi_phase = pickle.load(handle)

    templates = {}
    rxns = []
    for idx, rxn_smi in enumerate(clean_rxnsmi_phase):
        r = rxn_smi.split('>>')[0]
        p = rxn_smi.split('>>')[-1]
        rxns.append((idx, r, p))
    print('total training rxns:', len(rxns))

    num_cores = len(os.sched_getaffinity(0))
    logging.info(f'Parallelizing over {num_cores} cores')
    pool = multiprocessing.Pool(num_cores)
    for result in tqdm(pool.imap_unordered(get_tpl, rxns),
                        total=len(rxns)):
        idx, template = result
        if 'reaction_smarts' not in template:
            continue
        cano_temp = cano_smarts(template['reaction_smarts'])
        if cano_temp not in templates:
            templates[cano_temp] = 1
        else:
            templates[cano_temp] += 1

    templates = sorted(templates.items(), key=lambda x: x[1], reverse=True)
    templates = ['{}: {}\n'.format(p[0], p[1]) for p in templates]
    with open(args.data_folder / args.templates_file, 'w') as f:
        f.writelines(templates)

def match_templates(args):
    logging.info('Loading templates from file:', args.template_file)
    with open(args.data_folder / args.templates_file, 'r') as f:
        templates = f.readlines()
    temps_filtered = []
    for p in patterns:
        pa, cnt = p.strip().split(': ')
        if int(cnt) >= args.min_freq:
            temps_filtered.append(pa)
    logging.info('total number of template patterns:', len(temps_filtered))

    def find_template(task):
        rxn_idx, rxn_smi = task
        rxn_mol = Chem.MolFromSmiles(rxn_smi)
        [a.SetAtomMapNum(0) for a in rxn_mol.GetAtoms()]

        for idx, pattern in enumerate(temps_filtered):
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is None:
                logging.info('error: pattern_mol is None')
            try:
                matches = rxn_mol.GetSubstructMatches(pattern_mol,
                                                    useChirality=True) # TODO: should this be False? (in RetroXpert it's False)
            except:
                continue
            else:
                if len(matches) > 0:
                    return k, idx
        logging.info(rxn_smi)
        return rxn_idx, 9999 # no template matching

    with open(args.data_folder / f"{args.output_file_prefix}_prod_smis_{phase}.smi") as f:
        phase_prod_smi_nomap = pickle.load(f)

    logging.info('Matching against extracted templates')
    for phase in ['train', 'valid', 'test']:
        logging.info(f'Processing {phase}')
        with open(args.data_folder / f'{args.rxnsmi_file_prefix}_{phase}.pickle', 'rb') as handle:
            clean_rxnsmi_phase = pickle.load(handle)
        
        tasks = [
            (idx, rxn_smi) for idx, rxn_smi in enumerate(clean_rxnsmi_phase)
        ]

        num_cores = len(os.sched_getaffinity(0))
        logging.info(f'Parallelizing over {num_cores} cores')
        pool = multiprocessing.Pool(num_cores)

        # make CSV file to save labels (template_idx) & rxn data for monitoring training
        col_names = ['rxn_idx', 'prod_smi', 'rcts_smi', 'temp_idx', 'template']
        rows = []
        found = 0
        for result in tqdm(pool.imap_unordered(find_template, tasks),
                       total=len(tasks)):
            rxn_idx, template_idx = result

            rcts_smi_map = clean_rxnsmi_phase[rxn_idx].split('>>')[0]
            rcts_mol = Chem.MolFromSmiles(rcts_smi_map)
            [atom.ClearProp('molAtomMapNumber') for atom in rcts_mol.GetAtoms()]
            rcts_smi_nomap = Chem.MolToSmiles(rcts_mol, True)
            # Sometimes stereochem takes another canonicalization...
            rcts_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(rcts_smi_nomap), True)

            rows.append([
                rxn_idx,
                phase_prod_smi_nomap[rxn_idx],
                rcts_smi_nomap,
                template_idx,
                temps_filtered[template_idx]
            ])
            found += (template_idx != 9999)
        
        logging.info(f'Template coverage: {found / len(tasks) * 100:.2f}%')
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

if __name__ == '__main__':
    # NOTE: rmbr to use allmapped_canon_{phase}.pickle

    # setup args.data_folder