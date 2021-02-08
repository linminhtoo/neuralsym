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
from rdkit import RDLogger
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdchiral.template_extractor import extract_from_reaction

from prepare_data import get_tpl, cano_smarts

'''
(neuralsym) [linmin001@eofe8 neuralsym]$ python test_templates.py
C-S(=O)(=O)-O-[C&H2&D2&+0]-C.C-[N&H1&D2&+0]-C>>C-[C&H2&D2&+0]-[N&H0&D3&+0](-C)-C
Traceback (most recent call last):
  File "test_templates.py", line 67, in <module>
    test_get_and_match_template(sample_rxn_smi)
  File "test_templates.py", line 60, in test_get_and_match_template
    pattern_mol
RuntimeError: Pre-condition Violation
        getNumImplicitHs() called without preceding call to calcImplicitValence()
        Violation occurred on line 190 in file Code/GraphMol/Atom.cpp
        Failed Expression: d_implicitValence > -1
        RDKIT: 2020.09.1
        BOOST: 1_73
'''
def test_get_and_match_template(rxn_smi):
    r = rxn_smi.split('>>')[0]
    p = rxn_smi.split('>>')[-1]
    rxn = (0, r, p)
    
    # extract template
    idx, template = get_tpl(rxn)
    p_temp = cano_smarts(template['products']) # reaction_smarts
    r_temp = cano_smarts(template['reactants'])
    cano_temp = r_temp + '>>' + p_temp
    print(cano_temp)

    # match template
    prod_mol = Chem.MolFromSmiles(p)
    [atom.ClearProp('molAtomMapNumber') for atom in prod_mol.GetAtoms()]
    prod_smi_nomap = Chem.MolToSmiles(prod_mol, True)
    # Sometimes stereochem takes another canonicalization... (just in case)
    prod_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(prod_smi_nomap), True)
    
    rcts_mol = Chem.MolFromSmiles(r)
    [atom.ClearProp('molAtomMapNumber') for atom in rcts_mol.GetAtoms()]
    rcts_smi_nomap = Chem.MolToSmiles(rcts_mol, True)
    # Sometimes stereochem takes another canonicalization...
    rcts_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(rcts_smi_nomap), True)
    
    rxn_smi_nomap = rcts_smi_nomap + '>>' + prod_smi_nomap
    # rxn_mol = Chem.MolFromSmiles(rxn_smi_nomap)
    rxn_mol = rdChemReactions.ReactionFromSmarts(rxn_smi_nomap)
    # [a.SetAtomMapNum(0) for a in rxn_mol.GetAtoms()]
    # try:
    #     pattern_mol = Chem.MolFromSmarts(cano_temp)
    # except:
    #     pattern_mol = Chem.MolFromSmarts(cano_temp[1])
    pattern_mol = rdChemReactions.ReactionFromSmarts(template['reaction_smarts']) # instd of cano_temp
    matches = rdChemReactions.HasReactionSubstructMatch(
                                rxn_mol,
                                pattern_mol
                            )
    print(matches)

if __name__ == '__main__':
    sample_rxn_smi = "CS(=O)(=O)O[CH2:1][CH2:2][CH2:3][C:4]1([CH2:5][c:6]2[cH:7][cH:8][c:9]([F:10])[cH:11][cH:12]2)[CH2:13][CH2:14][N:15]([CH2:16][c:17]2[cH:18][c:19]([O:20][CH3:21])[c:22]([O:23][CH3:24])[c:25]([O:26][CH3:27])[cH:28]2)[C:29]1=[O:30].[cH:31]1[cH:32][cH:33][c:34]2[nH:35][c:36]([NH:37][CH:38]3[CH2:39][CH2:40][NH:41][CH2:42][CH2:43]3)[n:44][c:45]2[cH:46]1>>[CH2:1]([CH2:2][CH2:3][C:4]1([CH2:5][c:6]2[cH:7][cH:8][c:9]([F:10])[cH:11][cH:12]2)[CH2:13][CH2:14][N:15]([CH2:16][c:17]2[cH:18][c:19]([O:20][CH3:21])[c:22]([O:23][CH3:24])[c:25]([O:26][CH3:27])[cH:28]2)[C:29]1=[O:30])[N:41]1[CH2:40][CH2:39][CH:38]([NH:37][c:36]2[nH:35][c:34]3[cH:33][cH:32][cH:31][cH:46][c:45]3[n:44]2)[CH2:43][CH2:42]1"

    test_get_and_match_template(sample_rxn_smi)