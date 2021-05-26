import sys
import os
import pickle
import numpy as np
import rdkit
import random
import torch
import torch.nn as nn
import pandas as pd
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy import sparse
from tqdm import tqdm
from rdkit import RDLogger
from rdkit import Chem

from model import TemplateNN_Highway
from prepare_data import mol_smi_to_count_fp
from infer_config import infer_config

DATA_FOLDER = Path(__file__).resolve().parent / 'data'
CHECKPOINT_FOLDER = Path(__file__).resolve().parent / 'checkpoint'

class Proposer:
    def __init__(self, infer_config: Dict) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"Loading templates from file: {infer_config['templates_file']}")
        with open(DATA_FOLDER / infer_config['templates_file'], 'r') as f:
            templates = f.readlines()
        self.templates_filtered = []
        for p in templates:
            pa, cnt = p.strip().split(': ')
            if int(cnt) >= infer_config['min_freq']:
                self.templates_filtered.append(pa)
        print(f'Total number of template patterns: {len(self.templates_filtered)}')

        self.model, self.indices = self.build_model(infer_config)
        print('Done initializing proposer\n')

    def build_model(self, infer_config: Dict):
         # load model from checkpoint
        checkpoint = torch.load(
            CHECKPOINT_FOLDER / f"{infer_config['expt_name']}.pth.tar",
            map_location=self.device,
        )
        model = TemplateNN_Highway(
            output_size=len(self.templates_filtered),
            size=infer_config['hidden_size'],
            num_layers_body=infer_config['depth'],
            input_size=infer_config['final_fp_size']
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)

        indices = np.loadtxt(DATA_FOLDER / 'variance_indices.txt').astype('int')
        return model, indices

    def propose(self, 
                input_smiles: List[str],
                topk: int = 5,
                **kwargs) -> List[Dict[str, List]]:

        results = {}
        with torch.no_grad():
            for smi in tqdm(input_smiles, desc='Proposing precursors'):
                prod_fp = mol_smi_to_count_fp(smi, infer_config['radius'], infer_config['orig_fp_size'])
                logged = sparse.csr_matrix(np.log(prod_fp.toarray() + 1))
                final_fp = logged[:, self.indices]
                final_fp = torch.as_tensor(final_fp.toarray()).float().to(self.device)

                outputs = self.model(final_fp)
                outputs = nn.Softmax(dim=1)(outputs)
                preds = torch.topk(outputs, k=topk, dim=1)[1].squeeze(dim=0).cpu().numpy()
 
                results[smi] = []
                for idx in preds:
                    score = outputs[0, idx.item()].item()
                    template = self.templates_filtered[idx.item()]
                    rxn = rdchiralReaction(template)
                    prod = rdchiralReactants(smi)
                    try:
                        precs = rdchiralRun(rxn, prod)
                    except:
                        precs = 'N/A'
                    results[smi].append((precs, score)) # Tuple[precs, score] where precs is a List[str]
        return results

if __name__ == '__main__':
    proposer = Proposer(infer_config)
    sample_products = [
        'CCOC(C#Cc1cncc(S(C)(=O)=O)c1)(OCC)OCC',
        'COC(=O)c1cccc2[nH]c(NCC3CCNCC3)nc12',
        'CC(C)(C)OC(=O)N1CC[C@H](NC(=O)C(F)(F)F)C1',
    ]
    results = proposer.propose(sample_products, topk=5)
    print(results)

    # should see this, key = product_smi, value = List of Tuple[precs, score]
    # {'CCOC(C#Cc1cncc(S(C)(=O)=O)c1)(OCC)OCC': [(['C#CC(OCC)(OCC)OCC.CS(=O)(=O)c1cncc(Br)c1'], 0.2964268922805786), 
    # ([], 0.05189067870378494), ([], 0.022873425856232643), ([], 0.0173777025192976), ([], 0.01714753918349743)], 
    
    # 'COC(=O)c1cccc2[nH]c(NCC3CCNCC3)nc12': [(['COC(=O)c1cccc2[nH]c(NCC3CCN(C(=O)OC(C)(C)C)CC3)nc12'], 0.9992377758026123), 
    # (['COC(=O)c1cccc2[nH]c(NCC3CCN(C(=O)OCc4ccccc4)CC3)nc12'], 0.0002514408261049539), (['COC(=O)c1cccc2[nH]c(NCC3CCN(C(=O)C(F)(F)F)CC3)nc12'], 0.00024452427169308066), 
    # (['COC(=O)c1cccc2[nH]c(NCC3CCN(Cc4ccccc4)CC3)nc12'], 0.00012763732229359448), (['COC(=O)c1cccc2[nH]c(NCc3ccncc3)nc12'], 4.4018081098329276e-05)], 
    
    # 'CC(C)(C)OC(=O)N1CC[C@H](NC(=O)C(F)(F)F)C1': [(['CC(C)(C)OC(=O)N1CC[C@H](N)C1.O=C(OC(=O)C(F)(F)F)C(F)(F)F'], 0.7076814770698547), 
    # (['CC(C)(C)OC(=O)N1CC[C@H](N)C1.O=C(Br)C(F)(F)F'], 0.039315130561590195), ([], 0.031778812408447266), 
    # ([], 0.030593203380703926), ([], 0.01709393411874771)]}
