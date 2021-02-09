import logging
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union
from scipy import sparse

Tensor = torch.tensor

class FingerprintDataset(Dataset):
    """
    Dataset class for fingerprint representation of products for template relevance prediction
    """
    def __init__(
        self,
        prodfps_filename: str,
        labels_filename: str,
        root: Optional[str] = None,
    ):
        if root is None:
            root = Path(__file__).resolve().parent / 'data'
        else:
            root = Path(root)
        if (root / prodfps_filename).exists():
            logging.info("Loading pre-computed product fingerprints...")
            self.data = sparse.load_npz(root / prodfps_filename)
            self.data = self.data.tocsr()
        else:
            raise RuntimeError(
                f"Could not find precomputed product fingerprints at "
                f"{root / prodfps_filename}"
            )

        if (root / labels_filename).exists():
            logging.info("Loading labels...")
            self.labels = np.load(root / labels_filename)
        else:
            raise RuntimeError(
                f"Could not find labels at "
                f"{root / labels_filename}"
            )

    def __getitem__(self, idx: Union[int, Tensor]) -> Tuple[Tensor, Tensor, Union[int, List]]:
        """Returns tuple of product fingerprint, label (template index), and index of prod_smi (in CSV file)
        """
        # return idx for retrieving product SMILES & labels from CSV file
        if torch.is_tensor(idx):
            idx = idx.tolist()

        prod_fps = torch.as_tensor(
            self.data[idx].toarray()
        )
        labels = torch.as_tensor(
            self.labels[idx]
        )
        return prod_fps.float(), labels.long(), idx

    def __len__(self):
        return self.data.shape[0]