# Template Relevance Network for Retrosynthesis
A faithful (to the best of my ability) re-implementation (because the authors did not release the source code) of the expansion network in Segler's seminal [Nature paper](https://www.nature.com/articles/nature25978): **"Planning chemical syntheses with deep nueral networks and symbolic AI"**. In essence, NeuralSym is a feedforward network using Highway-ELU blocks, taking as input a product fingerprint (ECFP4), and producing as output an array of logits, one value for each template extracted from the training dataset. A softmax classification can be done across these logits to determine the most promising reaction template, which can be passed through RDChiral to generate the predicted precursors for this product. 

## Data files
This model has been re-trained on USPTO-50K dataset, which has ~50,000 atom-mapped reactions from US Patent data.
I have done some pre-processing on the original data from Schneider et al. to get slightly cleaner train/valid/test datasets, which have 39713/4989/5005 reaction SMILES respectively. These are in the 3 .pickle files in data/. From these 3 files, just do
```
    python prepare_data.py
```
to extract templates from training data, get the labels & so on. I have significantly optimized the code & it automatically parallelizes, so it shouldn't take long to prepare all the data on a standard 8/16-core machine, maybe 15 mins total (slowest step is the variance thresholding of 1 mil-dim fingerprints to 32681-dim fingerprints)

## Training
I have provided a sample ```train.sh``` file with sensible hyperparameters that achieved ~40% top-1 valid/test accuracies. Just do
```
    bash -i train.sh
```
On 1x RTX2080 one epoch takes 8 seconds, and the whole training finishes in <5 minutes. 
The training arguments can be found in ```train.py``` & should be self-explanatory. I plan to do a quick bayesian optimization using Optuna, but don't expect any fancy improvements (<1% probably).

## Inference
This is a work in progress in ```inference.py```. For my own project, I am first working on generating up to top-200 precursors across the entire train/valid/test datasets. But I will also work on a simple API that accepts a list of product SMILES and outputs a list of top-K dictionaries containing precursors & corresponding probabilities assigned by the model.

## Requirements & Setup instructions
RDKit, RDChiral & PyTorch are the main packages.
Tested on Python 3.6
```
    TORCH_VER=1.6.0
    CUDA_VER=10.1
    CUDA_CODE=cu101

    # ensure conda is initialized first
    conda create -n neuralsym python=3.6 tqdm pathlib typing scipy pandas joblib -y
    conda activate neuralsym

    conda install -y pytorch=${TORCH_VER} torchvision cudatoolkit=${CUDA_VER} torchtext -c pytorch
    conda install -y rdkit -c rdkit

    pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"
```