# Template Relevance Network for Retrosynthesis
A faithful re-implementation of the expansion network in Segler's seminal [Nature paper](https://www.nature.com/articles/nature25978): **"Planning chemical syntheses with deep nueral networks and symbolic AI"**. In essence, NeuralSym is a feedforward network using Highway-ELU blocks, taking as input a product fingerprint (ECFP4), and producing as output an array of logits, one value for each template extracted from the training dataset. A softmax classification can be done across these logits to determine the most promising reaction template, which can be passed through RDChiral to generate the predicted precursors for this product. 


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