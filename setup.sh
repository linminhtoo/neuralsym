TORCH_VER=1.6.0
CUDA_VER=10.1
CUDA_CODE=cu101

source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda create -n neuralsym python=3.6 tqdm pathlib typing scipy pandas joblib -y
conda activate neuralsym

conda install -y pytorch=${TORCH_VER} torchvision cudatoolkit=${CUDA_VER} torchtext -c pytorch
conda install -y rdkit -c rdkit

pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"