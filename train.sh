source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate neuralsym

python train.py \
    --expt_name 'debug' \
    --do_train \
    --do_test \
    --prodfps_prefix 50k_32681dim_2rad_prod_fps \
    --labels_prefix 50k_32681dim_2rad_labels \
    --csv_prefix 50k_32681dim_2rad_csv \
    --checkpoint \
    --random_seed 1337 \
    --learning_rate 1e-3 \
    --epochs 5 \
    --early_stop