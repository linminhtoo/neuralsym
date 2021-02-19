source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate neuralsym

# cannot use too high LR, will diverge slowly (loss increases > 20)
# higher bs --> faster training (using CPU)
# 8 sec/epoch on 1 GPU lmao
python train.py \
    --expt_name 'prelim_lr1e3_ep60_bs300_fixed' \
    --do_train \
    --do_test \
    --prodfps_prefix 50k_1000000dim_2rad_to_32681_prod_fps \
    --labels_prefix 50k_1000000dim_2rad_to_32681_labels \
    --csv_prefix 50k_1000000dim_2rad_to_32681_csv \
    --bs 300 \
    --bs_eval 300 \
    --random_seed 1337 \
    --learning_rate 1e-3 \
    --epochs 60 \
    --early_stop \
    --early_stop_patience 4 \
    --depth 1 \
    --hidden_size 100

    # --checkpoint \
    # don't checkpoint first as model is very big, 600M