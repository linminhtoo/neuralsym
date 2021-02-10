source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate neuralsym

# cannot use too high LR, will diverge slowly (loss increases > 20)
# higher bs --> faster training (using CPU)
python train.py \
    --expt_name 'prelim_lr1e3_ep60_bs800_depth3_hidden300' \
    --do_train \
    --do_test \
    --prodfps_prefix 50k_32681dim_2rad_prod_fps \
    --labels_prefix 50k_32681dim_2rad_labels \
    --csv_prefix 50k_32681dim_2rad_csv \
    --checkpoint \
    --bs 800 \
    --bs_eval 300 \
    --random_seed 1337 \
    --learning_rate 1e-3 \
    --epochs 60 \
    --early_stop \
    --early_stop_patience 4 \
    --depth 5 \
    --hidden_size 512

