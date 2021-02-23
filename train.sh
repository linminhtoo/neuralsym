source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate neuralsym

# cannot use too high LR, will diverge slowly (loss increases > 20)
# higher bs --> faster training (using CPU)
# 8 sec/epoch on 1 RTX2080
python train.py \
    --expt_name 'depth0_dim500_lr1e3_stop2_fac30_pat1_try' \
    --log_file 'depth0_dim500_lr1e3_stop2_fac30_pat1_try' \
    --do_train \
    --do_test \
    --prodfps_prefix 50k_1000000dim_2rad_to_32681_prod_fps \
    --labels_prefix 50k_1000000dim_2rad_to_32681_labels \
    --csv_prefix 50k_1000000dim_2rad_to_32681_csv \
    --bs 300 \
    --bs_eval 300 \
    --random_seed 1337 \
    --learning_rate 1e-3 \
    --epochs 30 \
    --early_stop \
    --early_stop_patience 2 \
    --depth 0 \
    --hidden_size 300 \
    --lr_scheduler_factor 0.3 \
    --lr_scheduler_patience 1 \
    --checkpoint

# python train.py \
#     --expt_name 'depth0_dim750_lr1e3_stop2_fac30_pat1' \
#     --log_file 'depth0_dim750_lr1e3_stop2_fac30_pat1' \
#     --do_train \
#     --do_test \
#     --prodfps_prefix 50k_1000000dim_2rad_to_32681_prod_fps \
#     --labels_prefix 50k_1000000dim_2rad_to_32681_labels \
#     --csv_prefix 50k_1000000dim_2rad_to_32681_csv \
#     --bs 300 \
#     --bs_eval 300 \
#     --random_seed 1337 \
#     --learning_rate 1e-3 \
#     --epochs 30 \
#     --early_stop \
#     --early_stop_patience 2 \
#     --depth 0 \
#     --hidden_size 750 \
#     --lr_scheduler_factor 0.3 \
#     --lr_scheduler_patience 1

# python train.py \
#     --expt_name 'depth0_dim1000_lr1e3_stop2_fac30_pat1' \
#     --log_file 'depth0_dim1000_lr1e3_stop2_fac30_pat1' \
#     --do_train \
#     --do_test \
#     --prodfps_prefix 50k_1000000dim_2rad_to_32681_prod_fps \
#     --labels_prefix 50k_1000000dim_2rad_to_32681_labels \
#     --csv_prefix 50k_1000000dim_2rad_to_32681_csv \
#     --bs 300 \
#     --bs_eval 300 \
#     --random_seed 1337 \
#     --learning_rate 1e-3 \
#     --epochs 30 \
#     --early_stop \
#     --early_stop_patience 2 \
#     --depth 0 \
#     --hidden_size 1000 \
#     --lr_scheduler_factor 0.3 \
#     --lr_scheduler_patience 1