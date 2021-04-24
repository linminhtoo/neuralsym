source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
conda activate neuralsym

# cannot use too high LR, will diverge slowly (loss increases > 20)
# higher bs --> faster training (using CPU)
# 8 sec/epoch on 1 RTX2080

# Highway, repeat with seed 77777777
python train.py \
    --model 'Highway' \
    --expt_name 'Highway_77777777_depth0_dim300_lr1e3_stop2_fac30_pat1' \
    --log_file 'Highway_77777777_depth0_dim300_lr1e3_stop2_fac30_pat1' \
    --do_train \
    --do_test \
    --prodfps_prefix 50k_1000000dim_2rad_to_32681_prod_fps \
    --labels_prefix 50k_1000000dim_2rad_to_32681_labels \
    --csv_prefix 50k_1000000dim_2rad_to_32681_csv \
    --bs 300 \
    --bs_eval 300 \
    --random_seed 77777777 \
    --learning_rate 1e-3 \
    --epochs 30 \
    --early_stop \
    --early_stop_patience 2 \
    --depth 0 \
    --hidden_size 300 \
    --lr_scheduler_factor 0.3 \
    --lr_scheduler_patience 1 \
    --checkpoint

# Highway, repeat with seed 20210423
# python train.py \
#     --model 'Highway' \
#     --expt_name 'Highway_20210423_depth0_dim300_lr1e3_stop2_fac30_pat1' \
#     --log_file 'Highway_20210423_depth0_dim300_lr1e3_stop2_fac30_pat1' \
#     --do_train \
#     --do_test \
#     --prodfps_prefix 50k_1000000dim_2rad_to_32681_prod_fps \
#     --labels_prefix 50k_1000000dim_2rad_to_32681_labels \
#     --csv_prefix 50k_1000000dim_2rad_to_32681_csv \
#     --bs 300 \
#     --bs_eval 300 \
#     --random_seed 20210423 \
#     --learning_rate 1e-3 \
#     --epochs 30 \
#     --early_stop \
#     --early_stop_patience 2 \
#     --depth 0 \
#     --hidden_size 300 \
#     --lr_scheduler_factor 0.3 \
#     --lr_scheduler_patience 1 \
#     --checkpoint

# FC, following Segler 2017 Neural, not better than Highway!
# python train.py \
#     --model 'FC' \
#     --expt_name 'FC512Elu_lr1e3_stop2_fac30_pat1' \
#     --log_file 'FC512Elu_lr1e3_stop2_fac30_pat1' \
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
#     --hidden_size 512 \
#     --lr_scheduler_factor 0.3 \
#     --lr_scheduler_patience 1 \
#     --checkpoint

# other experiments
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