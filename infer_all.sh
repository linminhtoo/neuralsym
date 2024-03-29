python3 infer_all.py \
    --csv_prefix 50k_1000000dim_2rad_to_32681_csv \
    --labels_prefix 50k_1000000dim_2rad_to_32681_labels \
    --templates_file 50k_training_templates \
    --rxn_smi_prefix 50k_clean_rxnsmi_noreagent_allmapped_canon \
    --log_file 'infer_77777777_highway_depth0_dim300' \
    --prodfps_prefix 50k_1000000dim_2rad_to_32681_prod_fps \
    --hidden_size 300 \
    --depth 0 \
    --topk 200 \
    --maxk 200 \
    --model Highway \
    --expt_name 'Highway_77777777_depth0_dim300_lr1e3_stop2_fac30_pat1' \
    --seed 77777777