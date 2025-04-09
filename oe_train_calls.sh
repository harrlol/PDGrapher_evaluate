#!/bin/bash

# 4/7, previously exported all oe data assuming no healthy counterparts, training with 5 folds

# List of items
# 04072000 retraining models for the ones that have healthy counterparts
cell_lines='A549 MCF7 PC3 VCAP A375 HT29 HEK293T HA1E'
with_healthy='A549 MCF7 PC3 VCAP'

# Log file
log_file="/home/b-evelyntong/hl/lincs_lvl3_oe/train_0407_big/training_log.txt"

# Loop through each cell line
for cell_line in $cell_lines; do

    echo "Processing ${cell_line}" | tee -a $log_file

    # Define the paths
    forward_path="/home/b-evelyntong/hl/lincs_lvl3_oe/torch_geometric_4_7/data_forward_${cell_line}.pt"
    backward_path="/home/b-evelyntong/hl/lincs_lvl3_oe/torch_geometric_4_7/data_backward_${cell_line}.pt"
    edge_index_path="/home/b-evelyntong/hl/lincs_lvl3_oe/torch_geometric_4_7/edge_index_${cell_line}.pt"
    splits_path="/home/b-evelyntong/hl/lincs_lvl3_oe/splits_4_7/genetic/${cell_line}/random/5fold/splits.pt"

    # first check if the cell has healthy counterparts
    if echo "$with_healthy" | grep -q "$cell_line"; then
        echo "Cell line ${cell_line} has healthy counterparts, using forward data." | tee -a $log_file
        python multiple_folds_test.py --forward_path $forward_path \
        --backward_path $backward_path --edge_index_path $edge_index_path --splits_path $splits_path --use_forward_data \
        >> $log_file 2>&1
    else
        echo "Cell line ${cell_line} does not have healthy counterparts, not using forward data." | tee -a $log_file
        python multiple_folds_test.py --forward_path $forward_path \
        --backward_path $backward_path --edge_index_path $edge_index_path --splits_path $splits_path \
        >> $log_file 2>&1
    fi

    if [ $? -eq 0 ]; then
        echo "${cell_line} training completed successfully." | tee -a $log_file
    else
        echo "Error occurred during training for ${cell_line}." | tee -a $log_file
        continue
    fi

    # Move output files
    output_dir="/home/b-evelyntong/hl/lincs_lvl3_oe/train_0407_big/"
    mkdir -p "$output_dir"
    mv "/home/b-evelyntong/hl/examples/PDGrapher/" "$output_dir"
    mv "/home/b-evelyntong/hl/lincs_lvl3_oe/train_0407_big/PDGrapher" "/home/b-evelyntong/hl/lincs_lvl3_oe/train_0407_big/${cell_line}"
    mkdir "/home/b-evelyntong/hl/examples/PDGrapher/"

done
