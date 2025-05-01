#!/bin/bash

# info
usage() {
  echo "Usage: $0 [-c \"CELL_LINE1 CELL_LINE2\"] [-w \"WITH_HEALTHY\"] [-l LOG_FILE] [-o OUTPUT_DIR]"
  echo "  -c  Cell lines to process (space-separated string)"
  echo "  -l  Path to the log file"
  echo "  -o  Output directory"
  echo "  -d  Torch data directory"
  echo "  -s  Splits directory"
  exit 1
}

# parse commands
while getopts ":c:d:s:f:o:e:p:" opt; do
  case $opt in
    c) cell_lines="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    d) torch_dir="$OPTARG" ;;
    s) splits_dir="$OPTARG" ;;
    p) embed="$OPTARG" ;;
    f) n_folds="$OPTARG" ;;
    e) n_epoch="$OPTARG" ;;
    *) usage ;;
  esac
done

# make sure outdir exists
mkdir -p "${output_dir}"

# default values
with_healthy='A549 MCF7 PC3 VCAP'
timestamp=$(date +"%Y-%m-%d_%H%M%S")
start_time=$(date +%s)
log_file="${output_dir}/training_log_${timestamp}.txt"

# echo some parameters
echo "Run started at ${timestamp}" | tee -a $log_file
echo "Cell lines: $cell_lines" | tee -a $log_file
echo "Torch data directory: $torch_dir" | tee -a $log_file
echo "Splits directory: $splits_dir" | tee -a $log_file
echo "Output directory: $output_dir" | tee -a $log_file
echo "Number of folds: $n_folds" | tee -a $log_file

cd ./PDGrapher
echo "##############################################################" | tee -a $log_file
echo "Using branch: $(git rev-parse --abbrev-ref HEAD)" | tee -a $log_file
echo "##############################################################" | tee -a $log_file
cd ..

# Loop through each cell line
for cell_line in $cell_lines; do

    echo "Processing ${cell_line}" | tee -a $log_file
    mkdir -p "${output_dir}/${cell_line}"

    # Define the paths
    forward_path="${torch_dir}/data_forward_${cell_line}.pt"
    backward_path="${torch_dir}/data_backward_${cell_line}.pt"
    edge_index_path="${torch_dir}/edge_index_${cell_line}.pt"
    splits_path="${splits_dir}/genetic/${cell_line}/random/${n_folds}fold/splits.pt"

    # first check if the cell has healthy counterparts, then check embedding
    if echo "$with_healthy" | grep -q "$cell_line"; then
        if [ -z "$embed" ]; then
            echo "Cell line ${cell_line} has healthy counterparts, using forward data." | tee -a $log_file
            python n_folds_train_call.py --forward_path $forward_path \
            --backward_path $backward_path --edge_index_path $edge_index_path \
            --splits_path $splits_path --n_epoch ${n_epoch} --use_forward_data \
            --output_dir "${output_dir}/${cell_line}" >> $log_file 2>&1
        else
            echo "Cell line ${cell_line} has healthy counterparts, using forward data and embedding." | tee -a $log_file
            python n_folds_train_call.py --forward_path $forward_path \
            --backward_path $backward_path --edge_index_path $edge_index_path \
            --splits_path $splits_path --embedding_path $embed --n_epoch ${n_epoch} \
            --output_dir "${output_dir}/${cell_line}" --use_forward_data >> $log_file 2>&1
        fi
    else
        if [ -z "$embed" ]; then
            echo "Cell line ${cell_line} does not have healthy counterparts, not using forward data." | tee -a $log_file
            python n_folds_train_call.py --forward_path $forward_path \
            --backward_path $backward_path --edge_index_path $edge_index_path \
            --splits_path $splits_path --n_epoch ${n_epoch} \
            --output_dir "${output_dir}/${cell_line}" >> $log_file 2>&1
        else
            echo "Cell line ${cell_line} does not have healthy counterparts, not using forward data and using embedding." | tee -a $log_file
            python n_folds_train_call.py --forward_path $forward_path \
            --backward_path $backward_path --edge_index_path $edge_index_path \
            --splits_path $splits_path --embedding_path $embed --n_epoch ${n_epoch} \
            --output_dir "${output_dir}/${cell_line}" >> $log_file 2>&1
        fi
    fi

    if [ $? -eq 0 ]; then
        echo "${cell_line} training completed successfully." | tee -a $log_file
    else
        echo "Error occurred during training for ${cell_line}." | tee -a $log_file
        continue
    fi

done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Training complete in ${elapsed} seconds." | tee -a $log_file