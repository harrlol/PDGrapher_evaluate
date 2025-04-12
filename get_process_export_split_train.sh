#!/bin/bash

# log file with timestamp
timestamp=$(date +"%Y-%m-%d_%H%M%S")
log_file="/home/b-evelyntong/hl/pipeline_log_${timestamp}.txt"
cell_lines_oe='A549 MCF7 PC3 VCAP A375 HT29 HEK293T HA1E'
cell_lines_xpr='A549 MCF7 PC3 A375 HT29 ES2 BICR6 YAPC AGS U251MG'

# both stdout and stderr will be logged to the file
# and also printed to the console
exec > >(tee $log_file) 2>&1

# note I do need to clone git repo before this, and migrate all my scripts over here (~/hl)
pdgrapher_dir="/home/b-evelyntong/hl/PDGrapher"
oe_dir="/home/b-evelyntong/hl/lincs_lvl3_oe"
mkdir -p ${oe_dir}
mkdir -p ${oe_dir}/raw
mkdir -p ${oe_dir}/raw/oe
mkdir -p ${oe_dir}/raw/xpr
mkdir -p ${oe_dir}/processed_npz
mkdir -p ${oe_dir}/processed_npz/oe
mkdir -p ${oe_dir}/processed_npz/xpr
mkdir -p ${oe_dir}/torch_export
mkdir -p ${oe_dir}/torch_export/oe
mkdir -p ${oe_dir}/torch_export/xpr
mkdir -p ${oe_dir}/splits
mkdir -p ${oe_dir}/splits/xpr
mkdir -p ${oe_dir}/splits/oe
ppi_file_dir="/home/b-evelyntong/hl/ppi_all_genes_edgelist.txt"
cosmic_file_dir="/home/b-evelyntong/hl/CosmicCLP_MutantExport_only_verified_and_curated.csv"


# Download the data for original xpr
echo "Downloading data for xpr..."
# wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/cellinfo_beta.txt -O ${oe_dir}/raw/xpr/cellinfo_beta.txt
# wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt -O ${oe_dir}/raw/xpr/geneinfo_beta.txt
# wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt -O ${oe_dir}/raw/xpr/instinfo_beta.txt
# wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt -O ${oe_dir}/raw/xpr/compoundinfo_beta.txt
# wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_ctl_n188708x12328.gctx -O ${oe_dir}/raw/xpr/level3_beta_ctl_n188708x12328.gctx
# wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_trt_xpr_n420583x12328.gctx -O ${oe_dir}/raw/xpr/level3_beta_trt_xpr_n420583x12328.gctx
# wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_trt_sh_n453175x12328.gctx -O ${oe_dir}/raw/xpr/level3_beta_trt_sh_n453175x12328.gctx

# # Download the data for oe
echo "Downloading data for oe..."
# wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/cellinfo_beta.txt -O ${oe_dir}/raw/oe/cellinfo_beta.txt
# wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt -O ${oe_dir}/raw/oe/geneinfo_beta.txt
# wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt -O ${oe_dir}/raw/oe/instinfo_beta.txt
# wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt -O ${oe_dir}/raw/oe/compoundinfo_beta.txt
# wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_ctl_n188708x12328.gctx -O ${oe_dir}/raw/oe/level3_beta_ctl_n188708x12328.gctx
# wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_trt_oe_n131668x12328.gctx -O ${oe_dir}/raw/oe/level3_beta_trt_oe_n131668x12328.gctx

# process xpr data
echo "Processing xpr data..."
# python process_lincs_pert_ctl_npz.py \
#     --data_dir "${oe_dir}/raw/xpr" \
#     --out_dir "${oe_dir}/processed_npz/xpr" \
#     --true_pert_type "trt_xpr"

# # process oe data
echo "Processing oe data..."
# python process_lincs_pert_ctl_npz.py \
#     --data_dir "${oe_dir}/raw/oe" \
#     --out_dir "${oe_dir}/processed_npz/oe" \
#     --true_pert_type "trt_oe"

# process health data for both directories
echo "Processing healthy data for xpr..."
# python process_lincs_healthy_npz.py \
#     --data_dir "${oe_dir}/raw/xpr" \
#     --out_dir "${oe_dir}/processed_npz/xpr"
echo "Processing healthy data for oe..."
# python process_lincs_healthy_npz.py \
#     --data_dir "${oe_dir}/raw/oe" \
#     --out_dir "${oe_dir}/processed_npz/oe"

# export pt for xpr
echo "Exporting pt for xpr..."
# python export_lincs_xpr_for_torch_geometric.py \
#   --data_root_dir "${oe_dir}/processed_npz/xpr" \
#   --ppi_edge_list_path "${ppi_file_dir}" \
#   --gene_info_path "${oe_dir}/processed_npz/xpr/gene_info.txt" \
#   --outdir "${oe_dir}/torch_export/xpr" \
#   --cosmic_file "${cosmic_file_dir}"

# # export pt for oe
echo "Exporting pt for oe..."
# python export_lincs_oe_for_torch_geometric.py \
#   --data_root_dir "${oe_dir}/processed_npz/oe" \
#   --ppi_edge_list_path "${ppi_file_dir}" \
#   --gene_info_path "${oe_dir}/processed_npz/oe/gene_info.txt" \
#   --outdir "${oe_dir}/torch_export/oe" \
#   --cosmic_file "${cosmic_file_dir}"

# generate splits for xpr
echo "Generating splits for xpr..."
python create_splits_pert.py \
  --nfolds 1 \
  --cell_lines_keep "${cell_lines_xpr}" \
  --out_dir "${oe_dir}/splits/xpr" \
  --data_root "${oe_dir}/torch_export/xpr"

# generate splits for oe
echo "Generating splits for oe..."
python create_splits_pert.py \
  --nfolds 1 \
  --cell_lines_keep "${cell_lines_oe}" \
  --out_dir "${oe_dir}/splits/oe" \
  --data_root "${oe_dir}/torch_export/oe"

# call a training run for xpr
echo "Training for xpr..."
bash train_calls.sh \
  -c  "${cell_lines_xpr}" \
  -d "/home/b-evelyntong/hl/lincs_lvl3_oe/torch_export/xpr" \
  -s "/home/b-evelyntong/hl/lincs_lvl3_oe/splits/xpr" \
  -o "/home/b-evelyntong/hl/training_history/train_0411_test/xpr" \
  -f 1 \
  -e 1

# call a training run for oe
echo "Training for oe..."
bash train_calls.sh \
  -c  "${cell_lines_oe}" \
  -d "/home/b-evelyntong/hl/lincs_lvl3_oe/torch_export/oe" \
  -s "/home/b-evelyntong/hl/lincs_lvl3_oe/splits/oe" \
  -o "/home/b-evelyntong/hl/training_history/train_0411_test/oe" \
  -f 1 \
  -e 1