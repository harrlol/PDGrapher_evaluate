#!/bin/bash

# log file with timestamp
timestamp=$(date +"%Y-%m-%d_%H%M%S")
log_file="~/hl/data_processing_log_${timestamp}.txt"

# both stdout and stderr will be logged to the file
# and also printed to the console
exec > >(tee -i $log_file)
exec 2>&1

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
ppi_file_dir="~/hl/ppi_all_genes_edgelist.txt"
cosmic_file_dir="~/hl/CosmicCLP_MutantExport_only_verified_and_curated.csv"

# Download the data for oe
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/cellinfo_beta.txt -O ${oe_dir}/raw/oe/cellinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt -O ${oe_dir}/raw/oe/geneinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt -O ${oe_dir}/raw/oe/instinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt -O ${oe_dir}/raw/oe/compoundinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_ctl_n188708x12328.gctx -O ${oe_dir}/raw/oe/level3_beta_ctl_n188708x12328.gctx
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_trt_oe_n131668x12328.gctx -O ${oe_dir}/raw/oe/level3_beta_trt_oe_n131668x12328.gctx

# Download the data for original xpr
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/cellinfo_beta.txt -O ${oe_dir}/raw/xpr/cellinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt -O ${oe_dir}/raw/xpr/geneinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt -O ${oe_dir}/raw/xpr/instinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt -O ${oe_dir}/raw/xpr/compoundinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_ctl_n188708x12328.gctx -O ${oe_dir}/raw/xpr/level3_beta_ctl_n188708x12328.gctx
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_trt_xpr_n420583x12328.gctx -O ${oe_dir}/raw/xpr/level3_beta_trt_xpr_n420583x12328.gctx
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_trt_sh_n453175x12328.gctx -O ${oe_dir}/raw/xpr/level3_beta_trt_sh_n453175x12328.gctx

# process xpr data
python process_lincs_oe_npz.py \
    --data_dir "${oe_dir}/raw/xpr" \
    --out_dir "${oe_dir}/processed_npz/xpr"

# process oe data
python process_lincs_oe_npz.py \
    --data_dir "${oe_dir}/raw/oe" \
    --out_dir "${oe_dir}/processed_npz/oe"

# process health data for both directories
python process_lincs_oe_healthy_npz.py \
    --data_dir "${oe_dir}/raw/xpr" \
    --out_dir "${oe_dir}/processed_npz/xpr"
python process_lincs_oe_healthy_npz.py \
    --data_dir "${oe_dir}/raw/oe" \
    --out_dir "${oe_dir}/processed_npz/oe"

# export pt
python export_lincs_oe_for_torch_geometric.py \
  --data_root_dir "${oe_dir}/processed_npz/oe" \
  --ppi_edge_list_path "${ppi_file_dir}" \
  --gene_info_path "${oe_dir}/processed_npz/oe/gene_info.txt" \
  --outdir "${oe_dir}/torch_export/oe" \
  --cosmic_file "${cosmic_file_dir}"

# need to change the cell line loop for xpr, but this should work for now...