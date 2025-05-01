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

branch="main"

# parse commands
while getopts ":f:o:e:b:c:" opt; do
  case $opt in
    o) project_dir="$OPTARG" ;;
    f) n_folds="$OPTARG" ;;
    e) n_epoch="$OPTARG" ;;
    c) cell_line_model="$OPTARG" ;;
    b) branch="$OPTARG" ;;
    *) usage ;;
  esac
done

# check current branch
cd PDGrapher || { echo "❌ Could not cd into PDGrapher"; exit 1; }
git fetch
current_branch=$(git rev-parse --abbrev-ref HEAD)

if [ "$current_branch" != "$branch" ]; then
    echo "🔄 Switching to branch: $branch"
    git checkout "$branch"
else
    echo "✅ Already on correct branch: $branch"
fi
cd ..

mkdir -p ${project_dir}
# log file with timestamp
timestamp=$(date +"%Y-%m-%d_%H%M%S")
log_file="${project_dir}/pipeline_log_${timestamp}.txt"

# specify all to use all cell lines, default small set
if [ "$cell_line_mode" = "all" ]; then
    cell_lines_oe='A549 MCF7 PC3 VCAP A375 HT29 HEK293T HA1E'
    cell_lines_xpr='A549 MCF7 PC3 A375 HT29 ES2 BICR6 YAPC AGS U251MG'
else
    cell_lines_oe='A549 A375'
    cell_lines_xpr='A549 A375'
fi

# both stdout and stderr will be logged to the file
# and also printed to the console
exec > >(tee $log_file) 2>&1

# note I do need to clone git repo before this, and migrate all my scripts over here (~/hl)
pdgrapher_dir="/home/b-evelyntong/hl/PDGrapher"
oe_dir="/home/b-evelyntong/hl/lincs_lvl3_oe"
ppi_file_dir="/home/b-evelyntong/hl/ppi_all_genes_edgelist.txt"
cosmic_file_dir="/home/b-evelyntong/hl/CosmicCLP_MutantExport_only_verified_and_curated.csv"

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
mkdir -p "/home/b-evelyntong/hl/training_history"



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
# python create_splits_pert.py \
#   --nfolds ${n_folds} \
#   --cell_lines_keep "${cell_lines_xpr}" \
#   --out_dir "${oe_dir}/splits/xpr" \
#   --data_root "${oe_dir}/torch_export/xpr"

# generate splits for oe
echo "Generating splits for oe..."
# python create_splits_pert.py \
#   --nfolds ${n_folds} \
#   --cell_lines_keep "${cell_lines_oe}" \
#   --out_dir "${oe_dir}/splits/oe" \
#   --data_root "${oe_dir}/torch_export/oe"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# check if we will require pre-computed embeddings
if [[ "$branch" == *"embed"* ]]; then
  # call a training run for xpr
  echo "Training for xpr..."
  bash train_calls.sh \
    -c  "${cell_lines_xpr}" \
    -d "${oe_dir}/torch_export/xpr" \
    -s "${oe_dir}/splits/xpr" \
    -o "${project_dir}/xpr" \
    -p "/home/b-evelyntong/hl/embedding_matrix_xpr.pt" \
    -f ${n_folds} \
    -e ${n_epoch}

  # call a training run for oe
  echo "Training for oe..."
  bash train_calls.sh \
    -c  "${cell_lines_oe}" \
    -d "${oe_dir}/torch_export/oe" \
    -s "${oe_dir}/splits/oe" \
    -o "${project_dir}/oe" \
    -p "/home/b-evelyntong/hl/embedding_matrix_oe.pt" \
    -f ${n_folds} \
    -e ${n_epoch}
else
  # call a training run for xpr
  echo "Training for xpr..."
  bash train_calls.sh \
    -c  "${cell_lines_xpr}" \
    -d "${oe_dir}/torch_export/xpr" \
    -s "${oe_dir}/splits/xpr" \
    -o "${project_dir}/xpr" \
    -f ${n_folds} \
    -e ${n_epoch}

  # call a training run for oe
  echo "Training for oe..."
  bash train_calls.sh \
    -c  "${cell_lines_oe}" \
    -d "${oe_dir}/torch_export/oe" \
    -s "${oe_dir}/splits/oe" \
    -o "${project_dir}/oe" \
    -f ${n_folds} \
    -e ${n_epoch}
fi

echo "Training completed for all cell lines at $(date +"%Y-%m-%d_%H%M%S")"