#!/bin/bash

### SELF NOTE ###
# directly execute this script with bash run_experiments.sh, and specify experiment variables inside the script
# defines, passes down, and creates experiment level variables, e.g. fold, epoch, cell line
# also collects results to be used for plotting

# define variables
cell_lines_oe='A549 MCF7 PC3 VCAP A375 HT29 HEK293T HA1E'
cell_lines_xpr='A549 MCF7 PC3 A375 HT29 ES2 BICR6 YAPC AGS U251MG' 
cell_lines_short='A549 A375' # remember to use --all on master_pipeline.sh to use all cell lines
date=$(date +%m%d)
reps=(A)
num_epochs=30
num_folds=2

# main loop
for rep in ${reps[@]}; do
    # dir at the experiment result level
    result_dir_name="report_${date}_${num_folds}_${num_epochs}_${rep}"
    result_dir="/home/b-evelyntong/hl/training_history/${result_dir_name}"
    mkdir -p "${result_dir}"

    # dir at the model level
    baseline_dir="/home/b-evelyntong/hl/training_history/train_${date}_baseline_${num_folds}_${num_epochs}_${rep}"
    mkdir -p "${baseline_dir}"
    gcnconvmod_dir="/home/b-evelyntong/hl/training_history/train_${date}_gcnconvmod_${num_folds}_${num_epochs}_${rep}"
    mkdir -p "${gcnconvmod_dir}"
    embedmod_dir="/home/b-evelyntong/hl/training_history/train_${date}_embedmod_${num_folds}_${num_epochs}_${rep}"    
    mkdir -p "${embedmod_dir}"
    bothmod_dir="/home/b-evelyntong/hl/training_history/train_${date}_bothmod_${num_folds}_${num_epochs}_${rep}"    
    mkdir -p "${bothmod_dir}"

    # run the baseline models
    bash master_pipeline.sh \
        -o ${baseline_dir} \
        -f ${num_folds} \
        -e ${num_epochs} \
        -b main

    # train the gcnconv modified models
    bash master_pipeline.sh \
        -o ${gcnconvmod_dir} \
        -f ${num_folds} \
        -e ${num_epochs} \
        -b feat/gcnconv-mod

    # train the embedding modified models
    bash master_pipeline.sh \
        -o ${embedmod_dir} \
        -f ${num_folds} \
        -e ${num_epochs} \
        -b feat/embed-mod

    # train models with both mods
    bash master_pipeline.sh \
        -o ${bothmod_dir} \
        -f ${num_folds} \
        -e ${num_epochs} \
        -b feat/gcnconv-embed-mod
    
    # move the metrics
    cp ${baseline_dir}/oe/training_log_*.txt ${result_dir}/baseline_oe.txt
    cp ${baseline_dir}/xpr/training_log_*.txt ${result_dir}/baseline_xpr.txt
    cp ${gcnconvmod_dir}/oe/training_log_*.txt ${result_dir}/gcnconvmod_oe.txt
    cp ${gcnconvmod_dir}/xpr/training_log_*.txt ${result_dir}/gcnconvmod_xpr.txt
    cp ${embedmod_dir}/oe/training_log_*.txt ${result_dir}/embedmod_oe.txt
    cp ${embedmod_dir}/xpr/training_log_*.txt ${result_dir}/embedmod_xpr.txt
    cp ${bothmod_dir}/oe/training_log_*.txt ${result_dir}/bothmod_oe.txt
    cp ${bothmod_dir}/xpr/training_log_*.txt ${result_dir}/bothmod_xpr.txt

    # evaluate the current run results
    echo "Evaluating results..."
    python compile_metrics_plot.py \
    --project_directory "${result_dir}" \
    --cell_lines "${cell_lines_short}"
done