#!/bin/bash

# 423_multi
# this is the script I use to manually change hyper parameters and automate experiments

# remember to use --all to use all cell lines
cell_lines_oe='A549 MCF7 PC3 VCAP A375 HT29 HEK293T HA1E'
cell_lines_xpr='A549 MCF7 PC3 A375 HT29 ES2 BICR6 YAPC AGS U251MG'
cell_lines_short='A549 A375'

date=0423
reps=(A B C)
num_epochs=1
num_folds=3

for rep in ${reps[@]}; do
    # definitions
    result_dir_name="report_${date}_${num_folds}_${num_epochs}_${rep}"
    result_dir="/home/b-evelyntong/hl/training_history/${result_dir_name}"
    mkdir -p "${result_dir}"
    baseline_dir="/home/b-evelyntong/hl/training_history/train_${date}_baseline_${num_folds}_${num_epochs}_${rep}"
    mod_dir="/home/b-evelyntong/hl/training_history/train_${date}_gcnmod_${num_folds}_${num_epochs}_${rep}"

    # run the baseline and modified models
    bash master_pipeline.sh \
        -o ${baseline_dir} \
        -f ${num_folds} \
        -e ${num_epochs} \
        -b main
    bash master_pipeline.sh \
        -o ${mod_dir} \
        -f ${num_folds} \
        -e ${num_epochs} \
        -b feat/gcnconv-mod

    # move the metrics
    cp ${baseline_dir}/oe/training_log_*.txt ${result_dir}/baseline_oe.txt
    cp ${baseline_dir}/xpr/training_log_*.txt ${result_dir}/baseline_xpr.txt
    cp ${mod_dir}/oe/training_log_*.txt ${result_dir}/gcnmod_oe.txt
    cp ${mod_dir}/xpr/training_log_*.txt ${result_dir}/gcnmod_xpr.txt

    # evaluate the current run results
    echo "Evaluating results..."
    python compile_metrics_plot.py \
    --project_directory "${result_dir}" \
    --cell_lines "${cell_lines_short}"
done