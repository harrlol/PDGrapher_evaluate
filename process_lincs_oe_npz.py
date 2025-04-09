'''
adapted from data/scripts/lincs/process_data.py, this script specifically processes lincs lvl3 overexpression data into npz
takes data directory path and desired output directory path as inputs, and populates output dir with npz files, while also making some plots and logs
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
import os.path as osp
import argparse
from collections import Counter
from random import sample
from sklearn.preprocessing import MinMaxScaler

def stats_data(inst_info_xpr, matrix_xpr, matrix_ctl, gene_info, outdir):
	dict_symbol_id = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))

	####Data exploration -- GE values of genes that are perturbed (!)
	#Get GE value for each gene perturbed by CRISPR
	values_pert = {}
	values_control = {}
	for i in range(len(inst_info_xpr)):
		gene_symbol = inst_info_xpr.at[i, 'cmap_name']
		if gene_symbol in dict_symbol_id:					#if the cmap_name of gene is in the gene_info
			sample_id = inst_info_xpr.at[i, 'sample_id']
			gene_id = dict_symbol_id[gene_symbol]
			if gene_id in values_pert:
				values_pert[gene_id].append(matrix_xpr.at[gene_id, sample_id])
			else:
				values_pert[gene_id] = [matrix_xpr.at[gene_id, sample_id]]

	for gene_symbol in list(set(inst_info_xpr['cmap_name'])):
		if gene_symbol in dict_symbol_id:				#if the cmap_name of gene is in the gene_info
			gene_id = dict_symbol_id[gene_symbol]
			values_control[gene_id] = [matrix_ctl.loc[gene_id]]

	for key in values_pert:
		values_pert[key] = np.mean(values_pert[key])

	for key in values_control:
		values_control[key] = np.mean(values_control[key])

	fig, (ax1, ax2) = plt.subplots(2, figsize=(16,6))
	ax1.hist(values_pert.values())
	ax2.hist(values_control.values())
	ax1.set_title('Values of perturbed genes (avg) - Over-expression')
	ax2.set_title('Values of genes in control (avg)')
	fig.savefig(osp.join(outdir,'exploration_ge_oe.png'))

	return


def stats_control(inst_info_ctl, log_handle):
    #Stats unique cell lines
    log_handle.write('Unique cell lines:\t{}:\n'.format(len(set(inst_info_ctl['cell_iname']))))
    for c in list(set(inst_info_ctl['cell_iname'])):
        log_handle.write('\t{}\n'.format(c))

    log_handle.write('\n\n')

	#Stats on dosages
    df_ctl = pd.DataFrame(inst_info_ctl[['cmap_name', 'cell_iname', 'pert_idose']].groupby(['cmap_name', 'cell_iname']).apply(lambda x: x['pert_idose'].unique()))
    df_ctl = pd.DataFrame([(i, len(df_ctl.loc[i][0])) for i in df_ctl.index], columns =['cmap_name-cell_line', 'n_doses'])
    log_handle.write('Stats on dosages and timepoints\n')
    log_handle.write('\n------\nHOW MANY DOSES ARE THERE FOR CMAP_NAME-CELL LINE PAIRS?\n------\n')
    for index,value in pd.Series.iteritems(pd.DataFrame(df_ctl['n_doses'])['n_doses'].describe()):
        log_handle.write('{}:\t{}\n'.format(index, value))
		
    log_handle.write('\n')
    log_handle.write('Number of pairs with more than 1 dose:\t{}/{}\n'.format(sum(df_ctl['n_doses']>1), len(df_ctl)))
    log_handle.write('Number of pairs with more than 2 doses:\t{}/{}\n\n'.format(sum(df_ctl['n_doses']>2), len(df_ctl)))

	#Stats on timepoints
    df_ctl = pd.DataFrame(inst_info_ctl[['cmap_name', 'cell_iname', 'pert_time']].groupby(['cmap_name', 'cell_iname']).apply(lambda x: x['pert_time'].unique()))
    df_ctl = pd.DataFrame([(i, len(df_ctl.loc[i][0])) for i in df_ctl.index], columns =['cmap_name-cell_line', 'n_times'])

    log_handle.write('\n------\nHOW MANY TIMEPOINTS ARE THERE FOR CMAP_NAME-CELL LINE PAIRS?\n------\n')
    for index,value in pd.Series.iteritems(pd.DataFrame(df_ctl['n_times'])['n_times'].describe()):
        log_handle.write('{}:\t{}\n'.format(index, value))

    log_handle.write('\n')
    log_handle.write('Number of pairs with more than 1 timepoint:\t{}/{}\n'.format(sum(df_ctl['n_times']>1), len(df_ctl['n_times'])))
    log_handle.write('Number of pairs with more than 2 timepoints:\t{}/{}\n\n'.format(sum(df_ctl['n_times']>2), len(df_ctl['n_times'])))

    log_handle.write('\nUSING THEM ALL FOR NOW\n')

	#Types of vectors
    log_handle.write('Number of vectors:\t{}:\n'.format(len(set(inst_info_ctl['cmap_name']))))
    df=pd.DataFrame.from_dict(Counter(inst_info_ctl['cmap_name']), orient='index')
    df = df.sort_values(by=0)
    for i, v in enumerate(zip(df.index, df[0])):
        log_handle.write('{}:\t{}\n'.format(v[0], v[1]))

	#Number of controls per cell line
    replicates = inst_info_ctl.groupby(['cell_iname']).size()
    log_handle.write('Number of replicates per cell line (different doses, times, vectors:\t{}:\n)'.format(replicates))
    df=pd.DataFrame.from_dict(Counter(inst_info_ctl['cell_iname']), orient='index')
    df = df.sort_values(by=0)
    for i, v in enumerate(zip(df.index, df[0])):
        log_handle.write('{}:\t{}\n'.format(v[0], v[1]))

    return


def load_and_eda(DATA_ROOT, outdir, log_handle):

    # load in metadata
    inst_info = pd.read_csv(os.path.join(DATA_ROOT, 'instinfo_beta.txt'), sep="\t", low_memory=False)
    gene_info = pd.read_csv(os.path.join(DATA_ROOT, 'geneinfo_beta.txt'), sep="\t", low_memory=False)

    # filters for only ctrl and oe and no failure
    inst_info_ctl = inst_info[np.logical_and(inst_info['pert_type'] == 'ctl_vector', inst_info['failure_mode'].isna()) ].reset_index(inplace=False, drop=True)
    inst_info_oe = inst_info[np.logical_and(inst_info['pert_type'] == 'trt_oe', inst_info['failure_mode'].isna()) ].reset_index(inplace=False, drop=True)

    # reads in oe matrix save as pd df
    f = h5py.File(os.path.join(DATA_ROOT, 'level3_beta_trt_oe_n131668x12328.gctx'), 'r')
    matrix_oe = f['0']['DATA']['0']['matrix'][:].transpose()
    gene_ids_oe = f['0']['META']['ROW']['id'][:]
    sample_ids_oe = f['0']['META']['COL']['id'][:]
    matrix_oe = pd.DataFrame(matrix_oe, columns = sample_ids_oe.astype(str), index = gene_ids_oe.astype(int))
    del f

    # reads in ctrl matrix save as pd df
    f = h5py.File(os.path.join(DATA_ROOT, 'level3_beta_ctl_n188708x12328.gctx'), 'r')
    matrix_ctl = f['0']['DATA']['0']['matrix'][:].transpose()
    gene_ids_ctl = f['0']['META']['ROW']['id'][:]					#not in the same order as gene_ids_xpr
    sample_ids_ctl = f['0']['META']['COL']['id'][:]
    matrix_ctl = pd.DataFrame(matrix_ctl, columns = sample_ids_ctl.astype(str), index = gene_ids_ctl.astype(int))
    del f
	
    ####################### runs stats check on data ###############################
    # log stats
    log_handle.write('CONTROL\n------\n')
    log_handle.write('Control entries in inst_info metadata:\t{}\n'.format(len(inst_info_ctl)))
    log_handle.write('Control entries in data matrix:\t{}\n'.format(len(sample_ids_ctl)))
    log_handle.write('Overlap between inst_info metadata and sample ids in data matrix:\t{}\n'.format(len(set(inst_info_ctl['sample_id']).intersection(set(sample_ids_ctl.astype(str))))))
    log_handle.write('\n------\n')
	
    log_handle.write('Over-expression\n------\n')
    log_handle.write('Over-expression entries in inst_info metadata:\t{}\n'.format(len(inst_info_oe)))
    log_handle.write('Over-expression entries in data matrix:\t{}\n'.format(len(sample_ids_oe)))
    log_handle.write('Overlap between inst_info metadata and sample ids in data matrix:\t{}\n'.format(len(set(inst_info_oe['sample_id']).intersection(set(sample_ids_oe.astype(str))))))

    # generate plot looking at perturbed gene expression in oe vs ctrl
    stats_data(inst_info_oe, matrix_oe, matrix_ctl, gene_info, outdir)
	#################################################################################
	
    #re-order gene_info based on the order in gene_ids_xpr (rows of data)
    gene_info.index = gene_info['gene_id']
    gene_info = gene_info.loc[gene_ids_oe.astype(int)].reset_index(inplace=False, drop=True)
    gene_info.to_csv(osp.join(outdir, 'gene_info.txt'), index=False)

    # filter matrix for samples found in metadata
    # oe
    list_ids = list(inst_info_oe['sample_id'])
    matrix_oe = matrix_oe[list_ids]

    # ctrl, added step because there exists samples in metadata not in matrix
    list_ids = list(inst_info_ctl['sample_id'])
    list_ids = list(set(list_ids).intersection(set(matrix_ctl.columns.astype(str)))) # find intersection
    matrix_ctl = matrix_ctl[list_ids]

    # remove metadata diff for future steps
    inst_info_ctl.index = inst_info_ctl['sample_id']; inst_info_ctl = inst_info_ctl.loc[list_ids].reset_index(inplace=False, drop=True)
	
    return inst_info_oe, matrix_oe, inst_info_ctl, matrix_ctl, gene_info, log_handle


def filter_by_cell_line(inst_info_oe, matrix_oe, inst_info_ctl, matrix_ctl, gene_info, log_handle):
    log_handle.write('Filtering for high nFeature cell lines A549, PC3, A375, HEK293T, HA1E, MCF7, HT29, VCAP\n------\n')
    ############### repeated their steps to identify cell_lines_to_keep #################
    # top cell lines that have > 4000 genes overexpressed:
    # A549, PC3, A375, HEK293T, HA1E, MCF7, HT29, VCAP
    # overlap of kept cell lines between xpr and oe is 
    # A549, PC3, A375, MCF7, HT29
    #####################################################################################

    # filter for kept cell lines
    # oe
    keep_cell_lines = ['A549', 'PC3', 'A375', 'HEK293T', 'HA1E', 'MCF7', 'HT29', 'VCAP']
    keep_index = []
    for i in range(len(inst_info_oe)):
        if inst_info_oe.at[i, 'cell_iname'] in keep_cell_lines:
            keep_index.append(i)
    inst_info_oe = inst_info_oe.loc[keep_index].reset_index(inplace=False, drop=True)  # filter from metadata
    list_ids = list(inst_info_oe['sample_id'])	# obtain sample ID from metadata
    matrix_oe = matrix_oe[list_ids]	 # filtered matrix by that sample id
    log_handle.write('Over-expression:\t{} datapoints\n'.format(matrix_oe.shape[1]))

    # ctrl
    keep_index = []
    for i in range(len(inst_info_ctl)):
        if inst_info_ctl.at[i, 'cell_iname'] in keep_cell_lines:
            keep_index.append(i)
    inst_info_ctl = inst_info_ctl.loc[keep_index].reset_index(inplace=False, drop=True)  # filter from metadata
    list_ids = list(inst_info_ctl['sample_id'])	 # obtain sample ID from metadata
    matrix_ctl = matrix_ctl[list_ids]  # filter matrix by that sample id

    # generate some stats on metadata for the cell lines selected
    stats_control(inst_info_ctl, log_handle)

    # filter samples by whether the gene used for perturbation is found in gene_info
    known_genes = list(set(gene_info['gene_symbol']))
    keep_index = []
    for i in range(len(inst_info_oe)):
        if inst_info_oe.at[i, 'cmap_name'] in known_genes:
            keep_index.append(i)
    inst_info_oe = inst_info_oe.loc[keep_index].reset_index(inplace=False, drop=True)  # filter from metadata
    list_ids = list(inst_info_oe['sample_id'])	# obtain sample ID from metadata
    matrix_oe = matrix_oe[list_ids]  # filter matrix by that sample id
    log_handle.write('CONTROL:\t{} datapoints\n'.format(matrix_ctl.shape[1]))
	
    return inst_info_oe, matrix_oe, inst_info_ctl, matrix_ctl, gene_info, log_handle, keep_cell_lines


def combine_lognorm_save(inst_info_oe, matrix_oe, inst_info_ctl, matrix_ctl, log_handle, outdir, keep_cell_lines):
    log_handle.write('Concatenating expression matrix and metadata\n------\n')
    # concatenate metadata
    metadata = pd.concat([inst_info_oe, inst_info_ctl], axis=0).reset_index(inplace=False, drop=True)
    metadata.to_csv(osp.join(outdir, 'all_metadata.txt'))
    matrix = pd.concat([matrix_oe, matrix_ctl], axis=1)
    del(matrix_oe)
    del(matrix_ctl)

    log_handle.write('Lognorm and min-max scaling\n------\n')
    # sample some matrix values and check distribution
    mat_val = matrix.values.flatten()
    sampling = sample(range(len(mat_val)), int(0.001*len(mat_val)))
    mat_val = mat_val[sampling]
    fig, ax = plt.subplots(figsize=(16,6))
    ax.hist(mat_val)
    ax.set_title('Histogram of values')
    fig.savefig(osp.join(outdir,'histogram_raw.png'))
    plt.close()

    # lognorm + minmax to 0,1
    matrix = np.log2(matrix + 1)
    scaler = MinMaxScaler((0,1))
    matrix = matrix.transpose()
    matrix = pd.DataFrame(scaler.fit_transform(matrix), columns = matrix.columns, index = matrix.index)
    matrix = matrix.transpose()

    # sample some matrix values and check distribution again
    mat_val = matrix.values.flatten()
    sampling = sample(range(len(mat_val)), int(0.001*len(mat_val)))
    mat_val = mat_val[sampling]
    fig, ax = plt.subplots(figsize=(16,6))
    ax.hist(mat_val)
    ax.set_title('Histogram of values')
    fig.savefig(osp.join(outdir,'histogram_lognorm.png'))
    plt.close()

    log_handle.write('Saving by cell line to npz format\n------\n')  
    log_handle.write('----------------\n----------------\nDATA MATRICES\n')
    log_handle.write('CELL\tPERT\t\tSIZE\tUNIQUE GENES/VECTORS\tUNIQUE CELL LINES\tAVG NUMBER OF 1\'s\n')
    # save by cell lines
    metadata.index = metadata['sample_id']
    metadata = metadata.loc[matrix.columns]	#sort metadata given by column order in data matrix (and filter samples that have been filtered out from matrix during binarization)
    for cell_line in keep_cell_lines:
        for pert_type in ['trt_oe', 'ctl_vector']:
            metadata_i = metadata[np.logical_and(metadata['cell_iname'] == cell_line, metadata['pert_type'] == pert_type)]
            data_i = matrix[metadata_i.index]
            metadata_i.to_csv(osp.join(outdir, 'cell_line_{}_pert_{}_metadata.txt'.format(cell_line, pert_type)), index=False)
            filename = 'cell_line_{}_pert_{}'.format(cell_line, pert_type)
            np.savez_compressed(osp.join(outdir, filename), data=data_i.values, row_ids = data_i.index, col_ids=data_i.columns)
            log_handle.write('{}\t{}\t\t{}\t{}\t{}\t{}\n'.format(cell_line, pert_type, len(metadata_i), len(set(metadata_i['pert_id'])),  len(set(metadata_i['cell_iname'])), np.mean(np.sum(data_i, 0))))
    log_handle.write('\n\n------\nSTATS\n------\n')		
		
    return


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process LINCS L3 OE data')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Root directory containing the data')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to save processed output')
    args = parser.parse_args()
    
    # get the variables
    DATA_ROOT = args.data_dir
    outdir = args.out_dir
    os.makedirs(outdir, exist_ok=True)

    # create log file
    log_handle = open(osp.join(outdir, 'process_data_lognorm.txt'), 'w')

    # begin processing
    inst_info_oe, matrix_oe, inst_info_ctl, matrix_ctl, gene_info, log_handle = load_and_eda(DATA_ROOT, outdir, log_handle)
    inst_info_oe, matrix_oe, inst_info_ctl, matrix_ctl, gene_info, log_handle, keep_cell_lines = filter_by_cell_line(inst_info_oe, matrix_oe, inst_info_ctl, matrix_ctl, gene_info, log_handle)
    combine_lognorm_save(inst_info_oe, matrix_oe, inst_info_ctl, matrix_ctl, log_handle, outdir, keep_cell_lines)

    log_handle.close()

if __name__ == "__main__":
    main()