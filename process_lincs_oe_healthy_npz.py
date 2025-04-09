'''
Process LINCS data of healhty cell lines
MCF10A, NL20, RWPE1
Will do some processing first and then rely on the functions in process_data.py
'''

import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp
import numpy as np
import argparse
from collections import Counter
import matplotlib.pyplot as plt
from random import sample
from sklearn.preprocessing import MinMaxScaler


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


def loads_data(DATA_ROOT, log_handle):

	# Loads metadata
	inst_info = pd.read_csv(os.path.join(DATA_ROOT, 'instinfo_beta.txt'), sep="\t", low_memory=False)
	inst_info_ctl_mcf10a = inst_info[np.logical_and(inst_info['cell_iname'] == 'MCF10A',np.logical_and(inst_info['pert_type'] == 'ctl_untrt', inst_info['failure_mode'].isna())) ].reset_index(inplace=False, drop=True)
	inst_info_ctl_nl20 = inst_info[np.logical_and(inst_info['cell_iname'] == 'NL20',np.logical_and(inst_info['pert_type'] == 'ctl_vehicle', inst_info['failure_mode'].isna())) ].reset_index(inplace=False, drop=True)
	inst_info_ctl_rwpe1 = inst_info[np.logical_and(inst_info['cell_iname'] == 'RWPE1',np.logical_and(inst_info['pert_type'] == 'ctl_vector', inst_info['failure_mode'].isna())) ].reset_index(inplace=False, drop=True)
	inst_info_ctl = pd.concat([inst_info_ctl_mcf10a, inst_info_ctl_nl20, inst_info_ctl_rwpe1])
	gene_info = pd.read_csv(os.path.join(DATA_ROOT, 'geneinfo_beta.txt'), sep="\t", low_memory=False)

	
	f = h5py.File(os.path.join(DATA_ROOT, 'level3_beta_ctl_n188708x12328.gctx'), 'r')
	matrix_ctl = f['0']['DATA']['0']['matrix'][:].transpose()
	gene_ids_ctl = f['0']['META']['ROW']['id'][:]					#not in the same order as gene_ids_xpr
	sample_ids_ctl = f['0']['META']['COL']['id'][:]
	matrix_ctl = pd.DataFrame(matrix_ctl, columns = sample_ids_ctl.astype(str), index = gene_ids_ctl.astype(int))
	del f

	# stats
	log_handle.write('CONTROL\n------\n')
	log_handle.write('Control entries in inst_info metadata:\t{}\n'.format(len(inst_info_ctl)))
	log_handle.write('Control entries in data matrix:\t{}\n'.format(len(sample_ids_ctl)))
	log_handle.write('Overlap between inst_info metadata and sample ids in data matrix:\t{}\n'.format(len(set(inst_info_ctl['sample_id']).intersection(set(sample_ids_ctl.astype(str))))))
	log_handle.write('\n------\n')

	# moving filtering up
	log_handle.write('Filtering to keep only those in metadata\n------\n')
	#CONTROL
	list_ids = list(inst_info_ctl['sample_id'])	#in metadata
	#extra steps
	list_ids = list(set(list_ids).intersection(set(matrix_ctl.columns.astype(str))))	#in metadata and in data matrix (some of metadata are not in data matrix)
	inst_info_ctl.index = inst_info_ctl['sample_id']; inst_info_ctl = inst_info_ctl.loc[list_ids].reset_index(inplace=False, drop=True) #remove entries from metadata that are not in data matrix
	matrix_ctl = matrix_ctl[list_ids]	#Filtered data matrix
	log_handle.write('CONTROL:\t{} datapoints\n\n\n'.format(matrix_ctl.shape[1]))

	return inst_info_ctl, gene_info, matrix_ctl


def normalize_and_save(inst_info_ctl, matrix_ctl, gene_info, log_handle, outdir):
	
	metadata = inst_info_ctl
	metadata.to_csv(osp.join(outdir, 'all_metadata_healthy.txt'))
	matrix = matrix_ctl

	matrix = np.log2(matrix + 1)
	scaler = MinMaxScaler((0,1))
	matrix = matrix.transpose()
	matrix = pd.DataFrame(scaler.fit_transform(matrix), columns = matrix.columns, index = matrix.index)
	matrix = matrix.transpose()
	mv = matrix.values.flatten()
	sampling = sample(range(len(mv)), int(0.1*len(mv)))
	mv = mv[sampling]

	fig, ax = plt.subplots(figsize=(16,6))
	ax.hist(mv)
	ax.set_title('Histogram of values')
	fig.savefig(osp.join(outdir,'histogram_healthy.png'))
	plt.close()

	log_handle.write('----------------\n----------------\nDATA MATRICES\n')
	log_handle.write('CELL\tPERT\t\tSIZE\tUNIQUE GENES/VECTORS\tUNIQUE CELL LINES\tAVG NUMBER OF 1\'s\n')
	metadata.index = metadata['sample_id']
	metadata = metadata.loc[matrix.columns]	#sort metadata given by column order in data matrix (and filter samples that have been filtered out from matrix during binarization)
	for cell_line, pert_type in zip(['MCF10A', 'NL20', 'RWPE1'],['ctl_untrt', 'ctl_vehicle', 'ctl_vector'] ):
		metadata_i = metadata[np.logical_and(metadata['cell_iname'] == cell_line, metadata['pert_type'] == pert_type)]
		data_i = matrix[metadata_i.index]
		metadata_i.to_csv(osp.join(outdir, 'cell_line_{}_pert_{}_metadata.txt'.format(cell_line, pert_type)), index=False)
		filename = 'cell_line_{}_pert_{}'.format(cell_line, pert_type)
		np.savez_compressed(osp.join(outdir, filename), data=data_i.values, row_ids = data_i.index, col_ids=data_i.columns)
		log_handle.write('{}\t{}\t\t{}\t{}\t{}\t{}\n'.format(cell_line, pert_type, len(metadata_i), len(set(metadata_i['cmap_name'])),  len(set(metadata_i['cell_iname'])), np.mean(np.sum(data_i, 0))))
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
    
	log_handle = open(osp.join(outdir, 'process_data_healthy_lognorm.txt'), 'w')
    
	inst_info_ctl, gene_info, matrix_ctl = loads_data(DATA_ROOT, log_handle)
	stats_control(inst_info_ctl, log_handle)
	normalize_and_save(inst_info_ctl, matrix_ctl, gene_info, log_handle, outdir)
	log_handle.close()


if __name__ == "__main__":
    main()






