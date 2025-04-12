##Create splits to be used for training across all models

import os
import os.path as osp
from sklearn.model_selection import KFold, train_test_split
import torch
import argparse
from sklearn.metrics import jaccard_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_splits_n_fold(dataset, splits_type, nfolds, outdir):
	kf = KFold(nfolds, shuffle=True, random_state=42)
	os.makedirs(outdir, exist_ok = True)

	#datasets forward and backward
	dataset_forward = dataset[0]; dataset_backward = dataset[1]
	splits = {}

	if splits_type =='random':
		i = 1
		if len(dataset_forward)> 0:
			for train_test_index_forward, train_test_index_backward in zip(kf.split(dataset_forward), kf.split(dataset_backward)):
				#Forward
				train_index_forward = train_test_index_forward[0]; test_index_forward = train_test_index_forward[1]
				train_index_backward = train_test_index_backward[0]; test_index_backward = train_test_index_backward[1]
				#Backward
				train_index_forward, val_index_forward = train_test_split(train_index_forward, test_size=0.2, random_state=42)
				train_index_backward, val_index_backward = train_test_split(train_index_backward, test_size=0.2, random_state=42)

				assert len(dataset_backward) == len(train_index_backward) +  len(val_index_backward) +  len(test_index_backward), 'Splitted datasets should have the same number of samples as full dataset'
				assert  len(set(test_index_forward).intersection(train_index_forward)) ==0, "Overlap between train and test indices should be zero"
				assert  len(set(test_index_forward).intersection(val_index_forward)) ==0, "Overlap between val and test indices should be zero"
				assert  len(set(train_index_forward).intersection(val_index_forward)) ==0, "Overlap between train and val indices should be zero"
				assert  len(set(test_index_backward).intersection(train_index_backward)) ==0, "Overlap between train and test indices should be zero"
				assert  len(set(test_index_backward).intersection(val_index_backward)) ==0, "Overlap between val and test indices should be zero"
				assert  len(set(train_index_backward).intersection(val_index_backward)) ==0, "Overlap between train and val indices should be zero"

				splits[i] = {'train_index_forward': train_index_forward, 
								'val_index_forward': val_index_forward,
								'test_index_forward': test_index_forward,
								'train_index_backward': train_index_backward,
								'val_index_backward': val_index_backward,
								'test_index_backward': test_index_backward}
				i += 1
		else:
			for train_test_index_backward in kf.split(dataset_backward):
				#Backward
				train_index_backward = train_test_index_backward[0]; test_index_backward = train_test_index_backward[1]
				train_index_backward, val_index_backward = train_test_split(train_index_backward, test_size=0.2, random_state=42)

				assert len(dataset_backward) == len(train_index_backward) +  len(val_index_backward) +  len(test_index_backward), 'Splitted datasets should have the same number of samples as full dataset'
				assert  len(set(test_index_backward).intersection(train_index_backward)) ==0, "Overlap between train and test indices should be zero"
				assert  len(set(test_index_backward).intersection(val_index_backward)) ==0, "Overlap between val and test indices should be zero"
				assert  len(set(train_index_backward).intersection(val_index_backward)) ==0, "Overlap between train and val indices should be zero"

				splits[i] = {'train_index_forward': None, 
								'val_index_forward': None,
								'test_index_forward': None,
								'train_index_backward': train_index_backward,
								'val_index_backward': val_index_backward,
								'test_index_backward': test_index_backward}
				i += 1
		
	torch.save(splits, osp.join(outdir,'splits.pt'))
	return


###Generate splits
def main():
	parser = argparse.ArgumentParser(description='Create splits')
	parser.add_argument('--nfolds', type=int, required=True, 
                        help='desired number of folds')
	parser.add_argument('--cell_lines_keep', type=str, required=True,
                        help='the cell lines you want to keep')
	parser.add_argument('--out_dir', type=str, required=True,
                        help='output directory')
	parser.add_argument('--data_root', type=str, required=True,
                        help='data directory')
	args = parser.parse_args()

	# get cell lines to keep
	cell_lines_str = args.cell_lines_keep
	cell_lines_keep = cell_lines_str.split()
	outdir_root = args.out_dir
	nfolds = args.nfolds
	data_root = args.data_root
	os.makedirs(outdir_root, exist_ok=True)
	
	for dataset_name in cell_lines_keep:
		for splits_type in ['random']:
			splits_setting = '{}fold'.format(nfolds)
			outdir = osp.join(outdir_root, '{}/{}/{}/{}'.format('genetic', dataset_name, splits_type, splits_setting))
			try:
				path = osp.join(data_root, 'data_forward_{}.pt'.format(dataset_name))
				dataset_forward = torch.load(path)
				path = osp.join(data_root, 'data_backward_{}.pt'.format(dataset_name))
				dataset_backward = torch.load(path)
				dataset = [dataset_forward, dataset_backward]
				create_splits_n_fold(dataset, splits_type, nfolds, outdir)
			except:
				continue

if __name__ == "__main__":
	main()