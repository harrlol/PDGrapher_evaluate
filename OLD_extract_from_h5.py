'''
Takes in an h5py file (most likely only work for TF Atlas for now), returns
1 h5 tree structure
2 various QC plots
3 cleaned metadat and log normmed expression matrix
'''

import torch
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from scipy.io import mmread
import matplotlib.pyplot as plt
import anndata as ad
import argparse
import h5py
import re
import os
import os.path as osp
import sys
from matplotlib_venn import venn2
from random import sample
from sklearn.preprocessing import MinMaxScaler

def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                try:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                try:
                    print(pre + '├── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')


def main(H5_PATH, OUT_DIR):
    print("begin processing h5py at " + H5_PATH)

    # begin inspecting the h5 file
    with h5py.File(H5_PATH, 'r') as f:

        # 1 make a tree and output it 
        original_stdout = sys.stdout
        with open(osp.join(OUT_DIR, "h5_tree.txt"), 'w') as tree_txt:
            sys.stdout = tree_txt
            h5_tree(f)
        sys.stdout = original_stdout

        # 2 create figures for QC
        # visualize the distribution of n_counts
        plt.figure()
        plt.hist(f['obs']['n_counts'][:], bins=50)
        plt.xlabel("n_counts")
        plt.ylabel("Frequency")
        plt.title("Distribution of n_counts")   
        plt.savefig(osp.join(OUT_DIR, 'n_counts_dist.png'))

        # visualize the distribution of n_genes
        plt.figure()
        plt.hist(f['obs']['n_genes'][:], bins=50)
        plt.xlabel("n_genes")
        plt.ylabel("Frequency")
        plt.title("Distribution of n_genes")
        plt.savefig(osp.join(OUT_DIR, 'n_genes_dist.png'))

        # 3 assemble metadata and expression matrix
        pert_dict = f['obs']['__categories']['TF'][:]
        pert_dict = [s.decode('utf-8') for s in pert_dict]
        pert_id = f['obs']['TF'][:].astype(int)
        pert_list = [pert_dict[i] for i in pert_id]
        sample_id = f['obs']['_index'][:].astype(str)

        metadata = pd.DataFrame({
            'pert_id': pert_id,
            'pert_gene': pert_list,
            '_index': sample_id
        })

        # note this loop will only add anything that is a dataset at directly one level below obs
        # it also assumes that all dataset type objects under obs are of the same dimension
        for key in f['obs'].keys():
            if key == 'TF' or key == "_index":
                continue

            item = f['obs'][key]
            if isinstance(item, h5py.Dataset):
                print(f"{key} is a dataset, add to metadata")
                metadata[key]=item
            elif isinstance(item, h5py.Group):
                print(f"{key} is a Group, skipping nested structure")
            else:
                print(f"{key} is an unknown type:", type(item))

        # 4 separating out ctrl from pert, and save metadata
        ctrl_samples = metadata['pert_gene'].str.contains("mCherry|GFP", case=False, na=False)
        metadata_out = pd.concat([metadata[~ctrl_samples], metadata[ctrl_samples]], axis=0)
        metadata_out.to_csv(osp.join(OUT_DIR, "metadata.csv"), index=False)

        # 5 use metadata to reorder gene matrix and out
        ordered_sample_id = metadata_out['_index'].tolist()

        # read in matrix SLOW
        print("Reading in X data object from h5 file...")
        matrix = f['X'][:].transpose()
        gene_ids = f['var']['_index'][:]
        sample_ids = f['obs']['_index'][:]
        print("Converting to pd datafrmae...")
        matrix_df = pd.DataFrame(matrix, columns = sample_ids.astype(str), index = gene_ids.astype(str)).astype(np.uint16)
        del matrix
    
        # random selections
        rand_gene = matrix_df.sample(n=1000, random_state=702).index        # randomly sample 1000 gene and record their index
        rand_gene_index = matrix_df.index.get_indexer(rand_gene)
        n_size = min(1000, int(0.0001 * matrix_df.shape[1]))     # df is gonna be huge
        rand_cell_index = np.random.default_rng(seed=702).choice(matrix_df.shape[1], size=n_size, replace=False)         # randomly sample 1000 cells

        # create more figures for log norm
        print("Creating expression distribution plot and log norm...")
        sample = matrix_df.iloc[rand_gene_index, rand_cell_index]
        plt.figure()
        plt.hist(sample)
        plt.xlabel("Expression Level")
        plt.ylabel("Frequency")
        plt.title("Distribution of Raw Expression Level")
        plt.savefig(osp.join(OUT_DIR, 'random_sample_expression_raw.png'))

        # log norm
        matrix_df = np.log2(matrix_df + 1)
        scaler = MinMaxScaler((0,1))
        matrix_df = matrix_df.transpose()
        matrix_df = pd.DataFrame(scaler.fit_transform(matrix_df), columns = matrix_df.columns, index = matrix_df.index)
        matrix_df = matrix_df.transpose()

        sample = matrix_df.iloc[rand_gene_index, rand_cell_index]
        plt.figure()
        plt.hist(sample)
        plt.xlabel("Expression Level")
        plt.ylabel("Frequency")
        plt.title("Distribution of Log Norm Expression Level")
        plt.savefig(osp.join(OUT_DIR, 'random_sample_expression_lognorm.png'))

        # reorder and out
        print("Reordering by metadata...")
        matrix_df = matrix_df[ordered_sample_id]
        print("Saving to npz format...")
        np.savez_compressed(osp.join(OUT_DIR, "matrix.csv"), data=matrix_df.values, row_ids = matrix_df.index, col_ids=matrix_df.columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process h5py files")
    parser.add_argument('--path_to_h5', type=str, dest='H5_PATH', help='File path for the h5 file you want to process')
    parser.add_argument('--output_directory', type=str, dest='OUT_DIR', help='Path to the output directory')
    args = parser.parse_args()

    main(args.H5_PATH, args.OUT_DIR)