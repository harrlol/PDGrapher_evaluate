'''
Takes in npz file and processes it into pytorch data file as done in export_data_for_torch_geometric.py
'''
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import re
import os.path as osp
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from tqdm import tqdm


def main():
    # attempting to make export_data_for_torch_geometric.py for joung's data
    matrix_file_path = "/home/b-evelyntong/hl/joung_data/03-21-t1/matrix.csv.npz"
    metadata_file_path = "/home/b-evelyntong/hl/joung_data/03-21-t1/metadata.csv"

    # load_ppi
    ppi = nx.read_edgelist("/home/b-evelyntong/hl/PDGrapher/data/processed/ppi/ppi_all_genes_edgelist_joung.txt")

    # load_gene_metadata (supposedly only gene_info is used)
    gene_info = pd.read_csv("/home/b-evelyntong/hl/joung_data/gene_info.txt", header=None)
    # gene_info = pd.read_csv("/home/b-evelyntong/hl/PDGrapher/data/processed/lincs/gene_info.txt")

    # load_data
    # loads in the actual matrix file, 44GB npz takes 3.5min and ~100GB memory
    obs_metadata = pd.read_csv(metadata_file_path)
    with np.load(matrix_file_path, allow_pickle=True) as arr:
        obs_data =arr['data']
        col_ids = arr['col_ids']
        row_ids = arr['row_ids']
    obs_data = pd.DataFrame(obs_data, columns= col_ids, index=row_ids)
    print('Number of observational datapoints:\t{}\n'.format(len(obs_metadata)))

    # split my data into "observational" (GFP and mCherry) and "interventional"
    ctrl_index = obs_metadata[obs_metadata['pert_gene'].str.contains("mCherry|GFP")]['_index']
    pert_index = obs_metadata[~obs_metadata['pert_gene'].str.contains("mCherry|GFP")]['_index']

    ctrl_data = obs_data[ctrl_index]
    pert_data = obs_data[pert_index]

    ctrl_metadata = obs_metadata[obs_metadata['pert_gene'].str.contains("mCherry|GFP")]
    pert_metadata = obs_metadata[~obs_metadata['pert_gene'].str.contains("mCherry|GFP")]

    del obs_data

    #1.Filter out obs and int data to keep only genes that are in the PPI
    gene_symbols_in_ppi = list(ppi.nodes())
    gene_symbols = gene_info[0].values.tolist()
    gene_symbols_in_ppi = list(set(gene_symbols) & set(gene_symbols_in_ppi))

    ctrl_data = ctrl_data.loc[gene_symbols_in_ppi]
    pert_data = pert_data.loc[gene_symbols_in_ppi]

    gene_id_sym_index = pert_metadata[['pert_id', 'pert_gene', '_index']]
    gene_id_sym_index = gene_id_sym_index.rename(columns={'pert_gene': 'pert_iso'})
    extracted_genes = [re.split('-', gene)[1] if '-' in gene else gene for gene in gene_id_sym_index['pert_iso']]
    gene_id_sym_index['pert_gene'] = extracted_genes


    #2. Filter out samples whose perturbations are not in the remaining genes (those in the PPI)
    keep = []
    for i, gene_symbol in gene_id_sym_index[['_index', 'pert_gene']].iterrows():
        if gene_symbol['pert_gene'] in gene_symbols_in_ppi:
            keep.append(gene_symbol['_index'])

    pert_metadata.index = pert_metadata['_index']; pert_metadata = pert_metadata.loc[keep].reset_index(inplace=False, drop=True)
    pert_data = pert_data[keep]
    gene_id_sym_index = gene_id_sym_index[gene_id_sym_index['_index'].isin(keep)]
    print('Number of interventional datapoints after keeping only those with perturbed genes in PPI:\t{}\n'.format(len(pert_metadata)))

    all_genes = pert_data.index

    unique_sym_id = gene_id_sym_index[['pert_id', 'pert_gene']].drop_duplicates()
    gene_symbol_to_index = dict(zip(unique_sym_id['pert_gene'], unique_sym_id['pert_id']))
    gene_index_to_ordered_index = dict(zip(unique_sym_id['pert_id'], range(len(unique_sym_id))))
    gene_symbol_to_ordered_index = dict(zip(unique_sym_id['pert_gene'], range(len(unique_sym_id))))
    gene_info = unique_sym_id.copy() 
    gene_info['ordered_index'] = [gene_index_to_ordered_index[i] for i in gene_info['pert_id']]

    real_gene_to_index = dict(zip(all_genes, range(len(all_genes))))
    ppi = nx.relabel_nodes(ppi, real_gene_to_index)

    pert_data.index = [real_gene_to_index[i] for i in pert_data.index]
    pert_data = pert_data.sort_index(inplace=False)
    ctrl_data.index = [real_gene_to_index[i] for i in ctrl_data.index]
    ctrl_data = ctrl_data.sort_index(inplace=False)

    #Assembling samples
    print("begin assembling data")
    edge_index = torch.LongTensor(np.array(ppi.edges()).transpose())
    edge_index = add_remaining_self_loops(edge_index)[0]
    edge_index = to_undirected(edge_index)
    number_of_nodes = ppi.number_of_nodes()

    #dict sample id: perturbed gene ordered index
    dict_sample_id_perturbed_gene_ordered_index = dict()
    for sample_index, pert_gene in zip(gene_id_sym_index['_index'], gene_id_sym_index['pert_gene']):
        dict_sample_id_perturbed_gene_ordered_index[sample_index] = real_gene_to_index[pert_gene]

    #these are helpers to sample from obs_data
    order = np.array(range(ctrl_data.shape[1]))
    np.random.shuffle(order)
    i = 0
    #shuffle obs data columns
    backward_data_list = []
    unique_names_pert = set()
    print("processing all samples...")
    for sample_id in tqdm(pert_data.columns, desc="Processing Samples"):
        binary_indicator_perturbation = np.zeros(len(pert_data))
        binary_indicator_perturbation[dict_sample_id_perturbed_gene_ordered_index[sample_id]] = 1
        #Get a random pre-intervention sample
        i = i % ctrl_data.shape[1]
        sample_index = order[i]
        ctrl_sample_id = ctrl_data.columns[i]
        ctrl_sample = ctrl_data[ctrl_data.columns[i]].values
        #concat initial node features and perturbation indicator
        control = torch.Tensor(ctrl_sample)
        perturbation = torch.Tensor(binary_indicator_perturbation)
        mutations = torch.Tensor(np.zeros(len(control)))
        # torch.Tensor(np.stack([obs_sample, binary_indicator_perturbation], 1))
        #post-intervention
        perturbed = torch.Tensor(pert_data[sample_id])
        #remove incoming edges to perturbed node
        # perturbed_node = dict_sample_id_perturbed_gene_ordered_index[sample_id]
        # edge_index_mutilated = edge_index[:, edge_index[1,:] != perturbed_node]
        
        gene_name = gene_id_sym_index[gene_id_sym_index['_index'] == sample_id]['pert_gene'].item()
        unique_names_pert.add(gene_name)
        data = Data(perturbagen_name = gene_name, control = control, perturbation=perturbation, perturbed = perturbed, gene_symbols = list(all_genes), mutations = mutations)
        data.num_nodes = number_of_nodes
        backward_data_list.append(data)
        i +=1

    # print('Samples forward:\t{}\n'.format(len(forward_data_list)))
    print('Samples backward:\t{}\n'.format(len(backward_data_list)))
    print('Unique perturbagens:\t{}\n'.format(len(unique_names_pert)))

    print('Saving data ...\n\n\n')
        
    outdir = '/home/b-evelyntong/hl/joung_data/torch_data'
    # torch.save(forward_data_list, osp.join(outdir, 'data_forward.pt'))
    torch.save(backward_data_list, osp.join(outdir, 'data_backward.pt'))
    torch.save(edge_index, osp.join(outdir, 'edge_index.pt'))

if __name__ == "__main__":
    main()