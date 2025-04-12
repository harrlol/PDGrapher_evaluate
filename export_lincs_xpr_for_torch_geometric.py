'''
adapted from export_data_for_torch_geometric.py, this script specifically processes
each npz to output pytorch files that is then fed into the model accordingly for
each cell line that may or may not have a healthy counterpart
'''
import torch
import numpy as np
import pandas as pd
import os.path as osp
import os
import argparse
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops, to_undirected


def load_data(data_root_dir, cell_line, log_handle):
    # loads observational data
    file = osp.join(data_root_dir, 'cell_line_{}_pert_ctl_vector.npz'.format(cell_line))
    file_metadata = osp.join(data_root_dir, 'cell_line_{}_pert_ctl_vector_metadata.txt'.format(cell_line))
    obs_metadata = pd.read_csv(file_metadata)
    with np.load(file, allow_pickle=True) as arr:
        obs_data =arr['data']
        col_ids = arr['col_ids']
        row_ids = arr['row_ids']
    obs_data = pd.DataFrame(obs_data, columns= col_ids, index=row_ids)
    log_handle.write('Number of observational datapoints:\t{}\n'.format(len(obs_metadata)))
	
    # loads interventional data
    file = osp.join(data_root_dir, 'cell_line_{}_pert_trt_xpr.npz'.format(cell_line))
    file_metadata = osp.join(data_root_dir, 'cell_line_{}_pert_trt_xpr_metadata.txt'.format(cell_line))
    int_metadata = pd.read_csv(file_metadata)
    with np.load(file, allow_pickle=True) as arr:
        int_data =arr['data']
        col_ids = arr['col_ids']
        row_ids = arr['row_ids']
    int_data = pd.DataFrame(int_data, columns= col_ids, index=row_ids)
    log_handle.write('Number of interventional datapoints:\t{}\n'.format(len(int_metadata)))
	
    return int_data, int_metadata, obs_data, obs_metadata


def load_healthy_data(data_root_dir, healthy, log_handle):
	healthy_data_path = osp.join(data_root_dir, 'cell_line_{}_pert_{}.npz'.format(healthy[0], healthy[1]))
	healthy_metadata_path = osp.join(data_root_dir, 'cell_line_{}_pert_{}_metadata.txt'.format(healthy[0], healthy[1]))
	healthy_metadata = pd.read_csv(healthy_metadata_path)
	with np.load(healthy_data_path, allow_pickle=True) as arr:
		healthy_data =arr['data']
		col_ids = arr['col_ids']
		row_ids = arr['row_ids']
	healthy_data = pd.DataFrame(healthy_data, columns= col_ids, index=row_ids)
	log_handle.write('Loading healthy cell line:\t{} Number of samples:\t{}\n'.format(healthy[0], healthy_data.shape[1]))
	return healthy_data, healthy_metadata


def map_cosmic_to_lincs(cosmic_data, cell_line, gene_info, dict_symbol_entrez, log_handle):
	cosmic_data = cosmic_data[cosmic_data['Sample name']==cell_line]
	log_handle.write('Mapping cosmic genes to lincs. Mapped: {}/{}\n'.format(len(set(cosmic_data['Gene name']).intersection(gene_info['gene_symbol'])), len(set(cosmic_data['Gene name']))))
	# filter genes not mapped to LINCS
	cosmic_data = cosmic_data[[gene_symbol in dict_symbol_entrez for gene_symbol in cosmic_data['Gene name']]]
	# save COSMIC mutations as entrez id (dataframe index)
	try:
		cosmic_mutations = list(set([dict_symbol_entrez[symbol] for symbol in cosmic_data['Gene name'].tolist()]))
	except:
		import pdb; pdb.set_trace()
	return cosmic_mutations


def filter_data_no_healthy(int_data, int_metadata, obs_data, obs_metadata, ppi, gene_info, log_handle):
	
    # gene symbols found in ppi
    gene_symbols_in_ppi = list(ppi.nodes())

    # filter genes in obs and int data for those found in PPI
    dict_symbol_id = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))  # key is gene symbol value is id
    gene_ids_in_ppi = [dict_symbol_id[i] for i in gene_symbols_in_ppi]  # get a list of gene id for genes found in ppi
    gene_info.index = gene_info['gene_id']; gene_info = gene_info.loc[gene_ids_in_ppi].reset_index(inplace=False, drop=True)  # filter gene info
    obs_data = obs_data.loc[gene_ids_in_ppi]  # filter obs data
    int_data = int_data.loc[gene_ids_in_ppi]  # filter int data

	# get intervention genes that are in the remaining genes
    keep = []
    for i, gene_symbol in enumerate(int_metadata['cmap_name']):
        if gene_symbol in gene_symbols_in_ppi:
            keep.append(int_metadata.at[i, 'sample_id'])
			
	# filter int data and metadata to keep only samples perturbed with these genes
    int_metadata.index = int_metadata['sample_id']; int_metadata = int_metadata.loc[keep].reset_index(inplace=False, drop=True)
    int_data = int_data[keep]
    log_handle.write('Number of interventional datapoints after keeping only those with perturbed genes in PPI:\t{}\n'.format(len(int_metadata)))
    log_handle.write('Assembling data...\n')

    return int_data, int_metadata, obs_data, obs_metadata, ppi, gene_info, 


def filter_data_with_healthy(healthy_data, healthy_metadata, cosmic_mutations, int_data, int_metadata, 
                             obs_data, obs_metadata, ppi, gene_info, log_handle):
	
    # gene symbols found in ppi
    gene_symbols_in_ppi = list(ppi.nodes())

    # filter genes in obs and int data for those found in PPI
    dict_symbol_id = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))  # key is gene symbol value is id
    gene_ids_in_ppi = [dict_symbol_id[i] for i in gene_symbols_in_ppi]  # get a list of gene id for genes found in ppi
    gene_info.index = gene_info['gene_id']; gene_info = gene_info.loc[gene_ids_in_ppi].reset_index(inplace=False, drop=True)  # filter gene info
    obs_data = obs_data.loc[gene_ids_in_ppi]  # filter obs data
    int_data = int_data.loc[gene_ids_in_ppi]  # filter int data

    # processing
    healthy_data = healthy_data.loc[gene_ids_in_ppi]
    cosmic_mutations = pd.DataFrame(cosmic_mutations)[[e in gene_ids_in_ppi for e in cosmic_mutations]][0].tolist()

	# get intervention genes that are in the remaining genes
    keep = []
    for i, gene_symbol in enumerate(int_metadata['cmap_name']):
        if gene_symbol in gene_symbols_in_ppi:
            keep.append(int_metadata.at[i, 'sample_id'])
			
	# filter int data and metadata to keep only samples perturbed with these genes
    int_metadata.index = int_metadata['sample_id']; int_metadata = int_metadata.loc[keep].reset_index(inplace=False, drop=True)
    int_data = int_data[keep]
    log_handle.write('Number of interventional datapoints after keeping only those with perturbed genes in PPI:\t{}\n'.format(len(int_metadata)))

    return healthy_data, healthy_metadata, cosmic_mutations, int_data, int_metadata, obs_data, obs_metadata, ppi, gene_info, 


def assemble_data_no_healthy(int_data, int_metadata, obs_data, obs_metadata, ppi, gene_info, log_handle):
    log_handle.write('Assembling data...\n')
	# reindexing
    gene_symbol_to_index = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))  # symbol to old id
    gene_index_to_ordered_index = dict(zip(gene_info['gene_id'], range(len(gene_info))))  # old id to new id (0,1,2,...)
    gene_info['ordered_index'] = [gene_index_to_ordered_index[i] for i in gene_info['gene_id']]  # add new id as new column in gene info 
    ppi = nx.relabel_nodes(ppi, gene_symbol_to_index)  # update ppi nodes from symbol to old id
    ppi = nx.relabel_nodes(ppi, gene_index_to_ordered_index)  # update ppi nodes from old id to new id
    int_data.index = [gene_index_to_ordered_index[i] for i in int_data.index]  # update int data row index from old index to new index
    int_data = int_data.sort_index(inplace=False)  # sort by new index
    obs_data.index = [gene_index_to_ordered_index[i] for i in obs_data.index]  # update obs data row index from old index to new index
    obs_data = obs_data.sort_index(inplace=False)  # sort by new index

	# obtain edge index
    edge_index = torch.LongTensor(np.array(ppi.edges()).transpose())
    edge_index = add_remaining_self_loops(edge_index)[0]
    edge_index = to_undirected(edge_index)
    number_of_nodes = ppi.number_of_nodes()

    ########## why is this commented out in the original code? ###########
	# remove incoming edges to perturbed nodes (mutated nodes)
	# mask = [e not in cosmic_mutations for e in edge_index[1,:]]
	# edge_index_mutilated = edge_index[:, mask]
	# edge_index_mutilated = add_remaining_self_loops(edge_index_mutilated)[0]
    #####################################################################

	# construct dict, sample id to new id for the perturbed gene in that sample
    dict_sample_id_perturbed_gene_ordered_index = dict()
    for sample_id, cmap_name in zip(int_metadata['sample_id'], int_metadata['cmap_name']):
        dict_sample_id_perturbed_gene_ordered_index[sample_id] = gene_index_to_ordered_index[gene_symbol_to_index[cmap_name]]

    # helps with random sampling
    order = np.array(range(obs_data.shape[1]))
    np.random.shuffle(order)  # shuffle obs data columns
    i = 0
	
    # reminant of with health counterpart
    forward_data_list = []
	
    # begin assembly
    backward_data_list = []
    unique_names_pert = set()
    for sample_id in int_data.columns:
	    # one hot encode which expressed gene was pertubed
        binary_indicator_perturbation = np.zeros(len(int_data))
        binary_indicator_perturbation[dict_sample_id_perturbed_gene_ordered_index[sample_id]] = 1
		
		# get a random pre-intervention sample
        i = i % obs_data.shape[1]
        sample_index = order[i]
        obs_sample_id = obs_data.columns[i]
        obs_sample = obs_data[obs_data.columns[i]].values  # an expression vector
		
		# concat initial node features and perturbation indicator
        diseased = torch.Tensor(obs_sample)
        intervention = torch.Tensor(binary_indicator_perturbation)
        mutations = torch.Tensor(np.zeros(len(diseased)))
		# torch.Tensor(np.stack([obs_sample, binary_indicator_perturbation], 1))
		
		# post-intervention
        treated = torch.Tensor(int_data[sample_id])
		
        ############# again, why commented out? ##############
		#remove incoming edges to perturbed node
		# perturbed_node = dict_sample_id_perturbed_gene_ordered_index[sample_id]
		# edge_index_mutilated = edge_index[:, edge_index[1,:] != perturbed_node]
		########################################################
		
        # put into torch geometric Data object
        gene_name = int_metadata[int_metadata['sample_id'] == sample_id]['cmap_name'].item()  # perturbed gene cmap name
        unique_names_pert.add(gene_name)
        data = Data(perturbagen_name = gene_name, diseased = diseased, intervention=intervention, treated = treated, gene_symbols = gene_info['gene_symbol'].tolist(), mutations = mutations)
        data.num_nodes = number_of_nodes
        backward_data_list.append(data)
        i +=1
    
    log_handle.write('Samples forward:\t{}\n'.format(len(forward_data_list)))
    log_handle.write('Samples backward:\t{}\n'.format(len(backward_data_list)))
    log_handle.write('Unique perturbagens:\t{}\n'.format(len(unique_names_pert)))
	
    return forward_data_list, backward_data_list, edge_index


def assemble_data_with_healthy(healthy_data, healthy_metadata, cosmic_mutations, int_data, int_metadata, 
                               obs_data, obs_metadata, ppi, gene_info, log_handle):
    log_handle.write('Assembling data...\n')
	# reindexing
    gene_symbol_to_index = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))  # symbol to old id
    gene_index_to_ordered_index = dict(zip(gene_info['gene_id'], range(len(gene_info))))  # old id to new id (0,1,2,...)
    gene_info['ordered_index'] = [gene_index_to_ordered_index[i] for i in gene_info['gene_id']]  # add new id as new column in gene info 
    ppi = nx.relabel_nodes(ppi, gene_symbol_to_index)  # update ppi nodes from symbol to old id
    ppi = nx.relabel_nodes(ppi, gene_index_to_ordered_index)  # update ppi nodes from old id to new id
    int_data.index = [gene_index_to_ordered_index[i] for i in int_data.index]  # update int data row index from old index to new index
    int_data = int_data.sort_index(inplace=False)  # sort by new index
    obs_data.index = [gene_index_to_ordered_index[i] for i in obs_data.index]  # update obs data row index from old index to new index
    obs_data = obs_data.sort_index(inplace=False)  # sort by new index

    healthy_data.index = [gene_index_to_ordered_index[i] for i in healthy_data.index]
    healthy_data = healthy_data.sort_index(inplace=False)
    cosmic_mutations = [gene_index_to_ordered_index[i] for i in cosmic_mutations]
    cosmic_vector = np.zeros(len(healthy_data))
    cosmic_vector[cosmic_mutations] = 1

	# obtain edge index
    edge_index = torch.LongTensor(np.array(ppi.edges()).transpose())
    edge_index = add_remaining_self_loops(edge_index)[0]
    edge_index = to_undirected(edge_index)
    number_of_nodes = ppi.number_of_nodes()

    ########## why is this commented out in the original code? ###########
	# remove incoming edges to perturbed nodes (mutated nodes)
	# mask = [e not in cosmic_mutations for e in edge_index[1,:]]
	# edge_index_mutilated = edge_index[:, mask]
	# edge_index_mutilated = add_remaining_self_loops(edge_index_mutilated)[0]
    #####################################################################
	
    # assemble forward data
    forward_data_list = []
    order = np.array(range(healthy_data.shape[1]))
    np.random.shuffle(order)
    i = 0
    dict_forward_sample_and_mutations = dict()  # saves the mutation vector used in forward
    for sample_id in obs_data.columns:
        # sample a random healthy expression vector
        i = i % healthy_data.shape[1]
        sample_index = order[i]
        healthy_sample = healthy_data[healthy_data.columns[i]].values
        healthy = torch.Tensor(healthy_sample)

        # randomize mutations. First select the percentage of mutations to include, then select the mutations
        perc_to_include = np.random.choice([0.25, 0.50, 0.75, 1], 1).item()
        if int_metadata['cell_mfc_name'][0].split('.')[0] == 'PC3':
            perc_to_include = 1
        cosmic_mutations_i = np.random.choice(cosmic_mutations, int(len(cosmic_mutations)* perc_to_include))
        cosmic_vector = np.zeros(len(healthy_data))
        cosmic_vector[cosmic_mutations_i] = 1
        mutations = torch.Tensor(cosmic_vector)

        #diseased
        diseased = torch.Tensor(obs_data[sample_id])
        data = Data(healthy = healthy, mutations=mutations, diseased=diseased, gene_symbols = gene_info['gene_symbol'].tolist())
        data.num_nodes = number_of_nodes
        forward_data_list.append(data)
        i +=1
        dict_forward_sample_and_mutations[sample_id] = mutations

    log_handle.write('finished data forward')

    # construct dict, sample id to new id for the perturbed gene in that sample
    dict_sample_id_perturbed_gene_ordered_index = dict()
    for sample_id, cmap_name in zip(int_metadata['sample_id'], int_metadata['cmap_name']):
        dict_sample_id_perturbed_gene_ordered_index[sample_id] = gene_index_to_ordered_index[gene_symbol_to_index[cmap_name]]
	
    # assemble backward data
    backward_data_list = []
    order = np.array(range(obs_data.shape[1]))
    np.random.shuffle(order)  # shuffle obs data columns
    i = 0
    unique_names_pert = set()
    for sample_id in int_data.columns:
	    # one hot encode which expressed gene was pertubed
        binary_indicator_perturbation = np.zeros(len(int_data))
        binary_indicator_perturbation[dict_sample_id_perturbed_gene_ordered_index[sample_id]] = 1
		
		# get a random pre-intervention sample
        i = i % obs_data.shape[1]
        sample_index = order[i]
        obs_sample_id = obs_data.columns[i]
        obs_sample = obs_data[obs_data.columns[i]].values  # an expression vector
		
		# concat initial node features and perturbation indicator
        diseased = torch.Tensor(obs_sample)
        intervention = torch.Tensor(binary_indicator_perturbation)
        # mutations = torch.Tensor(np.zeros(len(diseased)))  # was used in no healthy
        mutations = dict_forward_sample_and_mutations[obs_sample_id]
		# torch.Tensor(np.stack([obs_sample, binary_indicator_perturbation], 1))
		
		# post-intervention
        treated = torch.Tensor(int_data[sample_id])
		
        ############# again, why commented out? ##############
		#remove incoming edges to perturbed node
		# perturbed_node = dict_sample_id_perturbed_gene_ordered_index[sample_id]
		# edge_index_mutilated = edge_index[:, edge_index[1,:] != perturbed_node]
		########################################################
		
        # put into torch geometric Data object
        gene_name = int_metadata[int_metadata['sample_id'] == sample_id]['cmap_name'].item()  # perturbed gene cmap name
        unique_names_pert.add(gene_name)
        data = Data(perturbagen_name = gene_name, diseased = diseased, intervention=intervention, treated = treated, gene_symbols = gene_info['gene_symbol'].tolist(), mutations = mutations)
        data.num_nodes = number_of_nodes
        backward_data_list.append(data)
        i +=1
    
    log_handle.write('Samples forward:\t{}\n'.format(len(forward_data_list)))
    log_handle.write('Samples backward:\t{}\n'.format(len(backward_data_list)))
    log_handle.write('Unique perturbagens:\t{}\n'.format(len(unique_names_pert)))
	
    return forward_data_list, backward_data_list, edge_index


def save_data(forward_data_list, backward_data_list, edge_index, cell_line, outdir, log_handle):
    log_handle.write('Saving data {} ...\n\n\n'.format(cell_line))
    torch.save(forward_data_list, osp.join(outdir, 'data_forward_{}.pt'.format(cell_line)))
    torch.save(backward_data_list, osp.join(outdir, 'data_backward_{}.pt'.format(cell_line)))
    torch.save(edge_index, osp.join(outdir, 'edge_index_{}.pt'.format(cell_line)))
    return


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description="Process and export data for PDGrapher")
    
    # Add arguments for each path
    parser.add_argument("--data_root_dir", type=str, help="Root directory containing processed data")
    parser.add_argument("--ppi_edge_list_path", type=str, help="Path to PPI edge list file")
    parser.add_argument("--gene_info_path", type=str, help="Path to gene info file")
    parser.add_argument("--outdir", type=str, help="Directory to save output files")
    parser.add_argument("--cosmic_file", type=str, help="Path to COSMIC mutation file")
    args = parser.parse_args()

    # parse variables
    outdir = args.outdir
    data_root_dir = args.data_root_dir
    ppi_edge_list_path = args.ppi_edge_list_path
    gene_info_path = args.gene_info_path
    path_cosmic_file = args.cosmic_file

    os.makedirs(outdir, exist_ok=True)
    log_handle = open(osp.join(outdir, 'log_export_data.txt'), 'w')

    for cell_line, healthy in zip(['A549', 'MCF7', 'PC3'], [('NL20', 'ctl_vehicle'), ('MCF10A', 'ctl_untrt'), ('RWPE1', 'ctl_vector')]):
        # load in ppi
        ppi = nx.read_edgelist(ppi_edge_list_path)
		
        # load_gene_metadata
        gene_info = pd.read_csv(gene_info_path)
        dict_entrez_symbol = dict(zip(gene_info['gene_id'], gene_info['gene_symbol']))
        dict_symbol_entrez = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))

        # healthy data and cosmic
        healthy_data, healthy_metadata = load_healthy_data(data_root_dir, healthy, log_handle)
        cosmic_data = pd.read_csv(path_cosmic_file)
        log_handle.write('Loading COSMIC data. Number of cell lines:\t{}\n'.format(len(set(cosmic_data['Sample name']))))
        cosmic_mutations = map_cosmic_to_lincs(cosmic_data, cell_line, gene_info, dict_symbol_entrez, log_handle)

        # function calls
        int_data, int_metadata, obs_data, obs_metadata = load_data(data_root_dir, cell_line, log_handle)
        healthy_data, healthy_metadata, cosmic_mutations, int_data, int_metadata, obs_data, obs_metadata, ppi, gene_info = filter_data_with_healthy(healthy_data, healthy_metadata, cosmic_mutations, int_data, int_metadata, obs_data, obs_metadata, ppi, gene_info, log_handle)
        forward_data_list, backward_data_list, edge_index = assemble_data_with_healthy(healthy_data, healthy_metadata, cosmic_mutations, int_data, int_metadata, obs_data, obs_metadata, ppi, gene_info, log_handle)
        save_data(forward_data_list, backward_data_list, edge_index, cell_line, outdir, log_handle)

    # export cell lines without healthy counterparts, currently treating them as all without
    for cell_line, healthy in zip(['A375', 'HT29', 'ES2', 'BICR6', 'YAPC', 'AGS', 'U251MG'], [None, None, None, None, None, None, None]):
    # for cell_line, healthy in zip(['A549', 'MCF7', 'PC3', 'VCAP', 'A375', 'HT29', 'HEK293T', 'HA1E'], [None, None, None, None, None, None, None, None]):
		# load in ppi
        ppi = nx.read_edgelist(ppi_edge_list_path)
		
        # load_gene_metadata
        gene_info = pd.read_csv(gene_info_path)
        dict_entrez_symbol = dict(zip(gene_info['gene_id'], gene_info['gene_symbol']))
        dict_symbol_entrez = dict(zip(gene_info['gene_symbol'], gene_info['gene_id']))

        # function calls
        int_data, int_metadata, obs_data, obs_metadata = load_data(data_root_dir, cell_line, log_handle)
        int_data, int_metadata, obs_data, obs_metadata, ppi, gene_info = filter_data_no_healthy(int_data, int_metadata, obs_data, obs_metadata, ppi, gene_info, log_handle)
        forward_data_list, backward_data_list, edge_index = assemble_data_no_healthy(int_data, int_metadata, obs_data, obs_metadata, ppi, gene_info, log_handle)
        save_data(forward_data_list, backward_data_list, edge_index, cell_line, outdir, log_handle)

    
    log_handle.close()

if __name__ == "__main__":
    main()