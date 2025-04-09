'''
Takes in the raw .tab3.txt file downloaded from biogrid per Gonzalez et al's workflow, 
returns a plain txt file with just the genes and the edges
'''

import pandas as pd
import os.path as osp
import argparse
import sys


def main(txt_in_path, txt_out_dir):
    # reads in tab3 file
    df = pd.read_csv(txt_in_path, sep = '\t', low_memory=False)

    # filters for only protein in human, code 9606
    df = df[(df['Organism ID Interactor A'] == 9606) & (df['Organism ID Interactor B'] == 9606)]

    # pulls out only the pairs of interacting entities columns
    edges = df[['Official Symbol Interactor A', 'Official Symbol Interactor B']].dropna()

    # filters out self-loops and dups
    edges = edges[edges['Official Symbol Interactor A'] != edges['Official Symbol Interactor B']]
    edges = edges.drop_duplicates()

    # save
    edges.to_csv(osp.join(txt_out_dir, "ppi_edgelist.txt"), sep = '\t', index = False, header = False)
    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BIOGRID PPI edges")
    parser.add_argument('--path_to_tab3_txt', type=str, dest='txt_in_path', help='File path for the BIOGRID txt file you want to process')
    parser.add_argument('--output_directory', type=str, dest='txt_out_dir', help='Path to the output directory')
    args = parser.parse_args()

    main(args.txt_in_path, args.txt_out_dir)