import torch
import math
import pandas as pd

# Load ESM embeddings
gene_to_embedding = torch.load("/home/b-evelyntong/hl/gene_to_embedding.pt")

# Load gene_info
# Load the .pt file to get the gene_symbol ordering
ordering_xpr = torch.load("/home/b-evelyntong/hl/lincs_lvl3_oe/torch_export/xpr/data_backward_A549.pt")
gene_symbols_xpr = ordering_xpr[0].gene_symbols
ordering_oe = torch.load("/home/b-evelyntong/hl/lincs_lvl3_oe/torch_export/oe/data_backward_A549.pt")
gene_symbols_oe = ordering_oe[0].gene_symbols

# prep for xpr
embedding_dim = len(next(iter(gene_to_embedding.values())))
num_genes = len(gene_symbols_xpr)
embedding_matrix = torch.zeros((num_genes, embedding_dim))

for idx, gene_symbol in enumerate(gene_symbols_xpr):
    if gene_symbol in gene_to_embedding:
        embedding_matrix[idx] = gene_to_embedding[gene_symbol]
    else:
        print(f"Warning: No embedding found for {gene_symbol}, initializing randomly.")
        embedding_matrix[idx] = torch.empty(embedding_dim).uniform_(
            -math.sqrt(1 / num_genes), math.sqrt(1 / num_genes)
        )

torch.save(embedding_matrix, "/home/b-evelyntong/hl/embedding_matrix_xpr.pt")

# prep for oe
embedding_dim = len(next(iter(gene_to_embedding.values())))
num_genes = len(gene_symbols_oe)
embedding_matrix = torch.zeros((num_genes, embedding_dim))

for idx, gene_symbol in enumerate(gene_symbols_oe):
    if gene_symbol in gene_to_embedding:
        embedding_matrix[idx] = gene_to_embedding[gene_symbol]
    else:
        print(f"Warning: No embedding found for {gene_symbol}, initializing randomly.")
        embedding_matrix[idx] = torch.empty(embedding_dim).uniform_(
            -math.sqrt(1 / num_genes), math.sqrt(1 / num_genes)
        )

torch.save(embedding_matrix, "/home/b-evelyntong/hl/embedding_matrix_oe.pt")