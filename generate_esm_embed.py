import networkx as nx
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from Bio import Entrez, SeqIO
import torch
from tqdm import tqdm
Entrez.email = 'lidatou0708@gmail.com'

from Bio import Entrez, SeqIO

Entrez.email = "your_email@example.com"

def get_protein_sequence_from_gene(gene_symbol, organism="Homo sapiens"):
    try:
        # 1. Search the gene in NCBI Gene db
        search_handle = Entrez.esearch(
            db="gene",
            term=f"{gene_symbol}[Gene Name] AND {organism}[Organism]",
            retmode="xml"
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()

        if not search_results["IdList"]:
            print(f"No gene ID found for {gene_symbol}")
            return None

        gene_id = search_results["IdList"][0]

        # 2. Get gene summary and extract linked protein ID
        link_handle = Entrez.elink(dbfrom="gene", db="protein", id=gene_id)
        link_results = Entrez.read(link_handle)
        link_handle.close()

        # There can be multiple links—take first one
        protein_ids = [link["Id"] for linkset in link_results for link in linkset["LinkSetDb"][0]["Link"]]
        if not protein_ids:
            print(f"No linked protein ID for {gene_symbol}")
            return None
        protein_id = protein_ids[0]

        # 3. Fetch the protein FASTA sequence
        fetch_handle = Entrez.efetch(db="protein", id=protein_id, rettype="fasta", retmode="text")
        record = SeqIO.read(fetch_handle, "fasta")
        fetch_handle.close()

        return str(record.seq)

    except Exception as e:
        print(f"Error retrieving protein for {gene_symbol}: {e}")
        return None


# below is too slow, i downloaded sequence locally instead
G = nx.read_edgelist("/home/b-evelyntong/hl/ppi_all_genes_edgelist.txt")

uniprot_dict = {}
for record in SeqIO.parse("/home/b-evelyntong/hl/esm_resource/human_proteome.fasta", "fasta"):
    desc = record.description
    seq = str(record.seq)
    for token in desc.split():
        if token.startswith("GN="):
            gene = token.split("=")[1]
            if gene not in uniprot_dict:
                uniprot_dict[gene] = seq
            break

print("Number of unique genes in uniprot_dict:", len(uniprot_dict))
print("Number of unique genes in G:", len(G.nodes()))
# check overlap with G
overlap = set(G.nodes()).intersection(set(uniprot_dict.keys()))
print("Number of overlapping genes:", len(overlap))

# keep a list of genes not in uniprot_dict and print in the end
gene_symbol_to_sequence = {}
for gene_symbol in G.nodes():
    if gene_symbol in uniprot_dict:
        gene_symbol_to_sequence[gene_symbol] = uniprot_dict[gene_symbol]
    else:
        print(f"Gene {gene_symbol} not found in uniprot_dict, fetching from NCBI...")
        seq = get_protein_sequence_from_gene(gene_symbol)
        if seq:
            gene_symbol_to_sequence[gene_symbol] = seq
        else:
            print(f"Gene {gene_symbol} not found in NCBI either, skipping...")
            continue
        
# check dict 
print("Number of genes in gene_symbol_to_sequence:", len(gene_symbol_to_sequence))

# convert to embeddings
gene_to_embedding = {}
client = ESMC.from_pretrained("esmc_300m").to("cuda")
for gene_symbol, sequence in tqdm(gene_symbol_to_sequence.items()):
    try:
        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        embedding = output.embeddings.mean(dim=1).squeeze().detach().cpu()
        gene_to_embedding[gene_symbol] = embedding
    except Exception as e:
        print(f"Error processing {gene_symbol}: {e}")
        continue
print("Number of genes in gene_to_embedding:", len(gene_to_embedding))

torch.save(gene_to_embedding, "/home/b-evelyntong/hl/gene_to_embedding.pt")