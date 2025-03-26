"""
Module handling the creation and augmention of PPI networks.

Requires the following files:
Data/Processed/*.tsv - Preprocessed mapping files of proteins, p-values, genes, and SNPs.
Data/SQLite_DB/genomics.db - SQLite3 database containing GWAS and STRING data.

Author: Luke Krongard
"""

#-----------------------------------------------------------------------------------------------------------------------

import networkx as nx
import pandas as pd
import sqlite3
from typing import Dict, List, Tuple
from query import query_interactions_by_score

#-----------------------------------------------------------------------------------------------------------------------

def create_network(min_score: int = 400) -> nx.Graph:
    """
    Creates a protein-protein interaction network based on interactions with a minimum score.
    """
    # Connect to the database
    conn = sqlite3.connect('Data/SQLite_DB/genomics.db')

    # Get interactions from the database
    interactions = query_interactions_by_score(conn, min_score)

    # Close the connection
    conn.close()

    # Create a new graph
    network = nx.Graph()

    # Add nodes and edges to the graph
    for interaction in interactions:
        protein1, protein2, weight = interaction
        network.add_edge(protein1, protein2, weight=weight)
    
    print(f"Created network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
    return network

#-----------------------------------------------------------------------------------------------------------------------

def augment_network(network: nx.Graph, mappings: Dict[str, float], continuous: bool = False) -> nx.Graph:
    """
    Augments a network with nodes and edges based on mappings.
    """
    if not continuous:
        # Assign a score of 1 to all proteins in the mapping
        for protein in mappings:
            network.nodes[protein]['score'] = 1
    else:
        # Otherwise, assign the scaled robust sigmoid value to each protein
        for protein, score in mappings.items():
            network.nodes[protein]['score'] = score
    
    print(f"Augmented {len(mappings)} proteins in the network.")
    return network

#-----------------------------------------------------------------------------------------------------------------------

def main():
    # Set the file path
    file_path = 'Data/Processed/alcohol_processed.tsv'

    # Create the network
    network = create_network(min_score=0)

    # Load the mapping file
    mappings = pd.read_csv(file_path, sep='\t')

    # Convert the mappings to a dictionary
    mappings_dict = mappings.set_index('Protein')['Scaled_Robust_Sigmoid'].to_dict()

    # Augment the network
    network = augment_network(network, mappings_dict, continuous=False)

    # Save the network to a file
    nx.write_graphml(network, 'Data/Networks/alcohol_network.graphml')

if __name__ == '__main__':
    main()