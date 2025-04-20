"""
Module handling the creation of PPI networks.

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
from query import query_interactions_by_score, query_interactions_by_protein

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
    G = nx.Graph()

    # Add nodes and edges to the graph
    for interaction in interactions:
        protein1, protein2, weight = interaction
        G.add_edge(protein1, protein2, weight=(weight/1000))
    
    return G

#-----------------------------------------------------------------------------------------------------------------------

def create_network_from_seeds(seeds: List[str], num_hops: int = 3, min_score: int = 400) -> nx.Graph:
    """
    Creates a protein-protein interaction network based on a list of seed proteins.
    """
    # Connect to the database
    conn = sqlite3.connect('Data/SQLite_DB/genomics.db')

    # Create a new graph
    G = nx.Graph()
    
    # Initialize the set of proteins to process in this hop
    current_proteins = set(seeds)
    
    # Initialize the set of proteins already processed
    processed_proteins = set()
    
    # Iterate for each hop
    for hop in range(num_hops):
        print(f"Processing hop {hop+1}/{num_hops}: {len(current_proteins)} proteins to process")
        
        # Get the next set of proteins to process
        next_proteins = set()
        
        # Process each protein in the current hop
        for protein in current_proteins:
            # Skip if we've already processed this protein
            if protein in processed_proteins:
                continue
                
            # Mark this protein as processed
            processed_proteins.add(protein)
            
            # Get interactions for this protein
            interactions = query_interactions_by_protein(conn, protein, min_score)
            
            # Add edges to the graph
            for interaction in interactions:
                protein1, protein2, weight = interaction
                G.add_edge(protein1, protein2, weight=weight)
                
                # Add the neighbor to the next hop
                if protein2 not in processed_proteins:
                    next_proteins.add(protein2)
        
        # Update the set of proteins to process in the next hop
        current_proteins = next_proteins
        
        # Break if there are no more proteins to process
        if not current_proteins:
            print(f"No more proteins to process after hop {hop+1}")
            break
    
    # Close the connection
    conn.close()
    
    print(f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

#-----------------------------------------------------------------------------------------------------------------------

def main():
    G = create_network(min_score=400)
    print(f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

if __name__ == "__main__":
    main()