"""
Module handling network propagation algorithms--heat diffusion and random walk with restart.

Author: Luke Krongard
"""

#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse as sp
from network import create_network
from scipy.sparse.linalg import expm_multiply
from typing import Dict
from query import query_proteins_by_aliases_batch
import sqlite3

#-----------------------------------------------------------------------------------------------------------------------

def preprocess_graph(G: nx.Graph) -> nx.Graph:
    """
    Preprocess the graph by adding self-loops to isolated nodes.
    """
    G = G.copy()
    isolated = [n for n, deg in G.degree() if deg == 0]
    G.add_edges_from((n, n, {'weight': 1.0}) for n in isolated)
    return G

#-----------------------------------------------------------------------------------------------------------------------

def row_normalized_adjacency_matrix(G: nx.Graph) -> sp.spmatrix:
    """
    Returns the degree row-normalized adjacency matrix of a network as a sparse matrix.
    """
    A = nx.to_scipy_sparse_array(G, dtype=np.float32, format='csr')
    deg = np.asarray(A.sum(1)).ravel()
    inv_deg = np.divide(1.0, deg, where=deg > 0, dtype=np.float32)  
    W_D = -1 * sp.diags(inv_deg) @ A          # shape (n,n), CSR    
    W_D.sort_indices()
    return W_D

#-----------------------------------------------------------------------------------------------------------------------

def heat_diffusion(W_D: sp.csr_matrix, F0: np.ndarray, t: int) -> np.ndarray:
    """
    Returns the heat diffusion scores after t iterations using a precomputed row-normalized 
    adjacency matrix W_D
    """
    W_hat = W_D + sp.eye(W_D.shape[0], dtype=W_D.dtype, format='csr')
    return expm_multiply(-W_hat * t, F0).ravel()      # .A1 = flat ndarray

#-----------------------------------------------------------------------------------------------------------------------

def random_walk(W_D: sp.csr_matrix, F0: np.ndarray, alpha: float, max_iter: int = 1000, 
                epsilon: float = 1e-6) -> np.ndarray:
    """
    Returns the random walk scores after iterative updates using a precomputed 
    row-normalized adjacency matrix W_D.
    """
    F = F0.copy()
    restart = (1 - alpha) * F0
    for _ in range(max_iter):
        F_new = restart - alpha * (W_D @ F)
        if np.linalg.norm(F_new - F) < epsilon:
            return F_new
        F = F_new
    print("Warning: RWR hit max_iter without convergence.")
    return F

#-----------------------------------------------------------------------------------------------------------------------

def run_propagation(G: nx.Graph,
                    F0: np.ndarray,
                    t: float,
                    alpha: float) -> Dict[str, np.ndarray]:
    G = preprocess_graph(G)
    W_D = row_normalized_adjacency_matrix(G)
    return {
        'Heat Diffusion': heat_diffusion(W_D, F0, t),
        'Random Walk':   random_walk(W_D, F0, alpha)
    }

#-----------------------------------------------------------------------------------------------------------------------

def load_disgenet_set(file_path: str) -> pd.DataFrame:
    """
    Load protein set from a DisGeNET file.
    """
    # Load DisGeNET data
    disease_genes = pd.read_csv(file_path, sep='\t')['Gene']
    
    # Convert to list
    disease_genes_list = disease_genes.tolist()

    # Map to proteins
    conn = sqlite3.connect('Data/SQLite_DB/genomics.db') 
    proteins = query_proteins_by_aliases_batch(conn, disease_genes_list)
    conn.close()

    # Convert dictionaries to list of their values
    proteins_list = [item for sublist in proteins.values() for item in sublist]

    # Remove duplicates
    proteins_list = list(set(proteins_list))

    # Return the list as a dataframe
    return pd.DataFrame(proteins_list, columns=['Protein'])

#-----------------------------------------------------------------------------------------------------------------------

def main() -> None:
    G = create_network(min_score=400)
    opt_t = 2.3
    opt_alpha = 0.50

    # Load all OMIM sets
    alcohol_proteins = load_disgenet_set('Data/DisGeNET/alcoholism.tsv')

    # Construct F0
    F0 = np.zeros(G.number_of_nodes(), dtype=np.float32)
    for protein in alcohol_proteins:
        if protein in G.nodes():
            F0[G.nodes().index(protein)] = 1.0
    
    # Run propagation
    results = run_propagation(G, F0, opt_t, opt_alpha)

    # Rank the top-20 proteins that weren't in input set
    for algo, scores in results.items():
        ranked_indices = np.argsort(scores)[::-1]
        ranked_proteins = [list(G.nodes())[i] for i in ranked_indices if i not in alcohol_proteins][:20]
        print(f"Top 20 proteins for {algo}:")
        print(ranked_proteins)
        print("")
    # Save results to CSV
    df = pd.DataFrame({
        'Protein': list(G.nodes()),
        'Heat Diffusion': results['Heat Diffusion'],
        'Random Walk': results['Random Walk']
    })
    df.to_csv('propagation_results.csv', index=False)
    print("Propagation results saved to 'propagation_results.csv'.")


if __name__ == '__main__':
    main()