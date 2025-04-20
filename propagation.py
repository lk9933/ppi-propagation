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

def random_walk(W_D: sp.csr_matrix, F0: np.ndarray, alpha: float, max_iter: int = 100, 
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

def main() -> None:
    G = create_network(min_score=700)
    maps = pd.read_csv('Data/Processed/alcohol_processed.tsv', sep='\t',
                       index_col='Protein')['Scaled_Robust_Sigmoid']
    F0 = np.fromiter((maps.get(n, 0.0) for n in G.nodes()), dtype=np.float32)

    res = run_propagation(G, F0, t=5.0, alpha=0.8)

    df = pd.DataFrame({
        'Protein': list(G.nodes()),
        'HD_Score': res['Heat Diffusion'],
        'RW_Score': res['Random Walk']
    })
    df['HD_Rank'] = df['HD_Score'].rank(method='dense', ascending=False)
    df['RW_Rank'] = df['RW_Score'].rank(method='dense', ascending=False)
    df['Avg_Rank'] = (df['HD_Rank'] + df['RW_Rank']) / 2
    df.sort_values('Avg_Rank', inplace=True)
    df.to_csv('Results/propagation_results.tsv', sep='\t', index=False)
    print("✓ Results saved to Results/propagation_results.tsv")

if __name__ == '__main__':
    main()