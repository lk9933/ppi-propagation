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

def row_normalized_adjacency_matrix(G: nx.Graph) -> sp.spmatrix:
    """
    Returns the degree row-normalized adjacency matrix of a network as a sparse matrix.
    """
    # Get the adjacency matrix as a sparse CSR matrix
    A = nx.to_scipy_sparse_array(G, format='csr')
    
    # Compute node degrees using a sparse operation
    degrees = np.array(A.sum(axis=1)).flatten()
    
    # Calculate inverse degrees, avoiding division by zero
    with np.errstate(divide='ignore'):
        inv_deg = np.where(degrees > 0, 1.0 / degrees, 0)
    
    # Create a diagonal sparse matrix from inverse degrees
    D_inv = sp.diags(inv_deg)
    
    # Compute the row-normalized adjacency matrix using sparse matrix multiplication
    W_D = D_inv @ A
    
    return W_D

#-----------------------------------------------------------------------------------------------------------------------

def normalized_laplacian_matrix(G: nx.Graph) -> sp.spmatrix:
    """
    Returns the normalized Laplacian matrix of a network as a sparse matrix.
    """
    # Get the adjacency matrix as a sparse CSR matrix
    A = nx.to_scipy_sparse_array(G, format='csr')
    
    # Compute node degrees using a sparse operation
    degrees = np.array(A.sum(axis=1)).flatten()
    
    # Calculate inverse square root of degrees, avoiding division by zero
    with np.errstate(divide='ignore'):
        inv_sqrt_deg = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0)
    
    # Create a diagonal degree sparse matrix
    D = sp.diags(degrees)

    # Create a diagonal sparse matrix from inverse square root of degrees
    D_inv_sqrt = sp.diags(inv_sqrt_deg)

    # Compute the Laplacian matrix
    W_L = D - A
    
    # Compute the normalized Laplacian matrix using sparse matrix multiplication
    W_L_hat = D_inv_sqrt @ W_L @ D_inv_sqrt
    
    return W_L_hat

#-----------------------------------------------------------------------------------------------------------------------

def heat_diffusion(G: nx.Graph, F0: np.ndarray, t: int) -> np.ndarray:
    """
    Returns the heat diffusion scores after t iterations using a precomputed row-normalized 
    adjacency matrix (W_D) if provided.
    """
    W_D = row_normalized_adjacency_matrix(G)
    
    F0_sparse = sp.csr_matrix(F0.reshape(-1, 1))
    W_D_hat = -W_D + sp.eye(W_D.shape[0])
    F_t = expm_multiply(-W_D_hat * t, F0_sparse)
    
    return F_t.toarray().flatten()

#-----------------------------------------------------------------------------------------------------------------------

def random_walk(G: nx.Graph, F0: np.ndarray, alpha: float, max_iter: int = 100, 
                epsilon: float = 1e-6) -> np.ndarray:
    """
    Returns the random walk scores after iterative updates. Optionally reuses a precomputed 
    row-normalized adjacency matrix (W_D).
    """
    W_D = row_normalized_adjacency_matrix(G)

    F0_sparse = sp.csr_matrix(F0.reshape(-1, 1))
    F_i = F0_sparse.copy()
    restart = (1 - alpha) * F0_sparse

    for i in range(max_iter):
        F_i_prev = F_i.copy()
        F_i = restart + alpha * (W_D @ F_i_prev)
        diff = F_i - F_i_prev
        if sp.linalg.norm(diff) < epsilon:
            break

    if i == max_iter - 1:
        print("Warning: Max iterations reached without convergence.")

    return F_i.toarray().flatten()

#-----------------------------------------------------------------------------------------------------------------------

def run_propagation(G: nx.Graph, F0: np.ndarray, t: int, alpha: float) -> Dict[str, np.ndarray]:
    """
    Runs the heat diffusion and random walk algorithms.
    """
    F_hd = heat_diffusion(G, F0, t)
    F_rw = random_walk(G, F0, alpha)

    return {'Heat Diffusion': F_hd, 'Random Walk': F_rw}

#-----------------------------------------------------------------------------------------------------------------------

def main():
    # Load the network
    G = create_network(min_score=700)

    # Load the mappings
    mappings = pd.read_csv('Data/Processed/alcohol_processed.tsv', sep='\t', index_col='Protein')
    mappings = mappings['Scaled_Robust_Sigmoid'].to_dict()

    # Convert the mappings to a numpy array
    F0 = np.zeros(G.number_of_nodes())
    for i, node in enumerate(G.nodes):
        if node in mappings:
            F0[i] = mappings[node]

    # Run the network propagation algorithms
    results = run_propagation(G, F0, t=5, alpha=0.8)
    
    # Create a DataFrame with proteins and scores from both algorithms
    df = pd.DataFrame({'Protein': list(G.nodes)})
    
    # Add scores for each algorithm
    df['HD_Score'] = results['Heat Diffusion']
    df['RW_Score'] = results['Random Walk']
    
    # Calculate ranks for each algorithm (using dense rank to handle ties)
    df['HD_Rank'] = df['HD_Score'].rank(method='dense', ascending=False)
    df['RW_Rank'] = df['RW_Score'].rank(method='dense', ascending=False)
    
    # Calculate average rank
    df['Avg_Rank'] = (df['HD_Rank'] + df['RW_Rank']) / 2
    
    # Sort by average rank (ascending)
    df.sort_values(by='Avg_Rank', ascending=True, inplace=True)
    
    # Save results to TSV file
    output_path = 'Results/propagation_results.tsv'
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()