"""
Module handling network propagation algorithms--heat diffusion and random walk with restart.

Author: Luke Krongard
"""

#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from typing import Dict

#-----------------------------------------------------------------------------------------------------------------------

def adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """
    Returns the degree row-normalized adjacency matrix of a network.
    """
    # Get the adjacency matrix
    A = nx.to_scipy_sparse_array(G)

    # Get the degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    
    # Create inverse degree matrix (avoiding division by zero)
    with np.errstate(divide='ignore'):
        inv_deg = np.where(degrees > 0, 1.0 / degrees, 0)
    
    # Create diagonal matrix with inverse degrees
    D_inv = sp.diags(inv_deg)
    
    # Compute the normalized adjacency matrix
    W_D = D_inv @ A
    
    return W_D

#-----------------------------------------------------------------------------------------------------------------------

def heat_diffusion(G: nx.Graph, F0: np.ndarray, t: int) -> np.ndarray:
    """
    Returns the heat diffusion matrix after t iterations.
    """
    print("Running heat diffusion algorithm...")
    # Get the degree row-normalized adjacency matrix
    W_D = adjacency_matrix(G)

    # Add the identity matrix
    W_D_hat = -W_D + sp.eye(W_D.shape[0])
    
    # Efficiently compute the matrix exponential
    F_t = expm_multiply(-W_D_hat * t, F0)
    print("Heat diffusion completed.")

    return F_t

#-----------------------------------------------------------------------------------------------------------------------

def random_walk(G: nx.Graph, F0: np.ndarray, alpha: float, max_iter: int = 100, epsilon: float = 1e-6) -> np.ndarray:
    """
    Returns the random walk matrix after t iterations.
    """
    print("Running random walk algorithm...")
    # Get the degree row-normalized adjacency matrix
    W_D = adjacency_matrix(G)

    # Initialize vector of scores
    F_i = F0.copy()

    # Precompute restart vector
    restart = (1 - alpha) * F0

    # Iterate until convergence
    for i in range(max_iter):
        # Store the previous score vector
        F_i_prev = F_i.copy()

        # Update scores using the random walk formula
        F_i = restart + alpha * (W_D @ F_i_prev)

        # Check for convergence
        if np.linalg.norm(F_i - F_i_prev) < epsilon:
            print (f"Converged after {i} iterations.")
            break
    
    if i == max_iter - 1:
        print("Warning: Max iterations reached without convergence.")

    return F_i

#-----------------------------------------------------------------------------------------------------------------------

def run_propagation(G: nx.Graph, F0: np.ndarray, t: int, alpha: float) -> Dict[str, np.ndarray]:
    """
    Runs the heat diffusion and random walk algorithms.
    """
    print("Running network propagation algorithms...")
    F_hd = heat_diffusion(G, F0, t)
    F_rw = random_walk(G, F0, alpha)

    return {'Heat Diffusion': F_hd, 'Random Walk': F_rw}

#-----------------------------------------------------------------------------------------------------------------------

def main():
    # Load the network
    G = nx.read_graphml('Data/Networks/alcohol_network_900.graphml')

    # Load the mappings
    mappings = pd.read_csv('Data/Processed/alcohol_processed.tsv', sep='\t', index_col='Protein')
    mappings = mappings['Scaled_Robust_Sigmoid'].to_dict()

    # Convert the mappings to a numpy array
    F0 = np.zeros(G.number_of_nodes())
    for i, node in enumerate(G.nodes):
        if node in mappings:
            F0[i] = mappings[node]

    # Run the network propagation algorithms
    results = run_propagation(G, F0, t=3, alpha=0.7)
    
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
    output_path = 'Data/Results/propagation_results.tsv'
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()