"""
Module handling network propagation algorithms--heat diffusion and random walk with restart.

Author: Luke Krongard
"""

#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
import networkx as nx
import pandas as pd
from scipy.linalg import expm
from typing import Dict

#-----------------------------------------------------------------------------------------------------------------------

def adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """
    Returns the degree row-normalized adjacency matrix of a network.
    """
    # Get the adjacency matrix
    A = nx.to_numpy_array(G)

    # Get the degree matrix
    degrees = np.sum(A, axis=1)
    
    # Get the inverse of the degree matrix
    D_inv = np.zeros_like(A)
    np.fill_diagonal(D_inv, 1 / degrees)
    
    # Compute the degree row-normalized adjacency matrix
    W_D = -np.matmul(D_inv, A)

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
    W_D_hat = W_D + np.eye(W_D.shape[0])
    
    # Compute the heat diffusion matrix
    exp_matrix = expm(-W_D_hat * t)

    # Propagate scores
    F_t = np.matmul(exp_matrix, F0)
    print("Heat diffusion completed.")

    return F_t

#-----------------------------------------------------------------------------------------------------------------------

def random_walk(G: nx.Graph, F0: np.ndarray, alpha: float, max_iter: int = 1000, epsilon: float = 1e-6) -> np.ndarray:
    """
    Returns the random walk matrix after t iterations.
    """
    print("Running random walk algorithm...")
    # Get the degree row-normalized adjacency matrix
    W_D = adjacency_matrix(G)

    # Initialize the current and previous score vectors
    F_i = F0.copy()
    F_i_prev = np.zeros_like(F0)

    # Iterate until convergence
    for i in range(max_iter):
        # Store the previous score vector
        F_i_prev = F_i.copy()

        # Update scores using the random walk formula
        F_i = (1 - alpha) * F0 + alpha * np.matmul(W_D, F_i_prev)

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
    G = nx.read_graphml('Data/Networks/alcohol_network_700.graphml')

    # Load the mappings
    mappings = pd.read_csv('Data/Processed/alcohol_processed.tsv', sep='\t', index_col='Protein')
    mappings = mappings['Scaled_Robust_Sigmoid'].to_dict()

if __name__ == '__main__':
    main()