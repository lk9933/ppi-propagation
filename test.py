"""
Module handling cross-validation and evaluation of network propagation algorithms.
"""

#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import networkx as nx
from network import create_network, create_network_from_seeds, augment_network
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from propagation import run_propagation
from typing import Dict, List, Tuple

#-----------------------------------------------------------------------------------------------------------------------

def load_protein_scores(file_path: str) -> pd.DataFrame:
    """
    Loads preprocessed protein scores from a directory.
    """
    # Load the protein mappings
    data = pd.read_csv(file_path, sep='\t')

    # Drop unnecessary columns
    data.drop(columns=['Gene', 'SNP'], inplace=True)

    return data

#-----------------------------------------------------------------------------------------------------------------------

def prepare_network(protein_scores: pd.DataFrame, network_params: Tuple) -> nx.Graph:
    """
    Prepares the network for propagation using the protein mappings.
    """
    # Extract network parameters
    min_score, complete, num_hops, continuous = network_params

    # Prepare the network
    G = nx.Graph()
    if complete:
        G = create_network(min_score)
    else:
        seeds = protein_scores['Protein'].tolist()
        G = create_network_from_seeds(seeds, num_hops, min_score)

    # Augment the network
    G = augment_network(G, protein_scores, continuous)

    return G

#-----------------------------------------------------------------------------------------------------------------------




