"""
Module handling cross-validation and evaluation of network propagation algorithms.
"""

#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import networkx as nx
import sqlite3
from network import create_network, create_network_from_seeds
from query import query_proteins_by_aliases_batch
from sklearn.model_selection import StratifiedKFold
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve, precision_score, recall_score, auc
from visualize import generate_roc_curve, generate_prc_curve
from propagation import heat_diffusion, random_walk
from typing import Dict, Tuple

#-----------------------------------------------------------------------------------------------------------------------

def load_omim_set(file_path: str) -> pd.DataFrame:
    """
    Load protein set from an OMIM file.
    """
    # Load OMIM data
    genes = pd.read_csv(file_path)
    
    # Convert to list
    genes_list = genes['Gene'].tolist()

    # Map to proteins
    conn = sqlite3.connect('Data/SQLite_DB/genomics.db') 
    proteins = query_proteins_by_aliases_batch(conn, genes_list)
    conn.close()

    # Convert dictionaries to list of their values
    proteins_list = [item for sublist in proteins.values() for item in sublist]

    # Remove duplicates
    proteins_list = list(set(proteins_list))

    # Return the list as a dataframe
    return pd.DataFrame(proteins_list, columns=['Protein'])

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

def heat_diffusion_test(protein_set: pd.DataFrame, G: nx.Graph, t: float) -> dict:
    """
    Test the heat diffusion algorithm on a set of proteins focusing only on AUROC.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proteins = protein_set['Protein'].values
    labels = protein_set['Label'].values
    node_indices = {node: idx for idx, node in enumerate(G.nodes())}

    auc_scores = []
    
    for train_index, test_index in skf.split(proteins, labels):
        F0 = np.zeros(G.number_of_nodes())
        for i in train_index:
            if proteins[i] in G.nodes():
                F0[node_indices[proteins[i]]] = labels[i]
        
        F_hd = heat_diffusion(G, F0, t)
        
        y_true = labels[test_index]
        y_scores = [F_hd[node_indices[proteins[i]]] if proteins[i] in node_indices else 0 
                    for i in test_index]

        # Calculate ROC AUC for this fold
        auc = roc_auc_score(y_true, y_scores)
        auc_scores.append(auc)
    
    # Calculate mean AUROC
    mean_auc = np.mean(auc_scores)
    return {'t': t, 'roc_auc': mean_auc}

#-----------------------------------------------------------------------------------------------------------------------

def random_walk_test(protein_set: pd.DataFrame, G: nx.Graph, alpha: float) -> dict:
    """
    Test the random walk algorithm on a set of proteins focusing only on AUROC.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proteins = protein_set['Protein'].values
    labels = protein_set['Label'].values
    node_indices = {node: i for i, node in enumerate(G.nodes())}

    auc_scores = []
    
    for train_index, test_index in skf.split(proteins, labels):
        F0 = np.zeros(G.number_of_nodes())
        for i in train_index:
            if proteins[i] in G.nodes():
                F0[node_indices[proteins[i]]] = labels[i]
        
        F_rw = random_walk(G, F0, alpha)
        
        y_true = labels[test_index]
        y_scores = [F_rw[node_indices[proteins[i]]] if proteins[i] in node_indices else 0 
                    for i in test_index]

        auc = roc_auc_score(y_true, y_scores)
        auc_scores.append(auc)
    
    mean_auc = np.mean(auc_scores)
    return {'alpha': alpha, 'roc_auc': mean_auc}

#-----------------------------------------------------------------------------------------------------------------------

def heat_diffusion_nested_cv(protein_set: pd.DataFrame, G: nx.Graph, t_values: list) -> dict:
    """
    Nested cross-validation for heat diffusion.
    The dataset is split into 5 folds. For each outer iteration:
      - Test fold = fold[i]
      - Validation fold = fold[(i+1) % 5]
      - Training folds = the remaining 3 folds
    For each candidate t in t_values, we compute AUROC on the validation set using the training set.
    Then, we retrain on training+validation and evaluate on the outer test set.
    Additionally, we compute the average t across folds and perform a fixed CV evaluation.
    Returns the list of selected t values, outer AUROCs, mean outer AUROC,
    the average t, and the fixed CV results.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proteins = protein_set['Protein'].values
    labels = protein_set['Label'].values
    all_folds = list(skf.split(proteins, labels))
    node_indices = {node: idx for idx, node in enumerate(G.nodes())}
    
    outer_aucs = []
    best_ts = []
    
    for i in range(5):
        test_idx = all_folds[i][1]
        val_idx = all_folds[(i+1) % 5][1]
        train_idx = []
        for j in range(5):
            if j != i and j != ((i+1) % 5):
                train_idx.extend(all_folds[j][1])
        
        train_set = protein_set.iloc[train_idx]
        val_set = protein_set.iloc[val_idx]
        test_set = protein_set.iloc[test_idx]
        
        best_t = None
        best_val_auc = -1
        for t in t_values:
            F0_train = np.zeros(G.number_of_nodes())
            for prot in train_set['Protein'].values:
                if prot in G.nodes():
                    idx = node_indices[prot]
                    F0_train[idx] = train_set.loc[train_set['Protein'] == prot, 'Label'].iloc[0]
            F_hd_train = heat_diffusion(G, F0_train, t)
            val_y_true = val_set['Label'].values
            val_y_scores = [F_hd_train[node_indices[p]] if p in node_indices else 0 
                            for p in val_set['Protein'].values]
            val_auc = roc_auc_score(val_y_true, val_y_scores)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_t = t
        
        best_ts.append(best_t)
        train_val_set = pd.concat([train_set, val_set], ignore_index=True)
        F0_train_val = np.zeros(G.number_of_nodes())
        for prot in train_val_set['Protein'].values:
            if prot in G.nodes():
                idx = node_indices[prot]
                F0_train_val[idx] = train_val_set.loc[train_val_set['Protein'] == prot, 'Label'].iloc[0]
        F_hd_full = heat_diffusion(G, F0_train_val, best_t)
        test_y_true = test_set['Label'].values
        test_y_scores = [F_hd_full[node_indices[p]] if p in node_indices else 0 
                         for p in test_set['Protein'].values]
        test_auc = roc_auc_score(test_y_true, test_y_scores)
        outer_aucs.append(test_auc)
        print(f"Outer CV fold {i+1}: best t = {best_t}, Test AUROC = {test_auc:.4f}")
        
    mean_auc = np.mean(outer_aucs)
    print(f"Mean outer AUROC: {mean_auc:.4f}")
    
    avg_t = np.mean(best_ts)
    # Perform fixed parameter CV using the averaged t
    fixed_cv_results = heat_diffusion_test(protein_set, G, avg_t)
    print(f"Fixed CV with avg t = {avg_t}: Mean AUROC = {fixed_cv_results['roc_auc']:.4f}")
    
    return {
        'selected_t': best_ts,
        'outer_aucs': outer_aucs,
        'mean_outer_auc': mean_auc,
        'avg_t': avg_t,
        'fixed_cv': fixed_cv_results
    }

#-----------------------------------------------------------------------------------------------------------------------

def random_walk_nested_cv(protein_set: pd.DataFrame, G: nx.Graph, alpha_values: list) -> dict:
    """
    Nested cross-validation for random walk.
    Using the same 5-fold scheme (3 folds training, 1 fold validation, 1 fold testing).
    For each outer split, determine the best alpha on the validation set,
    retrain on training+validation and evaluate on test.
    Additionally, compute the average alpha and run a fixed CV evaluation.
    Returns the list of selected alpha values, outer AUROCs, mean outer AUROC,
    the average alpha, and the fixed CV results.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proteins = protein_set['Protein'].values
    labels = protein_set['Label'].values
    all_folds = list(skf.split(proteins, labels))
    node_indices = {node: i for i, node in enumerate(G.nodes())}
    
    outer_aucs = []
    best_alphas = []
    
    for i in range(5):
        test_idx = all_folds[i][1]
        val_idx = all_folds[(i+1) % 5][1]
        train_idx = []
        for j in range(5):
            if j != i and j != ((i+1) % 5):
                train_idx.extend(all_folds[j][1])
        
        train_set = protein_set.iloc[train_idx]
        val_set = protein_set.iloc[val_idx]
        test_set = protein_set.iloc[test_idx]
        
        best_alpha = None
        best_val_auc = -1
        for alpha in alpha_values:
            F0_train = np.zeros(G.number_of_nodes())
            for prot in train_set['Protein'].values:
                if prot in G.nodes():
                    idx = node_indices[prot]
                    F0_train[idx] = train_set.loc[train_set['Protein'] == prot, 'Label'].iloc[0]
            F_rw_train = random_walk(G, F0_train, alpha)
            val_y_true = val_set['Label'].values
            val_y_scores = [F_rw_train[node_indices[p]] if p in node_indices else 0 
                            for p in val_set['Protein'].values]
            val_auc = roc_auc_score(val_y_true, val_y_scores)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_alpha = alpha
        
        best_alphas.append(best_alpha)
        train_val_set = pd.concat([train_set, val_set], ignore_index=True)
        F0_train_val = np.zeros(G.number_of_nodes())
        for prot in train_val_set['Protein'].values:
            if prot in G.nodes():
                idx = node_indices[prot]
                F0_train_val[idx] = train_val_set.loc[train_val_set['Protein'] == prot, 'Label'].iloc[0]
        F_rw_full = random_walk(G, F0_train_val, best_alpha)
        test_y_true = test_set['Label'].values
        test_y_scores = [F_rw_full[node_indices[p]] if p in node_indices else 0 
                         for p in test_set['Protein'].values]
        test_auc = roc_auc_score(test_y_true, test_y_scores)
        outer_aucs.append(test_auc)
        print(f"Outer CV fold {i+1}: best alpha = {best_alpha}, Test AUROC = {test_auc:.4f}")
    
    mean_auc = np.mean(outer_aucs)
    print(f"Mean outer AUROC: {mean_auc:.4f}")
    
    avg_alpha = np.mean(best_alphas)
    # Perform fixed parameter CV using the averaged alpha
    fixed_cv_results = random_walk_test(protein_set, G, avg_alpha)
    print(f"Fixed CV with avg alpha = {avg_alpha}: Mean AUROC = {fixed_cv_results['roc_auc']:.4f}")
    
    return {
        'selected_alpha': best_alphas,
        'outer_aucs': outer_aucs,
        'mean_outer_auc': mean_auc,
        'avg_alpha': avg_alpha,
        'fixed_cv': fixed_cv_results
    }

#-----------------------------------------------------------------------------------------------------------------------

def run_tests(positive_set: pd.DataFrame, negative_set: pd.DataFrame, G: nx.Graph, disease: str) -> None:
    """
    Run tests for heat diffusion and random walk algorithms using only AUROC as performance metric.
    """
    print("Loading input sets...")
    input = pd.concat([positive_set, negative_set], ignore_index=True)

    input['Label'] = 0
    input.loc[input['Protein'].isin(positive_set['Protein']), 'Label'] = 1
    input.loc[input['Protein'].isin(negative_set['Protein']), 'Label'] = 0

    input.drop_duplicates(subset='Protein', inplace=True)

    missing_proteins = set(input['Protein'].tolist()) - set(G.nodes())
    if missing_proteins:
        print(f"Adding {len(missing_proteins)} missing proteins to the network.")
        G.add_nodes_from(missing_proteins)

    # Collect AUROC results for Heat Diffusion
    hd_results = []
    print("Testing Heat Diffusion...")
    t_values = [1, 2, 4, 8, 16]
    heat_diffusion_nested_cv(input, G, t_values)

    # Collect AUROC results for Random Walk
    rw_results = []
    print("Testing Random Walk...")
    alpha_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    random_walk_nested_cv(input, G, alpha_values)

    # Print out the best parameter based on AUROC for each algorithm
    best_hd = max(hd_results, key=lambda d: d['roc_auc'])
    best_rw = max(rw_results, key=lambda d: d['roc_auc'])
    
    print("\nOptimal Parameters based on AUROC:")
    print(f"Heat Diffusion: t = {best_hd['t']} with AUROC = {best_hd['roc_auc']:.4f}")
    print(f"Random Walk: α = {best_rw['alpha']} with AUROC = {best_rw['roc_auc']:.4f}")

#-----------------------------------------------------------------------------------------------------------------------

def disease_test(positive_file: str, negative_file: str, disease: str, file_type_pos: str, file_type_neg: str):
    """
    Load the disease set and run tests.
    """
    # Load the disease set
    if file_type_pos == 'omim':
        disease_set = load_omim_set(positive_file)
    elif file_type_pos == 'disgenet':
        disease_set = load_disgenet_set(positive_file)
    else:
        raise ValueError(f"Unsupported file type for positive set: {file_type_pos}")
    
    # Load the negative set
    if file_type_neg == 'omim':
        negative_set = load_omim_set(negative_file)
    elif file_type_neg == 'disgenet':
        negative_set = load_disgenet_set(negative_file)
    else:
        raise ValueError(f"Unsupported file type for negative set: {file_type_neg}")
    
    # Get the minimum number of proteins to sample
    min_proteins = min(len(disease_set), len(negative_set))

    # Randomly sample min_genes from both sets
    positive_set = disease_set.sample(n=min_proteins, random_state=42)
    negative_set = negative_set.sample(n=min_proteins, random_state=42)

    # Create the network from the protein set
    G = create_network(min_score=400)

    # Run tests
    run_tests(positive_set, negative_set, G, disease=disease)

#-----------------------------------------------------------------------------------------------------------------------

def main():
    # Example usage
    positive_file = 'Data/OMIM/Processed/epilepsy_genes.txt'
    negative_file = 'Data/OMIM/Processed/deafness_genes.txt'
    disease = 'epilepsy'
    file_type_pos = 'omim'
    file_type_neg = 'omim'

    disease_test(positive_file, negative_file, disease, file_type_pos, file_type_neg)

if __name__ == '__main__':
    main()