"""
Module handling cross-validation and evaluation of network propagation algorithms.

Author: Luke Krongard
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
import multiprocessing as mp
from multiprocessing import Lock, Queue
import time
from functools import partial
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve, precision_score, recall_score, auc
from visualize import generate_roc_curve, generate_prc_curve, generate_disease_roc_curve
from propagation import heat_diffusion, random_walk, row_normalized_adjacency_matrix, preprocess_graph
from typing import Dict, Tuple, List, Any
from queue import Empty  # Import Empty exception from queue module
import json
from sklearn.linear_model import LogisticRegression

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

def heat_diffusion_nested_cv(protein_set: pd.DataFrame, G: nx.Graph, t_values: list, disease: str) -> dict:
    """
    Nested cross-validation for heat diffusion.
    For each outer CV fold:
      - Determine the best t parameter on the validation set.
      - Retrain on training+validation and evaluate on the outer test set.
    Returns the selected t values, fold AUROCs, mean AUROC, mean t, and ROC curve data.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proteins = protein_set['Protein'].values
    labels = protein_set['Label'].values
    all_folds = list(skf.split(proteins, labels))
    node_indices = {node: idx for idx, node in enumerate(G.nodes())}
    
    outer_aucs = []
    best_ts = []
    fold_results = []  # to store summary of each fold
    all_fpr = []     # list to store false positive rates of each fold
    all_tpr = []     # list to store true positive rates of each fold

    G_processed = preprocess_graph(G)
    W_D = row_normalized_adjacency_matrix(G_processed)
    
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
        # Iterate through candidate t values
        for t in t_values:
            F0_train = np.zeros(G.number_of_nodes())
            for prot in train_set['Protein'].values:
                if prot in G.nodes():
                    idx = node_indices[prot]
                    F0_train[idx] = train_set.loc[train_set['Protein'] == prot, 'Label'].iloc[0]
            F_hd_train = heat_diffusion(W_D, F0_train, t)
            val_y_true = val_set['Label'].values
            val_y_scores = [F_hd_train[node_indices[p]] if p in node_indices else 0 
                            for p in val_set['Protein'].values]
            # Normalize scores
            train_scores = [
                F_hd_train[node_indices[p]]
                for p in train_set['Protein']
                if p in node_indices
            ]
            mu, sigma    = np.mean(train_scores), np.std(train_scores, ddof=0)
            val_y_scores = [0.0 if sigma == 0 else (s - mu) / sigma for s in val_y_scores]

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
                F0_train_val[idx] = train_val_set.loc[train_val_set['Protein'] == prot, 'Label'].iat[0]
        F_hd_full = heat_diffusion(W_D, F0_train_val, best_t)
        test_y_true = test_set['Label'].values
        test_y_scores = [F_hd_full[node_indices[p]] if p in node_indices else 0 
                         for p in test_set['Protein'].values]
        # Normalize test scores
        train_scores = [
            F_hd_full[node_indices[p]]
            for p in train_val_set['Protein']
            if p in node_indices
        ]
        mu, sigma    = np.mean(train_scores), np.std(train_scores, ddof=0)
        test_y_scores = [0.0 if sigma == 0 else (s - mu) / sigma for s in test_y_scores]

        # Calculate AUROC
        test_auc = roc_auc_score(test_y_true, test_y_scores)
        test_auc = round(test_auc, 4)  # Round to 4 decimals
        outer_aucs.append(test_auc)
        
        # Log fold result
        fold_result = {
            'Algorithm': 'Heat Diffusion',
            'Fold': i+1,
            'BestParameter': round(best_t, 2),  # Round to 2 decimals
            'Test_AUROC': test_auc
        }
        fold_results.append(fold_result)
        
        fpr, tpr, _ = roc_curve(test_y_true, test_y_scores)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
    
    mean_auc = round(np.mean(outer_aucs), 4)  # Round to 4 decimals
    
    avg_t = round(np.mean(best_ts), 2)  # Round to 2 decimals
    
    return {
        'selected_t': best_ts,
        'outer_aucs': outer_aucs,
        'mean_outer_auc': mean_auc,
        'avg_t': avg_t,
        'fold_results': fold_results,
        'all_fpr': all_fpr,
        'all_tpr': all_tpr
    }

#-----------------------------------------------------------------------------------------------------------------------

def random_walk_nested_cv(protein_set: pd.DataFrame, G: nx.Graph, alpha_values: list, disease: str) -> dict:
    """
    Nested cross-validation for random walk.
    For each outer CV fold:
      - Determine the best alpha on the validation set.
      - Retrain on training+validation and evaluate on the outer test set.
    Returns the selected alpha values, fold AUROCs, mean AUROC, mean alpha, and ROC curve data.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proteins = protein_set['Protein'].values
    labels = protein_set['Label'].values
    all_folds = list(skf.split(proteins, labels))
    node_indices = {node: i for i, node in enumerate(G.nodes())}
    
    outer_aucs = []
    best_alphas = []
    fold_results = []
    all_fpr = []
    all_tpr = []

    G_processed = preprocess_graph(G)
    W_D = row_normalized_adjacency_matrix(G_processed)
    
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
                    F0_train[idx] = train_set.loc[train_set['Protein'] == prot, 'Label'].iat[0]
            F_rw_train = random_walk(W_D, F0_train, alpha)
            val_y_true = val_set['Label'].values
            val_y_scores = [F_rw_train[node_indices[p]] if p in node_indices else 0 
                            for p in val_set['Protein'].values]
            # Normalize scores
            train_scores = [
                F_rw_train[node_indices[p]]
                for p in train_set['Protein']
                if p in node_indices
            ]
            mu, sigma    = np.mean(train_scores), np.std(train_scores, ddof=0)
            val_y_scores = [0.0 if sigma == 0 else (s - mu) / sigma for s in val_y_scores]

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
        F_rw_full = random_walk(W_D, F0_train_val, best_alpha)
        test_y_true = test_set['Label'].values
        test_y_scores = [F_rw_full[node_indices[p]] if p in node_indices else 0 
                         for p in test_set['Protein'].values]
        # Normalize test scores
        train_scores = [
            F_rw_full[node_indices[p]]
            for p in train_val_set['Protein']
            if p in node_indices
        ]
        mu, sigma    = np.mean(train_scores), np.std(train_scores, ddof=0)
        test_y_scores = [0.0 if sigma == 0 else (s - mu) / sigma for s in test_y_scores]

        test_auc = roc_auc_score(test_y_true, test_y_scores)
        test_auc = round(test_auc, 4)  # Round to 4 decimals
        outer_aucs.append(test_auc)
        
        # Log fold result
        fold_result = {
            'Algorithm': 'Random Walk',
            'Fold': i+1,
            'BestParameter': round(best_alpha, 2),  # Round to 2 decimals
            'Test_AUROC': test_auc
        }
        fold_results.append(fold_result)
        
        fpr, tpr, _ = roc_curve(test_y_true, test_y_scores)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
    
    mean_auc = round(np.mean(outer_aucs), 4)  # Round to 4 decimals
    
    avg_alpha = round(np.mean(best_alphas), 2)  # Round to 2 decimals
    
    return {
        'selected_alpha': best_alphas,
        'outer_aucs': outer_aucs,
        'mean_outer_auc': mean_auc,
        'avg_alpha': avg_alpha,
        'fold_results': fold_results,
        'all_fpr': all_fpr,
        'all_tpr': all_tpr
    }

#-----------------------------------------------------------------------------------------------------------------------

def write_summary_tsv(summary_rows: list, filename: str = "results_summary.tsv", disease: str = "") -> None:
    """
    Write the combined fold summary to a TSV file.
    Each row represents one outer CV fold result with the algorithm used, the best parameter and the test AUROC.
    """
    import csv
    # Specify fieldnames for the TSV file
    fieldnames = ['Algorithm', 'Fold', 'BestParameter', 'Test_AUROC']
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

#-----------------------------------------------------------------------------------------------------------------------

def write_fold_details_json(summary: dict, disease: str, algorithm: str):
    """
    Write per-disease fold-level data (FPR, TPR, etc.) to a JSON file for downstream visualization.
    The JSON file name will be: Results/<disease>_<algorithm>_folds.json
    """
    import os
    import json

    # Create Results directory if it doesn't exist
    os.makedirs("Results", exist_ok=True)
    
    filename = f"Results/{disease}_{algorithm}_folds.json"
    
    # Prepare output data based on algorithm type
    if algorithm == "HeatDiffusion":
        output_data = {
            "all_fpr": summary.get("all_fpr", []),
            "all_tpr": summary.get("all_tpr", []),
            "fold_aucs": summary.get("outer_aucs", []),
            "mean_outer_auc": summary.get("mean_outer_auc", 0.0),
            "best_params": summary.get("selected_t", []),
            "avg_param": summary.get("avg_t")
        }
    else:  # RandomWalk
        output_data = {
            "all_fpr": summary.get("all_fpr", []),
            "all_tpr": summary.get("all_tpr", []),
            "fold_aucs": summary.get("outer_aucs", []),
            "mean_outer_auc": summary.get("mean_outer_auc", 0.0),
            "best_params": summary.get("selected_alpha", []),
            "avg_param": summary.get("avg_alpha")
        }
    
    # Convert numpy arrays to lists for JSON serialization
    for key, value in output_data.items():
        if isinstance(value, list):
            for i, item in enumerate(value):
                if hasattr(item, 'tolist'):  # Check if it's a numpy array
                    output_data[key][i] = item.tolist()
    
    # Write to JSON file with error handling
    try:
        with open(filename, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Successfully wrote JSON data to {filename}")
    except Exception as e:
        print(f"Error writing JSON data to {filename}: {str(e)}")
        # Try to write to current directory as fallback
        fallback_filename = f"{disease}_{algorithm}_folds.json"
        try:
            with open(fallback_filename, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Fallback: wrote JSON data to {fallback_filename}")
        except Exception as e2:
            print(f"Critical error: Could not write JSON data: {str(e2)}")

#-----------------------------------------------------------------------------------------------------------------------

def run_tests(positive_set: pd.DataFrame, negative_set: pd.DataFrame, G: nx.Graph, disease: str) -> None:
    """
    Run tests for both heat diffusion and random walk algorithms with AUROC as performance metric.
    Status updates (progress messages) are printed to the terminal.
    The AUROC scores and best parameter details are collected and written to a summary TSV file.
    """
    input_set = pd.concat([positive_set, negative_set], ignore_index=True)

    # Assign labels: 1 for positive proteins, 0 for negatives
    input_set['Label'] = 0
    input_set.loc[input_set['Protein'].isin(positive_set['Protein']), 'Label'] = 1
    input_set.loc[input_set['Protein'].isin(negative_set['Protein']), 'Label'] = 0

    input_set.drop_duplicates(subset='Protein', inplace=True)
    print(f"Running tests for {disease} with {len(input_set)} proteins."
          f" # of Positives: {len(positive_set)}"
          f" # of Negatives: {len(negative_set)}"
          f" # of Overlapping Proteins: {len(set(positive_set['Protein']).intersection(set(negative_set['Protein'])))}")
    

    # Ensure all proteins are included in the network
    missing_proteins = set(input_set['Protein'].tolist()) - set(G.nodes())
    if missing_proteins:
        G.add_nodes_from(missing_proteins)

    # Run nested CV tests for Heat Diffusion
    t_values = [round(x, 1) for x in np.arange(0.5, 4.5, 0.5)]
    hd_summary = heat_diffusion_nested_cv(input_set, G, t_values, disease)

    # Run nested CV tests for Random Walk
    alpha_values = [round(x, 2) for x in np.arange(0.5, 0.95, 0.05)]
    rw_summary = random_walk_nested_cv(input_set, G, alpha_values, disease)

    # Combine fold-level results from both algorithms
    summary_rows = hd_summary['fold_results'] + rw_summary['fold_results']

    # Add overall summary rows with mean outer AUROC and mode hyperparameter values.
    summary_rows.append({
        'Algorithm': 'Heat Diffusion',
        'Fold': 'Overall',
        'BestParameter': hd_summary['avg_t'],
        'Test_AUROC': hd_summary['mean_outer_auc']
    })
    summary_rows.append({
        'Algorithm': 'Random Walk',
        'Fold': 'Overall',
        'BestParameter': rw_summary['avg_alpha'],
        'Test_AUROC': rw_summary['mean_outer_auc']
    })

    # Write the detailed summary results to a TSV file
    file_path = f"Results/{disease}_summary.tsv"
    write_summary_tsv(summary_rows, file_path, disease)

    # Generate ROC curves
    roc_output_hd = f"Results/{disease}_heat_diffusion_roc.png"
    roc_output_rw = f"Results/{disease}_random_walk_roc.png"
    generate_disease_roc_curve(hd_summary, disease, "Heat Diffusion", "t", hd_summary['avg_t'], hd_summary['mean_outer_auc'], roc_output_hd, "blue")
    generate_disease_roc_curve(rw_summary, disease, "Random Walk", "α", rw_summary['avg_alpha'], rw_summary['mean_outer_auc'], roc_output_rw, "red")

    # Write fold details to JSON files
    write_fold_details_json(hd_summary, disease, "HeatDiffusion")
    write_fold_details_json(rw_summary, disease, "RandomWalk")

#-----------------------------------------------------------------------------------------------------------------------

def disease_test(G: nx.Graph, positive_file: str, negative_file: str, disease: str, file_type_pos: str, file_type_neg: str):
    """
    Load the disease and negative sets from file and run the tests.
    """
    # Load the disease set using appropriate loader based on file type
    if file_type_pos == 'omim':
        positive_set = load_omim_set(positive_file)
    elif file_type_pos == 'disgenet':
        positive_set = load_disgenet_set(positive_file)
    else:
        raise ValueError(f"Unsupported file type for positive set: {file_type_pos}")
    
    # Load the negative set
    if file_type_neg == 'omim':
        negative_set = load_omim_set(negative_file)
    elif file_type_neg == 'disgenet':
        negative_set = load_disgenet_set(negative_file)
    else:
        raise ValueError(f"Unsupported file type for negative set: {file_type_neg}")
    
    # Determine the minimum sample size across both sets
    min_proteins = min(len(positive_set), len(negative_set))

    # Random sample to balance the input sets
    positive_set = positive_set.sample(n=min_proteins, random_state=42)
    negative_set = negative_set.sample(n=min_proteins, random_state=42)

    # Run the tests with the provided network and disease details
    run_tests(positive_set, negative_set, G, disease=disease)

#-----------------------------------------------------------------------------------------------------------------------

def disease_worker(args):
    """
    Worker function for parallel disease testing.
    
    Args:
        args: A tuple containing (G, positive_file, negative_file, disease, file_type_pos, file_type_neg)
    
    Returns:
        Disease name and completion status
    """
    try:
        G, positive_file, negative_file, disease, file_type_pos, file_type_neg = args
        
        start_time = time.time()
        
        disease_test(G, positive_file, negative_file, disease, file_type_pos, file_type_neg)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return disease, "completed"
    except Exception as e:
        return disease, f"failed: {str(e)}"

#-----------------------------------------------------------------------------------------------------------------------

def main():
    """
    Main entry point for running network propagation tests with multiprocessing.
    Generates the network and executes all disease tests in parallel.
    """
    start_time = time.time()
    
    # Create the network once to be used by all processes
    G = create_network(min_score=400)
    
    # Define all disease tests to run
    test_cases = [
        # (G, positive_file, negative_file, disease, file_type_pos, file_type_neg)
        (G, 'Data/OMIM/Processed/epilepsy_genes.txt', 'Data/OMIM/Processed/negative_genes.txt', 'epilepsy', 'omim', 'omim'),
        (G, 'Data/OMIM/Processed/diabetes_genes.txt', 'Data/OMIM/Processed/negative_genes.txt', 'diabetes', 'omim', 'omim'),
        (G, 'Data/DisGeNET/alcoholism.tsv', 'Data/OMIM/Processed/negative_genes.txt', 'alcohol_use_disorder', 'disgenet', 'omim'),
        (G, 'Data/DisGeNET/cocaine.tsv', 'Data/OMIM/Processed/negative_genes.txt', 'cocaine_use_disorder', 'disgenet', 'omim'),
        (G, 'Data/DisGeNET/tobacco.tsv', 'Data/OMIM/Processed/negative_genes.txt', 'tobacco_use_disorder', 'disgenet', 'omim'),
        (G, 'Data/DisGeNET/depression.tsv', 'Data/OMIM/Processed/negative_genes.txt', 'depression', 'disgenet', 'omim'),
        (G, 'Data/DisGeNET/adhd.tsv', 'Data/OMIM/Processed/negative_genes.txt', 'attention_deficit_disorder', 'disgenet', 'omim'),
        (G, 'Data/DisGeNET/anxiety.tsv', 'Data/OMIM/Processed/negative_genes.txt', 'anxiety', 'disgenet', 'omim'),
        (G, 'Data/DisGeNET/schizophrenia.tsv', 'Data/OMIM/Processed/negative_genes.txt', 'schizophrenia', 'disgenet', 'omim'),
        (G, 'Data/DisGeNET/bipolar.tsv', 'Data/OMIM/Processed/negative_genes.txt', 'bipolar_disorder', 'disgenet', 'omim'),
    ]
    
    # Determine optimal number of processes (number of tests or CPU cores, whichever is smaller)
    num_processes = min(len(test_cases), mp.cpu_count())
    
    print(f"Running tests in parallel using {num_processes} processes.")
    # Create process pool and run tests in parallel
    with mp.Pool(processes=num_processes) as pool:
        pool.map(disease_worker, test_cases)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"All tests completed in {total_time:.2f} seconds.")

if __name__ == '__main__':
    main()