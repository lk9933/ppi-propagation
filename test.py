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

def write_summary(file_path: str, algo: str, param_name: str, param_value, 
                  mean_auc: float, mean_prc_auc: float, mean_opt_threshold: float, 
                  mean_opt_f1: float, mean_precision: float, mean_recall: float):
    """Append formatted summary results to the given file with aligned columns."""
    # Shorten algorithm names
    if algo.lower() == "heat diffusion":
        algo_short = "HD"
    elif algo.lower() == "random walk":
        algo_short = "RWR"
    else:
        algo_short = algo

    with open(file_path, "a") as f:
        f.write(f"{algo_short:<5} {param_name + '=' + str(param_value):<10} "
                f"{mean_auc:<10.4f} {mean_prc_auc:<10.4f} {mean_opt_threshold:<15.4f} "
                f"{mean_opt_f1:<10.4f} {mean_precision:<10.4f} {mean_recall:<10.4f}\n")

#-----------------------------------------------------------------------------------------------------------------------

def heat_diffusion_test(protein_set: pd.DataFrame, G: nx.Graph, t: float, summary_file: str, disease: str) -> dict:
    """
    Test the heat diffusion algorithm on a set of proteins and visualize ROC/PRC curves.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proteins = protein_set['Protein'].values
    labels = protein_set['Label'].values
    node_indices = {node: idx for idx, node in enumerate(G.nodes())}

    auc_scores = []
    prc_auc_scores = []
    opt_thresholds = []
    opt_f1_scores = []
    precisions = []
    recalls = []
    # Accumulators for visualization
    all_fpr = []
    all_tpr = []
    fold_precisions = []
    fold_recalls = []
    all_y_true = []
    all_y_pred = []

    for train_index, test_index in skf.split(proteins, labels):
        F0 = np.zeros(G.number_of_nodes())
        for i in train_index:
            if proteins[i] in G.nodes():
                F0[node_indices[proteins[i]]] = labels[i]
        
        F_hd = heat_diffusion(G, F0, t)
        
        y_true = labels[test_index]
        y_scores = [F_hd[node_indices[proteins[i]]] if proteins[i] in node_indices else 0 
                    for i in test_index]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_scores)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        
        fold_prec, fold_rec, _ = precision_recall_curve(y_true, y_scores)
        fold_precisions.append(fold_prec)
        fold_recalls.append(fold_rec)
        
        metrics = calculate_auc_metrics(y_true, y_scores)
        auc_scores.append(metrics['roc_auc'])
        prc_auc_scores.append(metrics['pr_auc'])
        opt_thresholds.append(metrics['optimal_threshold'])
        opt_f1_scores.append(metrics['optimal_f1'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
    
    # Calculate mean metrics
    mean_auc = np.mean(auc_scores)
    mean_prc_auc = np.mean(prc_auc_scores)
    mean_opt_threshold = np.mean(opt_thresholds)
    mean_opt_f1 = np.mean(opt_f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    
    # Build results dictionary for visualization
    results_dict = {
        'all_fpr': all_fpr,
        'all_tpr': all_tpr,
        'fold_precisions': fold_precisions,
        'fold_recalls': fold_recalls,
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred,
        'auc': mean_auc,
        'fold_aucs': auc_scores
    }
    
    # Define output paths for the plots
    roc_output = f"Results/{disease}/ROC_Curves/heat_diffusion/t{t}/heat_diffusion_roc.png"
    prc_output = f"Results/{disease}/PRC_Curves/heat_diffusion/t{t}/precision_recall.png"
    
    # Visualize the ROC and PRC curves
    generate_roc_curve(results_dict, color='blue', algo_name="Heat Diffusion",
                       param_name="t", param_value=t, output_path=roc_output)
    generate_prc_curve(results_dict, color='blue', algo_name="Heat Diffusion",
                       param_name="t", param_value=t, output_path=prc_output)

    # Write summary results
    write_summary(summary_file, "Heat Diffusion", "t", t, mean_auc, mean_prc_auc, 
                  mean_opt_threshold, mean_opt_f1, mean_precision, mean_recall)
    
    # Return a summary dictionary for optimal parameter evaluation
    return {'t': t, 'roc_auc': mean_auc, 'prc_auc': mean_prc_auc}

#-----------------------------------------------------------------------------------------------------------------------

def random_walk_test(protein_set: pd.DataFrame, G: nx.Graph, alpha: float, summary_file: str, disease: str) -> dict:
    """
    Test the random walk algorithm on a set of proteins and visualize ROC/PRC curves.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proteins = protein_set['Protein'].values
    labels = protein_set['Label'].values
    node_indices = {node: i for i, node in enumerate(G.nodes())}

    auc_scores = []
    prc_auc_scores = []
    opt_thresholds = []
    opt_f1_scores = []
    precisions = []
    recalls = []
    # Accumulators for visualization
    all_fpr = []
    all_tpr = []
    fold_precisions = []
    fold_recalls = []
    all_y_true = []
    all_y_pred = []

    for train_index, test_index in skf.split(proteins, labels):
        F0 = np.zeros(G.number_of_nodes())
        for i in train_index:
            if proteins[i] in G.nodes():
                F0[node_indices[proteins[i]]] = labels[i]
        
        F_rw = random_walk(G, F0, alpha)
        
        y_true = labels[test_index]
        y_scores = [F_rw[node_indices[proteins[i]]] if proteins[i] in node_indices else 0 
                    for i in test_index]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_scores)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        
        fold_prec, fold_rec, _ = precision_recall_curve(y_true, y_scores)
        fold_precisions.append(fold_prec)
        fold_recalls.append(fold_rec)
        
        metrics = calculate_auc_metrics(y_true, y_scores)
        auc_scores.append(metrics['roc_auc'])
        prc_auc_scores.append(metrics['pr_auc'])
        opt_thresholds.append(metrics['optimal_threshold'])
        opt_f1_scores.append(metrics['optimal_f1'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
    
    # Calculate mean metrics
    mean_auc = np.mean(auc_scores)
    mean_prc_auc = np.mean(prc_auc_scores)
    mean_opt_threshold = np.mean(opt_thresholds)
    mean_opt_f1 = np.mean(opt_f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    
    results_dict = {
        'all_fpr': all_fpr,
        'all_tpr': all_tpr,
        'fold_precisions': fold_precisions,
        'fold_recalls': fold_recalls,
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred,
        'auc': mean_auc,
        'fold_aucs': auc_scores
    }
    
    roc_output = f"Results/{disease}/ROC_Curves/random_walk/alpha{alpha}/random_walk_roc.png"
    prc_output = f"Results/{disease}/PRC_Curves/random_walk/alpha{alpha}/precision_recall.png"
    
    generate_roc_curve(results_dict, color='red', algo_name="Random Walk",
                       param_name="α", param_value=alpha, output_path=roc_output)
    generate_prc_curve(results_dict, color='red', algo_name="Random Walk",
                       param_name="α", param_value=alpha, output_path=prc_output)

    # Write summary results
    write_summary(summary_file, "Random Walk", "α", alpha, mean_auc, mean_prc_auc, 
                  mean_opt_threshold, mean_opt_f1, mean_precision, mean_recall)
    
    # Return a summary dictionary for optimal parameter evaluation
    return {'alpha': alpha, 'roc_auc': mean_auc, 'prc_auc': mean_prc_auc}

#-----------------------------------------------------------------------------------------------------------------------

def calculate_auc_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> dict:
    """
    Calculate various AUC-related metrics using proper methods.
    """
    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_true, y_scores)
    
    # Calculate PR curve and PR-AUC using scikit-learn
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Calculate F1 at optimal threshold
    f1_scores = [f1_score(y_true, y_scores >= threshold) for threshold in thresholds]
    optimal_idx = np.argmax(f1_scores)
    
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'optimal_threshold': thresholds[optimal_idx] if len(thresholds) > 0 else 0.5,
        'optimal_f1': f1_scores[optimal_idx] if len(f1_scores) > 0 else 0.0,
        'precision': precision_score(y_true, y_scores >= thresholds[optimal_idx]) if len(thresholds) > 0 else 0.0,
        'recall': recall_score(y_true, y_scores >= thresholds[optimal_idx]) if len(thresholds) > 0 else 0.0
    }
    
    return metrics

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

    # Return the list as a dataframe
    return pd.DataFrame(proteins_list, columns=['Protein'])

#-----------------------------------------------------------------------------------------------------------------------

def run_tests(positive_set: pd.DataFrame, negative_set: pd.DataFrame, G: nx.Graph, disease: str) -> None:
    """
    Run tests for heat diffusion and random walk algorithms.
    """
    # Load the data sets
    print("Loading input sets...")
    input = pd.concat([positive_set, negative_set], ignore_index=True)

    # Label positive set genes as 1 and negative set genes as 0
    input['Label'] = 0
    input.loc[input['Protein'].isin(positive_set['Protein']), 'Label'] = 1
    input.loc[input['Protein'].isin(negative_set['Protein']), 'Label'] = 0

    # Remove duplicates
    input.drop_duplicates(subset='Protein', inplace=True)

    # Ensure every protein in the OMIM set is present in G
    missing_proteins = set(input['Protein'].tolist()) - set(G.nodes())
    if missing_proteins:
        print(f"Adding {len(missing_proteins)} missing proteins to the network.")
        G.add_nodes_from(missing_proteins)

    # Create the results summary file with header
    summary_file = f"Results/{disease}/results_summary.txt"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w") as f:
        f.write(f"{'Alg':<5} {'Param':<10} {'ROC_AUC':<10} {'PRC_AUC':<10} "
                f"{'Opt_Threshold':<15} {'Opt_F1':<10} {'Precision':<10} {'Recall':<10}\n")
        f.write("-" * 80 + "\n")

    # Collect summaries for Heat Diffusion
    hd_results = []
    print("Testing Heat Diffusion...")
    t_values = [1, 2, 4, 8, 16]
    for t in t_values:
        res = heat_diffusion_test(input, G, t=t, summary_file=summary_file, disease=disease)
        hd_results.append(res)

    # Collect summaries for Random Walk
    rw_results = []
    print("Testing Random Walk...")
    alpha_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for alpha in alpha_values:
        res = random_walk_test(input, G, alpha=alpha, summary_file=summary_file, disease=disease)
        rw_results.append(res)

    # Identify the optimal parameters for each algorithm
    best_hd_roc = max(hd_results, key=lambda d: d['roc_auc'])
    best_hd_prc = max(hd_results, key=lambda d: d['prc_auc'])
    best_rw_roc = max(rw_results, key=lambda d: d['roc_auc'])
    best_rw_prc = max(rw_results, key=lambda d: d['prc_auc'])

    # Append the final summary comparison to the summary file
    with open(summary_file, "a") as f:
        f.write("\nOptimal Parameters Summary:\n")
        f.write("---------- Heat Diffusion ----------\n")
        f.write(f"Best ROC AUC: t = {best_hd_roc['t']} (ROC_AUC = {best_hd_roc['roc_auc']:.4f})\n")
        f.write(f"Best PRC AUC: t = {best_hd_prc['t']} (PRC_AUC = {best_hd_prc['prc_auc']:.4f})\n")
        f.write("---------- Random Walk ----------\n")
        f.write(f"Best ROC AUC: α = {best_rw_roc['alpha']} (ROC_AUC = {best_rw_roc['roc_auc']:.4f})\n")
        f.write(f"Best PRC AUC: α = {best_rw_prc['alpha']} (PRC_AUC = {best_rw_prc['prc_auc']:.4f})\n")

#-----------------------------------------------------------------------------------------------------------------------
"""
def main():
    # Load the OMIM set
    epilepsy_set = load_omim_set('Data/OMIM/Processed/epilepsy_genes.txt')

    # Randomly sample 200 epilepsy genes
    positive_set = epilepsy_set.sample(n=200, random_state=42)

    # Load the negative set
    diabetes_set = load_omim_set('Data/OMIM/Processed/diabetes_genes.txt')

    # Randomly sample 200 diabetes genes
    negative_set = diabetes_set.sample(n=200, random_state=42)

    # Create the network
    protein_set = pd.concat([positive_set, negative_set], ignore_index=True)
    seeds = protein_set['Protein'].tolist()
    G = create_network_from_seeds(seeds, min_score=400, num_hops=2)

    # Run tests
    run_tests(positive_set, negative_set, G, disease="epilepsy")
"""

def main():
    # Load the OMIM set
    epilepsy_set = load_omim_set('Data/OMIM/Processed/epilepsy_genes.txt')

    # Randomly sample 200 epilepsy genes
    negative_set = epilepsy_set.sample(n=200, random_state=41)

    # Load the negative set
    diabetes_set = load_omim_set('Data/OMIM/Processed/diabetes_genes.txt')

    # Randomly sample 200 diabetes genes
    positive_set = diabetes_set.sample(n=200, random_state=41)

    # Create the network
    protein_set = pd.concat([positive_set, negative_set], ignore_index=True)
    seeds = protein_set['Protein'].tolist()
    G = create_network_from_seeds(seeds, min_score=400, num_hops=2)

    # Run tests
    run_tests(positive_set, negative_set, G, disease="diabetes")

if __name__ == '__main__':
    main()