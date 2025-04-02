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
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve, precision_score, recall_score
from propagation import heat_diffusion, random_walk
from typing import Dict, Tuple

#-----------------------------------------------------------------------------------------------------------------------

def prepare_network(protein_scores: pd.DataFrame, min_score: int, complete: bool, num_hops: int) -> nx.Graph:
    """
    Prepares the network for propagation using the protein mappings.
    """
    # Prepare the network
    G = nx.Graph()
    if complete:
        G = create_network(min_score)
    else:
        seeds = protein_scores['Protein'].tolist()
        G = create_network_from_seeds(seeds, num_hops, min_score)

    return G

#-----------------------------------------------------------------------------------------------------------------------

def assign_labels(protein_scores: pd.DataFrame, input_file: str) -> pd.DataFrame:
    """
    Assigns labels to the protein scores based on the input file
    """
    # Load the input file
    disease_associations = pd.read_csv(input_file, sep='\t')

    # Map the gene names to proteins
    protein_names = set(disease_associations['identifier'].unique().tolist())

    # Assign labels based on the presence of proteins in the gene map
    protein_scores['Label'] = protein_scores['Protein'].apply(lambda x: 1 if x in protein_names else 0)

    return protein_scores
    
#-----------------------------------------------------------------------------------------------------------------------

def run_heat_diffusion_cv(protein_scores, G, t, node_indices):
    """Run Heat Diffusion cross-validation for a specific t value."""
    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize results collection
    fold_aucs = []
    all_fpr = []
    all_tpr = []
    all_y_true = []
    all_y_pred = []
    
    # Perform k-fold cross-validation
    for fold, (train_index, test_index) in enumerate(kf.split(protein_scores, protein_scores['Label'])):
        # Split data
        train_data = protein_scores.iloc[train_index]
        test_data = protein_scores.iloc[test_index]
        
        # Prepare initial scores
        F0 = np.zeros(G.number_of_nodes())
        for node in G.nodes():
            if node in train_data['Protein'].values:
                i = node_indices[node]
                # Assign score based off protein_scores data SCALED_ROBUST_SIGMOID
                # F0[i] = train_data[train_data['Protein'] == node]['Scaled_Robust_Sigmoid'].iloc[0]
                F0[i] = 1
        
        # Run Heat Diffusion
        F_hd = heat_diffusion(G, F0, t)
        
        # Evaluate performance
        y_true = []
        y_pred = []
        
        for protein in test_data['Protein']:
            if protein in G:
                i = node_indices[protein]
                y_true.append(test_data.loc[test_data['Protein'] == protein, 'Label'].iloc[0])
                y_pred.append(F_hd[i])
        
        if len(y_true) > 0:
            # Calculate ROC
            try:
                roc_auc = roc_auc_score(y_true, y_pred)
                fold_aucs.append(roc_auc)
                
                # Store predictions for later ROC curve generation
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
                
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                all_fpr.append(fpr)
                all_tpr.append(tpr)
            except ValueError as e:
                print(f"Warning: {e}")
    
    # Calculate average AUC
    avg_auc = np.mean(fold_aucs) if fold_aucs else 0.0
    
    return {
        'auc': avg_auc,
        'fold_aucs': fold_aucs,
        'all_fpr': all_fpr,
        'all_tpr': all_tpr,
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred
    }


def run_random_walk_cv(protein_scores, G, alpha, node_indices):
    """Run Random Walk cross-validation for a specific alpha value."""
    # Similar structure to run_heat_diffusion_cv but for Random Walk
    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize results collection
    fold_aucs = []
    all_fpr = []
    all_tpr = []
    all_y_true = []
    all_y_pred = []
    
    # Perform k-fold cross-validation
    for fold, (train_index, test_index) in enumerate(kf.split(protein_scores, protein_scores['Label'])):
        # Split data
        train_data = protein_scores.iloc[train_index]
        test_data = protein_scores.iloc[test_index]
        
        # Prepare initial scores
        F0 = np.zeros(G.number_of_nodes())
        for node in G.nodes():
            if node in train_data['Protein'].values:
                i = node_indices[node]
                # F0[i] = train_data[train_data['Protein'] == node]['Scaled_Robust_Sigmoid'].iloc[0]
                F0[i] = 1
        
        # Run Random Walk
        F_rw = random_walk(G, F0, alpha)
        
        # Evaluate performance
        y_true = []
        y_pred = []
        
        for protein in test_data['Protein']:
            if protein in G:
                i = node_indices[protein]
                y_true.append(test_data.loc[test_data['Protein'] == protein, 'Label'].iloc[0])
                y_pred.append(F_rw[i])
        
        if len(y_true) > 0:
            # Calculate ROC
            try:
                roc_auc = roc_auc_score(y_true, y_pred)
                fold_aucs.append(roc_auc)
                
                # Store predictions for later ROC curve generation
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
                
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                all_fpr.append(fpr)
                all_tpr.append(tpr)
            except ValueError as e:
                print(f"Warning: {e}")
    
    # Calculate average AUC
    avg_auc = np.mean(fold_aucs) if fold_aucs else 0.0
    
    return {
        'auc': avg_auc,
        'fold_aucs': fold_aucs,
        'all_fpr': all_fpr,
        'all_tpr': all_tpr,
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred
    }


def generate_roc_curve(results, color, algo_name, param_name, param_value, output_path, f1_metrics=None):
    """Generate a ROC curve for a single algorithm with specific parameters."""
    plt.figure(figsize=(10, 8))
    
    # Plot individual fold curves with transparency
    for i in range(len(results['all_fpr'])):
        plt.plot(results['all_fpr'][i], results['all_tpr'][i], color=color, lw=0.8, alpha=0.15)
    
    # Create smooth interpolation for average curve
    mean_fpr, mean_tpr = interpolate_roc_curve(results)
    
    mean_auc = results['auc']
    std_auc = np.std(results['fold_aucs'])
    
    # Plot the final smooth curve
    plt.plot(mean_fpr, mean_tpr, color=color, lw=3, 
            label=f'{algo_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})', antialiased=True)
    
    # If F1 metrics are provided, mark the optimal threshold point
    if f1_metrics:
        # We need to find the point on the ROC curve closest to the F1 threshold
        threshold = f1_metrics['threshold']
        
        # Calculate ROC curve for all data points to find the threshold point
        fpr_all, tpr_all, thresholds_all = roc_curve(results['all_y_true'], results['all_y_pred'])
        
        # Find the index closest to our F1 threshold
        idx = np.argmin(np.abs(thresholds_all - threshold)) if len(thresholds_all) > 0 else 0
        
        if idx < len(fpr_all):
            # Mark the threshold point on the curve
            plt.plot(fpr_all[idx], tpr_all[idx], 'o', markersize=10, 
                    color='black', markerfacecolor=color)
            
            # Add an annotation for the F1 score
            plt.annotate(f"F1: {f1_metrics['f1']:.3f}\nThreshold: {threshold:.3f}",
                        xy=(fpr_all[idx], tpr_all[idx]),
                        xytext=(fpr_all[idx]+0.15, tpr_all[idx]-0.15),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                        fontsize=10)
    
    # Finalize plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{algo_name} ROC Curve ({param_name}={param_value})', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_comparison_roc(hd_results, rw_results, best_t, best_alpha, output_path):
    """Generate a ROC curve comparing the best Heat Diffusion and Random Walk results."""
    plt.figure(figsize=(10, 8))
    
    # Process and plot data for both algorithms
    for algo, results, color, param_name, param_value in [
        ('Heat Diffusion', hd_results, 'blue', 't', best_t), 
        ('Random Walk', rw_results, 'red', 'α', best_alpha)
    ]:
        # Plot individual fold curves with transparency
        for i in range(len(results['all_fpr'])):
            plt.plot(results['all_fpr'][i], results['all_tpr'][i], color=color, lw=0.8, alpha=0.15)
        
        # Create smooth interpolation for average curve
        mean_fpr, mean_tpr = interpolate_roc_curve(results)
        
        mean_auc = results['auc']
        std_auc = np.std(results['fold_aucs'])
        
        # Plot the final smooth curve
        label = f'{algo} ({param_name}={param_value}, AUC = {mean_auc:.3f} ± {std_auc:.3f})'
        plt.plot(mean_fpr, mean_tpr, color=color, lw=3, label=label, antialiased=True)
    
    # Finalize plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Comparison of Best Propagation Algorithms', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def interpolate_roc_curve(results):
    """Interpolate ROC curve data points to create a smooth curve."""
    mean_fpr = np.linspace(0, 1, 2000)
    mean_tpr = np.zeros_like(mean_fpr)
    
    # Interpolate each fold's curve
    for i in range(len(results['all_fpr'])):
        if len(results['all_fpr'][i]) > 3:
            try:
                # Use spline interpolation for smoothness
                x_temp = results['all_fpr'][i]
                y_temp = results['all_tpr'][i]
                
                if len(x_temp) != len(np.unique(x_temp)):
                    # Linear interpolation for duplicates
                    interp_tpr = np.interp(mean_fpr, x_temp, y_temp)
                else:
                    # Spline interpolation
                    spl = make_interp_spline(x_temp, y_temp, k=3)
                    interp_tpr = spl(mean_fpr)
                    interp_tpr = np.clip(interp_tpr, 0, 1)
            except:
                interp_tpr = np.interp(mean_fpr, x_temp, y_temp)
        else:
            interp_tpr = np.interp(mean_fpr, results['all_fpr'][i], results['all_tpr'][i])
        
        mean_tpr += interp_tpr
    
    mean_tpr /= len(results['all_fpr'])
    
    # Ensure monotonicity
    for i in range(1, len(mean_tpr)):
        mean_tpr[i] = max(mean_tpr[i], mean_tpr[i-1])
    
    # Apply smoothing
    window_size = 10
    if len(mean_tpr) > window_size*3:
        smoothed_tpr = np.zeros_like(mean_tpr)
        for i in range(len(mean_tpr)):
            start = max(0, i - window_size//2)
            end = min(len(mean_tpr), i + window_size//2 + 1)
            smoothed_tpr[i] = np.mean(mean_tpr[start:end])
        smoothed_tpr[0] = mean_tpr[0]
        smoothed_tpr[-1] = mean_tpr[-1]
        mean_tpr = smoothed_tpr
    
    return mean_fpr, mean_tpr


def find_optimal_threshold(y_true, y_pred):
    """Find optimal threshold that maximizes F1 score."""
    # Initialize variables
    thresholds = np.arange(0.0, 1.00, 0.05)
    best_f1 = 0
    best_threshold = 0.0

    # Iterate through thresholds to find the best F1 score
    for threshold in thresholds:
        y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Return the best threshold     
    return best_threshold

def calculate_f1_metrics(results):
    """Calculate F1 score and related metrics using optimal threshold."""
    y_true = results['all_y_true']
    y_pred = results['all_y_pred']
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_true, y_pred)
    
    # Make binary predictions using optimal threshold
    y_pred_binary = (np.array(y_pred) >= optimal_threshold).astype(int)
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'threshold': optimal_threshold
    }


def evaluate_heat_diffusion(protein_scores, G, node_indices, t_values, output_file):
    """Evaluate heat diffusion algorithm with different t values."""
    print("Running Heat Diffusion cross-validation...")
    best_hd_auc = 0
    best_t = 0
    hd_results_cache = {}
    
    output_file.write("1. HEAT DIFFUSION EVALUATION\n")
    output_file.write("============================\n\n")
    output_file.write("Evaluation Metrics:\n\n")
    output_file.write(f"{'t':<5} {'AUC':<8} {'F1':<8} {'Precision':<10} {'Recall':<10} {'Threshold':<10}\n")
    output_file.write("-" * 55 + "\n")
    
    # Evaluate Heat Diffusion for different t values
    for t in t_values:
        print(f"Computing Heat Diffusion results for t={t}...")
        hd_results = run_heat_diffusion_cv(protein_scores, G, t, node_indices)
        hd_results_cache[t] = hd_results
        hd_auc = hd_results['auc']
        
        # Calculate F1 and related metrics
        f1_metrics = calculate_f1_metrics(hd_results)
        
        # Generate individual ROC curve for this parameter
        output_path = f"Results/sud/ROC_Curves/heat_diffusion/t{t}/heat_diffusion_roc.png"
        generate_roc_curve(hd_results, 'blue', 'Heat Diffusion', 't', t, output_path, f1_metrics)
        
        # Track best parameter
        if hd_auc > best_hd_auc:
            best_hd_auc = hd_auc
            best_t = t
        
        # Write results to file
        output_file.write(f"{t:<5} {hd_auc:.4f}   {f1_metrics['f1']:.4f}   {f1_metrics['precision']:.4f}      {f1_metrics['recall']:.4f}      {f1_metrics['threshold']:.4f}\n")
    
    # Write best Heat Diffusion parameter
    output_file.write(f"\nBest Heat Diffusion parameter: t={best_t} (AUC={best_hd_auc:.4f})\n\n")
    
    return best_t, hd_results_cache[best_t]


def evaluate_random_walk(protein_scores, G, node_indices, alpha_values, output_file):
    """Evaluate random walk algorithm with different alpha values."""
    print("Running Random Walk cross-validation...")
    best_rw_auc = 0
    best_alpha = 0
    rw_results_cache = {}
    
    output_file.write("\n\n2. RANDOM WALK EVALUATION\n")
    output_file.write("============================\n\n")
    output_file.write("Evaluation Metrics:\n\n")
    output_file.write(f"{'t':<5} {'AUC':<8} {'F1':<8} {'Precision':<10} {'Recall':<10} {'Threshold':<10}\n")
    output_file.write("-" * 55 + "\n")
    
    # Evaluate Random Walk for different alpha values
    for alpha in alpha_values:
        print(f"Computing Random Walk results for alpha={alpha}...")
        rw_results = run_random_walk_cv(protein_scores, G, alpha, node_indices)
        rw_results_cache[alpha] = rw_results
        rw_auc = rw_results['auc']

        # Calculate F1 and related metrics
        f1_metrics = calculate_f1_metrics(rw_results)
        
        # Generate individual ROC curve for this parameter
        output_path = f"Results/sud/ROC_Curves/random_walk/alpha{alpha}/random_walk_roc.png"
        generate_roc_curve(rw_results, 'red', 'Random Walk', 'α', alpha, output_path, f1_metrics)
        
        # Track best parameter
        if rw_auc > best_rw_auc:
            best_rw_auc = rw_auc
            best_alpha = alpha
        
        # Write results to file
        output_file.write(f"{alpha:<5} {rw_auc:.4f}   {f1_metrics['f1']:.4f}   {f1_metrics['precision']:.4f}      {f1_metrics['recall']:.4f}      {f1_metrics['threshold']:.4f}\n")
    
    # Write best Random Walk parameter
    output_file.write(f"\nBest Random Walk parameter: alpha={best_alpha} (AUC={best_rw_auc:.4f})\n")
    
    return best_alpha, rw_results_cache[best_alpha]


def main():
    # Load protein scores
    protein_scores = pd.read_csv('Data/Processed/sud_complete_processed.tsv', sep='\t')

    # Assign labels based on disease associations
    protein_scores = assign_labels(protein_scores, 'Data/STRING/sud_proteins.tsv')

    # Prepare the network
    G = prepare_network(protein_scores, min_score=400, complete=False, num_hops=2)
    
    # Parameters to test
    t_values = [2, 4, 6, 8]
    alpha_values = [0.2, 0.4, 0.6, 0.8]
    
    # Create results directory
    os.makedirs("Results/sud", exist_ok=True)
    
    # Pre-compute the node indices for each protein
    node_indices = {}
    for protein in protein_scores['Protein']:
        if protein in G:
            node_indices[protein] = list(G.nodes()).index(protein)

    # Open a file to write summary results
    with open("Results/sud/summary_results.txt", "w") as f:
        f.write("Network Propagation Algorithm Evaluation\n")
        f.write("=======================================\n\n")
        f.write(f"Dataset: sud_processed.tsv\n")
        f.write(f"Network size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n\n")
        
        # Evaluate Heat Diffusion
        best_t, best_hd_results = evaluate_heat_diffusion(protein_scores, G, node_indices, t_values, f)
        
        # Evaluate Random Walk
        best_alpha, best_rw_results = evaluate_random_walk(protein_scores, G, node_indices, alpha_values, f)
        
        # Generate comparison ROC curve for the best parameters
        output_path = f"Results/sud/ROC_Curves/comparison/best_algorithms_comparison.png"
        generate_comparison_roc(best_hd_results, best_rw_results, best_t, best_alpha, output_path)
        
        # Compare the two methods
        f.write("\n\n3. COMPARISON OF METHODS\n")
        f.write("========================\n\n")
        
        # Calculate F1 metrics for best models
        hd_f1_metrics = calculate_f1_metrics(best_hd_results)
        rw_f1_metrics = calculate_f1_metrics(best_rw_results)
        
        f.write(f"Metric          Heat Diffusion (t={best_t})    Random Walk (alpha={best_alpha})\n")
        f.write(f"---------------------------------------------------------------------------\n")
        f.write(f"AUC             {best_hd_results['auc']:.4f}                {best_rw_results['auc']:.4f}\n")
        f.write(f"F1 Score        {hd_f1_metrics['f1']:.4f}                {rw_f1_metrics['f1']:.4f}\n")
        f.write(f"Precision       {hd_f1_metrics['precision']:.4f}                {rw_f1_metrics['precision']:.4f}\n")
        f.write(f"Recall          {hd_f1_metrics['recall']:.4f}                {rw_f1_metrics['recall']:.4f}\n")
        f.write(f"Best Threshold  {hd_f1_metrics['threshold']:.4f}                {rw_f1_metrics['threshold']:.4f}\n\n")
        
        # Determine best method based on AUC
        auc_difference = best_hd_results['auc'] - best_rw_results['auc']
        f1_difference = hd_f1_metrics['f1'] - rw_f1_metrics['f1']
        
        f.write(f"AUC difference: {abs(auc_difference):.4f} favoring {('Heat Diffusion' if auc_difference > 0 else 'Random Walk')}\n")
        f.write(f"F1 difference: {abs(f1_difference):.4f} favoring {('Heat Diffusion' if f1_difference > 0 else 'Random Walk')}\n")
    
    print(f"\nEvaluation complete. Results saved to Results/sud/summary_results.txt")
    print(f"ROC curves saved to Results/sud/ROC_Curves/")


if __name__ == "__main__":
    main()

