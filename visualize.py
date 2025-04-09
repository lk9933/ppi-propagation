"""
Module for visualizing cross-validation results of network propagation algorithms.
"""

#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, precision_recall_curve

#-----------------------------------------------------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------------------------------------------------

def generate_prc_curve(results, color, algo_name, param_name, param_value, output_path, f1_metrics=None):
    """Generate a Precision-Recall Curve for a single algorithm with specific parameters."""
    plt.figure(figsize=(10, 8))
    
    # Plot individual fold curves with transparency
    for i in range(len(results['fold_precisions'])):
        plt.plot(results['fold_recalls'][i], results['fold_precisions'][i],
                 color=color, lw=0.8, alpha=0.15)
    
    # Aggregate overall precision-recall curve from all predictions
    precision_all, recall_all, _ = precision_recall_curve(results['all_y_true'], results['all_y_pred'])
    # Sort recall and precision for smooth plotting
    sorted_idx = np.argsort(recall_all)
    recall_all = recall_all[sorted_idx]
    precision_all = precision_all[sorted_idx]
    
    # Plot the aggregated precision-recall curve
    plt.plot(recall_all, precision_all, color=color, lw=3,
             label=f'{algo_name} (Aggregated)', antialiased=True)
    
    # Optionally annotate with F1 metrics if provided
    if f1_metrics:
        threshold = f1_metrics['threshold']
        plt.annotate(f"F1: {f1_metrics['f1']:.3f}\nThreshold: {threshold:.3f}",
                     xy=(recall_all[0], precision_all[0]),
                     xytext=(recall_all[0]+0.1, precision_all[0]-0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=10)
    
    # Finalize the PRC plot
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{algo_name} PRC Curve ({param_name}={param_value})', fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    
    # Ensure directory exists and save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

#-----------------------------------------------------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------------------------------------------------

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