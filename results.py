"""
Module to aggregate and visualize results.

Author: Luke Krongard
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_results(results_dir="Results"):
    """
    Aggregate all disease summary files (e.g., *_summary.tsv) into a single DataFrame.
    Assumes each file contains rows with 'Algorithm', 'Test_AUROC', 'BestParameter', 'Fold', etc.
    """
    all_data = []
    for file in os.listdir(results_dir):
        if file.endswith("_summary.tsv"):
            path = os.path.join(results_dir, file)
            disease_name = file.replace("_summary.tsv", "")
            df = pd.read_csv(path, sep="\t")
            df["Disease"] = disease_name
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def load_fold_jsons(results_dir="Results"):
    """
    Loads fold-level ROC data from JSON files (written by test.py).
    Creates a dict keyed by (disease, algorithm), containing FPR/TPR, AUC, and parameters.
    """
    fold_data = {}
    for file in os.listdir(results_dir):
        if file.endswith("_folds.json"):
            path = os.path.join(results_dir, file)
            # Example file: "epilepsy_HeatDiffusion_folds.json"
            with open(path, "r") as f:
                data = json.load(f)
            # Parse the disease name & algorithm
            base = file.replace("_folds.json", "")
            # e.g., "epilepsy_HeatDiffusion"
            parts = base.split("_")
            # The last part is the algorithm, everything else is disease
            algo = parts[-1]
            disease = "_".join(parts[:-1])
            fold_data[(disease, algo)] = data
    return fold_data

# filepath: /Users/lukekrongard/Documents/Spring 2025 IW/results.py
def create_summary_table(results_df):
    """
    For each disease and algorithm (Fold == "Overall"), show BestParameter and mean AUROC.
    Then append a row "All diseases" for each algorithm, indicating the overall mean parameter 
    (across all diseases) and overall mean AUROC. Finally, bold the algorithm with the best 
    mean AUROC (on the "All diseases" row).
    """
    # Filter only the final fold rows
    df_overall = results_df[results_df["Fold"] == "Overall"].copy()
    # Restrict to desired columns
    df_overall = df_overall[["Disease", "Algorithm", "BestParameter", "Test_AUROC"]]

    # Build "All diseases" rows by grouping across diseases
    agg_df = df_overall.groupby("Algorithm", as_index=False).agg({
        "BestParameter": "mean",
        "Test_AUROC": "mean"
    })
    agg_df["Disease"] = "All diseases"
    # Round the aggregates
    agg_df["BestParameter"] = agg_df["BestParameter"].round(2)
    agg_df["Test_AUROC"] = agg_df["Test_AUROC"].round(4)

    # Concatenate disease-level rows with "All diseases"
    final_df = pd.concat([df_overall, agg_df], ignore_index=True)

    # Determine which algorithm has the highest All-diseases AUROC
    best_algo = agg_df.loc[agg_df["Test_AUROC"].idxmax(), "Algorithm"]

    # Apply conditional formatting so that we bold only the best algo row in "All diseases"
    def highlight_best(row):
        if row["Disease"] == "All diseases" and row["Algorithm"] == best_algo:
            return ["font-weight: bold"] * len(row)
        return ["" for _ in row]

    # Create a Styler
    styled = (
        final_df.style
        .apply(highlight_best, axis=1)         # highlight the single best row
        .background_gradient(
            subset=["Test_AUROC"],
            cmap="YlGnBu"
        )
    )
    return styled

def create_performance_comparison(results_df):
    """
    Draw box plots and bar charts for performance (AUROC) across folds.
    """
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=results_df, x="Algorithm", y="Test_AUROC")
    plt.title("Performance Distribution (AUROC) by Algorithm")
    plt.tight_layout()
    plt.show()

    # Bar chart for overall scores
    overall = results_df[results_df["Fold"] == "Overall"]
    plt.figure(figsize=(8, 5))
    sns.barplot(data=overall, x="Algorithm", y="Test_AUROC", capsize=.2)
    plt.title("Overall AUROC Scores")
    plt.tight_layout()
    plt.show()

def create_hyperparameter_analysis(results_df):
    """
    Plot AUROC vs. parameter values (Heat Diffusion t, Random Walk α) to show sensitivity.
    Optionally, create a heatmap if multiple parameter combos exist in the data.
    """
    per_fold = results_df[results_df["Fold"] != "Overall"]
    if per_fold.empty:
        print("No per-fold data found for hyperparameter analysis.")
        return

    plt.figure(figsize=(8, 5))
    for algo in per_fold["Algorithm"].unique():
        subset = per_fold[per_fold["Algorithm"] == algo].sort_values("BestParameter")
        plt.plot(subset["BestParameter"], subset["Test_AUROC"], marker="o", label=algo)
    plt.legend()
    plt.title("AUROC vs. Parameter Values")
    plt.xlabel("Parameter (t or α)")
    plt.ylabel("AUROC")
    plt.tight_layout()
    plt.show()

    # Example heatmap (requires multiple parameter combos)
    # pivot_heat = per_fold.pivot_table(
    #     index="BestParameter", columns="Algorithm", values="Test_AUROC", aggfunc="mean"
    # )
    # sns.heatmap(pivot_heat, annot=True, cmap="YlOrBr")
    # plt.title("Parameter Performance Heat Map")
    # plt.tight_layout()
    # plt.show()

def create_roc_visualization(fold_data):
    """
    Overlaid ROC curves for each disease, with each algorithm in a different color.
    Uses data from the JSON fold files. Each fold is shown as a thin line, and the average as thick.
    """
    for (disease, algo), data in fold_data.items():
        if not data.get("all_fpr") or not data.get("all_tpr"):
            continue
        plt.figure(figsize=(8, 6))
        # Plot each fold
        for fpr, tpr in zip(data["all_fpr"], data["all_tpr"]):
            plt.plot(fpr, tpr, lw=0.8, alpha=0.2, label=None, color="blue")

        # Plot average curve
        mean_auc = data.get("mean_outer_auc", 0.0)
        # Simple average interpolation
        mean_fpr = np.linspace(0, 1, 200)
        sum_tpr = np.zeros_like(mean_fpr)
        for fpr, tpr in zip(data["all_fpr"], data["all_tpr"]):
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            sum_tpr += interp_tpr
        mean_tpr = sum_tpr / len(data["all_fpr"])
        label_str = f"{algo} (Mean AUROC={mean_auc:.3f})"
        plt.plot(mean_fpr, mean_tpr, lw=2, label=label_str, color="blue")

        plt.plot([0, 1], [0, 1], "k--", lw=1.5)
        disease_name = disease.replace("_", " ").title()
        plt.title(f"ROC for {disease_name} - {algo}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

def create_disease_specific_results(results_df):
    """
    Create grouped bar charts to compare algorithm performance across multiple disorders.
    Only includes 'Overall' folds in the figure.
    """
    subset = results_df[results_df["Fold"] == "Overall"]
    if subset.empty:
        print("No overall results found for disease-specific comparisons.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(data=subset, x="Disease", y="Test_AUROC", hue="Algorithm")
    plt.xticks(rotation=45, ha="right")
    plt.title("Disease-Specific Overall AUROC Comparison")
    plt.tight_layout()
    plt.show()

def main():
    # 1) Load summary-level results
    results_df = load_all_results("Results")
    if results_df.empty:
        print("No result files found. Run tests first.")
        return

    # 2) Load fold-level ROC data from JSON
    fold_data = load_fold_jsons("Results")

    # Create and print the summary table
    summary_table = create_summary_table(results_df)
    print("Summary Table (Console View):")
    print(summary_table.data.to_string())  # console output

    # export summary table as a tsv
    summary_table.data.to_csv("summary_table.tsv", sep="\t", index=False)

    # 1. Summary Table with Visual Elements
    # (For a Jupyter notebook, you can display 'summary_table' directly.)

    # # 2. Performance comparison
    # create_performance_comparison(results_df)

    # # 3. Hyperparameter analysis
    # create_hyperparameter_analysis(results_df)

    # # 4. ROC Visualization
    # # Overlaid for each disease–algorithm, from the new JSON fold data
    # create_roc_visualization(fold_data)

    # # 5. Disease-specific results
    # create_disease_specific_results(results_df)

if __name__ == "__main__":
    main()