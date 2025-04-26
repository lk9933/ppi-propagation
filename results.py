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
import numpy as np

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
            with open(path, "r") as f:
                data = json.load(f)
            # Parse the disease name & algorithm
            base = file.replace("_folds.json", "")
            parts = base.split("_")
            # normalize algorithm names to match summary labels
            algo_raw = parts[-1]
            name_map = {"HeatDiffusion": "Heat Diffusion", "RandomWalk": "Random Walk"}
            algo = name_map.get(algo_raw, algo_raw)
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

def plot_roc_comparison(fold_data,
                        algorithms=None,
                        fpr_grid=None):
    import matplotlib.pyplot as plt

    # determine available algorithm names in fold_data
    available = sorted({a for (_, a) in fold_data.keys()})
    # filter or derive algorithms list
    if algorithms:
        selected = [algo for algo in algorithms if algo in available]
        missing = set(algorithms) - set(selected)
        if missing:
            print(f"Warning: requested algorithms {missing} not found in fold_data; available: {available}")
        algorithms = selected
    else:
        algorithms = available

    if fpr_grid is None:
        fpr_grid = np.linspace(0, 1, 200)
    # derive algorithm list if not provided based on fold_data keys
    # ensure final algorithm list
    algorithms = sorted(algorithms)

    # assign one distinct color per algorithm
    palette = plt.get_cmap("tab10")
    colors = {algo: palette(i) for i, algo in enumerate(algorithms)}

    # create figure similar to visualize.py
    plt.figure(figsize=(10, 8))

    for algo in algorithms:
        # collect interpolated TPRs and per-disease AUCs for this algo
        all_tprs = []
        aucs = []
        for (disease, a), data in fold_data.items():
            if a != algo:
                continue
            # collect mean AUC per disease
            if "mean_outer_auc" in data:
                aucs.append(data["mean_outer_auc"])
            # plot each fold's curve using JSON all_fpr/all_tpr
            for fpr_vals, tpr_vals in zip(data.get("all_fpr", []), data.get("all_tpr", [])):
                fpr = np.array(fpr_vals)
                tpr = np.array(tpr_vals)
                # thin transparent lines for individual folds
                plt.plot(fpr, tpr, color=colors[algo], lw=0.8, alpha=0.15)
                interp_tpr = np.interp(fpr_grid, fpr, tpr)
                all_tprs.append(interp_tpr)

        if not all_tprs:
            continue

        # compute mean TPR and plot as solid line
        mean_tpr = np.mean(all_tprs, axis=0)
        # compute AUC statistics and bold mean curve
        mean_auc = np.mean(aucs) if aucs else 0
        std_auc = np.std(aucs) if aucs else 0
        label = f"{algo} (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
        plt.plot(fpr_grid, mean_tpr,
                 color=colors[algo],
                 lw=3,
                 label=label)

    # diagonal line for reference
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6, label="Chance")
    # add grid behind curves
    plt.grid(alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Comparison of ROC Across Diseases", fontsize=14)
    # only show legend if there are labelled artists
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(loc="lower right")
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

    # # export summary table as a tsv
    summary_table.data.to_csv("summary_table.tsv", sep="\t", index=False)

    # 3) Plot ROC curves for each algorithm
    # algorithms = results_df["Algorithm"].unique()
    #plot_roc_comparison(fold_data, algorithms)

if __name__ == "__main__":
    main()