import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_algorithm_roc(json_dir, fpr_grid=None, output_dir=None):
    """
    Reads all '*_folds.json' in json_dir, groups by algorithm, and for each:
      - plots each disease's mean ROC (averaged over its 5 folds) in light color
      - computes and plots the overall mean ROC curve (averaged across diseases)
        with bold line and AUC stats
    """
    # 1) Load and group data by algorithm
    algo_data = {}
    for fname in os.listdir(json_dir):
        if not fname.endswith("_folds.json"):
            continue
        path = os.path.join(json_dir, fname)
        with open(path) as f:
            data = json.load(f)

        # infer algorithm name
        base = fname.replace("_folds.json", "")
        key = base.split("_")[-1]
        algo = "Heat Diffusion" if key == "HeatDiffusion" else \
               "Random Walk"     if key == "RandomWalk"   else key

        algo_data.setdefault(algo, []).append(data)

    # 2) FPR grid for interpolation
    if fpr_grid is None:
        fpr_grid = np.linspace(0, 1, 200)

    # ensure output directory
    if output_dir is None:
        output_dir = json_dir
    os.makedirs(output_dir, exist_ok=True)

    # 3) Color map
    color_map = {
        "Heat Diffusion": "blue",
        "Random Walk": "red"
    }

    # 4) Plot per algorithm
    for algo, entries in algo_data.items():
        bg_color = color_map.get(algo, "gray")
        plt.figure(figsize=(8, 6))

        disease_means = []
        aucs = []

        # background = each disease's average curve
        for data in entries:
            # interpolate each fold to fpr_grid
            fold_interp = [
                np.interp(fpr_grid, fpr, tpr)
                for fpr, tpr in zip(data["all_fpr"], data["all_tpr"])
            ]
            # compute the disease-level mean TPR
            mean_tpr_disease = np.mean(fold_interp, axis=0)
            disease_means.append(mean_tpr_disease)

            # plot it
            plt.plot(fpr_grid, mean_tpr_disease,
                     color=bg_color, lw=1, alpha=0.3)

            # collect for overall AUC stats
            if "mean_outer_auc" in data:
                aucs.append(data["mean_outer_auc"])

        # overall mean curve across diseases
        mean_tpr = np.mean(disease_means, axis=0)
        mean_auc = np.mean(aucs) if aucs else np.nan
        std_auc  = np.std(aucs)  if aucs else np.nan
        label = f"{algo} (Mean AUROC = {mean_auc:.3f})"

        # overlay mean ROC
        plt.plot(fpr_grid, mean_tpr,
                 color=bg_color, lw=3, label=label)

        # diagonal chance line
        plt.plot([0, 1], [0, 1],
                 linestyle="--", color="gray", alpha=0.6)

        # labels & styling
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{algo} ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        # save as high‑quality vector PDF
        file_name = f"{algo.replace(' ', '_')}_ROC.pdf"
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

def plot_method_agreement(json_dir, output_dir=None):
    """
    Reads all '*_folds.json' in json_dir, extracts mean_outer_auc for
    Heat Diffusion and Random Walk per disease, and plots them against each other.
    """
    # collect per-disease AUCs
    aucs = {"Heat Diffusion": {}, "Random Walk": {}}
    for fname in os.listdir(json_dir):
        if not fname.endswith("_folds.json"):
            continue
        path = os.path.join(json_dir, fname)
        with open(path) as f:
            data = json.load(f)
        base = fname.replace("_folds.json", "")
        parts = base.split("_")
        key = parts[-1]
        algo = ("Heat Diffusion" if key=="HeatDiffusion"
                else "Random Walk" if key=="RandomWalk"
                else key)
        disease = "_".join(parts[:-1])
        if "mean_outer_auc" in data:
            aucs.setdefault(algo, {})[disease] = data["mean_outer_auc"]

    # find diseases with both methods
    common = set(aucs["Heat Diffusion"]).intersection(aucs["Random Walk"])
    x = [aucs["Heat Diffusion"][d] for d in common]
    y = [aucs["Random Walk"][d]   for d in common]

    # ensure output directory
    if output_dir is None:
        output_dir = json_dir
    os.makedirs(output_dir, exist_ok=True)

    # scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.7)
    # identity line
    plt.plot([0,1], [0,1], linestyle="--", color="gray", alpha=0.6)
    # zoom in axes to focus on 0.65–0.95 range
    plt.xlim(0.70, 0.95)
    plt.ylim(0.70, 0.95)
    # set tick intervals to 0.05
    plt.xticks(np.arange(0.70, 0.96, 0.05))
    plt.yticks(np.arange(0.70, 0.96, 0.05))
    plt.xlabel("Heat Diffusion AUROC")
    plt.ylabel("Random Walk AUROC")
    plt.title("Method Agreement: AUROC per Disease")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # save as PDF
    out_path = os.path.join(output_dir, "method_agreement_scatter.pdf")
    plt.savefig(out_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_optimal_params_histograms(json_dir, t_mod=(1.0, 5.0), a_mod=(0.1, 0.5),
                                   bins=30, output_dir=None):
    """
    Reads all '*_folds.json' in json_dir, collects 'optimal_t' and
    'optimal_alpha' from each file, then plots side-by-side histograms
    highlighting moderate ranges.
    """
    t_vals, a_vals = [], []
    for fname in os.listdir(json_dir):
        if not fname.endswith("_folds.json"):
            continue
        path = os.path.join(json_dir, fname)
        with open(path) as f:
            data = json.load(f)
        base = fname.replace("_folds.json", "")
        key = base.split("_")[-1]
        # pull optimal parameters directly from JSON top‐level
        if key == "HeatDiffusion" and "best_params" in data:
            t_vals.extend(data["best_params"])
        elif key == "RandomWalk" and "best_params" in data:
            a_vals.extend(data["best_params"])

    if not t_vals or not a_vals:
        raise ValueError("No optimal parameters found for HeatDiffusion or RandomWalk.")

    if output_dir is None:
        output_dir = json_dir
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(t_vals, bins=bins, color='blue', edgecolor='black')
    ax1.set(title="Optimal t (Heat Diffusion)", xlabel="t", ylabel="Count")
    ax1.set_xlim(0.0, 4.5)
    ax1.set_xticks(np.arange(0.0, 4.6, 0.5))
    ax1.grid(alpha=0.3)

    ax2.hist(a_vals, bins=bins, color='red', edgecolor='black')
    ax2.set(title="Optimal α (Random Walk)", xlabel="α", ylabel="Count")
    ax2.set_xlim(0.45, 0.95)
    ax2.set_xticks(np.arange(0.45, 1.0, 0.05))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "optimal_params_histograms.pdf")
    plt.savefig(out_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Example usage
    json_dir = "Results"  # Directory containing the JSON files
    # specify an output directory if different, else PDFs go into the same folder
    plot_algorithm_roc(json_dir, output_dir="Results/roc_pdfs")
    plot_method_agreement(json_dir, output_dir="Results/roc_pdfs")
    plot_optimal_params_histograms(json_dir, output_dir="Results/roc_pdfs")
