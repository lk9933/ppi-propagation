import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

CATEGORY_MAP = {
    'psychiatric': {
        'anxiety', 'depression', 'bipolar', 'schizophrenia',
        'attention_deficit_disorder'
    },
    'substance': {
        'alcohol_use_disorder', 'cocaine_use_disorder', 'tobacco_use_disorder'
    },
    'control': {
        'diabetes', 'epilepsy'
    }
}
CATEGORY_COLORS = {
    'psychiatric': 'blue',
    'substance': 'red',
    'control': 'green'
}

def aggregate(files):
    all_data = []
    for file in files:
        df = pd.read_csv(file, sep='\t')
        df['mean_auroc'] = df.iloc[:, 1:].mean(axis=1)
        df = df[[df.columns[0], 'mean_auroc']]
        df.columns = ['param', 'mean_auroc']
        all_data.append(df)
    combined = pd.concat(all_data)
    return combined.groupby('param', as_index=False)['mean_auroc'].mean()

def plot_points(df, title, xlabel, output_path, color, xlim=None, xticks=None, ylim=None):
    plt.figure(figsize=(10, 6))
    plt.plot(df['param'], df['mean_auroc'], color=color, linewidth=2)
    plt.scatter(df['param'], df['mean_auroc'], color=color, edgecolor='black', s=60)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Average AUROC', fontsize=12, labelpad=10)
    plt.title(title, fontsize=14)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    if xticks is not None:
        plt.xticks(xticks, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()

def categorize_files(files):
    from collections import defaultdict
    buckets = defaultdict(list)
    for f in files:
        parts = os.path.basename(f).replace('.tsv','').split('_')
        # Remove the last 2 tokens (e.g., "alpha"/"t" and "aucs")
        disease = '_'.join(parts[:-2])
        for cat, diseases in CATEGORY_MAP.items():
            if disease in diseases:
                buckets[cat].append(f)
                break
    return buckets


def aggregate_and_plot(results_dir='Results/'):
    alpha_files = glob.glob(os.path.join(results_dir, '*_alpha_aucs.tsv'))
    t_files = glob.glob(os.path.join(results_dir, '*_t_aucs.tsv'))

    # split into category buckets
    alpha_buckets = categorize_files(alpha_files)
    t_buckets    = categorize_files(t_files)

    # aggregate data per category
    alpha_data = {cat: aggregate(files) for cat, files in alpha_buckets.items()}
    t_data     = {cat: aggregate(files) for cat, files in t_buckets.items()}

    # Configure axis limits/ticks
    alpha_min = min(df['param'].min() for df in alpha_data.values())
    alpha_max = max(df['param'].max() for df in alpha_data.values())
    alpha_ticks = sorted(set(itertools.chain.from_iterable(
        df['param'].unique() for df in alpha_data.values())))
    alpha_xlim = (alpha_min - 0.05, alpha_max + 0.05)

    t_min = min(df['param'].min() for df in t_data.values())
    t_max = max(df['param'].max() for df in t_data.values())
    t_ticks = sorted(set(itertools.chain.from_iterable(
        df['param'].unique() for df in t_data.values())))
    t_xlim = (t_min - 0.5, t_max + 0.5)

    # === α plot ===
    plt.figure(figsize=(10, 6))
    for cat, df in alpha_data.items():
        color = CATEGORY_COLORS[cat]
        plt.plot(df['param'], df['mean_auroc'], label=cat.capitalize(),
                 color=color, linewidth=2)
        plt.scatter(df['param'], df['mean_auroc'], color=color,
                    edgecolor='black', s=60)
    plt.xlabel('α (Restart Probability)', fontsize=12)
    plt.ylabel('Average AUROC', fontsize=12, labelpad=10)
    plt.title('AUROC vs α for RWR (by disease class)', fontsize=14)
    plt.xlim(alpha_xlim); plt.ylim(0.725, 0.875)
    plt.xticks(alpha_ticks, fontsize=10); plt.yticks(fontsize=10)
    plt.grid(True, axis='y'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'rwr_alpha_auroc.pdf'), format='pdf')
    plt.close()

    # === t plot ===
    plt.figure(figsize=(10, 6))
    for cat, df in t_data.items():
        color = CATEGORY_COLORS[cat]
        plt.plot(df['param'], df['mean_auroc'], label=cat.capitalize(),
                 color=color, linewidth=2)
        plt.scatter(df['param'], df['mean_auroc'], color=color,
                    edgecolor='black', s=60)
    plt.xlabel('t (Diffusion Time)', fontsize=12)
    plt.ylabel('Average AUROC', fontsize=12, labelpad=10)
    plt.title('AUROC vs t for Heat Diffusion (by disease class)', fontsize=14)
    plt.xlim(t_xlim); plt.ylim(0.65, 0.9)
    plt.xticks(t_ticks, fontsize=10); plt.yticks(fontsize=10)
    plt.grid(True, axis='y'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'hd_t_auroc.pdf'), format='pdf')
    plt.close()

if __name__ == "__main__":
    aggregate_and_plot()
    print("Plots saved to Results/")