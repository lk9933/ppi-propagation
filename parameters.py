import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np

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

def aggregate_and_plot(results_dir='Results/'):
    alpha_files = glob.glob(os.path.join(results_dir, '*_alpha_aucs.tsv'))
    t_files = glob.glob(os.path.join(results_dir, '*_t_aucs.tsv'))

    alpha_data = aggregate(alpha_files)
    t_data = aggregate(t_files)

    alpha_min, alpha_max = alpha_data['param'].min(), alpha_data['param'].max()
    alpha_ticks = sorted(alpha_data['param'].unique())
    alpha_xlim = (alpha_min - 0.05, alpha_max + 0.05)

    t_min, t_max = t_data['param'].min(), t_data['param'].max()
    t_ticks = sorted(t_data['param'].unique())
    t_xlim = (t_min - 0.5, t_max + 0.5)

    plot_points(alpha_data, 'AUROC vs α for RWR', 'α (Restart Probability)',
                os.path.join(results_dir, 'rwr_alpha_auroc.pdf'), color='red',
                xlim=alpha_xlim, xticks=alpha_ticks, ylim=(0.79, 0.81))

    plot_points(t_data, 'AUROC vs t for Heat Diffusion', 't (Diffusion Time)',
                os.path.join(results_dir, 'hd_t_auroc.pdf'), color='blue',
                xlim=t_xlim, xticks=t_ticks, ylim=(0.72, 0.82))

# Execute
aggregate_and_plot('Results/')