""" 
Plot classification performance across different models.
"""

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import scienceplots
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import itertools


def plot_violin_with_statistical_analysis(data, shuffled_data, data_type, metric_name='F1'):
    """
    Plot violin plot comparing Shuffled, Shared, and Personalized models with statistical analysis.
    
    Args:
        data: DataFrame with true data results
        shuffled_data: DataFrame with shuffled data results  
        data_type: String indicating dataset type (e.g., "MDD_ACPI")
        metric_name: String indicating metric to plot (default: 'F1')
    """
    # Filter data for scale = 1.0 (100% edges)
    df_violin = data[data['scale'] == 1.0].copy()
    df_violin_shuffled = shuffled_data[shuffled_data['scale'] == 1.0].copy()

    # Rename categories for clarity
    df_violin['RD'] = df_violin['RD'].replace({'com': 'Shared', 'spec': 'Personalized'})
    df_violin_shuffled['RD'] = df_violin_shuffled['RD'].replace({'com': 'Shuffled'})
    df_violin_shuffled = df_violin_shuffled[df_violin_shuffled['RD'] == 'Shuffled']

    # Combine datasets
    all_data = pd.concat([df_violin, df_violin_shuffled])
    analyze_list = ['Shuffled', 'Shared', 'Personalized']

    # Print descriptive statistics
    print("Descriptive Statistics:")
    for c_model in analyze_list:
        c_data = all_data[all_data['RD'] == c_model]
        print(f"{c_model}: {c_data[metric_name].mean():.3f} ± {c_data[metric_name].std():.3f}")

    # Statistical significance testing
    pairs = list(itertools.combinations(analyze_list, 2))
    p_values = []

    print("\nStatistical Significance Testing:")
    for pair in pairs:
        data1 = all_data[all_data['RD'] == pair[0]][metric_name]
        data2 = all_data[all_data['RD'] == pair[1]][metric_name]
        
        # Mann-Whitney U test (non-parametric)
        _, p = mannwhitneyu(data1, data2, alternative='two-sided')
        p_values.append(p)
        print(f"{pair[0]} vs {pair[1]}: Raw p-value = {p:.4f}")

    # FDR correction
    rejected, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')
    print("\nFDR Corrected Results:")
    for i, pair in enumerate(pairs):
        significance = '*' if corrected_p[i] < 0.05 else ''
        print(f"{pair[0]} vs {pair[1]}: p-FDR = {corrected_p[i]:.2e} {significance}")

    # Create violin plot
    plt.figure(figsize=(7/2.54, 6/2.54))  # Small publication-ready size
    
    # Violin plot with custom colors
    ax = sns.violinplot(
        x='RD', y=metric_name, data=all_data,
        palette=["lightgray", "skyblue", "salmon"],
        inner=None, cut=0, saturation=0.8,
        order=["Shuffled", 'Shared', 'Personalized']
    )

    # Add individual data points
    sns.stripplot(
        data=all_data, x='RD', y=metric_name,
        order=["Shuffled", 'Shared', 'Personalized'],
        jitter=0.2, palette=["dimgray", "royalblue", "firebrick"],
        alpha=0.8, size=1.5
    )

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 105)
    ax.set_xlabel('')
    ax.set_ylabel('F1-score', fontsize=14)
    
    plt.xticks(rotation=10)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()

    # Save plots
    plt.savefig(f"{result_folder}/{data_type}/avg_performance.png", dpi=600, format='png')
    plt.savefig(f"{result_folder}/{data_type}/avg_performance.pdf", dpi=600, format='pdf')
    plt.show()


def plot_performance_across_scales(data, shuffled_data, data_type, metric_name='F1'):
    """
    Plot performance across different edge percentage scales with confidence intervals.
    
    Args:
        data: DataFrame with true data results
        shuffled_data: DataFrame with shuffled data results
        data_type: String indicating dataset type
        metric_name: String indicating metric to plot (default: 'F1')
    """
    # Prepare data
    df_data = data.copy()
    df_shuffled = shuffled_data.copy()
    
    # Rename categories
    df_data['RD'] = df_data['RD'].replace({'com': 'Shared', 'spec': 'Personalized'})
    df_shuffled['RD'] = df_shuffled['RD'].replace({'com': 'Shuffled'})
    df_shuffled = df_shuffled[df_shuffled['RD'] == 'Shuffled']

    # Combine datasets
    all_data = pd.concat([df_data, df_shuffled]).reset_index()
    analyze_list = ['Shuffled', 'Shared', 'Personalized']

    # Create line plot
    plt.figure(figsize=(8/2.54, 7.1/2.54))
    
    ax = sns.lineplot(
        x='scale', y=metric_name, hue='RD', data=all_data,
        marker='o', palette=["gray", "blue", "red"],
        ci=95, legend=False, alpha=0.8,
        hue_order=analyze_list
    )

    # Styling
    plt.ylim([0, 100])
    plt.xticks([0.1, 0.2, 0.5, 1.0], ['10%', '20%', '50%', '100%'], rotation=60)
    plt.xlabel('Percentage of top edges', fontsize=14)
    plt.ylabel('F1-score', fontsize=15)
    plt.tick_params(axis='both', labelsize=12)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='Shuffled', markerfacecolor='gray'),
        Line2D([0], [0], color='blue', lw=2, label='Shared', markerfacecolor='blue'),
        Line2D([0], [0], color='red', lw=2, label='Personalized', markerfacecolor='red'),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, frameon=False,
              bbox_to_anchor=(0.35, 1.05), loc='lower center', ncol=3, borderaxespad=0.)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    # Save plots
    plt.savefig(f"{result_folder}/{data_type}/f1_scale.png", dpi=300, format='png')
    plt.savefig(f"{result_folder}/{data_type}/f1_scale.pdf", dpi=300, format='pdf')
    plt.show()


def main():
    """
    Main function to execute the plotting pipeline.
    """
    # Configuration for publication-quality plots
    config = {
        'pdf.fonttype': 42,
        'ps.fonttype': 42, 
        'axes.linewidth': 1.5
    }
    rcParams.update(config)
    plt.style.use('ieee')
    plt.rcParams['font.family'] = 'Arial'

    # Analysis parameters
    plot_violin_flag = True
    plot_percentage_flag = True
    
    # File paths
    result_folder = "./data"
    data_type_list = ["MDD_ACPI"]
    file_name = "performance_repeat"

    # Process each dataset
    for data_type in data_type_list:
        print(f"Processing {data_type}")
        print("-" * 60)
        
        # Load data files
        fullfile_name = f"{result_folder}/{data_type}/{file_name}.csv"
        shuffled_fullfile_name = f"{result_folder}/{data_type}/{file_name}_random.csv"
        
        true_data = pd.read_csv(fullfile_name)
        shuffled_data = pd.read_csv(shuffled_fullfile_name)

        # Generate plots
        if plot_violin_flag:
            plot_violin_with_statistical_analysis(true_data, shuffled_data, data_type)

        if plot_percentage_flag:
            plot_performance_across_scales(true_data, shuffled_data, data_type)


if __name__ == "__main__":
    # Global variable for result folder (consider passing as parameter in production)
    result_folder = "./data"
    main()
