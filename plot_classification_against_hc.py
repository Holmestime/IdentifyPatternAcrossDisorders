"""
Classification Performance Visualization Module

This module provides functions to visualize and compare classification performance
between actual data and random baselines across different scales and models.

"""

import os
import pickle
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import scienceplots
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests


class ClassificationVisualizer:
    """
    A class for visualizing classification performance comparisons.
    """
    
    def __init__(self, result_folder: str):
        """
        Initialize the visualizer with configuration.
        
        Args:
            result_folder (str): Path to the result folder for saving plots
        """
        self.result_folder = result_folder
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Configure matplotlib and seaborn plotting style."""
        config = {
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.linewidth': 1.5
        }
        rcParams.update(config)
        plt.style.use('ieee')
        plt.rcParams['font.family'] = 'Arial'
    
    def _perform_statistical_analysis(self, data: pd.DataFrame, 
                                    analyze_list: list, 
                                    metric_name: str = 'F1') -> None:
        """
        Perform statistical analysis and print results.
        
        Args:
            data (pd.DataFrame): Data for analysis
            analyze_list (list): List of groups to analyze
            metric_name (str): Name of the metric column
        """
        # Print descriptive statistics
        print("\n=== Descriptive Statistics ===")
        for model in analyze_list:
            model_data = data[data['RD'] == model]
            mean_val = model_data[metric_name].mean()
            std_val = model_data[metric_name].std()
            print(f"{model}: {mean_val:.3f} ± {std_val:.3f}")
        
        # Perform pairwise comparisons
        pairs = list(itertools.combinations(analyze_list, 2))
        p_values = []
        
        print("\n=== Statistical Significance Tests ===")
        for pair in pairs:
            data1 = data[data['RD'] == pair[0]][metric_name]
            data2 = data[data['RD'] == pair[1]][metric_name]
            
            # Mann-Whitney U test (non-parametric)
            _, p = mannwhitneyu(data1, data2, alternative='two-sided')
            p_values.append(p)
            print(f"{pair[0]} vs {pair[1]}: Raw p-value = {p:.4f}")
        
        # FDR correction
        rejected, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')
        
        print("\n=== FDR Corrected Results ===")
        for i, pair in enumerate(pairs):
            significance = '*' if corrected_p[i] < 0.05 else ''
            print(f"{pair[0]} vs {pair[1]}: p-FDR = {corrected_p[i]:.2e} {significance}")
    
    def plot_violin_comparison(self, data: pd.DataFrame, 
                             shuffled_data: pd.DataFrame,
                             data_type: str, 
                             metric_name: str = 'F1') -> None:
        """
        Create violin plot comparing actual vs shuffled performance.
        
        Args:
            data (pd.DataFrame): Actual performance data
            shuffled_data (pd.DataFrame): Shuffled/random baseline data
            data_type (str): Type of data being analyzed
            metric_name (str): Performance metric to plot
        """
        # Prepare data
        df_violin = data[data['scale'] == 1.0].copy()
        df_violin_shuffled = shuffled_data[shuffled_data['scale'] == 1.0].copy()
        
        # Standardize labels
        df_violin['RD'] = df_violin['RD'].replace({'com': 'Shared'})
        df_violin_shuffled['RD'] = df_violin_shuffled['RD'].replace({'com': 'Shuffled'})
        df_violin_shuffled = df_violin_shuffled[df_violin_shuffled['RD'] == 'Shuffled']
        
        # Combine datasets
        all_data = pd.concat([df_violin, df_violin_shuffled])
        analyze_list = ['Shuffled', 'Shared']
        
        # Perform statistical analysis
        self._perform_statistical_analysis(all_data, analyze_list, metric_name)
        
        # Create visualization
        plt.figure(figsize=(6.7 / 2.54, 5.5 / 2.54))
        
        # Violin plot
        ax = sns.violinplot(
            x='RD',
            y=metric_name,
            data=all_data,
            palette=["lightgray", "skyblue"],
            inner=None,
            cut=0,
            saturation=0.8,
            order=["Shuffled", 'Shared']
        )
        
        # Add individual data points
        sns.stripplot(
            data=all_data,
            x='RD',
            y=metric_name,
            order=["Shuffled", 'Shared'],
            jitter=0.2,
            palette=["dimgray", "royalblue"],
            alpha=0.8,
            size=1.5
        )
        
        # Styling
        self._style_plot(ax, metric_name, y_label='F1-score', y_lim=(0, 103))
        plt.xticks(rotation=10)
        
        # Save and display
        output_path = f"{self.result_folder}/{data_type}/hc_compare.pdf"
        self._save_and_show_plot(output_path)
    
    def plot_scale_comparison(self, data: pd.DataFrame, 
                            shuffled_data: pd.DataFrame,
                            data_type: str, 
                            metric_name: str = 'F1') -> None:
        """
        Create line plot showing performance across different scales.
        
        Args:
            data (pd.DataFrame): Actual performance data
            shuffled_data (pd.DataFrame): Shuffled/random baseline data
            data_type (str): Type of data being analyzed
            metric_name (str): Performance metric to plot
        """
        # Prepare data
        df_violin = data.copy()
        df_violin_shuffled = shuffled_data.copy()
        
        # Standardize labels
        df_violin['RD'] = df_violin['RD'].replace({
            'com': 'Shared', 
            'spec': 'Personalized'
        })
        df_violin_shuffled['RD'] = df_violin_shuffled['RD'].replace({'com': 'Shuffled'})
        df_violin_shuffled = df_violin_shuffled[df_violin_shuffled['RD'] == 'Shuffled']
        
        # Combine datasets
        all_data = pd.concat([df_violin, df_violin_shuffled]).reset_index()
        analyze_list = ['Shuffled', 'Shared']
        
        # Create visualization
        plt.figure(figsize=(8/2.54, 8/2.54))
        
        ax = sns.lineplot(
            x='scale',
            y=metric_name,
            hue='RD',
            data=all_data,
            marker='o',
            style=False,
            palette=["gray", "blue"],
            ci=95,
            legend=False,
            alpha=0.8,
            hue_order=analyze_list,
            linewidth=2
        )
        
        # Customize axes
        plt.ylim([0, 110])
        plt.xticks([0.1, 0.2, 0.5, 1.0], ['10%', '20%', '50%', '100%'], rotation=60)
        plt.xlabel('Percentage of top edges', fontsize=16)
        plt.ylabel('F1-score', fontsize=16)
        plt.tick_params(axis='both', labelsize=12)
        
        # Add custom legend
        self._add_custom_legend(ax, analyze_list)
        
        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        
        # Save and display
        output_path = f"{self.result_folder}/{data_type}/hc_f1_scale_percentage.pdf"
        self._save_and_show_plot(output_path)
    
    def _style_plot(self, ax, metric_name: str, y_label: str = None, y_lim: tuple = None):
        """Apply common styling to plots."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if y_lim:
            ax.set_ylim(y_lim)
        
        ax.set_xlabel('')
        
        if y_label:
            ax.set_ylabel(y_label, fontsize=16)
        
        plt.tick_params(axis='both', labelsize=14)
        plt.tight_layout()
    
    def _add_custom_legend(self, ax, analyze_list: list):
        """Add custom legend to the plot."""
        colors = ["gray", "blue"]
        legend_elements = [
            Line2D([0], [0], color=colors[i], lw=2, label=label, markerfacecolor=colors[i])
            for i, label in enumerate(analyze_list)
        ]
        
        ax.legend(
            handles=legend_elements,
            fontsize=12,
            frameon=False,
            bbox_to_anchor=(0.5, 1.05),
            loc='lower center',
            ncol=len(analyze_list),
            borderaxespad=0.
        )
    
    def _save_and_show_plot(self, output_path: str):
        """Save plot to file and display."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=600, format='pdf')
        plt.show()


def main():
    """
    Main function to execute the visualization pipeline.
    """
    # Configuration
    RESULT_FOLDER = "./data"
    DATA_TYPE_LIST = ["MULTI"]  # ["MDD", "ABIDE", "ACPI", "MDD_ACPI", "MDD_ABIDE", "ABIDE_ACPI", "MULTI"]
    FILE_NAME = "performance_hc_compare"
    PLOT_VIOLIN_FLAG = True
    PLOT_PERCENTAGE_FLAG = True
    
    # Initialize visualizer
    visualizer = ClassificationVisualizer(RESULT_FOLDER)
    
    # Process each data type
    for data_type in DATA_TYPE_LIST:
        print(f"\n{'='*60}")
        print(f"Processing: {data_type}")
        print(f"{'='*60}")
        
        # Load data
        data_file = f"{RESULT_FOLDER}/{data_type}/{FILE_NAME}.csv"
        shuffled_file = f"{RESULT_FOLDER}/{data_type}/{FILE_NAME}_random.csv"
        
        try:
            true_data = pd.read_csv(data_file)
            shuffled_data = pd.read_csv(shuffled_file)
            
            # Generate plots
            if PLOT_VIOLIN_FLAG:
                print("\nGenerating violin plot...")
                visualizer.plot_violin_comparison(true_data, shuffled_data, data_type)
            
            if PLOT_PERCENTAGE_FLAG:
                print("\nGenerating scale comparison plot...")
                visualizer.plot_scale_comparison(true_data, shuffled_data, data_type)
                
        except FileNotFoundError as e:
            print(f"Error: Could not find data files for {data_type}")
            print(f"Expected files: {data_file}, {shuffled_file}")
            continue
        except Exception as e:
            print(f"Error processing {data_type}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
