"""
This script plots the performance of regression models across different disorders and scales.
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
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
import itertools


class RegressionPerformancePlotter:
    """
    A class to visualize regression performance across different disorders and scales.
    Supports ABIDE and MDD datasets with various scoring types.
    """
    
    def __init__(self, result_folder):
        """
        Initialize the plotter with result folder path.
        
        Args:
            result_folder (str): Path to the folder containing results
        """
        self.result_folder = result_folder
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Configure matplotlib plotting style."""
        config = {
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.linewidth': 1.5
        }
        rcParams.update(config)
        plt.style.use('ieee')
        plt.rcParams['font.family'] = 'Arial'
    
    def _preprocess_data(self, data, data_type, score_type, include_afc=False):
        """
        Preprocess data by renaming columns based on data type and score type.
        
        Args:
            data (DataFrame): Input data
            data_type (str): Either "ABIDE" or "MDD"
            score_type (str): Score type ("HAMD" or "ados_total")
            include_afc (bool): Whether to include AFC data
            
        Returns:
            DataFrame: Preprocessed data
        """
        data = data.copy()
        
        if data_type == "ABIDE":
            data['RD'] = data['RD'].replace({
                'ados_total_com': 'Shared', 
                'ados_total_spec': 'Personalized',
                'ados_total_afc': 'AFC'
            })
            
        elif data_type == "MDD":
            if score_type == "HAMD":
                data = data[data['RD'].str.contains('hamd')]
                data['RD'] = data['RD'].replace({
                    'hamd_com': 'Shared', 
                    'hamd_spec': 'Personalized',
                    'hamd_afc': 'AFC'
                })
        
        # Remove AFC data if not needed
        if not include_afc:
            data = data[data['RD'] != 'AFC']
            
        return data
    
    def _preprocess_shuffled_data(self, shuffled_data, data_type, score_type):
        """
        Preprocess shuffled data for baseline comparison.
        
        Args:
            shuffled_data (DataFrame): Shuffled data
            data_type (str): Either "ABIDE" or "MDD"
            score_type (str): Score type
            
        Returns:
            DataFrame: Preprocessed shuffled data
        """
        shuffled_data = shuffled_data.copy()
        
        if data_type == "ABIDE":
            shuffled_data['RD'] = shuffled_data['RD'].replace({'ados_total_com': 'Shuffled'})
            shuffled_data = shuffled_data[shuffled_data['RD'] == 'Shuffled']
            
        elif data_type == "MDD":
            if score_type == "HAMD":
                shuffled_data = shuffled_data[shuffled_data['RD'].str.contains('hamd')]
                shuffled_data['RD'] = shuffled_data['RD'].replace({'hamd_com': 'Shuffled'})
                shuffled_data = shuffled_data[shuffled_data['RD'] == 'Shuffled']
                
        return shuffled_data
    
    def plot_scale_performance(self, data, data_type, score_type, include_random=False, shuffled_data=None):
        """
        Plot regression performance across different edge percentage scales.
        
        Args:
            data (DataFrame): Main performance data
            data_type (str): Dataset type ("ABIDE" or "MDD")
            score_type (str): Score type
            include_random (bool): Whether to include random baseline
            shuffled_data (DataFrame): Shuffled data for baseline comparison
        """
        # Preprocess main data
        processed_data = self._preprocess_data(data, data_type, score_type)
        
        # Handle shuffled data if provided
        if include_random and shuffled_data is not None:
            processed_shuffled = self._preprocess_shuffled_data(shuffled_data, data_type, score_type)
            all_data = pd.concat([processed_data, processed_shuffled]).reset_index()
            analyze_list = ['Shuffled', 'Shared', 'Personalized']
            colors = ["gray", "blue", "red"]
            figure_size = (8.2 / 2.54, 7.5 / 2.54)
        else:
            all_data = processed_data
            analyze_list = ['Shared', 'Personalized']
            colors = ["blue", "red"]
            figure_size = (9 / 2.54, 6.5 / 2.54)
        
        # Create the plot
        plt.figure(figsize=figure_size)
        
        ax = sns.lineplot(
            x='scale',
            y='CC',
            hue='RD',
            data=all_data,
            marker='o',
            style=False,
            palette=colors,
            ci=95,
            legend=False,
            alpha=0.8,
            hue_order=analyze_list
        )
        
        # Customize plot appearance
        plt.xticks([0.1, 0.2, 0.5, 1.0], ['10%', '20%', '50%', '100%'], rotation=60)
        plt.xlabel('Percentage of top edges', fontsize=14)
        plt.ylabel('Regression-CC', fontsize=15)
        plt.tick_params(axis='both', labelsize=12)
        
        # Set y-axis limits for MDD data
        if data_type == "MDD":
            plt.ylim([-0.1, 0.35])
        
        # Create custom legend
        if include_random:
            legend_elements = [
                Line2D([0], [0], color='gray', lw=2, label='Shuffled'),
                Line2D([0], [0], color='blue', lw=2, label='Shared'),
                Line2D([0], [0], color='red', lw=2, label='Personalized'),
            ]
            ax.legend(handles=legend_elements, fontsize=9, frameon=False,
                     bbox_to_anchor=(0.4, 1.05), loc='lower center', ncol=3, borderaxespad=0.)
        else:
            legend_elements = [
                Line2D([0], [0], color='blue', lw=2, label='Shared'),
                Line2D([0], [0], color='red', lw=2, label='Personalized'),
            ]
            ax.legend(handles=legend_elements, fontsize=10, frameon=False)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        
        # Save plots
        self._save_plot(f"com_specific_scale_{score_type}", data_type)
        plt.show()
    
    def plot_violin_comparison(self, data, data_type, score_type, include_random=False, 
                             shuffled_data=None, metric_name='CC'):
        """
        Plot violin plots comparing different model types.
        
        Args:
            data (DataFrame): Main performance data
            data_type (str): Dataset type
            score_type (str): Score type
            include_random (bool): Whether to include random baseline
            shuffled_data (DataFrame): Shuffled data for baseline
            metric_name (str): Metric to plot
        """
        # Preprocess data
        processed_data = self._preprocess_data(data, data_type, score_type)
        
        if include_random and shuffled_data is not None:
            processed_shuffled = self._preprocess_shuffled_data(shuffled_data, data_type, score_type)
            all_data = pd.concat([processed_data, processed_shuffled])
            analyze_list = ['Shuffled', 'Shared', 'Personalized']
            
            # Perform statistical analysis
            self._perform_statistical_analysis(all_data, analyze_list, metric_name)
            
            # Color schemes for violin and scatter plots
            violin_colors = {'Shuffled': 'lightgray', 'Shared': 'skyblue', 'Personalized': 'salmon'}
            scatter_colors = {'Shuffled': 'dimgray', 'Shared': 'royalblue', 'Personalized': 'firebrick'}
            rotation = 15
        else:
            all_data = processed_data
            analyze_list = ['Shared', 'Personalized']
            violin_colors = {'Shared': 'skyblue', 'Personalized': 'salmon'}
            scatter_colors = {'Shared': 'royalblue', 'Personalized': 'firebrick'}
            rotation = 10
        
        # Create the plot
        plt.figure(figsize=(6 / 2.54, 5 / 2.54))
        
        # Plot violin plot
        sns.violinplot(
            data=all_data,
            x='RD',
            y=metric_name,
            order=analyze_list,
            palette=[violin_colors[dt] for dt in analyze_list],
            inner=None,
            saturation=0.8
        )
        
        # Plot scatter points
        sns.stripplot(
            data=all_data,
            x='RD',
            y=metric_name,
            order=analyze_list,
            jitter=0.2,
            palette=[scatter_colors[dt] for dt in analyze_list],
            alpha=0.8,
            size=1.5
        )
        
        # Customize appearance
        sns.despine()
        plt.xlabel('')
        plt.ylabel("Regression-CC", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tick_params(axis='both', labelsize=10)
        plt.xticks(rotation=rotation)
        plt.tight_layout()
        
        # Save plots
        self._save_plot(f"com_specific_compare_{score_type}", data_type)
        plt.show()
    
    def _perform_statistical_analysis(self, data, analyze_list, metric_name):
        """
        Perform statistical analysis and print results.
        
        Args:
            data (DataFrame): Data for analysis
            analyze_list (list): List of categories to analyze
            metric_name (str): Metric column name
        """
        # Print descriptive statistics
        for c_model in analyze_list:
            c_data = data[data['RD'] == c_model]
            print(f"{c_model}, {c_data[metric_name].mean():.3f} +- {c_data[metric_name].std():.3f}")
        
        # Perform pairwise comparisons
        pairs = list(itertools.combinations(analyze_list, 2))
        p_values = []
        
        print("\nSignificance test results:")
        
        for pair in pairs:
            data1 = data[data['RD'] == pair[0]][metric_name]
            data2 = data[data['RD'] == pair[1]][metric_name]
            
            # Mann-Whitney U test (non-parametric)
            _, p = mannwhitneyu(data1, data2, alternative='two-sided')
            p_values.append(p)
            print(f"{pair[0]} vs {pair[1]}: Raw p-value = {p:.4f}")
        
        # FDR correction
        rejected, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')
        
        print("\n### FDR corrected results ###")
        for i, pair in enumerate(pairs):
            significance = '*' if corrected_p[i] < 0.05 else ''
            print(f"{pair[0]} vs {pair[1]}: p-FDR = {corrected_p[i]:.2e} {significance}")
    
    def _save_plot(self, filename, data_type):
        """
        Save plot in both PNG and PDF formats.
        
        Args:
            filename (str): Base filename
            data_type (str): Dataset type for folder structure
        """
        base_path = f"{self.result_folder}/{data_type}/{filename}"
        plt.savefig(f"{base_path}.png", dpi=600, format='png')
        plt.savefig(f"{base_path}.pdf", dpi=600, format='pdf')


def main():
    """Main execution function."""
    # Configuration
    data_type = "ABIDE"
    plot_violin_flag = True
    plot_percentage_flag = True
    result_folder = "./data"
    
    # Initialize plotter
    plotter = RegressionPerformancePlotter(result_folder)
    
    # Load data
    data_df = pd.read_csv(os.path.join(result_folder, data_type, "performance.csv"))
    data_df_shuffled = pd.read_csv(os.path.join(result_folder, data_type, "performance_random.csv"))
    
    # Plot violin plots (scale=1 only)
    if plot_violin_flag:
        selected_data = data_df[data_df['scale'] == 1]
        selected_shuffled = data_df_shuffled[data_df_shuffled['scale'] == 1]
        
        if data_type == "MDD":
            # Plot for HAMD score only
            plotter.plot_violin_comparison(
                selected_data, data_type, "HAMD", 
                include_random=True, shuffled_data=selected_shuffled
            )
        else:
            # Plot for ABIDE dataset
            plotter.plot_violin_comparison(
                selected_data, data_type, "ados_total",
                include_random=True, shuffled_data=selected_shuffled
            )
    
    # Plot performance across different scales
    if plot_percentage_flag:
        if data_type == "MDD":
            # Plot for HAMD score only
            score_data = data_df[data_df['RD'].str.contains('hamd')]
            score_shuffled = data_df_shuffled[data_df_shuffled['RD'].str.contains('hamd')]
            
            plotter.plot_scale_performance(
                score_data, data_type, "HAMD",
                include_random=True, shuffled_data=score_shuffled
            )
        else:
            # Plot for ABIDE dataset
            ados_data = data_df[data_df['RD'].str.contains('ados_total')]
            ados_shuffled = data_df_shuffled[data_df_shuffled['RD'].str.contains('ados_total')]
            
            plotter.plot_scale_performance(
                ados_data, data_type, "ados_total",
                include_random=True, shuffled_data=ados_shuffled
            )


if __name__ == "__main__":
    main()
