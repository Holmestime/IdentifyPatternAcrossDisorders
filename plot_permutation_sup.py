"""
Permutation Results Plotting Module

This module provides functionality to plot permutation results for different disorder
configurations in neuroimaging studies. It handles multiple disorder types and their
combinations, generating publication-ready plots.
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mat73 import loadmat


class PermutationPlotter:
    """
    A class to handle plotting of permutation results for disorder analysis.
    
    This class manages different disorder configurations and generates plots
    showing correlation coefficients across different shared basis numbers.
    """
    
    def __init__(self, root_path="./data"):
        """
        Initialize the PermutationPlotter with dataset configurations.
        
        Args:
            root_path (str): Root directory path containing all datasets
        """
        self.root_path = root_path
        self.configs = self._initialize_configs()
        self.title_mapping = self._initialize_title_mapping()
        
    def _initialize_configs(self):
        """
        Initialize dataset configurations for all supported disorders.
        
        Returns:
            dict: Configuration dictionary with dataset paths and parameters
        """
        return {
            "MDD": {
                "dataset_folder": os.path.join(self.root_path, "MDD"),
                "all_num": 24,
            },
            "ABIDE": {
                "dataset_folder": os.path.join(self.root_path, "ABIDE"), 
                "all_num": 18
            },
            "ACPI": {
                "dataset_folder": os.path.join(self.root_path, "ACPI"),
                "all_num": 6
            },
            "MDD_ABIDE": {
                "dataset_folder": os.path.join(self.root_path, "MDD_ABIDE"),
                "all_num": 36
            },
            "MDD_ACPI": {
                "dataset_folder": os.path.join(self.root_path, "MDD_ACPI"),
                "all_num": 12
            },
            "ABIDE_ACPI": {
                "dataset_folder": os.path.join(self.root_path, "ABIDE_ACPI"),
                "all_num": 12
            },
            "MULTI": {
                "dataset_folder": os.path.join(self.root_path, "MULTI"),
                "all_num": 18
            }
        }
    
    def _initialize_title_mapping(self):
        """
        Initialize mapping from dataset configs to display titles.
        
        Returns:
            dict: Mapping of configuration names to display titles
        """
        return {
            "MDD": "MDD",
            "ABIDE": "ASD",
            "ACPI": "MJUser",
            "MDD_ABIDE": "MDD&ASD",
            "MDD_ACPI": "MDD&MJUser",
            "ABIDE_ACPI": "ASD&MJUser",
            "MULTI": "MDD&ASD&MJUser"
        }
    
    def _setup_plot_style(self):
        """Configure matplotlib style settings for publication-ready plots."""
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 22
    
    def _load_permutation_data(self, file_path):
        """
        Load permutation data from .mat file.
        
        Args:
            file_path (str): Path to the .mat file containing permutation results
            
        Returns:
            numpy.ndarray or None: Loaded correlation coefficient data
        """
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found")
            return None
            
        try:
            data = loadmat(file_path)
            return data['result_cc']
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _configure_axes(self, ax):
        """
        Configure plot axes with custom x-tick labels and styling.
        
        Args:
            ax: Matplotlib axes object to configure
        """
        # Set x-axis to start from 1
        plt.xlim(left=1)
        
        # Configure x-tick labels to be +1 of actual values
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        xticks = ax.get_xticks()
        
        # Adjust labels: add 1 to integer ticks, empty for non-integer
        new_labels = []
        for x in xticks:
            if x == int(x):  # Check if tick is an integer
                new_labels.append(str(int(x + 1)))
            else:
                new_labels.append('')  # Empty label for non-integer ticks
        
        ax.set_xticklabels(new_labels)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        
        # Style the plot
        plt.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def plot_permutation_results(self, dataset_config, plot_output_folder):
        """
        Plot permutation results for a specific disorder configuration.
        
        Args:
            dataset_config (str): Configuration name (e.g., 'MDD', 'ABIDE', 'MDD_ABIDE')
            plot_output_folder (str): Output directory for saving plots
            
        Raises:
            ValueError: If dataset_config is not recognized
        """
        # Validate configuration
        if dataset_config not in self.configs:
            raise ValueError(f"Unknown dataset configuration: {dataset_config}")
        
        # Get configuration and title
        config = self.configs[dataset_config]
        title_name = self.title_mapping[dataset_config]
        
        # Extract configuration parameters
        dataset_folder = config["dataset_folder"]
        all_num = config["all_num"]
        permute_site = all_num - 1  # Calculate permute_site as all_num - 1
        
        # Create output directory
        os.makedirs(plot_output_folder, exist_ok=True)
        
        # Setup plot style
        self._setup_plot_style()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Load and plot data
        file_path = os.path.join(dataset_folder, f"permutation_cc_{permute_site}.mat")
        result_cc = self._load_permutation_data(file_path)
        
        if result_cc is not None:
            # Plot the mean correlation coefficients
            plt.plot(np.mean(result_cc, axis=0), 
                    color='blue', 
                    label=f"Permute site {permute_site} / {all_num}", 
                    linewidth=2)
        
        # Configure axes and labels
        ax = plt.gca()
        self._configure_axes(ax)
        
        plt.xlabel("Shared basis number")
        plt.ylabel("CC")
        plt.title(f"Determination of $C$ - {title_name}")
        plt.legend(frameon=False)
        
        # Save and display
        pdf_file_name = os.path.join(plot_output_folder, f"permutation_results_{dataset_config}.pdf")
        plt.savefig(pdf_file_name, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main function to generate plots for all disorder configurations.
    
    This function creates a PermutationPlotter instance and generates plots
    for single disorders, pairwise combinations, and multi-disorder analysis.
    """
    # Initialize plotter
    plotter = PermutationPlotter()
    base_result_path = "./data"
    
    # Define disorder groups for organized processing
    disorder_groups = {
        "Single disorders": ["MDD", "ABIDE", "ACPI"],
        "Pairwise disorders": ["MDD_ABIDE", "MDD_ACPI", "ABIDE_ACPI"],
        "Multi-disorder": ["MULTI"]
    }
    
    # Generate plots for each group
    for group_name, disorders in disorder_groups.items():
        print(f"\nProcessing {group_name}...")
        
        for disorder in disorders:
            output_path = os.path.join(base_result_path, disorder)
            print(f"  Plotting {disorder}...")
            
            try:
                plotter.plot_permutation_results(disorder, output_path)
                print(f"  ✓ Successfully plotted {disorder}")
            except Exception as e:
                print(f"  ✗ Error plotting {disorder}: {e}")


if __name__ == "__main__":
    main()
