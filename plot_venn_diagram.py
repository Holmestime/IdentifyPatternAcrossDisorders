"""
Venn Diagram Visualization for Disorder-Specific Functional Connectivity Patterns

This script creates a three-way Venn diagram showing shared and unique 
functional connectivity patterns across Major Depressive Disorder (MDD), 
Autism Spectrum Disorder (ASD), and Marijuana Users (MJ Users).
"""

import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
from matplotlib import rcParams


def setup_plot_config():
    """Configure matplotlib parameters for publication-quality plots."""
    config = {
        "font.family": 'Arial',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 300.0,
        'pdf.fonttype': 42,  # Ensure text is editable in PDF
        'ps.fonttype': 42
    }
    rcParams.update(config)


def define_connectivity_patterns():
    """Define core functional connectivity patterns for each disorder group."""
    # Disorder-specific patterns (unique to each group)
    mdd_core = {"VN-SMN(-)"}
    asd_core = {"VN-VN(+)", "FPN-LIN(+)"}
    mj_core = {"DMN-LIN(-)"}
    
    # Shared patterns between disorder pairs
    mdd_asd = {"DMN-VAN(-)", "DMN-Sub(-)"}
    mdd_mj = {"SMN-Sub(+)", "DMN-SMN(+)"}
    asd_mj = {"Sub-Sub(-)"}
    
    # Common pattern across all three groups
    common_all = {"DMN-Sub(+)"}
    
    return mdd_core, asd_core, mj_core, mdd_asd, mdd_mj, asd_mj, common_all


def create_venn_diagram(mdd_core, asd_core, mj_core, mdd_asd, mdd_mj, asd_mj, common_all):
    """Create and customize the three-way Venn diagram."""
    # Set figure size (convert from cm to inches)
    plt.figure(figsize=(20 / 2.54, 20 / 2.54))
    
    # Create Venn diagram with subset counts
    # Order: (MDD only, ASD only, MDD∩ASD, MJ only, MDD∩MJ, ASD∩MJ, MDD∩ASD∩MJ)
    venn = venn3(subsets=(1, 1, 3, 1, 3, 3, 1),
                 set_labels=('MDD', 'ASD', 'MJUser'),
                 set_colors=('#1f77b4', '#ff7f0e', '#2ca02c'),
                 alpha=0.6)
    
    # Set text labels for each region
    venn.get_label_by_id('100').set_text('\n'.join(mdd_core))      # MDD only
    venn.get_label_by_id('010').set_text('\n'.join(asd_core))      # ASD only
    venn.get_label_by_id('001').set_text('\n'.join(mj_core))       # MJ only
    venn.get_label_by_id('110').set_text('\n'.join(mdd_asd))       # MDD ∩ ASD
    venn.get_label_by_id('101').set_text('\n'.join(mdd_mj))        # MDD ∩ MJ
    venn.get_label_by_id('011').set_text('\n'.join(asd_mj))        # ASD ∩ MJ
    venn.get_label_by_id('111').set_text('\n'.join(common_all))    # All three
    
    return venn


def format_venn_diagram(venn):
    """Apply formatting to the Venn diagram text elements."""
    # Format set labels (disorder names)
    for text in venn.set_labels:
        text.set_fontsize(24)
        text.set_fontweight('bold')
    
    # Format subset labels (connectivity patterns)
    for text in venn.subset_labels:
        text.set_fontsize(14)


def add_legend_and_title():
    """Add legend explanation and title to the plot."""
    # Add connectivity direction legend
    plt.text(-0.65, -0.55, 
             "+ Hyper\n- Hypo\nconnectivity",
             fontsize=18, 
             fontweight='bold')
    
    # Add main title
    plt.title("Key Triple-Disease Shared Patterns", 
              fontsize=24, 
              pad=20, 
              fontweight='bold')


def save_figure(save_path=None):
    """Save the figure in multiple formats if path is provided."""
    if save_path:
        plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')


def main():
    """Main function to create the Venn diagram visualization."""
    # Setup plot configuration
    setup_plot_config()
    
    # Define connectivity patterns
    mdd_core, asd_core, mj_core, mdd_asd, mdd_mj, asd_mj, common_all = define_connectivity_patterns()
    
    # Create Venn diagram
    venn = create_venn_diagram(mdd_core, asd_core, mj_core, mdd_asd, mdd_mj, asd_mj, common_all)
    
    # Format the diagram
    format_venn_diagram(venn)
    
    # Add legend and title
    add_legend_and_title()
    
    # Apply tight layout
    plt.tight_layout()
    
    # Uncomment to save figure
    # save_figure("path/to/save/venn_diagram")
    
    # Display the plot
    plt.show()


# Execute the main function
if __name__ == "__main__":
    main()