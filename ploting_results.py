import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

print("Generating publication-ready plots...")

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.8

# Professional color palette
colors = {
    'U-Net3D': '#2E86AB',           # Deep blue
    'U-Net3D_Simple': '#A23B72',    # Deep rose
    '3D DeepLabV3+': '#F18F01',     # Orange
    'DeepLabV3+_Simple': '#C73E1D'  # Red-orange
}

# --- Data from results.py ---

dice_data = {
    'U-Net3D': {'WT': 0.884, 'TC': 0.835, 'ET': 0.776},
    'U-Net3D_Simple': {'WT': 0.871, 'TC': 0.818, 'ET': 0.758},
    '3D DeepLabV3+': {'WT': 0.879, 'TC': 0.846, 'ET': 0.788},
    'DeepLabV3+_Simple': {'WT': 0.864, 'TC': 0.829, 'ET': 0.769}
}

hd95_data = {
    'U-Net3D': {'WT': 6.34, 'TC': 5.21, 'ET': 4.87},
    'U-Net3D_Simple': {'WT': 7.12, 'TC': 5.89, 'ET': 5.34},
    '3D DeepLabV3+': {'WT': 5.89, 'TC': 4.67, 'ET': 4.23},
    'DeepLabV3+_Simple': {'WT': 7.45, 'TC': 5.76, 'ET': 5.12}
}

efficiency_data = {
    'U-Net3D': {
        'params': 15.2,
        'avg_dice': np.mean([0.884, 0.835, 0.776]),
        'color': colors['U-Net3D']
    },
    'U-Net3D_Simple': {
        'params': 4.8,
        'avg_dice': np.mean([0.871, 0.818, 0.758]),
        'color': colors['U-Net3D_Simple']
    },
    '3D DeepLabV3+': {
        'params': 18.6,
        'avg_dice': np.mean([0.879, 0.846, 0.788]),
        'color': colors['3D DeepLabV3+']
    },
    'DeepLabV3+_Simple': {
        'params': 5.3,
        'avg_dice': np.mean([0.864, 0.829, 0.769]),
        'color': colors['DeepLabV3+_Simple']
    }
}

stats_data = {
    'TC_dice_p': 0.032,
    'ET_dice_p': 0.045
}


# --- 1. Generate Figure 2: Dice Score Comparison ---

print("Generating Figure 2: Dice Score Comparison...")
df_list_dice = []
for model, regions in dice_data.items():
    for region, dice in regions.items():
        df_list_dice.append({'Model': model, 'Region': region, 'Dice Score': dice})

df_dice = pd.DataFrame(df_list_dice)

fig, ax = plt.subplots(figsize=(10, 6))
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})

# Custom bar plot with better spacing
x = np.arange(len(['WT', 'TC', 'ET']))
width = 0.2
regions = ['WT', 'TC', 'ET']

for i, (model, model_data) in enumerate(dice_data.items()):
    values = [model_data[region] for region in regions]
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, values, width, label=model,
                   color=colors[model], edgecolor='black', linewidth=0.8,
                   alpha=0.85)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8, rotation=0)

# Add significance markers with brackets
def add_significance_bracket(ax, x_pos, y_start, p_value, height=0.012):
    """Add a bracket with p-value annotation"""
    bracket_h = height
    ax.plot([x_pos - 0.05, x_pos - 0.05, x_pos + 0.05, x_pos + 0.05],
            [y_start, y_start + bracket_h, y_start + bracket_h, y_start],
            'k-', linewidth=1.2)
    ax.text(x_pos, y_start + bracket_h + 0.002,
            f'*\n$p={p_value}$',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add significance for TC and ET
y_tc = max([dice_data[model]['TC'] for model in dice_data.keys()]) + 0.008
y_et = max([dice_data[model]['ET'] for model in dice_data.keys()]) + 0.008

add_significance_bracket(ax, 1, y_tc, stats_data['TC_dice_p'])
add_significance_bracket(ax, 2, y_et, stats_data['ET_dice_p'])

ax.set_ylim(0.74, 0.92)
ax.set_ylabel('Dice Similarity Coefficient (DSC)', fontweight='bold')
ax.set_xlabel('Tumor Sub-Region', fontweight='bold')
ax.set_title('Segmentation Performance Across Tumor Regions', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(regions)
ax.legend(frameon=True, shadow=True, fancybox=True, loc='upper right')
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('dice_score_comparison.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('dice_score_comparison.pdf', bbox_inches='tight', facecolor='white')
print("Saved dice_score_comparison.png and .pdf")


# --- 2. Generate Figure 3: HD95 Comparison ---

print("Generating Figure 3: HD95 Comparison...")
df_list_hd95 = []
for model, regions in hd95_data.items():
    for region, hd95 in regions.items():
        df_list_hd95.append({'Model': model, 'Region': region, 'HD95 (mm)': hd95})

df_hd95 = pd.DataFrame(df_list_hd95)

fig, ax = plt.subplots(figsize=(10, 6))
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})

# Custom bar plot
for i, (model, model_data) in enumerate(hd95_data.items()):
    values = [model_data[region] for region in regions]
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, values, width, label=model,
                   color=colors[model], edgecolor='black', linewidth=0.8,
                   alpha=0.85)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

ax.set_ylabel('95th Percentile Hausdorff Distance (mm)', fontweight='bold')
ax.set_xlabel('Tumor Sub-Region', fontweight='bold')
ax.set_title('Boundary Accuracy Assessment (Lower is Better)', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(regions)
ax.legend(frameon=True, shadow=True, fancybox=True, loc='upper right')
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.set_ylim(0, max(df_hd95['HD95 (mm)']) * 1.15)

plt.tight_layout()
plt.savefig('hd95_comparison.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('hd95_comparison.pdf', bbox_inches='tight', facecolor='white')
print("Saved hd95_comparison.png and .pdf")


# --- 3. Generate Figure 4: Performance-Efficiency Trade-off ---

print("Generating Figure 4: Performance-Efficiency Trade-off...")
fig, ax = plt.subplots(figsize=(10, 7))
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})

# Plot points with connecting lines
model_names = list(efficiency_data.keys())
params_list = [efficiency_data[m]['params'] for m in model_names]
dice_list = [efficiency_data[m]['avg_dice'] for m in model_names]

# Draw Pareto frontier for reference
pareto_models = ['DeepLabV3+_Simple', '3D DeepLabV3+']
pareto_params = [efficiency_data[m]['params'] for m in pareto_models]
pareto_dice = [efficiency_data[m]['avg_dice'] for m in pareto_models]
ax.plot(pareto_params, pareto_dice, '--', color='gray', alpha=0.4,
        linewidth=1.5, label='Pareto Frontier', zorder=1)

# Plot each model
for model_name, data in efficiency_data.items():
    # Main scatter point
    scatter = ax.scatter(
        data['params'],
        data['avg_dice'],
        s=400,
        c=data['color'],
        edgecolors='black',
        linewidth=2,
        alpha=0.9,
        zorder=3,
        marker='o'
    )

    # Add elegant labels with white background boxes
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor=data['color'], alpha=0.9, linewidth=1.5)

    # Adjust label positions for clarity
    if 'Simple' in model_name:
        va_pos = 'top'
        y_offset = -0.0025
    else:
        va_pos = 'bottom'
        y_offset = 0.0025

    ax.annotate(f"{model_name}\n{data['params']:.1f}M params",
                xy=(data['params'], data['avg_dice']),
                xytext=(0, y_offset * 200),
                textcoords='offset points',
                ha='center', va=va_pos,
                fontsize=9,
                bbox=bbox_props,
                zorder=4)

ax.set_title('Model Performance versus Computational Efficiency',
             fontweight='bold', pad=20)
ax.set_xlabel('Model Parameters (Millions)', fontweight='bold')
ax.set_ylabel('Average Dice Score (WT + TC + ET)', fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Set nice limits
ax.set_xlim(3, 20)
ax.set_ylim(0.810, 0.845)

# Add custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5,
           label='Pareto Frontier')
]
ax.legend(handles=legend_elements, loc='lower right',
         frameon=True, shadow=True, fancybox=True)

plt.tight_layout()
plt.savefig('efficiency_tradeoff.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('efficiency_tradeoff.pdf', bbox_inches='tight', facecolor='white')
print("Saved efficiency_tradeoff.png and .pdf")

print("\nAll publication-ready plots generated successfully!")
print("Generated both PNG (600 dpi) and PDF versions for LaTeX.")
print("\nKey improvements:")
print("- Professional serif fonts (Times New Roman)")
print("- Publication-quality 600 dpi resolution")
print("- Elegant color palette with good contrast")
print("- PDF versions for vector graphics in LaTeX")
print("- Enhanced legends with shadows and frames")
print("- Value labels on bars for precise reading")
print("- Statistical significance brackets")
print("- Grid styling optimized for print")
print("- Proper spacing and margins")