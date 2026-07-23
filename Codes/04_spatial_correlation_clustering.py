"""
Spatial Correlation Matrix & Hierarchical Clustering
----------------------------------------------------
This script imports pre-computed developmental turning points, calculates strict 
pairwise spatial correlations across 19 standard EEG channels, and performs 
hierarchical clustering to group structurally similar trajectories.

Inputs:
    - Pre-computed sex-stratified turning points (CSV)
    - Feature trajectories with model weights for significance filtering (CSV)
    - Channel and feature mapping definitions (Excel/CSV)

Outputs:
    - Master annotated correlation matrix with exemplar topomaps (PNG)
    - Detailed feature topomap grids per cluster (PNG)
"""

import os
import math
import textwrap
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from scipy.stats import pearsonr
import scipy.cluster.hierarchy as sch
import mne

# =========================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================
# --- Analytical Parameters ---
TARGET_CLUSTERS = 12 
THRESHOLD_WEIGHT = 0.95

# --- Directory Paths (Repository Standard) ---
DATA_DIR = os.path.join('data')
OUTPUT_DIR = os.path.join('outputs', 'Correlation_Clustering')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input Files
TURNING_POINTS_FILE = os.path.join(DATA_DIR, 'Robust_Turning_Points_SexStratified.csv')
AIC_FILE = os.path.join(DATA_DIR, 'Feature_Cleaned_interactive_Trajectories_Covariate_Sex.csv')
CHANNEL_MAP_FILE = os.path.join(DATA_DIR, 'to_19_channels.xlsx')
FEATURE_MAP_FILE = os.path.join(DATA_DIR, 'All_feats_categories.csv')

# --- MNE & Warnings ---
warnings.filterwarnings('ignore', category=RuntimeWarning) # Suppress correlation warnings for flat arrays
mne.set_log_level('WARNING') # Suppress noisy MNE rendering outputs

# --- Professional Aesthetics ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42 

channels_19 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

# =========================================================================
# 2. LOAD DATA & MAPPINGS (Significance Filtering)
# =========================================================================
print("Loading datasets and filtering masks...")
try:
    df_aic = pd.read_csv(AIC_FILE)
    mapping_df = pd.read_excel(CHANNEL_MAP_FILE)
    mappingfeatures_df = pd.read_csv(FEATURE_MAP_FILE, encoding='latin1')
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing required data files. Please ensure data is in the '{DATA_DIR}' directory. {e}")

# Build Feature and Channel Maps
feature_name_map = dict(zip(mappingfeatures_df['Feature_Code'], mappingfeatures_df['Base_Feature Name']))
channel_map = dict(zip(mapping_df['Channel'].astype(str).str.strip().str.lower(), mapping_df['Standard_19'].astype(str).str.strip()))

# Extract and map names in the AIC file to create a mask of valid features
extracted = df_aic['Feature'].str.extract(r'^(.*?)[-_]?(ch\d+|average|avgch|avg)$', flags=re.IGNORECASE, expand=True)
df_aic['Base_Feature'] = extracted[0].map(feature_name_map).fillna(extracted[0])
df_aic['Channel'] = extracted[1].str.lower().map(channel_map).fillna(extracted[1].str.lower())

# Identify strictly significant feature-channel pairs
valid_channels = df_aic[
    (df_aic['Best_Weight'] >= THRESHOLD_WEIGHT) & 
    (~df_aic['Class'].isin(['Linear', 'Invariant', 'Null', 'Constant', 'None'])) &
    (df_aic['Channel'].isin(channels_19))
]
valid_pairs = valid_channels[['Base_Feature', 'Channel']].drop_duplicates()

# =========================================================================
# 3. INGEST ROBUST TURNING POINTS 
# =========================================================================
print("Processing pre-computed turning points...")
df_turning = pd.read_csv(TURNING_POINTS_FILE)

# Standardize Feature and Channel names to match our internal pipeline
df_turning['Base_Feature'] = df_turning['Feature'].map(feature_name_map).fillna(df_turning['Feature'])

# Average the critical ages between Males and Females
df_metrics = df_turning.groupby(['Base_Feature', 'Channel'])['Critical_Age'].mean().reset_index()
df_metrics.rename(columns={'Critical_Age': 'Critical_age'}, inplace=True)

# Enforce biological boundaries and significance mask
df_metrics = df_metrics[(df_metrics['Critical_age'] >= 5) & (df_metrics['Critical_age'] <= 85)]
df_metrics = df_metrics.merge(valid_pairs, on=['Base_Feature', 'Channel'], how='inner')

# =========================================================================
# 4. STRICT PAIRWISE CORRELATION (Intersect Only)
# =========================================================================
print("\n--- Processing Strict Pairwise Spatial Correlations ---")
feature_space_df = df_metrics.pivot(index='Channel', columns='Base_Feature', values='Critical_age').reindex(channels_19)
X_raw = feature_space_df.T.values
feature_names = np.array(feature_space_df.columns.tolist())

feature_means, feature_stds = np.nanmean(X_raw, axis=1, keepdims=True), np.nanstd(X_raw, axis=1, keepdims=True)
feature_stds[feature_stds == 0] = 1.0 
X_normalized = (X_raw - feature_means) / feature_stds

valid_channel_counts = np.sum(~np.isnan(X_normalized), axis=1)
robust_mask = valid_channel_counts >= 11
X_robust = X_normalized[robust_mask, :]
features_robust = feature_names[robust_mask].tolist()
n_features = len(features_robust)

print("Calculating exact Pearson correlations based strictly on overlapping channels...")
r_matrix = np.zeros((n_features, n_features))
p_matrix = np.ones((n_features, n_features))
dist_matrix_condensed = []

for i in range(n_features):
    for j in range(i+1, n_features): 
        v1, v2 = X_robust[i], X_robust[j]
        valid_mask = ~np.isnan(v1) & ~np.isnan(v2)
        
        if np.sum(valid_mask) >= 11:
            r, p = pearsonr(v1[valid_mask], v2[valid_mask])
            if np.isnan(r): r, p = 0.0, 1.0
        else:
            r, p = 0.0, 1.0 
            
        r_matrix[i, j] = r_matrix[j, i] = r
        p_matrix[i, j] = p_matrix[j, i] = p
        
        distance = 1.0 - r if p < 0.05 else 2.0
        dist_matrix_condensed.append(distance)

np.fill_diagonal(r_matrix, 1.0)
np.fill_diagonal(p_matrix, 0.0)

# =========================================================================
# 5. HARDCODED HIERARCHICAL CLUSTERING & ORPHAN FILTERING
# =========================================================================
print(f"\n--- Extracting {TARGET_CLUSTERS} Hardcoded Structural Clusters ---")
linkage_matrix = sch.linkage(dist_matrix_condensed, method='average')
cluster_labels = sch.fcluster(linkage_matrix, t=TARGET_CLUSTERS, criterion='maxclust')

sort_idx = sch.leaves_list(linkage_matrix)
sorted_features = [features_robust[i] for i in sort_idx]
sorted_r_matrix = r_matrix[sort_idx, :][:, sort_idx]
sorted_p_matrix = p_matrix[sort_idx, :][:, sort_idx]
sorted_labels = cluster_labels[sort_idx]

sorted_r_matrix_sig = np.where(sorted_p_matrix < 0.05, sorted_r_matrix, np.nan)

cluster_blocks = []
for cid in np.unique(sorted_labels):
    indices = np.where(sorted_labels == cid)[0]
    if len(indices) < 2: continue # Ignore orphans
        
    start_idx, end_idx = indices[0], indices[-1]
    
    intra_cluster_r = sorted_r_matrix[indices, :][:, indices]
    mean_corrs = np.nanmean(intra_cluster_r, axis=1)
    exemplar_local_idx = np.argmax(mean_corrs)
    exemplar_global_idx = indices[exemplar_local_idx]
    colors_pool = matplotlib.colormaps['tab10'](np.linspace(0, 1, max(len(cluster_blocks), 2)))    
    
    cluster_blocks.append({
        'id': cid, 'start': start_idx, 'end': end_idx, 'size': len(indices),
        'exemplar_name': sorted_features[exemplar_global_idx],
        'exemplar_data': X_robust[sort_idx[exemplar_global_idx]],
        'all_names': [sorted_features[i] for i in indices],
        'all_data': [X_robust[sort_idx[i]] for i in indices],
        'color': colors_pool[len(cluster_blocks) % len(colors_pool)] 
    })

print(f"--> Extracted {len(cluster_blocks)} robust multi-feature networks.")

# =========================================================================
# 6. VISUALIZATION (Matrix + Triangular Borders + Diagonal Topomaps)
# =========================================================================
print("Rendering Master Annotated Figure...")

fig = plt.figure(figsize=(20, 20), dpi=300)
ax_heat = fig.add_axes([0.05, 0.1, 0.75, 0.8])

mask = np.triu(np.ones_like(sorted_r_matrix_sig, dtype=bool), k=1)

# CUSTOM DIVERGING COLORMAP: Charcoal (-1) to Pure White (0) to Premium Gold (+1)
colors_charcoal_gold = ["#d4af37", "#ffffff", "#2f3640"] 
custom_diverge = mcolors.LinearSegmentedColormap.from_list("CharcoalGold", colors_charcoal_gold)

sns.heatmap(
    sorted_r_matrix_sig, mask=mask, cmap=custom_diverge, center=0, vmin=-1, vmax=1,
    xticklabels=sorted_features, yticklabels=sorted_features,
    linewidths=0.5, square=True, ax=ax_heat,
    cbar_kws={'label': 'Pearson Correlation (r), p-val<0.05', 'shrink': 0.4, 'pad': 0.04}
)
cbar = ax_heat.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Pearson Correlation (r), p-val<0.05', fontsize=16)
ax_heat.set_title(f" ", fontsize=18, fontweight='bold', pad=20)
ax_heat.tick_params(axis='x', rotation=90, labelsize=9)
ax_heat.tick_params(axis='y', rotation=0, labelsize=9)

# MNE Setup
montage = mne.channels.make_standard_montage('standard_1020')
info = mne.create_info(ch_names=channels_19, sfreq=100, ch_types='eeg')
info.set_montage(montage)
mask_params = dict(marker='x', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
colors_pool = matplotlib.colormaps['tab10'](np.linspace(0, 1, max(len(cluster_blocks), 2)))

bbox = ax_heat.get_position()

# 6A: PARALLEL DIAGONAL ALIGNMENT CALCULATION
map_size = 0.09 
min_dist = map_size + 0.02 

positions = []
for block in cluster_blocks:
    start, end = block['start'], block['end']
    width = end - start + 1
    c = start + (width / 2.0)
    
    # Project parallel to the diagonal
    offset = n_features * 0.12  
    data_x = c + offset 
    data_y = c - offset               
    
    fig_x = bbox.x0 + (data_x / n_features) * bbox.width
    fig_y = bbox.y1 - (data_y / n_features) * bbox.height 
    
    positions.append({
        'block': block, 'fig_x': fig_x, 'fig_y': fig_y, 
        'start': start, 'end': end, 'width': width, 'c': c
    })

# Parallel Diagonal Anti-Collision
for i in range(1, len(positions)):
    prev_y = positions[i-1]['fig_y']
    curr_y = positions[i]['fig_y']
    
    if prev_y - curr_y < min_dist:
        shift_amount = min_dist - (prev_y - curr_y)
        positions[i]['fig_y'] -= shift_amount
        positions[i]['fig_x'] += shift_amount 

# 6B: DRAWING PASS
global_im = None 

for i, pos in enumerate(positions):
    block = pos['block']
    color = colors_pool[i]
    start, end, width = pos['start'], pos['end'], pos['width']
    
    # TRIANGULAR BORDER
    vertices = [
        (start, start),           
        (start, end + 1),         
        (end + 1, end + 1)        
    ]
    triangle = patches.Polygon(vertices, closed=True, linewidth=3, edgecolor=color, facecolor='none', zorder=10)
    ax_heat.add_patch(triangle)
    
    adjusted_data_x = (pos['fig_x'] - bbox.x0) * n_features / bbox.width
    adjusted_data_y = (bbox.y1 - pos['fig_y']) * n_features / bbox.height
    
    # Add Topomap Axes
    ax_topo = fig.add_axes([pos['fig_x'] - map_size/2, pos['fig_y'] - map_size/2, map_size, map_size])
    
    feat_data = block['exemplar_data']
    insignificant_mask = np.isnan(feat_data)
    feat_data_plot = feat_data.copy()
    feat_data_plot[insignificant_mask] = 0.0
    
    global_im, _ = mne.viz.plot_topomap(
        feat_data_plot, info, axes=ax_topo, show=False, 
        cmap='Blues', vlim=(-2.5, 2.5), contours=4,
        mask=insignificant_mask, mask_params=mask_params, size=0.95
    )
    
    wrapped_name = "\n".join(textwrap.wrap(block['exemplar_name'], width=18))
    ax_topo.set_title(f"{wrapped_name}", fontsize=10, fontweight='bold', color=color, pad=4)
    
    diagonal_center = start + (width / 2.0)
    ax_heat.annotate('', xy=(diagonal_center, diagonal_center), xytext=(adjusted_data_x, adjusted_data_y),
                     arrowprops=dict(arrowstyle="-", color=color, lw=2, linestyle='dotted'), 
                     annotation_clip=False)

# 6C: SLEEK GLOBAL COLORBAR FOR TOPOMAPS
if global_im is not None:
    lowest_y = positions[-1]['fig_y'] - (map_size / 2) - 0.04
    col_x = positions[-1]['fig_x'] + 0.04
    
    cbar_width = 0.08
    cbar_height = 0.008
    cbar_ax = fig.add_axes([col_x - cbar_width/2, lowest_y, cbar_width, cbar_height])
    
    cb = fig.colorbar(global_im, cax=cbar_ax, orientation='horizontal')
    cb.set_ticks([-2.5, 0, 2.5])
    cb.ax.set_xticklabels(['Early', ' ', 'Late'], fontsize=14, fontweight='bold')
    cb.outline.set_linewidth(1.0)

output_path = os.path.join(OUTPUT_DIR, "Parallel_Diagonal_Annotated_Matrix.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
plt.close()

print(f"\n[SUCCESS] Parallel Diagonal figure generated and saved to:\n--> {output_path}")

# =========================================================================
# 7. GENERATE FEATURE GRIDS FOR EACH CLUSTER
# =========================================================================
print("\n--- Generating Topomap Grids for Each Cluster ---")

for block in cluster_blocks:
    cid = block['id']
    names = block['all_names']
    data_list = block['all_data']
    cluster_color = block['color']
    n_items = len(names)
    
    cols = min(4, n_items)
    rows = math.ceil(n_items / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5), dpi=300)
    fig.suptitle(f"Cluster {cid} Topographies (n={n_items})", fontsize=20, fontweight='bold', color=cluster_color, y=1.02)
    
    if n_items == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
    global_grid_im = None
    
    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < n_items:
            feat_data = data_list[i]
            insignificant_mask = np.isnan(feat_data)
            feat_data_plot = feat_data.copy()
            feat_data_plot[insignificant_mask] = 0.0
            
            global_grid_im, _ = mne.viz.plot_topomap(
                feat_data_plot, info, axes=ax, show=False, 
                cmap='Blues', vlim=(-2.5, 2.5), contours=4,
                mask=insignificant_mask, mask_params=mask_params
            )
            
            wrapped_name = "\n".join(textwrap.wrap(names[i], width=20))
            ax.set_title(wrapped_name, fontsize=11, fontweight='bold', pad=8)
        else:
            ax.axis('off') 
            
    if global_grid_im is not None:
        plt.tight_layout(rect=[0, 0.08, 1, 1]) 
        cbar_ax = fig.add_axes([0.3, 0.03, 0.4, 0.02]) 
        cb = fig.colorbar(global_grid_im, cax=cbar_ax, orientation='horizontal')
        cb.set_ticks([-2.5, 0, 2.5])
        cb.ax.set_xticklabels(['Early', ' ', 'Late'], fontsize=12, fontweight='bold')
        cb.outline.set_linewidth(1.0)
    else:
        plt.tight_layout()
    
    grid_path = os.path.join(OUTPUT_DIR, f"Cluster_{cid}_Detailed_Grid.png")
    plt.savefig(grid_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Saved grid for Cluster {cid} ({n_items} features) -> {grid_path}")

print("\n[SUCCESS] All analyses and figures completed.")