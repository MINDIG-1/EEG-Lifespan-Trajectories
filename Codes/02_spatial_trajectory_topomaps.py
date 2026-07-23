"""
Spatial Trajectory Topomaps
---------------------------
This script generates global topographic maps (topomaps) illustrating the spatial 
distribution and relative strength of age-related EEG trajectories. It clusters 
features by domain (Complexity, Spectral, Morphological) and applies a split-sign 
sorting mechanism to separate early/midlife developmental models.

Inputs:
    - Analyzed feature trajectories with covariates (CSV)
    - Channel spatial mapping definitions (Excel)
    - Feature category definitions (CSV)

Outputs:
    - Master spatial topomap figure (PNG)
"""

import os
import re
import textwrap
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import mne

# =========================================================
# 1. SETUP & CONFIGURATION
# =========================================================
# --- Global Constants ---
SIG_THRESHOLD = 0.95
COLOR_PURPLE = "#5A3286"
COLOR_TEAL = "#2B7A8E"

# --- MNE & Warnings ---
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING') # Suppresses noisy MNE console output during rendering

# --- Directory Setup ---
DATA_DIR = os.path.join('data')
OUTPUT_DIR = os.path.join('outputs', 'Spatial_Topomaps')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Aesthetics ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 1.5

# =========================================================
# 2. LOAD & PREPARE DATA
# =========================================================
data_file = os.path.join(DATA_DIR, 'Feature_Cleaned_interactive_Trajectories_Covariate_Sex.csv')
mapping_feat_file = os.path.join(DATA_DIR, 'All_feats_categories.csv')
mapping_chan_file = os.path.join(DATA_DIR, 'to_19_channels.xlsx')

try:
    df = pd.read_csv(data_file)
    mappingfeatures_df = pd.read_csv(mapping_feat_file, encoding='latin1')
    mapping_df = pd.read_excel(mapping_chan_file)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing required data files. Please ensure data is in the '{DATA_DIR}' directory. {e}")

# Filter and extract base feature/channel names
df = df[df['Best_Weight'] >= 0.0].copy() 
extracted = df['Feature'].str.extract(r'^(.*?)[-_]?(ch\d+|average|avgch|avg)$', flags=re.IGNORECASE, expand=True)
df['Base_Feature'] = extracted[0]
df['Raw_Ch'] = extracted[1].str.lower()
df['Abs_Strength'] = np.abs(df['Relative_Strength'])

# Map strictly to the 10-20 standard 19-channel system
channel_map = dict(zip(mapping_df['Channel'].astype(str).str.strip().str.lower(), mapping_df['Standard_19'].astype(str).str.strip()))
df['Channel'] = df['Raw_Ch'].map(channel_map).fillna(df['Raw_Ch'])
df.loc[df['Raw_Ch'].isin(['average', 'avgch', 'avg']), 'Channel'] = 'Avg'

# =========================================================
# 3. COLORMAP DEFINITION
# =========================================================
nodes = [0.0, 0.425, 0.49, 0.51, 0.575, 1.0]
cmap_colors = [COLOR_PURPLE, "#BFADD3", "#FFFFFF", "#FFFFFF", "#B9D7DF", COLOR_TEAL] 
custom_cmap = mcolors.LinearSegmentedColormap.from_list("PurpWhiteTeal", list(zip(nodes, cmap_colors)))
norm = mpl.colors.Normalize(vmin=-1, vmax=1)

# =========================================================
# 4. GLOBAL-PRIORITY SPLIT-SIGN SORTING
# =========================================================
domains = ['Complexity', 'Spectral', 'Morphological']
row_feature_map = {}

for domain in domains:
    df_dom = df[df['Category'] == domain].copy()
    summary_data = []
    
    for feat, group in df_dom.groupby('Base_Feature'):
        max_weight_anywhere = group['Best_Weight'].max()
        avg_row = group[group['Channel'] == 'Avg']
        
        if not avg_row.empty:
            avg_strength = avg_row['Abs_Strength'].values[0]
            avg_weight = avg_row['Best_Weight'].values[0]
            global_beta = avg_row['Effect_Size_Beta'].values[0]
        else:
            avg_strength, avg_weight = 0, 0
            global_beta = group.loc[group['Abs_Strength'].idxmax(), 'Effect_Size_Beta']
            
        summary_data.append({
            'Base_Feature': feat,
            'Max_Weight_Anywhere': max_weight_anywhere,
            'Avg_Strength': avg_strength, 
            'Avg_Weight': avg_weight,
            'Global_Beta': global_beta
        })
        
    df_sum = pd.DataFrame(summary_data)
    df_valid = df_sum[df_sum['Max_Weight_Anywhere'] >= SIG_THRESHOLD]
    
    # Split into Negative (Inverse) and Positive (Quadratic) priority lists
    neg_feats = df_valid[df_valid['Global_Beta'] < 0].sort_values(['Avg_Strength', 'Avg_Weight'], ascending=[False, False]).head(7)['Base_Feature'].tolist()
    pos_feats = df_valid[df_valid['Global_Beta'] >= 0].sort_values(['Avg_Strength', 'Avg_Weight'], ascending=[False, False]).head(7)['Base_Feature'].tolist()
    
    # Pad lists to ensure uniform grid sizing (7 slots each + 1 center divider)
    neg_feats += [None] * (7 - len(neg_feats))
    pos_feats += [None] * (7 - len(pos_feats))
    row_feature_map[domain] = neg_feats + [None] + pos_feats

# =========================================================
# 5. MASTER PLOTTING
# =========================================================
print("Rendering topomaps (this may take a moment)...")

fig = plt.figure(figsize=(30, 15))
gs = GridSpec(3, 15, wspace=0.01, hspace=0.3, top=0.78) 

def style_icon_ax(ax):
    """Helper function to style the descriptive trajectory icons."""
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']: ax.spines[spine].set_linewidth(2); ax.spines[spine].set_color('#7f8c8d')

# Draw the legend icons at the top of the figure
x_dummy = np.linspace(0, 1, 50)

ax_i3 = fig.add_axes([0.14, 0.84, 0.03, 0.04])
ax_i3.plot(x_dummy, 1 - np.exp(-5 * x_dummy), color=COLOR_PURPLE, lw=3)
style_icon_ax(ax_i3); fig.text(0.18, 0.86, "Steep Early Rise", va='center', fontsize=14)

ax_i1 = fig.add_axes([0.58, 0.84, 0.03, 0.04])
ax_i1.plot(x_dummy, np.exp(-4 * x_dummy), color=COLOR_PURPLE, lw=3)
style_icon_ax(ax_i1); fig.text(0.62, 0.86, "Steep Early Decay", va='center', fontsize=14)

ax_i2 = fig.add_axes([0.31, 0.84, 0.03, 0.04])
ax_i2.plot(x_dummy, -4*(x_dummy-0.5)**2 + 1, color=COLOR_TEAL, lw=3)
style_icon_ax(ax_i2); fig.text(0.35, 0.86, "Midlife Decay\n(Inverted U-Shape)", va='center', fontsize=14)

ax_i4 = fig.add_axes([0.75, 0.84, 0.03, 0.04])
ax_i4.plot(x_dummy, 4*(x_dummy-0.5)**2, color=COLOR_TEAL, lw=3)
style_icon_ax(ax_i4); fig.text(0.79, 0.86, "Midlife Restoration\n(U-shape)", va='center', fontsize=14)

# Define Spatial Configurations for MNE
spatial_channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
mne_ch_mapping = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
mne_ch_names = [mne_ch_mapping.get(ch, ch) for ch in spatial_channels]
info = mne.create_info(ch_names=mne_ch_names, sfreq=200, ch_types='eeg')
info.set_montage(mne.channels.make_standard_montage('standard_1005'))
name_map = dict(zip(mappingfeatures_df['Feature_Code'], mappingfeatures_df['Base_Feature Name']))

for row_idx, domain in enumerate(domains):
    domain_features = row_feature_map[domain]
    fig.text(0.10, 0.73 - (row_idx * 0.28), f"{domain}", fontsize=22, fontweight='bold', ha='right', va='center')
    
    for col_idx, feat in enumerate(domain_features):
        ax_head = fig.add_subplot(gs[row_idx, col_idx])
        if feat is None: 
            ax_head.axis('off')
            if col_idx != 7: ax_head.text(0.5, 0.5, 'No features\nidentified', ha='center', va='center', fontsize=13, color='#95a5a6', fontstyle='italic')
            continue
        
        # --- SPATIAL TOPOMAP LOGIC (Masking Low Confidence Channels) ---
        data_array = np.zeros(len(spatial_channels))
        mask_array = np.zeros(len(spatial_channels), dtype=bool) 
        
        for j, ch in enumerate(spatial_channels):
            row_data = df[(df['Base_Feature'] == feat) & (df['Channel'] == ch)]
            
            if not row_data.empty and row_data.iloc[0]['Best_Weight'] >= SIG_THRESHOLD:
                s, m = row_data.iloc[0]['Relative_Strength'], row_data.iloc[0]['Class']
                intensity = min(max(0.15 + s, 0), 1)
                data_array[j] = -intensity if m == 'Inverse' else intensity
            else:
                mask_array[j] = True # Mark for 'X' drawing

        mne.viz.plot_topomap(
            data_array, info, axes=ax_head, show=False, cmap=custom_cmap, vlim=(-1, 1), contours=4,
            sphere=(0.0, 0.0, 0.0, 0.11), extrapolate='head', mask=mask_array, 
            mask_params=dict(marker='x', markeredgecolor='black', markeredgewidth=1.5, markersize=6)
        )
        ax_head.set_title(textwrap.fill(str(name_map.get(feat, feat)), width=15), fontsize=10, pad=8)
        
        # --- GLOBAL AVG BAR (With 'X' Marker for low significance) ---
        ax_avg = ax_head.inset_axes([0.1, -0.15, 0.8, 0.08])
        ax_avg.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='#bdc3c7', lw=0.8, linestyle='--'))
        
        avg_row = df[(df['Base_Feature'] == feat) & (df['Channel'] == 'Avg')]
        
        if not avg_row.empty:
            asig, acls = avg_row.iloc[0]['Relative_Strength'], avg_row.iloc[0]['Class']
            ai = min(max(0.15 + asig, 0), 1)
            av = -ai if acls == 'Inverse' else ai
            ax_avg.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=custom_cmap(norm(av)), edgecolor='black', lw=0, clip_on=False))
            
            if avg_row.iloc[0]['Best_Weight'] < SIG_THRESHOLD:
                ax_avg.plot([0, 1], [0, 1], color='black', lw=1.5, clip_on=False)
                ax_avg.plot([0, 1], [1, 0], color='black', lw=1.5, clip_on=False)
        else:
            ax_avg.plot([0, 1], [0, 1], color='black', lw=1.5, clip_on=False)
            ax_avg.plot([0, 1], [1, 0], color='black', lw=1.5, clip_on=False)
        
        if row_idx == 0 and col_idx == 0:
            ax_avg.text(-0.15, 0.5, "Global Avg", transform=ax_avg.transAxes, ha='right', va='center', fontsize=11, fontweight='bold')
        ax_avg.axis('off')

# =========================================================
# 6. AESTHETIC GRIDS & BACKGROUND PANELS
# =========================================================
fig.patches.extend([
    patches.Rectangle((0.11, 0.12), 0.38, 0.68, transform=fig.transFigure, facecolor=COLOR_PURPLE, alpha=0.03, zorder=-10),
    patches.Rectangle((0.53, 0.12), 0.38, 0.68, transform=fig.transFigure, facecolor=COLOR_TEAL, alpha=0.03, zorder=-10)
])
fig.add_artist(Line2D([0.05, 0.95], [0.81, 0.81], transform=fig.transFigure, color='#7f8c8d', lw=2.5))
fig.add_artist(Line2D([0.51, 0.51], [0.12, 0.81], transform=fig.transFigure, color='#bdc3c7', lw=2, linestyle='--'))
for y in [0.59, 0.31]: fig.add_artist(Line2D([0.08, 0.92], [y, y], transform=fig.transFigure, color='#ecf0f1', lw=2))

cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.015])
mpl.colorbar.ColorbarBase(cbar_ax, cmap=custom_cmap, norm=norm, orientation='horizontal').set_ticks([])
cbar_ax.text(0.5, 1.5, 'Normalized Effect Magnitude', ha='center', fontsize=14, fontweight='bold', transform=cbar_ax.transAxes)

# Dedicated Legend Box
legend_elements = [
    patches.Patch(color=COLOR_TEAL, label='Quadratic Model'),
    patches.Patch(color=COLOR_PURPLE, label='Inverse Model'),
    Line2D([0], [0], marker='x', color='black', lw=0, markeredgewidth=2, markersize=10, label='No Significance')
]
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.98), 
           fontsize=14, title='Significance & Topology', title_fontsize=16, 
           frameon=True, facecolor='#f9f9f9', edgecolor='#bdc3c7')

# Save Master Output
plot_filename = os.path.join(OUTPUT_DIR, 'Master_Sign_Split_Topomaps.png')
plt.savefig(plot_filename, dpi=600, bbox_inches='tight')
print(f"Complete. Master figure successfully saved to: {plot_filename}")