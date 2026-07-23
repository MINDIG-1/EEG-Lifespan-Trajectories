"""
Sex-Stratified Developmental Turning Points Generator
---------------------------------------------------
This script evaluates high-confidence age-related EEG trajectories and recalculates 
Ordinary Least Squares (OLS) regressions on the raw dataset. It computes the 
mathematical derivatives of Quadratic and Inverse models to pinpoint exact critical 
ages (turning points or stabilization points) for both males and females.

Inputs:
    - Analyzed feature trajectories with model weights (CSV)
    - Channel spatial mapping definitions (Excel)
    - Raw harmonized EEG dataset (CSV)

Outputs:
    - Robust Turning Points dataset (CSV)
    - Composite density and distributional scatter figure (PNG/SVG)
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import ks_2samp

# =========================================================
# 1. SETUP & CONFIGURATION
# =========================================================
warnings.filterwarnings('ignore')

# --- Analytical Thresholds ---
STRENGTH_THRESHOLD = 0.0
WEIGHT_THRESHOLD = 0.95

# --- Directory Paths (Repository Standard) ---
DATA_DIR = os.path.join('data')
OUTPUT_DIR = os.path.join('outputs', 'Turning_Points')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input Files
AIC_FILE = os.path.join(DATA_DIR, 'Feature_Cleaned_interactive_Trajectories_Covariate_Sex.csv')
CHANNEL_MAP_FILE = os.path.join(DATA_DIR, 'to_19_channels.xlsx')
RAW_EEG_FILE = os.path.join(DATA_DIR, 'All_feats_new_EC_CLEANED.csv')

# Output Files
CSV_OUT = os.path.join(OUTPUT_DIR, 'Robust_Turning_Points_SexStratified.csv')
PNG_OUT = os.path.join(OUTPUT_DIR, 'Nature_DensityTop_ScatterViolin_SexStratified.png')
SVG_OUT = os.path.join(OUTPUT_DIR, 'Nature_DensityTop_ScatterViolin_SexStratified.svg')

# --- Professional Aesthetics ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

SEX_PALETTE = {'Female': '#CD6155', 'Male': '#2980b9'}
SEX_ORDER = ['Female', 'Male']
CHANNELS_19 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

# =========================================================
# 2. LOAD DATA & MAPPINGS
# =========================================================
print("Loading datasets and mapping definitions...")
try:
    df = pd.read_csv(AIC_FILE)
    mapping_df = pd.read_excel(CHANNEL_MAP_FILE)
    df_raw = pd.read_csv(RAW_EEG_FILE)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing required data file. Ensure it is in the '{DATA_DIR}' directory. {e}")

channel_map = dict(zip(
    mapping_df['Channel'].astype(str).str.strip().str.lower(),
    mapping_df['Standard_19'].astype(str).str.strip()
))

# =========================================================
# 3. FILTER & EXTRACT CHANNELS
# =========================================================
df['Filtered_Strength'] = df['Relative_Strength']
df.loc[df['Relative_Strength'] < STRENGTH_THRESHOLD, 'Filtered_Strength'] = 0
df.loc[df['Best_Weight'] < WEIGHT_THRESHOLD, 'Filtered_Strength'] = 0

extracted = df['Feature'].str.extract(r'^(.*?)[-_]?(ch\d+|average|avgch|avg)$', flags=re.IGNORECASE, expand=True)
df['Base_Feature'] = extracted[0]
df['Raw_Ch'] = extracted[1].str.lower()

df['Channel'] = df['Raw_Ch'].map(channel_map).fillna(df['Raw_Ch'])
df.loc[df['Raw_Ch'].isin(['average', 'avg', 'avgch']), 'Channel'] = 'Avg'

# Remove average channel
df = df[df['Channel'] != 'Avg']

print("\nExtracting features that passed topographic thresholds...")
active_df = df[df['Filtered_Strength'] > 0].copy()
idx = (active_df.groupby('Feature')['Filtered_Strength'].transform('max') == active_df['Filtered_Strength'])
final_features = active_df[idx]

print(f"--> Found {len(final_features)} high-confidence features for transition analysis.")

# =========================================================
# 4. ROBUST TURNING POINT IDENTIFICATION (OLS & Derivatives)
# =========================================================
transition_results = []
age_range = np.linspace(5, 85, 1000)

for _, row in final_features.iterrows():
    feature, target_col = row['Base_Feature'], row['Feature']
    best_model, category = row['Best_Model'], row['Category']

    if target_col not in df_raw.columns:
        continue

    # Determine Model Formula
    if 'Quadratic' in best_model:
        formula = 'y_target ~ (age + I(age**2)) * C(sex)' if 'Interactive' in best_model else 'y_target ~ age + I(age**2) + C(sex)'
    elif 'Inverse' in best_model:
        formula = 'y_target ~ I(1/age) * C(sex)' if 'Interactive' in best_model else 'y_target ~ I(1/age) + C(sex)'
    else:
        continue

    # Isolate data and fit model safely
    df_sub = df_raw[['age', 'sex', target_col]].dropna().copy()
    df_sub.rename(columns={target_col: 'y_target'}, inplace=True)

    try:
        model = smf.ols(formula, data=df_sub).fit()
    except Exception:
        continue # Skip if OLS fails to converge

    preds_F = model.predict(pd.DataFrame({'age': age_range, 'sex': ['F'] * 1000})).values
    preds_M = model.predict(pd.DataFrame({'age': age_range, 'sex': ['M'] * 1000})).values

    # Determine Critical Ages based on curve derivatives
    if 'Quadratic' in best_model:
        crit_F = age_range[np.argmin(np.abs(np.gradient(preds_F, age_range)))]
        crit_M = age_range[np.argmin(np.abs(np.gradient(preds_M, age_range)))]
        tt_F = tt_M = "Reversal (Inc to Dec / Dec to Inc)"
        
    elif 'Inverse' in best_model:
        sigma_noise = np.sqrt(model.mse_resid)
        threshold = sigma_noise / np.sqrt(len(df_sub))
        
        stable_F = np.where(np.abs(np.gradient(preds_F, age_range)) < threshold)[0]
        stable_M = np.where(np.abs(np.gradient(preds_M, age_range)) < threshold)[0]
        
        crit_F = age_range[stable_F[0]] if len(stable_F) > 0 else np.nan
        crit_M = age_range[stable_M[0]] if len(stable_M) > 0 else np.nan
        tt_F = tt_M = "Statistical Stabilization"

    # Save bounded results
    if not np.isnan(crit_F) and 5 <= crit_F <= 85:
        transition_results.append({'Feature': feature, 'Channel': row['Channel'], 'Category': category, 
                                   'Model': best_model, 'Sex': 'Female', 'Transition_Type': tt_F, 'Critical_Age': crit_F})
    if not np.isnan(crit_M) and 5 <= crit_M <= 85:
        transition_results.append({'Feature': feature, 'Channel': row['Channel'], 'Category': category, 
                                   'Model': best_model, 'Sex': 'Male', 'Transition_Type': tt_M, 'Critical_Age': crit_M})

# Save Turning Points
df_tp = pd.DataFrame(transition_results)
df_tp.to_csv(CSV_OUT, index=False)
print(f"--> Saved robust turning points to: {CSV_OUT}")

# =========================================================
# 5. PUBLICATION-STYLE VISUALIZATION
# =========================================================
print("\nGenerating Sex-Stratified Publication-Ready Visualization...")

fig = plt.figure(figsize=(20, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.25)

ax_kde = fig.add_subplot(gs[0])
ax_violins = fig.add_subplot(gs[1], sharex=ax_kde)

# --- 5A. KDE PANEL ---
sns.kdeplot(
    data=df_tp, x='Critical_Age', hue='Sex', hue_order=SEX_ORDER, palette=SEX_PALETTE,
    fill=True, alpha=0.2, linewidth=2.5, bw_adjust=0.7, ax=ax_kde
)

# Statistical Comparison (Kolmogorov-Smirnov)
ages_F = df_tp[df_tp['Sex'] == 'Female']['Critical_Age']
ages_M = df_tp[df_tp['Sex'] == 'Male']['Critical_Age']

if len(ages_F) > 0 and len(ages_M) > 0:
    ks_stat, ks_p = ks_2samp(ages_F, ages_M)
    median_F, median_M = np.median(ages_F), np.median(ages_M)
    
    ks_p_str = f"{ks_p:.4f}" if ks_p >= 0.0001 else f"{ks_p:.2e}"
    stats_text = (f"Kolmogorov-Smirnov Test\nD = {ks_stat:.3f}\np = {ks_p_str}\n\n"
                  f"Female Median = {median_F:.1f} y\nMale Median = {median_M:.1f} y")

    ax_kde.text(
        0.75, 0.95, stats_text, transform=ax_kde.transAxes,
        fontsize=11, fontweight='bold', va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.92, edgecolor='#bdc3c7')
    )

    print("\n================ SEX DIFFERENCE STATISTICS ================")
    print(f"K-S Statistic (D): {ks_stat:.4f}")
    print(f"K-S p-value      : {ks_p:.4e}")
    print(f"Female Median TP : {median_F:.2f} years")
    print(f"Male Median TP   : {median_M:.2f} years")
    print("===========================================================")

# KDE Formatting
ax_kde.set_ylabel('Density', fontsize=13, fontweight='bold')
ax_kde.set_xlabel('')
ax_kde.set_xlim(5, 85)
ax_kde.set_ylim(bottom=0)
ax_kde.grid(True, axis='x', linestyle='--', alpha=0.5)
sns.despine(ax=ax_kde, left=True)
ax_kde.tick_params(axis='x', length=0, labelbottom=False)
ax_kde.set_yticks([])

# --- 5B. VIOLIN + STRIP PANEL ---
sns.stripplot(
    data=df_tp, x='Critical_Age', y='Sex', hue='Sex', order=SEX_ORDER, hue_order=SEX_ORDER, 
    palette=SEX_PALETTE, ax=ax_violins, size=6, alpha=0.6, jitter=0.15, 
    edgecolor='white', linewidth=0.5, legend=False, zorder=1
)

sns.violinplot(
    data=df_tp, x='Critical_Age', y='Sex', hue='Sex', order=SEX_ORDER, hue_order=SEX_ORDER, 
    palette=SEX_PALETTE, ax=ax_violins, inner='box', alpha=0.4, linewidth=1.5, cut=0, 
    zorder=2, legend=False
)

# Violin Formatting
ax_violins.set_xlabel('Age (Years)', fontsize=13, fontweight='bold', labelpad=10)
ax_violins.set_ylabel('Sex', fontsize=13, fontweight='bold')
ax_violins.tick_params(axis='y', labelsize=13, length=0)
ax_violins.set_xticks(np.arange(5, 90, 10))
ax_violins.tick_params(axis='x', labelsize=12)
ax_violins.grid(axis='x', linestyle='--', alpha=0.5)
ax_violins.grid(axis='y', linestyle='-', alpha=0.1)
sns.despine(ax=ax_violins, left=True)

# =========================================================
# 6. SAVE FIGURES
# =========================================================
plt.tight_layout()
plt.savefig(PNG_OUT, dpi=300, bbox_inches='tight')
plt.savefig(SVG_OUT, format='svg', bbox_inches='tight')

print(f"\n[SUCCESS] Figures successfully exported to '{OUTPUT_DIR}'")