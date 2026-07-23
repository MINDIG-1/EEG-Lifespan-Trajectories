"""
Multi-Model Inference & Trajectory Analysis
-------------------------------------------
This script performs Akaike Information Criterion (AIC) based Multi-Model Inference 
to identify fundamental age-related trajectories for neuroharmonized EEG features. 
It evaluates baseline additive models (Null, Linear, Quadratic, Inverse) and 
subsequently tests for sex-specific interactive effects.

Inputs:
    - Cleaned, harmonized EEG dataset (CSV)
    - Feature category mapping (CSV)

Outputs:
    - Comprehensive statistical summary of feature trajectories (CSV)
    - Sample trajectory plots for top features (PNG)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import zscore
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress non-critical statsmodels warnings for clean console output
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & LOAD DATA
# ==========================================
# Define relative paths for GitHub repository structure
DATA_DIR = os.path.join('data')
OUTPUT_DIR = os.path.join('outputs', 'AIC_Cleaned_interactive_features_sexcovars')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'Sample_Plots')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# File names (Update these if your repository data files change names)
DATA_FILE = os.path.join(DATA_DIR, 'Neuro_harmonized_outputEC_CLEANED.csv')
CATEGORY_FILE = os.path.join(DATA_DIR, 'All_feats_categories.csv')

# --- AESTHETICS FOR PLOTS ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# Load Data
try:
    df = pd.read_csv(DATA_FILE)
    df_ctg = pd.read_csv(CATEGORY_FILE)
    print(f"Successfully loaded datasets.")
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing required data files. Please ensure data is in the '{DATA_DIR}' directory. {e}")

# --- AUTO-DETECT & STANDARDIZE COLUMNS ---
# Standardize Age
for col in df.columns:
    if 'age' in col.lower().strip():
        df.rename(columns={col: 'Age'}, inplace=True)
        break

# Standardize Sex
possible_names = ['sex', 'gender', 'gen', 'm/f', 'group']
for col in df.columns:
    if col.lower().strip() in possible_names:
        df.rename(columns={col: 'Sex'}, inplace=True)
        break

# Standardize Categories
for col in df_ctg.columns:
    if 'category' in col.lower().strip():
        df_ctg.rename(columns={col: 'Category'}, inplace=True)
    if 'feature' in col.lower().strip():
        df_ctg.rename(columns={col: 'Feature'}, inplace=True)

# Isolate numeric EEG features
numeric_cols = df.select_dtypes(include=[np.number]).columns
features_list = [col for col in numeric_cols if col not in ['Age', 'Sex'] and 'unnamed' not in col.lower()]

print(f"Analyzing {len(features_list)} features using Hierarchical 2-Step Modeling...")
features_to_plot = features_list[:5] 
full_results = []      

# ==========================================
# 2. MAIN FEATURE LOOP
# ==========================================
for feature in features_list:
    
    # Isolate data, drop missing values, and calculate Z-scores efficiently
    subset = df[['Age', 'Sex', feature]].dropna().copy()
    subset['y_scaled'] = zscore(subset[feature])

    # --------------------------------------------------------
    # STEP 1: Trajectory Shape Selection (Additive Models Only)
    # --------------------------------------------------------
    base_formulas = {
        'Null':      'y_scaled ~ C(Sex)',
        'Linear':    'y_scaled ~ Age + C(Sex)',
        'Quadratic': 'y_scaled ~ Age + I(Age**2) + C(Sex)',
        'Inverse':   'y_scaled ~ I(1/Age) + C(Sex)'
    }

    aic_scores = {}
    base_models = {}

    for name, formula in base_formulas.items():
        try:
            res = smf.ols(formula, data=subset).fit()
            aic_scores[name] = res.aic
            base_models[name] = res
        except Exception:
            aic_scores[name] = np.inf

    # Calculate Akaike Weights strictly across the 4 base shapes
    min_aic = min(aic_scores.values())
    deltas = {k: v - min_aic for k, v in aic_scores.items()}
    rel_lik = {k: np.exp(-0.5 * v) for k, v in deltas.items()}
    sum_lik = sum(rel_lik.values())
    weights = {k: v / sum_lik for k, v in rel_lik.items()}

    base_shape = max(weights, key=weights.get)
    base_shape_weight = weights[base_shape]
    cls = 'Invariant' if base_shape == 'Null' else base_shape

    # --------------------------------------------------------
    # STEP 2: Dimorphism Evaluation (Extended Models)
    # --------------------------------------------------------
    sex_p_value, sex_beta = np.nan, np.nan
    effect_size_beta, effect_size_pvalue = np.nan, np.nan
    interaction_beta, interaction_pvalue = np.nan, np.nan
    effect_direction = "Invariant"
    best_weight = base_shape_weight

    if base_shape == 'Null':
        best_model = 'Null'
        winning_res = base_models['Null']
        sex_terms = [t for t in winning_res.params.index if 'Sex' in t]
        if sex_terms:
            sex_p_value = winning_res.pvalues[sex_terms[0]]
            sex_beta = winning_res.params[sex_terms[0]]
            
    else:
        # Retrieve the additive fit from Step 1
        add_res = base_models[base_shape]
        
        # Build and fit the Interactive version of the winning shape
        add_formula = base_formulas[base_shape]
        int_formula = add_formula.replace('+ C(Sex)', '* C(Sex)')
        
        try:
            int_res = smf.ols(int_formula, data=subset).fit()
            # Model selection: interactive vs additive
            if int_res.aic < add_res.aic:
                best_model = f"{base_shape}_Interactive"
                winning_res = int_res
            else:
                best_model = f"{base_shape}_Additive"
                winning_res = add_res
        except Exception:
            int_res = None
            best_model = f"{base_shape}_Additive"
            winning_res = add_res

        # Extract Main Effects (From Additive Model to avoid multicollinearity)
        sex_terms = [t for t in add_res.params.index if 'Sex' in t and ':' not in t]
        if sex_terms:
            sex_p_value = add_res.pvalues[sex_terms[0]]
            sex_beta = add_res.params[sex_terms[0]]

        if base_shape == 'Linear': base_term_str = 'Age'
        elif base_shape == 'Quadratic': base_term_str = 'I(Age ** 2)'
        elif base_shape == 'Inverse': base_term_str = 'I(1 / Age)'

        if base_term_str in add_res.params:
            effect_size_beta = add_res.params[base_term_str]
            effect_size_pvalue = add_res.pvalues[base_term_str]
            
            if base_shape == 'Linear': effect_direction = "Increasing" if effect_size_beta > 0 else "Decreasing"
            elif base_shape == 'Quadratic': effect_direction = "Accelerating Up (U)" if effect_size_beta > 0 else "Accelerating Down (n)"
            elif base_shape == 'Inverse': effect_direction = "Decreasing (Drop)" if effect_size_beta > 0 else "Increasing (Rise)"

        # Extract Interaction Effect (From Interactive Model)
        if int_res is not None:
            int_terms = [t for t in int_res.params.index if base_term_str in t and 'Sex' in t]
            if int_terms:
                interaction_beta = int_res.params[int_terms[0]]
                interaction_pvalue = int_res.pvalues[int_terms[0]]
                if best_model.endswith('Interactive'):
                    effect_direction += " (Divergent)"
                else:
                    effect_direction += " (Parallel)"
                
    full_results.append({
        'Feature': feature,
        'Class': cls,
        'Composite_Shape_Weight': round(base_shape_weight, 4), 
        'Sex_Pvalue': sex_p_value,
        'Sex_Beta': sex_beta,
        'Effect_Size_Beta': effect_size_beta,  
        'Effect_Size_Pvalue': effect_size_pvalue,       
        'Interaction_Beta': interaction_beta,       
        'Interaction_Pvalue': interaction_pvalue,   
        'Direction': effect_direction,              
        'Best_Model': best_model,
        'Best_Weight': round(best_weight, 4)
    })

    # ============================================================
    # VISUALIZATION SECTION
    # ============================================================
    if feature in features_to_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sexes = subset['Sex'].unique()
        colors = sns.color_palette("Set1", len(sexes))
        color_map = dict(zip(sexes, colors))
        
        sns.scatterplot(data=subset, x='Age', y='y_scaled', hue='Sex', 
                        palette=color_map, alpha=0.3, edgecolor=None, s=30, ax=ax)
        
        age_seq = np.linspace(5, 85, 200)
        
        for sex in sexes:
            pred_df = pd.DataFrame({'Age': age_seq, 'Sex': [sex] * len(age_seq)})
            pred_y = winning_res.predict(pred_df)
            line_style = '-' if best_model != 'Null' else '--'
            ax.plot(age_seq, pred_y, color=color_map[sex], linewidth=3, linestyle=line_style, label=f'Fit: {sex}')

        clean_name = feature.replace('_avgch', '').replace('_', ' ')
        ax.set_title(f"{clean_name}\nWinner: {best_model} (Shape Weight: {base_shape_weight:.2f})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Age (Years)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Standardized Feature (Z-score)", fontsize=12, fontweight='bold')
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Sex", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        sns.despine()
        plt.tight_layout()
        
        plot_filename = os.path.join(PLOTS_DIR, f"{feature}_Fit.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig) 

print(f"Sample plots saved in: {PLOTS_DIR}")

# ==========================================
# 3. GENERATE CSV OUTPUT & CALCULATE RELATIVE STRENGTH
# ==========================================
final_df = pd.DataFrame(full_results)

# Calculate Relative Strength (Min-Max Scaling of absolute effect sizes per class)
final_df['abs_effect_beta'] = final_df['Effect_Size_Beta'].abs()
max_abs_beta_per_class = final_df.groupby('Class')['abs_effect_beta'].transform('max')
min_abs_beta_per_class = final_df.groupby('Class')['abs_effect_beta'].transform('min')

# Subtraction handles the range natively without needing a redundant absolute value call
final_df['Relative_Strength'] = (final_df['abs_effect_beta'] - min_abs_beta_per_class) / (max_abs_beta_per_class - min_abs_beta_per_class)
final_df['Relative_Strength'] = final_df['Relative_Strength'].round(4)
final_df.drop(columns=['abs_effect_beta'], inplace=True)

# Merge with Category Data
final_df.sort_values(by='Feature', ascending=False, inplace=True)
df_ctg_mapped = df_ctg[['Feature', 'Category']].copy()
final_df = final_df.merge(df_ctg_mapped, on='Feature', how='left')

# Reorder columns logically
cols = final_df.columns.tolist()
cols.remove('Category')
cols.remove('Feature')
cols.remove('Relative_Strength')

insert_idx = cols.index('Effect_Size_Pvalue') + 1
cols.insert(insert_idx, 'Relative_Strength')

final_df = final_df[['Category', 'Feature'] + cols]

# Save Final Output
output_file = os.path.join(OUTPUT_DIR, 'Feature_Cleaned_interactive_Trajectories_Covariate_Sex.csv')
final_df.to_csv(output_file, index=False)
print(f"\n[SUCCESS] Detailed table saved to '{output_file}'")