"""
PCLR Fitness Prediction Evaluation Pipeline

Overview:
    This script conducts a comparative evaluation of the addition of PCLR embeddings 
    against standard clinical baselines for predicting various fitness and
    health phenotypes (e.g., VO2 Max, Grip Strength, LVEF) in the UK Biobank dataset.

    It assesses the incremental predictive value of adding embeddings to a baseline
    model consisting of Age, Sex, and BMI.

Workflow:
    1.  Data Loading: Merges target labels, clinical demographics, and PCLR embeddings.
    2.  Inference: Loads pre-trained Lasso regression models (Basic vs. Full) and Scalers.
    3.  Evaluation: Calculates performance metrics (R2, MAE) on a held-out test set.
    4.  Interpretability:
        - Computes feature importance via Lasso coefficients.
        - Aggregates contribution by feature group (Age vs. Sex vs. BMI vs. Embeddings).
    5.  Visualization: Generates a suite of 4 plots per task:
        - Prediction Scatter (Predicted vs. Measured).
        - Top 25 Feature Importance.
        - Stacked Feature Contribution (Basic vs. Full).
        - Incremental R2 Improvement.

Outputs:
    - CSV summary of R2 and MAE improvements.
    - Raw prediction CSVs for analysis (Measured,Predicted_Basic,Predicted_Full)
    - Plots for all tasks.

"""

import os
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------- 1. Configuration & Constants -------------------

BASE_DIR = "/cluster/work/grlab/projects/tmp_imankowski"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(DATA_DIR, "submission_data")

AGE2_SEX_BMI2_PATH = os.path.join(DATA_DIR, "eid_age2_sex_bmi2.csv")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "pclr_emb_i2.csv")
LABELS_BASE_DIR = os.path.join(DATA_DIR, "labels/real_ukbb_values/")

PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predicted_labels")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summary")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

for d in [PREDICTION_DIR, SUMMARY_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Color Palette (Color-Blind Friendly)
COLORS = {
    'Age': '#D55E00',            # Vermillion
    'Sex': '#009E73',            # Bluish Green
    'BMI': '#CC79A7',            # Reddish Purple
    'PCLR Embeddings': '#0072B2' # Blue
}

FITNESS_TASKS = {
    'VO2max_i1': {
        'id': '30038-1.0',
        'file': "labels_VO2max_instance_1.csv",
        'name': "VO2 Max (Instance 1)",
        'unit': "mL/kg/min"
    },
    'VO2max_i0': {
        'id': '30038-0.0',
        'file': "labels_VO2max_instance_0.csv",
        'name': "VO2 Max (Instance 0)",
        'unit': "mL/kg/min"
    },
    'LVEF': {
        'id': '31060-2.0',
        'file': "31060-2.0_LVEF.csv",
        'name': "Left Ventricular Ejection Fraction (LVEF)",
        'unit': "%"
    },

    'Walking_Pace': {
        'id': '924-2.0',
        'file': "walking_pace_i2.csv",
        'name': "Walking Pace",
        'unit': "Category"
    },
    'Able_To_Walk_10Min': {
        'id': '6017-1.0',
        'file': "6017-1.0_ABLE_TO_WALK_10MIN.csv",
        'name': "Able to walk 10 min",
        'unit': "Binary"
    }, 

    'FEV1_Z_Score': {
        'id': '20256-0.0',
        'file': "20256-0.0_FEV1_Z_SCORE.csv",
        'name': "FEV1 Z-Score",
        'unit': "SD"
    },

    'FEV1_Z_Score': {
        'id': '20256-0.0',
        'file': "20256-0.0_FEV1_Z_SCORE.csv",
        'name': "FEV1 Z-Score",
        'unit': "SD"
    },

    'Systolic_BP': {
        'id': 'SYSTOLIC_BP_mean', 
        'file': "SYSTOLIC_BP_2_mean.csv",
        'name': "Systolic Blood Pressure",
        'unit': "mmHg"
    },
    'Diastolic_BP': {
        'id': 'DIASTOLIC_BP_mean', 
        'file': "DIASTOLIC_BP_2_mean.csv",
        'name': "Diastolic Blood Pressure",
        'unit': "mmHg"
    },
    'Grip_Strength_Left': {
        'id': '46-2.0',
        'file': "grip_left_i2.csv",
        'name': "Grip Strength (Left)",
        'unit': "kg"
    },
    'Grip_Strength_Right': {
        'id': '47-2.0',
        'file': "grip_right_i2.csv",
        'name': "Grip Strength (Right)",
        'unit': "kg"
    },
    'Body_Fat_Percentage': {
        'id': '23099-2.0',
        'file': "23099-2.0_BODY_FAT_PERCENTAGE.csv",
        'name': "Body Fat Percentage",
        'unit': "%"
    },
    'Whole_Body_Fat_Mass': {
        'id': '23100-2.0',
        'file': "23100-2.0_WHOLE_BODY_FAT_MASS.csv",
        'name': "Whole Body Fat Mass",
        'unit': "kg"
    },
    'Trunk_Fat_Percentage': {
        'id': '23127-2.0',
        'file': "23127-2.0_TRUNK_FAT_PERCENTAGE.csv",
        'name': "Trunk Fat Percentage",
        'unit': "%"
    },
    'Trunk_Fat_Mass': {
        'id': '23128-2.0',
        'file': "23128-2.0_TRUNK_FAT_MASS.csv",
        'name': "Trunk Fat Mass",
        'unit': "kg"
    },
    'FEV1_i2': {
        'id': '3063-2.0',
        'file': "3063-2.0_FEV1.csv",
        'name': "Forced Expiratory Volume in 1 Second(FEV1) (Instance 2)",
        'unit': "L"
    },
    'Forced_Vital_Capacity': {
        'id': '3062-2.0',
        'file': "3062-2.0_FORCED_VITAL_CAPACITY.csv",
        'name': "Forced Vital Capacity (FVC)",
        'unit': "L"
    },
    'FEV1_FVC_Ratio_Z': {
        'id': '20258-0.0',
        'file': "20258-0.0_FEV1_FVC_RATIO_Z_SCORE.csv",
        'name': "FEV1/FVC Ratio Z-Score",
        'unit': "SD"
    },

    'FEV1_Predicted': {
        'id': '20153-0.0', 
        'file': "20153-0.0_FEV1_PREDICTED.csv",
        'name': "Forced Expiratory Volume in 1 Second (FEV1) Predicted",
        'unit': "L"
    },
    'FEV1_Predicted_Percentage': {
        'id': '20154-0.0', 
        'file': "20154-0.0_FEV1_PREDICTED_PERCENTAGE.csv",
        'name': "Forced Expiratory Volume in 1 Second (FEV1) Predicted Percentage",
        'unit': "%"
    },
    'Heel_BMD_Tscore': {
        'id': '78-0.0',
        'file': "78-0.0_HBM_T_SCORE.csv",
        'name': "Heel Bone Mineral Density T-Score",
        'unit': "T-Score"
    },

    'Health_Scale_Today': {
        'id': '120103-0.0',
        'file': "120103-0.0_SCALE_TO_INDICATE_HOW_HEALTH_IS_TODAY.csv",
        'name': "Subjective Health Scale",
        'unit': "Score"
    },
    
    'Tiredness_Freq': {
        'id': '2080-2.0',
        'file': "2080-2.0_FREQUENCY_OF_TIREDNESS.csv",
        'name': "Tiredness Frequency",
        'unit': "Score"
    },
    'Reaction_Time': {
        'id': '20023-2.0',
        'file': "20023-2.0_REACTION_TIME.csv",
        'name': "Reaction Time",
        'unit': "ms"
    },
    'Health_Score_England': {
        'id': '26413-0.0',
        'file': "26413-0.0_HEALTH_SCORE_ENGLAND.csv",
        'name': "Health Score (England)",
        'unit': "Score"
    },
    'Health_Score_Scotland': {
        'id': '26420-0.0',
        'file': "26420-0.0_HEALTH_SCORE_SCOTLAND.csv",
        'name': "Health Score (Scotland)",
        'unit': "Score"
    },
    'Health_Score_Wales': {
        'id': '26430-0.0',
        'file': "26430-0.0_HEALTH_SCORE_WALES.csv",
        'name': "Health Score (Wales)",
        'unit': "Score"
    }
}

# -------------------- 2. Helper Functions -------------------

def clean_feature_name(name):
    """
    Get clean labels
    e.g., 'pclr_output_103' -> 'PCLR Emb. 103'
    """
    name_lower = name.lower()
    
    if 'pclr_output' in name_lower:
        # Extract number
        match = re.search(r'\d+', name)
        num = match.group() if match else '?'
        return f"PCLR Emb. {num}"
    
    if name_lower == 'bmi': return "BMI"
    if name_lower == 'age': return "Age"
    if name_lower == 'sex': return "Sex"
    return name.replace('_', ' ').title()

def get_model_paths(task_id):
    """
    Constructs paths. Handles standard IDs and BP string IDs.
    """
    lower_id = task_id.lower() 
    base_name = f"{lower_id}_lasso" 
    
    full_model = os.path.join(MODEL_DIR, f"{base_name}_model.pkl")
    full_scaler = os.path.join(MODEL_DIR, f"{base_name}_scaler.pkl")
    
    basic_model = os.path.join(MODEL_DIR, f"basic_{base_name}_model.pkl")
    basic_scaler = os.path.join(MODEL_DIR, f"basic_{base_name}_scaler.pkl")
    
    return (basic_model, basic_scaler), (full_model, full_scaler)

def calculate_feature_contributions(model, scaler, X_df):
    if not hasattr(model, 'coef_'):
        return {}
    
    X_scaled = scaler.transform(X_df.values)
    coeffs = model.coef_
    
    contributions = np.abs(X_scaled * coeffs) 
    mean_contributions = np.mean(contributions, axis=0) 
    
    agg = {'Age': 0.0, 'Sex': 0.0, 'BMI': 0.0, 'PCLR Embeddings': 0.0}
    feature_names = X_df.columns
    
    for name, value in zip(feature_names, mean_contributions):
        name_lower = name.lower()
        if 'age' in name_lower: agg['Age'] += value
        elif 'sex' in name_lower: agg['Sex'] += value
        elif 'bmi' in name_lower: agg['BMI'] += value
        elif 'pclr' in name_lower: agg['PCLR Embeddings'] += value
        
    total = sum(agg.values())
    if total == 0: return agg
    return {k: v / total for k, v in agg.items()}

def load_and_prep_data(label_file, target_id):
    path = os.path.join(LABELS_BASE_DIR, label_file)
    if not os.path.exists(path):
        return None, None, None, None

    labels_df = pd.read_csv(path).rename(columns={'eid': 'eid'})
    clinical_df = pd.read_csv(AGE2_SEX_BMI2_PATH)
    pclr_emb = pd.read_csv(EMBEDDINGS_PATH)
    
    merged = labels_df.merge(clinical_df, on='eid', how='inner')
    merged = merged.merge(pclr_emb, on='eid', how='inner')
    
    for col in ['age', 'bmi', 'sex']:
        merged[col] = merged[col].astype(float)

    for col in ['test_type_bike', 'test_type_treadmill', 'test_type_rower']:
        merged[col] = 0.0
    merged['test_type_bike'] = 1.0 

    clinical_cols = ['age', 'sex', 'bmi', 'test_type_bike', 'test_type_treadmill', 'test_type_rower']
    pclr_cols = [c for c in merged.columns if 'pclr_output' in c]
    
    full_cols = clinical_cols + pclr_cols + [target_id]
    data = merged.dropna(subset=full_cols).reset_index(drop=True)
    
    return data, clinical_cols, pclr_cols, target_id

# -------------------- 3. Visualization Functions -------------------

def plot_scatter(y_true, y_pred, title, unit, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color=COLORS['PCLR Embeddings'], s=10, label='Participants')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity (x=y)')
    
    plt.xlabel(f"Measured [{unit}]")
    plt.ylabel(f"Predicted [{unit}]")
    plt.title(f"Predicted vs. Measured: {title}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, title, filename):
    if not hasattr(model, 'coef_'):
        return

    coeffs = model.coef_.flatten()
    
    cleaned_names = [clean_feature_name(f) for f in feature_names]
    
    fi_df = pd.DataFrame({'feature': cleaned_names, 'raw_feature': feature_names, 'importance': coeffs})
    fi_df['abs_importance'] = fi_df['importance'].abs()
    
    fi_df = fi_df.sort_values('abs_importance', ascending=False).head(25).iloc[::-1]

    bar_colors = []
    for feat in fi_df['raw_feature']: 
        f_lower = feat.lower()
        if 'age' in f_lower: bar_colors.append(COLORS['Age'])
        elif 'sex' in f_lower: bar_colors.append(COLORS['Sex'])
        elif 'bmi' in f_lower: bar_colors.append(COLORS['BMI'])
        elif 'pclr' in f_lower: bar_colors.append(COLORS['PCLR Embeddings'])
        else: bar_colors.append('gray')

    plt.figure(figsize=(10, 8))
    plt.barh(fi_df['feature'], fi_df['importance'], color=bar_colors)
    
    legend_elements = [
        Line2D([0], [0], color=COLORS['Age'], lw=4, label='Age'),
        Line2D([0], [0], color=COLORS['Sex'], lw=4, label='Sex'),
        Line2D([0], [0], color=COLORS['BMI'], lw=4, label='BMI'),
        Line2D([0], [0], color=COLORS['PCLR Embeddings'], lw=4, label='PCLR Embeddings')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.title(f"Top 25 Feature Importances: {title}")
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_stacked_contribution(basic_contrib, full_contrib, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 5), sharey=True)
    
    order = ['Age', 'Sex', 'BMI', 'PCLR Embeddings']
    
    def plot_bar(ax, contribs, bar_title):
        bottom = 0
        for key in order:
            val = contribs.get(key, 0)
            if val > 0:
                ax.bar(0, val, bottom=bottom, color=COLORS[key], width=0.6, label=key)
                if val > 0.04: 
                    ax.text(0, bottom + val/2, f"{val:.1%}", ha='center', va='center', 
                            color='white', fontweight='bold', fontsize=9)
                bottom += val
        ax.set_title(bar_title)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    plot_bar(ax1, basic_contrib, "Basic Model")
    plot_bar(ax2, full_contrib, "Full Model")
    
    ax1.set_ylabel("Feature Contribution (Normalized)")
    ax1.set_ylim(0, 1.05)
    
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0))
    
    plt.suptitle(f"Feature Contribution: {title}", y=0.95)
    plt.tight_layout(rect=[0, 0.1, 1, 0.92])
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_incremental_r2(r2_basic, r2_full, title, filename):
    plt.figure(figsize=(5, 5))
    
    bars = plt.bar(['Basic Model', 'Full Model'], [r2_basic, r2_full], 
                   color=['gray', COLORS['PCLR Embeddings']], width=0.6)
    
    plt.ylabel("Coefficient of Determination ($R^2$)")
    plt.title(f"Incremental Value: {title}")
    plt.ylim(0, max(r2_full, r2_basic) * 1.2 if max(r2_full, r2_basic) > 0 else 1.0)
    
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            plt.text(bar.get_x() + bar.get_width()/2, h, f"{h:.3f}", 
                     ha='center', va='bottom', fontweight='bold')
            
    imp = ((r2_full - r2_basic) / r2_basic) * 100 if r2_basic > 0 else 0
    plt.text(0.5, max(r2_full, r2_basic)*1.1 if max(r2_full, r2_basic) > 0 else 0.5, 
             f"+{imp:.1f}% Imp.", ha='center', color='black', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# -------------------- 4. Main Loop -------------------

if __name__ == '__main__':
    print(f"Starting Evaluation on Test Sets (Split: {TEST_SIZE})...")
    results = []

    for task_key, info in FITNESS_TASKS.items():
        print(f"\nProcessing: {info['name']}...")
        
        df, clin_cols, pclr_cols, target = load_and_prep_data(info['file'], info['id'])
        if df is None or df.empty:
            print(f"  -> Skipped (Data file not found for {task_key})")
            continue
            
        # Split Data (Test Set Only)
        X_clin = df[clin_cols]
        X_full = df[clin_cols + pclr_cols]
        y = df[target].values
        
        _, X_clin_test, _, y_test = train_test_split(X_clin, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        _, X_full_test, _, _ = train_test_split(X_full, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        # Check Model Existence
        (basic_path, basic_scaler_path), (full_path, full_scaler_path) = get_model_paths(info['id'])
        
        if not os.path.exists(full_path):
            print(f"  -> Skipped (Model file not found: {full_path})")
            continue

        # Load Models & Predict
        try:
            model_basic = joblib.load(basic_path)
            scaler_basic = joblib.load(basic_scaler_path)
            model_full = joblib.load(full_path)
            scaler_full = joblib.load(full_scaler_path)
        except Exception as e:
            print(f"  -> Error loading models: {e}")
            continue
        
        X_clin_test_s = scaler_basic.transform(X_clin_test.values)
        y_pred_basic = model_basic.predict(X_clin_test_s)
        
        X_full_test_s = scaler_full.transform(X_full_test.values)
        y_pred_full = model_full.predict(X_full_test_s)
        
        # Metrics
        r2_basic = r2_score(y_test, y_pred_basic)
        r2_full = r2_score(y_test, y_pred_full)
        mae_full = mean_absolute_error(y_test, y_pred_full)
        
        # Feature Contribution
        contrib_basic = calculate_feature_contributions(model_basic, scaler_basic, X_clin_test)
        contrib_full = calculate_feature_contributions(model_full, scaler_full, X_full_test)
        
        # Generate All 4 Plots
        plot_scatter(y_test, y_pred_full, info['name'], info['unit'], 
                        os.path.join(FIGURE_DIR, f"scatter_{task_key}.png"))
        
        full_feature_names = list(clin_cols) + list(pclr_cols)
        plot_feature_importance(model_full, full_feature_names, info['name'],
                                os.path.join(FIGURE_DIR, f"importance_{task_key}.png"))

        plot_stacked_contribution(contrib_basic, contrib_full, info['name'],
                                    os.path.join(FIGURE_DIR, f"stacked_importance_{task_key}.png"))
        
        plot_incremental_r2(r2_basic, r2_full, info['name'],
                            os.path.join(FIGURE_DIR, f"incremental_{task_key}.png"))
        
        # Save Predictions & Log
        pred_df = pd.DataFrame({
            'Measured': y_test,
            'Predicted_Basic': y_pred_basic,
            'Predicted_Full': y_pred_full
        })
        pred_df.to_csv(os.path.join(PREDICTION_DIR, f"preds_{task_key}.csv"), index=False)
        
        results.append({
            'Task': info['name'],
            'N_Test': len(y_test),
            'R2_Basic': r2_basic,
            'R2_Full': r2_full,
            'Improvement': r2_full - r2_basic,
            'MAE_Full': mae_full
        })
        print(f"Done. R2 basic: {r2_basic:.3f}, R2 full: {r2_full:.3f}")

    if results:
        pd.DataFrame(results).to_csv(os.path.join(SUMMARY_DIR, "final_results_summary.csv"), index=False)
        print("\nPipeline Complete. Summary saved.")
