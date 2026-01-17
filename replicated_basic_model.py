"""
VO2max Prediction Script (Fixed Linear Model)

Sex Encoding Configuration:
The UK Biobank has follows the encoding: 0 = Female, 1= Male. 
The sex coefficient is negative, resulting in lower VO2max, probably for females.
Thus model weights probably expect the encoding: 0 = Male, 1 = Female.

How to handle the input data:
1. If input data is (0: Female, 1: Male):
   KEEP the line: clinical_df['sex'] = 1 - clinical_df['sex']

2. If input data is (0: Male, 1: Female):
   DELETE or COMMENT OUT the line: clinical_df['sex'] = 1 - clinical_df['sex']
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any

LABELS_FILE = "/cluster/work/grlab/projects/tmp_imankowski/data/labels/real_ukbb_values/30038-1.0.csv"
AGE2_SEX_BMI2_PATH = "/cluster/work/grlab/projects/tmp_imankowski/data/eid_age2_sex_bmi2.csv"
OUTPUT_FILE = "/cluster/work/grlab/projects/tmp_imankowski/data/replicated_basic_vo2_predictions.csv"


TARGET_LABEL = "30038-1.0"

#  Fixed Model Parameters
BASIC_PARAMS: Dict[str, Any] = {
    "(Intercept)": [43.13745938, None, None],
    "age": [-5.235623211 , 45.4056738, 19.22841],
    "sex": [-10.41131428, None, None],
    "bmi": [-4.763900724, 25.8824301 , 4.898251],
    "test_type_bike": [-8.983549601, None, None],
}

def predict_basic(data: pd.DataFrame, params: dict) -> pd.Series:
    """
    Predicts VO2 using the fixed coefficients
    The formula is: VO2_predicted = Intercept + sum(Coefficient_i * Scaled_Feature_i)
    """
    
    intercept_coef = params['(Intercept)'][0]
    predictions = pd.Series([intercept_coef] * len(data), index=data.index)


    data['age'] = data['age'].astype(float)  
    data['bmi'] = data['bmi'].astype(float)
    

    # Since ALL participants used the bike, their prediction should include the bike coefficient.
    data['test_type_bike'] = 1.0 

    
    # Iterate and apply linear model
    for term, (coef, mean, var) in params.items():
        if term == '(Intercept)':
            continue

        feature_values = data[term].astype(float)
        
        # Apply normalization: (X - mean) / sqrt(var)
        if mean is not None and var is not None:
            if var <= 1e-9:
                scaled_feature = (feature_values - mean) / 1.0 # Stabilize near-zero variance
            else:
                scaled_feature = (feature_values - mean) / np.sqrt(var)
        else:
            scaled_feature = feature_values

        predictions += coef * scaled_feature
        
    return predictions

def evaluate_metrics(true_values: pd.Series, predicted_values: pd.Series, target_label: str):
    """Calculates and prints R, R2, MAE, MSE, and RMSE."""
    
    true_values = true_values.astype(float)
    predicted_values = predicted_values.astype(float)

    mean_true_values = tf.reduce_mean(true_values)
    mean_predicted_values = tf.reduce_mean(predicted_values)

    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    
    r2 = r2_score(true_values, predicted_values)
    r_matrix = np.corrcoef(true_values, predicted_values)
    r = r_matrix[0, 1]
    
    print(f"Target Variable: {target_label}")
    print("\n--- Model Evaluation Metrics ---")
    print (f"Mean of True VO2max Values: {mean_true_values}")
    print(f"Mean of Predicted VO2max Values: {mean_predicted_values}")
    print("------------------------------")
    print(f"Pearson Correlation (R): {r:.4f}")
    print(f"Coefficient of Determination (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("------------------------------")
    return {'R': r, 'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    
    # 1. Load clinical and label data
    try:
        labels_df = pd.read_csv(LABELS_FILE)

        clinical_df = pd.read_csv(AGE2_SEX_BMI2_PATH)

        # Delete the following line for encoding (0: female, 1: male)
        # Keep it for encoding (0: male, 1: female)
        clinical_df['sex'] = 1 - clinical_df['sex'] 
        
        full_labels_df = labels_df.merge(clinical_df, on='eid', how='inner')
        full_labels_df['eid'] = full_labels_df['eid'].astype(int)

    except Exception as e:
        print(f"\nERROR: Could not load data files. Check paths and formats.")
        print(f"Detail: {e}")
        exit()
    


    # Apply Fixed Linear Model to get predictions
    predictions = predict_basic(full_labels_df.copy(), BASIC_PARAMS)
    
    # Save results
    full_labels_df.loc[:, 'vo2_predicted_basic'] = predictions.values
    

    # Evaluate Metrics
    if TARGET_LABEL in full_labels_df.columns:
        true_values = full_labels_df[TARGET_LABEL].astype(float)
        valid_indices = true_values.notna()
        
        evaluate_metrics(
            true_values=true_values[valid_indices],
            predicted_values=predictions[valid_indices],
            target_label=TARGET_LABEL
        )
    else:
        print(f"Cannot calculate metrics. Target column '{TARGET_LABEL}' not found or is empty.")
    
    output_path = OUTPUT_FILE 
    
    save_cols = ['eid', TARGET_LABEL, 'vo2_predicted_basic']
    save_cols_existing = [col for col in save_cols if col in full_labels_df.columns]
    
    full_labels_df[save_cols_existing].to_csv(output_path, index=False)
    print(f"\nPrediction complete. Results saved to: {output_path}")
