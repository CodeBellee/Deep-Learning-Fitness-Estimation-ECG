"""
Basic Fitness Parameter Prediction Pipeline ('Basic' Features Only)

This script uses age, sex and BMI to predict various fitness parameters.

Usage:
    1. Configure the 'Configuration' section below.
    2. Select the model type (LASSO, XGBOOST, MLP, ELASTICNET).
    3. Add/uncomment the desired targets in the 'TARGETS_TO_RUN' list.
    4. Run the script.
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from typing import Dict, Any, List

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm


# -------------------------- CONFIGURATION --------------------------

# Choose Model: 'LASSO', 'XGBOOST', 'MLP', or 'ELASTICNET'
MODEL_USED = 'LASSO'

BASE_DIR = "/cluster/work/grlab/projects/tmp_imankowski"
LABELS_BASE_DIR = os.path.join(BASE_DIR, "data/labels/real_ukbb_values")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/labels/predicted_values")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")

AGE2_SEX_BMI2_PATH = os.path.join(BASE_DIR, "data/eid_age2_sex_bmi2.csv")

BATCH_SIZE = 64
TEST_SIZE = 0.2
RANDOM_STATE = 42
ALPHA_VALUE = 1.0


# -------------------------- TARGET DEFINITIONS --------------------------

FITNESS_PARAMS = {
    'VO2max_i0': '30038-0.0',
    'VO2max_i1': '30038-1.0',
    'GRIP_STRENGTH_LEFT_i2': '46-2.0',
    'GRIP_STRENGTH_RIGHT_i2': '47-2.0',
    'WALKING_PACE_i2': '924-2.0',
    'FEV1_i0': '20150-0.0',
    'HBM_T_SCORE': '78-0.0',
    'HEALTH_SCORE_ENGLAND': '26413-0.0',
    'HEALTH_SCORE_WALES': '26430-0.0',
    'HEALTH_SCORE_SCOTLAND': '26420-0.0',
    'REACTION_TIME_2': '20023-2.0',
    'BODY_FAT_PERCENTAGE_2': '23099-2.0',
    'WHOLE_BODY_FAT_MASS_2': '23100-2.0',
    'TRUNK_FAT_PERCENTAGE_2': '23127-2.0',
    'TRUNK_FAT_MASS_2': '23128-2.0',
    'HEEL_BONE_DENSITY_T_SCORE': '78-0.0',
    'HEEL_BONE_DENSITY_LEFT_2': '4106-2.0',
    'WALK_CYCLE_UNAIDED_10M_0': '6017-0.0',
    'WALK_CYCLE_UNAIDED_10M_1': '6017-1.0',
    'FEV1_2': '3063-2.0',
    'FEV1_FVC_RATIO_Z': '20258-0.0',
    'FEV1_Z_SCORE': '20256-0.0',
    'FEV1_PREDICTED': '20153-0.0',
    'FEV1_PREDICTED_PCT': '20154-0.0',
    'FORCED_VITAL_CAPACITY_2': '3062-2.0',
    'LVEF': '31060-2.0',
    'HEALTH_TODAY_SCALE': '120103-0.0',
    'FREQUENCY_OF_TIREDNESS_2': '2080-2.0',
    # If blood_pressure_averager.py was used,
    # the file and column names will be named DIASTOLIC_BP_mean / SYSTOLIC_BP_mean
    'DIASTOLIC_BP': 'DIASTOLIC_BP_mean', 
    'SYSTOLIC_BP': 'SYSTOLIC_BP_mean'
}

#  Choose Targets to run here
TARGETS_TO_RUN = [
    FITNESS_PARAMS['VO2max_i0'],
    # FITNESS_PARAMS['VO2max_i1'],
    # FITNESS_PARAMS['GRIP_STRENGTH_LEFT_i2'],
    # FITNESS_PARAMS['GRIP_STRENGTH_RIGHT_i2'],
    # FITNESS_PARAMS['HEALTH_TODAY_SCALE'],
    FITNESS_PARAMS['DIASTOLIC_BP'],
    FITNESS_PARAMS['WALKING_PACE_i2']
]


# -------------------------- HELPER FUNCTIONS --------------------------

def get_label_path(target_id: str) -> str:
    """Generates the full path to the label CSV based on the target ID."""
    filename = f"{target_id}.csv"
    return os.path.join(LABELS_BASE_DIR, filename)

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepares clinical features and ensures correct data types."""
    feature_df = data.copy()
    
    cols_to_float = ['age', 'bmi', 'sex']
    for col in cols_to_float:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].astype(float)
    
    # UK biobank exercise tests were all performed on a bike
    feature_df['test_type_bike'] = 1.0
    feature_df['test_type_treadmill'] = 0.0
    feature_df['test_type_rower'] = 0.0
    
    return feature_df

def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculates evaluation metrics and prints them in the specific format."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    valid_idx = np.logical_and(~np.isnan(y_true), ~np.isnan(y_pred))
    y_true_valid = y_true[valid_idx]
    y_pred_valid = y_pred[valid_idx]
    
    n = len(y_true_valid)
    if n < 2:
        return {'R': 0, 'R2': 0, 'MAE': 0, 'MSE': 0, 'RMSE': 0}

    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    mse = mean_squared_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_valid, y_pred_valid)
    r = np.corrcoef(y_true_valid, y_pred_valid)[0, 1]

    # 95% CI for R
    r_clipped = np.clip(r, -0.99999999, 0.99999999)
    z = 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))
    se_z = 1.0 / np.sqrt(n - 3)
    z_critical = norm.ppf(0.975)
    
    z_lower = z - z_critical * se_z
    z_upper = z + z_critical * se_z
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    
    print(f"Number of valid samples (n): {n}")
    print(f"Pearson Correlation (R): {r:.4f} (95% CI: [{r_lower:.4f}, {r_upper:.4f}])")
    print(f"Coefficient of Determination (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    return {'R': r, 'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}

def get_basic_features(data: pd.DataFrame, target_col: str):
    """
    Identifies available CLINICAL features and prepares X, y for training.
    Strictly excludes PCLR features.
    """
    clinical_features = ['age', 'sex', 'bmi', 'test_type_bike', 'test_type_treadmill', 'test_type_rower']
    # Note: PCLR features are intentionally omitted here
    
    available_features = [f for f in clinical_features if f in data.columns]
    
    X = data[available_features].copy()
    y = data[target_col].copy()
    
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y, available_features


# -------------------------- MODEL TRAINERS --------------------------

def train_custom_lasso_model(data: pd.DataFrame, target_col: str):
    print("\n--- Training Custom Model ---")

    X, y, features = get_basic_features(data, target_col)
    
    print(f"Using {len(features)} features for training")
    print(f"Training on {len(X)} samples after removing NaN values")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    base_model = Lasso(alpha=ALPHA_VALUE, random_state=RANDOM_STATE, max_iter=2000)
    param_grid = {'alpha': np.logspace(-3, 2, 50)}
    
    grid_model = GridSearchCV(base_model, param_grid, scoring='r2', cv=10, n_jobs=4, verbose=1)
    grid_model.fit(X_train_scaled, y_train)
    
    model = grid_model.best_estimator_
    print(f"\nOptimal alpha found via CV: {grid_model.best_params_['alpha']:.4f}")
    
    # Metrics
    print("\n---Training Set Performance ---")
    y_train_pred = model.predict(X_train_scaled)
    calculate_metrics(y_train, y_train_pred)

    print("\n--- Test Set Performance ---")
    y_test_pred = model.predict(X_test_scaled)
    calculate_metrics(y_test, y_test_pred)
    
    # Top Features
    print("\n--- Top 20 Most Important Features ---")
    feature_importance = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_
    })
    feature_importance['abs_coef'] = feature_importance['coefficient'].abs()
    feature_importance = feature_importance.sort_values('abs_coef', ascending=False)
    print(feature_importance.head(20).to_string(index=False))
    
    return model, scaler, features, X_test_scaled, y_test

def train_custom_xgboost_model(data: pd.DataFrame, target_col: str):
    print("\n--- Training XGBoost Regressor Model ---")

    X, y, features = get_basic_features(data, target_col)
    
    print(f"Using {len(features)} features for training")
    print(f"Training on {len(X)} samples after removing NaN values")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    base_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, tree_method='hist', random_state=RANDOM_STATE)
    param_grid = {
        'max_depth': [3, 4, 5],
        'gamma': [0, 0.5, 1],
        'reg_lambda': [0.1, 1, 10],
        'n_estimators': [100],
        'learning_rate': [0.1]
    }
    
    grid_model = GridSearchCV(base_model, param_grid, scoring='r2', cv=10, n_jobs=-1, verbose=1)
    grid_model.fit(X_train_scaled, y_train)
    
    model = grid_model.best_estimator_
    print(f"\nOptimal XGBoost parameters found via CV: {grid_model.best_params_}")
    
    # Metrics
    print("\n--- Training Set Performance ---")
    y_train_pred = model.predict(X_train_scaled)
    calculate_metrics(y_train, y_train_pred)

    print("\n--- Test Set Performance ---")
    y_test_pred = model.predict(X_test_scaled)
    calculate_metrics(y_test, y_test_pred)
    
    # Top Features
    print("\n--- Top 20 Most Important Features (Gain) ---")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print(feature_importance.head(20).to_string(index=False))
    
    return model, scaler, features, X_test_scaled, y_test

def train_custom_elasticnet_model(data: pd.DataFrame, target_col: str):
    print("\n--- Training Elastic Net Model ---")
    
    X, y, features = get_basic_features(data, target_col)
    
    print(f"Using {len(features)} features for training")
    print(f"Training on {len(X)} samples after removing NaN values")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    base_model = ElasticNet(random_state=RANDOM_STATE, max_iter=5000)
    param_grid = {
        'alpha': np.logspace(-3, 2, 20),
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    }
    
    grid_model = GridSearchCV(base_model, param_grid, scoring='r2', cv=10, n_jobs=-1, verbose=1)
    grid_model.fit(X_train_scaled, y_train)
    
    model = grid_model.best_estimator_
    print(f"\nOptimal params found: alpha={grid_model.best_params_['alpha']:.4f}, l1_ratio={grid_model.best_params_['l1_ratio']:.4f}")
    
    # Metrics
    print("\n---Training Set Performance ---")
    y_train_pred = model.predict(X_train_scaled)
    calculate_metrics(y_train, y_train_pred)

    print("\n--- Test Set Performance ---")
    y_test_pred = model.predict(X_test_scaled)
    calculate_metrics(y_test, y_test_pred)
    
    # Top Features
    print("\n--- Top 20 Most Important Features ---")
    feature_importance = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_
    })
    feature_importance['abs_coef'] = feature_importance['coefficient'].abs()
    feature_importance = feature_importance.sort_values('abs_coef', ascending=False)
    print(feature_importance.head(20).to_string(index=False))
    
    return model, scaler, features, X_test_scaled, y_test

def train_custom_mlp_model(data: pd.DataFrame, target_col: str):
    print("\n--- Training Multi-Layer Perceptron (MLP) Model ---")
    
    X, y, features = get_basic_features(data, target_col)
    
    print(f"Using {len(features)} features for training")
    print(f"Training on {len(X)} samples after removing NaN values")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    input_dim = X_train_scaled.shape[1]
    
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='dense_1'),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001), name='dense_2'),
        Dropout(0.3),
        Dense(64, activation='relu', name='dense_3'),
        Dense(1, name='output_layer')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mse'])
    print(model.summary())
    
    early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopper],
        verbose=1,
        shuffle=True
    )
    
    print("\n---Training Set Performance ---")
    y_train_pred = model.predict(X_train_scaled).flatten()
    calculate_metrics(y_train, y_train_pred)
    
    print("\n--- Test Set Performance ---")
    y_test_pred = model.predict(X_test_scaled).flatten()
    calculate_metrics(y_test, y_test_pred)
    
    print("\n--- Feature Importance ---")
    print("Feature importance for MLP requires separate techniques (e.g., SHAP).")
    
    return model, scaler, features, X_test_scaled, y_test

MODEL_FUNCTIONS = {
    "LASSO": train_custom_lasso_model,
    "XGBOOST": train_custom_xgboost_model,
    "MLP": train_custom_mlp_model,
    "ELASTICNET": train_custom_elasticnet_model
}


# -------------------------- MAIN EXECUTION --------------------------

if __name__ == '__main__':

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    print(f"Model used: {MODEL_USED}")

    # Load clinical data once
    try:
        clinical_df = pd.read_csv(AGE2_SEX_BMI2_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: Clinical data file not found. {e}")
        exit()

    # Loop through all selected targets
    for target_label in TARGETS_TO_RUN:

        print("\n" + "-"*50)
        print(f" Target evaluation of target {target_label} starts now ")
        print("-"*50 + "\n")
        
        # Load Label Data
        label_file = get_label_path(target_label)
        if not os.path.exists(label_file):
            print(f"WARNING: Label file not found at {label_file}. Skipping...")
            continue
            
        try:
            labels_df = pd.read_csv(label_file)
            labels_df = labels_df.rename(columns={'eid': 'eid'})
            
            # Merge clinical and labels (No PCLR merge here!)
            full_data = labels_df.merge(clinical_df, on='eid', how='inner')
            full_data['eid'] = full_data['eid'].astype(int)
            print(f"Loaded label data for {len(full_data)} participants")
            
            full_data = prepare_features(full_data)
            
        except Exception as e:
            print(f"Error preparing data for {target_label}: {e}")
            continue

        # Train Model
        train_fn = MODEL_FUNCTIONS.get(MODEL_USED.upper())
        if not train_fn:
            print(f"Error: Model {MODEL_USED} not implemented.")
            break
            
        model, scaler, features, X_test_scaled, y_test = train_fn(full_data, target_label)
        
        print(f"\n--- Model used: {MODEL_USED} ---")

        # Save Model & Scaler
        model_filename = os.path.join(MODEL_SAVE_DIR, f"basic_{target_label.lower()}_{MODEL_USED.lower()}_model.pkl")
        scaler_filename = os.path.join(MODEL_SAVE_DIR, f"basic_{target_label.lower()}_{MODEL_USED.lower()}_scaler.pkl")
        
        if MODEL_USED != 'MLP': 
            joblib.dump(model, model_filename)
        else:
            model.save(model_filename.replace('.pkl', '.h5'))
        joblib.dump(scaler, scaler_filename)
        
        print(f"\nModel saved to: {model_filename}")
        print(f"Scaler saved to: {scaler_filename}")

        # Predict on Full Dataset
        print("\n--- Full Dataset Performance ---")
        X_full = full_data[features]
        X_full_scaled = scaler.transform(X_full)
        full_preds = model.predict(X_full_scaled)
        if MODEL_USED == 'MLP': full_preds = full_preds.flatten()
        
        # Metrics on full data
        calculate_metrics(full_data[target_label], full_preds)

        # Save Results
        pred_col = "vo2_predicted_custom" # Maintained variable name from basic_params for consistency
        full_data[pred_col] = full_preds
        
        out_file = os.path.join(OUTPUT_DIR, f"basic_{target_label.lower()}_{MODEL_USED.lower()}.csv")
        save_cols = ['eid', target_label, pred_col]
        
        # Only save columns that actually exist
        save_cols_existing = [col for col in save_cols if col in full_data.columns]
        full_data[save_cols_existing].to_csv(out_file, index=False)
        print(f"\nPredictions saved to: {out_file}")
