"""
Blood Pressure Mean Calculator for UK Biobank Data.

This script reads two CSV files representing different
of blood pressure measurements, aligns them by patient ID ('eid'), calculates
the mean value, and exports the result.
"""

import pandas as pd
import os

def process_blood_pressure(path_2_0, path_2_1, output_path, label_col_name):
    try:
        df0 = pd.read_csv(path_2_0)
        df1 = pd.read_csv(path_2_1)
    except FileNotFoundError as e:
        print(f"Error finding file: {e}")
        return

    if 'eid' in df0.columns:
        df0.set_index('eid', inplace=True)
    if 'eid' in df1.columns:
        df1.set_index('eid', inplace=True)
    
    merged = df0.join(df1, lsuffix='_0', rsuffix='_1', how='outer')

    merged['mean_value'] = merged.mean(axis=1)

    output_df = merged[['mean_value']].rename(columns={'mean_value': label_col_name})

    output_df.to_csv(output_path)
    print(f"Successfully saved: {output_path}")

# --- Configuration ---

base_dir = "/cluster/work/grlab/projects/tmp_imankowski/data/labels/real_ukbb_values"

dia_path_0 = f"{base_dir}/4079-2.0.csv"
dia_path_1 = f"{base_dir}/4079-2.1.csv"
dia_output = f"{base_dir}/DIASTOLIC_BP_mean.csv" 

sys_path_0 = f"{base_dir}/4080-2.0.csv"
sys_path_1 = f"{base_dir}/4080-2.1.csv"
sys_output = f"{base_dir}/SYSTOLIC_BP_mean.csv" 

if __name__ == "__main__":
    process_blood_pressure(dia_path_0, dia_path_1, dia_output, "DIASTOLIC_BP_mean")
    process_blood_pressure(sys_path_0, sys_path_1, sys_output, "SYSTOLIC_BP_mean")
