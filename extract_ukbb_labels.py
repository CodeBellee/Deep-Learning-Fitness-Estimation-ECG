"""
UKBB Fitness Parameter Extraction Script

Reads UK Biobank CSV data, extracts specific fitness/health columns, 
filters out invalid values (NaN, 'Prefer not to answer'), and 
saves individual CSV files for each parameter.

Usage: python create_fitness_params.py
"""

import os
import sys
import pandas as pd

INPUT_FILE_PATH = "/cluster/work/grlab/projects/projects2025-dataspectrum4cvd/ukbb/raw/Main/ukb679928.csv"
OUTPUT_DIR = "/cluster/work/grlab/projects/tmp_imankowski/data/labels/real_ukbb_values/"

# Some values to exclude as they encode values we do not want to predict (e.g. -3: Prefer not to answer, -7: None of the above)
# To customize per field if necessary
GLOBAL_EXCLUSION_VALUES = [-3, -7, -818]

# --- Field Dictionary (Reference) ---
# Mapping readable names to UKBB Field IDs
UKBB_FIELDS = {
    'WALKING_PACE_i2': "924-2.0",
    'ABLE_TO_WALK_OR_CYCLE_10_MIN_0': '6017-0.0',
    'ABLE_TO_WALK_OR_CYCLE_10_MIN_1': '6017-1.0',
    
    'FEV1_i0': "20150-0.0",  # Best measure
    'FEV1_0': '3063-0.0',
    'FEV1_2': '3063-2.0',
    'FEV1_FVC_RATIO_Z_SCORE': '20258-0.0',
    'FEV1_Z_SCORE': '20256-0.0',
    'FEV1_PREDICTED': '20153-0.0',
    'FORCED_VITAL_CAPACITY_2': '3062-2.0',
    'PEAK_EXPIRATORY_FLOW_0': '2064-0.0',

    'BODY_FAT_PERCENTAGE_2': '23099-2.0',
    'WHOLE_BODY_FAT_MASS_2': '23100-2.0',
    'TRUNK_FAT_PERCENTAGE_2': '23127-2.0',
    'TRUNK_FAT_MASS_2': '23128-2.0',
    
    'HEEL_BONE_DENSITY_T_SCORE_0': '78-0.0',
    'HEEL_BONE_DENSITY_AUTO_LEFT_2': '4106-2.0',

    'LVEF': '31060-2.0',
    'OVERALL_ACCELERATION_AVG': '90012-0.0',
    'HEALTH_SCORE_ENGLAND': '26413-0.0',
    
    'FREQUENCY_OF_TIREDNESS_2': '2080-2.0',
    'HEALTH_SCALE_TODAY_2005': '120103-0.0'
}

# Add the name of the data field as listed above that you want to process in this run
FIELDS_TO_PROCESS = [
    'WALKING_PACE_i2',
    'FEV1_i0'
]

EID_COL = "eid"

def process_and_save_column(df, col_name, col_id, output_dir):
    """
    Extracts a column, cleans NaNs and specific exclusion values, and saves to CSV.
    """
    print(f"\nProcessing {col_name} (ID: {col_id})...")
    
    sub_df = df.dropna(subset=[col_id]).copy()
    
    # Filter out exclusion values (e.g. -3, -7)
    initial_count = len(sub_df)
    sub_df = sub_df[~sub_df[col_id].isin(GLOBAL_EXCLUSION_VALUES)]
    filtered_count = initial_count - len(sub_df)
    
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} rows containing exclusion codes {GLOBAL_EXCLUSION_VALUES}")

    sub_df = sub_df[[EID_COL, col_id]]
    
    output_filename = f"{col_id}.csv"
    output_path = os.path.join(output_dir, output_filename)
    sub_df.to_csv(output_path, index=False)
    
    print(f"Successfully created: {output_path}")
    print(f"Entries: {len(sub_df)}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    required_ids = [UKBB_FIELDS[name] for name in FIELDS_TO_PROCESS]
    required_cols = [EID_COL] + required_ids

    print(f"Reading data from: {INPUT_FILE_PATH}")
    
    try:
        # Check header first to ensure columns exist
        header_df = pd.read_csv(INPUT_FILE_PATH, nrows=0)
        available_cols = set(header_df.columns)
        
        missing_cols = [col for col in required_cols if col not in available_cols]
        if missing_cols:
            print("\nERROR: The following columns are missing from the input file:")
            for col in missing_cols:
                print(f" - {col}")
            sys.exit(1)
            
        print("All required columns found. Loading data...")

        df = pd.read_csv(INPUT_FILE_PATH, usecols=required_cols)
        print(f"Total participants loaded: {len(df)}")

        for name in FIELDS_TO_PROCESS:
            col_id = UKBB_FIELDS[name]
            process_and_save_column(df, name, col_id, OUTPUT_DIR)

        print("\nScript finished successfully.")

    except FileNotFoundError:
        print(f"Error: The input file was not found at {INPUT_FILE_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
