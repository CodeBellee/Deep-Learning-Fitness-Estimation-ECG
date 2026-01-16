"""
UK Biobank Demographics Extractor

This script processes raw UK Biobank data to create a clean dataset containing 
Participant ID (eid), Sex, Age, and BMI. 


Methodology:
1. Loads main data (Sex, BMI) and age data.
2. Performs an inner join on 'eid'.
3. Removes rows with any missing values in target columns.
4. Exports to CSV.
"""
import pandas as pd
import sys

MAIN_FILE_PATH = '/cluster/work/grlab/projects/projects2025-dataspectrum4cvd/ukbb/raw/Main/ukb679928.csv'
AGE_FILE_PATH = '/cluster/work/grlab/projects/projects2025-dataspectrum4cvd/ukbb/processed/age_cohorts.csv'
OUTPUT_FILE_NAME = '/cluster/work/grlab/projects/tmp_imankowski/data/eid_age2_sex_bmi2.csv'

MAIN_COLUMNS = {
    'eid': 'eid',
    '31-0.0': 'sex',
    '21001-2.0': 'bmi' # instance 2
}

AGE_COLUMNS = {
    'participant.eid': 'eid',
    'participant.p21003_i2': 'age' # instance 2
}

try:
    # 1. Load the main UKBB data and select/rename columns
    print(f"Loading main file: {MAIN_FILE_PATH}")
    df_main = pd.read_csv(
        MAIN_FILE_PATH,
        usecols=MAIN_COLUMNS.keys()
    )
    df_main.rename(columns=MAIN_COLUMNS, inplace=True)
    print(f"Main data loaded with {len(df_main)} rows.")

    # 2. Load the age cohort data and select/rename columns
    print(f"Loading age file: {AGE_FILE_PATH}")
    df_age = pd.read_csv(
        AGE_FILE_PATH,
        usecols=AGE_COLUMNS.keys()
    )

    df_age.rename(columns=AGE_COLUMNS, inplace=True)
    print(f"Age data loaded with {len(df_age)} rows.")

    # Merge the two DataFrames on 'eid' (Participant ID)
    print("Merging dataframes on 'eid'...")
    df_merged = pd.merge(df_main, df_age, on='eid', how='inner') # 'inner' join ensures only eids present in BOTH files are kept
    print(f"Dataframe merged. Total rows: {len(df_merged)}")

    # 4. Filter: Drop rows where any of the target columns is missing
    target_cols = ['eid', 'age', 'sex', 'bmi']
    df_cleaned = df_merged.dropna(subset=target_cols)

    final_row_count = len(df_cleaned)

    # 5. Select the column order and save to CSV
    df_final = df_cleaned[target_cols]
    df_final.to_csv(OUTPUT_FILE_NAME, index=False)

    print(f"Successfully created and saved the file: {OUTPUT_FILE_NAME}")
    print(f"FINAL ROW COUNT: {final_row_count}")
    print(f"\n--- Output Summary ---\n{OUTPUT_FILE_NAME} has {final_row_count} rows.")

except FileNotFoundError as e:
    print(f"Error: One of the input files was not found: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    sys.exit(1)
