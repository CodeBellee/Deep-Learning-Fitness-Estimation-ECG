"""
UK Biobank Cohort Characterization and Table 1 Generation

This script processes UK Biobank tabular data to generate a baseline characteristics 
table for a specific study cohort. 

Key steps:
1. Cohort Definition: Filters participants based on the availability of resting ECG 
   XML files in the specified directory.
2. Data Extraction: Reads age and clinical biomarkers (blood pressure, lipids, 
   anthropometry) from UKBB tabular extracts.
3. Data Harmonization: Applies a specific priority logic to merge data across 
   instances (prioritizing Instance 2 -> Average(1,3) -> Instance 1 -> Instance 0).
4. Reporting: Generates a TableOne summary and exports it to LaTeX.

"""

from tableone import TableOne
import pandas as pd
import numpy as np
import os
import re


TABULAR_INPUT_FILE_PATH = "/cluster/work/grlab/projects/projects2025-dataspectrum4cvd/ukbb/raw/Main/ukb679928.csv"
AGE_INPUT_FILE_PATH = "/cluster/work/grlab/projects/projects2025-dataspectrum4cvd/ukbb/processed/age_cohorts.csv"
RESTING_ECG_PATH = "/cluster/work/grlab/projects/projects2025-dataspectrum4cvd/ukbb/raw/Bulk/ECG"
OUTPUT_DIR = "/cluster/work/grlab/projects/tmp_imankowski/data/"


# -------------------------------- 1. Get eIDs --------------------------------

xml_files = [f for f in os.listdir(RESTING_ECG_PATH) if f.endswith('.xml')]
eids_set = set()
for filename in xml_files:
    match = re.match(r'(\d{7})', filename)
    if match:
        eids_set.add(int(match.group(1)))

print(f'Extracted {len(eids_set)} unique eIDs from resting ECG XML filenames.')


# -------------------------------- 2. Extract Age --------------------------------
cols_to_keep_age = ['participant.eid', 'participant.p21003_i2']
df_age = pd.read_csv(AGE_INPUT_FILE_PATH, usecols=cols_to_keep_age)

age_rename_map = {
    'participant.eid' : 'eid',
    'participant.p21003_i2' : 'Age (years)'
}
df_age.rename(columns=age_rename_map, inplace=True)

# Keep only EIDs found in the ECG folder
df_age = df_age[df_age['eid'].isin(eids_set)]
print(f"Age data filtered to {len(df_age)} participants based on ECG EIDs.")

# -------------------------------- 3. Extract Tabular Data --------------------------------
cols_to_keep = [
    'eid',
    '31-0.0', # Sex
    '21001-0.0', '21001-1.0', '21001-2.0', '21001-3.0', # BMI, body size measurement.
    '23104-2.0', '23104-3.0', # BMI impedance 
    '30038-0.0', '30038-1.0', # VO2max
    '4080-0.0', '4080-0.1', '4080-1.0', '4080-1.1', '4080-2.0', '4080-2.1', '4080-3.0', '4080-3.1', #blood pressure
    '4079-0.0', '4079-0.1', '4079-1.0', '4079-1.1', '4079-2.0', '4079-2.1', '4079-3.0', '4079-3.1',  #blood pressure
    '30760-0.0', '30760-1.0', # HDL cholesterol
    '30780-0.0', '30780-1.0', # LDL direct
    '30870-0.0', '30870-1.0', # Triglycerides
    '30710-0.0', '30710-1.0', # C-reactive protein (CRP)
    '23400-0.0', '23400-1.0', # Total Cholesterol
    '21021-0.0', '21021-1.0', '21021-2.0', '21021-3.0', # Arterial Stiffness
    '20116-0.0', '20116-1.0', '20116-2.0', '20116-3.0', # Smoking
    '21000-0.0' # Ethnic background
]

df_tabular = pd.read_csv(TABULAR_INPUT_FILE_PATH, usecols=cols_to_keep)

# Keep only EIDs found in the ECG folder
df_tabular = df_tabular[df_tabular['eid'].isin(eids_set)]
print(f"Tabular data filtered to {len(df_tabular)} participants based on ECG EIDs.")

# -------------------------------- 4. Data Processing Logic --------------------------------

df_tabular['31-0.0'] = df_tabular['31-0.0'].map({0.0: 'Female', 1.0: 'Male'})

def apply_priority_logic(df, field_code, field_name, is_bp=False):
    """
    Applies the priority logic (2 -> Avg(1,3) -> 1 -> 0) to a given field.
    """
    if is_bp:
        column_prefix = field_code
        for instance in ['0', '1', '2', '3']:
            col0 = f"{column_prefix}-{instance}.0"
            col1 = f"{column_prefix}-{instance}.1"
            avg_col = f"{column_prefix}_{instance}_avg"
            
            c0 = pd.to_numeric(df[col0], errors='coerce') if col0 in df.columns else pd.Series(np.nan, index=df.index)
            c1 = pd.to_numeric(df[col1], errors='coerce') if col1 in df.columns else pd.Series(np.nan, index=df.index)
            
            df[avg_col] = np.where(
                c0.notna() & c1.notna(), (c0 + c1) / 2.0,
                np.where(c0.notna(), c0, c1)
            )

        c2 = df.get(f"{column_prefix}_2_avg", pd.Series(np.nan, index=df.index))
        c1 = df.get(f"{column_prefix}_1_avg", pd.Series(np.nan, index=df.index))
        c3 = df.get(f"{column_prefix}_3_avg", pd.Series(np.nan, index=df.index))
        c0 = df.get(f"{column_prefix}_0_avg", pd.Series(np.nan, index=df.index))
        
    else: 
        c2 = pd.to_numeric(df.get(f"{field_code}-2.0", pd.Series(np.nan, index=df.index)), errors='coerce')
        c1 = pd.to_numeric(df.get(f"{field_code}-1.0", pd.Series(np.nan, index=df.index)), errors='coerce')
        c3 = pd.to_numeric(df.get(f"{field_code}-3.0", pd.Series(np.nan, index=df.index)), errors='coerce')
        c0 = pd.to_numeric(df.get(f"{field_code}-0.0", pd.Series(np.nan, index=df.index)), errors='coerce')


    # Apply Priority Logic (2 -> Avg(1,3) -> 1 -> 0)
    final_series = c2.copy() if c2 is not None else pd.Series(np.nan, index=df.index)
    
    avg_13 = (c1 + c3) / 2
    final_series = final_series.fillna(avg_13) 
    final_series = final_series.fillna(c1) 
    final_series = final_series.fillna(c0)

    df[field_name] = final_series
    
    return df

def process_biomarkers(df):
    """calls apply_priority_logic for all fields"""
    df = apply_priority_logic(df, '4080', 'Systolic Blood Pressure (mmHg)', is_bp=True)
    df = apply_priority_logic(df, '4079', 'Diastolic Blood Pressure (mmHg)', is_bp=True)
    df = apply_priority_logic(df, '30038', 'VO2max per kg estimated (ml/kg/min)')
    df = apply_priority_logic(df, '30760', 'HDL Cholesterol (mmol/L)')
    df = apply_priority_logic(df, '30780', 'LDL Cholesterol (mmol/L)')
    df = apply_priority_logic(df, '30870', 'Trigylcerides (mmol/L)')
    df = apply_priority_logic(df, '30710', 'hs-CRP (mg/L)')
    df = apply_priority_logic(df, '23400', 'Total Cholesterol (mmol/l)')
    df = apply_priority_logic(df, '21001', 'BMI, body size measures (kg/m^2)')
    df = apply_priority_logic(df, '23104', 'BMI, body composition by impedance (kg/m^2)')
    df = apply_priority_logic(df, '21021', 'Arterial Stiffness Index')
    df = apply_priority_logic(df, '20116', 'Smoking Status')
    df['Smoker'] = (df['Smoking Status'] == 2)
    return df


df_tabular = process_biomarkers(df_tabular.copy()) 

# -------------------------------- 5. Ethnicity Mapping --------------------------------
group_mapping = {
    1: 'White', 1001: 'White', 1002: 'White', 1003: 'White',
    2: 'Mixed', 2001: 'Mixed', 2002: 'Mixed', 2003: 'Mixed', 2004: 'Mixed',
    3: 'Asian', 3001: 'Asian', 3002: 'Asian', 3003: 'Asian', 3004: 'Asian', 5: 'Asian',
    4: 'Black', 4001: 'Black', 4002: 'Black', 4003: 'Black',
    6: 'Other', -1: 'None', -3: 'None'
}
df_tabular['Ethnic Background'] = df_tabular['21000-0.0'].map(group_mapping)

# -------------------------------- 6. Merge & Prepare TableOne (STRICT MERGE APPLIED) --------------------------------

data = df_tabular.merge(df_age, on='eid', how='left')

print(f"Final merged data size (N for Overall/no VO2 max at I1): {len(data)}")

data.drop(columns=[col for col in data.columns if re.match(r'(4080|4079)_[0123]_avg|Smoking Status Code', col)], inplace=True, errors='ignore')
data.rename(columns={'31-0.0': 'Sex'}, inplace=True)


# -------------------------------- 7. Prepare the Subset for TableOne --------------------------------
vo2max_i1_col = '30038-1.0'

eids_with_vo2max_i1 = df_tabular[pd.to_numeric(df_tabular[vo2max_i1_col], errors='coerce').notna()]['eid'].unique()

print(f"Number of participants with VO2 max at Instance 1: {len(eids_with_vo2max_i1)}")

# Grouping columns
cohort_label_overall = 'no VO2max at I1' 
cohort_label_subset = 'VO2max I1 Subset'

data['Analysis Cohort'] = cohort_label_overall 
data.loc[data['eid'].isin(eids_with_vo2max_i1), 'Analysis Cohort'] = cohort_label_subset


# -------------------------------- 8. Define TableOne Variables and Create Table --------------------------------
columns = [
    'Age (years)', 'Sex', 'Ethnic Background',
    'BMI, body size measures (kg/m^2)', 'BMI, body composition by impedance (kg/m^2)',
    'VO2max per kg estimated (ml/kg/min)', 'Systolic Blood Pressure (mmHg)',
    'Diastolic Blood Pressure (mmHg)', 'Total Cholesterol (mmol/l)', 'HDL Cholesterol (mmol/L)',
    'LDL Cholesterol (mmol/L)', 'Trigylcerides (mmol/L)', 'hs-CRP (mg/L)',
    'Arterial Stiffness Index', 'Smoker'
]

categorical = ['Sex', 'Ethnic Background', 'Smoker']
continuous = [
    'Age (years)', 'BMI, body size measures (kg/m^2)',
    'BMI, body composition by impedance (kg/m^2)', 'VO2max per kg estimated (ml/kg/min)',
    'Systolic Blood Pressure (mmHg)', 'Diastolic Blood Pressure (mmHg)', 'Total Cholesterol (mmol/l)',
    'HDL Cholesterol (mmol/L)', 'LDL Cholesterol (mmol/L)',
    'Trigylcerides (mmol/L)', 'hs-CRP (mg/L)', 'Arterial Stiffness Index'
]

cohort_column = 'Analysis Cohort'

# Create TableOne
mytable = TableOne(
    data,
    columns=columns,
    categorical=categorical,
    continuous=continuous,
    groupby=cohort_column,
    overall=True,
    pval=False,
    missing=True
)

# Output
print("\n--- TableOne Summary (Fixed Filtering) ---")
print(mytable.tabulate(tablefmt="github"))

latex_output_path = os.path.join(OUTPUT_DIR, 'tableone_cohorts.tex')
latex_code = mytable.to_latex(
    caption="Baseline Characteristics (Fixed Filtering, Stratified)",
    label="tab:baseline_fixed_filtering",
    float_format='%.1f'     
)

with open(latex_output_path, 'w') as f:
    f.write(latex_code)

print(f"\nLaTeX code saved to: {latex_output_path}")
