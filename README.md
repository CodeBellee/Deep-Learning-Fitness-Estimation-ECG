# Deep-Learning-Fitness-Estimation-ECG
This project applies deep learning to estimate VO2max and other CRF metrics using resting ECG data from ~8,700 UK Biobank participants. By utilizing pre-trained PCLR embeddings and transfer learning, the model eliminates the need for costly and risky maximal exercise testing.

How I used the files:

1. extract_ukbb_labels.py
   input: INPUT_FILE_PATH = "/cluster/work/grlab/projects/projects2025-dataspectrum4cvd/ukbb/raw/Main/ukb679928.csv"
   output:
   explanation: Create .csv files with create create_fitness_params.py.
   
3. blood_pressure_averager.py
   Special case: blood pressure CSV files were creates with BP_mean.py as teh average of two measurements was calculated.
   
4. process_ukbb_demographics.csv file was created with sex, age (from instace 2), BMI (from instance 2) from participants with a resting ECG data file from insatnce 2
5. extract_pclr_embeddings.py
   PCLR embeddings were extracted and saved for taining and/or predition.
   The PCLR encoder is needed for this step. You can find it here: https://github.com/broadinstitute/ml4h/tree/master/model_zoo/PCLR
   
6. preprocessing.py
   input:
   output:
   Preprocessing the raw ECG data form insatnce 2. modified code from Elias Pinter.

Model Replication:
Hardcoded weights from the paper of Khurshid et al. (2024) were used to predict VO2max of participants of the UK Biobank.

7. replicated_basic_model.py
   Basic Model
   
8. replicated_Deep_VO2_model.py

Training Models on UK Biobank data:

9. basic_model.py creates a 'Basic Model' (Input: Age, Sex, BMI, 320 ECG embeddings) with the machine learning techniques: Lasso, ElasicNet, XGBoost, MLP
This file was used to generate the Basic Model for all machine learning techniques for the VO2max evaluation.


10. full_model.py
   input:
   output: a) saves Full Models for all fitness parameters saved in /cluster/work/grlab/projects/tmp_imankowski/models
           b) prints evaluation metrics (r (95 %CI), R^2, MAE, MSE, RMSE)
   File creates a 'Full Model' (Input: Age, Sex, BMI, 320 ECG embeddings) with the Machine Learning techniques: Lasso, ElasicNet, XGBoost, MLP
   The Lasso Model was run and saved for all fitness metrics. It was saved under /cluster/work/grlab/projects/tmp_imankowski/models
   The other models (ElasicNet, XGBoost, MLP) were run and saved only for the VO2max
   The file also generated the evaluation metrics (r (95 %CI), R^2, MAE, MSE, RMSE)
   
Tables and Figures:


11. ukbb_ecg_cohort_table1.py
    Tableone for classic metrics such as age, sex, ethnic background. Instance Priority 2 -> average(1,3) -> 1 -> 0. Data Mixed from instances.
     Metrics and instance priority inspired from Arjuns Thesis.

12. generate_fitness_summary.py
    Tableone for fitness metrics with trained models and matching instances

13. visualisation

