# Deep Learning Fitness Estimation from ECG

This project applies deep learning to estimate **VO2max** and other **Cardiorespiratory Fitness (CRF)** metrics using resting 12-lead ECG data from ~8,700 UK Biobank participants.

By utilizing pre-trained PCLR (Patient Contrastive Learning of Representations) embeddings and transfer learning, this framework eliminates the need for costly and risky maximal exercise testing, predicting fitness phenotypes directly from the raw ECG signal and basic clinical demographics.

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Workflow Pipeline](#workflow-pipeline)
  - [1. Data Extraction](#1-data-extraction)
  - [2. Signal Preprocessing](#2-signal-preprocessing)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Model Training & Replication](#4-model-training--replication)
- [Analysis & Tables](#analysis--tables)
- [References](#references)

## Project Overview

The core objective of this repository is to assess the incremental predictive value of adding ECG embeddings to standard clinical variables (Age, Sex, BMI).

* **Dataset:** UK Biobank (Resting ECGs from Instance 2).
* **Methodology:** Comparative analysis between a "Basic Model" (Demographics only) and a "Full Model" (Demographics + PCLR Embeddings).
* **Replication:** Includes scripts to replicate the "Deep ECG-VO2" model using weights from *Khurshid et al. (2024)*.

## Prerequisites

### Data Requirements
To run these scripts, you need access to:
1.  **UK Biobank Tabular Data:** The main phenotype CSV (e.g., `ukb679928.csv`).
2.  **Raw ECG Data:** Bulk Field 20205 (XML format).
3.  **PCLR Encoder:** The pre-trained model weights (`PCLR.h5`) from the [Broad Institute ML4H repository](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/PCLR).

### Dependencies
* Python 3.x
* TensorFlow
* Pandas / NumPy
* Scikit-Learn / XGBoost
* TableOne (for cohort tables)

## Workflow Pipeline

### 1. Data Extraction
Extract specific fitness phenotypes and clean the data.

* **`extract_ukbb_labels.py`**: Reads the main UKBB CSV and extracts targets like Walking Pace, FEV1, and Body Fat %. It filters out nulls and "Prefer not to answer" codes.
* *Note:* Blood pressure labels should be averaged (mean of two measurements) prior to use.

### 2. Signal Preprocessing
Convert raw XML data into analysis-ready arrays.

* **`preprocessing.py`**: 
    * Parses UKBB XML files.
    * Resamples signals to 4096 time steps.
    * Scales voltage ($\mu V \rightarrow mV$).
    * Saves outputs as `.npy` files.
    * *Credit:* Modified from code by Elias Pinter.

### 3. Feature Engineering
Extract deep learning features from the ECGs.

* **`extract_pclr_embeddings.py`**: Loads the preprocessed `.npy` ECGs and passes them through the pre-trained PCLR encoder to generate 320-dimensional embedding vectors.

### 4. Model Training & Replication

#### A. Model Replication (Validation)
* **`replicated_deep_vo2_model.py`**: Validates the pipeline by applying hardcoded weights from the *Khurshid et al. (2024)* paper to the UKBB cohort to predict VO2max.

#### B. Training New Models
We train two variations to test hypothesis:

1.  **`basic_model.py` (Baseline)**: Trains Lasso, ElasticNet, XGBoost, and MLP models using **only** clinical features (Age, Sex, BMI).
2.  **`full_model.py` (Deep Learning)**: Trains the same models using Clinical features **+ 320 PCLR ECG embeddings**.
    * *Output:* Saves trained models and generates evaluation metrics ($R$, $R^2$, MAE, MSE, RMSE).

## Analysis & Tables

Scripts for generating statistics and figures.

* **`ukbb_ecg_cohort_table1.py`**: Generates "Table 1" (Baseline Characteristics). It uses a specific logic for harmonizing data across instances (Priority: Instance 2 $\rightarrow$ Avg(1,3) $\rightarrow$ 1 $\rightarrow$ 0).
* **`generate_fitness_summary.py`**: Generates summary statistics ("Table 3.2") for the target fitness metrics.
* **`performance_analysis.py`**: 
    * Compares Basic vs. Full models.
    * Generates scatter plots (Measured vs. Predicted).
    * Calculates feature importance and incremental $R^2$ improvement.

## References

1.  **Deep ECG-VO2:** Khurshid et al., "Deep learned representations of the resting 12-lead electrocardiogram to predict at peak exercise." (2024).
2.  **PCLR:** [Broad Institute ML4H Model Zoo](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/PCLR).
