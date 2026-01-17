"""
Description: 
    Loads preprocessed raw ECG signals (.npy), extracts 320-dimensional 
    embeddings using a pre-trained PCLR model, and saves the output to a CSV. 
    The resulting dataframe is keyed by Participant ID (eid) to allow for 
    downstream merging with clinical metadata.

Input:  Directory of .npy ECG files and PCLR encoder
Output: CSV file containing eids and feature vectors (pclr_output_0 ... pclr_output_319)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from typing import Dict, Any
import joblib

ECG_PATH    = "/cluster/work/grlab/projects/tmp_imankowski/data/preprocessed_ep_rECG_i2"
MODEL_PATH  = "/cluster/work/grlab/projects/tmp_imankowski/data/PCLR.h5"
OUTPUT_FILE = "/cluster/work/grlab/projects/tmp_imankowski/data/pclr_emb_i2.csv"

BATCH_SIZE = 64
SIGNAL_LENGTH = 4096
NUM_LEADS = 12


def _load_ecg_py(path_bytes):
    """NumPy function to load the raw ECG data from .npy."""
    path = path_bytes.decode("utf-8")
    arr = np.load(path).astype(np.float32)
    return arr


def tf_load_ecg(path, eid):
    """TensorFlow map function to load and shape the ECG."""
    ecg = tf.numpy_function(_load_ecg_py, [path], tf.float32)
    ecg.set_shape((SIGNAL_LENGTH, NUM_LEADS))
    return ecg, eid


def extract_features(encoder: Model, ecg_paths: np.ndarray, eids: np.ndarray):
    """
    Uses the PCLR encoder to extract 320-dim embeddings for all ECGs.
    output: Dataframe with eid, pclr_output_0, ..., pclr_output_319, label (vo2 max)
    """
    print(f"\n--- Starting PCLR Feature Extraction for {len(eids)} samples ---")
    
    ecg_ds = tf.data.Dataset.from_tensor_slices((ecg_paths, eids))
    ecg_ds = (
        ecg_ds
        .map(tf_load_ecg, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    

    embeddings = encoder.predict(ecg_ds, verbose=1)
    predicted_eids = eids[:len(embeddings)]
    
    emb_df = pd.DataFrame(embeddings, index=predicted_eids)
    emb_df.index.name = 'eid'
    
    # Rename columns to match feature names (pclr_output_0 to pclr_output_319)
    column_names = {i: f'pclr_output_{i}' for i in range(embeddings.shape[1])}
    emb_df = emb_df.rename(columns=column_names).reset_index()
    
    return emb_df


# --- MAIN EXECUTION ---

if __name__ == '__main__':
    
    # Setup TensorFlow environment
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Get ECG eids and paths
    ecg_paths = []
    eids = []

    
    for filename in os.listdir(ECG_PATH):
        path = os.path.join(ECG_PATH, filename)
        eid_str = filename.rsplit('.', 1)[0]
        eid = int(eid_str)
        ecg_paths.append(path)
        eids.append(eid)
    
    ecg_paths = np.array(ecg_paths)
    eids = np.array(eids)
    print(f"Found {len(eids)} participants with ECG data")

    # Load the PCLR Encoder model
    try:
        full_encoder = load_model(MODEL_PATH, compile=False)
        pclr_encoder = Model(inputs=full_encoder.input, outputs=full_encoder.output)
        print("PCLR encoder loaded successfully")
        
    except Exception as e:
        print(f"\nERROR: Could not load PCLR model from {MODEL_PATH}.")
        print(f"Detail: {e}")
        exit()

    pclr_emb = extract_features(pclr_encoder, ecg_paths, eids)
    
    # Save results
    pclr_emb.to_csv(OUTPUT_FILE, index=False)
    print(f"\nEmbeddings saved to: {OUTPUT_FILE}")
    
