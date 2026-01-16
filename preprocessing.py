"""
ECG Signal Preprocessing Utility
--------------------------------
This script processes UK Biobank resting ECG data from XML format into 
standardized NumPy arrays. 

Key Features:
- Parallel processing using multiprocessing for high throughput.
- Resamples signals to a fixed length (4096 samples).
- Scales units from microvolts (uV) to millivolts (mV).
- Standardizes lead ordering (I, II, III, aVR, aVL, aVF, V1-V6).
- Saves output as (4096, 12) float32 .npy files.

Usage:
    Update the 'EIDS_PATH', 'ECG_PATH', and 'OUT_FOLDER' constants within the 
    script to point to your local data directories before execution.
"""
from pathlib import Path
import csv
import xml.etree.ElementTree as ET
import numpy as np
import os
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

EIDS_PATH = '/cluster/work/grlab/projects/tmp_imankowski/data/labels/manifest_ecg_rest.csv'
ECG_PATH    = '/cluster/work/grlab/projects/projects2025-dataspectrum4cvd/ukbb/raw/Bulk/ECG'
OUT_FOLDER  = '/cluster/work/grlab/projects/tmp_imankowski/data/preprocessed_ep_rECG_i2'
os.makedirs(OUT_FOLDER, exist_ok=True)

# only use ECGs from instance ECG_INSTANCE
ECG_INSTANCE = 2

# fixed lead order
LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]
LEAD_TO_IDX = {l: i for i, l in enumerate(LEADS)}
SIGNAL_LEN = 4096


def process_eid(eid: str):
    """Worker: convert one ECG XML (for a single eid) to resampled + scaled .npy."""
    xml_filename = f"{eid}_20205_{ECG_INSTANCE}_0.xml"
    xml_path = os.path.join(ECG_PATH, xml_filename)
    out_path = os.path.join(OUT_FOLDER, f"{eid}.npy")
    

    if not os.path.exists(xml_path):
        return eid, False, "XML not found"

    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        return eid, False, f"XML parse error: {e}"

    # (4096, 12) array, all leads
    ecg = np.zeros((SIGNAL_LEN, len(LEADS)), dtype=np.float32)

    try:
        for wf in root.findall("./StripData/WaveformData"):
            lead_name = wf.attrib.get("lead")

            idx = LEAD_TO_IDX[lead_name]
            data = np.fromstring(wf.text.strip(), sep=",", dtype=np.float32)

            # resample to 4096
            resampled = np.interp(
                np.linspace(0.0, 1.0, SIGNAL_LEN),
                np.linspace(0.0, 1.0, data.shape[0]),
                data,
            ).astype(np.float32)

            # scale microvolts -> millivolts
            resampled /= 1000.0

            ecg[:, idx] = resampled

        # save (4096, 12) array as .npy
        np.save(out_path, ecg)
        return eid, True, None

    except Exception as e:
        return eid, False, f"NPY write error: {e}"


def main():
    # collect all eids from scv file with eids
    with open(EIDS_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    eids = [row["eid"] for row in rows]

    n_workers = cpu_count()

    with Pool(processes=n_workers) as pool:
        for eid, ok, err in tqdm(
            pool.imap_unordered(process_eid, eids),
            total=len(eids),
            desc="Converting ECG XML → NPY (parallel)",
        ):
            if not ok:
                tqdm.write(f"{eid}: {err}")


if __name__ == "__main__":
    main()
