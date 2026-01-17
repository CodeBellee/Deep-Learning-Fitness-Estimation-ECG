"""
Deep ECG-VO2 Replication Model

Description:
    This script implements the "Deep ECG-VO2" (Basic + PCLR) model as described 
    by Khurshid et al. (2024). It utilizes a pre-trained PCLR (Patient Contrastive 
    Learning of Representations) encoder to extract 320-dimensional embeddings 
    from 12-lead ECG signals. These embeddings are then integrated with 
    clinical features (age, sex, BMI) through a fixed-coefficient linear 
    regression model to predict VO2 max.

Model Details:
    - Model Architecture: Basic Clinical Features + PCLR ECG Embeddings.
    - Input: 12-lead ECG (.npy files, 4096 samples) and clinical CSV data, PCLR encoder
    - Output: Predicted VO2 max values (mL/kg/min) and evaluation
    - Parameters: Coefficients and scaling factors are derived from the 
      supplementary materials of the Khurshid et al. study.

Usage:
    - Ensure ECG files are named by 'eid' and stored in the specified ECG_PATH.
    - Ensure the PCLR encoder (.h5) is available at MODEL_PATH.
    - The script performs feature extraction and linear inference sequentially.


Reference: Khurshid et al., "Deep learned representations of the resting 12-lead electrocardiogram to predict at peak exercise."
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any


ECG_PATH    = "/cluster/work/grlab/projects/tmp_imankowski/data/preprocessed_ep_VO2_i1_rECG_i2"
LABELS_FILE = "/cluster/work/grlab/projects/tmp_imankowski/data/labels/real_ukbb_values/30038-1.0.csv"
AGE2_SEX_BMI2_PATH = "/cluster/work/grlab/projects/tmp_imankowski/data/eid_age2_sex_bmi2.csv"
MODEL_PATH  = "/cluster/work/grlab/projects/tmp_imankowski/data/PCLR.h5"

OUTPUT_FILE = "/cluster/work/grlab/projects/tmp_imankowski/data/replicated_deep_ecg_vo2_predictions.csv"

TARGET_LABEL = "30038-1.0"
BATCH_SIZE = 64
SIGNAL_LENGTH = 4096
NUM_LEADS = 12

DEEP_ECG_VO2_PARAMS: Dict[str, Any] = {
    # Basic Features (Pages 3-4)
    "(Intercept)": [39.42936232, None, None],
    "age": [-2.522921776, 45.4056738, 19.2284085],
    "sex": [-4.358716985, None, None],
    "bmi": [-3.021876721, 25.8824301, 4.89825069],
    "test_type_bike": [-6.687211652, None, None],
    "test_type_treadmill": [0.0, None, None],
    "test_type_rower": [0.0, None, None],
    # PCLR Embeddings (320 features)
    "pclr_output_0": [0.148974119, 1.49757739, 1.25336251],
    "pclr_output_1": [0.201471518, 1.4395947, 1.80083661],
    "pclr_output_2": [0.197460849, 2.06085344, 2.69233884],
    "pclr_output_3": [0.155841582, 1.9336512, 2.59133112],
    "pclr_output_4": [0.0, 0.94172909, 1.41948921],
    "pclr_output_5": [0.0, 2.10617496, 2.71041333],
    "pclr_output_6": [0.0, 2.51889324, 2.54681674],
    "pclr_output_7": [0.0, 2.01589794, 2.38114314],
    "pclr_output_8": [0.049585647, 1.42741893, 2.3193154],
    "pclr_output_9": [0.0, 1.27281774, 1.45410427],
    "pclr_output_10": [0.0, 1.87638397, 2.26644863],
    "pclr_output_11": [0.0, 1.71532417, 2.30505027],
    "pclr_output_12": [0.0, 1.15172952, 1.878187],
    "pclr_output_13": [0.0, 2.52915857, 3.25310709],
    "pclr_output_14": [0.0, 0.97056106, 1.65253814],
    "pclr_output_15": [0.0, 1.46257073, 2.53792558],
    "pclr_output_16": [0.0, 1.25707826, 1.75960251],
    "pclr_output_17": [0.0, 2.65339803, 2.88406637],
    "pclr_output_18": [0.0, 1.81320734, 2.08272051],
    "pclr_output_19": [0.0, 1.57679402, 2.39824657],
    "pclr_output_20": [0.027496439, 1.30758231, 1.31511039],
    "pclr_output_21": [0.107016563, 1.84227237, 1.84530181],
    "pclr_output_22": [-0.069538412, 0.93325, 1.42689217],
    "pclr_output_23": [0.0, 1.31928476, 2.11185039],
    "pclr_output_24": [-0.880894757, 0.50404455, 1.15274568],
    "pclr_output_25": [0.0, 2.48927446, 2.13926756],
    "pclr_output_26": [0.0, 1.70225812, 2.55151309],
    "pclr_output_27": [0.0, 0.48190097, 0.68036743],
    "pclr_output_28": [-0.129330365, 0.83311627, 1.40355844],
    "pclr_output_29": [-0.469502682, 2.54771039, 2.20680305],
    "pclr_output_30": [0.0, 1.14102543, 2.08873501],
    "pclr_output_31": [0.0, 1.42398341, 2.00559063],
    "pclr_output_32": [-1.074725926, 1.61824293, 2.0941349],
    "pclr_output_33": [0.0, 1.22833162, 1.87370168],
    "pclr_output_34": [-1.07601847, 0.74702225, 1.79059082],
    "pclr_output_35": [0.0, 0.94841636, 1.58999938],
    "pclr_output_36": [0.0, 2.29701872, 2.04374044],
    "pclr_output_37": [0.0, 2.92214635, 2.39785217],
    "pclr_output_38": [0.0, 0.7956176, 1.55924987],
    "pclr_output_39": [0.0, 1.61840732, 2.05172985],
    "pclr_output_40": [-0.027183483, 0.51811184, 1.10962812],
    "pclr_output_41": [0.683933629, 2.69708607, 2.49509735],
    "pclr_output_42": [0.0, 5.22500243, 3.47034934],
    "pclr_output_43": [0.0, 1.71701299, 1.86554481],  # CORRECTED
    "pclr_output_44": [0.0, 0.48792009, 1.19237208],
    "pclr_output_45": [0.49063298, 1.52284553, 1.83160526],
    "pclr_output_46": [0.0, 1.14576111, 1.80910059],
    "pclr_output_47": [0.0, 0.58935353, 1.32222676],
    "pclr_output_48": [0.0, 2.12547194, 1.95151821],
    "pclr_output_49": [0.03637642, 2.39926828, 2.26330425],
    "pclr_output_50": [0.0, 1.25465094, 2.08907426],
    "pclr_output_51": [0.0, 2.98717328, 3.4903483],
    "pclr_output_52": [0.0, 0.93260558, 1.87640195],
    "pclr_output_53": [0.0, 2.0625568, 2.36539885],
    "pclr_output_54": [0.0, 1.07688038, 1.46166992],
    "pclr_output_55": [0.518785398, 1.73275764, 1.16605968],
    "pclr_output_56": [0.0, 1.6228801, 2.72987279],
    "pclr_output_57": [0.0, 1.29188397, 2.40251639],
    "pclr_output_58": [0.0, 0.49886388, 1.12057055],      #
    "pclr_output_59": [-0.101083668, 1.17102203, 2.18502834],
    "pclr_output_60": [0.0, 2.35975999, 2.85336955],
    "pclr_output_61": [0.00169397, 1.16929443, 1.92256332],
    "pclr_output_62": [0.23101011, 2.37776138, 2.72927501],
    "pclr_output_63": [0.0, 1.45200521, 2.07717096],
    "pclr_output_64": [0.378410412, 2.29688033, 2.47473183],
    "pclr_output_65": [-0.127299014, 1.95146638, 2.2496957],
    "pclr_output_66": [0.0, 2.44155118, 2.28037521],
    "pclr_output_67": [0.0, 1.34757001, 2.48331465],  # CORRECTED
    "pclr_output_68": [0.0, 1.21077539, 2.46706823],  # CORRECTED
    "pclr_output_69": [0.068226177, 2.34865584, 3.48535103],  # CORRECTED
    "pclr_output_70": [0.0, 0.7621336, 1.43331644],
    "pclr_output_71": [-0.056232563, 1.16668365, 1.71641714],
    "pclr_output_72": [0.0, 1.86829086, 2.14646078],  # CORRECTED
    "pclr_output_73": [-0.429751133, 1.70075268, 2.54578805],
    "pclr_output_74": [0.0, 2.57171366, 3.31569623],
    "pclr_output_75": [0.0, 1.51868082, 1.97518165],
    "pclr_output_76": [-0.334797715, 1.13833135, 1.50992665],
    "pclr_output_77": [0.0, 0.7602404, 1.31149262],  # CORRECTED
    "pclr_output_78": [-0.092420216, 1.47484159, 1.91856792],
    "pclr_output_79": [0.0, 1.55556323, 1.65410917],  # CORRECTED
    "pclr_output_80": [0.0, 1.35642381, 2.39836271],  # CORRECTED
    "pclr_output_81": [0.0, 1.05263052, 1.80983737],
    "pclr_output_82": [0.0, 0.95704253, 2.12358119],
    "pclr_output_83": [0.150593047, 1.75125534, 2.86055303],
    "pclr_output_84": [0.0, 1.56431301, 1.65765174],
    "pclr_output_85": [0.0, 1.06124185, 2.12778982],
    "pclr_output_86": [0.0, 0.30322034, 0.77854529],  # CORRECTED
    "pclr_output_87": [0.0, 0.4184037, 0.30608754],  # CORRECTED
    "pclr_output_88": [0.0, 3.65413955, 2.31265522],  # CORRECTED
    "pclr_output_89": [-0.937660201, 1.20771869, 1.74010176],  # CORRECTED
    "pclr_output_90": [0.0, 1.1130913, 1.95439776],  # CORRECTED
    "pclr_output_91": [0.0, 0.36041742, 1.00558223],  # CORRECTED
    "pclr_output_92": [0.0, 1.33911093, 2.55232705],  # CORRECTED
    "pclr_output_93": [0.0, 1.39175097, 1.65883503],
    "pclr_output_94": [-0.058210198, 0.15191849, 0.98413665],
    "pclr_output_95": [0.0, 0.73985778, 1.55743041],
    "pclr_output_96": [0.0, 1.2088637, 1.71820709],
    "pclr_output_97": [0.0, 1.01686162, 2.05849463],
    "pclr_output_98": [-0.1880952, 1.45144563, 1.87674562],  # CORRECTED
    "pclr_output_99": [0.511592136, 1.44049729, 1.81966165],
    "pclr_output_100": [0.0, 2.93295178, 3.78095987],
    "pclr_output_101": [0.0, 1.14009676, 1.81718861],
    "pclr_output_102": [0.453830423, 2.55332723, 2.79851765],
    "pclr_output_103": [0.146942391, 2.47006687, 2.46619298],  # CORRECTED
    "pclr_output_104": [-0.473426773, 0.81357957, 1.54773543],
    "pclr_output_105": [0.0, 1.21093181, 2.08484477],
    "pclr_output_106": [0.0, 1.92795475, 2.80328368],
    "pclr_output_107": [0.0, 1.53389677, 2.08173665],
    "pclr_output_108": [0.0, 1.45091412, 2.00101274],
    "pclr_output_109": [0.080093009, 1.23795383, 1.67181122],
    "pclr_output_110": [0.176081202, 1.9801926, 1.71651331],
    "pclr_output_111": [-0.348673871, 1.28153615, 1.67563288],
    "pclr_output_112": [0.0, 1.35896291, 1.40391599],
    "pclr_output_113": [-0.165270443, 1.54206156, 1.88027165],
    "pclr_output_114": [-0.028987043, 1.75269263, 1.41780746],  # CORRECTED
    "pclr_output_115": [0.0, 1.10278616, 1.46612177],
    "pclr_output_116": [0.0, 1.17333313, 1.89999252],
    "pclr_output_117": [0.0, 1.22175391, 1.63526162],
    "pclr_output_118": [-0.102437525, 1.18847759, 1.77596643],
    "pclr_output_119": [0.718545419, 1.2914633, 1.08349809],
    "pclr_output_120": [1.580071616, 1.85277009, 1.91557208],
    "pclr_output_121": [0.0, 0.92273518, 1.72973425],
    "pclr_output_122": [-0.167631066, 0.91955042, 2.52992642],
    "pclr_output_123": [-0.825266141, 0.80287255, 1.22023543],
    "pclr_output_124": [-0.421658605, 2.31576135, 2.84812247],
    "pclr_output_125": [0.620557477, 2.28032887, 2.00783005],
    "pclr_output_126": [0.0, 1.61009787, 2.52659522],
    "pclr_output_127": [0.0, 1.14240894, 1.69967555],
    "pclr_output_128": [1.014726443, 1.88131286, 2.36560046],
    "pclr_output_129": [0.052954137, 0.97092884, 1.31618514],
    "pclr_output_130": [-0.073753511, 1.21370738, 1.68694067],
    "pclr_output_131": [0.0, 1.53803222, 1.99452634],
    "pclr_output_132": [-0.100428634, 0.52984927, 1.18296353],
    "pclr_output_133": [0.0, 2.57623141, 2.62284165],
    "pclr_output_134": [-0.296134938, 0.98389123, 1.23573513],
    "pclr_output_135": [0.0, 0.94500145, 1.57151835],
    "pclr_output_136": [0.0, 1.27616561, 1.98959479],
    "pclr_output_137": [0.172884934, 1.30547747, 1.51019424],
    "pclr_output_138": [0.0, 2.8678706, 2.2323234],
    "pclr_output_139": [0.077565255, 2.76235613, 3.76744067],
    "pclr_output_140": [0.0, 1.5637931, 2.60129913],
    "pclr_output_141": [-0.026378633, 0.88729453, 1.56367503],
    "pclr_output_142": [-0.172093548, 1.46935428, 2.00407106],
    "pclr_output_143": [-0.39790882, 1.28525045, 2.33582865],
    "pclr_output_144": [0.0, 1.77892407, 1.75866619],
    "pclr_output_145": [0.826458833, 2.04814763, 1.60862135],
    "pclr_output_146": [0.019214348, 1.6000115, 2.75084721],
    "pclr_output_147": [0.574932504, 2.57763283, 3.69848824],
    "pclr_output_148": [0.0, 0.99148788, 2.00191424],
    "pclr_output_149": [0.0, 2.55460038, 3.11279115],
    "pclr_output_150": [0.0, 1.79892117, 2.70175277],
    "pclr_output_151": [0.0, 0.77464026, 1.63069855],
    "pclr_output_152": [0.0, 1.04272055, 1.88152508],
    "pclr_output_153": [0.0, 1.2331542, 1.68087813],
    "pclr_output_154": [0.0, 1.8250892, 1.79631829],
    "pclr_output_155": [0.143874096, 0.65601802, 1.20234136],
    "pclr_output_156": [0.0, 3.16240881, 2.91337172],
    "pclr_output_157": [0.094848425, 0.71628405, 0.90908515],
    "pclr_output_158": [0.0, 0.91956909, 1.417499],
    "pclr_output_159": [0.0, 1.83074409, 2.4406152],
    "pclr_output_160": [0.0, 1.33054252, 2.59029208],
    "pclr_output_161": [0.202124617, 1.58577965, 2.68631597],
    "pclr_output_162": [0.0, 1.62601641, 2.01044581],
    "pclr_output_163": [0.256048041, 0.88789717, 1.50340348],
    "pclr_output_164": [0.0, 2.7473667, 2.97231499],
    "pclr_output_165": [0.0, 1.71407966, 2.39094793],
    "pclr_output_166": [-0.336972015, 1.1585236, 1.49634562],
    "pclr_output_167": [0.0, 2.08649787, 2.15089086],  # CORRECTED
    "pclr_output_168": [-0.107063005, 2.2382583, 1.86179363],  # CORRECTED
    "pclr_output_169": [0.0, 1.48648736, 1.54097067],
    "pclr_output_170": [0.0, 1.21218912, 1.75811623],
    "pclr_output_171": [0.0, 2.6465834, 2.82028179],
    "pclr_output_172": [0.0, 0.58047895, 1.327316],
    "pclr_output_173": [0.0, 2.02382818, 2.28941592],
    "pclr_output_174": [-0.065399703, 1.2078755, 1.87520198],
    "pclr_output_175": [-0.000662706, 1.27169339, 2.06006406],
    "pclr_output_176": [0.0, 1.59824042, 2.90303982],
    "pclr_output_177": [0.0, 1.1068009, 2.4426194],
    "pclr_output_178": [0.0, 1.58224419, 2.77061207],
    "pclr_output_179": [0.0, 3.28058033, 3.35741155],
    "pclr_output_180": [-0.560585747, 1.68583606, 1.4741124],
    "pclr_output_181": [0.0, 2.08565284, 2.14458725],
    "pclr_output_182": [0.17775444, 1.89574055, 2.81992796],
    "pclr_output_183": [0.0, 1.56000152, 2.73451645],
    "pclr_output_184": [0.0, 0.96876186, 1.88779674],
    "pclr_output_185": [-0.000836845, 2.81754993, 2.96041107],
    "pclr_output_186": [0.0, 1.03661388, 1.83591157],
    "pclr_output_187": [0.0, 1.84195065, 1.83314684],
    "pclr_output_188": [2.389223522, 1.31977748, 0.98434834],
    "pclr_output_189": [0.0, -0.0035709, 0.00030383],
    "pclr_output_190": [0.0, 1.34144874, 1.72763435],
    "pclr_output_191": [0.0, 0.97435961, 1.76967294],
    "pclr_output_192": [0.0, 1.61478583, 2.75492907],
    "pclr_output_193": [0.0, 2.06612707, 2.41088639],
    "pclr_output_194": [-0.176657034, 0.85996735, 1.77239014],
    "pclr_output_195": [0.309723071, 1.60950189, 2.10126602],
    "pclr_output_196": [0.214402647, 1.8504626, 1.79150389],
    "pclr_output_197": [0.0, 1.24695937, 1.58555788],
    "pclr_output_198": [0.0, 1.74977497, 2.39732857],
    "pclr_output_199": [0.0, 1.28176471, 2.26469336],
    "pclr_output_200": [-0.482492783, 1.03810931, 1.56493019],
    "pclr_output_201": [0.0, 1.21872787, 1.91857419],
    "pclr_output_202": [-0.550437789, 0.76635199, 1.11742709],
    "pclr_output_203": [0.0, 1.5161407, 2.58956264],
    "pclr_output_204": [0.0, 1.05781086, 1.73620227],
    "pclr_output_205": [0.0, 1.23956647, 1.72241633],
    "pclr_output_206": [0.0, 2.51188011, 2.95634278],
    "pclr_output_207": [0.152625095, 1.59064999, 2.2918741],
    "pclr_output_208": [0.312393092, 1.52046325, 1.71839184],
    "pclr_output_209": [0.0, 1.21370108, 2.04509051],
    "pclr_output_210": [0.0, 2.61793718, 2.89303429],
    "pclr_output_211": [0.0, 1.29104923, 2.07972231],
    "pclr_output_212": [0.0, 0.82508018, 1.42076224],
    "pclr_output_213": [0.0, 1.16404858, 1.92743057],
    "pclr_output_214": [0.0, 0.81005354, 0.82054454],
    "pclr_output_215": [0.0, 4.15762242, 2.65655718],
    "pclr_output_216": [0.164148454, 3.56502383, 2.59295555],
    "pclr_output_217": [0.0, 0.94295533, 1.47206702],
    "pclr_output_218": [0.0, 1.77575066, 2.21806666],
    "pclr_output_219": [0.0, 0.57065153, 1.36046874],
    "pclr_output_220": [0.0, 1.63277328, 1.99107372],
    "pclr_output_221": [0.0, 1.55776371, 2.27202661],
    "pclr_output_222": [-0.364446455, 0.69466761, 1.24822662],
    "pclr_output_223": [0.0, 3.08495865, 1.89091379],
    "pclr_output_224": [0.0, 1.41791527, 1.96346636],
    "pclr_output_225": [0.017906687, 0.93487611, 1.95826968],
    "pclr_output_226": [0.0, 0.77833356, 1.58797746],
    "pclr_output_227": [0.0, 2.32439784, 2.98266723],
    "pclr_output_228": [0.0, 1.71807763, 2.19604182],
    "pclr_output_229": [0.0, 0.85145671, 1.45537594],
    "pclr_output_230": [0.0, 1.10810807, 1.70873582],
    "pclr_output_231": [-0.091760044, 0.38879164, 0.91827746],
    "pclr_output_232": [0.0, 2.20432062, 2.55923243],
    "pclr_output_233": [0.036225474, 1.81620796, 1.37280963],
    "pclr_output_234": [0.0, 1.3722781, 2.1366196],
    "pclr_output_235": [0.0, 4.09247849, 3.73273909],
    "pclr_output_236": [0.0, 0.73464125, 1.5309768],
    "pclr_output_237": [0.0, 1.11023142, 1.88604364],
    "pclr_output_238": [0.0, 2.03028006, 3.12048109],
    "pclr_output_239": [0.0, 2.01042009, 2.84193651],
    "pclr_output_240": [-0.736416949, 1.54472413, 1.64836188],
    "pclr_output_241": [-0.318409466, 0.57533346, 0.58830334],
    "pclr_output_242": [0.0, 3.55183312, 2.39984502],
    "pclr_output_243": [0.093378477, 1.93507461, 2.64401772],
    "pclr_output_244": [0.0, 1.07385615, 1.48135804],
    "pclr_output_245": [0.0, 2.52034655, 2.74213485],
    "pclr_output_246": [-0.24462345, 1.83880809, 2.40443235],
    "pclr_output_247": [0.0, 0.62454056, 1.4240379],
    "pclr_output_248": [-0.067551818, 0.50518713, 0.76666406],
    "pclr_output_249": [0.0, 1.24478152, 2.03116645],
    "pclr_output_250": [-0.372182713, 0.83310236, 1.26544451],
    "pclr_output_251": [0.103023923, 0.73188455, 1.70111294],
    "pclr_output_252": [-0.11760845, 1.36411913, 2.03772501],
    "pclr_output_253": [0.0, 1.42141383, 1.81290189],
    "pclr_output_254": [0.0, 0.68045075, 1.223293],
    "pclr_output_255": [0.494342806, 0.43137576, 0.41784607],
    "pclr_output_256": [0.0, 1.35323645, 1.66774545],
    "pclr_output_257": [-0.088791071, 1.113475, 1.63152075],
    "pclr_output_258": [0.308810041, 2.42203771, 1.93501869],
    "pclr_output_259": [0.0, 0.96163408, 1.60837465],
    "pclr_output_260": [0.0, 1.4477891, 1.99612219],
    "pclr_output_261": [0.0, 1.91235515, 2.37688321],
    "pclr_output_262": [0.0, 2.5397773, 3.84225955],
    "pclr_output_263": [0.0, 1.24676888, 1.7811911],
    "pclr_output_264": [0.0, 1.35991998, 2.06333389],
    "pclr_output_265": [0.0, 0.87590688, 2.19754439],
    "pclr_output_266": [0.0, 0.77121483, 1.2469901],
    "pclr_output_267": [0.0, 0.27853221, 0.80470138],
    "pclr_output_268": [0.067547936, 2.03975241, 2.54607399],
    "pclr_output_269": [0.0, 1.45582452, 2.1259793],
    "pclr_output_270": [-0.583560615, 1.69918892, 2.31399683],
    "pclr_output_271": [0.009426868, 1.76795604, 2.26602102],
    "pclr_output_272": [0.084236391, 1.14354069, 1.77799919],
    "pclr_output_273": [0.0, 1.91738901, 2.28801093],
    "pclr_output_274": [0.254437507, 1.53687354, 1.99075661],
    "pclr_output_275": [0.0, 1.70469552, 2.36183673],
    "pclr_output_276": [0.0, 1.66290779, 1.50635083],
    "pclr_output_277": [0.0, 1.52562434, 1.80712986],
    "pclr_output_278": [-0.223475127, 1.24331232, 1.09412075],
    "pclr_output_279": [0.0, 0.81707654, 1.57314035],
    "pclr_output_280": [0.0, 2.64791272, 2.55748077],
    "pclr_output_281": [0.0, 1.51246105, 2.56345493],
    "pclr_output_282": [0.0, 1.07537866, 2.56120448],
    "pclr_output_283": [0.0, 0.919191, 1.52221512],
    "pclr_output_284": [-0.125176766, 2.65323699, 2.95406033],
    "pclr_output_285": [0.0, 4.12285599, 2.5450327],
    "pclr_output_286": [0.0, 1.88453233, 2.02901022],
    "pclr_output_287": [0.0, 2.20660265, 3.27718336],
    "pclr_output_288": [0.0, 1.58234083, 1.55628715],
    "pclr_output_289": [0.066982135, 3.165761, 3.73816954],
    "pclr_output_290": [-0.162540521, 0.91668378, 1.63235108],
    "pclr_output_291": [0.0, 1.97850423, 2.31738996],
    "pclr_output_292": [0.0, 2.4413066, 2.75045641],
    "pclr_output_293": [0.0, 0.90256533, 1.4430694],
    "pclr_output_294": [0.0, 1.98496874, 2.3705703],
    "pclr_output_295": [-0.572563353, 0.7066295, 1.49376475],
    "pclr_output_296": [0.331301146, 1.17114315, 1.84294381],
    "pclr_output_297": [0.0, 2.81873428, 3.24764803],
    "pclr_output_298": [0.0, 1.1179844, 2.07518384],
    "pclr_output_299": [0.0, 1.29559086, 2.59263416],
    "pclr_output_300": [-0.030093789, 0.95783367, 1.43549894],
    "pclr_output_301": [0.0, 2.09916402, 2.70341324],
    "pclr_output_302": [0.0, 1.09273111, 2.14201764],
    "pclr_output_303": [-0.901736689, 0.86160328, 1.24487467],
    "pclr_output_304": [0.0, 2.24198872, 3.52615917],
    "pclr_output_305": [0.0, 1.05770183, 1.85989406],
    "pclr_output_306": [-0.724532796, 1.30018756, 1.8784644],
    "pclr_output_307": [0.0, 1.33051757, 2.00081861],
    "pclr_output_308": [0.0, 3.22203146, 3.61256236],
    "pclr_output_309": [0.0, 2.51763006, 2.93805914],
    "pclr_output_310": [0.0, 0.87541686, 1.79926013],
    "pclr_output_311": [0.0, 1.6580118, 1.85692637],
    "pclr_output_312": [0.0, 1.28232692, 2.14695401],
    "pclr_output_313": [-0.073808218, 1.03006776, 1.89352417],
    "pclr_output_314": [0.0, 2.94266915, 3.50212952],
    "pclr_output_315": [0.715020665, 0.9866166, 1.1738772],
    "pclr_output_316": [0.0, 0.53381676, 1.2381162],
    "pclr_output_317": [0.0, 1.66839289, 2.08838554],
    "pclr_output_318": [0.255754517, 0.95656172, 1.72580845],
    "pclr_output_319": [0.0, 1.37712642, 2.18924738],
}


# --- HELPER FUNCTIONS ---

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

def extract_features(data_frame: pd.DataFrame, encoder: Model, ecg_paths: np.ndarray, eids: np.ndarray):
    """
    Uses the PCLR encoder to extract 320-dim embeddings for all ECGs.
    output: dataframe with eid, pclr_output_0, ..., pclr_output_319, label (vo2 max)
    """
    print(f"\n--- Starting PCLR Feature Extraction for {len(eids)} samples ---")
    
    # 1. Create TensorFlow Dataset for ECGs
    # Pass eids as the 'label' so we can map predictions back to the DataFrame
    ecg_ds = tf.data.Dataset.from_tensor_slices((ecg_paths, eids))
    ecg_ds = (
        ecg_ds
        .map(tf_load_ecg, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # 2. Extract Embeddings (PCLR outputs)
    embeddings = encoder.predict(ecg_ds, verbose=1)
    
    # 3. Get EIDs corresponding to the embeddings (order is preserved by predict)
    predicted_eids = eids[:len(embeddings)]
    
    # 4. Create embedding DataFrame
    emb_df = pd.DataFrame(embeddings, index=predicted_eids)
    emb_df.index.name = 'eid'
    
    # 5. Rename columns to match fixed linear model parameters (pclr_output_0 to pclr_output_319)
    column_names = {i: f'pclr_output_{i}' for i in range(embeddings.shape[1])}
    emb_df = emb_df.rename(columns=column_names).reset_index()

    merged_df = data_frame.merge(emb_df, on='eid', how='inner')
    
    return merged_df

def deep_ecg_vo2_predict(data: pd.DataFrame, params: dict) -> pd.Series:
    """
    Predicts VO2 using the fixed coefficients and scale factors of the Deep ECG-VO2 model.
    The formula is: VO2_predicted = Intercept + sum(Coefficient_i * Scaled_Feature_i)
    """
    
    intercept_coef = params['(Intercept)'][0]
    print(f"interpect = {intercept_coef}")
    predictions = pd.Series([intercept_coef] * len(data), index=data.index)

    # --- 1. Clinical Feature Preparation --- 
    # Since ALL participants used the bike, their prediction should include the bike coefficient.
    data['test_type_bike'] = 1.0 
    data['test_type_treadmill'] = 0.0
    data['test_type_rower'] = 0.0
    
    # --- 2. Iterate and apply linear model ---
    for term, (coef, mean, var) in params.items():
        if term == '(Intercept)':
            continue
        
        if coef == 0.0:
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
    
    # Metrics
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    
    r2 = r2_score(true_values, predicted_values)
    r_matrix = np.corrcoef(true_values, predicted_values)
    r = r_matrix[0, 1]
    
    print("\n--- Model Evaluation Metrics ---")
    print(f"Target Variable: {target_label}")
    print("------------------------------")
    print(f"Pearson Correlation (R): {r:.4f}")
    print(f"Coefficient of Determination (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("------------------------------")
    return {'R': r, 'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}

# --- 4. MAIN EXECUTION ---

if __name__ == '__main__':
    
    # Setup TensorFlow environment
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # 1. Load clinical and label data
    try:
        labels_df = pd.read_csv(LABELS_FILE)
        labels_df = labels_df.rename(columns={'eid': 'eid'})

        clinical_df = pd.read_csv(AGE2_SEX_BMI2_PATH)
        # Keep the following line for encoding (0= Male, 1= Female), otherwiese delete it
        clinical_df['sex'] = 1 - clinical_df['sex']
        
        full_labels_df = labels_df.merge(clinical_df, on='eid', how='inner')
        full_labels_df['eid'] = full_labels_df['eid'].astype(int)


    except Exception as e:
        print(f"\nERROR: Could not load data files. Check paths and formats.")
        print(f"Detail: {e}")
        exit()
    
    # 2. Map ECG paths and filter to available data
    ecg_paths = []
    eids = []
    
    for eid in full_labels_df['eid'].unique():
        path = os.path.join(ECG_PATH, f"{eid}.npy")
        if os.path.exists(path):
            ecg_paths.append(path)
            eids.append(eid)
    
    ecg_paths = np.array(ecg_paths)
    eids = np.array(eids)
    full_data_for_prediction = full_labels_df[full_labels_df['eid'].isin(eids)].copy()

    print(f"Loaded {len(eids)} ECG samples with corresponding clinical/label data.")

    # 3. Load the PCLR Encoder model
    try:
        pclr_encoder = load_model(MODEL_PATH, compile=False)
        
    except Exception as e:
        print(f"\nERROR: Could not load PCLR model from {MODEL_PATH}.")
        print(f"Detail: {e}")
        exit()

    # 4. Extract PCLR Features and combine with clinical data
    full_data_for_prediction = extract_features(full_data_for_prediction, pclr_encoder, ecg_paths, eids)

    # 5. Apply Fixed Linear Model to get predictions
    predictions = deep_ecg_vo2_predict(full_data_for_prediction.copy(), DEEP_ECG_VO2_PARAMS)
    
    # 6. Save results
    full_data_for_prediction.loc[:, 'vo2_predicted_fixed_linear'] = predictions.values
    

    # 7. Evaluate Metrics
    if TARGET_LABEL in full_data_for_prediction.columns:
        true_values = full_data_for_prediction[TARGET_LABEL].astype(float)
        valid_indices = true_values.notna()

        evaluate_metrics(
            true_values=true_values[valid_indices],
            predicted_values=predictions[valid_indices],
            target_label=TARGET_LABEL
        )
    else:
        print(f"Cannot calculate metrics. Target column '{TARGET_LABEL}' not found or is empty.")


    output_path = OUTPUT_FILE 
    
    save_cols = ['eid', TARGET_LABEL, 'vo2_predicted_fixed_linear']
    save_cols_existing = [col for col in save_cols if col in full_data_for_prediction.columns]
    
    full_data_for_prediction[save_cols_existing].to_csv(output_path, index=False)
    print(f"\nPrediction complete. Results saved to: {output_path}")
