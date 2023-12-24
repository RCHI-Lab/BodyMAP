import numpy as np
import torch
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

MAX_PRESSURE_SYNTH = 100.0 # 170.0 # synth
MAX_DEPTH_SYNTH = 2215.0 # synth
MAX_PMAP_SYNTH = 100.0
MAX_PRESSURE_REAL = 100.0 # 620.0 # real 
MAX_DEPTH_REAL = 2575.0 # real
MAX_PMAP_REAL = 100.0

USE_TILL = 1024
USE_FOR_INFER = 5

SMPL_FEMALE = SMPL_Layer(
    center_idx=0,
    gender='female',
    model_root='../smpl_models/smpl').to(DEVICE)

SMPL_MALE = SMPL_Layer(
    center_idx=0,
    gender='male',
    model_root='../smpl_models/smpl').to(DEVICE)

POSE = torch.tensor(SMPL_FEMALE.smpl_data['pose'].r).unsqueeze(0).float().to(DEVICE)

FACES = SMPL_FEMALE.th_faces.unsqueeze(0).int().to(DEVICE)

MODALITY_TO_FEATURE_SIZE = {
    'both' : 64, 
    'depth' : 64,
    'pressure' : 16,
}

X_BUMP = -0.0143*2
Y_BUMP = -(0.0286*64*1.04 - 0.0286*64)/2 - 0.0143*2

# Set your base path which has the file structure as shown in the readme
BASE_PATH = '/home/yeojinj/Desktop/abhishek'