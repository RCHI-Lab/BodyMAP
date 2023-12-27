import numpy as np

import torch 
import torch.nn.functional as F
from constants import MAX_PMAP_REAL, FACES, DEVICE


X_BUMP = -0.0143*2
Y_BUMP = -(0.0286*64*1.04 - 0.0286*64)/2 - 0.0143*2
EPS = torch.tensor([1e-8]).to(DEVICE)
FACES_RENDER = FACES.long()


def get_projected_pressure(verts, pmap):
    verts[:, :, 0] = 64 - (verts[:, :, 0] - Y_BUMP)/ (0.0286*1.04)
    verts[:, :, 1] = (verts[:, :, 1] - X_BUMP)/(0.0286)
    verts[:, :, 0] = verts[:, :, 0] + verts[:, :, 0].floor().detach() - verts[:, :, 0].detach()
    verts[:, :, 1] = verts[:, :, 1] + verts[:, :, 1].floor().detach() - verts[:, :, 1].detach()

    projected_pressure = torch.zeros((verts.shape[0], 1, 64, 27)).to(DEVICE)

    for i in range(64):
        for j in range(27):
            mask = ((verts[:, :, 0] == i) & (verts[:, :, 1] == j) & (verts[:, :, 2] >= 0)).float()
            projected_pressure[:, 0, i, j] = (pmap*mask).sum(dim=-1)/(mask.sum(dim=-1) + EPS)
    return torch.nan_to_num(projected_pressure)*MAX_PMAP_REAL

