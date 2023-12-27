import torch 
import torch.nn as nn 

from constants import DEVICE, MODALITY_TO_FEATURE_SIZE
from MeshEstimator import MeshEstimator
from WSPMEModel1 import PMEstimator as WSPME1


MODEL_FN_DICT = {
    'WSPME1' : WSPME1, 
}


class WSPMMModel(nn.Module):

    def __init__(self, model_fn, feature_size, out_size, vertex_size, batch_size, modality, indexing_mode=0):
        super(WSPMMModel, self).__init__()
        model_type = MODEL_FN_DICT.get(model_fn, None)
        if model_type is None:
            print (f'ERROR model_fn {model_fn} is not valid')
            exit(-1)
        self.pme = model_type(feature_size, out_size, vertex_size, modality)
        self.mesh_model = MeshEstimator(batch_size)
    
    def forward(self, depth_map, pressure_map, batch_gender):
        smpl_pred, pmap_pred, img_feat = self.pme(depth_map, pressure_map)
        if smpl_pred is not None:
            mesh_pred = self.mesh_model(smpl_pred.clone(), batch_gender)
        else:
            mesh_pred = None
        return mesh_pred, pmap_pred, img_feat, smpl_pred

    def infer(self, depth_map, pressure_map, batch_gender):
        smpl_pred, pmap_pred, img_feat = self.pme.infer(depth_map, pressure_map)
        if smpl_pred is not None:
            mesh_pred = self.mesh_model.infer(smpl_pred.clone(), batch_gender)
        else:
            mesh_pred = None
        return mesh_pred, pmap_pred, img_feat, smpl_pred

    def mesh_infer_gt(self, x_gt, batch_gender):
        with torch.no_grad():
            x_gt = x_gt.to(DEVICE)
            mesh_pred = self.mesh_model.infer(x_gt, batch_gender, is_gt=True)
        return mesh_pred


if __name__ == '__main__':
    model_fn = 'WSPME1'
    image_size_type = 'resized224' # ['resized224', 'original']
    modality = 'both'
    feature_size = 392 if image_size_type == 'resized224' else MODALITY_TO_FEATURE_SIZE[modality]
    out_size = 88
    vertex_size = 6890
    batch_size = 2
    indexing_mode = 0

    model = WSPMMModel(
        model_fn, 
        feature_size,
        out_size, 
        vertex_size,
        batch_size, 
        modality,
        indexing_mode=indexing_mode).to(DEVICE)

    if image_size_type == 'resized224':
        depth_map = torch.rand(2, 1, 224, 224).to(DEVICE)
        pressure_map = torch.rand(2, 1, 224, 224).to(DEVICE)
    else:
        depth_map = torch.rand(2, 1,  128, 54).to(DEVICE)
        pressure_map = torch.rand(2, 1, 64, 27).to(DEVICE)
    
    with torch.no_grad():
        batch_gender = torch.tensor([[1, 0], [0, 1]])
        mesh_pred, pmap_pred, img_feat, smpl_pred = model(depth_map, pressure_map, batch_gender)

    if pmap_pred is not None:
        print ('pmap pred', pmap_pred.shape)
    if img_feat is not None:
        print ('img_feat', img_feat.shape)
    if smpl_pred is not None:
        print ('smpl pred', smpl_pred.shape)
        for k in mesh_pred:
            print (k, mesh_pred[k].shape)

