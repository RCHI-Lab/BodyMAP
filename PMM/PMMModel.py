import torch 
import torch.nn as nn 

from constants import DEVICE, MODALITY_TO_FEATURE_SIZE
from MeshEstimator import MeshEstimator
from PMEModel1 import PMEstimator as PME1
from PMEModel13 import PMEstimator as PME13


MODEL_FN_DICT = {
    'PME1' : PME1, 
    'PME13': PME13,
}


class PMMModel(nn.Module):

    def __init__(self, model_fn, feature_size, out_size, vertex_size, batch_size, modality, use_contact=False, indexing_mode=0):
        super(PMMModel, self).__init__()
        model_type = MODEL_FN_DICT.get(model_fn, None)
        if model_type is None:
            print (f'ERROR model_fn {model_fn} is not valid')
            exit(-1)
        self.pme = model_type(feature_size, out_size, vertex_size, modality, use_contact)
        self.mesh_model = MeshEstimator(batch_size)
    
    def forward(self, depth_map, pressure_map, batch_gender):
        smpl_pred, pmap_pred, contact_pred = self.pme(depth_map, pressure_map)
        mesh_pred = self.mesh_model.infer(smpl_pred, batch_gender)
        return mesh_pred, pmap_pred, contact_pred, smpl_pred

    def infer(self, depth_map, pressure_map, batch_gender):
        smpl_pred, pmap_pred, contact_pred = self.pme.infer(depth_map, pressure_map)
        mesh_pred = self.mesh_model.infer(smpl_pred, batch_gender)
        return mesh_pred, pmap_pred, contact_pred, smpl_pred

    def mesh_infer_gt(self, x_gt, batch_gender):
        with torch.no_grad():
            x_gt = x_gt.to(DEVICE)
            mesh_pred = self.mesh_model.infer(x_gt, batch_gender, is_gt=True)
        return mesh_pred


if __name__ == '__main__':
    model_fn = 'PME13'
    image_size_type = 'resized224' # ['resized224', 'original']
    modality = 'both'
    feature_size = 392 if image_size_type == 'resized224' else MODALITY_TO_FEATURE_SIZE[modality]
    out_size = 88
    vertex_size = 6890
    batch_size = 2
    use_contact = True

    model = PMMModel(
        model_fn, 
        feature_size,
        out_size, 
        vertex_size,
        batch_size, 
        modality, 
        use_contact).to(DEVICE)

    if image_size_type == 'resized224':
        depth_map = torch.rand(2, 1, 224, 224).to(DEVICE)
        pressure_map = torch.rand(2, 1, 224, 224).to(DEVICE)
    else:
        depth_map = torch.rand(2, 1,  128, 54).to(DEVICE)
        pressure_map = torch.rand(2, 1, 64, 27).to(DEVICE)
    
    with torch.no_grad():
        batch_gender = torch.tensor([[1, 0], [0, 1]])
        mesh_pred, pmap_pred, contact_pred, smpl_pred = model(depth_map, pressure_map, batch_gender)

    if pmap_pred is not None:
        print ('pmap pred', pmap_pred.shape)
    if contact_pred is not None:
        print ('contact_pred', contact_pred.shape)
    if smpl_pred is not None:
        print ('smpl pred', smpl_pred.shape)
        for k in mesh_pred:
            print (k, mesh_pred[k].shape)

