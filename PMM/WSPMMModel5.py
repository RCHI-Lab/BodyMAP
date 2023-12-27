import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as models

from constants import DEVICE, MODALITY_TO_FEATURE_SIZE, X_BUMP, Y_BUMP
from MeshEstimator import MeshEstimator

from PMEModel16 import PMEstimator as PME16


MODEL_FN_DICT = {
    'PME16' : PME16,
}


class WSPMMModel(nn.Module):

    def __init__(self, model_fn, feature_size, out_size, vertex_size, batch_size, modality, indexing_mode=0):
        super(WSPMMModel, self).__init__()
        model_type = MODEL_FN_DICT.get(model_fn, None)
        if model_type is None:
            print (f'ERROR model_fn {model_fn} is not valid')
            exit(-1)
        self.modality = modality
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.upsample = nn.Upsample(scale_factor=2)
        self.pme = model_type(modality, vertex_size, indexing_mode=indexing_mode, use_contact=False)
        self.use_z = True

    def _prep_input(self, depth_map, pressure_map):
        if self.modality == 'pressure':
            x = pressure_map
        elif self.modality == 'depth':
            x = depth_map
        else:
            if pressure_map.shape != depth_map.shape:
                pressure_map = self.upsample(pressure_map)
            x = torch.cat((depth_map, pressure_map), dim=1)
        return x

    def _get_grid(self, verts):
        Y = 64 - (verts[:, :, 0:1] - Y_BUMP)/ (0.0286*1.04)
        X = (verts[:, :, 1:2] - X_BUMP)/(0.0286)
        Y = torch.clamp(Y, min=0, max=63)
        X = torch.clamp(X, min=0, max=26)
        Y = 2*(Y/63) - 1
        X = 2*(X/26) - 1
        return torch.cat((X, Y), dim=-1).unsqueeze(-2)

    def _prep_features(self, smpl_verts, image_features, encoder_features, local_features):
        grid = self._get_grid(smpl_verts.clone())
        start_features = F.grid_sample(image_features, grid, align_corners=True).squeeze(-1).permute(0, 2, 1)
        encoder_features = F.grid_sample(encoder_features, grid, align_corners=True).squeeze(-1).permute(0, 2, 1)
        lp_features = F.grid_sample(local_features.clone(), grid, align_corners=True).squeeze(-1).permute(0, 2, 1)
        gl_features = self.flatten(local_features)
        if self.use_z:
            start_features[smpl_verts[:, :, -1] < 0] = 0.0
            encoder_features[smpl_verts[:, :, -1] < 0] = 0.0
            lp_features[smpl_verts[:, :, -1] < 0] = 0.0
        return start_features, encoder_features, lp_features, gl_features

    def forward(self, depth_map, pressure_map, gender, smpl_verts, img_feat):
        image_features = self._prep_input(depth_map, pressure_map).clone()
        encoder_features = img_feat.clone() 
        local_features = self.global_pool(encoder_features)

        start_features, encoder_features, lp_features, gl_features = self._prep_features(smpl_verts, image_features, encoder_features, local_features)

        pmap_pred, _ = self.pme(smpl_verts, start_features, encoder_features, lp_features, gl_features)
        
        return None, pmap_pred, None, None

    def infer(self, depth_map, pressure_map, gender, smpl_verts, img_feat):
        return self.forward(depth_map, pressure_map, gender, smpl_verts, img_feat)


if __name__ == '__main__':
    model_fn = 'PME16'
    image_size_type = 'resized224' # ['resized224', 'original']
    modality = 'both'
    feature_size = 392 if image_size_type == 'resized224' else MODALITY_TO_FEATURE_SIZE[modality]
    out_size = 88
    vertex_size = 6890
    batch_size = 2
    indexing_mode = 9

    model = WSPMMModel(
        model_fn, 
        feature_size,
        out_size, 
        vertex_size,
        batch_size, 
        modality,
        indexing_mode).to(DEVICE)

    if image_size_type == 'resized224':
        depth_map = torch.rand(2, 1, 224, 224).to(DEVICE)
        pressure_map = torch.rand(2, 1, 224, 224).to(DEVICE)
    else:
        depth_map = torch.rand(2, 1,  128, 54).to(DEVICE)
        pressure_map = torch.rand(2, 1, 64, 27).to(DEVICE)
    
    img_feat = torch.rand(2, 512, 7, 7).to(DEVICE)
    smpl_verts = torch.rand(2, 6890, 3).to(DEVICE)
    
    with torch.no_grad():
        batch_gender = torch.tensor([[1, 0], [0, 1]])
        mesh_pred, pmap_pred, img_feat, smpl_pred = model(depth_map, pressure_map, batch_gender, smpl_verts, img_feat)

    print ('pmap pred', pmap_pred.shape)

