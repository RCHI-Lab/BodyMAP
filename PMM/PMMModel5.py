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


class PMMModel(nn.Module):

    def __init__(self, model_fn, feature_size, out_size, vertex_size, batch_size, modality, use_contact=False, indexing_mode=0):
        super(PMMModel, self).__init__()
        model_type = MODEL_FN_DICT.get(model_fn, None)
        if model_type is None:
            print (f'ERROR model_fn {model_fn} is not valid')
            exit(-1)
        self.modality = modality
        self.out_size = out_size 
        self.in_channels = 2 if self.modality == 'both' else 1
        self.encoder = models.__dict__['resnet18'](pretrained=False)
        self.encoder.conv1 = nn.Conv2d(
            in_channels=self.in_channels, out_channels=64, \
            kernel_size=(7, 7), stride=(2, 2), \
            padding=(3, 3), bias=False
        )
        self.encoder = nn.Sequential(*list(self.encoder.children()))[:-2]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=512, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=out_size)
        self.mesh_model = MeshEstimator(batch_size)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)
        self.pme = model_type(modality, vertex_size, indexing_mode=indexing_mode, use_contact=use_contact)
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

    def _forward_smpl(self, depth_map, pressure_map):
        x = self._prep_input(depth_map, pressure_map)
        image_features = x.clone()             # [B, M, 224, 224]

        out = self.encoder(x)
        encoder_features = out.clone()         # [B, 512, 7, 7] 
        out = self.global_pool(out)            
        local_features = out.clone()           # [B, 512, 1, 1]
        out = self.flatten(out)                # [B, 512]    

        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        smpl_pred = self.fc3(out)
        return smpl_pred, local_features, encoder_features, image_features

    def _get_grid(self, verts):
        Y = 64 - (verts[:, :, 0:1] - Y_BUMP)/ (0.0286*1.04)
        X = (verts[:, :, 1:2] - X_BUMP)/(0.0286)
        Y = torch.clamp(Y, min=0, max=63) # scale to pressure mat size of 64 x 27
        X = torch.clamp(X, min=0, max=26)
        Y = 2*(Y/63) - 1 # scale to [-1, 1] to scale to any size
        X = 2*(X/26) - 1 
        return torch.cat((X, Y), dim=-1).unsqueeze(-2)

    def _prep_features(self, smpl_verts, image_features, encoder_features, local_features):
        grid = self._get_grid(smpl_verts.clone())
        start_features = F.grid_sample(image_features, grid, align_corners=True).squeeze(-1).permute(0, 2, 1)
        encoder_features = F.grid_sample(encoder_features, grid, align_corners=True).squeeze(-1).permute(0, 2, 1)
        lp_features = F.grid_sample(local_features.clone(), grid, align_corners=True).squeeze(-1).permute(0, 2, 1)
        gl_features = self.flatten(local_features)
        if self.use_z: # set the values for the vertices which are on top (z is reversed so checking for negative) as zero
            start_features[smpl_verts[:, :, -1] < 0] = 0.0
            encoder_features[smpl_verts[:, :, -1] < 0] = 0.0
            lp_features[smpl_verts[:, :, -1] < 0] = 0.0
        return start_features, encoder_features, lp_features, gl_features

    def forward(self, depth_map, pressure_map, gender):
        smpl_pred, local_features, encoder_features, image_features = self._forward_smpl(depth_map, pressure_map)
        mesh_pred = self.mesh_model.infer(smpl_pred, gender)
        smpl_verts = mesh_pred['out_verts'].clone() # [B, 6890, 3]

        start_features, encoder_features, lp_features, gl_features = self._prep_features(smpl_verts, 
                                                                                        image_features, 
                                                                                        encoder_features, 
                                                                                        local_features)

        pmap_pred, contact_pred = self.pme(smpl_verts, start_features, encoder_features, lp_features, gl_features)
        
        return mesh_pred, pmap_pred, contact_pred, smpl_pred

    def infer(self, depth_map, pressure_map, gender):
        smpl_pred, local_features, encoder_features, image_features = self._forward_smpl(depth_map, pressure_map)
        mesh_pred = self.mesh_model.infer(smpl_pred, gender)
        smpl_verts = mesh_pred['out_verts'].clone() # [B, 6890, 3]

        start_features, encoder_features, lp_features, gl_features = self._prep_features(smpl_verts, 
                                                                                        image_features, 
                                                                                        encoder_features, 
                                                                                        local_features)

        pmap_pred, contact_pred = self.pme.infer(smpl_verts, start_features, encoder_features, lp_features, gl_features)
        
        return mesh_pred, pmap_pred, contact_pred, smpl_pred

    def mesh_infer_gt(self, x_gt, batch_gender):
        with torch.no_grad():
            x_gt = x_gt.to(DEVICE)
            mesh_pred = self.mesh_model.infer(x_gt, batch_gender, is_gt=True)
        return mesh_pred


if __name__ == '__main__':
    model_fn = 'PME16'
    image_size_type = 'resized224' # ['resized224', 'original']
    modality = 'both'
    feature_size = 392 if image_size_type == 'resized224' else MODALITY_TO_FEATURE_SIZE[modality]
    out_size = 88
    vertex_size = 6890
    batch_size = 2
    use_contact = True
    indexing_mode = 7

    model = PMMModel(
        model_fn, 
        feature_size,
        out_size, 
        vertex_size,
        batch_size, 
        modality, 
        use_contact, 
        indexing_mode).to(DEVICE)

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

