import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from constants import *

''' WS - pose  '''


class PMEstimator(nn.Module):
    
    def __init__(self, feature_size, out_size, vertex_size, modality):
        super(PMEstimator, self).__init__()
        self.out_size = out_size
        self.vertex_size = vertex_size
        self.modality = modality
        self.in_channels = 2 if self.modality == 'both' else 1
        self.encoder = models.__dict__['resnet18'](pretrained=False)
        self.encoder.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, \
                                            kernel_size=(7, 7), stride=(2, 2), \
                                            padding=(3, 3), bias=False) 
        self.encoder = nn.Sequential(*list(self.encoder.children()))[:-2]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=512, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=out_size)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)

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
    
    def forward(self, depth_map, pressure_map):
        '''
        Estimated params - shape - [B, 88]
        0-10    Body Shape (10)
        10-13   root xyz (3)
        13-19   root angle (atan2 3)
        19-88   joint angles (69)
        '''
        x = self._prep_input(depth_map, pressure_map)
        out = self.encoder(x)
        img_feat = out.clone()         # [B, 512, 7, 7]
        out = self.global_pool(out)            
        out = self.flatten(out)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))       
        smpl_pred = self.fc3(out)
        return smpl_pred, None, img_feat 

    def infer(self, depth_map, pressure_map):
        return self.forward(depth_map, pressure_map)


if __name__ == '__main__':

    image_size_type = 'resized224' # ['resized224', 'original']
    feature_size = 392 if image_size_type == 'resized224' else 64
    modality = 'both'           # ['depth', 'pressure', 'both']
    out_size = 88
    vertex_size = 6890

    model = PMEstimator(
                        feature_size, 
                        out_size, 
                        vertex_size,
                        modality).to(DEVICE)


    if image_size_type == 'resized224':
        depth_map = torch.rand(2, 1, 224, 224).to(DEVICE)
        pressure_map = torch.rand(2, 1, 224, 224).to(DEVICE)
    else:
        depth_map = torch.rand(2, 1,  128, 54).to(DEVICE)
        pressure_map = torch.rand(2, 1, 64, 27).to(DEVICE)

    with torch.no_grad():
        final_params, final_pmap, img_feat = model.infer(depth_map, pressure_map)
        print ('final_params', final_params.shape)
        # print ('final pmap', final_pmap.shape)
        print ('img_feat', img_feat.shape)

