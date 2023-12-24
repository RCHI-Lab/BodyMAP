import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from constants import *

# PME Model 16 - PointNet 


class PointNetFeat(nn.Module):

    def __init__(self, in_size=3):
        super(PointNetFeat, self).__init__()
        self.fc1 = nn.Linear(in_features=in_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=512)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, vert_features, global_im_features=None):
        B, N, D = vert_features.shape 
        
        out = self.relu(self.fc1(vert_features))
        out = self.relu(self.fc2(out))
        global_feat = out.clone()                       # B, N, 64
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = torch.max(out, dim=1)[0]          
        out = out.unsqueeze(dim=1).repeat(1, N, 1)      # B, N, 512
        if global_im_features is not None:
            global_im_features = global_im_features.unsqueeze(dim=1).repeat(1, N, 1)
            out = torch.cat((global_im_features, global_feat, out), dim=-1)
        else:
            out = torch.cat((global_feat, out), dim=-1)
        return out


class PMEstimator(nn.Module):
    
    def __init__(self, modality, vertex_size, indexing_mode=0, use_contact=False):
        super(PMEstimator, self).__init__()
        if modality == 'both':
            mod_size = 2
        else:
            mod_size = 1
        self.indexing_mode = indexing_mode
        indexing_dict = self._prep_indexing_dict(mod_size)
        in_size, fc1_size = indexing_dict[indexing_mode]

        self.use_contact = use_contact

        self.pnf = PointNetFeat(in_size)

        self.fc1 = nn.Linear(in_features=fc1_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)

        if indexing_mode in [3, 5, 8, 9, 12, 13, 15, 16]:
            self.fc_enc = nn.Linear(in_features=512, out_features=16)

        self.fc_pmap = nn.Linear(in_features=64, out_features=1)
        if self.use_contact:
            self.fc_contact = nn.Linear(in_features=64, out_features=2)
        else:
            self.fc_contact = None
        
        self.relu = nn.LeakyReLU(0.1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def _prep_indexing_dict(self, mod_size):
        return {
            0 : (3, 576),                               # only_verts
            1 : (3 + mod_size, 576),                    # verts + start_features
            3 : (3 + 16, 576),                          # verts + enc_red_features
            5 : (3 + mod_size + 16, 576),               # verts + start_features + enc_red_features
            6 : (3, 576 + 512),                         # verts | gl_features
            7 : (3 + mod_size, 576 + 512),              # verts + start_features | gl_features
            8 : (3 + 16, 576 + 512),                    # verts + enc_red_features | gl_features
            9 : (3 + mod_size + 16, 576 + 512),         # verts + start_features + enc_red_features | gl_features
            11: (mod_size, 576),                        # start_features
            12: (mod_size + 16, 576),                   # start_features + enc_red_features
            13: (mod_size + 16, 576 + 512),             # start_features + enc_red_features | gl_features
            14: (mod_size, 576 + 512),                  # start_features | gl_features
            15: (16, 576),                              # enc_red_features 
            16: (16, 576 + 512)                         # enc_red_features | gl_features
        }

    def _get_features(self, smpl_verts, start_features, encoder_features, lp_features, gl_features):
        if self.indexing_mode == 0:          # only_verts
            pnf_features = smpl_verts
            global_features = None
        elif self.indexing_mode == 1:        # verts + start_features
            pnf_features = torch.cat((smpl_verts, start_features), dim=-1)
            global_features = None
        elif self.indexing_mode == 3:        # verts + enc_red_features
            encoder_features = self.fc_enc(encoder_features)
            pnf_features = torch.cat((smpl_verts, encoder_features), dim=-1)
            global_features = None
        elif self.indexing_mode == 5:        # verts + start_features + enc_red_features
            encoder_features = self.fc_enc(encoder_features)
            pnf_features = torch.cat((smpl_verts, start_features, encoder_features), dim=-1)
            global_features = None
        elif self.indexing_mode == 6:        # verts | gl_features
            pnf_features = smpl_verts
            global_features = gl_features
        elif self.indexing_mode == 7:        # verts + start_features | gl_features
            pnf_features = torch.cat((smpl_verts, start_features), dim=-1)
            global_features = gl_features
        elif self.indexing_mode == 8:        # verts + enc_red_features | gl_features
            encoder_features = self.fc_enc(encoder_features)
            pnf_features = torch.cat((smpl_verts, encoder_features), dim=-1)
            global_features = gl_features
        elif self.indexing_mode == 9:        # verts + start_features + enc_red_features | gl_features
            encoder_features = self.fc_enc(encoder_features)
            pnf_features = torch.cat((smpl_verts, start_features, encoder_features), dim=-1)
            global_features = gl_features
        elif self.indexing_mode == 11:       # start_features
            pnf_features = start_features
            global_features = None
        elif self.indexing_mode == 12:       # start_features + encoder_features
            encoder_features = self.fc_enc(encoder_features)
            pnf_features = torch.cat((start_features, encoder_features), dim=-1)
            global_features = None
        elif self.indexing_mode == 13:       # start_features + encoder_features | gl_features
            encoder_features = self.fc_enc(encoder_features)
            pnf_features = torch.cat((start_features, encoder_features), dim=-1)
            global_features = gl_features
        elif self.indexing_mode == 14:       # start_features | gl_features
            pnf_features = start_features
            global_features = gl_features
        elif self.indexing_mode == 15:       # encoder_features
            encoder_features = self.fc_enc(encoder_features)
            pnf_features = encoder_features
            global_features = None
        elif self.indexing_mode == 16:       # encoder_features | gl_features
            encoder_features = self.fc_enc(encoder_features)
            pnf_features = encoder_features
            global_features = gl_features
        else:
            print ('ERROR: invalid indexing mode')
            exit(-1)
        
        return pnf_features, global_features

    def forward(self, smpl_verts, start_features, encoder_features, lp_features, gl_features):
        pnf_features, global_features = self._get_features(smpl_verts, 
                                                            start_features, 
                                                            encoder_features, 
                                                            lp_features, 
                                                            gl_features)
        out = self.pnf(pnf_features, global_features)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        in_pmap, in_contact = out.clone(), out.clone()
        out_pmap = self.fc_pmap(in_pmap).squeeze(-1)
        if self.use_contact:
            out_contact = self.fc_contact(in_contact).permute(0, 2, 1)
            out_contact = self.log_softmax(out_contact)
        else:
            out_contact = None
        return out_pmap, out_contact

    def infer(self, smpl_verts, start_features, encoder_features, lp_features, gl_features):
        out_pmap, log_contact = self.forward(smpl_verts, start_features, encoder_features, lp_features, gl_features)
        if self.use_contact:
            contact = log_contact.argmax(dim=1)
            out_pmap = out_pmap * contact
        return out_pmap, log_contact


if __name__ == '__main__':

    image_size_type = 'resized224' # ['resized224', 'original']
    feature_size = 392 if image_size_type == 'resized224' else 64
    vertex_size = 6890
    use_contact = True
    indexing_mode = 7
    modality = 'both'

    model = PMEstimator(
                        modality,
                        vertex_size,
                        indexing_mode=indexing_mode,
                        use_contact=use_contact
                        ).to(DEVICE)

    smpl_verts = torch.rand(2, 6890, 3).to(DEVICE)
    start_features = torch.rand(2, 6890, 2).to(DEVICE)
    encoder_features = torch.rand(2, 6890, 512).to(DEVICE)
    lp_features = torch.rand(2, 6890, 8).to(DEVICE)
    gl_features = torch.rand(2, 512).to(DEVICE)
    
    with torch.no_grad():
        pmap, contact = model.infer(smpl_verts, start_features, encoder_features, lp_features, gl_features)
        print ('pmap', pmap.shape)
        print ('contact', contact.shape)

