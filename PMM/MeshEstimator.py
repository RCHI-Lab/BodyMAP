import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from constants import *
import angle_utils


class MeshEstimator(nn.Module):
    
    def __init__(self, batch_size):
        super(MeshEstimator, self).__init__()
        self.bounds = 2 * torch.Tensor(
            np.array([[-0.5933865286111969, 0.5933865286111969], [-2*np.pi, 2*np.pi], [-1.215762200416361, 1.215762200416361],
                        [-1.5793940868065197, 0.3097956806], [-0.5881754611, 0.5689768556],[-0.5323249722, 0.6736965222],
                        [-1.5793940868065197, 0.3097956806], [-0.5689768556, 0.5881754611],[-0.6736965222, 0.5323249722],
                        [-np.pi / 3, np.pi / 3], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],
                        [-0.02268926111, 2.441713561], [-0.01, 0.01], [-0.01, 0.01],  # knee
                        [-0.02268926111, 2.441713561], [-0.01, 0.01], [-0.01, 0.01],
                        [-np.pi / 3, np.pi / 3], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],
                        [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],
                        # ankle, pi/36 or 5 deg
                        [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],
                        # ankle, pi/36 or 5 deg
                        [-np.pi / 3, np.pi / 3], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],
                        [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01],  # foot
                        [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01],  # foot

                        [-np.pi / 3, np.pi / 3], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],  # neck
                        [-1.551596394 * 1 / 3, 2.206094311 * 1 / 3],  [-2.455676183 * 1 / 3, 0.7627082389 * 1 / 3],  [-1.570795 * 1 / 3, 2.188641033 * 1 / 3],
                        [-1.551596394 * 1 / 3, 2.206094311 * 1 / 3],  [-0.7627082389 * 1 / 3, 2.455676183 * 1 / 3],  [-2.188641033 * 1 / 3, 1.570795 * 1 / 3],
                        [-np.pi / 3, np.pi / 3], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],  # head
                        [-1.551596394 * 2 / 3, 2.206094311 * 2 / 3],  [-2.455676183 * 2 / 3, 0.7627082389 * 2 / 3],   [-1.570795 * 2 / 3, 2.188641033 * 2 / 3],
                        [-1.551596394 * 2 / 3, 2.206094311 * 2 / 3],  [-0.7627082389 * 2 / 3, 2.455676183 * 2 / 3],   [-2.188641033 * 2 / 3, 1.570795 * 2 / 3],
                        [-0.01, 0.01], [-2.570867817, 0.04799651389], [-0.01, 0.01],  # elbow
                        [-0.01, 0.01], [-0.04799651389, 2.570867817], [-0.01, 0.01],  # elbow
                        [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],
                        # wrist, pi/36 or 5 deg
                        [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],
                        # wrist, pi/36 or 5 deg
                        [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01],  # hand
                        [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01]])).float().to(DEVICE)
    
        self.verts_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]

        v_template_f = SMPL_FEMALE.th_v_template.squeeze(0)         # 6890, 3
        shapedirs_f = SMPL_FEMALE.th_shapedirs.permute(0, 2, 1)     # 6890, 10, 3
        J_regressor_f = SMPL_FEMALE.th_J_regressor.permute(1, 0)    # 6890, 24
        posedirs_f = SMPL_FEMALE.th_posedirs                        # 6890, 3, 207
        # posedirs_f = posedirs_f[self.verts_list, :, :]              # 10, 3, 207
        weights_f = SMPL_FEMALE.th_weights                          # 6890, 24
        # weights_f = weights_f[self.verts_list, :]                   # 10, 24

        v_template_m = SMPL_MALE.th_v_template.squeeze(0)
        shapedirs_m = SMPL_MALE.th_shapedirs.permute(0, 2, 1)
        J_regressor_m = SMPL_MALE.th_J_regressor.permute(1, 0)
        posedirs_m = SMPL_MALE.th_posedirs
        # posedirs_m = posedirs_m[self.verts_list, :, :]
        weights_m = SMPL_MALE.th_weights
        # weights_m = weights_m[self.verts_list, :]

        self.parents = np.array(SMPL_FEMALE.kintree_parents).astype(np.int32)

        self.N = batch_size
        shapedirs_f = shapedirs_f.unsqueeze(0).repeat(self.N, 1, 1, 1).permute(0, 2, 1, 3).unsqueeze(0)         # 6890, 10, 3 -> 1, 6890, 10, 3 -> N, 6890, 10, 3 -> N, 10, 6890, 3 -> 1, N, 10, 6890, 3 
        shapedirs_m = shapedirs_m.unsqueeze(0).repeat(self.N, 1, 1, 1).permute(0, 2, 1, 3).unsqueeze(0)
        self.shapedirs = torch.cat((shapedirs_f, shapedirs_m), 0)                                               # 2 x N x B x R x D
        self.B = self.shapedirs.size()[2]  # this is 10
        self.R = self.shapedirs.size()[3]  # this is 6890, or num of verts
        self.D = self.shapedirs.size()[4]  # this is 3, or num dimensions
        self.R_used = self.R # 10
        self.shapedirs = self.shapedirs.permute(1, 0, 2, 3, 4).view(self.N, 2, self.B * self.R * self.D)        # N X 2 X B X R X D ->  N, 2, B*R*D

        v_template_f = v_template_f.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)                         # 6890, 3 -> 1, N, 6890, 3
        v_template_m = v_template_m.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
        self.v_template = torch.cat((v_template_f, v_template_m), 0)                                            # 2 X N X R X D
        self.v_template = self.v_template.permute(1, 0, 2, 3).view(self.N, 2, self.R * self.D)                  # N X 2 X R*D

        self.J_regressor = torch.cat((J_regressor_f.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0),
                                        J_regressor_m.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)), 0)                                                                      # 2 X N X R X 24
        self.J_regressor = self.J_regressor.permute(1, 0, 2, 3).view(self.N, 2, self.R * 24)                    # N X 2 X R*24

        self.posedirs = torch.cat((posedirs_f.unsqueeze(0).repeat(self.N, 1, 1, 1).unsqueeze(0),
                                    posedirs_m.unsqueeze(0).repeat(self.N, 1, 1, 1).unsqueeze(0)), 0)           # 2 X N X V X 3 X 207
        self.posedirs = self.posedirs.permute(1, 0, 2, 3, 4).view(self.N, 2, self.R_used * self.D * 207)        # N X 2 X V*D*207

        self.weights_repeat = torch.cat((weights_f.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0), 
                                        weights_m.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)), 0)           # 2 X N X V X 24
        self.weights_repeat = self.weights_repeat.permute(1, 0, 2, 3).view(self.N, 2, self.R_used * 24)         # N X 2 X V*24

    def forward(self, x, batch_gender, is_gt=False):
        '''
        Input 
        ------
        x : [B, 88] -> output from CNN model 
        batch_gender : [B, 2] -> input gender 

        Output
        ------
        out_betas           :   [B, 10]         -> body shape 
        out_joint_angles:   :   [B, 72]         -> joint angles 
        out_root_shift      :   [B, 3]          -> root shift xyz
        out_root_angles     :   [B, 6]          -> root angles (atan2)
        out_verts           :   [B, 6890, 3]    -> 3D vertices 
        out_verts_red       :   [B, 10, 3]      -> subset of 3D vertices
        out_joint_pos       :   [B, 24]         -> Joint locations
        '''
        if x is None:
            return {}

        batch_gender = batch_gender.float().to(DEVICE)

        if not is_gt:
            # Comment and below and check performance 
            # x = x*0.01 # commenting this to check if perf degrades

            # Shift xyz position 
            x[:, 10] += (0.6 - 0.286)
            x[:, 11] += (1.2 - 0.286)
            x[:, 12] += (0.1)

        # [B, 88] -> [B, 91]
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        x = F.pad(x, (0, 3, 0, 0))
        x = x.squeeze(0)
        x = x.squeeze(0)

        # preserving joint angles 
        x[:, 22:91] = x[:, 19:88].clone() 
        
        # root rotation angle
        x[:, 19] = torch.atan2(x[:, 16].clone(), x[:, 13].clone()) #pitch x, y
        x[:, 20] = torch.atan2(x[:, 17].clone(), x[:, 14].clone()) #roll x, y
        x[:, 21] = torch.atan2(x[:, 18].clone(), x[:, 15].clone()) #yaw x, y

        if not is_gt:
            # clip beta parameters
            x[:, :10] /= 3.
            x[:, :10] = x[:, :10].tanh()
            x[:, :10] *= 3

            # apply bounds to joint angles 
            x[:, 19:91] -= torch.mean(self.bounds[0:72, 0:2], dim=1)
            x[:, 19:91] *= (2. / torch.abs(self.bounds[0:72, 0] - self.bounds[0:72, 1]))
            x[:, 19:91] = x[:, 19:91].tanh()
            x[:, 19:91] /= (2. / torch.abs(self.bounds[0:72, 0] - self.bounds[0:72, 1]))
            x[:, 19:91] += torch.mean(self.bounds[0:72, 0:2], dim=1)

        # form rotation matrix
        joint_angles_matrix = angle_utils.batch_rodrigues(x[:, 19:91].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)

        out_betas = x[:, 0:10].clone()
        out_joint_angles = x[:, 19:91].clone()
        out_root_shift = x[:, 10:13].clone()
        out_root_angles = x[:, 13:19].clone()

        use_betas = x[:, 0:10].clone()
        use_root_shift = x[:, 10:13].clone()

        batch_gender = batch_gender.unsqueeze(1)

        batch_size_infer = batch_gender.shape[0]


        # verts prediction
        shapedirs = torch.bmm(batch_gender, self.shapedirs[0:batch_size_infer, :, :])\
                    .view(batch_size_infer, self.B, self.R*self.D)                                # N, B, R*D
        betas_shapedirs_mult = torch.bmm(use_betas.unsqueeze(1), shapedirs)\
                                    .squeeze(1)\
                                    .view(batch_size_infer, self.R, self.D)                       # N, R, D
        v_template = torch.bmm(batch_gender, self.v_template[0:batch_size_infer, :, :])\
                            .view(batch_size_infer, self.R, self.D)                               # N, R, D
        v_shaped = betas_shapedirs_mult + v_template                                              # N, R, D
        J_regressor_repeat = torch.bmm(batch_gender, self.J_regressor[0:batch_size_infer, :, :])\
                                    .view(batch_size_infer, self.R, 24)                           # N, R, 24
        Jx = torch.bmm(v_shaped[:, :, 0].unsqueeze(1), J_regressor_repeat).squeeze(1)             # N, 24
        Jy = torch.bmm(v_shaped[:, :, 1].unsqueeze(1), J_regressor_repeat).squeeze(1)             # N, 24
        Jz = torch.bmm(v_shaped[:, :, 2].unsqueeze(1), J_regressor_repeat).squeeze(1)             # N, 24
        # these are the joint locations with home pose (pose is 0 degree on all angles)
        joint_locations_pred = torch.stack([Jx, Jy, Jz], dim=2)                                   # N, 24, 3

        new_joint_locations_pred, A_pred = angle_utils.batch_global_rigid_transformation(\
                                            joint_angles_matrix, joint_locations_pred, \
                                            self.parents, rotate_base=False)
        new_joint_locations_pred = new_joint_locations_pred - joint_locations_pred[:, 0:1, :] + use_root_shift.unsqueeze(1)

        # assemble a reduced form of the transformed mesh
        # v_shaped_red = torch.stack([v_shaped[:, self.verts_list[0], :],
        #                             v_shaped[:, self.verts_list[1], :],  # head
        #                             v_shaped[:, self.verts_list[2], :],  # l knee
        #                             v_shaped[:, self.verts_list[3], :],  # r knee
        #                             v_shaped[:, self.verts_list[4], :],  # l ankle
        #                             v_shaped[:, self.verts_list[5], :],  # r ankle
        #                             v_shaped[:, self.verts_list[6], :],  # l elbow
        #                             v_shaped[:, self.verts_list[7], :],  # r elbow
        #                             v_shaped[:, self.verts_list[8], :],  # l wrist
        #                             v_shaped[:, self.verts_list[9], :],  # r wrist
        #                             ]).permute(1, 0, 2)
        v_shaped_red = v_shaped

        pose_feature = (joint_angles_matrix[:, 1:, :, :]).sub(1.0, torch.eye(3).float().to(DEVICE)).view(-1, 207)   # N, 207
        posedirs_repeat = torch.bmm(batch_gender, self.posedirs[0:batch_size_infer, :, :]) \
            .view(batch_size_infer, self.R_used * self.D, 207) \
            .permute(0, 2, 1)                                                                                       # N, 207, 30
        v_posed = torch.bmm(pose_feature.unsqueeze(1), posedirs_repeat).view(-1, self.R_used, self.D)               # N, 10, 3
        v_posed = v_posed.clone() + v_shaped_red
        weights_repeat = torch.bmm(batch_gender, self.weights_repeat[0:batch_size_infer, :, :]) \
            .squeeze(1) \
            .view(batch_size_infer, self.R_used, 24)
        T = torch.bmm(weights_repeat, A_pred.view(batch_size_infer, 24, 16)).view(batch_size_infer, -1, 4, 4)
        v_posed_homo = torch.cat([v_posed, torch.ones(batch_size_infer, v_posed.shape[1], 1).float().to(DEVICE)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts_pred = v_homo[:, :, :3, 0] - joint_locations_pred[:, 0:1, :] + use_root_shift.unsqueeze(1)
        verts_red = torch.stack([
                                verts_pred[:, self.verts_list[0], :],
                                verts_pred[:, self.verts_list[1], :],  # head
                                verts_pred[:, self.verts_list[2], :],  # l knee
                                verts_pred[:, self.verts_list[3], :],  # r knee
                                verts_pred[:, self.verts_list[4], :],  # l ankle
                                verts_pred[:, self.verts_list[5], :],  # r ankle
                                verts_pred[:, self.verts_list[6], :],  # l elbow
                                verts_pred[:, self.verts_list[7], :],  # r elbow
                                verts_pred[:, self.verts_list[8], :],  # l wrist
                                verts_pred[:, self.verts_list[9], :],  # r wrist
                                ]).permute(1, 0, 2)
        
        if not is_gt:
            verts_offset = torch.Tensor(verts_red.clone().detach().cpu().numpy()).float().to(DEVICE)

            # joint locations 
            targets_est_detached = torch.Tensor(new_joint_locations_pred.clone().detach().cpu().numpy()).float().to(DEVICE)
            synth_joint_addressed = [3, 15, 4, 5, 7, 8, 18, 19, 20, 21]
            for real_joint in range(10):
                verts_offset[:, real_joint, :] = verts_offset[:, real_joint, :] - targets_est_detached[:, synth_joint_addressed[real_joint], :]

            for real_joint in range(10):
                new_joint_locations_pred[:, synth_joint_addressed[real_joint], :] = new_joint_locations_pred[:, synth_joint_addressed[real_joint], :].clone() \
                                    + torch.add(-1, 1) * (new_joint_locations_pred[:, synth_joint_addressed[real_joint], :].clone() + verts_offset[:, real_joint, :])


        new_joint_locations_pred = new_joint_locations_pred.contiguous().view(-1, 72)

        verts_pred_final = verts_pred[:, :, [1, 0, 2]]
        verts_pred_final[:, :, -1] *= -1.0

        return {
            'out_betas'         : out_betas, 
            'out_joint_angles'  : out_joint_angles, 
            'out_root_shift'    : out_root_shift, 
            'out_root_angles'   : out_root_angles, 
            'out_verts'         : verts_pred_final, 
            'out_verts_red'     : verts_red,
            'out_joint_pos'     : new_joint_locations_pred
        }        

    def infer(self, x, batch_gender, is_gt=False):
        return self.forward(x, batch_gender, is_gt=is_gt)

    
if __name__ == '__main__':
    mesh_model = MeshEstimator(2).to(DEVICE)

    pme_pred = torch.rand(2, 88).float().to(DEVICE)
    batch_gender = torch.tensor([[1, 0], [0, 1]])
    mesh_pred = mesh_model(pme_pred, batch_gender)

    for k in mesh_pred:
        print (k, mesh_pred[k].shape)

