import numpy as np 
import os
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom

from constants import *
import utils
import angle_utils


class SLPDataset():

    def __init__(self, data_path):
        super(SLPDataset, self).__init__()
        self.pressure_data_dir = os.path.join(data_path, 'SLP_filtered', 'danaLab') # change to SLP
        self.slp_real_cleaned_data_dir = os.path.join(data_path, 'slp_real_cleaned')
        self.pmap_data_dir = os.path.join(data_path, 'GT_BP_data', 'slp2')
        self.verts_data_dir = os.path.join(data_path, 'GT_BP_data', 'slp2')
        self.slp_labels_dir = os.path.join(data_path, 'SLP_SMPL_fits', 'fits')

        self.load_per_person = 45
        self.gender_male = [0, 1]
        self.gender_female = [1, 0]
        self.pm_adjust_cm = [-1, 4]

    def prepare_dataset(self, data_lines, cover_str, load_verts=False, for_infer=False, train_on_real='all'):
        pressure_x = []
        depth_x = []
        label_y = []
        pmap_y = []
        verts_y = []
        names_y = []
        
        phys_arr = np.load(os.path.join(self.pressure_data_dir, 'physiqueData.npy'))
        phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]

        slp_T_cam_all = np.load(os.path.join(self.slp_real_cleaned_data_dir, 'slp_T_cam_0to102.npy'), allow_pickle=True) # 102, 45, 3
        depth_all = np.load(os.path.join(self.slp_real_cleaned_data_dir, f'depth_{cover_str}_cleaned_0to102.npy'), allow_pickle=True) # 102, 45, 128, 54

        if train_on_real == 'all':
            start_pose = 0
            stop_pose = self.load_per_person
        elif train_on_real == 'lay':
            start_pose = 0
            stop_pose = 15 
        elif train_on_real == 'side':
            start_pose = 15
            stop_pose = self.load_per_person
        else:
            print (f'ERROR: not implemented train on real mode: {train_on_real}')
            exit(-1)

        for person_num_str in data_lines:
            body_mass = phys_arr[int(person_num_str) -1][0]
            body_height = phys_arr[int(person_num_str) - 1][1]
            gender = phys_arr[int(person_num_str) - 1][2]
            if gender == 0:
                gender_label = self.gender_female
                gender_str = 'f'
            else:
                gender_label = self.gender_male
                gender_str = 'm'
        
            pressure_path = os.path.join(self.pressure_data_dir, person_num_str, 'PMarray', cover_str)
            pmap_path = os.path.join(self.pmap_data_dir, person_num_str, cover_str)
            verts_path = os.path.join(self.verts_data_dir, person_num_str, cover_str)

            for i in range(start_pose, stop_pose, 1):
                pressure_image = np.load(os.path.join(pressure_path, f'{(i+1):06d}.npy')).astype(np.float) # 192, 84
                pressure_image = pressure_image[0:191 - self.pm_adjust_cm[1], 0:84]
                if np.shape(pressure_image)[0] < 190:
                    pressure_image = np.concatenate((np.zeros((190 - np.shape(pressure_image)[0], np.shape(pressure_image)[1])), pressure_image), axis = 0) # 190, 84
                pressure_image = pressure_image[:, 3 - self.pm_adjust_cm[0] : 80 - self.pm_adjust_cm[0]] # 190, 77
                pressure_image = gaussian_filter(pressure_image, sigma=0.5/0.345)
                pressure_image = zoom(pressure_image, (0.335, 0.355), order=1) # 64, 27
                pressure_image = (pressure_image * ((body_mass * 9.81) / (np.sum(pressure_image) * 0.0264 * 0.0286))) * (1 / 133.322) # normalizing by mass, converting to mmHg # 64, 27

                pressure_image = np.clip(pressure_image, 0, 100)

                depth_image = depth_all[int(person_num_str) - 1, i] # 128, 5
                depth_image[127:, :] = 0.0

                pose_data = utils.load_pickle(os.path.join(self.slp_labels_dir, f'p{int(person_num_str):03d}', f'proc_{(i+1):02d}.pkl'))
                pose_data['slp_T_cam'] = np.array(slp_T_cam_all[int(person_num_str) - 1, i])
                R_root = angle_utils.matrix_from_dir_cos_angles(pose_data['global_orient'])
                flip_root_euler = np.pi
                flip_root_R = angle_utils.eulerAnglesToRotationMatrix([flip_root_euler, 0.0, 0.0])
                pose_data['global_orient'] = angle_utils.dir_cos_angles_from_matrix(np.matmul(flip_root_R, R_root))
                body_shape = pose_data['betas'] # 10
                joint_angles = np.array(list(pose_data['global_orient']) + list(pose_data['body_pose'])) # 72
                root_xyz_shift = pose_data['O_T_slp'] + pose_data['slp_T_cam'] + pose_data['cam_T_Bo'] + pose_data['Bo_T_Br'] # 3
                root_xyz_shift[1:] *= -1
                # gt_markers = np.array(pose_data['markers_xyz_m']) + np.array([12/1000., -35/1000., 0.0])
                gt_markers = np.array(pose_data['markers_xyz_m'])
                gt_markers *= 1000.0
                gt_markers = gt_markers.reshape(-1)

                label = np.concatenate((
                    gt_markers, 
                    body_shape, 
                    joint_angles,
                    root_xyz_shift, 
                    gender_label, [1],
                    [body_mass], 
                    [body_height/100.]
                ), axis=0)

                pmap = np.load(os.path.join(pmap_path, f'{(i+1):06d}_gt_pmap.npy'))
                
                pressure_x.append(pressure_image)
                depth_x.append(depth_image)
                label_y.append(label)
                pmap_y.append(pmap)
                names_y.append(f'real_{cover_str}_{int(person_num_str):03d}_{gender_str}_{(i+1)}')

                if load_verts:
                    verts = np.load(os.path.join(verts_path, f'{(i+1):06d}_gt_vertices.npy'))
                    verts_y.append(verts)

                
                if for_infer and i >= start_pose + 1:
                    break
            
            if for_infer and len(names_y) >= 10:
                break
        
        data_pressure_x = np.expand_dims(np.array(pressure_x), -1)
        data_depth_x = np.expand_dims(np.array(depth_x), -1)
        data_label_y = np.array(label_y)
        data_pmap_y = np.array(pmap_y)
        data_verts_y = np.array(verts_y)
        data_names_y = np.array(names_y)

        print ('SLP', cover_str, \
                'dpx', data_pressure_x.shape, \
                data_pressure_x.max(),
                'ddx', data_depth_x.shape, \
                'dy', data_label_y.shape, \
                'data_pmap', data_pmap_y.shape, \
                'data_verts', data_verts_y.shape, \
                'data names', data_names_y.shape)

        return data_pressure_x, data_depth_x, data_label_y, data_pmap_y, data_verts_y, data_names_y
    
