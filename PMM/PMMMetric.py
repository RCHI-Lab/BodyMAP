import json
import numpy as np
from tqdm import tqdm 

import torch
from constants import *


FACES_NP = FACES.squeeze(0).to("cpu").numpy()

TRANS_PREV_GT = np.identity(4)
TRANS_PREV_GT[0, 3] = -2.0
TRANS_NEXT_GT = np.identity(4)
TRANS_NEXT_GT[0, 3] = -2.0
TRANS_NEXT_GT[1, 3] = 1.0

TRANS_PREV_PD = np.identity(4)
TRANS_NEXT_PD = np.identity(4)
TRANS_NEXT_PD[1, 3] = 1.0


def get_triangle_area_vert_weight(verts, faces, verts_idx_red = None):

    #first we need all the triangle areas
    tri_verts = verts[faces, :]
    a = np.linalg.norm(tri_verts[:,0]-tri_verts[:,1], axis = 1)
    b = np.linalg.norm(tri_verts[:,1]-tri_verts[:,2], axis = 1)
    c = np.linalg.norm(tri_verts[:,2]-tri_verts[:,0], axis = 1)
    s = (a+b+c)/2
    A = np.sqrt(s*(s-a)*(s-b)*(s-c))

    A = np.swapaxes(np.stack((A, A, A)), 0, 1) #repeat the area for each vert in the triangle
    A = A.flatten()
    faces = np.array(faces).flatten()
    i = np.argsort(faces) #sort the faces and the areas by the face idx
    faces_sorted = faces[i]
    A_sorted = A[i]
    last_face = 0
    area_minilist = []
    area_avg_list = []
    face_sort_list = [] #take the average area for all the trianges surrounding each vert
    for vtx_connect_idx in range(np.shape(faces_sorted)[0]):
        if faces_sorted[vtx_connect_idx] == last_face and vtx_connect_idx != np.shape(faces_sorted)[0]-1:
            area_minilist.append(A_sorted[vtx_connect_idx])
        elif faces_sorted[vtx_connect_idx] > last_face or vtx_connect_idx == np.shape(faces_sorted)[0]-1:
            if len(area_minilist) != 0:
                area_avg_list.append(np.mean(area_minilist))
            else:
                area_avg_list.append(0)
            face_sort_list.append(last_face)
            area_minilist = []
            last_face += 1
            if faces_sorted[vtx_connect_idx] == last_face:
                area_minilist.append(A_sorted[vtx_connect_idx])
            elif faces_sorted[vtx_connect_idx] > last_face:
                num_tack_on = np.copy(faces_sorted[vtx_connect_idx] - last_face)
                for i in range(num_tack_on):
                    area_avg_list.append(0)
                    face_sort_list.append(last_face)
                    last_face += 1
                    if faces_sorted[vtx_connect_idx] == last_face:
                        area_minilist.append(A_sorted[vtx_connect_idx])

    area_avg = np.array(area_avg_list)
    area_avg_red = area_avg[area_avg > 0] #find out how many of the areas correspond to verts facing the camera

    norm_area_avg = area_avg/np.sum(area_avg_red)
    norm_area_avg = norm_area_avg*np.shape(area_avg_red) #multiply by the REDUCED num of verts

    if verts_idx_red is not None:
        try:
            norm_area_avg = norm_area_avg[verts_idx_red]
        except:
            norm_area_avg = norm_area_avg[verts_idx_red[:-1]]

    return norm_area_avg


def get_area_norm(verts, gt=False):
    if gt:
        trans_prev, trans_next = TRANS_PREV_GT.copy(), TRANS_NEXT_GT.copy()
    else:
        trans_prev, trans_next = TRANS_PREV_PD.copy(), TRANS_NEXT_PD.copy()
    
    verts_edit = verts.copy()
    
    smpl_verts_quad = np.concatenate((verts_edit, np.ones((verts.shape[0], 1))), axis = 1)
    smpl_verts_quad = np.swapaxes(smpl_verts_quad, 0, 1)
    smpl_verts = np.swapaxes(np.matmul(trans_prev, smpl_verts_quad), 0, 1)[:, 0:3] # gt over pressure mat

    vertices_pimg = np.array(smpl_verts)
    faces_pimg = np.array(FACES_NP.copy())

    vertices_pimg[:, 0] = vertices_pimg[:, 0] + trans_next[0, 3] - trans_prev[0, 3]
    vertices_pimg[:, 1] = vertices_pimg[:, 1] + trans_next[1, 3] - trans_prev[1, 3]

    area_norm = get_triangle_area_vert_weight(vertices_pimg, faces_pimg, None)
    return area_norm


def create_metric_dict(epoch=-1):
    return {
            'epoch' : epoch,
            '3D MPJPE' : torch.tensor(0).float().to(DEVICE),
            'v2vP' : torch.tensor(0).float().to(DEVICE), 
            'count' : 0,
            }


def PMMMetric(model, test_loader, writer=None, epoch=-1, pmap_norm=False, infer_pmap=False, infer_smpl=False, MOD1=None):
    dict_map = {
            'uncover' : create_metric_dict(epoch), 
            'cover1' :  create_metric_dict(epoch), 
            'cover2' :  create_metric_dict(epoch), 
            'synth' :   create_metric_dict(epoch),
            'f' :       create_metric_dict(epoch), 
            'm' :       create_metric_dict(epoch),
            'overall' : create_metric_dict(epoch),
            }

    model.eval()
    with torch.no_grad():
        for _, batch_pressure_images, _, batch_depth_images, batch_labels, batch_pmap, batch_verts, batch_names in tqdm(iter(test_loader), desc='metric'):
            batch_depth_images = batch_depth_images.to(DEVICE)
            batch_pressure_images = batch_pressure_images.to(DEVICE)
            batch_labels_copy = batch_labels.clone().to(DEVICE)
            batch_pmap = batch_pmap.to(DEVICE)            

            if MOD1 is not None:
                batch_mesh_pred, _, img_feat, _ = MOD1.infer(batch_depth_images.clone(), batch_pressure_images.clone(), batch_labels[:, 157:159].clone())
                _, batch_pmap_pred, _, _ = model(batch_depth_images, batch_pressure_images, batch_labels[:, 157:159], batch_mesh_pred['out_verts'].clone(), img_feat)
            else:
                batch_mesh_pred, batch_pmap_pred, _, _ = model.infer(batch_depth_images, batch_pressure_images, batch_labels[:, 157:159])

            batch_labels = batch_labels.to(DEVICE)
            if infer_smpl:
                batch_mesh_pred['out_joint_pos'] = batch_mesh_pred['out_joint_pos'].reshape(-1, 24, 3)
            else:
                batch_mesh_pred = {
                    'out_joint_pos' : (batch_labels_copy[:, :72]/1000.).reshape(-1, 24, 3),
                    'out_verts' : batch_verts,
                    }
            if not infer_pmap:
                batch_pmap_pred = batch_pmap.clone()

            
            for i, file_name in enumerate(batch_names):
                file_name_contents = file_name.split('_')
                loss = torch.norm(batch_labels_copy[i, :72].reshape(24, 3)/1000. - batch_mesh_pred['out_joint_pos'][i], dim=1)
                
                gt_pmap = batch_pmap[i]
                pd_pmap = batch_pmap_pred[i]
                if pmap_norm:
                    if batch_names[i][0] == 's':
                        pd_pmap *= MAX_PMAP_SYNTH
                        gt_pmap *= MAX_PMAP_SYNTH
                    elif batch_names[i][0] == 'r':
                        pd_pmap *= MAX_PMAP_REAL
                        gt_pmap *= MAX_PMAP_REAL
                    else:
                        print ('ERROR: Invaid data category in metric calculation')
                        exit(-1)
                gt_area_norm = torch.tensor(get_area_norm(batch_verts[i].numpy(), gt=True)).float().to(DEVICE)
                pd_area_norm = torch.tensor(get_area_norm(batch_mesh_pred['out_verts'][i].to("cpu").numpy(), gt=(not infer_smpl))).float().to(DEVICE)

                gt_pmap_norm = gt_pmap * gt_area_norm
                pd_pmap_norm = pd_pmap * pd_area_norm

                loss_pmap = torch.nn.functional.mse_loss(pd_pmap_norm, gt_pmap_norm, reduction='none')
                    
                metric_dict_data = dict_map[file_name_contents[1]]
                metric_dict_gender = dict_map[file_name_contents[3]]
                metric_dict_overall = dict_map['overall']
                for metric_dict in [metric_dict_data, metric_dict_gender, metric_dict_overall]:
                    metric_dict['3D MPJPE'] += loss.sum()
                    metric_dict['v2vP'] += loss_pmap.sum()
                    metric_dict['count'] += 1

        for mdict in dict_map.values():
            # divide by 24 for 24 joint positions, multiiply by 1000 for error in mm
            mdict['3D MPJPE'] = round(((mdict['3D MPJPE']/(mdict['count']*24))*1000).item(), 6)
            mdict['v2vP'] = (mdict['v2vP']/(mdict['count']*6890)).item()
            mdict['v2vP'] = round(133.32 * 133.32 * (1 / 1000000) * mdict['v2vP'], 6)

    if writer is not None:
        for mkey, mdict in dict_map.items():
            mdict_str = json.dumps(mdict)
            writer.add_text(f'{mkey}', mdict_str)
    return dict_map

