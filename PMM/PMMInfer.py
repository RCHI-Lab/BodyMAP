import numpy as np 
import os

import torch
from constants import *
import viz_utils


def PMMInfer(model, infer_loader, writer=None, save_gt=False, \
            epoch=-1, pmap_norm=False, infer_pmap=False, infer_smpl=False, MOD1=None):
    model.eval()
    with torch.no_grad():
        for batch_original_pressure, \
            batch_pressure_images, \
            batch_original_depth, \
            batch_depth_images, \
            batch_labels, \
            batch_pmap, \
            batch_verts, \
            batch_names \
                in iter(infer_loader):
            batch_depth_images = batch_depth_images.to(DEVICE)
            batch_pressure_images = batch_pressure_images.to(DEVICE)
            
            if MOD1 is not None:
                mesh_pred, _, img_feat, _ = MOD1.infer(batch_depth_images.clone(), batch_pressure_images.clone(), batch_labels[:, 157:159].clone())
                _, pmap_pred, _, _ = model(batch_depth_images, batch_pressure_images, batch_labels[:, 157:159], mesh_pred['out_verts'].clone(), img_feat)
            else:
                mesh_pred, pmap_pred, _, _ = model.infer(batch_depth_images, batch_pressure_images, batch_labels[:, 157:159])
            num_images = batch_pmap.shape[0]

            if not infer_smpl:
                mesh_pred = {'out_verts' : batch_verts}

            for i in range(num_images):
                if infer_pmap:
                    gt_pmap = batch_pmap[i]
                    pd_pmap = pmap_pred[i]
                    if pmap_norm:
                        if batch_names[i][0] == 's':
                            pd_pmap *= MAX_PMAP_SYNTH
                            gt_pmap *= MAX_PMAP_SYNTH
                        elif batch_names[i][0] == 'r':
                            pd_pmap *= MAX_PMAP_REAL
                            gt_pmap *= MAX_PMAP_REAL
                        else:
                            print ('ERROR: Invalid data category in infer')
                            exit(-1)
                    rest_verts = viz_utils.get_rest_verts(batch_labels[i][72:82].float().to(DEVICE), batch_labels[i][157])
                else:
                    rest_verts = None

                if save_gt:
                    render_pose = viz_utils.render_trimesh(batch_verts[i], torch.tensor(np.ones((6890, 3)) * [0, 1, 0]), edit=True) # green - gt
                    if infer_pmap:
                        color_gt_pmap = viz_utils.generate_colors(gt_pmap)
                        render_pmap = viz_utils.render_trimesh(rest_verts, color_gt_pmap, edit=False)
                        render_both = viz_utils.render_trimesh(batch_verts[i], color_gt_pmap, edit=True)
                    else:
                        render_both = render_pose
                    try:
                        gt_image = viz_utils.generate_viz(batch_original_pressure[i].numpy().copy(), \
                                                            batch_original_depth[i].numpy().copy(), \
                                                            batch_verts[i].numpy().copy())
                    except Exception as e:
                        print ('Exception occured in generating gt image', e)
                        gt_image = None
                    if writer:
                        # writer.add_image(f'Mod_{batch_names[i]}/pressure', batch_pressure_images[i].to("cpu").numpy())
                        # writer.add_image(f'Mod_{batch_names[i]}/depth', batch_depth_images[i].to("cpu").numpy())
                        # if gt_image is not None:
                        #     writer.add_image(f'Ex_{batch_names[i]}/gt_data', gt_image, dataformats='HWC')
                        writer.add_mesh(f'Ex_{batch_names[i]}/gt', 
                                        vertices=torch.as_tensor(render_both.vertices, dtype=torch.float).unsqueeze(0),
                                        colors=torch.as_tensor(render_both.visual.vertex_colors[:, :3], dtype=torch.int).unsqueeze(0),
                                        faces=torch.as_tensor(render_both.faces, dtype=torch.int).unsqueeze(0))
                        if infer_pmap:
                            writer.add_mesh(f'Pmap_{batch_names[i]}/gt', 
                                            vertices=torch.as_tensor(render_pmap.vertices, dtype=torch.float).unsqueeze(0),
                                            colors=torch.as_tensor(render_pmap.visual.vertex_colors[:, :3], dtype=torch.int).unsqueeze(0),
                                            faces=torch.as_tensor(render_pmap.faces, dtype=torch.int).unsqueeze(0))
                            writer.add_mesh(f'Pose_{batch_names[i]}/gt', 
                                            vertices=torch.as_tensor(render_pose.vertices, dtype=torch.float).unsqueeze(0),
                                            colors=torch.as_tensor(render_pose.visual.vertex_colors[:, :3], dtype=torch.int).unsqueeze(0),
                                            faces=torch.as_tensor(render_pose.faces, dtype=torch.int).unsqueeze(0))

                render_pose = viz_utils.render_trimesh(mesh_pred['out_verts'][i], torch.tensor(np.ones((6890, 3)) * [0, 0, 1]), edit=True) # blue - pred
                if infer_pmap:
                    color_pd_pmap = viz_utils.generate_colors(pd_pmap)
                    render_pmap = viz_utils.render_trimesh(rest_verts, color_pd_pmap, edit=False)
                    render_both = viz_utils.render_trimesh(mesh_pred['out_verts'][i], color_pd_pmap, edit=True)
                else:
                    render_both = render_pose
                try:
                    pred_image = viz_utils.generate_viz(batch_original_pressure[i].numpy().copy(), \
                                                        batch_original_depth[i].numpy().copy(), \
                                                        mesh_pred['out_verts'][i].to("cpu").numpy().copy())
                except Exception as e:
                    print ('Exception occured in generating pred image', e)
                    pred_image = None
                if writer:
                    # if pred_image is not None:
                    #     writer.add_image(f'Ex_{batch_names[i]}/pred_data', pred_image, global_step=epoch, dataformats='HWC')
                    writer.add_mesh(f'Ex_{batch_names[i]}/pred', 
                                        vertices=torch.as_tensor(render_both.vertices, dtype=torch.float).unsqueeze(0),
                                        colors=torch.as_tensor(render_both.visual.vertex_colors[:, :3], dtype=torch.int).unsqueeze(0), 
                                        faces=torch.as_tensor(render_both.faces, dtype=torch.int).unsqueeze(0), 
                                        global_step=epoch)
                    if infer_pmap:
                        writer.add_mesh(f'Pmap_{batch_names[i]}/pred', 
                                            vertices=torch.as_tensor(render_pmap.vertices, dtype=torch.float).unsqueeze(0),
                                            colors=torch.as_tensor(render_pmap.visual.vertex_colors[:, :3], dtype=torch.int).unsqueeze(0), 
                                            faces=torch.as_tensor(render_pmap.faces, dtype=torch.int).unsqueeze(0), 
                                            global_step=epoch)
                        writer.add_mesh(f'Pose_{batch_names[i]}/pred', 
                                            vertices=torch.as_tensor(render_pose.vertices, dtype=torch.float).unsqueeze(0),
                                            colors=torch.as_tensor(render_pose.visual.vertex_colors[:, :3], dtype=torch.int).unsqueeze(0), 
                                            faces=torch.as_tensor(render_pose.faces, dtype=torch.int).unsqueeze(0), 
                                            global_step=epoch)

