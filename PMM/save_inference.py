import argparse
import json
import numpy as np 
import os 
from tqdm import tqdm 

import torch 

from PMMTrainerDataset import prepare_loader
from constants import DEVICE, MAX_PMAP_REAL, BASE_PATH


def load_model(model_path, opts_path):
    model = torch.load(model_path).to(DEVICE)
    model.eval()

    with open(opts_path, 'r') as f:
        opts = json.load(f)
    return model, opts


def parse_args():
    parser = argparse.ArgumentParser(description='save inference results')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Full path of model weights (pth) file')
    parser.add_argument('--opts_path', type=str, required=True, 
                        help='Full path of exp.json file saved after training')
    parser.add_argument('--save_path', type=str, required=True, 
                        help='Full path of directory to save inference results')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    model, opts = load_model(args.model_path, args.opts_path)
    writer_path = args.save_path
    os.makedirs(writer_path, exist_ok=True)

    data_path = os.path.join(BASE_PATH, "BodyPressure/data_BP")
    val_file = os.path.join(BASE_PATH, "BodyMAP/data_files/real_val.txt")

    test_loader, _ = prepare_loader(data_path, [val_file, None], opts['batch_size'], opts['image_size_type'], \
                                'normal', opts['normalize_pressure'], opts['normalize_depth'], \
                                False, False, training=False)

    model.eval()
    if opts['WS']:
        MOD1 = torch.load(opts['load_MOD1_path']).to(DEVICE)
        MOD1.eval()
    else:
        MOD1 = None

    print (f'Starting testing for {opts["name"]}')
    with torch.no_grad():
        for _, batch_pressure_images, _, batch_depth_images, batch_labels, _, _, batch_names in tqdm(iter(test_loader), desc='model test run'):
            batch_depth_images = batch_depth_images.to(DEVICE)
            batch_pressure_images = batch_pressure_images.to(DEVICE)

            if MOD1 is not None:
                batch_mesh_pred, batch_pmap_pred, _, _ = model.infer(batch_depth_images, batch_pressure_images, batch_labels[:, 157:159])
            else:
                batch_mesh_pred, _, img_feat, _ = MOD1.infer(batch_depth_images.clone(), batch_pressure_images.clone(), batch_labels[:, 157:159].clone())
                _, batch_pmap_pred, _, _ = model(batch_depth_images, batch_pressure_images, batch_labels[:, 157:159], batch_mesh_pred['out_verts'].clone(), img_feat)

            batch_mesh_pred['out_joint_pos'] = batch_mesh_pred['out_joint_pos'].reshape(-1, 24, 3)

            if opts['normalize_pressure']:
                batch_pmap_pred *= MAX_PMAP_REAL

            for pd_betas, pd_jtr, pd_verts, pd_pmap, name in zip(batch_mesh_pred['out_betas'], 
                                                                batch_mesh_pred['out_joint_pos'], 
                                                                batch_mesh_pred['out_verts'], 
                                                                batch_pmap_pred, batch_names):
                name_split = name.split('_')
                write_dir = os.path.join(writer_path, f'{int(name_split[2]):05d}', f'{name_split[1]}')
                os.makedirs(write_dir, exist_ok=True)

                np.save(os.path.join(write_dir, f'{int(name_split[-1]):06d}_pd_betas.npy'), pd_betas.to("cpu").numpy())
                np.save(os.path.join(write_dir, f'{int(name_split[-1]):06d}_pd_vertices.npy'), pd_verts.to("cpu").numpy())
                np.save(os.path.join(write_dir, f'{int(name_split[-1]):06d}_pd_jtr.npy'), pd_jtr.to("cpu").numpy() * 1000.)
                np.save(os.path.join(write_dir, f'{int(name_split[-1]):06d}_pd_pmap.npy'), pd_pmap.to("cpu").numpy())
    print ('done')

