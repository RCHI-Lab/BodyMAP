import argparse
import cv2
import numpy as np 
import imageio
import os 
from PIL import Image
from scipy.ndimage.interpolation import zoom
import sys
import trimesh
from tqdm import tqdm 

sys.path.append('../PMM')
import torch 
import viz_utils
import utils
from constants import DEVICE, FACES, BASE_PATH
from SLPDataset import SLPDataset


faces = FACES.squeeze(0).to("cpu").long().numpy()
DATA_PATH = os.path.join(BASE_PATH, 'BodyPressure', 'data_BP')
DATA_LINES = utils.load_data_lines(os.path.join(BASE_PATH, 'BodyMAP', 'data_files', 'real_val.txt'))


def norm(vert):
    vert[:, 0] += 0.0286*2
    vert[:, 1] += 0.0143

    shift_both_amount = -np.min([-0.1, np.min(vert[:, 1])])

    vert_quad = np.concatenate((vert, np.ones((vert.shape[0], 1))), axis = 1)
    vert_quad = np.swapaxes(vert_quad, 0, 1)
    
    transform_A = np.identity(4)
    transform_A[1, 3] = shift_both_amount
    vert_A = np.swapaxes(np.matmul(transform_A, vert_quad), 0, 1)[:, 0:3]

    vert_A *= (64*0.0286 / 1.92)
    vert_A[:, 0] += (1.92 - 64*0.0286)/2
    vert_A[:, 1] += (0.84 - 27*0.0286)/2

    vert_A = vert_A - np.mean(vert_A, axis=0)
    vert_A[:, 2] *= -1
    return vert_A


def gen_mesh(vert, colors):
    all_verts = np.array(vert)
    all_verts[:, 2] *= -1
    faces_red = np.array(faces)
    mesh = trimesh.base.Trimesh(vertices=all_verts, faces=faces_red, vertex_colors=colors)
    return mesh


def get_data(cover_str):
    data_dict = {}
    data_pressure_x, data_depth_x, data_label_y, data_pmap_y, data_verts_y, data_names_y = SLPDataset(DATA_PATH).prepare_dataset(DATA_LINES, cover_str, load_verts=True, for_infer=False)
    data_label_y = torch.tensor(data_label_y).float().to(DEVICE)
    data_pmap_y = torch.tensor(data_pmap_y).float()
    data_verts_y = data_verts_y
    
    for pressure, depth, label, pmap, verts, name in zip(data_pressure_x, data_depth_x, data_label_y, data_pmap_y, data_verts_y, data_names_y):
        name_split = name.split("_")
        name_filtered = "_".join(name_split[1:3] + name_split[4:])
        data_dict[name_filtered] = {
            'pressure' : pressure, 
            'depth'    : depth, 
            'label'    : label, 
            'pmap'     : pmap, 
            'vert'     : verts,
            'name'     : name_filtered,
        }
    return data_dict


def get_image(color_arr_in, PTr_A2B):
        sz_B = [84, 192]
        color_arr = cv2.warpPerspective(color_arr_in, PTr_A2B, tuple(sz_B)).astype(np.int16)
        image_zoom = 4.585/2
        color_arr_r = zoom(color_arr[:, :, 0], image_zoom, order=1)
        color_arr_g = zoom(color_arr[:, :, 1], image_zoom, order=1)
        color_arr_b = zoom(color_arr[:, :, 2], image_zoom, order=1)
        color_arr = np.stack((color_arr_r, color_arr_g, color_arr_b), axis=2)
        color_arr = color_arr[:, 1:-1]  
        return color_arr


def load_RGB_images(cover_type, p_idx, pose_idx):
    PTrA = np.load(os.path.join(DATA_PATH, 'SLP', 'danaLab', f'{p_idx:05}', 'align_PTr_RGB.npy'))
    PTrB = np.eye(3)

    PTr_A2B = np.dot(np.linalg.inv(PTrB), PTrA)
    PTr_A2B = PTr_A2B / np.linalg.norm(PTr_A2B)

    uncover = imageio.imread(os.path.join(DATA_PATH, 'SLP', 'danaLab', f'{p_idx:05}', 'RGB', 'uncover', f'image_{pose_idx:06d}.png'))
    uncover = get_image(uncover, PTr_A2B)

    cover = imageio.imread(os.path.join(DATA_PATH, 'SLP', 'danaLab', f'{p_idx:05}', 'RGB', cover_type, f'image_{pose_idx:06d}.png'))
    cover = get_image(cover, PTr_A2B)

    return uncover, cover


def parse_args():
    parser = argparse.ArgumentParser(description='save visualization results')
    parser.add_argument('--cover_type', type=str, default='cover1', choices=['uncover', 'cover1', 'cover2'],  
                        help='Cover type of test example')
    parser.add_argument('--p_idx', type=int, default=81,   
                        help='Participant idx to visualize. p_idx should be between 81 and 102 (included)')
    parser.add_argument('--pose_idx', type=int, default=1,   
                        help='Pose idx to visualize. pose_idx should be between 1 and 45 (included)')
    parser.add_argument('--files_dir', type=str, default='none',   
                        help='Full path where model inference results are saved. If not passed, the code will visualize GT data')
    parser.add_argument('--viz_type', type=str, default='image', choices=['image', 'video'],   
                        help='Visualization type: image or video')
    parser.add_argument('--save_path', type=str, required=True, 
                        help='Full path of directory to save viz results')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_args()
    is_model = args.files_dir != 'none'

    data_dict = get_data(args.cover_type)
    name = f'{args.cover_type}_{args.p_idx:03}_{args.pose_idx}'

    ref1, ref2 = load_RGB_images(args.cover_type, args.p_idx, args.pose_idx)
    ref = np.ones((520, 412, 3)).astype(int) * 255
    ref[40:480, 10:201, :] = ref1[:, :, :3]
    ref[40:480, 211:-10, :] = ref2[:, :, :3]
    ref = ref.astype(np.uint8)

    pressure = viz_utils.gen_pressure(data_dict[name]['pressure'])
    depth = viz_utils.gen_depth(data_dict[name]['depth'])
    inp = np.ones((520, 413, 3)).astype(int) * 255 
    inp[40:480, 10:195, :] = depth
    inp[40:480, 205:-30, :] = pressure
    inp = inp.astype(np.uint8)

    if is_model:
        model_str = 'Model'
        pmap = np.load(os.path.join(args.files_dir, f'{args.p_idx:05}', args.cover_type, f'{args.pose_idx:06}_pd_pmap.npy'))
        pmap = torch.tensor(pmap).float()
        vert = np.load(os.path.join(args.files_dir, f'{args.p_idx:05}', args.cover_type, f'{args.pose_idx:06}_pd_vertices.npy'))
        vert = vert.reshape(-1, 3)
    else:
        model_str = 'GT'
        pmap = data_dict[name]['pmap']
        vert = data_dict[name]['vert']
        print ('Plotting GT data. Please pass files_dir to plot model inferences')
    pmap_colors = viz_utils.generate_colors(pmap).to("cpu").numpy()
    vert = norm(vert)
    mesh = gen_mesh(vert, pmap_colors)

    if args.viz_type == 'video':
        file_save_path = os.path.join(args.save_path, f'{model_str}_{name}.mp4')
        final_renders = []
        renders = viz_utils.render_video(mesh)
        for render in renders:
            render = np.hstack((ref, inp, render))
            render = cv2.putText(render, 'Reference', (40, 30), 0, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
            render = cv2.putText(render, 'Reference', (245, 30), 0, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
            render = cv2.putText(render, 'Depth', (475, 30), 0, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
            render = cv2.putText(render, '2D Pressure', (625, 30), 0, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
            render = cv2.putText(render, 'Output', (900, 30), 0, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
            image = Image.fromarray(render)
            image = image.resize((1080, int(render.shape[0]*1080/render.shape[1])))
            final_renders.append(image)
        imageio.mimsave(file_save_path, final_renders, fps=10)
    else: # image
        file_save_path = os.path.join(args.save_path, f'{model_str}_{name}.png')
        front_image = viz_utils.render_scene(mesh, rotate=False)
        back_image = viz_utils.render_scene(mesh, rotate=True)
        render = np.hstack((ref, inp, front_image, back_image))
        render = cv2.putText(render, 'Reference', (40, 30), 0, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
        render = cv2.putText(render, 'Reference', (245, 30), 0, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
        render = cv2.putText(render, 'Depth', (475, 30), 0, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
        render = cv2.putText(render, '2D Pressure', (625, 30), 0, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
        render = cv2.putText(render, 'Body Mesh - Pressure', (925, 30), 0, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
        image = Image.fromarray(render.astype(np.uint8))
        image.save(file_save_path)
    
    print (f'file saved at: {file_save_path}')

