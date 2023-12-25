import matplotlib.cm as cm
import numpy as np 
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
from scipy.ndimage.interpolation import zoom
import sys
import trimesh

import torch 

from constants import *
import angle_utils

base_path = os.path.join(BASE_PATH, 'caps_main')

sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, 'smpl'))

from smpl.smpl_webuser.serialization import load_model

smpl_female = load_model(os.path.join(base_path, 'smpl_models', 'smpl', 'SMPL_FEMALE.pkl'))
smpl_male = load_model(os.path.join(base_path, 'smpl_models', 'smpl', 'SMPL_MALE.pkl'))

# HUMAN_MAT = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.05, 0.05, 0.25, 0.5], metallicFactor=0.6, roughnessFactor=0.5, alphaMode='BLEND')
HUMAN_MAT = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.05, 0.05, 0.8, 0.0], metallicFactor=0.6, roughnessFactor=0.5)
INFERNO = cm.get_cmap('inferno', 255)
JET = cm.get_cmap('jet', 255)
MIDDLE_FILLER = 100

def get_rest_verts(body_shape, gender):
    if gender == 1: #FEMALE
        smpl_model = SMPL_FEMALE
    else: # MALE
        smpl_model = SMPL_MALE

    smpl_verts, _ = smpl_model(POSE, th_betas=body_shape.reshape(1, -1))
    return smpl_verts[0]


def get_gt_verts(data_label):
    data_label = data_label.to("cpu").numpy()

    betas_gt = data_label[72:82]
    angles_gt = data_label[82:154]
    root_shift_est_gt = data_label[154:157]
    gender = data_label[157]
    root_shift_est_gt[1] *= -1
    root_shift_est_gt[2] *= -1

    R_root = angle_utils.matrix_from_dir_cos_angles(angles_gt[0:3])
    flip_root_euler = np.pi
    flip_root_R = angle_utils.eulerAnglesToRotationMatrix([flip_root_euler, 0.0, 0.0])
    angles_gt[0:3] = angle_utils.dir_cos_angles_from_matrix(np.matmul(flip_root_R, R_root))

    if gender == 1:
        smpl_model = smpl_female
    else:
        smpl_model = smpl_male
    
    for beta in range(betas_gt.shape[0]):
        smpl_model.betas[beta] = betas_gt[beta]
    for angle in range(angles_gt.shape[0]):
        smpl_model.pose[angle] = angles_gt[angle]

    smpl_verts_gt = np.array(smpl_model.r)
    for s in range(root_shift_est_gt.shape[0]):
        smpl_verts_gt[:, s] += (root_shift_est_gt[s] - float(smpl_model.J_transformed[0, s]))

    # smpl_verts_gt = np.concatenate((-(smpl_verts_gt[:, 1:2]),# - 0.286 + 0.0143),
    #                                 smpl_verts_gt[:, 0:1],# - 0.286 + 0.0143,
    #                                 smpl_verts_gt[:, 2:3]), axis=1)

    smpl_verts_gt[:, 1:2] *= -1
    
    return torch.tensor(smpl_verts_gt)


def generate_colors(pmap):
    verts_color_jet = np.clip(cm.jet(pmap.to("cpu").numpy()/30.)[:, 0:3], a_min=0.0, a_max=1.0)
    return torch.tensor(verts_color_jet).float().to(DEVICE)
    

def render_trimesh(vertices, colors, edit=False):
    # trimesh.vertice.shape = 6890, 3
    # trimesh.faces.shape = 13776, 3

    if edit:
        vertices_copy = vertices.to("cpu").clone()
        vertices_copy = vertices_copy[:, [1, 0, 2]]
        vertices_copy[:, -1] *= -1.0
    else:
        vertices_copy = vertices.to("cpu").clone()

    return trimesh.base.Trimesh(vertices=vertices_copy.numpy(),
                                faces=FACES.to("cpu").squeeze(0).numpy(),
                                vertex_colors=colors.to("cpu").numpy())


def generate_mesh(smpl_verts, smpl_faces):
    smpl_verts[:, 0] += 0.0286*2 #vertical
    smpl_verts[:, 1] += 0.0143

    shift_estimate_sideways = np.max([0.8, np.max(smpl_verts[:, 1])]) + 0.1
    shift_both_amount = -np.min([-0.1, np.min(smpl_verts[:, 1])])

    smpl_verts_quad = np.concatenate((smpl_verts, np.ones((smpl_verts.shape[0], 1))), axis = 1)
    smpl_verts_quad = np.swapaxes(smpl_verts_quad, 0, 1)
    
    transform_A = np.identity(4)
    transform_A[1, 3] = shift_both_amount
    smpl_verts_A = np.swapaxes(np.matmul(transform_A, smpl_verts_quad), 0, 1)[:, 0:3]

    smpl_verts_A *= (64*0.0286 / 1.92)
    smpl_verts_A[:, 0] += (1.92 - 64*0.0286)/2
    smpl_verts_A[:, 1] += (0.84 - 27*0.0286)/2


    shift =  int(7.5*(shift_estimate_sideways+shift_both_amount)/0.0286)+179

    mesh = trimesh.base.Trimesh(vertices=np.array(smpl_verts_A), faces=smpl_faces)
    return pyrender.Mesh.from_trimesh(mesh, material=HUMAN_MAT, smooth=True), shift


def generate_scene(mesh, rotate=False):
    scene = pyrender.Scene()

    scene.add(mesh)
    camera_pose = np.eye(4)

    camera_pose[0, 0] = np.cos(np.pi/2)
    camera_pose[0, 1] = np.sin(np.pi/2)
    camera_pose[1, 0] = -np.sin(np.pi/2)
    camera_pose[1, 1] = np.cos(np.pi/2)
    rot_udpim = np.eye(4)

    rot_y = 180*np.pi/180.
    rot_udpim[1,1] = np.cos(rot_y)
    rot_udpim[2,2] = np.cos(rot_y)
    rot_udpim[1,2] = np.sin(rot_y)
    rot_udpim[2,1] = -np.sin(rot_y)
    camera_pose = np.matmul(rot_udpim,  camera_pose)

    camera_pose[0, 3] = 0
    camera_pose[1, 3] = 1.2
    camera_pose[2, 3] = -1.0

    if rotate:
        rotation_matrix = np.array([[1, 0, 0, 0],
                                    [0, np.cos(np.pi), -np.sin(np.pi), 0],
                                    [0, np.sin(np.pi), np.cos(np.pi), 0],
                                    [0, 0, 0, 1]])
        camera_pose = np.dot(rotation_matrix, camera_pose)

    magnify =(1+MIDDLE_FILLER/880)*(64*.0286)

    camera = pyrender.OrthographicCamera(xmag=magnify, ymag=magnify)

    scene.add(camera, pose=camera_pose)
    
    light = pyrender.SpotLight(color=np.ones(3), intensity=250.0, innerConeAngle=np.pi / 10.0, outerConeAngle=np.pi / 2.0)
    light_pose = np.copy(camera_pose)
    light_pose[0, 3] = 0.8
    light_pose[1, 3] = -0.5
    light_pose[2, 3] = -2.5

    light_pose2 = np.copy(camera_pose)
    light_pose2[0, 3] = 2.5
    light_pose2[1, 3] = 1.0
    light_pose2[2, 3] = -5.0

    light_pose3 = np.copy(camera_pose)
    light_pose3[0, 3] = 1.0
    light_pose3[1, 3] = 5.0
    light_pose3[2, 3] = -4.0

    scene.add(light, pose=light_pose)
    scene.add(light, pose=light_pose2)
    scene.add(light, pose=light_pose3)

    r = pyrender.OffscreenRenderer(600, 880+MIDDLE_FILLER)
    render, _ = r.render(scene)

    zeros_append = np.zeros((render.shape[0], 191, 3)).astype(np.uint8) + 255
    im_to_show = np.concatenate((zeros_append, render), axis = 1)
    return im_to_show


def render_scene(mesh, camera_pose=None, rotate=False):
    if rotate:
        lval = 300
        HUMAN_MAT = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.05, 0.05, 0.25, 0.5], metallicFactor=0.6, roughnessFactor=0.5, alphaMode='BLEND')
    else:
        lval = 150
        HUMAN_MAT = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.05, 0.05, 0.25, 0.5], metallicFactor=0.6, roughnessFactor=0.5, alphaMode='BLEND')

    
    scene = pyrender.Scene(ambient_light=(lval, lval, lval))
    color_mesh = pyrender.Mesh.from_trimesh(
            mesh,
            smooth=False)
    scene.add(color_mesh)
    wire_mesh = pyrender.Mesh.from_trimesh(
        mesh,
        material=HUMAN_MAT,
        wireframe=True,
        smooth=False)
    scene.add(wire_mesh)
    if camera_pose is None:
        camera_pose = np.eye(4)

        camera_pose[0, 0] = np.cos(np.pi/2)
        camera_pose[0, 1] = np.sin(np.pi/2)
        camera_pose[1, 0] = -np.sin(np.pi/2)
        camera_pose[1, 1] = np.cos(np.pi/2)
        rot_udpim = np.eye(4)

        rot_y = 180*np.pi/180.
        rot_udpim[1,1] = np.cos(rot_y)
        rot_udpim[2,2] = np.cos(rot_y)
        rot_udpim[1,2] = np.sin(rot_y)
        rot_udpim[2,1] = -np.sin(rot_y)
        camera_pose = np.matmul(rot_udpim,  camera_pose)

        camera_pose[0, 3] = -1
        camera_pose[1, 3] = 0.5
        camera_pose[2, 3] = -1.0


        if rotate:
            rotation_matrix = np.array([[1, 0, 0, 0],
                                        [0, np.cos(np.pi), -np.sin(np.pi), 0],
                                        [0, np.sin(np.pi), np.cos(np.pi), 0],
                                        [0, 0, 0, 1]])
            camera_pose = np.dot(rotation_matrix, camera_pose)


    magnify =(1+MIDDLE_FILLER/880)*(64*.0286)

    camera = pyrender.OrthographicCamera(xmag=magnify, ymag=magnify)

    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=50.0, innerConeAngle=np.pi / 10.0, outerConeAngle=np.pi / 2.0)
    light_pose = np.copy(camera_pose)
    light_pose[0, 3] = 0.8
    light_pose[1, 3] = -0.5
    light_pose[2, 3] = -2.5

    light_pose2 = np.copy(camera_pose)
    light_pose2[0, 3] = 2.5
    light_pose2[1, 3] = 1.0
    light_pose2[2, 3] = -5.0

    light_pose3 = np.copy(camera_pose)
    light_pose3[0, 3] = 1.0
    light_pose3[1, 3] = 5.0
    light_pose3[2, 3] = -4.0

    scene.add(light, pose=light_pose)
    scene.add(light, pose=light_pose2)
    scene.add(light, pose=light_pose3)

    r = pyrender.OffscreenRenderer(600, 880 + MIDDLE_FILLER)
    render, _ = r.render(scene)
    render = np.array(render)[40:560, 60:300, :]
    return render


def render_video(mesh):
    renders = []
    HUMAN_MAT = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.05, 0.05, 0.15, 0.5], metallicFactor=0.6, roughnessFactor=0.5)
    color_mesh = pyrender.Mesh.from_trimesh(
            mesh,
            smooth=False)
    wire_mesh = pyrender.Mesh.from_trimesh(
        mesh,
        material=HUMAN_MAT,
        wireframe=True)

    magnify =(1+MIDDLE_FILLER/880)*(64*.0286)

    r = pyrender.OffscreenRenderer(600, 880 + MIDDLE_FILLER)

    for rot in range(0, 360*2 + 5, 10):
        scene = pyrender.Scene(ambient_light=(400, 400, 400))
        scene.add(wire_mesh)
        scene.add(color_mesh)
        camera = pyrender.OrthographicCamera(xmag=magnify, ymag=magnify)

        camera_pose = np.eye(4)
        camera_pose[0, 0] = np.cos(np.pi/2)
        camera_pose[0, 1] = np.sin(np.pi/2)
        camera_pose[1, 0] = -np.sin(np.pi/2)
        camera_pose[1, 1] = np.cos(np.pi/2)

        rot_udpim = np.eye(4)
        rot_y = 180*np.pi/180.
        rot_udpim[1,1] = np.cos(rot_y)
        rot_udpim[2,2] = np.cos(rot_y)
        rot_udpim[1,2] = np.sin(rot_y)
        rot_udpim[2,1] = -np.sin(rot_y)
        camera_pose = np.matmul(rot_udpim,  camera_pose)

        camera_pose[0, 3] = -1
        camera_pose[1, 3] = 0.5
        camera_pose[2, 3] = -1.0

        rotation_matrix = np.array([[1, 0, 0, 0],
                                    [0, np.cos(np.radians(rot)), -np.sin(np.radians(rot)), 0],
                                    [0, np.sin(np.radians(rot)), np.cos(np.radians(rot)), 0],
                                    [0, 0, 0, 1]])
        camera_pose = np.dot(rotation_matrix, camera_pose)

        scene.add(camera, pose=camera_pose)

        render, _ = r.render(scene)
        render = np.array(render)[40:560, 60:300, :]
        renders.append(render)


        # break
    
    return renders


def gen_depth(depth):
    depth = depth.squeeze(-1)
    depth = zoom(depth, 3.435, order=1)
    depth -= 1700
    depth = depth.astype(float)/ 500.
    depth = INFERNO(np.clip(depth, a_min=0, a_max=1))[:, :, 0:3]*255.
    depth = depth.astype(np.uint8)
    return depth 


def gen_pressure(pressure):
    pressure = pressure.squeeze(-1)
    pressure = zoom(pressure, (3.435*2, 3.435*2/1.04), order=0)
    pressure = (np.clip((JET(pressure/40)[:, :, 0:3] + 0.1), a_min=0, a_max=1)*255).astype(np.uint8)
    return pressure


def generate_viz(pressure, depth, verts):
    mesh, shift = generate_mesh(verts, np.array(smpl_female.f))
    im_to_show = generate_scene(mesh)

    depth = gen_depth(depth)
    im_to_show[int(MIDDLE_FILLER/2):int(MIDDLE_FILLER/2)+440, 3:np.shape(depth)[1]+3, :] = depth

    pressure = gen_pressure(pressure)
    im_to_show[int(MIDDLE_FILLER/2):int(MIDDLE_FILLER/2)+440, shift: shift+178, :] = pressure[:, :, 0:3]
    return im_to_show

