import numpy as np
import os 

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from constants import *
import utils
import trans_utils
from BPDataset import BPDataset
from SLPDataset import SLPDataset


def prepare_transforms(image_size_type):
    transforms = [trans_utils.ToTensor]
    if image_size_type == "resized224":
        transforms.append(trans_utils.Resize)
    return transforms


def prepare_loader(data_path, data_files, batch_size, image_size_type, \
                    normalize_pressure, normalize_depth):
    print ('Starting infer dataset prepare')
    data_transforms = prepare_transforms(image_size_type=image_size_type)

    dataset = PMMInferDataset(data_path=data_path, 
                                    data_files=data_files, 
                                    transforms=data_transforms,
                                    normalize_pressure=normalize_pressure, 
                                    normalize_depth=normalize_depth)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    print ('Prepared infer dataset')
    print ()
    return loader, dataset


class PMMInferDataset(Dataset):

    def __init__(self, data_path, data_files, transforms, normalize_pressure, normalize_depth):
        super(PMMInferDataset, self).__init__()
        self.data_path = data_path
        self.transforms = transforms
        self.normalize_pressure = normalize_pressure
        self.normalize_depth = normalize_depth

        self._prepare_dataset(data_files)
    
    def _concat_data_returns(self, data_returns):
        self.data_pressure_x = utils.concatenate(self.data_pressure_x, data_returns[0])
        self.data_depth_x = utils.concatenate(self.data_depth_x, data_returns[1])
        self.data_label_y = utils.concatenate(self.data_label_y, data_returns[2])
        self.data_pmap_y = utils.concatenate(self.data_pmap_y, data_returns[3])
        self.data_verts_y = utils.concatenate(self.data_verts_y, data_returns[4])
        self.data_names_y = utils.concatenate(self.data_names_y, data_returns[5])

    def _prepare_dataset(self, data_files):
        real_file, synth_file = data_files
        self.data_pressure_x = None
        self.data_depth_x = None
        self.data_label_y = None
        self.data_pmap_y = None
        self.data_verts_y = None
        self.data_names_y = None

        data_lines = utils.load_data_lines(real_file)
        for cover_str in ['uncover', 'cover1', 'cover2']:
            data_returns = SLPDataset(self.data_path).prepare_dataset(data_lines, cover_str, load_verts=True, for_infer=True)
            self._concat_data_returns(data_returns)
        data_lines = utils.load_data_lines(synth_file)
        data_returns = BPDataset(self.data_path).prepare_dataset(data_lines, load_verts=True, for_infer=True)
        self._concat_data_returns(data_returns)

        self.data_pmap_y = torch.tensor(self.data_pmap_y).float()
        self.data_verts_y = torch.tensor(self.data_verts_y).float()
        self.data_label_y = torch.tensor(self.data_label_y).float()
        
        print ('Mixed', \
                'dpx', self.data_pressure_x.shape, \
                'ddx', self.data_depth_x.shape, \
                'dy', self.data_label_y.shape, \
                'data_pmap', self.data_pmap_y.shape, \
                'data_verts', self.data_verts_y.shape, \
                'data names', self.data_names_y.shape)

    def __len__(self):
        return self.data_pressure_x.shape[0]
    
    def _apply_transforms(self, pressure_image, depth_image, pmap):
        images = [pressure_image, depth_image]
        for transform_fn in self.transforms:
            images, pmap = transform_fn(images, pmap)
        pressure_image, depth_image = images
        return pressure_image, depth_image, pmap

    def __getitem__(self, index):
        '''
        Label contents
        0-72    3D Marker Positions
        72-82   Body Shape Params
        82-154  Joint Angles 
        154-157 Root XYZ Shift
        157-159 Gender
        159     [1]
        160     Body Mass
        161     Body Height
        '''
        '''
        Returns --- 
        original_pressure_image
        pressure_image
        original_depth_image
        depth_image 
        label 
        pmap 
        verts
        file_name
        '''
        label = self.data_label_y[index]
        pmap = self.data_pmap_y[index]
        verts = self.data_verts_y[index]
        file_name = self.data_names_y[index]

        org_pressure_image = self.data_pressure_x[index].copy()
        pressure_image = self.data_pressure_x[index]
        if self.normalize_pressure:
            if file_name[0] == 'r':
                pressure_image /= MAX_PRESSURE_REAL
                pmap /= MAX_PMAP_REAL
            else:
                pressure_image /= MAX_PRESSURE_SYNTH
                pmap /= MAX_PMAP_SYNTH
        
        org_depth_image = self.data_depth_x[index].copy()
        depth_image = self.data_depth_x[index]
        if self.normalize_depth:
            if file_name[0] == 'r':
                depth_image /= MAX_DEPTH_REAL
            else:
                depth_image /= MAX_DEPTH_SYNTH
        
        pressure_image, depth_image, pmap = self._apply_transforms(pressure_image, depth_image, pmap)
        
        return org_pressure_image, pressure_image.float(), \
                org_depth_image, depth_image.float(), \
                label, pmap, verts, file_name
                
