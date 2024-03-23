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


def prepare_transforms(image_size_type, is_affine, is_erase, train=False):
    transforms = []
    if image_size_type == "resized224":
        transforms.append(trans_utils.Resize)
    if train:
        if is_affine:
            transforms.append(trans_utils.RandomAffine) # just translating images, rotation handled separately
        if is_erase:
            transforms.append(trans_utils.RandomCutOut)
            transforms.append(trans_utils.PixelDropout)
    return transforms


def prepare_loader(data_path, data_files, batch_size, image_size_type, \
                    exp_type, normalize_pressure, normalize_depth, \
                    is_affine, is_erase, training=False, train_on_real='all'):
    data_transforms = prepare_transforms(image_size_type=image_size_type, 
                                        is_affine=is_affine, 
                                        is_erase=is_erase, 
                                        train=training)

    dataset = PMMTrainerDataset(data_path=data_path, 
                                data_files=data_files, 
                                transforms=data_transforms, 
                                exp_type=exp_type, 
                                normalize_pressure=normalize_pressure, 
                                normalize_depth=normalize_depth, 
                                training=training, 
                                is_affine=is_affine, 
                                train_on_real=train_on_real)
    
    # shuffle, drop_last should be True for training case
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=training, drop_last=training, num_workers=4)
    return loader, dataset


def prepare_dataloaders(data_path, train_files, val_files, batch_size, image_size_type, \
                        exp_type, normalize_pressure, normalize_depth, is_affine, is_erase, train_on_real='all'):
    print ('Starting train dataset prepare')
    train_loader, train_dataset = prepare_loader(data_path=data_path, 
                                                data_files=train_files, 
                                                batch_size=batch_size, 
                                                image_size_type=image_size_type, 
                                                exp_type=exp_type, 
                                                normalize_pressure=normalize_pressure, 
                                                normalize_depth=normalize_depth,
                                                is_affine=is_affine, 
                                                is_erase=is_erase, 
                                                training=True, 
                                                train_on_real=train_on_real)
    print ('Prepared train dataset')
    print ()
    
    print ('Starting val dataset prepare')
    val_loader, val_dataset = prepare_loader(data_path=data_path, 
                                            data_files=val_files, 
                                            batch_size=batch_size, 
                                            image_size_type=image_size_type, 
                                            exp_type=exp_type, 
                                            normalize_pressure=normalize_pressure, 
                                            normalize_depth=normalize_depth,
                                            is_affine=False, 
                                            is_erase=False, 
                                            training=False, 
                                            train_on_real='all')
    print ('Prepared val dataset')
    print ()

    return train_loader, val_loader, train_dataset, val_dataset


class PMMTrainerDataset(Dataset):

    def __init__(self, data_path, data_files, transforms, exp_type, \
                normalize_pressure, normalize_depth, training, is_affine, train_on_real):
        super(PMMTrainerDataset, self).__init__()
        self.data_path = data_path
        self.transforms = transforms
        self.exp_type = exp_type
        self.normalize_pressure = normalize_pressure
        self.normalize_depth = normalize_depth
        self.training = training
        self.load_verts = (not training)
        self.is_affine = is_affine
        self.train_on_real = train_on_real

        self._prepare_dataset(data_files)
    
    def _concat_data_returns(self, data_returns):
        self.data_pressure_x = utils.concatenate(self.data_pressure_x, data_returns[0])
        self.data_depth_x = utils.concatenate(self.data_depth_x, data_returns[1])
        self.data_label_y = utils.concatenate(self.data_label_y, data_returns[2])
        self.data_pmap_y = utils.concatenate(self.data_pmap_y, data_returns[3])
        self.data_verts_y = utils.concatenate(self.data_verts_y, data_returns[4])
        self.data_names_y = utils.concatenate(self.data_names_y, data_returns[5])

    def _purge_for_overfitting(self, use_till):
        self.data_pressure_x = self.data_pressure_x[:use_till, ...]
        self.data_depth_x = self.data_depth_x[:use_till, ...]
        self.data_label_y = self.data_label_y[:use_till, ...]
        self.data_pmap_y = self.data_pmap_y[:use_till, ...]
        self.data_verts_y = self.data_verts_y[:use_till, ...]
        self.data_names_y = self.data_names_y[:use_till, ...]

    def _prepare_dataset(self, data_files):
        real_file, synth_file = data_files
        self.data_pressure_x = None
        self.data_depth_x = None
        self.data_label_y = None
        self.data_pmap_y = None
        self.data_verts_y = None
        self.data_names_y = None

        if real_file is not None:
            data_lines = utils.load_data_lines(real_file)
            for cover_str in ['uncover', 'cover1', 'cover2']:
                data_pressure_x, data_depth_x, data_label_y, data_pmap_y, data_verts_y, data_names_y = \
                    SLPDataset(self.data_path).prepare_dataset(data_lines, cover_str, load_verts=self.load_verts, train_on_real=self.train_on_real)
                self._concat_data_returns((data_pressure_x, data_depth_x, data_label_y, data_pmap_y, data_verts_y, data_names_y))
            
            if self.exp_type == 'overfit':
                self._purge_for_overfitting(USE_TILL)
        else:
            print ('No real data used')

        if synth_file is not None:
            data_lines = utils.load_data_lines(synth_file)
            data_pressure_x, data_depth_x, data_label_y, data_pmap_y, data_verts_y, data_names_y = \
                BPDataset(self.data_path).prepare_dataset(data_lines, load_verts=self.load_verts)
            self._concat_data_returns((data_pressure_x, data_depth_x, data_label_y, data_pmap_y, data_verts_y, data_names_y))

            if self.exp_type == 'overfit':
                self._purge_for_overfitting(2*USE_TILL)
        else:
            print ('No synth data used')

        self.data_pressure_x = torch.tensor(self.data_pressure_x).float().permute(0, 3, 1, 2)
        self.data_depth_x = torch.tensor(self.data_depth_x).float().permute(0, 3, 1, 2)
        self.data_pmap_y = torch.tensor(self.data_pmap_y).float()
        self.data_verts_y = torch.tensor(self.data_verts_y).float()
        
        print ('Mixed', \
                'dpx', self.data_pressure_x.shape, \
                'ddx', self.data_depth_x.shape, \
                'dy', self.data_label_y.shape, \
                'data_pmap', self.data_pmap_y.shape, \
                'data_verts', self.data_verts_y.shape, \
                'data names', self.data_names_y.shape)

    def __len__(self):
        return self.data_pressure_x.shape[0]
    
    def _apply_transforms(self, label, pressure_image, depth_image, pmap):
        if self.training and self.is_affine:
            label[82:154], pressure_image, depth_image = trans_utils.Rotate(label[82:154], pressure_image, depth_image)
        images = [pressure_image, depth_image]
        for transform_fn in self.transforms:
            images, pmap = transform_fn(images, pmap)
        pressure_image, depth_image = images
        return torch.tensor(label).float(), pressure_image, depth_image, pmap

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
        pressure_image
        depth_image 
        label 
        pmap 
        verts
        file_name
        '''
        file_name = self.data_names_y[index]
        label = self.data_label_y[index]
        pmap = self.data_pmap_y[index]

        org_pressure_image = self.data_pressure_x[index].clone()
        pressure_image = self.data_pressure_x[index].clone()
        if self.normalize_pressure:
            if file_name[0] == 'r':
                pressure_image /= MAX_PRESSURE_REAL
                pmap /= MAX_PMAP_REAL
            else:
                pressure_image /= MAX_PRESSURE_SYNTH
                pmap /= MAX_PMAP_SYNTH
        
        org_depth_image = self.data_depth_x[index].clone()
        depth_image = self.data_depth_x[index].clone()
        if self.normalize_depth:
            if file_name[0] == 'r':
                depth_image /= MAX_DEPTH_REAL
            else:
                depth_image /= MAX_DEPTH_SYNTH
        
        label, pressure_image, depth_image, pmap = self._apply_transforms(label.copy(), pressure_image.clone(), depth_image.clone(), pmap.clone())
        verts = self.data_verts_y[index] if self.load_verts else np.array([])
        return org_pressure_image, pressure_image.float(), org_depth_image, depth_image.float(), label, pmap, verts, file_name

