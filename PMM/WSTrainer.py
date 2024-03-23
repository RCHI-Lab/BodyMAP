from collections import defaultdict
import json
import numpy as np
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from WSPMMModel2 import WSPMMModel as WSPMM2
from WSPMMModel5 import WSPMMModel as WSPMM5
from PMMTrainerDataset import prepare_dataloaders
from PMMInferDataset import prepare_loader as prepare_inferloader
from PMMInfer import PMMInfer
from PMMMetric import PMMMetric
from constants import *
import loss_utils



MODEL_FN_DICT = {
    'WSPMM2' : WSPMM2,
    'WSPMM5' : WSPMM5,
}


class WSTrainer():
    
    def __init__(self, args):
        self.args = args 
        print (f'Starting WS PM Trainer Object for {self.args["name"]} {self.args["exp_type"]} model')

        self.criterion = nn.L1Loss(reduction='mean')
        self.criterion_mse = nn.MSELoss(reduction='mean')
        self.criterion_contact = nn.NLLLoss(reduction='mean')

        if self.args.get('load_MOD1_path', None) is not None:
            self.MOD1 = torch.load(self.args['load_MOD1_path']).to(DEVICE)
            self.MOD1.eval()
        else:
            self.MOD1 = None 

    def _load_data(self):
        start_time = time.time()
        if self.args['real_train_file'] is None and self.args['synth_train_file'] is None:
            print ('ERROR: No dataset being used')
            exit(-1)
        self.train_loader, self.val_loader, train_dataset, val_dataset = prepare_dataloaders(self.args['data_path'], \
                                                                (self.args['real_train_file'] if self.args['use_real'] else None, \
                                                                self.args['synth_train_file'] if self.args['use_synth'] else None), \
                                                                (self.args['real_val_file'] if self.args['use_real'] else None, \
                                                                None), # setting synth val file as None to fasten up experiments
                                                                self.args['batch_size'], self.args['image_size_type'], \
                                                                self.args['exp_type'], \
                                                                self.args['normalize_pressure'], self.args['normalize_depth'], \
                                                                self.args['is_affine'], self.args['is_erase'], \
                                                                self.args['train_on_real'])
        
        self.infer_loader, _ = prepare_inferloader(self.args['data_path'], (self.args['real_val_file'], self.args['synth_val_file']), \
                                            self.args['batch_size'], self.args['image_size_type'], \
                                            self.args['normalize_pressure'], self.args['normalize_depth'])

        self.metric_loader = self.val_loader
        print ('Using val dataset for metric testing as well')

        end_time = time.time()
        self.args['dataset_setup_time'] = (end_time - start_time)
        print (f'Dataset Setup Time = {self.args["dataset_setup_time"]: .0f} s')
        self.args['train_len'] = len(train_dataset)
        self.args['val_len'] = len(val_dataset)

    def _get_losses(self, mesh_pred, pmap_pred, mesh_gt, labels_gt, pmap_gt, pi_org):
        if self.args['smpl_loss']:
            betas_loss = self.criterion(mesh_pred['out_betas'], labels_gt[:, 72:82])

            # try changing to l1 as in paper
            joint_angles_loss = self.criterion_mse(mesh_pred['out_joint_angles'][:, 3:], labels_gt[:, 85:154]) # skip root rotation from loss

            root_angle_loss = self.criterion(mesh_pred['out_root_angles'][:, :3], torch.cos(labels_gt[:, 82:85])) + \
                                self.criterion(mesh_pred['out_root_angles'][:, 3:], torch.sin(labels_gt[:, 82:85]))

            joint_pos_loss = ((mesh_gt['out_joint_pos'] - mesh_pred['out_joint_pos']) + 1e-7) ** 2
            joint_pos_loss = joint_pos_loss.reshape(-1, 24, 3).sum(dim=-1).sqrt().mean()

            v2v_loss = ((mesh_gt['out_verts'] - mesh_pred['out_verts'])**2).sum(dim=-1).sqrt().mean()

            smpl_loss = betas_loss * (1/1.728158146914805) + \
                    joint_angles_loss * (1/0.29641429463719227) + \
                    root_angle_loss * (1/0.3684988513298487) + \
                    joint_pos_loss * (1/0.1752780723422608)
        else:
            v2v_loss = smpl_loss = torch.tensor(0).float().to(DEVICE)

        if self.args['pmap_loss']:
            pd_proj_pressure = loss_utils.get_projected_pressure(mesh_pred['out_verts'].clone(), pmap_pred)
            gt_proj_pressure = pi_org.clone()

            proj_pressure_loss = self.criterion_mse(pd_proj_pressure, gt_proj_pressure)
            
            up_halves = pmap_pred[mesh_pred['out_verts'][:, :, -1] < 0]
            pmap_reg_loss = self.criterion_mse(up_halves, torch.zeros_like(up_halves))
        else:
            proj_pressure_loss = torch.tensor(0).float().to(DEVICE)
            pmap_reg_loss = torch.tensor(0).float().to(DEVICE)

        loss =  self.args['lambda_smpl_loss'] * smpl_loss + \
                self.args['lambda_v2v_loss'] * v2v_loss * (1/0.1752780723422608) + \
                self.args['lambda_proj_loss'] * proj_pressure_loss + \
                self.args['lambda_preg_loss'] * pmap_reg_loss

        return {
            'total_loss'        : loss,
            'smpl_loss'         : smpl_loss,
            'v2v_loss'          : v2v_loss,
            'proj_pressure_loss': proj_pressure_loss,
            'pmap_reg_loss'     : pmap_reg_loss,
        }

    def _train_epoch(self, model):
        model.train()
        running_losses = defaultdict(float)
        with torch.autograd.set_detect_anomaly(True):
            for batch_org_pressure_images, \
                batch_pressure_images, \
                batch_org_depth_images, \
                batch_depth_images, batch_labels, batch_pmap, _, _ in iter(self.train_loader):
                self.opt.zero_grad()

                batch_org_pressure_images = batch_org_pressure_images.to(DEVICE)
                batch_pressure_images = batch_pressure_images.to(DEVICE)
                batch_depth_images = batch_depth_images.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                batch_pmap = batch_pmap.to(DEVICE)
                batch_labels_copy = batch_labels.clone()

                if self.MOD1 is not None:
                    batch_mesh_pred, _, img_feat, _ = self.MOD1.infer(batch_depth_images.clone(), batch_pressure_images.clone(), batch_labels[:, 157:159].clone())
                    mesh_gt = self.MOD1.mesh_infer_gt(torch.cat((
                                            batch_labels[:, 72:82], 
                                            batch_labels[:, 154:157],
                                            torch.cos(batch_labels[:, 82:85]),
                                            torch.sin(batch_labels[:, 82:85]),
                                            batch_labels[:, 85:154],
                                        ), axis=1), batch_labels[:, 157:159])
                    _, batch_pmap_pred, _, _ = model(batch_depth_images, batch_pressure_images, batch_labels[:, 157:159], batch_mesh_pred['out_verts'].clone(), img_feat)
                else:
                    batch_mesh_pred, batch_pmap_pred, _, _ = model(batch_depth_images, batch_pressure_images, batch_labels[:, 157:159])
                    mesh_gt = model.mesh_infer_gt(torch.cat((
                                            batch_labels[:, 72:82], 
                                            batch_labels[:, 154:157],
                                            torch.cos(batch_labels[:, 82:85]),
                                            torch.sin(batch_labels[:, 82:85]),
                                            batch_labels[:, 85:154],
                                        ), axis=1), batch_labels[:, 157:159])
                mesh_gt = {
                            'out_verts' : mesh_gt['out_verts'],
                            'out_joint_pos' : mesh_gt['out_joint_pos']
                        }

                losses = self._get_losses(batch_mesh_pred, batch_pmap_pred, mesh_gt, batch_labels_copy, batch_pmap, batch_org_pressure_images)
                for k in losses:
                    running_losses[k] += losses[k].item()
                losses['total_loss'].backward()

                self.opt.step()
        for k in running_losses:
            running_losses[k] /= self.args['train_len']
        return running_losses

    def _validate_epoch(self, model):
        model.eval()
        running_losses = defaultdict(float)
        with torch.no_grad():
            for batch_org_pressure_images, \
                batch_pressure_images, \
                batch_org_depth_images, \
                batch_depth_images, \
                batch_labels, batch_pmap, batch_verts, _ in iter(self.val_loader):

                batch_org_pressure_images = batch_org_pressure_images.to(DEVICE)
                batch_pressure_images = batch_pressure_images.to(DEVICE)
                batch_depth_images = batch_depth_images.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                batch_verts = batch_verts.to(DEVICE)
                batch_pmap = batch_pmap.to(DEVICE)
                batch_labels_copy = batch_labels.clone()
                mesh_gt = {
                    'out_verts' : batch_verts,
                    'out_joint_pos' : batch_labels_copy[:, :72]/1000.,
                }

                if self.MOD1 is not None:
                    batch_mesh_pred, _, img_feat, _ = self.MOD1.infer(batch_depth_images.clone(), batch_pressure_images.clone(), batch_labels[:, 157:159].clone())
                    _, batch_pmap_pred, _, _ = model(batch_depth_images, batch_pressure_images, batch_labels[:, 157:159], batch_mesh_pred['out_verts'].clone(), img_feat)
                else:
                    batch_mesh_pred, batch_pmap_pred, _, _ = model(batch_depth_images, batch_pressure_images, batch_labels[:, 157:159])

                losses = self._get_losses(batch_mesh_pred, batch_pmap_pred, mesh_gt, batch_labels_copy, batch_pmap, batch_org_pressure_images)
                for k in losses:
                    running_losses[k] += losses[k].item()
        for k in running_losses:
            running_losses[k] /= self.args['val_len']
        return running_losses

    def _train_model(self, model):
        model = model.to(DEVICE)
        print (f"Starting training for experiment - {self.args['name']} {self.args['exp str']}")
        print (f"Starting model training for {self.args['epochs'] - self.starting_epoch} epochs starting from {self.starting_epoch}")

        for e in tqdm(range(self.starting_epoch, self.args['epochs'], 1)):
            train_losses = self._train_epoch(model)
            val_losses = self._validate_epoch(model)

            for k in train_losses:
                self.writer.add_scalars(f'Loss/{k}', {'train': train_losses[k], 'val': val_losses[k]}, e)

            self.writer.add_scalar('Learning_rate', self.opt.param_groups[0]['lr'], e)
            
            if e % self.args['epochs_val_viz'] == 0:
                PMMInfer(model, self.infer_loader, writer=self.writer, save_gt=(e==0), epoch=e, pmap_norm=self.args['normalize_pressure'], infer_pmap=self.args['infer_pmap'], infer_smpl=self.args['infer_smpl'], MOD1=self.MOD1)

            if e % self.args['epochs_metric'] == 0:
                PMMMetric(model, self.metric_loader, writer=self.writer, epoch=e, pmap_norm=self.args['normalize_pressure'], infer_pmap=self.args['infer_pmap'], infer_smpl=self.args['infer_smpl'], MOD1=self.MOD1)
            
            if e % self.args['epochs_save'] == 0 or e % self.args['epochs_metric'] == 0:
                self._save_model(model, e)
                model.to(DEVICE)

        PMMInfer(model, self.infer_loader, writer=self.writer, save_gt=(e==0), epoch=e, pmap_norm=self.args['normalize_pressure'], infer_pmap=self.args['infer_pmap'], infer_smpl=self.args['infer_smpl'], MOD1=self.MOD1)
        metric = PMMMetric(model, self.metric_loader, writer=self.writer, epoch=e, pmap_norm=self.args['normalize_pressure'], infer_pmap=self.args['infer_pmap'], infer_smpl=self.args['infer_smpl'], MOD1=self.MOD1)
        self.args['metric'] = metric
        self._save_model(model, self.args['epochs'])
        return model

    def _save_model(self, model, epoch):
        model = model.to("cpu")
        if epoch % self.args['epochs_metric'] == 0:
            torch.save(model, os.path.join(self.output_path, f'modelPMM_{epoch}.pth'))
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'opt_state_dict' : self.opt.state_dict(),
        }, os.path.join(self.output_path , 'modelPMM.pt'))

    def _save_args(self):
        args_str = json.dumps(self.args)
        with open(os.path.join(self.output_path, 'exp.json'), 'w') as f:
            f.write(args_str)
        self.writer.add_text('Args', args_str)

    def _setup(self):
        self._load_data()
        run_folder = 'runs'
        model_folder = self.args['name']
        self.args['save_path'] = os.path.join(self.args['save_path'], f'{self.args["exp_type"]}')

        self.output_path = os.path.join(self.args['save_path'], 'exps', model_folder)
        for path in [self.output_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        self.writer = SummaryWriter(os.path.join(self.args['save_path'], run_folder, self.args['name']))
        self._save_args()

    def _setup_model(self):
        main_model_fn = MODEL_FN_DICT.get(self.args['main_model_fn'], None)
        if main_model_fn is None:
            print ('ERROR: invalid main_model_fn')
            exit(-1)
        model = main_model_fn(
                        model_fn=self.args['model_fn'], \
                        feature_size=self.args['feature_size'], \
                        out_size=self.args['out_size'], \
                        vertex_size=self.args['vertex_size'], \
                        batch_size=self.args['batch_size'], \
                        modality=self.args['modality'], \
                        indexing_mode=self.args['indexing_mode'])
        for param in model.parameters():
            param.requires_grad = True
        try:
            for param in model.mesh_model.parameters():
                param.requires_grad = False
        except:
            pass
        self.opt = optim.Adam(model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        return model

    def _load_model(self, model):
        is_loaded = False
        model_path = os.path.join(self.output_path, 'modelPMM.pt')
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.opt.load_state_dict(checkpoint['opt_state_dict'])
                for state in self.opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(DEVICE)
                self.starting_epoch = checkpoint['epoch'] + 1
                is_loaded = True
                print (f'Model loaded. Last epoch = {checkpoint["epoch"]}')
            except Exception as e:
                print (f'Model loading failed {e}')
                is_loaded = False
                self.starting_epoch = 0
        else:
            is_loaded = False
            self.starting_epoch = 0
        return model

    def train_model(self):
        self._setup()

        start_time = time.time()
        model = self._setup_model()
        model = self._load_model(model)
        model = self._train_model(model)
        end_time = time.time()
        self.args['training_time'] = (end_time - start_time)
        print (f'model trained in time = {self.args["training_time"]: .0f} s')
        self._save_args()

        self.writer.flush()
        self.writer.close()
    
