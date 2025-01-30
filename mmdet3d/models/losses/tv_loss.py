# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from mmdet.models.builder import LOSSES
import torch
import torch.nn as nn
import torch.nn.functional as F

def tv_3d(voxels, weight):
    # bs, Z, H, W, C = voxels.size()
    tv_z = torch.pow(voxels[:, 1:, :, :, :] - voxels[:, :-1, :, :, :], 2).sum()
    tv_h = torch.pow(voxels[:, :, 1:, :, :] - voxels[:, :, :-1, :, :], 2).sum()
    tv_w = torch.pow(voxels[:, :, :, 1:, :] - voxels[:, :, :, :-1, :], 2).sum()
    return weight * (tv_z + tv_h + tv_w) / voxels.numel()

@LOSSES.register_module()
class TVLoss3D(nn.Module):
    def __init__(self, loss_weight=0.01,  loss_name='loss_tv', density_weight = 5, semantics_weight = .2):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name 
        self.density_weight = density_weight 
        self.semantics_weight = semantics_weight 
    def forward(self, voxel_outs):

        density_tv = tv_3d(voxel_outs[0], self.density_weight)
        semantics_tv = tv_3d(voxel_outs[1], self.semantics_weight)
        return self.loss_weight * (density_tv + semantics_tv)
    
    @property
    def loss_name(self):
        return self._loss_name

