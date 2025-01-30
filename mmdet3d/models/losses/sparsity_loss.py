# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from mmdet.models.builder import LOSSES
import torch
import torch.nn as nn
import torch.nn.functional as F

@LOSSES.register_module()
class OccSparsityLoss(nn.Module):
    def __init__(self, loss_weight=0.1, loss_name='loss_sparsity'):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name 
        
    def forward(self, density):
        return self.loss_weight * density.norm(dim=-1).mean()
    
    @property
    def loss_name(self):
        return self._loss_name

