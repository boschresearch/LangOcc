# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from ..builder import LOSSES
import torch.nn as nn
import torch

@LOSSES.register_module()
class CosSimLoss(nn.Module):
    def __init__(self, loss_weight=1.0, loss_name='loss_cossim'):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.cossim = nn.CosineSimilarity(dim=-1)

    def forward(self, pred, target, ray_weights, **kwargs):
        loss = (ray_weights.squeeze() * (1 - self.cossim(pred, target))).mean()

        return self.loss_weight * loss
    
@LOSSES.register_module()
class CosSimWeightedMSELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduce_by= 'mean', loss_name='loss_cossimMSE', stop_cossim=False):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.cosim_coef = 2
        self.reduce_by = reduce_by
        self.stop_cossim = stop_cossim

    def forward(self, pred, target, ray_weights, **kwargs):
        neg_cossim = self.cosim_coef * (1 - self.cossim(pred, target))
        ray_weights = ray_weights.squeeze() * neg_cossim
        if self.stop_cossim:
            ray_weights = ray_weights.detach() # do not compute gradients through cosine similarity
        if self.reduce_by == 'mean':
            loss = (ray_weights * self.loss_fn(pred, target).mean(-1)).mean()
        else:
            loss = (ray_weights * self.loss_fn(pred, target).sum(-1)).mean()
            
        return self.loss_weight * loss    