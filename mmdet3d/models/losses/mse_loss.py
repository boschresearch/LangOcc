# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from ..builder import LOSSES
import torch.nn as nn
import torch

@LOSSES.register_module()
class CustomMSELoss(nn.Module):
    def __init__(self, loss_weight=1.0, loss_name='loss_custommse'):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred, target, ray_weights, **kwargs):
        loss = (ray_weights * self.loss_fn(pred, target)).sum(dim=-1).mean()
        return self.loss_weight * loss