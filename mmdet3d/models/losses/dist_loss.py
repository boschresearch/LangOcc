# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from mmdet.models.builder import LOSSES
import torch.nn as nn
from torch_efficient_distloss import eff_distloss

@LOSSES.register_module()
class DistortionLoss(nn.Module):
    def __init__(self,
                 loss_weight=0.01,
                 loss_name='loss_dist'):
        super().__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, weights, distances, intervals):
        '''
        Efficient O(N) realization of distortion loss.
        There are B rays each with N sampled points.
        weights:        Float tensor in shape [B,N]. Volume rendering weights of each point.
        distances:      Float tensor in shape [B,N]. Midpoint distance to camera of each point.
        intervals:      Float tensor in shape [B,N]. The query interval of each point.
        '''

        loss = self.loss_weight * eff_distloss(weights, distances, intervals)
        return loss

    @property
    def loss_name(self):
        return self._loss_name 