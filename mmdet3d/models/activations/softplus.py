# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from mmcv.cnn.bricks import ACTIVATION_LAYERS
import torch.nn as nn
import torch.nn.functional as F

@ACTIVATION_LAYERS.register_module()
class Softplus(nn.Module):
    __constants__ = ['beta', 'threshold']
    beta: int
    threshold: int

    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        return F.softplus(input, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)