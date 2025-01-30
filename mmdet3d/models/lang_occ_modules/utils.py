# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import List, Tuple
from pyquaternion import Quaternion
import numpy as np
import copy 

def create_img(ray_dataset, coor, outputs, img_size):
    ray_indices = torch.cat([torch.cat((q[0].new_full( (q[0].shape[0], 1) , i), q[0]), axis=-1) for i, q in enumerate(ray_dataset)])
    coor = torch.cat(coor)[:, [1,0]]
    index = torch.cat((ray_indices, coor), dim=-1)[:, 1:]
    image_dict = {}
    for module in ['depth', 'semantics']:
        module_image = torch.zeros(img_size, dtype=torch.uint8)
        module_image[tuple(index.T)] = torch.tensor(outputs[f'{module}'])
        image_dict[module] = module_image.cpu().numpy()
    return image_dict
    
def interpolate_values_langocc(values, samples, cam_idxs, pc_range, current_timestep):
    r_min = pc_range[0:3]
    r_max = pc_range[3:6]
    
    # scale to [-1, 1] for torch grid_sample
    samples_scaled = ((samples - r_min)/(r_max - r_min)) * 2 - 1

    sampled_values = F.grid_sample(
        values.permute(0, 4, 1, 2, 3),
        samples_scaled.unsqueeze(1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(2).permute(0, 2, 3, 1)
    
    return sampled_values
