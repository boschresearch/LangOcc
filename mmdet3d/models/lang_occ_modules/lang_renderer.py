# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from abc import abstractmethod
import math
from typing import List

import nerfacc
from nerfacc.pdf import importance_sampling
from nerfacc.data_specs import RayIntervals
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import HEADS, build_loss, build_head

from .samplers import Sampler, UniformSampler, PDFSampler
from .utils import interpolate_values_langocc
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss

color_map = np.array(
        [[0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier orangey
        [255, 192, 203, 255],  # bicycle pink
        [255, 255, 0, 255],  # bus yellow
        [0, 150, 245, 255],  # car blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle dark orange
        [255, 0, 0, 255],  # pedestrian red
        [255, 240, 150, 255],  # traffic_cone light yellow
        [135, 60, 0, 255],  # trailer brown
        [160, 32, 240, 255],  # truck purple
        [255, 0, 255, 255],  # driveable_surface dark pink
        [139, 137, 137, 255],  # other_flat dark red
        [75, 0, 75, 255],  # sidewalk dark purple
        [150, 240, 80, 255],  # terrain light green
        [230, 230, 250, 255],  # manmade white
        [0, 175, 0, 255],  # vegetation green
        [0, 255, 127, 255],  # ego car dark cyan
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ])

class RenderModule(nn.Module):
    def __init__(self, output_size, name, loss=None, use_uncertainty=False, **kwargs):
        super().__init__()
        self.output_size = output_size
        self.name = name

        self.loss = build_loss(loss)

    @abstractmethod
    def get_value(self, samples, samples_start, samples_end, values, ray_indices, pc_range):
        """Get the value to integrate over"""
        
    @abstractmethod
    def get_value_oob(self, samples, samples_centers, values, ray_indices, oob_mask, pc_range):
        """Get the value to integrate over, with oob masking"""
    
    @abstractmethod
    def get_loss(self, pred, gt, ray_weights=None):
        """Compute the pixel-wise loss"""

    @abstractmethod
    def background_model(self, result, renderer):
        """Apply background model"""

    @abstractmethod
    def get_output(self, result, renderer):
        """Convert output to a plottable image"""

@HEADS.register_module()
class LanguageRenderModule(RenderModule):
    def __init__(self, loss_cfg= dict(type='CosSimLoss', loss_weight=1.0), num_classes=17, *args, **kwargs):
        super().__init__(1, 'language', *args, loss=loss_cfg, **kwargs)
        self.num_classes = num_classes
        self.color_map = color_map
        
    def get_loss(self, pred, gt, ray_weights=None):
        if type(self.loss) == CrossEntropyLoss:
            loss = self.loss(pred.view(-1, 17), gt.view(-1).long())
        else:
            loss = self.loss(pred, gt, ray_weights)
        return loss
    
    def get_value(self, samples, samples_start, samples_end, values, ray_indices, pc_range):
        # Value to integrate over = predicted class logits from semantic head
        # -> Interpolate features at sample positions
        sampled_features = interpolate_values_langocc(values[1], samples, ray_indices, pc_range, None)
        return sampled_features
    
    def get_output(self, result, renderer):
        return result
    
@HEADS.register_module()
class SH_RGBRenderModule(RenderModule):
    def __init__(self, degree=3, loss_cfg= dict(type='L1Loss', loss_weight=1.0), *args, **kwargs):
        super().__init__(1, 'RGB', *args, loss=loss_cfg, **kwargs)
        self.levels = degree 
        self.n_coeffs = self.levels ** 2
        
    def get_loss(self, pred, gt, ray_weights=None):
        loss = self.loss(pred, gt, ray_weights)
        return loss
    
    def components_from_spherical_harmonics(self, directions):
        """
        Returns value for each component of spherical harmonics.

        Args:
            levels: Number of spherical harmonic levels to compute.
            directions: Spherical harmonic coefficients
        """
        components = torch.zeros((*directions.shape[:-1], self.n_coeffs), device=directions.device)

        assert 1 <= self.levels <= 5, f"SH levels must be in [1,4], got {self.levels}"
        assert directions.shape[-1] == 3, f"Direction input should have three dimensions. Got {directions.shape[-1]}"

        x = directions[..., 0]
        y = directions[..., 1]
        z = directions[..., 2]

        xx = x**2
        yy = y**2
        zz = z**2

        # l0
        components[..., 0] = 0.28209479177387814

        # l1
        if self.levels > 1:
            components[..., 1] = 0.4886025119029199 * y
            components[..., 2] = 0.4886025119029199 * z
            components[..., 3] = 0.4886025119029199 * x

        # l2
        if self.levels > 2:
            components[..., 4] = 1.0925484305920792 * x * y
            components[..., 5] = 1.0925484305920792 * y * z
            components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
            components[..., 7] = 1.0925484305920792 * x * z
            components[..., 8] = 0.5462742152960396 * (xx - yy)

        # l3
        if self.levels > 3:
            components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
            components[..., 10] = 2.890611442640554 * x * y * z
            components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
            components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
            components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
            components[..., 14] = 1.445305721320277 * z * (xx - yy)
            components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

        # l4
        if self.levels > 4:
            components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
            components[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
            components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
            components[..., 19] = 0.6690465435572892 * y * z * (7 * zz - 3)
            components[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
            components[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
            components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
            components[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
            components[..., 24] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) 
        return torch.cat((components.unsqueeze(-2), components.unsqueeze(-2), components.unsqueeze(-2)), dim=-2)
    
    def get_value(self, samples, samples_start, samples_end, values, ray_indices, pc_range):
        # Sample SH values from grid
        sampled_features = interpolate_values_langocc(values[2], samples, ray_indices, pc_range, None)
        return sampled_features
    
    def get_output(self, result, renderer):
        return result
    
@HEADS.register_module()
class RGBRenderModule(RenderModule):
    def __init__(self,  loss_cfg= dict(type='L1Loss', loss_weight=1.0), *args, **kwargs):
        super().__init__(1, 'RGB', *args, loss=loss_cfg, **kwargs)
        
    def get_loss(self, pred, gt, ray_weights=None):
        loss = self.loss(pred, gt, ray_weights)
        return loss
    
    def get_value(self, samples, samples_start, samples_end, values, ray_indices, pc_range):
        # Sample RGB values from grid
        sampled_features = interpolate_values_langocc(values[2], samples, ray_indices, pc_range, None)
        return sampled_features
    
    def get_output(self, result, renderer):
        return result

@HEADS.register_module()
class LangRenderer(nn.Module):
    def __init__(self,
                coarse_sampler: Sampler = None,
                fine_sampler: Sampler = None,
                render_modules: List[dict] = None,
                render_range = [0.05, 40],
                grid_cfg=None,
                pc_range=None,
                prop_samples_per_ray=50,
                samples_per_ray=50,
                use_proposal=True,
                render_frame_ids=[0],
                dist_loss=False,
                dynamic_weight=0):
        super().__init__()

        # Create render modules
        assert render_modules is not None, "Please provide at least one render module."
        self.render_modules = nn.ModuleList()
        for render_module in render_modules:
            self.render_modules.append(build_head(render_module))

        self.near = render_range[0]
        self.far = render_range[1]
        self.coarse_sampler = coarse_sampler if coarse_sampler is not None else UniformSampler()
        self.fine_sampler = fine_sampler if fine_sampler is not None else PDFSampler()
        self.samples_per_ray = samples_per_ray
        self.prop_samples_per_ray = prop_samples_per_ray
        self.use_proposal = use_proposal
        self.zero_weights = None
        self.dynamic_weight = dynamic_weight

        assert len(render_frame_ids)>0, "Please provide at least a single render frame"
        self.current_timestep = render_frame_ids.index(0)
        self.render_frame_ids = render_frame_ids
        self.interpolate_fn = interpolate_values_langocc

        # Grid config
        self.pc_range = pc_range
        self.dist_loss = build_loss(dist_loss) if dist_loss else None

        # evaluation scenes
        self.eval_scenes = []

    def generate_samples(self, origins, directions, cam_idxs, density):
        if self.use_proposal:
            # Initial uniform sampling
            init_samples_start, init_samples_end, init_samples_centers = self.coarse_sampler(origins, directions, self.prop_samples_per_ray, self.near, self.far)
            samples = self.create_3d_samples(origins, directions, init_samples_centers)
            init_density = self.interpolate_fn(density, samples, cam_idxs, self.pc_range, self.current_timestep)
            _, init_transmittance, _ = nerfacc.render_weight_from_density(init_samples_start, init_samples_end, init_density.squeeze(-1))
            
            # pdf sampler using nerfacc functions
            cdfs = 1.0 - torch.cat([init_transmittance, torch.zeros_like(init_transmittance[..., :1])], dim=-1)
            intervals = RayIntervals(vals=torch.cat([init_samples_start, init_samples_end[..., -1:]], dim=-1))
            sampled_intervals, sampled_samples = importance_sampling(intervals, cdfs, self.samples_per_ray)
            samples_start, samples_end, samples_centers = sampled_intervals.vals[..., :-1], sampled_intervals.vals[..., 1:], sampled_samples.vals #TODO: THIS SOMEHOW GIVES MORE VALS??
        else:
            # only use coarse sampling (e.g., uniform)
            samples_start, samples_end, samples_centers = self.coarse_sampler(origins, directions, self.samples_per_ray, self.near, self.far)

        return samples_start, samples_end, samples_centers 
    
    def create_3d_samples(self, origins, directions, sample_centers):
        samples = origins[..., None, :] + sample_centers[..., None] * directions[..., None, :]
        return samples

    def forward(self, voxel_outs, ray_dataset, bda, return_intermediate=False, **kwargs):
        if not isinstance(self.pc_range, torch.Tensor):
            self.pc_range = voxel_outs[0].new_tensor(self.pc_range)

        # Disentangle ray dataset
        cam_idxs = ray_dataset[..., :2].int()
        origins = ray_dataset[..., 2:5]
        directions = ray_dataset[..., 5:8]

        samples_start, samples_end, samples_centers = self.generate_samples(origins, directions, cam_idxs, voxel_outs[0])
        samples = self.create_3d_samples(origins, directions, samples_centers)
        densities = self.interpolate_fn(voxel_outs[0], samples, cam_idxs, self.pc_range, self.current_timestep).squeeze(-1)
        weights, _, _ = nerfacc.render_weight_from_density(samples_start, samples_end, densities)
        self.zero_weights = (weights.sum(-1) == 0).nonzero() # rays with 0 weight -> Will produce high gradient, thus mask out

        results = {}
        for module in self.render_modules:
            values = module.get_value(samples, samples_start, samples_end, voxel_outs, cam_idxs, self.pc_range)
            if type(module) == SH_RGBRenderModule:
                # combine sh values to rgb
                sh_components = module.components_from_spherical_harmonics(directions)
                values = (sh_components.unsqueeze(2) * values.view(*values.shape[:3], 3, -1)).sum(-1).clip(0, 1)
            result = nerfacc.accumulate_along_rays(weights, values)
            if module.name == "language":
                # scale clip embeds to unit vector
                result = F.normalize(result, dim=-1) # this will automatically prevent division by zero
            results[f'{module.name}'] = result

        if self.dist_loss is not None:
            results['loss_dist']  = self.dist_loss(weights, samples_centers, samples_end-samples_start)

        if return_intermediate:
            return results, densities.cpu(), samples.cpu()
        else:
            return results

    def forward_test(self, voxel_outs, ray_dataset, return_intermediate=False, ray_batch_size=20000, **kwargs):
        with torch.no_grad():
            if not isinstance(self.pc_range, torch.Tensor):
                self.pc_range = voxel_outs[0].new_tensor(self.pc_range)

            results = {k:[] for k in [m.name for m in self.render_modules]}
            if 'max_depths' in kwargs and kwargs['max_depths']:
                results['max_depth'] = []
            if 'clipped_depths' in kwargs:
                results['clipped_depth'] = []
            n_batches = ray_dataset.shape[1] // ray_batch_size + 1

            for i in range(n_batches):
                cam_idxs = ray_dataset[..., ray_batch_size*i:(i+1)*ray_batch_size, :2].int()
                origins = ray_dataset[..., ray_batch_size*i:(i+1)*ray_batch_size, 2:5]
                directions = ray_dataset[..., ray_batch_size*i:(i+1)*ray_batch_size, 5:8]

                samples_start, samples_end, samples_centers = self.generate_samples(origins, directions, cam_idxs, voxel_outs[0])
                samples = self.create_3d_samples(origins, directions, samples_centers)
                densities = self.interpolate_fn(voxel_outs[0], samples, cam_idxs, self.pc_range, self.current_timestep).squeeze(-1)
                weights, _, _ = nerfacc.render_weight_from_density(samples_start, samples_end, densities)

                for module in self.render_modules:
                    values = module.get_value(samples, samples_start, samples_end, voxel_outs, cam_idxs, self.pc_range)
                    if type(module) == SH_RGBRenderModule:
                        # combine sh values to rgb
                        sh_components = module.components_from_spherical_harmonics(directions)
                        values = (sh_components.unsqueeze(2) * values.view(*values.shape[:3], 3, -1)).sum(-1).clip(0, 1)
                    result = nerfacc.accumulate_along_rays(weights, values)
                    if module.name == "language":
                        # scale clip embeds to unit vector
                        result = F.normalize(result, dim=-1) # this will automatically prevent division by zero
                    results[f'{module.name}'].append(result.cpu())

                    if module.name == 'depth' and 'max_depths' in kwargs and kwargs['max_depths']:
                        max_density_along_rays = torch.argmax(densities, dim=-1)
                        results['max_depth'].append(torch.gather(samples_centers, 2, max_density_along_rays.unsqueeze(-1)))
                    if module.name == 'depth' and 'clipped_depths' in kwargs:
                        clipped_densities = densities.clone()
                        clipped_densities[clipped_densities < kwargs['clipped_depths']] = 0.
                        weights_clipped, _, _ = nerfacc.render_weight_from_density(samples_start, samples_end, clipped_densities)
                        results['clipped_depth'].append(nerfacc.accumulate_along_rays(weights_clipped, values))
            
            for k in results.keys():
                results[k] = torch.cat(results[k], dim=1).cpu().numpy()

            if return_intermediate:
                return results, densities.cpu(), samples.cpu()
            else:
                return results
    
    def calculate_losses(self, pred, ray_dataset, static_mask=None):

        depths = ray_dataset[..., -4:-3]
        rgb = ray_dataset[..., -3:]
        embeddings = ray_dataset[..., 8:-4] 

        gt = {'depth': depths, 'language': embeddings, 'RGB':rgb}
        ray_weights = torch.ones_like(ray_dataset[..., 0])
        ray_weights[tuple(self.zero_weights.T)] = 0

        if static_mask is not None:
            # Increase weight for all dynamic rays
            ray_weights = ray_weights + (~static_mask * self.dynamic_weight)

        losses_dict = {}
        for module in self.render_modules:
            losses_dict[f'loss_render_{module.name}'] = module.get_loss(pred[module.name], gt[module.name], ray_weights.unsqueeze(-1))

        if self.dist_loss is not None:
            losses_dict['loss_dist'] = pred['loss_dist']
        return losses_dict
    
    def get_outputs(self, results):
        out = {}
        for module in self.render_modules:
            out_t = module.get_output(results[module.name].detach(), self)
            out[module.name] = out_t
        return out