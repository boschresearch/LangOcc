# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
from abc import abstractmethod

class Sampler(nn.Module):
    "Abstract Sampler superclass in case some common logic is needed in the future."
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, rays, num_samples, near, far, *args, **kwargs):
        "Generate Ray Samples"

    def linear_scale_to_target_space(self, samples, near, far):
        return samples * far + (1 - samples) * near

    
class UniformSampler(Sampler):
    def __init__(self, single_jitter=True) -> None:
        super().__init__()
        self.single_jitter = single_jitter

    def forward(self, origins, directions, num_samples, near, far):
        b, num_rays, _ = directions.shape
        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(directions.device)
        
        # add random jitter to bin borders (except beginning and end)
        jitter_scale = lambda x: x * (1 / num_samples) + (- 1 / num_samples / 2)
        jitter = torch.rand((b, num_rays, num_samples - 1), dtype=bins.dtype, device=bins.device)
        jitter = torch.cat((jitter.new_zeros(b, num_rays, 1), jitter_scale(jitter), jitter.new_zeros(b, num_rays, 1) ), dim=-1)
        bins = bins + jitter

        bin_upper = bins[..., 1:]
        bin_lower = bins[..., :-1]
        bin_centers = (bin_lower + bin_upper) / 2.0

        # scale to metric space (meters)
        samples_start = self.linear_scale_to_target_space(bin_lower, near, far)
        samples_end = self.linear_scale_to_target_space(bin_upper, near, far)
        samples_center = self.linear_scale_to_target_space(bin_centers, near, far)

        return samples_start, samples_end, samples_center

class PDFSampler(Sampler):
    def __init__(self, single_jitter=True) -> None:
        super().__init__()
        self.histogram_padding = 0.01
        self.single_jitter = single_jitter

    def forward(self, origins, directions, num_samples, near, far, weights=None, existing_bins=None):
    # def forward(self, rays, num_samples, near, far, weights=None, existing_bins=None):
        assert weights is not None and existing_bins is not None
        
        # nerf studio version
        weights = weights + self.histogram_padding # add small amount to weights
        num_bins = num_samples + 1
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(1e-5 - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        # construct pdf and cdf
        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        # create uniform stratified samples 
        u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
        u = u.expand(size=(*cdf.shape[:-1], num_bins))
        if self.single_jitter:
            rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
        else:
            rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
        u = (u + rand).contiguous()

        existing_bins = torch.cat((existing_bins[0], existing_bins[1][..., -1:]), dim=-1)

        num_initial_samples = weights.shape[-1]
        inds = torch.searchsorted(cdf, u, side="right")
        below = torch.clamp(inds - 1, 0, num_initial_samples)
        above = torch.clamp(inds, 0, num_initial_samples)
        cdf_g0 = torch.gather(cdf, -1, below)
        bins_g0 = torch.gather(existing_bins, -1, below)
        cdf_g1 = torch.gather(cdf, -1, above)
        bins_g1 = torch.gather(existing_bins, -1, above)

        t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        bins = bins.detach()

        lower = bins[:, :-1]
        upper = bins[:, 1:]
        center = (upper + lower) / 2.

        return lower, upper, center