# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import torch
import torch.nn as nn
from mmdet3d.models.losses import CosSimLoss

class ClipReducer(nn.Module):
    def __init__(self, embed_size, target_size):
        super().__init__()
        self.U = nn.parameter.Parameter(
            torch.zeros((embed_size, target_size)),
        )
        nn.init.normal_(self.U, std=0.1)
        self.embed_size = embed_size
        self.target_size = target_size
        self.loss_fn = ReduceLoss()

    def forward(self, embed):
        # embed of size [N, embed_size] has to be in unit length already
        bottleneck = self.reduce(embed)
        recon = self.reconstruct(bottleneck)
        return bottleneck, recon
    
    def reduce(self, embed):
        bottleneck = torch.matmul(embed, self.U)
        bottleneck = bottleneck / bottleneck.norm(dim=-1, keepdim=True)
        return bottleneck
    
    def reconstruct(self, bottleneck):
        recon = torch.matmul(bottleneck, self.U.T)
        recon = recon / recon.norm(dim=-1, keepdim=True)
        return recon
    
    def loss(self, recon, embed):
        return self.loss_fn(recon, embed)

class ClipReducerAE(nn.Module):
    def __init__(self, embed_size=512, target_size=128, loss_fn='MSE'):
        super().__init__()
        self.encoder = nn.Sequential(
           nn.Linear(embed_size, embed_size),
           nn.LeakyReLU(0.1),
           nn.Linear(embed_size, target_size*3),
           nn.LeakyReLU(0.1),
           nn.Linear(target_size*3, target_size*2),
           nn.LeakyReLU(0.1),
           nn.Linear(target_size*2, target_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(target_size, target_size*2),
            nn.LeakyReLU(0.1),
            nn.Linear(target_size*2, target_size*3),
            nn.LeakyReLU(0.1),
            nn.Linear(target_size*3, embed_size),
            nn.LeakyReLU(0.1),
            nn.Linear(embed_size, embed_size)
        )

        self.loss_fn = CosSimLoss() if loss_fn=='Cos' else nn.MSELoss()

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)
        

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight.data, std=0.1)
            nn.init.normal_(m.bias.data, std=0.1)

    def forward(self, embed):
        bottleneck = self.reduce(embed)
        recon = self.reconstruct(bottleneck)
        return bottleneck, recon

    def reduce(self, embed):
        bottleneck = self.encoder(embed)
        bottleneck = bottleneck / bottleneck.norm(dim=-1, keepdim=True)
        return bottleneck
    
    def reconstruct(self, bottleneck):
        recon = self.decoder(bottleneck)
        recon = recon / recon.norm(dim=-1, keepdim=True)
        return recon
    
    def loss(self, recon, embed):
        return self.loss_fn(recon, embed)

class KL_loss(nn.Module):
    def __init__(self, loss_weight=.1):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, mean, log_var):
        kl_loss = -0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp())
        return self.loss_weight * kl_loss

class ReduceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon, embed):
        loss = torch.arccos((embed * recon).sum(-1)).mean()
        return loss
    
class ClassSeparationLossMSE(nn.Module):
    def __init__(self, class_map):
        super().__init__()
        self.class_map = torch.tensor(class_map)
        # or use 0 as wrong class
        self.targets = torch.stack([(self.class_map == i).int()*2-1 for i in self.class_map]).float()
        self.loss_fn = nn.MSELoss()

    def forward(self, bottleneck):
        # Same class embeddings should be closer, while different class embeddings further away
        pairwise_sim = bottleneck @ bottleneck.T
        loss = self.loss_fn(pairwise_sim, self.targets)
        return loss

class ClassSeparationLoss(nn.Module):
    def __init__(self, class_map):
        super().__init__()
        self.class_map = torch.tensor(class_map)
        self.targets = torch.stack([(self.class_map == i).int()*2-1 for i in self.class_map]).float().flatten()
        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(self, bottleneck):
        bottleneck_pred = bottleneck.unsqueeze(1).repeat(1, len(bottleneck), 1).flatten(0, 1)
        bottleneck_target = bottleneck.unsqueeze(0).repeat(len(bottleneck), 1, 1).flatten(0, 1).detach()
        loss = self.loss_fn(bottleneck_pred, bottleneck_target, self.targets)
        return loss

class ClassSeparationLossMargin(nn.Module):
    def __init__(self, class_map, margin = 1.1):
        super().__init__()
        self.class_map = torch.tensor(class_map)
        self.targets = torch.stack([(self.class_map == i).int() for i in self.class_map]).float().flatten()
        self.cossim = nn.CosineSimilarity()
        self.margin = margin

    def forward(self, bottleneck):
        bottleneck_pred = bottleneck.unsqueeze(1).repeat(1, len(bottleneck), 1).flatten(0, 1)
        bottleneck_target = bottleneck.unsqueeze(0).repeat(len(bottleneck), 1, 1).flatten(0, 1).detach()

        # pairwise distance
        distance = 1 - self.cossim(bottleneck_pred, bottleneck_target)
        loss = self.targets * distance + (1 - self.targets) * torch.clamp(self.margin - distance, 0)
        
        return loss.mean()

def load_reducer(ckpt_root, reducer_cfg):
    vocabulary_version, sep_weight, separator, reducer_type, reduced_size, use_templates = reducer_cfg
    load_path = os.path.join(ckpt_root, f'reducer_{vocabulary_version}_{sep_weight}_{separator}_{reducer_type}_{reduced_size}{"_templates" if use_templates else ""}.pth')
    reducer_state_dict = torch.load(load_path)

    # instantiate selected reducer_type
    if reducer_type == 'U':
        reducer = ClipReducer(512, int(reduced_size))
    else:
        assert False, "Not implemented yet"

    reducer.eval()
    reducer.load_state_dict(reducer_state_dict)
    return reducer