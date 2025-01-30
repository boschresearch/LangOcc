# Copyright (c) Phigent Robotics. All rights reserved.

# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from .bevdet import BEVStereo4D
import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_loss, build_head
from mmcv.cnn.bricks.conv_module import ConvModule
import os
import numpy as np
from mmcv.cnn.bricks.transformer import (build_feedforward_network)
from mmdet3d.models.lang_occ_modules import vocabulary, class_to_nusc_v1_map, nusc_v1_to_class_map, nusc_v2_to_class_map, class_weights, load_reducer
from mmdet3d.models.lang_occ_modules import ClipReducer
from mmdet3d.models.losses import OccSparsityLoss

@DETECTORS.register_module()
class LangOcc(BEVStereo4D):

    def __init__(self,
                 out_dim=64,
                 use_mask=False,
                 num_classes=18, 
                 density_decoder=None,
                 language_decoder=None,
                 rgb_decoder=None,
                 renderer=None,
                 tv_loss=False,
                 loss_occ_density=None,
                 loss_occ_language=None,
                 eval_threshold_range=[.05, .2, .5],
                 class_embeddings_path=None,
                 vocabulary_version=[1],
                 class_weights_version=0,
                 reducer_cfg=None,
                 semantic_rays=False,
                 ckpt_root='./ckpts',
                 average_embed_preds=False,
                 lss_depth_loss=False,
                 sparsity_loss_weight=0,
                 use_templates=False,
                 **kwargs):
        super(LangOcc, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ_density = build_loss(loss_occ_density) if loss_occ_density is not None else None
        self.loss_occ_language = build_loss(loss_occ_language) if loss_occ_language is not None else None
        self.align_after_view_transfromation = False
        self.eval_threshold_range = eval_threshold_range
        self.class_embeddings = None
        self.class_embeddings_path = class_embeddings_path
        self.vocabulary_version = vocabulary_version
        self.class_weights_version = class_weights_version
        self.nusc_to_class_map = nusc_v1_to_class_map
        self.use_templates = use_templates

        self.semantic_rays = semantic_rays
        self.lss_depth_loss = lss_depth_loss

        self.average_embed_preds = average_embed_preds 

        if reducer_cfg is not None:
            self.reducer = load_reducer(ckpt_root, reducer_cfg)
            self.reducer.U.requires_grad = False
        else:
            self.reducer = None

        self.density_decoder = build_feedforward_network(density_decoder) if density_decoder is not None else None
        self.language_decoder = build_feedforward_network(language_decoder) if language_decoder is not None else None
        self.rgb_decoder = build_feedforward_network(rgb_decoder) if rgb_decoder is not None else None
        self.renderer = build_head(renderer) if renderer is not None else None

        # additional losses
        self.loss_tv =  build_loss(dict(type='TVLoss3D')) if tv_loss else None # TODO adapt to language features
        self.loss_sparsity = OccSparsityLoss(loss_weight=sparsity_loss_weight) if sparsity_loss_weight > 0 else None

    def load_class_embeddings(self, device):
        if self.class_embeddings is not None:
            pass
        else:
            self.class_embeddings = []
            self.mapping_tensor = []
            self.class_mappings = []

            # Load all chosen vocabularies
            for version in self.vocabulary_version:
                embed_filename = f'class_embeddings_v{version}{"_templates" if self.use_templates else ""}.npz'

                class_embedding = torch.tensor(np.load(os.path.join(self.class_embeddings_path,
                                            embed_filename))['arr_0'], device=device).float()
                if self.reducer is not None:
                    with torch.no_grad():
                        class_embedding = self.reducer.reduce(class_embedding)

                voc_map = torch.tensor(vocabulary[version][1], device=device)
                mapping_tensor = torch.zeros((len(class_embedding), len(self.nusc_to_class_map)), device=device)
                mapping_tensor[torch.arange(len(class_embedding)), voc_map] = 1

                self.class_embeddings.append(class_embedding)
                self.mapping_tensor.append(mapping_tensor)
                self.class_mappings.append(voc_map)

    def embeds_to_classes(self, preds):
        """mapping from embeddings to classes"""
        # Load class embeddings
        self.load_class_embeddings(preds.device)
        
        if self.average_embed_preds:
            voxel_probs = [preds @ embed.T for embed in self.class_embeddings]
            mapped_probs = torch.stack([(v @ mapping) / mapping.sum(0) for v, mapping in zip(voxel_probs, self.mapping_tensor)])
            pred_classes = (mapped_probs * class_weights.to(mapped_probs.device)[:, self.class_weights_version]).argmax(-1)
        else:
            weights = [class_weights[:, self.class_weights_version][map_tensor.argmax(1)].to(preds.device) for map_tensor in self.mapping_tensor]
            pred_classes = torch.stack([class_map[((preds @ embed.T) * w).argmax(-1)] for w, class_map, embed in zip(weights, self.class_mappings, self.class_embeddings)])

        # map back from vocabulary to target class space
        pred_classes = self.nusc_to_class_map[pred_classes]
        return pred_classes

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    origins=None,
                    directions=None,
                    lang_ray_dataset=None,
                    return_embeds=False,
                    return_classes=True,
                    render_preds=False,
                    clip_low_density_regions=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        
        out_dict = {}

        density_pred = self.density_decoder(voxel_feats)
        if self.language_decoder is not None:
            embeddings_pred = self.language_decoder(voxel_feats) 
            embeddings_pred = embeddings_pred / embeddings_pred.norm(dim=-1, keepdim=True) if not self.semantic_rays else embeddings_pred
            if return_classes:
                semantics_pred = self.embeds_to_classes(embeddings_pred) if not self.semantic_rays else embeddings_pred.argmax(-1).unsqueeze(1)
            else:
                semantics_pred = torch.zeros([*density_pred.shape[:-1]], device=density_pred.device)
        else:
            semantics_pred = torch.zeros([*density_pred.shape[:-1]], device=density_pred.device)

        # clip low density regions
        if clip_low_density_regions:
            density_pred[density_pred<1e-1] = 0
            # remove roof predictions
            density_pred[..., 11:, :] = 0
        
        free_space = torch.stack([density_pred.squeeze() < tr for tr in self.eval_threshold_range])
        # render if specified
        if render_preds:
            voxel_outs = [density_pred.permute(0, 3, 2, 1, 4), embeddings_pred.permute(0, 3, 2, 1, 4), None]
            render_preds = self.renderer.forward_test(voxel_outs, lang_ray_dataset[0], **kwargs)
            out_dict['render'] = render_preds

        out_dict['occupancy'] = semantics_pred.squeeze(1).to(torch.uint8).cpu().numpy()
        out_dict['free_space'] =  free_space.cpu().numpy()

        # Also return predicted language embeddings
        if return_embeds:
            out_dict['embeddings'] = embeddings_pred.squeeze().cpu().numpy()

        return [out_dict]

    def forward_train(self,
                      # standard inputs
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      lang_ray_dataset=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      gt_depth=None,
                      # For weighting of dynamic rays
                      static_mask=None,
                      #kwargs
                      **kwargs):
        
        # imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda  = self.prepare_inputs(img_inputs)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()

        if self.lss_depth_loss:
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
            losses['loss_depth'] = loss_depth

        voxel_feats = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1) # bcdhw->bwhdc

        density_pred = self.density_decoder(voxel_feats)
        lang_pred = self.language_decoder(voxel_feats) if self.language_decoder is not None else torch.zeros([*density_pred.shape])
        lang_pred = F.normalize(lang_pred, dim=-1) if not self.semantic_rays and lang_pred is not None else lang_pred
        rgb_pred = self.rgb_decoder(voxel_feats) if self.rgb_decoder is not None else None

        if self.loss_sparsity is not None:
            losses['loss_sparsity'] = self.loss_sparsity(density_pred)
            
        if self.renderer is not None:
            voxel_outs = [density_pred.permute(0, 3, 2, 1, 4), lang_pred.permute(0, 3, 2, 1, 4), 
                          rgb_pred.permute(0, 3, 2, 1, 4) if rgb_pred is not None else None]

            if self.loss_tv is not None:
                losses['loss_tv'] = self.loss_tv(voxel_outs)

            render_preds = self.renderer(voxel_outs, lang_ray_dataset, img_inputs[-1])
            render_losses = self.renderer.calculate_losses(render_preds, lang_ray_dataset, static_mask)
            losses.update(render_losses)

        return losses