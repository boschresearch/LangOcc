# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer
from mmdet3d.models import builder
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.runner.base_module import BaseModule

class SimpleBasicBlock(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super(SimpleBasicBlock, self).__init__()
        self.conv = ConvModule(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        
        if channels_in != channels_out:
             self.skip_conv = ConvModule(
                  channels_in, channels_out,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias=False,
                  conv_cfg=dict(type='Conv3d'),
                  act_cfg=None
             )
        else:
             self.skip_conv = None

    def forward(self, x):
          skip = self.skip_conv(x) if self.skip_conv is not None else x
          return self.conv(x) + skip

@FEEDFORWARD_NETWORK.register_module()
class PointDecoder(BaseModule):
     """
     Decoder that predicts values for individual points.
     """
     def __init__(self,
                init_cfg = dict(type='Xavier', layer=['Linear'], distribution='uniform', bias=0.),
                in_channels = 256,
                embed_dims = 256,
                num_hidden_layers=1,
                num_classes=1,
                ffn_drop=0,
                bias_init = None,
                act_cfg=dict(type='ReLU', inplace=True),
                final_act_cfg=None,
    ):
          super().__init__(init_cfg=init_cfg)  
          self.embed_dims = embed_dims
          self.activate = build_activation_layer(act_cfg)
          self.final_activate = build_activation_layer(final_act_cfg) if final_act_cfg is not None else None
          self.num_classes = num_classes

          layers = []
          for _ in range(num_hidden_layers):
               layers.extend(
                    [
                    nn.Linear(in_channels, embed_dims), 
                    self.activate,
                    nn.Dropout(ffn_drop)
                    ]
               )
               in_channels = embed_dims
          layers.append(nn.Linear(embed_dims, num_classes))
          self.layers = nn.Sequential(*layers)

          # initialize bias of last linear layer to represent data distribution
          if bias_init is not None:
               self.layers[-1].bias.data = torch.tensor(bias_init, dtype=torch.float32)

     def forward(self, x):
          x = self.layers(x)
          if self.final_activate is not None:
               x = self.final_activate(x)
          return x

@FEEDFORWARD_NETWORK.register_module()
class VoxelDecoder(BaseModule):
     def __init__(self,
          init_cfg = dict(type='Xavier', layer=['Conv3d'], distribution='uniform', bias=0.),
          embed_dims=256,
          in_channels=256,
          num_layers=2,
          out_layer=None
    ):
          super().__init__(init_cfg=init_cfg)  
          self.embed_dims = embed_dims
          
          layers = []
          

          for i in range(num_layers):
               layers.extend(
                    [
                    SimpleBasicBlock(embed_dims if i!=0 else in_channels, embed_dims)
                    ]
               )
          self.out_layer = nn.Linear(embed_dims, out_layer) if out_layer is not None else None


          self.layers = nn.Sequential(*layers)

     def forward(self, voxel_features):
          out = self.layers(voxel_features.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

          if self.out_layer is not None:
               out = self.out_layer(out)

          return out