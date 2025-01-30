# Copyright (c) OpenMMLab. All rights reserved.
from .aspp_head import ASPPHead
from .maskclip_head import MaskClipHead
from .maskclip_plus_head import MaskClipPlusHead
from .aspp_headv2 import ASPPHeadV2

__all__ = [
   'ASPPHead', 'MaskClipHead', 'MaskClipPlusHead', 'ASPPHeadV2'
]
