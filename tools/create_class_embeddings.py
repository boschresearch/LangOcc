# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
from argparse import ArgumentParser
import os
import clip
import numpy as np
from mmdet3d.models.lang_occ_modules import create_class_embeddings

cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
img_size = (900, 1600)
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__=="__main__":
    parser = ArgumentParser(description="Render occupancy (pred and/or gt)")

    parser.add_argument("--save-path", type=str, help="Path to save directory", default='data/embeddings')
    parser.add_argument("--use-templates", help="Use prompt templates for class embeddings", action='store_true')
    parser.add_argument("--vocabulary", type=int, help='Version of the vocabulary', default=1)
    args = parser.parse_args()

    # create target directory
    save_path = os.path.join(args.save_path, 'MaskCLIP')
    os.makedirs(save_path, exist_ok=True)

    print(f"Starting creation of class embeddings:")
    print(f'Vocabulary Version: {args.vocabulary}')
    print(f'Using Templates: {args.use_templates}')

    # init clip encoder
    print("Loading CLIP model...")
    model, _ = clip.load('ViT-B/16', device)
    print("Done.")

    # create class embeddings
    create_class_embeddings(args, model, use_templates=args.use_templates, device=device)    