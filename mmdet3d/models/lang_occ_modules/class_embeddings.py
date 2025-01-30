# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import os
import clip
import numpy as np
from .vocabulary import vocabulary, templates

def create_class_embeddings(settings, model, use_templates=False, device=torch.device('cuda')):
    print("Starting to create vocabulary embeddings...")
    # load vocabulary
    vocab = vocabulary[settings.vocabulary]
    class_text = vocab[2]

    # compute class embeddings
    if not use_templates:
        class_tokens = torch.cat([clip.tokenize(c) for c in class_text]).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            class_embeddings = model.encode_text(class_tokens)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

    else:
        all_class_embeds = []
        for i, c in enumerate(class_text):
            texts = [template.format(c) for template in templates]  # format with class
            class_tokens = torch.cat([clip.tokenize(c) for c in texts]).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                class_embeddings = model.encode_text(class_tokens)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                all_class_embeds.append(class_embedding)
        class_embeddings = torch.stack(all_class_embeds)

    # store class embeddings
    filename = os.path.join(settings.save_path, f'MaskCLIP', f'class_embeddings_v{settings.vocabulary}{"_templates" if use_templates else ""}.npz')
    np.savez_compressed(filename, class_embeddings.cpu().numpy())
    print("Done!")