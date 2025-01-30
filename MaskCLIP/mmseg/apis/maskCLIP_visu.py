# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import mmcv 
import os
from tqdm import tqdm
import numpy as np
from mmcv.parallel import DataContainer as DC
from sklearn.decomposition import PCA
# from mmseg.utils import vocabulary
from mmdet3d.models.lang_occ_modules import vocabulary
from PIL import Image
import pickle as pkl

color_map = np.array([
    [0, 150, 245],  # 0 car blue
    [160, 32, 240],  # 1 truck purple
    [135, 60, 0],  # 2 trailer brown
    [255, 255, 0],  # 3 bus yellow
    [0, 255, 255],  # 4 construction_vehicle cyan
    [255, 192, 203],  # 5 bicycle pink
    [200, 180, 0],  # 6 motorcycle dark orange
    [255, 0, 0],  # 7 pedestrian red
    [255, 240, 150],  # 8 traffic_cone light yellow
    [255, 120, 50],  # 9 barrier orangey
    [255, 0, 255],  # 10 driveable_surface dark pink
    [139, 137, 137],  # 11 other_flat grey
    [75, 0, 75],  # 12 sidewalk dark purple
    [150, 240, 80],  # 13 terrain light green
    [230, 230, 250],  # 14 manmade white
    [0, 175, 0],  # 15 vegetation green
    [0, 0, 0] # 16 background black
], dtype=np.uint8)/ 255.

class_embeddings_path = "../data/embeddings/MaskCLIP"
all_cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

def create_MaskCLIP_visu(model, data_loader, save_dir, cams, selected_vocabulary=1, samples_per_scene=3):
    cam_names = [all_cams[cam] for cam in cams]
    model.eval()
    device = torch.device('cuda') if next(model.parameters()).is_cuda else torch.device('cpu')
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    class_embeds = torch.tensor(np.load(os.path.join(class_embeddings_path,f'class_embeddings_v{selected_vocabulary}.npz'))['arr_0'], device=device)
    voc = vocabulary[selected_vocabulary]
    class_mapping = torch.tensor(voc[1], device=device)
    mapping_tensor = torch.zeros((len(class_embeds), len(color_map)), device=device).half()
    mapping_tensor[torch.arange(len(class_embeds)), class_mapping] = 1
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i_iter, (batch_indices, data, cam) in enumerate(tqdm(zip(loader_indices, data_loader, cam_names), total=len(data_loader))):

        scene_name = data['img_metas'].data[0][0][0]['scene_name']
        sample_token = data['img_metas'].data[0][0][0]['sample_token']

        if save_dir is not None:
            save_dir_cur = os.path.join(save_dir, scene_name)
            if not os.path.exists(save_dir_cur):
                os.makedirs(save_dir_cur)
            save_path = os.path.join(save_dir_cur, f"{sample_token}_{cam}.npz")
            
        full_embeds = []
        for i, metas in enumerate(data['img_metas'].data[0][0]):
            img_c = data['img'][0][:, i]
            data_c = {'img_metas': [DC([[metas]], cpu_only=True)], 'img': [img_c]}
            if metas['cam'] != cam:
                continue
            # test_img = data_c['img'][0].data[0].permute(1, 2, 0).cpu().numpy()
            with torch.no_grad():
                result = model(return_loss=False, **data_c)
                result[0] = result[0] / np.linalg.norm(result[0], axis=0, keepdims=True)

            full_embeds.append(result[0])
        
        full_embeds = torch.tensor(full_embeds[0], device=device).permute(1, 2, 0).half()
        # Use vocabulary to create masks
        sims = (full_embeds @ class_embeds.T)
        # mapped_sims = (sims @ mapping_tensor)
        # labels = mapped_sims.argmax(-1)
        labels = class_mapping[sims.argmax(-1)]
        label_img = color_map[labels.cpu().numpy()]

        # Fit a PCA to the embeds
        embeds_flat = full_embeds.cpu().numpy().reshape(-1, 512)
        pca = PCA(n_components=3).fit(embeds_flat)
        transformed = pca.transform(embeds_flat)
        comp_min, comp_max = transformed.min(axis=0), transformed.max(axis=0)
        transformed_norm = (transformed - comp_min) / (comp_max - comp_min)
        pca_img = transformed_norm.reshape(900, 1600, 3)

        # load image
        sample = dataset.nusc.get('sample', sample_token)
        img_path = os.path.join(dataset.nusc.dataroot, dataset.nusc.get('sample_data', sample['data'][cam])['filename'])
        img = np.array(Image.open(img_path)) / 255.
        # create overlay images
        overlay_labels = (img * .5 + label_img * .5)
        overlay_pca = (img * .5 + pca_img * .5)
        # Save the images
        np.savez_compressed(save_path, overlay_labels=overlay_labels, overlay_pca=overlay_pca)
        # np.savez_compressed(save_path, pca=pca_img, labels=label_img, overlay_labels=overlay_labels, overlay_pca=overlay_pca)

        # Also save the pca model
        with open(os.path.join(save_dir_cur, f"{sample_token}_{cam}_pca.pkl"), 'wb') as f:
            pkl.dump(pca, f)

        torch.cuda.empty_cache()

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()