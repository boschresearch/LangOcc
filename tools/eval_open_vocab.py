# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import argparse
import clip
import numpy as np
from tqdm import tqdm
import os
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.models.lang_occ_modules import load_reducer, templates
from mmcv.runner import load_checkpoint
import torch
import torch.nn.functional as F
import mmcv
from pyquaternion import Quaternion
from sklearn.metrics import average_precision_score
from matplotlib import cm
try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
import warnings

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

def load_points(pts_filename, file_client):
    try:
        pts_bytes = file_client.get(pts_filename)
        points = np.frombuffer(pts_bytes, dtype=np.float32)
    except ConnectionError:
        mmcv.check_file_exist(pts_filename)
        if pts_filename.endswith('.npy'):
            points = np.load(pts_filename)
        else:
            points = np.fromfile(pts_filename, dtype=np.float32)

    return points

def create_pc_grid(grid_cfg):

    Z = int((grid_cfg['Z_BOUND'][1] - grid_cfg['Z_BOUND'][0]) / grid_cfg['Z_BOUND'][2])
    H = int((grid_cfg['X_BOUND'][1] - grid_cfg['X_BOUND'][0]) / grid_cfg['X_BOUND'][2])
    W = int((grid_cfg['Y_BOUND'][1] - grid_cfg['Y_BOUND'][0]) / grid_cfg['Y_BOUND'][2])

    xs = torch.linspace(0.5 * grid_cfg['X_BOUND'][2] + grid_cfg['X_BOUND'][0], grid_cfg['X_BOUND'][1] - 0.5 * grid_cfg['X_BOUND'][2], W).view(W, 1, 1).expand(W, H, Z)
    ys = torch.linspace(0.5 * grid_cfg['Y_BOUND'][2] + grid_cfg['Y_BOUND'][0], grid_cfg['Y_BOUND'][1] - 0.5 * grid_cfg['Y_BOUND'][2], H).view(1, H, 1).expand(W, H, Z)
    zs = torch.linspace(0.5 * grid_cfg['Z_BOUND'][2] + grid_cfg['Z_BOUND'][0], grid_cfg['Z_BOUND'][1] - 0.5 * grid_cfg['Z_BOUND'][2], Z).view(1, 1, Z).expand(W, H, Z)
    
    ref_3d = torch.stack((xs, ys, zs), -1)
    ref_3d = ref_3d.flatten(0, 2)
    return ref_3d

pc_range = torch.tensor([-40., -40., -1.0, 40., 40., 5.4])
cfg_root = 'configs/lang_occ'
colormap = cm.get_cmap('magma')

if __name__=='__main__':
    # create argument parser
    parser = argparse.ArgumentParser(description='Open Vocabulary Benchmark')
    parser.add_argument('--cfg', type=str, help='Path to config file', required=True)
    parser.add_argument('--vocabulary', type=int, help='Path to data directory', default=1)
    parser.add_argument('--fs-index', type=int, help='Which threshold to use', default=0)
    parser.add_argument('--ckpt', type=str, help='Which ckpt to use', default='epoch_18_ema')
    parser.add_argument('--ckpt-root', type=str, help='Which ckpt to use', default='./work_dirs')
    parser.add_argument('--use-templates', action='store_true', help='Use prompt templates for class embeddings')
    args = parser.parse_args()
    
    torch.set_warn_always(False)
    warnings.filterwarnings("ignore")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg_path = os.path.join(cfg_root, f'{args.cfg}.py')
    cfg = Config.fromfile(cfg_path)
    cfg = compat_cfg(cfg)

    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    file_client_args=dict(backend='disk')
    file_client = mmcv.FileClient(**file_client_args)
    # load model
    print("Loading model...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    ckpt_path = os.path.join(args.ckpt_root, args.cfg, f'{args.ckpt}.pth')
    checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')
    model = model.to(device)
    model.eval()
    print("Done.")

    # Load targets and queries
    retrieval_path = 'data/retrieval_benchmark'
    retrievals = os.path.join(retrieval_path, 'retrieval_anns_all.csv')
    cfg.data.test['ann_file'] = 'data/bevdetv2-nuscenes_infos_val.pkl'
    dataset = build_dataset(cfg.data.test)
    pipeline = dataset.pipeline

    # Generate query embeddings
    print("Generate Query Embeddings")
    lang_encoder, _ = clip.load('ViT-B/16', device)

    # Load targets and queries
    tokens = []
    text_queries = []
    sample_splits = []
    for line in open(retrievals, 'r').readlines():
        line = line.strip()
        token, split, ann_path, pts_path, query = line.split(';')
        tokens.append(token)
        text_queries.append(query)
        sample_splits.append(split)

    # compute class embeddings
    if not args.use_templates:
        class_tokens = torch.cat([clip.tokenize(c) for c in text_queries]).to(device)
        with torch.no_grad():
            query_embeddings = lang_encoder.encode_text(class_tokens).float()
            query_embeddings /= query_embeddings.norm(dim=-1, keepdim=True)
    else:
        all_query_embeds = []
        for i, c in enumerate(text_queries):
            texts = [template.format(c) for template in templates]  # format with class
            class_tokens = torch.cat([clip.tokenize(c) for c in texts]).to(device)
            with torch.no_grad():
                query_embeddings_templates = lang_encoder.encode_text(class_tokens)
                query_embeddings_templates /= query_embeddings_templates.norm(dim=-1, keepdim=True)
                query_embedding = query_embeddings_templates.mean(dim=0)
                query_embedding /= query_embedding.norm()
                all_query_embeds.append(query_embedding)
        query_embeddings = torch.stack(all_query_embeds).float()

    if model.reducer is not None:
        query_embeddings = model.reducer.reduce(query_embeddings.float())

    print("Done.")

    print("Loading annotation files")
    annos_train = mmcv.load('data/bevdetv2-nuscenes_infos_train.pkl', file_format='pkl')['infos']
    annos_val = mmcv.load('data/bevdetv2-nuscenes_infos_val.pkl', file_format='pkl')['infos']
    annos_test = mmcv.load('data/bevdetv2-nuscenes_infos_test.pkl', file_format='pkl')['infos']
    print("Done.")

    grid_cfg_formatted = {"X_BOUND": [pc_range[0], pc_range[3], .4],
    "Y_BOUND": [pc_range[1], pc_range[4], .4],
    "Z_BOUND": [pc_range[2], pc_range[5], .4],
    "BEV_DOWNSCALE": 1}
    grid = create_pc_grid(grid_cfg_formatted)
    dummy_time = 0.0
    all_mAPs = []
    all_mAPs_visible = []

    mAPs_train = []
    mAPs_test = []
    mAPs_val = []
    mAPs_valtest = []
    mAPs_train_visible = []
    mAPs_val_visible = []
    mAPs_valtest_visible = []
    mAPs_test_visible = []

    token_idx = 0
    for token, query, split in tqdm(zip(tokens, text_queries, sample_splits), total=len(tokens)):
        if split == 'test':
            annos = annos_test
        elif split == 'val':
            annos = annos_val
        elif split == 'train':
            annos = annos_train
        else:
            assert False, "Wrong split!"

        index, info = [(i, a) for i, a in enumerate(annos) if a['token'] == token][0]
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            lidar_idx=info['lidar_idx'],
            panoptic_filename=info['panoptic_path'],
            scene_name=info['scene_name'],
            ann_infos = info['ann_infos'] if 'ann_infos' in info else None
        )
        input_dict.update(dict(curr=info))
        info_adj_list = []
        adj_id_list = list(range(*cfg.multi_adj_frame_id_cfg))
        adj_id_list.append(cfg.multi_adj_frame_id_cfg[1])
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)
            if not annos[select_id]['scene_name'] == info[
                    'scene_name']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(annos[select_id])
        input_dict.update(dict(adjacent=info_adj_list))

        results = pipeline(input_dict)
        # transformations = calculate_transformations(results)
        results = {
            'img_metas' : [[results['img_metas'][0].data]],
            'points' : [[results['points'][0].data]],
            'img_inputs' : [[inp.unsqueeze(0).to(device) for inp in results['img_inputs'][0]]],
        }

        # Forward model
        with torch.no_grad():
            out = model(return_loss=False, rescale=True, **results, return_embeds=True, return_classes=False)
            occupancy, free_space, embeds = out[0]['occupancy'], out[0]['free_space'], torch.tensor(out[0]['embeddings'], device=device)

        # Load points and targets
        annotations = np.load(os.path.join(retrieval_path, 'annotations', f'{token}__retrieval.npy')) 
        matched_points = np.load(os.path.join(retrieval_path, 'matching_points', f'{token}__points.npy'))
        points = results['points'][0][0]
        
        # Transform to ego
        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = Quaternion(input_dict['curr']['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = np.array(input_dict['curr']['lidar2ego_translation'])
        lidar2ego = torch.tensor(lidar2ego)
        points_in_ego = (lidar2ego[:3, :3] @ points[:, :3, None]).squeeze(-1) + lidar2ego[:3, 3]
        
        # Filter out OOB points (also from GT)
        oob_mask = (points_in_ego[:, 0] > pc_range[0]) & (points_in_ego[:, 0] < pc_range[3]) & (
                            points_in_ego[:, 1] > pc_range[1]) & (points_in_ego[:, 1] < pc_range[4]) & (
                            points_in_ego[:, 2] > pc_range[2]) & (points_in_ego[:, 2] < pc_range[5])
        matched_mask = torch.zeros(len(points), dtype=torch.bool)
        matched_mask[matched_points] = True

        # Trilinear interpolation
        sample_points = ((points_in_ego - pc_range[:3]) / (pc_range[3:] - pc_range[:3]) * 2 - 1).to(device)
        sampled_embeds = F.grid_sample(
            embeds.permute(3, 0, 1, 2).unsqueeze(0),
            # sample_points[None, :, None, None, :],
            sample_points[None, :, None, None, [2,1,0]],
            align_corners=False
        ).squeeze(-1).squeeze(-1).squeeze(0).permute(1, 0)
        sampled_embeds = sampled_embeds / sampled_embeds.norm(dim=1, keepdim=True)

        # Load query embedding
        query_embedding = torch.tensor(query_embeddings[token_idx].reshape(1, -1), device=device)

        # compute similarity
        similarity = (sampled_embeds @ query_embedding.float().T).cpu().numpy()
        probs = (similarity + 1 ) / 2
        
        # compute AP / PrecisionRecall
        annotations = torch.tensor(annotations).squeeze()
        matched_oob = torch.logical_and(matched_mask, oob_mask)

        visible_annotations = annotations[matched_oob]
        visible_probs = probs.squeeze()[matched_oob]
        mAP = average_precision_score(annotations[oob_mask].squeeze().cpu().numpy(), probs.squeeze()[oob_mask])
        mAP_visible = average_precision_score(visible_annotations.squeeze().cpu().numpy(),visible_probs.squeeze())

        all_mAPs.append(mAP)
        all_mAPs_visible.append(mAP_visible)
        if split == 'train':
            mAPs_train.append(mAP)
            mAPs_train_visible.append(mAP_visible)
        elif split == 'val':
            mAPs_val.append(mAP)
            mAPs_val_visible.append(mAP_visible)
            mAPs_valtest.append(mAP)
            mAPs_valtest_visible.append(mAP_visible)
        elif split == 'test':
            mAPs_test.append(mAP)
            mAPs_test_visible.append(mAP_visible)
            mAPs_valtest.append(mAP)
            mAPs_valtest_visible.append(mAP_visible)

        token_idx += 1

        
    print(f"mAP: {np.mean(all_mAPs)}, mAP visible: {np.mean(all_mAPs_visible)}")
    print(f"mAP_train: {np.mean(mAPs_train)}, mAP_train visible: {np.mean(mAPs_train_visible)}")
    print(f"mAP_val: {np.mean(mAPs_val)}, mAP_val visible: {np.mean(mAPs_val_visible)}")
    print(f"mAP_test: {np.mean(mAPs_test)}, mAP_test visible: {np.mean(mAPs_test_visible)}")
    print(f"mAP_valtest: {np.mean(mAPs_valtest)}, mAP_valtest visible: {np.mean(mAPs_valtest_visible)}")