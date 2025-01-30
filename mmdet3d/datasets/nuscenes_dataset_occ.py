# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import os.path as osp
import mmcv
import gc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes


from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, Metric_FScore, IoU
from mmdet3d.models.lang_occ_modules.vocabulary import class_to_nusc_v1_map

colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])


@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):

    def __init__(self, *args, eval_threshold_range=[.05, .2, .5], threshold_save_indices=None, num_classes=18,
                 gt_root='data/gts', gt_root2='data/gtsv2', eval_metrics = ['mIoU'],**kwargs):
        super().__init__(*args, **kwargs)

        self.set_eval_threshold_range(eval_threshold_range, threshold_save_indices)
        self.gt_root=gt_root
        self.gt_root_v2=gt_root2
        self.num_classes = num_classes
        self.eval_metrics = eval_metrics

    def set_eval_threshold_range(self, eval_range, indices = None):
        self.eval_threshold_range = eval_range
        if indices is None:
            self.threshold_save_indices = list(range(len(eval_range)))
        else:
            self.threshold_save_indices = indices

    def get_data_info(self, index):
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict
    
    def evaluate_mIoU(self, occ_results, eval_dict):
        n_vocabs = occ_results[0]['occupancy'].shape[0]
        metrics = [[Metric_mIoU(
        num_classes=self.num_classes,
        use_lidar_mask=False,
        use_image_mask=True, eval_tr=i) for i in self.eval_threshold_range] for _ in range(n_vocabs)]
            
        general_iou = [IoU(use_image_mask=True, eval_tr=i) for i in self.eval_threshold_range]
        
        for index, occ_pred in enumerate(occ_results):
            info = self.data_infos[index]

            occ_path = os.path.join(self.gt_root, info['scene_name'], info['token'], 'labels.npz')
            occ_gt = np.load(occ_path)
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)

            preds = occ_pred['occupancy']
            for i, t in enumerate(self.eval_threshold_range):
                preds_i = preds.copy()
                preds_i[:, occ_pred['free_space'][i]] = 17
                general_iou[i].add_batch(preds_i[0], gt_semantics, mask_lidar, mask_camera)
                for v in range(n_vocabs):
                    metrics[v][i].add_batch(preds_i[v], gt_semantics, mask_lidar, mask_camera)

        top_mIoU = 0
        top_IoU = 0
        for i, t in enumerate(self.eval_threshold_range):
            print("############################")
            print(f"Eval threshold {t}:")
            general_iou_metric = general_iou[i]
            iou = general_iou_metric.count_miou()[1][0]
            eval_dict.update({f'IoU_{t}': iou})
            if iou > top_IoU:
                top_IoU = iou

            for v in range(n_vocabs):
                print(f"\nVocabulary {v}:")
                metric = metrics[v][i]
                miou = metric.count_miou()[2]
                eval_dict.update({f'mIoU_{v}_{t}': miou})
                if miou > top_mIoU:
                    top_mIoU = miou
            print("############################")
                    
        eval_dict.update({'top_mIoU': top_mIoU})
        eval_dict.update({'top_IoU': top_IoU})

        return eval_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, save_dir=None, **eval_kwargs):
        print('\nStarting Evaluation...')
        
        eval_dict = {}
        eval_metrics = self.eval_metrics if 'metric' not in eval_kwargs else eval_kwargs['metric']

        ## Save Occupancy ##
        if save_dir is not None:
            self.save_occupancy(occ_results, save_dir)
            self.save_render_results(occ_results, save_dir)

        ## mIoU for Occ3D v1 ##
        if 'mIoU' in eval_metrics:
            eval_dict = self.evaluate_mIoU(occ_results, eval_dict)
            
        return eval_dict

    def save_render_results(self, results, out_path):
        mmcv.mkdir_or_exist(osp.join(out_path, 'render'))
        all_renders = {}
        for index, output in enumerate(results):
            if 'render_depth' not in output.keys() or 'render_semantics' not in output.keys():
                continue
            info = self.data_infos[index]
            scene_name, token = info['scene_name'], info['token']
            if scene_name not in all_renders.keys():
                all_renders[scene_name] = {}
            all_renders[scene_name][token] = np.stack((output['render_depth'] , output['render_semantics']))

        for scene, preds in all_renders.items():
            out_file_occ = osp.join(out_path, 'render', f'{scene}.npz')
            np.savez_compressed(out_file_occ, **preds)

    def save_occupancy(self, results, out_path):
        mmcv.mkdir_or_exist(osp.join(out_path, 'occupancy'))
        mmcv.mkdir_or_exist(osp.join(out_path, 'free_space'))
        all_occs = {}
        all_fs = {}
        for index, output in enumerate(results):
            info = self.data_infos[index]
            scene_name, token = info['scene_name'], info['token']
            if scene_name not in all_occs.keys():
                all_occs[scene_name] = {}
                all_fs[scene_name] = {}
            if isinstance(output, np.ndarray):
                all_occs[scene_name][token] = output
            else:
                all_occs[scene_name][token] = output['occupancy']
                all_fs[scene_name][token] = output['free_space'][self.threshold_save_indices]

        for scene, preds in all_occs.items():
            out_file_occ = osp.join(out_path, 'occupancy', f'{scene}.npz')
            np.savez_compressed(out_file_occ, **preds)
            
        for scene, preds in all_fs.items():
            out_file_fs = osp.join(out_path, 'free_space', f'{scene}.npz')
            np.savez_compressed(out_file_fs, **preds)
