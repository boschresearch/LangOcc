# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
# Code adapted from POP-3D (https://github.com/vobecant/POP3D/blob/main/generate_projections_nuscenes.py)

# Use as:
# python tools/generate_rays.py --exact

import os
import copy
import argparse
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from tqdm import tqdm


class NuScenesMatchDataset(Dataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(
        self,
        phase,
        nusc_root,
        version="test",
        shuffle=False,
        save_dir=None,
        overwrite=False,
        exact_projections=False,
        **kwargs,
    ):
        self.phase = phase
        self.shuffle = shuffle

        self.H, self.W = None, None
        self.overwrite = overwrite
        self.exact_projections = exact_projections
        self.max_dist = 40

        self.pc_range = [-40., -40., -1.0, 40., 40., 5.4] # in ego frame

        assert save_dir is not None
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        else:
            self.nusc = NuScenes(
                version=f"v1.0-{version}", dataroot=nusc_root, verbose=True
            )
        # print(f'NuScenes loaded with {len(self.nusc)} samples.')
        self.camera_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        self.list_keyframes = []
        # a skip ratio can be used to reduce the dataset size and accelerate experiments
        skip_ratio = 1
        skip_counter = 0
        if phase in ("train", "val", "test", "mini_train", "mini_val"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_scans(scene)

    def create_list_of_scans(self, scene):
        # Get first and last keyframe in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        list_data = []
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            list_data.append(current_sample["data"])
            current_sample_token = current_sample["next"]

        # Add new scans in the list
        self.list_keyframes.extend(list_data)

    def map_pointcloud_to_image(self, data, min_dist: float = 1.0, store_points=False, ):
        pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)

        sample = self.nusc.get('sample', pointsensor['sample_token'])
        scene_name = self.nusc.get('scene', sample['scene_token'])['name']

        save_dir = os.path.join(self.save_dir, scene_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{sample["token"]}.npz')
        if os.path.exists(save_path) and not self.overwrite:
            return
        
        all_pixels = []
        all_depths = []
        all_points = []

        for i, camera_name in enumerate(self.camera_list):
            
            cam = self.nusc.get("sample_data", data[camera_name])

            pc = copy.deepcopy(pc_original)
            if self.H is None and self.W is None:
                im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))
                self.H, self.W = im.shape[:2]

            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            cs_record = self.nusc.get(
                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            )
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # EDIT: Filter out points that are out of occupancy range (in ego frame at lidar sweep time)
            points_ego_lidar = pc.points.T[:, :3]
            oob_mask = (points_ego_lidar[:, 0] > self.pc_range[0]) & (points_ego_lidar[:, 0] < self.pc_range[3]) & (
                points_ego_lidar[:, 1] > self.pc_range[1]) & (points_ego_lidar[:, 1] < self.pc_range[4]) & (
                points_ego_lidar[:, 2] > self.pc_range[2]) & (points_ego_lidar[:, 2] < self.pc_range[5])

            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get(
                "calibrated_sensor", cam["calibrated_sensor_token"]
            )
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cs_record["camera_intrinsic"]),
                normalize=True,
            )

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to avoid
            # seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            # mask = np.logical_and(mask, depths < self.max_dist)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < self.W - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < self.H - 1)
            mask = np.logical_and(mask, oob_mask)
            matching_points = np.where(mask)[0]
            if self.exact_projections:
                matching_pixels = np.flip(points[matching_points], axis=1).astype(np.float32)
            else:
                matching_pixels = np.round(np.flip(points[matching_points], axis=1)).astype(np.uint16)
            matching_depths = depths[matching_points]

            all_pixels.append(matching_pixels)
            all_depths.append(matching_depths)
            all_points.append(matching_points)

        # create cam index array
        # cam_indices = np.array([i for i, pixels in enumerate(all_pixels) for _ in range(len(pixels))]).astype(np.uint8)
        cam_indices = np.concatenate([np.full(p.shape[0], i) for i,p in enumerate(all_pixels)]).astype(np.uint8)
        all_pixels = np.concatenate(all_pixels, axis=0)
        all_depths = np.concatenate(all_depths, axis=0)#.astype(np.float16)
        all_points = np.concatenate(all_points, axis=0).astype(np.uint32)

        if store_points:
            np.savez_compressed(save_path, pixels=all_pixels, depths=all_depths, cam_indices=cam_indices, points=all_points)
        else:
            np.savez_compressed(save_path, pixels=all_pixels, depths=all_depths, cam_indices=cam_indices)

    def __len__(self):
        return len(self.list_keyframes)

    def __getitem__(self, idx):
        self.map_pointcloud_to_image(self.list_keyframes[idx])
        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--nusc_root', type=str, default='../data/nuscenes')
    parser.add_argument('--rays-dir', type=str, default='../data/rays')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'])
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--exact', action='store_true')

    args = parser.parse_args()

    nusc_root = args.nusc_root
    rays_root = args.rays_dir
    if not os.path.exists(rays_root):
        os.makedirs(rays_root)
  
    for phase in args.splits:
    # for phase in ['train','val','test']:
        print(f'\n\nGenerating rays for "{phase}" split.')
        if phase == 'mini_train' or phase =='mini_val':
            version = 'mini'
        elif phase == 'test':
            version = 'test'
        else:
            version = 'trainval'
        dataset = NuScenesMatchDataset(phase=phase, nusc_root=nusc_root, save_dir=rays_root, version=version, overwrite=args.overwrite, exact_projections=args.exact)
        for _ in tqdm(dataset):
            pass