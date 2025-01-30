# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import math
import os
import os.path as osp
import tempfile
import time
import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
#from torch.utils.tensorboard._utils import figure_to_image
from tqdm import tqdm
from mmcv.parallel import DataContainer as DC
import gc


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={},
                    start=None, end=None,
                    projections_dir="/nfs/datasets/nuscenes/features/projections",
                    save_dir='/nfs/datasets/nuscenes/maskclip_features_projections_all',
                    features=None,
                    verbose=False,
                    overwrite=False,
                    sample_features=False,
                    patches=False,
                    rescale=None,
                    ):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    save_dir = f'{save_dir}{"_patches" if patches else""}'

    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_s = time.time()
    for i_iter, (batch_indices, data) in enumerate(tqdm(zip(loader_indices, data_loader), total=len(data_loader))):

        scene_name = data['img_metas'].data[0][0][0]['scene_name']
        sample_token = data['img_metas'].data[0][0][0]['sample_token']

        if save_dir is not None:
            save_dir_cur = os.path.join(save_dir, scene_name)
            if not os.path.exists(save_dir_cur):
                os.makedirs(save_dir_cur)
            save_path = os.path.join(save_dir_cur, f"{sample_token}.npz")
            if os.path.exists(save_path) and not show and not overwrite:
                continue
        projections_path = os.path.join(projections_dir, scene_name, f'{sample_token}.npz')
        projections = np.load(projections_path)
        pixels, _, cam_indices = projections['pixels'], projections['depths'], projections['cam_indices']
        all_projections = np.zeros((pixels.shape[0], 512))
        full_embeds = []
        for i, metas in enumerate(data['img_metas'].data[0][0]):
            img_c = data['img'][0][:, i]
            cam_mask = cam_indices == i
            data_c = {'img_metas': [DC([[metas]], cpu_only=True)], 'img': [img_c]}

            with torch.no_grad():
                result = model(return_loss=False, patches=patches, **data_c)
                result[0] = result[0] / np.linalg.norm(result[0], axis=0, keepdims=True)

            # Use exact projections to sample features from the feature map
            if not patches:
                if sample_features:
                    features = torch.tensor(result[0]).unsqueeze(0)
                    sample_points = torch.tensor([pixels[cam_mask].astype(np.float32)]).unsqueeze(0)
                    sample_points = sample_points / torch.tensor(metas['ori_shape'][:2]) * 2 - 1
                    point_features = F.grid_sample(
                        features,
                        sample_points[..., [1,0]],
                        align_corners=False,
                    ).squeeze().T
                else:
                    rows, cols = pixels[cam_mask].astype(int).T
                    point_features = result[0][:, rows, cols].T

                all_projections[cam_mask] = point_features
            else:
                full_embeds.append(result[0])

        if patches:
            full_embeds = torch.tensor(np.stack(full_embeds))
            if rescale is not None:
                full_embeds = F.interpolate(full_embeds, scale_factor=rescale, mode='bilinear', align_corners=False)
            np.savez_compressed(save_path, features=full_embeds.cpu().numpy().astype(np.float16))
        else:
            np.savez_compressed(save_path, features=all_projections.astype(np.float16))

        torch.cuda.empty_cache()
        del all_projections
        del full_embeds
        # gc.collect()

        if i_iter == 0:
            print(f'saved to {save_path}.')

        # if efficient_test:
        #     result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        # if format_only:
        #     result = dataset.format_results(
        #         result, indices=batch_indices, **format_args)
        # if pre_eval:
        #     # TODO: adapt samples_per_gpu > 1.
        #     # only samples_per_gpu=1 valid now
        #     result = dataset.pre_eval(result, indices=batch_indices)
        #     results.extend(result)
        # else:
        #     results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return None


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={}):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def vis_output(model, data_loader, config_name, num_vis,
               highlight_rule, black_bg, pavi=False):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    # if pavi:
    #     from pavi import SummaryWriter
    #     writer = SummaryWriter(config_name, project='maskclip')
    # else:
    #     try:
    #         from torch.utils.tensorboard import SummaryWriter
    #         writer = SummaryWriter('vis/{}'.format(config_name))
    #     except:
    #         writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('vis/{}'.format(config_name))
    except:
        writer = None

    if len(highlight_rule) == 0:
        highlight_names = []
    elif highlight_rule == 'zs5':
        highlight_names = [10, 14, 1, 18, 8, 20, 19, 5, 9, 16]
    else:
        rank, splits, strategy = highlight_rule.split('_')
        rank, splits = int(rank), int(splits)
        all_index = list(range(len(class_names)))
        if strategy == 'itv':
            highlight_names = all_index[(rank - 1)::splits]
        elif strategy == 'ctn':
            classes_per_split = len(class_names) // splits
            highlight_names = all_index[(rank - 1) * classes_per_split: rank * classes_per_split]

    count = 0
    class_names = list(dataset.CLASSES)
    palette = dataset.PALETTE
    for batch_indices, data in zip(loader_indices, data_loader):
        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        gt = dataset.get_gt_seg_map_by_idx(batch_indices[0]) if black_bg else None

        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        img, img_meta = imgs[0], img_metas[0]

        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        with torch.no_grad():
            seg_logit = model(return_loss=False, rescale=True, return_logit=True, **data)
        seg_logit = seg_logit[0]
        seg_pred = seg_logit.argmax(axis=0)

        if len(class_names) + 1 == seg_logit.shape[0] and count == 0:
            class_names = ['background'] + class_names
            palette = [(0, 0, 0)] + palette
            highlight_names = [i + 1 for i in highlight_names]

        img_seg = model.module.show_result(img_show, [seg_pred], opacity=0.5,
                                           palette=palette,
                                           classes=class_names, gt=gt)

        filename = img_metas[0]['ori_filename']
        # seg_logit = np.exp(seg_logit*100)
        seg_logit = (seg_logit == seg_logit.max(axis=0, keepdims=True))
        fig = activation_matplotlib(seg_logit, img_show, img_seg, class_names, highlight_names)
        # writer.add_figure(filename, fig)


        # img = figure_to_image(fig)
        img=None

        if writer is not None:
            writer.add_image(filename, img)

        batch_size = img_tensor.size(0)
        for _ in range(batch_size):
            prog_bar.update()

        count += 1
        if count == num_vis:
            break
    if writer is not None:
        writer.close()


def activation_matplotlib(seg_logit, image, image_seg, class_names, highlight_names):
    total = len(class_names) + 1 if image_seg is None else len(class_names) + 2
    row, col = math.ceil((total) / 5), 5
    fig = plt.figure(figsize=(6 * col, 3 * row))
    count, class_idx = 0, 0
    for _ in range(row):
        for _ in range(col):
            if count == 0:
                ax = fig.add_subplot(row, col, count + 1, xticks=[], yticks=[], title='image')
                plt.imshow(image[..., ::-1])
                count += 1
                if image_seg is not None:
                    ax = fig.add_subplot(row, col, count + 1, xticks=[], yticks=[], title='seg')
                    plt.imshow(image_seg[..., ::-1])
                    count += 1

            if count == total:
                return fig

            ax = fig.add_subplot(row, col, count + 1, xticks=[], yticks=[], title=class_names[class_idx])
            if class_idx in highlight_names:
                ax.set_title(class_names[class_idx], color='r')
            plt.imshow(seg_logit[class_idx])
            count += 1
            class_idx += 1

    # fig.tight_layout()
    return fig
