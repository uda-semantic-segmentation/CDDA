# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Provide args as argument to main()
# - Snapshot source code
# - Build UDA model instead of regular one

import argparse
import copy
import datetime
import os
import os.path as osp
import sys
import time
import random
from collections import Counter
from gc import get_threshold

import numpy as np
import mmcv
import torch
import torchvision
from mmcv.utils import Config, DictAction
from mmseg.datasets import build_dataset
from datetime import datetime
from mmseg.datasets import build_dataloader
import matplotlib.pyplot as plt
import cv2




def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args(args):
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args





def create_matrix(seg_img, depth_img, cls_num):
    temp_depth = torch.zeros((cls_num, 256))
    valid_indices = torch.nonzero((seg_img < cls_num) & (depth_img <= 255))
    seg_vals = seg_img[valid_indices[:, 0], valid_indices[:, 1]].long()

    depth_vals = depth_img[valid_indices[:, 0], valid_indices[:, 1]].long()

    counts = Counter(zip(seg_vals.tolist(), depth_vals.tolist()))
    for (seg, depth), count in counts.items():
        temp_depth[seg, depth] += count

    return temp_depth.cuda()




def cul_mix(croped_labels_path, croped_depth_path, matrix_path, nums, datasetName):
    print('=========cul_mix')
    total_mean_matrix2 = torch.zeros([nums, 19, 256]).cuda()
    for i in range(nums):
        print(f'{i}=======get {datasetName} depth distribution...')
        path_sudolbl = os.path.join(croped_labels_path, f'{i}.png')
        path_dep = os.path.join(croped_depth_path, f'{i}.png')
        dep = cv2.imread(path_dep, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
        gt = cv2.imread(path_sudolbl)
        dep = np.asarray(dep).astype(int)

        dep = dep.astype(np.uint8)
        gt = torch.tensor(gt).cuda()
        dep = torch.tensor(dep).cuda()

        matrix_temp = create_matrix(gt[:, :, 0], dep, 19)
        total_mean_matrix2[i] = matrix_temp
    torch.save(total_mean_matrix2, matrix_path)


def main(args, dep_args):
    args = parse_args(args)

    assert not dep_args.not_get_images or not dep_args.not_get_distributions
    cfg = Config.fromfile(dep_args.config)

    dataset_name = dep_args.dataset_name
    source_preprocessed_path = dep_args.source_preprocessed_path

    croped_images_path = os.path.join(source_preprocessed_path, 'images')
    croped_labels_path = os.path.join(source_preprocessed_path, 'labels')
    croped_depth_path = os.path.join(source_preprocessed_path, 'depth')



    os.makedirs(croped_images_path, exist_ok=True)
    os.makedirs(croped_labels_path, exist_ok=True)
    os.makedirs(croped_depth_path, exist_ok=True)

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    meta = dict()

    # set random seeds
    if args.seed is None and 'seed' in cfg:
        args.seed = cfg['seed']
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.splitext(osp.basename(args.config))[0]

    dataset = [build_dataset(cfg.data.train)]
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset]
    data_loaders = data_loaders[0]

    if dataset_name == 'synthia':
        nums = 9400
    elif dataset_name == 'gta':
        nums = 24966
    else:
        print('=============plz input right dataset name!!!!')
        exit(0)
    count = nums if dep_args.not_get_images else -1
    for j, batch in enumerate(data_loaders):
        if count < nums:
            # save images, gts and depth
            src_imgs = batch['img'].data[0]
            src_gts = batch['gt_semantic_seg'].data[0]
            src_metas = batch['img_metas'].data[0]

            for i in range(cfg.data.samples_per_gpu):
                src_dep = src_metas[i]['dep'].data[0]
                count += 1

                print(count)

                src_img = np.array(src_imgs[i].cpu())
                src_label_gt = np.array(src_gts[i].cpu(), dtype=np.uint8)
                src_depth = np.array(src_dep.cpu(), dtype=np.uint8)

                path_dep_src = os.path.join(croped_depth_path, f'{count}.png')
                path_img_src = os.path.join(croped_images_path, f'{count}.png')
                path_gt_src = os.path.join(croped_labels_path, f'{count}.png')

                src_img = torch.Tensor(src_img)
                mean = [123.675, 116.28, 103.53]
                std = [58.395, 57.12, 57.375]
                MEAN = [-meann / stdd for meann, stdd in zip(mean, std)]
                STD = [1 / stdd for stdd in std]
                nor = torchvision.transforms.Normalize(mean=MEAN, std=STD)
                src_img = nor(src_img)
                src_img = np.array(src_img)
                src_img = src_img.transpose(1, 2, 0)
                src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
                src_label_gt = src_label_gt.transpose(1, 2, 0)

                cv2.imwrite(path_dep_src, src_depth)
                cv2.imwrite(path_img_src, src_img)
                cv2.imwrite(path_gt_src, src_label_gt)

                print('---------------------------------')
        else:
            print('calculating......')
            # Get depth distributions of source domain
            source_matrix_path = os.path.join(source_preprocessed_path, f'{dataset_name}_{nums}_distribution.pth')
            if not dep_args.not_get_distributions:
                cul_mix(croped_labels_path, croped_depth_path, source_matrix_path, nums, dataset_name)
            break


if __name__ == '__main__':
    main(sys.argv[1:])
