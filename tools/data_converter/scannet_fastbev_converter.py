"""
ScanNet → FastBEV 数据转换器

将 ScanNet 的 posed_images + scannet_instance_data 转换为 FastBEV 可用的 PKL 格式

生成的 PKL 格式与 NuScenesMultiViewDataset.get_data_info() 兼容:
{
    'metainfo': {...},
    'data_list': [
        {
            'sample_idx': str,
            'lidar_points': {'lidar_path': str, 'num_pts_feats': int},
            'img_paths': [str, ...],           # 多帧图像路径
            'intrinsics': [np.array(3x3), ...], # 每帧内参
            'extrinsics': [np.array(4x4), ...], # 每帧外参 (world-to-camera)
            'axis_align_matrix': np.array(4x4),
            'instances': [{'bbox_3d': [x,y,z,w,h,l], 'bbox_label_3d': int}, ...],
            ...
        },
        ...
    ]
}

用法:
    python tools/data_converter/scannet_fastbev_converter.py \
        --scannet_root dataset/scannet \
        --out_dir dataset/scannet/mm3Ddetection \
        --num_frames 6 \
        --frame_stride 10
"""
import argparse
import os
import pickle
import numpy as np
from os import path as osp


# ScanNet 18 类检测 (NYU40 ID → 内部索引)
CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
           'bookshelf', 'picture', 'counter', 'desk', 'curtain',
           'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
           'garbagebin')

CAT_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
CAT_ID_TO_LABEL = {cat_id: i for i, cat_id in enumerate(CAT_IDS)}


def get_split_scenes(scannet_root, split='train'):
    """从 mm3Ddetection 中已有的 PKL 读取场景列表"""
    pkl_path = osp.join(scannet_root, 'mm3Ddetection', f'scannet_infos_{split}.pkl')
    if osp.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        scenes = []
        for info in data['data_list']:
            scene_id = info['lidar_points']['lidar_path'].replace('.bin', '')
            scenes.append(scene_id)
        return scenes

    # 备选：从 split 文件读取
    split_file = osp.join(scannet_root, f'{split}.txt')
    if osp.exists(split_file):
        with open(split_file) as f:
            return [line.strip() for line in f if line.strip()]

    raise FileNotFoundError(f'找不到 {split} 集的场景列表')


def load_existing_info(scannet_root, scene_id, split='train'):
    """从已有 PKL 中加载场景的标注信息"""
    pkl_path = osp.join(scannet_root, 'mm3Ddetection', f'scannet_infos_{split}.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    for info in data['data_list']:
        if info['lidar_points']['lidar_path'].replace('.bin', '') == scene_id:
            return info
    return None


def get_camera_info_for_scene(posed_images_root, scene_id, num_frames=6, frame_stride=1):
    """
    从 posed_images 目录加载相机信息

    Args:
        posed_images_root: posed_images 根目录
        scene_id: 场景 ID
        num_frames: 使用的帧数
        frame_stride: 帧间隔

    Returns:
        img_paths: 图像路径列表
        intrinsics: 内参列表 [3x3]
        extrinsics: 外参列表 [4x4] (world-to-camera)
    """
    scene_dir = osp.join(posed_images_root, scene_id)
    if not osp.exists(scene_dir):
        return None, None, None

    color_dir = osp.join(scene_dir, 'color')
    pose_dir = osp.join(scene_dir, 'pose')
    intrinsic_file = osp.join(scene_dir, 'intrinsic_color.txt')

    if not osp.exists(color_dir) or not osp.exists(pose_dir):
        return None, None, None

    # 读取内参
    intrinsic = np.loadtxt(intrinsic_file, dtype=np.float32)
    if intrinsic.shape == (4, 4):
        intrinsic = intrinsic[:3, :3]

    # 获取所有帧
    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith('.jpg')])
    pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.txt')])

    assert len(color_files) == len(pose_files), \
        f'{scene_id}: color({len(color_files)}) != pose({len(pose_files)})'

    total_frames = len(color_files)

    # 选择帧：均匀采样 num_frames 帧
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        # 从中间区域均匀采样（避免开头和结尾的不稳定帧）
        start = total_frames // 10
        end = total_frames - total_frames // 10
        indices = np.linspace(start, end - 1, num_frames, dtype=int).tolist()

    img_paths = []
    intrinsics_list = []
    extrinsics_list = []

    for idx in indices:
        img_path = osp.join('posed_images', scene_id, 'color', color_files[idx])
        pose_file = osp.join(pose_dir, pose_files[idx])

        # 读取 camera-to-world 位姿
        camera_to_world = np.loadtxt(pose_file, dtype=np.float32)
        if camera_to_world.shape != (4, 4):
            continue

        # world-to-camera = inv(camera-to-world)
        world_to_camera = np.linalg.inv(camera_to_world)

        img_paths.append(img_path)
        intrinsics_list.append(intrinsic.copy())
        extrinsics_list.append(world_to_camera)

    return img_paths, intrinsics_list, extrinsics_list


def convert_scannet_to_fastbev(scannet_root, out_dir, num_frames=6, frame_stride=1):
    """
    转换 ScanNet 数据为 FastBEV 格式

    Args:
        scannet_root: ScanNet 数据根目录
        out_dir: 输出目录
        num_frames: 每场景使用的帧数
        frame_stride: 帧间隔
    """
    posed_images_root = osp.join(scannet_root, 'posed_images')

    for split in ['train', 'val']:
        print(f'\n{"="*60}')
        print(f'处理 {split} 集...')
        print(f'{"="*60}')

        scenes = get_split_scenes(scannet_root, split)
        print(f'共 {len(scenes)} 个场景')

        data_list = []
        skipped = 0

        for i, scene_id in enumerate(scenes):
            # 加载已有标注
            existing_info = load_existing_info(scannet_root, scene_id, split)
            if existing_info is None:
                print(f'  [{i+1}] {scene_id}: 跳过 (无标注)')
                skipped += 1
                continue

            # 加载相机信息
            img_paths, intrinsics, extrinsics = get_camera_info_for_scene(
                posed_images_root, scene_id, num_frames, frame_stride)

            if img_paths is None or len(img_paths) == 0:
                print(f'  [{i+1}] {scene_id}: 跳过 (无图像)')
                skipped += 1
                continue

            # 构建 FastBEV 兼容的 info
            new_info = {
                'sample_idx': scene_id,
                'lidar_points': existing_info['lidar_points'].copy(),
                'pts_semantic_mask_path': existing_info.get('pts_semantic_mask_path'),
                'pts_instance_mask_path': existing_info.get('pts_instance_mask_path'),
                'axis_align_matrix': existing_info['axis_align_matrix'],
                'instances': existing_info['instances'].copy(),
                # FastBEV 需要的相机信息
                'img_paths': img_paths,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,
            }

            data_list.append(new_info)

            if (i + 1) % 50 == 0:
                print(f'  [{i+1}/{len(scenes)}] 已处理...')

        print(f'  完成! 有效场景: {len(data_list)}, 跳过: {skipped}')

        # 保存 PKL
        out_pkl = osp.join(out_dir, f'scannet_fastbev_infos_{split}.pkl')
        out_data = {
            'metainfo': {
                'categories': {cls: idx for idx, cls in enumerate(CLASSES)},
                'dataset': 'scannet_fastbev',
                'info_version': '2.0',
                'num_frames': num_frames,
            },
            'data_list': data_list,
        }
        with open(out_pkl, 'wb') as f:
            pickle.dump(out_data, f)
        print(f'  保存: {out_pkl}')
        print(f'  大小: {osp.getsize(out_pkl) / 1024:.1f} KB')


def main():
    parser = argparse.ArgumentParser(description='ScanNet → FastBEV 数据转换')
    parser.add_argument('--scannet_root', type=str,
                        default='dataset/scannet',
                        help='ScanNet 数据根目录')
    parser.add_argument('--out_dir', type=str,
                        default='dataset/scannet/mm3Ddetection',
                        help='输出目录')
    parser.add_argument('--num_frames', type=int, default=6,
                        help='每场景使用的帧数 (默认 6)')
    parser.add_argument('--frame_stride', type=int, default=1,
                        help='帧间隔 (默认 1)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    convert_scannet_to_fastbev(
        args.scannet_root, args.out_dir, args.num_frames, args.frame_stride)


if __name__ == '__main__':
    main()
