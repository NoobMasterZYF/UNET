"""
Weather Nowcasting Model — UNet Optical Flow Version.
============================================================================
修改内容：
  1. 光流计算从 TV-L1 替换为预训练的 U-Net 光流
  2. 新增 U-Net 光流预训练流程（自监督，合成数据）
  3. TS 计算修正为多标签二分类版本（原版适用于多分类）
  4. 新增特征重要性分析（Permutation Importance）

用法：
  1. 首次运行会自动预训练 U-Net 光流（约需 20-30 分钟，GPU）
  2. 预训练模型缓存到 unet_flow_pretrained.pth，后续直接加载
  3. 其余流程与原代码一致：数据加载 → 调参 → 训练最终模型 → 分析
============================================================================
"""

# ============================================================================
# 0. 导入模块
# ============================================================================
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import re
import glob
import random
import sys
import gc
import copy
import logging
import itertools
import ast
import multiprocessing

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, accuracy_score, roc_auc_score,
    average_precision_score, f1_score as sk_f1_score,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── UNet 光流模块 ──
from unet_optical_flow import (
    UNetOpticalFlow,
    pretrain_unet_flow,
    compute_unet_optical_flow,
    get_unet_flow_model,
)

# ============================================================================
# 1. 路径和参数配置
# ============================================================================
ROOT_DIR = 'D:/CMA-HKQX-2024'
info_excel = os.path.join(ROOT_DIR, 'dataset-for-training', 'infomation.xlsx')
label_dir = os.path.join(ROOT_DIR, 'GHA-SCW-Datasets', 'Label')
base_dir = os.path.join(ROOT_DIR, 'dataset-for-training')
save_path = os.path.join(base_dir, 'saved_dataset_unet.npz')  # 新文件名，区分旧版
output_dir = os.path.join(base_dir, 'output_unet')            # 新输出目录
os.makedirs(output_dir, exist_ok=True)

# 模型参数
time_steps = 6
forecast_steps = 3
img_size_radar = (400, 400)
img_size_satellite = (200, 200)
num_angles = 15
channels_satellite = 3
awos_features_dim = 9
batch_size = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 2. 样本筛选
# ============================================================================

def sample_selection(info_excel_path, label_base_dir, time_steps=6, forecast_steps=3):
    """从 infomation.xlsx 和标签文件中筛选有效样本候选"""

    def check_data_completeness(date_str, input_ts_list, label_ts_str, label_path):
        missing_reasons = []

        # AWOS
        awos_path = os.path.join(base_dir, date_str, 'AWS', f"{date_str}_AWS.csv")
        if not os.path.exists(awos_path):
            missing_reasons.append(f"AWOS: missing {awos_path}")

        # Label
        if not os.path.exists(label_path):
            missing_reasons.append(f"Label: missing {label_path}")

        # Radar (15 files per timestamp)
        radar_dir = os.path.join(base_dir, date_str, 'radar_img')
        if os.path.exists(radar_dir):
            for ts_str in input_ts_list:
                files = [f for f in os.listdir(radar_dir)
                         if f.startswith(ts_str + '_') and f.endswith('.jpg')]
                if len(files) < 15:
                    missing_reasons.append(
                        f"Radar: found {len(files)}/15 files for {ts_str}")
        else:
            missing_reasons.append(f"Radar: directory missing {radar_dir}")

        # Satellite (I/U/W channels)
        for ts_str in input_ts_list:
            parts = ts_str.split('_')
            if len(parts) != 2:
                missing_reasons.append(f"Invalid ts format: {ts_str}")
                continue
            timestamp_no_ss = f"{parts[0]}{parts[1][:4]}"
            for prefix, sub in zip(['IEC', 'UEC', 'WEC'], ['I', 'U', 'W']):
                sat_path = os.path.join(
                    base_dir, date_str, 'cloud_img', sub,
                    f"{prefix}{timestamp_no_ss}_GH4.jpg")
                if not os.path.exists(sat_path):
                    missing_reasons.append(f"Satellite {prefix}: missing {sat_path}")

        if missing_reasons:
            print(f"  Discarding: {date_str}, {label_ts_str} — {'; '.join(missing_reasons)}")
            return False
        return True

    info_df = pd.read_excel(info_excel_path)
    dates = info_df['filename'].astype(str).tolist()
    print(f"Step 1: Loaded {len(dates)} dates from infomation.xlsx")

    all_positive = []
    all_negative = []

    for date_str in dates:
        try:
            formatted_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
        except ValueError:
            print(f"Warning: invalid date format {date_str}")
            continue

        label_path = os.path.join(label_base_dir, f"{formatted_date}.xlsx")
        if not os.path.exists(label_path):
            print(f"Warning: label file missing: {label_path}")
            continue

        df = pd.read_excel(label_path)
        df['timestamp'] = pd.to_datetime(df['时间(LT)'])
        timestamps = df['timestamp'].tolist()
        labels = df[['雷暴', '短时强降水', '大风']].values.tolist()

        start_skip, end_skip = 10, 10
        if len(timestamps) < start_skip + end_skip + 1:
            continue

        for i in range(start_skip, len(timestamps) - end_skip):
            label_idx = i + forecast_steps
            if label_idx >= len(timestamps):
                continue

            target_ts = timestamps[i]
            label_ts = timestamps[label_idx]
            target_labels = labels[label_idx]

            # Filter: label time 7:00–22:00
            if not (7 <= label_ts.hour <= 22):
                continue

            input_timestamps = timestamps[i - time_steps + 1: i + 1]
            if len(input_timestamps) < time_steps:
                continue

            input_ts_list = [dt.strftime('%Y%m%d_%H%M00') for dt in input_timestamps]
            label_ts_str = label_ts.strftime('%Y%m%d_%H%M00')

            if check_data_completeness(date_str, input_ts_list, label_ts_str, label_path):
                if any(target_labels):
                    all_positive.append((formatted_date, target_ts, tuple(target_labels)))
                else:
                    all_negative.append((formatted_date, target_ts))

    print(f"Step 2: Potential positive: {len(all_positive)}, negative: {len(all_negative)}")

    # Save potential pool
    classes_list = ['雷暴', '短时强降水', '大风']
    potential_data = []
    for date, ts, lbls in all_positive:
        sample_classes = ','.join([classes_list[j] for j in range(3) if lbls[j] == 1])
        potential_data.append({
            'Type': 'Positive', 'Date': date,
            'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'Classes': sample_classes, 'Labels': str(list(lbls))
        })
    for date, ts in all_negative:
        potential_data.append({
            'Type': 'Negative', 'Date': date,
            'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'Classes': '', 'Labels': '[0, 0, 0]'
        })

    if potential_data:
        df_potential = pd.DataFrame(potential_data)
        df_potential.to_excel(os.path.join(output_dir, 'potential_samples.xlsx'), index=False)
        print(f"Potential samples saved ({len(df_potential)} rows)")

    return all_positive, all_negative


# ============================================================================
# 3. 数据加载函数
# ============================================================================

def find_nearest_ts(base_dir, yyyymmdd, target_ts_str, max_delta=10):
    """查找最近的有效时间戳作为替补"""
    target_dt = datetime.strptime(target_ts_str, '%Y%m%d_%H%M%S')
    available_files = glob.glob(
        os.path.join(base_dir, yyyymmdd, 'radar_img', '*_50kM.jpg'))
    available_ts = []
    for f in available_files:
        parts = os.path.basename(f).split('_')
        if len(parts) >= 2:
            ts = parts[0] + '_' + parts[1]
            try:
                available_ts.append((datetime.strptime(ts, '%Y%m%d_%H%M%S'), ts))
            except ValueError:
                continue

    nearest = None
    min_delta = None
    for dt, ts in available_ts:
        delta = abs((dt - target_dt).total_seconds()) / 60.0
        if delta <= max_delta and (min_delta is None or delta < min_delta):
            min_delta = delta
            nearest = ts
    return nearest


def load_radar_images(base_dir, dates_timestamps, img_size=(400, 400), num_angles=15):
    """加载雷达图像 (JPG, 15仰角)"""
    radar_data = {}
    all_timestamps = {}
    pattern = re.compile(r'(\d{8}_\d{6})_(\d+(\.\d+)?)_50kM\.jpg', re.IGNORECASE)

    for date_str in dates_timestamps:
        yyyymmdd = date_str.replace('-', '')
        date_dir = os.path.join(base_dir, yyyymmdd, 'radar_img')
        if not os.path.exists(date_dir):
            print(f"Warning: radar dir missing: {date_dir}")
            continue

        date_radar = {}
        required_ts = set(dates_timestamps[date_str])

        for filename in sorted(os.listdir(date_dir)):
            match = pattern.match(filename)
            if match and match.group(1) in required_ts:
                timestamp = match.group(1)
                angle = float(match.group(2))
                int_angle = min(max(int(round(angle)), 0), num_angles - 1)

                if timestamp not in date_radar:
                    date_radar[timestamp] = np.zeros(
                        (num_angles, img_size[0], img_size[1], 1), dtype=np.float32)

                img_path = os.path.join(date_dir, filename)
                img = np.array(Image.open(img_path).convert('L')) / 255.0
                date_radar[timestamp][int_angle] = img[:, :, np.newaxis]

        sorted_ts = sorted(date_radar.keys())
        radar_array = np.array([date_radar[ts] for ts in sorted_ts])
        radar_data[date_str] = radar_array
        all_timestamps[date_str] = sorted_ts
        print(f"Radar {date_str}: {len(sorted_ts)} timestamps")

    return radar_data, all_timestamps


def load_satellite_images(base_dir, dates_timestamps, img_size=(200, 200)):
    """加载卫星云图 (JPG, IEC/UEC/WEC 三通道)"""
    satellite_data = {}
    all_timestamps = {}
    pattern = re.compile(r'(IEC|UEC|WEC)(\d{8}\d{4})_GH4\.jpg', re.IGNORECASE)
    channel_map = {'IEC': 0, 'UEC': 1, 'WEC': 2}
    subdirs = {'IEC': 'I', 'UEC': 'U', 'WEC': 'W'}

    for date_str in dates_timestamps:
        date_dir_str = date_str.replace('-', '')
        date_dir = os.path.join(base_dir, date_dir_str, 'cloud_img')
        if not os.path.exists(date_dir):
            print(f"Warning: satellite dir missing: {date_dir}")
            continue

        date_satellite_dict = {}
        date_ts_list = []

        for channel_prefix, subdir in subdirs.items():
            sub_dir = os.path.join(date_dir, subdir)
            if not os.path.exists(sub_dir):
                continue

            for filename in sorted(os.listdir(sub_dir)):
                match = pattern.match(filename)
                if not match or match.group(1) != channel_prefix:
                    continue

                timestamp_raw = match.group(2)
                try:
                    dt = datetime.strptime(timestamp_raw, '%Y%m%d%H%M')
                except ValueError:
                    continue
                full_ts = dt.strftime('%Y%m%d_%H%M%S')

                if full_ts not in dates_timestamps.get(date_str, []):
                    continue

                img_path = os.path.join(sub_dir, filename)
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize((img_size[1], img_size[0]), Image.Resampling.LANCZOS)
                    img_array = np.array(img) / 255.0
                except Exception as e:
                    print(f"Warning: cannot read {img_path}: {e}")
                    continue

                channel_idx = channel_map[channel_prefix]
                if full_ts not in date_satellite_dict:
                    date_satellite_dict[full_ts] = np.zeros(
                        (img_size[0], img_size[1], 3), dtype=np.float32)

                date_satellite_dict[full_ts][:, :, channel_idx] = img_array
                if full_ts not in date_ts_list:
                    date_ts_list.append(full_ts)

        sorted_ts = sorted(date_ts_list)
        if sorted_ts:
            satellite_array = np.array([date_satellite_dict[ts] for ts in sorted_ts])
            satellite_data[date_str] = satellite_array
            all_timestamps[date_str] = sorted_ts
            print(f"Satellite {date_str}: {len(sorted_ts)} timestamps")
        else:
            print(f"Warning: {date_str} no satellite data loaded")

    return satellite_data, all_timestamps


def load_awos_data(base_dir, dates_timestamps):
    """加载 AWOS 气象站数据 (CSV, 9维特征)"""
    awos_data = {}
    all_timestamps = {}
    features = ['10分风向', '10分风速', '气压', '温度', '湿度',
                '24h_变温度', '3h_变温度', '24h_变气压', '3h_变气压']

    scaler = StandardScaler()

    for date_str in dates_timestamps:
        yyyymmdd = date_str.replace('-', '')
        awos_path = os.path.join(base_dir, yyyymmdd, 'AWS', f"{yyyymmdd}_AWS.csv")
        if not os.path.exists(awos_path):
            print(f"Warning: AWOS file missing: {awos_path}")
            continue

        try:
            df = pd.read_csv(awos_path, encoding='gb18030', sep=',')
        except Exception as e:
            print(f"Error reading {awos_path}: {e}")
            continue

        required_cols = features + ['datetime']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: missing columns in {awos_path}: {missing_cols}")
            continue

        df = df.dropna(subset=required_cols)

        # Parse timestamps
        parsed_ts = []
        for t in df['datetime']:
            dt = None
            for fmt in ['%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M:%S',
                         '%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M']:
                try:
                    dt = datetime.strptime(t.strip(), fmt)
                    break
                except ValueError:
                    pass
            parsed_ts.append(dt.strftime('%Y%m%d_%H%M%S') if dt else None)

        df['timestamp'] = parsed_ts
        required_ts = set(dates_timestamps[date_str])
        mask = df['timestamp'].isin(required_ts)
        df_filtered = df[mask]

        if df_filtered.empty:
            print(f"Warning: {date_str} no matching AWOS timestamps")
            continue

        awos_values = df_filtered[features].values.astype(np.float32)
        awos_values = scaler.fit_transform(awos_values)

        awos_data[date_str] = awos_values
        all_timestamps[date_str] = df_filtered['timestamp'].tolist()
        print(f"AWOS {date_str}: {len(df_filtered)} rows")

    return awos_data, all_timestamps


def load_labels(label_base_dir, dates_timestamps):
    """加载标签数据 (Excel, 雷暴/短时强降水/大风 三标签)"""
    labels_data = {}
    all_timestamps = {}

    for date_str in dates_timestamps:
        label_path = os.path.join(label_base_dir, f"{date_str}.xlsx")
        if not os.path.exists(label_path):
            print(f"Warning: label file missing: {label_path}")
            continue

        df = pd.read_excel(label_path)
        df['timestamp'] = pd.to_datetime(df['时间(LT)']).apply(
            lambda dt: dt.strftime('%Y%m%d_%H%M%S'))

        required_ts = set(dates_timestamps[date_str])
        mask = df['timestamp'].isin(required_ts)
        df_filtered = df[mask]

        if df_filtered.empty:
            print(f"Warning: {date_str} no matching label timestamps")
            continue

        labels_values = df_filtered[['雷暴', '短时强降水', '大风']].values.astype(np.float32)
        labels_data[date_str] = labels_values
        all_timestamps[date_str] = df_filtered['timestamp'].tolist()
        print(f"Labels {date_str}: {len(df_filtered)} rows")

    return labels_data, all_timestamps


# ============================================================================
# 4. 样本构建（★ UNet 光流替换 TV-L1）
# ============================================================================

def create_samples(radar_data, satellite_data, awos_data, labels_data,
                   samples_list, all_timestamps,
                   optical_flow_model=None,  # ★ 新增：预训练的 UNet 光流模型
                   time_steps=6, forecast_steps=3):
    """
    构建样本，使用 U-Net 光流替代 TV-L1。
    """
    samples = {'radar': [], 'satellite': [], 'awos': [], 'optical_flow': [],
               'labels': [], 'indices': [], 'timestamps': []}
    skipped_count = 0

    flow_device = next(optical_flow_model.parameters()).device if optical_flow_model else 'cpu'

    for idx, (date, label_ts, substituted) in enumerate(samples_list):
        try:
            # Input timestamps
            input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i)
                             for i in range(time_steps)]
            input_str_list = [ts.strftime('%Y%m%d_%H%M%S') for ts in input_ts_list]
            label_str = label_ts.strftime('%Y%m%d_%H%M%S')

            # Apply substitutions
            for i, ts_str in enumerate(input_str_list):
                if ts_str in substituted:
                    input_str_list[i] = substituted[ts_str]
            if label_str in substituted:
                label_str = substituted[label_str]

            # Get data
            date_radar = radar_data.get(date, np.array([]))
            date_satellite = satellite_data.get(date, np.array([]))
            date_awos = awos_data.get(date, np.array([]))
            date_labels = labels_data.get(date, np.array([]))
            date_ts = all_timestamps.get(date, [])

            if not date_ts or date_radar.size == 0:
                raise ValueError("No data for this date")

            # Compute indices
            input_indices = [date_ts.index(ts_str) for ts_str in input_str_list
                             if ts_str in date_ts]
            label_idx = date_ts.index(label_str) if label_str in date_ts else -1

            if len(input_indices) != time_steps or label_idx == -1:
                raise ValueError(f"Incomplete timestamps: input {len(input_indices)}/{time_steps}")

            if max(input_indices) >= len(date_radar) or label_idx >= len(date_labels):
                raise IndexError("Index out of range")

            # ─────────────────────────────────────────────────────
            # ★ 核心修改：U-Net 光流替代 TV-L1
            # ─────────────────────────────────────────────────────
            radar_3rd = date_radar[input_indices, 2, :, :, 0]  # (6, 400, 400)

            if optical_flow_model is not None:
                optical_flow = compute_unet_optical_flow(
                    optical_flow_model, radar_3rd, device=flow_device)
            else:
                # Fallback: zero flow (should not happen if model is loaded)
                print(f"Warning: no optical flow model, using zeros for sample {idx}")
                optical_flow = np.zeros(
                    (time_steps - 1, radar_3rd.shape[1], radar_3rd.shape[2], 2),
                    dtype=np.float32)

            # Pad to time_steps if needed
            if len(optical_flow) < time_steps - 1:
                pad = time_steps - 1 - len(optical_flow)
                optical_flow = np.concatenate([
                    optical_flow,
                    np.tile(optical_flow[-1:], (pad, 1, 1, 1))
                ], axis=0)
            # ─────────────────────────────────────────────────────

            # Extract data
            radar_data_sample = date_radar[input_indices]
            satellite_data_sample = date_satellite[input_indices]
            awos_data_sample = date_awos[input_indices]
            optical_flow_sample = optical_flow
            labels_sample = date_labels[label_idx]

            samples['radar'].append(radar_data_sample)
            samples['satellite'].append(satellite_data_sample)
            samples['awos'].append(awos_data_sample)
            samples['optical_flow'].append(optical_flow_sample)
            samples['labels'].append(labels_sample)
            samples['timestamps'].append(label_str)
            samples['indices'].append(idx)

        except Exception as e:
            print(f"Warning: sample {idx} on {date} skipped: {e}")
            skipped_count += 1
            continue

    for key in samples:
        samples[key] = np.array(samples[key])

    print(f"Created {len(samples['indices'])} samples, skipped {skipped_count}")
    return samples


# ============================================================================
# 5. 数据增强
# ============================================================================

def augment_rare(samples, rare_multiplier=3):
    """过采样稀有类别（短时强降水、大风）"""
    augmented = {k: list(samples[k]) for k in samples}

    transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ])

    for i in range(len(samples['labels'])):
        lbl = samples['labels'][i]
        if lbl[1] or lbl[2]:  # Rare class
            for _ in range(rare_multiplier - 1):
                # Augment radar
                radar = samples['radar'][i]
                aug_radar = np.zeros_like(radar)
                for t in range(radar.shape[0]):
                    for a in range(radar.shape[1]):
                        img = radar[t, a]
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                        if img_tensor.shape[0] == 1:
                            img_tensor = img_tensor.repeat(3, 1, 1)
                        aug_tensor = transform(img_tensor)
                        if aug_tensor.shape[0] == 3:
                            aug_tensor = aug_tensor.mean(dim=0, keepdim=True)
                        aug_radar[t, a] = aug_tensor.permute(1, 2, 0).numpy()

                # Augment satellite
                satellite = samples['satellite'][i]
                aug_satellite = np.zeros_like(satellite)
                for t in range(satellite.shape[0]):
                    img = satellite[t]
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                    aug_tensor = transform(img_tensor)
                    aug_satellite[t] = aug_tensor.permute(1, 2, 0).numpy()

                aug_optical_flow = samples['optical_flow'][i]
                aug_awos = (samples['awos'][i] +
                            np.random.normal(0, 0.01, samples['awos'][i].shape))

                augmented['radar'].append(aug_radar)
                augmented['satellite'].append(aug_satellite)
                augmented['awos'].append(aug_awos)
                augmented['optical_flow'].append(aug_optical_flow)
                augmented['labels'].append(samples['labels'][i])
                augmented['indices'].append(samples['indices'][i])
                augmented['timestamps'].append(samples['timestamps'][i])

    for k in augmented:
        augmented[k] = np.array(augmented[k])

    print(f"After augmentation: {len(augmented['labels'])} (was {len(samples['labels'])})")
    return augmented


# ============================================================================
# 6. 主数据加载（★ 包含 UNet 预训练逻辑）
# ============================================================================

def load_data():
    """
    完整数据加载流程：
      样本筛选 → 负样本优化 → UNet 光流预训练 → 数据加载 → 样本构建 → 增强 → 保存
    """
    global save_path  # 使用新的文件名

    # 6a. 如果有缓存直接加载
    if os.path.exists(save_path):
        print(f"Loading cached dataset from {save_path}")
        loaded = np.load(save_path, allow_pickle=True)
        samples = {key: loaded[key] for key in loaded.files}
        print(f"Loaded {len(samples['indices'])} samples")

        # Output negative sample info
        negative_samples = []
        for i, lbl in enumerate(samples['labels']):
            if np.all(lbl == [0, 0, 0]):
                negative_samples.append({
                    'Index': samples['indices'][i],
                    'Timestamp': samples['timestamps'][i],
                    'Labels': str(list(lbl)),
                    'Type': 'Negative'
                })
        if negative_samples:
            df_neg = pd.DataFrame(negative_samples)
            df_neg.to_excel(os.path.join(output_dir, 'negative_samples.xlsx'), index=False)
            print(f"Negative samples: {len(df_neg)}")
        return samples

    # 6b. 获取候选样本
    all_positive, all_negative = sample_selection(info_excel, label_dir)
    print(f"Candidate positive: {len(all_positive)}, negative: {len(all_negative)}")

    # Save sample names
    save_sample_names(all_positive, all_negative, output_dir)

    # 6c. 构建时间戳字典
    all_samples_list = [(date, target_ts, {}) for date, target_ts, _ in all_positive] + \
                       [(date, target_ts, {}) for date, target_ts in all_negative]
    dates_timestamps = {}
    for date, label_ts, _ in all_samples_list:
        input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i) for i in range(time_steps)]
        required_str = [ts.strftime('%Y%m%d_%H%M%S') for ts in input_ts_list + [label_ts]]
        dates_timestamps.setdefault(date, []).extend(required_str)
        dates_timestamps[date] = sorted(set(dates_timestamps[date]))

    # 6d. 加载雷达数据用于负样本优化
    radar_data, radar_ts = load_radar_images(base_dir, dates_timestamps)

    # 6e. 负样本优化：基于雷达特征的余弦相似度
    def extract_radar_features(date, label_ts, radar_data, radar_ts, time_steps=6):
        try:
            input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i) for i in range(time_steps)]
            input_str_list = [ts.strftime('%Y%m%d_%H%M%S') for ts in input_ts_list]
            date_ts_list = radar_ts.get(date, [])
            if not date_ts_list:
                return None
            input_indices = [date_ts_list.index(ts_str) for ts_str in input_str_list
                             if ts_str in date_ts_list]
            if len(input_indices) != time_steps:
                return None
            radar_seq = radar_data[date][input_indices]
            features = []
            for t in range(time_steps):
                slice_t = radar_seq[t].flatten()
                features.extend([np.mean(slice_t), np.var(slice_t)])
            return np.array(features)
        except Exception as e:
            print(f"Feature extraction failed for {date}, {label_ts}: {e}")
            return None

    pos_features = []
    pos_keys = []
    for date, label_ts, _ in all_positive:
        feat = extract_radar_features(date, label_ts, radar_data, radar_ts)
        if feat is not None:
            pos_features.append(feat)
            pos_keys.append((date, label_ts))

    neg_features = []
    neg_keys = []
    for date, target_ts in all_negative:
        feat = extract_radar_features(date, target_ts, radar_data, radar_ts)
        if feat is not None:
            neg_features.append(feat)
            neg_keys.append((date, target_ts))

    pos_features = np.array(pos_features)
    neg_features = np.array(neg_features)
    print(f"Extracted features — positive: {len(pos_features)}, negative: {len(neg_features)}")

    if len(pos_features) > 0 and len(neg_features) > 0:
        similarities = cosine_similarity(neg_features, pos_features).mean(axis=1)
        sorted_indices = np.argsort(similarities)[::-1]
        num_delete = int(len(neg_features) * 0.3)
        to_keep_indices = sorted_indices[num_delete:]
        all_negative = [neg_keys[i] for i in to_keep_indices]
        print(f"Hard negative filtering: {len(neg_keys)} → {len(all_negative)}")

    # Check for manual negative samples
    manual_neg_path = os.path.join(base_dir, 'negative_samples0.xlsx')
    if os.path.exists(manual_neg_path):
        try:
            df_manual = pd.read_excel(manual_neg_path)
            all_negative = []
            for _, row in df_manual.iterrows():
                ts_str = row['Timestamp']
                dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                all_negative.append((dt.strftime('%Y-%m-%d'), dt))
            print(f"Using manual negatives: {len(all_negative)} from {manual_neg_path}")
        except Exception as e:
            print(f"Warning: failed to read manual negatives: {e}")

    del radar_data, radar_ts
    gc.collect()

    # 6f. 分离稀有正样本
    rare_positive = [c for c in all_positive if c[2][1] == 1 or c[2][2] == 1]
    other_positive = [c for c in all_positive if c not in rare_positive]
    print(f"Rare positive: {len(rare_positive)}, Other positive: {len(other_positive)}")

    # 6g. 验证正样本完整性
    target_positive = len(all_positive)
    negative_ratio = 2.5
    substituted_timestamps = {}

    def validate_candidates(candidates, max_count=None):
        valid = []
        for candidate in candidates:
            if max_count and len(valid) >= max_count:
                break
            date, label_ts, lbls = candidate
            input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i) for i in range(6)]
            required_str = [ts.strftime('%Y%m%d_%H%M%S') for ts in input_ts_list + [label_ts]]

            all_exist = True
            sample_substituted = {}
            for ts_str in required_str:
                yyyymmdd = ts_str[:8]
                radar_pattern = os.path.join(
                    base_dir, yyyymmdd, 'radar_img', f'{ts_str[:8]}_{ts_str[9:15]}_*_50kM.jpg')
                files = glob.glob(radar_pattern)
                if len(files) == 0:
                    substitute = find_nearest_ts(base_dir, yyyymmdd, ts_str)
                    if substitute:
                        sample_substituted[ts_str] = substitute
                    else:
                        all_exist = False
                        break
            if all_exist:
                valid.append((date, label_ts, lbls))
                if sample_substituted:
                    substituted_timestamps[(date, label_ts.strftime('%Y%m%d_%H%M%S'))] = sample_substituted
        return valid

    valid_rare = validate_candidates(rare_positive)
    num_other_needed = max(0, target_positive - len(valid_rare))
    random.shuffle(other_positive)
    valid_other = validate_candidates(other_positive, num_other_needed)

    valid_positive = valid_rare + valid_other
    print(f"Valid positive: rare {len(valid_rare)}, other {len(valid_other)}, total {len(valid_positive)}")

    # 6h. 负样本
    target_negative = int(len(valid_positive) * negative_ratio)
    valid_negative = []
    candidates_neg = list(all_negative)
    random.shuffle(candidates_neg)
    for candidate in candidates_neg:
        if len(valid_negative) >= target_negative:
            break
        valid_negative.append(candidate)

    # 6i. 构建样本列表和日期时间戳
    samples_list = [(date, label_ts,
                     substituted_timestamps.get((date, label_ts.strftime('%Y%m%d_%H%M%S')), {}))
                    for date, label_ts, _ in valid_positive] + \
                   [(date, label_ts, {}) for date, label_ts in valid_negative]
    print(f"Total samples: positive {len(valid_positive)}, negative {len(valid_negative)}")

    dates_timestamps = {}
    for date, label_ts, substituted in samples_list:
        input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i) for i in range(6)]
        required_str = [ts.strftime('%Y%m%d_%H%M%S') for ts in input_ts_list + [label_ts]]
        for i, ts_str in enumerate(required_str):
            if ts_str in substituted:
                required_str[i] = substituted[ts_str]
        dates_timestamps.setdefault(date, []).extend(required_str)
        dates_timestamps[date] = sorted(set(dates_timestamps[date]))

    # ─────────────────────────────────────────────────────────
    # ★ 6j. 预训练/加载 UNet 光流模型
    # ─────────────────────────────────────────────────────────
    unet_model_path = os.path.join(base_dir, 'unet_flow_pretrained.pth')
    print("\n" + "=" * 60)
    print("UNet Optical Flow Model Setup")
    print("=" * 60)

    optical_flow_model = get_unet_flow_model(
        base_dir=base_dir,
        dates=list(dates_timestamps.keys()),
        model_path=unet_model_path,
        device=str(device),
        force_retrain=False,
    )
    print("=" * 60 + "\n")

    # 6k. 加载所有数据
    radar_data, radar_ts = load_radar_images(base_dir, dates_timestamps)
    satellite_data, satellite_ts = load_satellite_images(base_dir, dates_timestamps)
    awos_data, awos_ts = load_awos_data(base_dir, dates_timestamps)
    labels_data, labels_ts = load_labels(label_dir, dates_timestamps)

    # 6l. 创建样本（★ 传入 UNet 光流模型）
    samples = create_samples(
        radar_data, satellite_data, awos_data, labels_data,
        samples_list, radar_ts,
        optical_flow_model=optical_flow_model,  # ★
    )

    # 6m. 增强稀有样本
    samples = augment_rare(samples, rare_multiplier=3)

    # 6n. 统计
    rare_count = sum(1 for lbl in samples['labels'] if any(lbl[1:]))
    wind_count = sum(1 for lbl in samples['labels'] if lbl[2])
    rain_count = sum(1 for lbl in samples['labels'] if lbl[1])
    print(f"Final stats — Rare: {rare_count}, Wind: {wind_count}, Rain: {rain_count}")

    # 6o. 保存
    if len(samples['indices']) > 0:
        print(f"Saving dataset to {save_path}")
        np.savez(save_path, **samples)

    return samples


# ============================================================================
# 7. 模型架构
# ============================================================================

class TwoDCNN(nn.Module):
    """2D CNN 特征提取器"""
    def __init__(self, input_channels, feature_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)

        if input_channels in [15, 2]:
            h, w = 400, 400
        else:
            h, w = 200, 200
        h, w = h // 4, w // 4
        self.fc = nn.Linear(64 * h * w, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvLSTM2d(nn.Module):
    """2D Convolutional LSTM"""
    def __init__(self, input_size, hidden_size, kernel_size=3, num_layers=1, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            in_channels=input_size + hidden_size,
            out_channels=4 * hidden_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True
        )

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        batch_size, seq_len, _, height, width = input_tensor.size()

        if hidden_state is None:
            h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                              height, width, device=input_tensor.device)
            c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                              height, width, device=input_tensor.device)
        else:
            h_t, c_t = hidden_state

        h_t_new = torch.zeros_like(h_t)
        c_t_new = torch.zeros_like(c_t)
        layer_output = []

        for layer in range(self.num_layers):
            h_t_layer = h_t[layer].clone()
            c_t_layer = c_t[layer].clone()
            output_inner = []

            for t in range(seq_len):
                combined = torch.cat((input_tensor[:, t], h_t_layer), dim=1)
                gates = self.conv(combined)
                i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)

                i_gate = torch.sigmoid(i_gate)
                f_gate = torch.sigmoid(f_gate)
                o_gate = torch.sigmoid(o_gate)
                c_tilde = torch.tanh(c_gate)

                c_t_layer_new = f_gate * c_t_layer + i_gate * c_tilde
                h_t_layer_new = o_gate * torch.tanh(c_t_layer_new)

                output_inner.append(h_t_layer_new.clone())
                h_t_layer = h_t_layer_new
                c_t_layer = c_t_layer_new

            layer_output.append(torch.stack(output_inner, dim=1))
            h_t_new[layer] = h_t_layer.clone()
            c_t_new[layer] = c_t_layer.clone()

        layer_output = torch.stack(layer_output, dim=0)
        return layer_output[-1], (h_t_new, c_t_new)


class WeatherModel(nn.Module):
    """多模态天气预测模型 (Radar + Satellite + AWOS + Optical Flow)"""
    def __init__(self, time_steps=6, use_optical_flow=False):
        super().__init__()
        self.time_steps = time_steps
        self.use_optical_flow = use_optical_flow

        self.radar_cnn = TwoDCNN(input_channels=15, feature_dim=64)
        self.satellite_cnn = TwoDCNN(input_channels=3, feature_dim=64)
        self.conv_lstm = ConvLSTM2d(input_size=64, hidden_size=64, kernel_size=3,
                                    num_layers=1, batch_first=True)
        self.fc_awos = nn.Linear(9, 16)

        input_dim = 64 + 64 + 16
        if use_optical_flow:
            self.flow_cnn = TwoDCNN(input_channels=2, feature_dim=64)
            input_dim += 64

        self.fc = nn.Linear(input_dim, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, radar_data, satellite_data, awos_data, optical_flow=None):
        batch_size = radar_data.size(0)

        # Radar features
        radar_features = []
        for t in range(self.time_steps):
            radar_t = radar_data[:, t].squeeze(-1)
            radar_features.append(self.radar_cnn(radar_t))
        radar_features = torch.stack(radar_features, dim=1)

        # Satellite features
        satellite_features = []
        for t in range(self.time_steps):
            satellite_t = satellite_data[:, t].permute(0, 3, 1, 2)
            satellite_features.append(self.satellite_cnn(satellite_t))
        satellite_features = torch.stack(satellite_features, dim=1)

        # ConvLSTM
        rf = radar_features.unsqueeze(-1).unsqueeze(-1)
        _, (h_n, _) = self.conv_lstm(rf)
        radar_out = h_n[-1].view(batch_size, -1)

        sf = satellite_features.unsqueeze(-1).unsqueeze(-1)
        _, (h_n, _) = self.conv_lstm(sf)
        satellite_out = h_n[-1].view(batch_size, -1)

        # AWOS
        awos_out = self.relu(self.fc_awos(awos_data[:, -1]))

        # Optical flow (optional)
        if self.use_optical_flow and optical_flow is not None:
            flow_features = []
            flow_seq_len = optical_flow.size(1)
            for t in range(min(flow_seq_len, self.time_steps)):
                flow_t = optical_flow[:, t]
                if flow_t.size(1) != 2:
                    flow_t = flow_t.permute(0, 3, 1, 2)
                flow_features.append(self.flow_cnn(flow_t))
            flow_features = torch.stack(flow_features, dim=1)
            ff = flow_features.unsqueeze(-1).unsqueeze(-1)
            _, (h_n, _) = self.conv_lstm(ff)
            flow_out = h_n[-1].view(batch_size, -1)
            features = torch.cat([radar_out, satellite_out, awos_out, flow_out], dim=1)
        else:
            features = torch.cat([radar_out, satellite_out, awos_out], dim=1)

        return self.sigmoid(self.fc(features))


# ============================================================================
# 8. 损失函数
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification"""
    def __init__(self, alpha, gamma, device=None):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
        self.gamma = gamma
        self.device = device

    def forward(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha[None, :] * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class CombinedLoss(nn.Module):
    """Focal Loss + Dice Loss"""
    def __init__(self, alpha, gamma, focal_weight=0.7, dice_weight=0.3, smooth=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def dice_loss(self, inputs, targets):
        intersection = (inputs * targets).sum(dim=0)
        sum_pred = inputs.sum(dim=0)
        sum_true = targets.sum(dim=0)
        dice = (2. * intersection + self.smooth) / (sum_pred + sum_true + self.smooth)
        return 1 - dice.mean()

    def forward(self, inputs, targets):
        return (self.focal_weight * self.focal(inputs, targets) +
                self.dice_weight * self.dice_loss(inputs, targets))


# ============================================================================
# 9. 评估指标（★ 修正 TS 为多标签版本）
# ============================================================================

def compute_ts_score(y_true, y_pred, class_idx=None):
    """
    ★ 修正版 TS (Threat Score / CSI) — 适用于多标签二分类。
    TS = TP / (TP + FP + FN)，对每个类别独立计算二元混淆矩阵。
    """
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("Inputs must be 2D arrays for multi-label")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    n_classes = y_true.shape[1]
    ts_scores = []
    for i in range(n_classes):
        if class_idx is not None and i != class_idx:
            continue
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        if cm.shape[0] > 1 and cm.shape[1] > 1:
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
        else:
            tp = fp = fn = 0
        ts = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        ts_scores.append(ts)

    return np.mean(ts_scores) if ts_scores else 0.0


def compute_metrics(y_true, y_pred, y_prob):
    """
    计算多标签评估指标：Accuracy, F1_macro, ROC_AUC, TS, POD, FAR, CSI
    """
    metrics = {}

    # Per-class accuracy average
    per_class_acc = [accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    metrics['accuracy'] = np.mean(per_class_acc)

    # F1 macro
    metrics['f1_macro'] = sk_f1_score(y_true, y_pred, average='macro', zero_division=0)

    # ROC AUC per class
    auc_scores = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0 and len(np.unique(y_true[:, i])) > 1:
            auc_scores.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    metrics['roc_auc'] = np.mean(auc_scores) if auc_scores else 0.0

    # TS (★ 修正版)
    metrics['ts'] = compute_ts_score(y_true, y_pred)

    # POD, FAR, CSI per class
    pod_scores, far_scores, csi_scores = [], [], []
    for i in range(y_true.shape[1]):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        if cm.shape[0] > 1 and cm.shape[1] > 1:
            tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
        else:
            tp = fp = fn = 0
        pod_scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        far_scores.append(fp / (tp + fp) if (tp + fp) > 0 else 0.0)
        csi_scores.append(tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0)

    metrics['pod'] = np.mean(pod_scores)
    metrics['far'] = np.mean(far_scores)
    metrics['csi'] = np.mean(csi_scores)

    return metrics


# ============================================================================
# 10. 训练函数
# ============================================================================

def train_model(model, train_loader, val_loader, timestamp_list, model_name='model',
                lr=0.002, alpha=None, gamma=5, num_epochs=100, patience=20,
                trial_id=None, is_final=False, focal_weight=0.7):
    """训练一个模型，返回 best_metrics 和 errors"""

    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, 'output.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Dynamic alpha
    train_labels = np.concatenate([batch[4].numpy() for batch in train_loader])
    label_dist = np.mean(train_labels, axis=0)
    if alpha is None:
        alpha = [1 / max(d, 1e-6) for d in label_dist]
        alpha = [a / sum(alpha) for a in alpha]
    print(f"Dynamic alpha: {alpha}")

    criterion = CombinedLoss(alpha=alpha, gamma=gamma,
                             focal_weight=focal_weight, dice_weight=0.3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # File paths
    suffix = '_final' if is_final else f'_trial_{trial_id}' if trial_id is not None else ''
    model_file = os.path.join(output_dir, f'model_{model_name}{suffix}.pth')
    metrics_file = os.path.join(output_dir, f'metrics_{model_name}{suffix}.csv')
    error_file = os.path.join(output_dir, f'errors_{model_name}{suffix}.csv')

    # Skip if already exists
    if os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} already exists, skipping")
        metrics = pd.read_csv(metrics_file).to_dict('records')[0]
        errors = pd.read_csv(error_file).to_dict('records') if os.path.exists(error_file) else []
        return metrics, errors

    val_labels = np.concatenate([batch[4].numpy() for batch in val_loader])
    print(f"Train label distribution: {label_dist}")
    print(f"Val label distribution: {np.mean(val_labels, axis=0)}")

    best_ts = 0
    best_metrics = {}
    errors = []
    patience_counter = 0
    epoch_records = []
    train_loss_list, val_loss_list, lr_history = [], [], []

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        train_loss = 0
        for batch in train_loader:
            radar_data, satellite_data, awos_data, optical_flow, labels, indices, _ = \
                [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(radar_data, satellite_data, awos_data, optical_flow)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        train_loss_list.append(train_loss)
        lr_history.append(optimizer.param_groups[0]['lr'])

        # ── Val ──
        model.eval()
        val_loss = 0
        y_true, y_pred, y_prob = [], [], []
        y_errors = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                radar_data, satellite_data, awos_data, optical_flow, labels, indices, batch_indices = \
                    [x.to(device) for x in batch]
                outputs = model(radar_data, satellite_data, awos_data, optical_flow)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                y_errors.extend([
                    (idx.item(), pred.tolist(), label.tolist(), timestamp_list[batch_idx.item()])
                    for idx, pred, label, batch_idx in zip(indices, preds, labels, batch_indices)
                ])
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(outputs.cpu().numpy())

        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics['val_loss'] = val_loss

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"TS={metrics['ts']:.4f}, POD={metrics['pod']:.4f}, FAR={metrics['far']:.4f}")

        epoch_records.append({
            'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
            'accuracy': metrics['accuracy'], 'f1_macro': metrics['f1_macro'],
            'roc_auc': metrics['roc_auc'], 'ts': metrics['ts'],
            'pod': metrics['pod'], 'far': metrics['far'], 'csi': metrics['csi']
        })

        if metrics['ts'] > best_ts:
            best_ts = metrics['ts']
            best_metrics = metrics.copy()
            errors = y_errors
            checkpoint = {
                'state_dict': model.state_dict(),
                'train_loss': train_loss_list,
                'val_loss': val_loss_list,
                'lr_history': lr_history
            }
            torch.save(checkpoint, model_file)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    pd.DataFrame(epoch_records).to_csv(metrics_file, index=False)
    pd.DataFrame(errors, columns=['index', 'prediction', 'label', 'timestamp']).to_csv(
        error_file, index=False)

    print(f"Best metrics for {model_name}: {best_metrics}")
    return best_metrics, errors


# ============================================================================
# 11. 超参数调优
# ============================================================================

def hyperparameter_tune(train_loader, val_loader, timestamp_list, model_name='model'):
    """Grid search over hyperparameters"""
    param_grid = {
        'lr': [0.0001, 0.0005, 0.001],
        'alpha': [[0.8, 0.9, 0.95], [0.85, 0.95, 0.98], [0.85, 0.95, 1.0]],
        'gamma': [3, 4, 5],
        'num_epochs': [50],
        'focal_weight': [0.6, 0.7, 0.8],
        'patience': [50]
    }

    default_params = {
        'lr': 0.0005, 'alpha': [0.8, 0.95, 1.0], 'gamma': 3,
        'focal_weight': 0.7, 'num_epochs': 50, 'patience': 50
    }

    results = []
    params_file = os.path.join(output_dir, f'trial_params_{model_name}.csv')

    # Load existing parameters
    saved_params = {}
    if os.path.exists(params_file):
        saved_params_df = pd.read_csv(params_file)
        for _, row in saved_params_df.iterrows():
            saved_params[int(row['trial'])] = {
                'lr': row['lr'], 'alpha': eval(row['alpha']),
                'gamma': row['gamma'], 'num_epochs': row['num_epochs'],
                'patience': row['patience'], 'focal_weight': row['focal_weight']
            }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))

    for trial_idx, combo in enumerate(all_combinations):
        trial_id = trial_idx + 1
        metrics_file = os.path.join(output_dir, f'metrics_{model_name}_trial_{trial_id}.csv')

        # Skip if done
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            metrics = {
                'accuracy': metrics_df['accuracy'].max(),
                'f1_macro': metrics_df['f1_macro'].max(),
                'roc_auc': metrics_df['roc_auc'].max(),
                'ts': metrics_df['ts'].max(),
                'pod': metrics_df['pod'].max() if 'pod' in metrics_df.columns else None,
                'far': metrics_df['far'].min() if 'far' in metrics_df.columns else None,
                'csi': metrics_df['csi'].max() if 'csi' in metrics_df.columns else None,
            }
            results.append({**saved_params.get(trial_id, {}), **metrics, 'trial': trial_id})
            continue

        params = dict(zip(keys, combo))
        print(f"\nTrial {trial_id}/{len(all_combinations)}: {params}")

        # Save trial params
        trial_params = {
            'trial': trial_id, 'lr': params['lr'], 'alpha': str(params['alpha']),
            'gamma': params['gamma'], 'num_epochs': params['num_epochs'],
            'patience': params['patience'], 'focal_weight': params['focal_weight']
        }
        pd.DataFrame([trial_params]).to_csv(
            params_file, mode='a', header=not os.path.exists(params_file), index=False)

        model = WeatherModel(use_optical_flow=(model_name == 'flow')).to(device)
        model.apply(lambda m: nn.init.xavier_uniform_(m.weight)
                    if isinstance(m, (nn.Conv2d, nn.Linear)) else None)

        try:
            best_metrics, errors = train_model(
                model, train_loader, val_loader, timestamp_list, model_name,
                lr=params['lr'], alpha=params['alpha'], gamma=params['gamma'],
                num_epochs=params['num_epochs'], patience=params['patience'],
                trial_id=trial_id, focal_weight=params['focal_weight']
            )
            results.append({**params, **best_metrics, 'trial': trial_id})
        except Exception as e:
            print(f"Trial {trial_id} failed: {e}")
            continue

    # Save results
    results_file = os.path.join(output_dir, f'hyperparam_results_{model_name}.csv')
    pd.DataFrame(results).to_csv(results_file, index=False)

    if results:
        best_trial = max(results, key=lambda x: x['ts'])
        print(f"\nBest trial for {model_name}: {best_trial}")
        return best_trial
    else:
        print(f"No successful trials for {model_name}")
        return default_params


# ============================================================================
# 12. 分析与可视化（★ 特征重要性 + Loss曲线 + 混淆矩阵 + ROC + PR）
# ============================================================================

def analyze_features_and_errors(model_flow, model_no_flow, val_data, val_labels,
                                val_timestamps, checkpoint_flow, checkpoint_no_flow):
    """
    全面分析模型性能：
      - Loss/LR 曲线
      - Permutation 特征重要性
      - 混淆矩阵
      - ROC / PR 曲线
      - 概率分布直方图
      - 时间序列误差分布
      - Per-class 指标对比
    """

    # ── 12a. Loss / LR 曲线 ──
    def plot_loss_curve(train_loss, val_loss, title, prefix):
        if not train_loss or not val_loss:
            return
        epochs = np.arange(1, len(train_loss) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(title)
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{prefix}_loss_curve.png'))
        plt.close()
        pd.DataFrame({'epoch': epochs, 'train_loss': train_loss, 'val_loss': val_loss}).to_csv(
            os.path.join(output_dir, f'{prefix}_loss.csv'), index=False)

    def plot_lr_curve(lr_history, title, prefix):
        if not lr_history:
            return
        epochs = np.arange(1, len(lr_history) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, lr_history)
        plt.xlabel('Epoch'); plt.ylabel('LR'); plt.title(title)
        plt.yscale('log'); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{prefix}_lr_curve.png'))
        plt.close()

    for ckpt, name in [(checkpoint_flow, 'flow'), (checkpoint_no_flow, 'no_flow')]:
        plot_loss_curve(ckpt.get('train_loss', []), ckpt.get('val_loss', []),
                        f'Loss Curve ({name})', name)
        plot_lr_curve(ckpt.get('lr_history', []), f'LR Curve ({name})', name)

    # ── 12b. 预测 ──
    val_labels_np = np.array(val_labels)
    classes = ['thunderstorm', 'heavy_rain', 'strong_wind']

    val_data_flow = [v.clone().detach().cpu() if isinstance(v, torch.Tensor) else v for v in val_data]
    val_data_no_flow = val_data_flow[:3]

    preds_flow_prob = predict_fn(model_flow, val_data_flow)
    preds_no_flow_prob = predict_fn(model_no_flow, val_data_no_flow)

    preds_flow_bin = (preds_flow_prob > 0.5).astype(int)
    preds_no_flow_bin = (preds_no_flow_prob > 0.5).astype(int)

    metrics_flow = compute_metrics(val_labels_np, preds_flow_bin, preds_flow_prob)
    metrics_no_flow = compute_metrics(val_labels_np, preds_no_flow_bin, preds_no_flow_prob)

    print(f"\n{'='*60}")
    print("Model Performance Comparison")
    print(f"{'='*60}")
    for name, m in [('With UNet Flow', metrics_flow), ('Without Flow', metrics_no_flow)]:
        print(f"\n{name}:")
        for k in ['accuracy', 'f1_macro', 'roc_auc', 'ts', 'pod', 'far', 'csi']:
            print(f"  {k}: {m[k]:.4f}")

    pd.DataFrame([metrics_flow]).to_csv(os.path.join(output_dir, 'performance_flow.csv'), index=False)
    pd.DataFrame([metrics_no_flow]).to_csv(os.path.join(output_dir, 'performance_no_flow.csv'), index=False)

    # ── 12c. ★ 特征重要性 (Permutation Importance) ──
    def custom_permutation_importance(model, data_list, y_true, features, n_repeats=5):
        base_prob = predict_fn(model, data_list)
        base_bin = (base_prob > 0.5).astype(int)
        base_score = compute_ts_score(y_true, base_bin)
        importances = np.zeros((len(features), n_repeats))

        for i in range(len(features)):
            for r in range(n_repeats):
                shuf_data = [d.clone() for d in data_list]
                shuf_idx = torch.randperm(shuf_data[i].shape[0])
                shuf_data[i] = shuf_data[i][shuf_idx]
                shuf_prob = predict_fn(model, shuf_data)
                shuf_bin = (shuf_prob > 0.5).astype(int)
                shuf_score = compute_ts_score(y_true, shuf_bin)
                importances[i, r] = base_score - shuf_score

        return {'importances_mean': np.mean(importances, axis=1),
                'importances_std': np.std(importances, axis=1),
                'importances': importances}

    try:
        # Flow model
        features_flow = ['Radar', 'Satellite', 'AWOS', 'Optical Flow']
        imp_flow = custom_permutation_importance(model_flow, val_data_flow, val_labels_np, features_flow)
        print("\nFeature Importance (With UNet Flow):")
        for i, (m, s) in enumerate(zip(imp_flow['importances_mean'], imp_flow['importances_std'])):
            print(f"  {features_flow[i]}: {m:.4f} ± {s:.4f}")

        # Bar plot
        sorted_idx = np.argsort(imp_flow['importances_mean'])
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), imp_flow['importances_mean'][sorted_idx],
                 xerr=imp_flow['importances_std'][sorted_idx])
        plt.yticks(range(len(sorted_idx)), [features_flow[i] for i in sorted_idx])
        plt.xlabel('Importance (Mean TS Decrease)')
        plt.title('Feature Importance — With UNet Optical Flow')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'feature_importance_flow.png'))
        plt.close()

        df_imp = pd.DataFrame({
            'feature': features_flow,
            'importance_mean': imp_flow['importances_mean'],
            'importance_std': imp_flow['importances_std']
        })
        df_imp.to_csv(os.path.join(output_dir, 'feature_importance_flow.csv'), index=False)

        # No-flow model
        features_no_flow = ['Radar', 'Satellite', 'AWOS']
        imp_no_flow = custom_permutation_importance(model_no_flow, val_data_no_flow, val_labels_np, features_no_flow)
        print("\nFeature Importance (Without Flow):")
        for i, (m, s) in enumerate(zip(imp_no_flow['importances_mean'], imp_no_flow['importances_std'])):
            print(f"  {features_no_flow[i]}: {m:.4f} ± {s:.4f}")

        sorted_idx = np.argsort(imp_no_flow['importances_mean'])
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), imp_no_flow['importances_mean'][sorted_idx],
                 xerr=imp_no_flow['importances_std'][sorted_idx])
        plt.yticks(range(len(sorted_idx)), [features_no_flow[i] for i in sorted_idx])
        plt.xlabel('Importance (Mean TS Decrease)')
        plt.title('Feature Importance — Without Optical Flow')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'feature_importance_no_flow.png'))
        plt.close()

    except Exception as e:
        print(f"Feature importance failed: {e}")

    # ── 12d. 混淆矩阵 ──
    def plot_confusion_matrix(true_labels, pred_bin, mode):
        for i, cls in enumerate(classes):
            cm = confusion_matrix(true_labels[:, i], pred_bin[:, i])
            plt.figure(figsize=(6, 4))
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title(f'Confusion Matrix ({mode} - {cls})')
            plt.colorbar()
            for x in range(2):
                for y in range(2):
                    plt.text(y, x, str(cm[x, y]), ha='center',
                             color='white' if cm[x, y] > cm.max() / 2 else 'black')
            plt.xticks([0, 1], ['Neg', 'Pos']); plt.yticks([0, 1], ['Neg', 'Pos'])
            plt.ylabel('True'); plt.xlabel('Predicted')
            plt.savefig(os.path.join(output_dir, f'cm_{mode}_{cls}.png'))
            plt.close()

    plot_confusion_matrix(val_labels_np, preds_flow_bin, 'flow')
    plot_confusion_matrix(val_labels_np, preds_no_flow_bin, 'no_flow')

    # ── 12e. ROC 曲线 ──
    for mode, preds_prob in [('flow', preds_flow_prob), ('no_flow', preds_no_flow_prob)]:
        plt.figure(figsize=(8, 6))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(val_labels_np[:, i], preds_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0, 1]); plt.ylim([0, 1.05])
        plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.title(f'ROC Curve ({mode})')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'roc_curve_{mode}.png'))
        plt.close()

    # ── 12f. PR 曲线 ──
    for mode, preds_prob in [('flow', preds_flow_prob), ('no_flow', preds_no_flow_prob)]:
        plt.figure(figsize=(8, 6))
        for i, cls in enumerate(classes):
            precision, recall, _ = precision_recall_curve(val_labels_np[:, i], preds_prob[:, i])
            ap = average_precision_score(val_labels_np[:, i], preds_prob[:, i])
            plt.step(recall, precision, where='post', label=f'{cls} (AP={ap:.2f})')
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'PR Curve ({mode})')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'pr_curve_{mode}.png'))
        plt.close()

    # ── 12g. 概率分布直方图 ──
    for mode, preds_prob in [('flow', preds_flow_prob), ('no_flow', preds_no_flow_prob)]:
        for i, cls in enumerate(classes):
            pos = preds_prob[val_labels_np[:, i] == 1, i]
            neg = preds_prob[val_labels_np[:, i] == 0, i]
            plt.figure(figsize=(8, 6))
            if len(pos) > 0:
                plt.hist(pos, bins=20, alpha=0.5, label='Positive', density=True)
            if len(neg) > 0:
                plt.hist(neg, bins=20, alpha=0.5, label='Negative', density=True)
            plt.xlabel('Predicted Probability'); plt.ylabel('Density')
            plt.title(f'Probability Distribution ({mode} - {cls})')
            plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'prob_hist_{mode}_{cls}.png'))
            plt.close()

    # ── 12h. 模型对比柱状图 ──
    for metric_name in ['pod', 'far', 'csi', 'ts']:
        plt.figure(figsize=(8, 6))
        x = np.arange(len(classes))
        width = 0.35
        flow_vals = []
        no_flow_vals = []
        for i in range(len(classes)):
            cm_f = confusion_matrix(val_labels_np[:, i], preds_flow_bin[:, i])
            cm_nf = confusion_matrix(val_labels_np[:, i], preds_no_flow_bin[:, i])
            if cm_f.shape[0] > 1:
                tp_f, fp_f, fn_f = cm_f[1, 1], cm_f[0, 1], cm_f[1, 0]
            else:
                tp_f = fp_f = fn_f = 0
            if cm_nf.shape[0] > 1:
                tp_nf, fp_nf, fn_nf = cm_nf[1, 1], cm_nf[0, 1], cm_nf[1, 0]
            else:
                tp_nf = fp_nf = fn_nf = 0

            if metric_name == 'pod':
                flow_vals.append(tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0)
                no_flow_vals.append(tp_nf / (tp_nf + fn_nf) if (tp_nf + fn_nf) > 0 else 0)
            elif metric_name == 'far':
                flow_vals.append(fp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0)
                no_flow_vals.append(fp_nf / (tp_nf + fp_nf) if (tp_nf + fp_nf) > 0 else 0)
            elif metric_name == 'csi':
                flow_vals.append(tp_f / (tp_f + fn_f + fp_f) if (tp_f + fn_f + fp_f) > 0 else 0)
                no_flow_vals.append(tp_nf / (tp_nf + fn_nf + fp_nf) if (tp_nf + fn_nf + fp_nf) > 0 else 0)
            elif metric_name == 'ts':
                flow_vals.append(tp_f / (tp_f + fn_f + fp_f) if (tp_f + fn_f + fp_f) > 0 else 0)
                no_flow_vals.append(tp_nf / (tp_nf + fn_nf + fp_nf) if (tp_nf + fn_nf + fp_nf) > 0 else 0)

        plt.bar(x - width/2, flow_vals, width, label='With UNet Flow')
        plt.bar(x + width/2, no_flow_vals, width, label='Without Flow')
        plt.xlabel('Class'); plt.ylabel(metric_name.upper())
        plt.title(f'Per-Class {metric_name.upper()} Comparison')
        plt.xticks(x, classes); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'comparison_{metric_name}.png'))
        plt.close()

    print("\nAnalysis complete. All figures saved to:", output_dir)


def predict_fn(model, data):
    """预测函数，用于特征重要性分析"""
    if len(data) == 4:
        radar, satellite, awos, flow = data
    elif len(data) == 3:
        radar, satellite, awos = data
        flow = None
    else:
        raise ValueError(f"Unexpected inputs: {len(data)}")

    model.eval()
    with torch.no_grad():
        radar = radar.to(device)
        satellite = satellite.to(device)
        awos = awos.to(device)
        if model.use_optical_flow and flow is not None:
            flow = flow.to(device)
            outputs = model(radar, satellite, awos, flow)
        else:
            outputs = model(radar, satellite, awos)
    return outputs.cpu().numpy()


# ============================================================================
# 13. 辅助函数
# ============================================================================

def save_sample_names(all_positive, all_negative, output_dir):
    """保存正负样本名称到 txt"""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'positive_samples.txt'), 'w', encoding='utf-8') as f:
        f.write("正样本列表\n" + "=" * 60 + "\n")
        for i, (date, ts, labels) in enumerate(all_positive, 1):
            label_names = []
            if labels[0] == 1: label_names.append("雷暴")
            if labels[1] == 1: label_names.append("短时强降水")
            if labels[2] == 1: label_names.append("大风")
            f.write(f"{i:3d}. {date} {ts.strftime('%Y-%m-%d %H:%M:%S')} [{', '.join(label_names)}]\n")

    with open(os.path.join(output_dir, 'negative_samples.txt'), 'w', encoding='utf-8') as f:
        f.write("负样本列表\n" + "=" * 60 + "\n")
        for i, (date, ts) in enumerate(all_negative, 1):
            f.write(f"{i:3d}. {date} {ts.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Sample names saved to {output_dir}")


def run_analysis():
    """独立运行分析（从缓存加载数据）"""
    samples = load_data()
    n = len(samples['radar'])
    split = int(0.8 * n)

    val_data = [
        torch.FloatTensor(samples['radar'][split:]),
        torch.FloatTensor(samples['satellite'][split:]),
        torch.FloatTensor(samples['awos'][split:]),
        torch.FloatTensor(samples['optical_flow'][split:]),
    ]
    val_labels = samples['labels'][split:]
    val_timestamps = samples['timestamps'][split:]

    # Load checkpoints
    checkpoint_flow = torch.load(os.path.join(output_dir, 'model_flow_final.pth'),
                                 map_location=device)
    model_flow = WeatherModel(use_optical_flow=True).to(device)
    if isinstance(checkpoint_flow, dict) and 'state_dict' in checkpoint_flow:
        model_flow.load_state_dict(checkpoint_flow['state_dict'])
    else:
        model_flow.load_state_dict(checkpoint_flow)

    checkpoint_no_flow = torch.load(os.path.join(output_dir, 'model_no_flow_final.pth'),
                                    map_location=device)
    model_no_flow = WeatherModel(use_optical_flow=False).to(device)
    if isinstance(checkpoint_no_flow, dict) and 'state_dict' in checkpoint_no_flow:
        model_no_flow.load_state_dict(checkpoint_no_flow['state_dict'])
    else:
        model_no_flow.load_state_dict(checkpoint_no_flow)

    analyze_features_and_errors(model_flow, model_no_flow, val_data, val_labels,
                                val_timestamps, checkpoint_flow, checkpoint_no_flow)


# ============================================================================
# 14. 主函数
# ============================================================================

def main():
    """完整流程：数据加载 → 调参 → 训练最终模型 → 分析"""
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 14a. 加载数据（含 UNet 光流预训练）
    samples = load_data()
    print(f"Loaded {len(samples['radar'])} samples")

    # 14b. 准备 Tensor
    radar_tensor = torch.FloatTensor(samples['radar'])
    satellite_tensor = torch.FloatTensor(samples['satellite'])
    awos_tensor = torch.FloatTensor(samples['awos'])
    flow_tensor = torch.FloatTensor(samples['optical_flow'])
    label_tensor = torch.FloatTensor(samples['labels'])
    indices = torch.LongTensor(samples['indices'])
    timestamp_list = samples['timestamps']

    print(f"Tensor shapes — Radar: {radar_tensor.shape}, Flow: {flow_tensor.shape}, "
          f"Labels: {label_tensor.shape}")

    # Label distribution
    label_array = label_tensor.numpy()
    print(f"Label counts — 雷暴: {label_array[:,0].sum():.0f}, "
          f"短时强降水: {label_array[:,1].sum():.0f}, 大风: {label_array[:,2].sum():.0f}")

    # 14c. 分割
    train_idx, val_idx = train_test_split(range(len(label_tensor)), test_size=0.2, random_state=42)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # 14d. DataLoaders
    dataset = TensorDataset(
        radar_tensor, satellite_tensor, awos_tensor, flow_tensor,
        label_tensor, indices, torch.arange(len(radar_tensor))
    )
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    num_workers = min(8, multiprocessing.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # 14e. 超参数调优
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning — With UNet Flow")
    print("=" * 60)
    best_trial_flow = hyperparameter_tune(train_loader, val_loader, timestamp_list, model_name='flow')

    print("\n" + "=" * 60)
    print("Hyperparameter Tuning — Without Flow")
    print("=" * 60)
    best_trial_no_flow = hyperparameter_tune(train_loader, val_loader, timestamp_list, model_name='no_flow')

    # 14f. 训练最终模型
    print("\n" + "=" * 60)
    print("Training Final Models")
    print("=" * 60)

    # With flow
    model_flow = WeatherModel(use_optical_flow=True).to(device)
    model_flow.apply(lambda m: nn.init.xavier_uniform_(m.weight)
                     if isinstance(m, (nn.Conv2d, nn.Linear)) else None)
    alpha_flow = ast.literal_eval(best_trial_flow['alpha']) if isinstance(best_trial_flow['alpha'], str) else best_trial_flow['alpha']
    metrics_flow, _ = train_model(
        model_flow, train_loader, val_loader, timestamp_list, model_name='flow',
        lr=best_trial_flow['lr'], alpha=alpha_flow, gamma=best_trial_flow['gamma'],
        num_epochs=best_trial_flow['num_epochs'], patience=best_trial_flow['patience'],
        trial_id=0, is_final=True, focal_weight=best_trial_flow['focal_weight']
    )

    # Without flow
    model_no_flow = WeatherModel(use_optical_flow=False).to(device)
    model_no_flow.apply(lambda m: nn.init.xavier_uniform_(m.weight)
                        if isinstance(m, (nn.Conv2d, nn.Linear)) else None)
    alpha_no_flow = ast.literal_eval(best_trial_no_flow['alpha']) if isinstance(best_trial_no_flow['alpha'], str) else best_trial_no_flow['alpha']
    metrics_no_flow, _ = train_model(
        model_no_flow, train_loader, val_loader, timestamp_list, model_name='no_flow',
        lr=best_trial_no_flow['lr'], alpha=alpha_no_flow, gamma=best_trial_no_flow['gamma'],
        num_epochs=best_trial_no_flow['num_epochs'], patience=best_trial_no_flow['patience'],
        trial_id=0, is_final=True, focal_weight=best_trial_no_flow['focal_weight']
    )

    # 14g. 对比总结
    comparison = {
        'Model': ['NoFlow', 'UNetFlow'],
        'Accuracy': [metrics_no_flow['accuracy'], metrics_flow['accuracy']],
        'F1_macro': [metrics_no_flow['f1_macro'], metrics_flow['f1_macro']],
        'ROC_AUC': [metrics_no_flow['roc_auc'], metrics_flow['roc_auc']],
        'TS': [metrics_no_flow['ts'], metrics_flow['ts']],
        'POD': [metrics_no_flow['pod'], metrics_flow['pod']],
        'FAR': [metrics_no_flow['far'], metrics_flow['far']],
        'CSI': [metrics_no_flow['csi'], metrics_flow['csi']],
    }
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    print("\n" + "=" * 60)
    print("FINAL COMPARISON: UNet Flow vs No Flow")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    # 14h. 运行分析
    print("\n" + "=" * 60)
    print("Running Detailed Analysis")
    print("=" * 60)
    run_analysis()

    print("\nAll done! Results saved to:", output_dir)


if __name__ == "__main__":
    main()
