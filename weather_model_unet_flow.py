"""
Weather Nowcasting Model — UNet Optical Flow Version (Single File).
============================================================================
  1. UNet 光流模型 + 自监督预训练（替代 TV-L1）
  2. TS 计算修正为多标签二分类版本
  3. 特征重要性分析（Permutation Importance）
  4. 单文件，无需额外 import

用法：
  python weather_model_unet_flow.py
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
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# 1. 路径和参数配置
# ============================================================================
ROOT_DIR = 'D:/CMA-HKQX-2024'
info_excel = os.path.join(ROOT_DIR, 'dataset-for-training', 'infomation.xlsx')
label_dir = os.path.join(ROOT_DIR, 'GHA-SCW-Datasets', 'Label')
base_dir = os.path.join(ROOT_DIR, 'dataset-for-training')
save_path = os.path.join(base_dir, 'saved_dataset_unet.npz')
output_dir = os.path.join(base_dir, 'output_unet')
os.makedirs(output_dir, exist_ok=True)

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
# 2. U-Net Optical Flow 模块（内联，无需外部 import）
# ============================================================================

class DoubleConv(nn.Module):
    """Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """DoubleConv + stride-2 downsampling"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        skip = self.conv(x)
        down = self.down(skip)
        return down, skip


class DecoderBlock(nn.Module):
    """Upsample -> concat skip -> DoubleConv"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNetOpticalFlow(nn.Module):
    """
    U-Net for dense optical flow estimation.
    Input:  (B, 2, H, W)  -- two consecutive frames concatenated
    Output: (B, 2, H, W)  -- (u, v) flow field
    """
    def __init__(self, base_ch=64):
        super().__init__()
        self.enc1 = EncoderBlock(2, base_ch)
        self.enc2 = EncoderBlock(base_ch, base_ch * 2)
        self.enc3 = EncoderBlock(base_ch * 2, base_ch * 4)
        self.enc4 = EncoderBlock(base_ch * 4, base_ch * 8)
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)
        self.dec4 = DecoderBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.dec3 = DecoderBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec1 = DecoderBlock(base_ch * 2, base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 2, kernel_size=3, padding=1)

    def forward(self, x):
        d1, s1 = self.enc1(x)
        d2, s2 = self.enc2(d1)
        d3, s3 = self.enc3(d2)
        d4, s4 = self.enc4(d3)
        bn = self.bottleneck(d4)
        u4 = self.dec4(bn, s4)
        u3 = self.dec3(u4, s3)
        u2 = self.dec2(u3, s2)
        u1 = self.dec1(u2, s1)
        return self.out_conv(u1)


# ── Synthetic Flow Data Generator ──

def generate_random_flow_field(h, w, max_displacement=20.0):
    """Generate smooth random flow field using low-frequency sine waves."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = np.zeros((h, w), dtype=np.float32)
    v = np.zeros((h, w), dtype=np.float32)
    n_components = random.randint(3, 8)
    for _ in range(n_components):
        freq_x = random.uniform(0.5, 3.0) * np.pi / w
        freq_y = random.uniform(0.5, 3.0) * np.pi / h
        amp_u = random.uniform(0.3, 1.0) * max_displacement / n_components
        amp_v = random.uniform(0.3, 1.0) * max_displacement / n_components
        phase_x = random.uniform(0, 2 * np.pi)
        phase_y = random.uniform(0, 2 * np.pi)
        u += amp_u * np.sin(freq_y * yy + phase_y) * np.cos(freq_x * xx + phase_x)
        v += amp_v * np.cos(freq_y * yy + phase_y) * np.sin(freq_x * xx + phase_x)
    u += random.uniform(-max_displacement * 0.3, max_displacement * 0.3)
    v += random.uniform(-max_displacement * 0.3, max_displacement * 0.3)
    return np.stack([u, v], axis=-1).astype(np.float32)


def warp_image(img, flow):
    """Warp image using backward flow with bilinear sampling."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    src_x = np.clip(xx - flow[:, :, 0], 0, w - 1)
    src_y = np.clip(yy - flow[:, :, 1], 0, h - 1)
    x0 = np.floor(src_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(src_y).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = src_x - x0.astype(np.float32)
    wy = src_y - y0.astype(np.float32)
    warped = (
        (1 - wx) * (1 - wy) * img[y0, x0] +
        wx * (1 - wy) * img[y0, x1] +
        (1 - wx) * wy * img[y1, x0] +
        wx * wy * img[y1, x1]
    )
    return warped.astype(np.float32)


def generate_synthetic_batch(images, batch_size=16, img_size=(400, 400),
                              max_displacement=20.0, device='cpu'):
    """Generate batch of (img1, img2, gt_flow) from random warps."""
    img1_list, img2_list, flow_list = [], [], []
    for _ in range(batch_size):
        idx = random.randint(0, len(images) - 1)
        img = images[idx].astype(np.float32)
        if img.shape != img_size:
            img = np.array(Image.fromarray(img).resize(
                (img_size[1], img_size[0]), Image.Resampling.LANCZOS))
        flow = generate_random_flow_field(img_size[0], img_size[1], max_displacement)
        warped = warp_image(img, flow)
        img1_list.append(img)
        img2_list.append(warped)
        flow_list.append(flow)
    img1_batch = torch.from_numpy(np.stack(img1_list)).unsqueeze(1).to(device)
    img2_batch = torch.from_numpy(np.stack(img2_list)).unsqueeze(1).to(device)
    flow_batch = torch.from_numpy(np.stack(flow_list)).permute(0, 3, 1, 2).to(device)
    return img1_batch, img2_batch, flow_batch


# ── Loss for Optical Flow Pre-training ──

class EPEWithSmoothnessLoss(nn.Module):
    """End Point Error + Edge-aware smoothness loss."""
    def __init__(self, smoothness_weight=0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight

    def forward(self, pred_flow, gt_flow, img1):
        epe = torch.mean((pred_flow - gt_flow) ** 2)
        flow_grad_x = torch.abs(pred_flow[:, :, :, 1:] - pred_flow[:, :, :, :-1])
        flow_grad_y = torch.abs(pred_flow[:, :, 1:, :] - pred_flow[:, :, :-1, :])
        img_grad_x = torch.abs(img1[:, :, :, 1:] - img1[:, :, :, :-1])
        img_grad_y = torch.abs(img1[:, :, 1:, :] - img1[:, :, :-1, :])
        weight_x = torch.exp(-img_grad_x * 5.0)
        weight_y = torch.exp(-img_grad_y * 5.0)
        smoothness = (weight_x * flow_grad_x).mean() + (weight_y * flow_grad_y).mean()
        return epe + self.smoothness_weight * smoothness, epe.item(), smoothness.item()


# ── Pre-training Loop ──

def pretrain_unet_flow(base_date_dir, dates, save_path,
                        img_size=(400, 400), num_epochs=50, batch_size=16,
                        max_displacement=20.0, device='cuda'):
    """Pre-train UNet on synthetic flow data from real radar images."""
    print(f"=== UNet Optical Flow Pre-training ===")
    print(f"Loading radar images...")

    all_images = []
    pattern = re.compile(r'(\d{8}_\d{6})_(\d+(\.\d+)?)_50kM\.jpg', re.IGNORECASE)

    for date_str in dates:
        yyyymmdd = date_str.replace('-', '')
        date_dir = os.path.join(base_date_dir, yyyymmdd, 'radar_img')
        if not os.path.exists(date_dir):
            continue
        date_images = []
        for filename in sorted(os.listdir(date_dir)):
            match = pattern.match(filename)
            if match:
                angle = float(match.group(2))
                if abs(angle - 3.0) > 0.5:
                    continue
                img_path = os.path.join(date_dir, filename)
                img = np.array(Image.open(img_path).convert('L')) / 255.0
                if img.shape != img_size:
                    img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(
                        (img_size[1], img_size[0]), Image.Resampling.LANCZOS)) / 255.0
                date_images.append(img.astype(np.float32))
        if date_images:
            step = max(1, len(date_images) // 20)
            all_images.extend(date_images[::step])
            print(f"  {date_str}: {len(date_images)} images, kept {len(date_images[::step])}")

    print(f"Total images for synthetic generation: {len(all_images)}")

    model = UNetOpticalFlow(base_ch=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = EPEWithSmoothnessLoss(smoothness_weight=0.1)
    steps_per_epoch = 200
    model.train()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_epe, epoch_smooth = 0.0, 0.0
        for step in range(steps_per_epoch):
            img1, img2, gt_flow = generate_synthetic_batch(
                all_images, batch_size, img_size, max_displacement, device)
            pair = torch.cat([img1, img2], dim=1)
            pred_flow = model(pair)
            loss, epe, smooth = criterion(pred_flow, gt_flow, img1)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_epe += epe
            epoch_smooth += smooth
        scheduler.step()
        avg_loss = epoch_epe / steps_per_epoch + 0.1 * epoch_smooth / steps_per_epoch
        print(f"Epoch {epoch+1}/{num_epochs} | EPE: {epoch_epe/steps_per_epoch:.4f} | "
              f"Smooth: {epoch_smooth/steps_per_epoch:.4f} | Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved to {save_path}")

    print(f"Pre-training complete. Best loss: {best_loss:.4f}")
    return model


# ── UNet Optical Flow Inference ──

def compute_unet_optical_flow(model, radar_sequence, device='cuda'):
    """Compute optical flow between consecutive radar frames using UNet."""
    model.eval()
    model.to(device)
    T = radar_sequence.shape[0]
    optical_flow = []
    for t in range(T - 1):
        img1 = torch.from_numpy(radar_sequence[t]).float().unsqueeze(0).unsqueeze(0).to(device)
        img2 = torch.from_numpy(radar_sequence[t + 1]).float().unsqueeze(0).unsqueeze(0).to(device)
        pair = torch.cat([img1, img2], dim=1)
        with torch.no_grad():
            flow = model(pair)
        optical_flow.append(flow.squeeze(0).permute(1, 2, 0).cpu().numpy())
    return np.array(optical_flow, dtype=np.float32)


def get_unet_flow_model(base_dir, dates, model_path=None, device='cuda',
                         force_retrain=False):
    """Load or pre-train UNet optical flow model."""
    if model_path is None:
        model_path = os.path.join(base_dir, 'unet_flow_pretrained.pth')
    model = UNetOpticalFlow(base_ch=64).to(device)
    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading pre-trained UNet optical flow from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Pre-training UNet optical flow from scratch...")
        model = pretrain_unet_flow(base_dir, dates, model_path, device=device)
    model.eval()
    return model


# ============================================================================
# 3. 样本筛选
# ============================================================================

def sample_selection(info_excel_path, label_base_dir, time_steps=6, forecast_steps=3):

    def check_data_completeness(date_str, input_ts_list, label_ts_str, label_path):
        missing_reasons = []
        awos_path = os.path.join(base_dir, date_str, 'AWS', f"{date_str}_AWS.csv")
        if not os.path.exists(awos_path):
            missing_reasons.append(f"AWOS: missing {awos_path}")
        if not os.path.exists(label_path):
            missing_reasons.append(f"Label: missing {label_path}")
        radar_dir = os.path.join(base_dir, date_str, 'radar_img')
        if os.path.exists(radar_dir):
            for ts_str in input_ts_list:
                files = [f for f in os.listdir(radar_dir)
                         if f.startswith(ts_str + '_') and f.endswith('.jpg')]
                if len(files) < 15:
                    missing_reasons.append(f"Radar: {len(files)}/15 files for {ts_str}")
        else:
            missing_reasons.append(f"Radar: dir missing {radar_dir}")
        for ts_str in input_ts_list:
            parts = ts_str.split('_')
            if len(parts) != 2:
                continue
            timestamp_no_ss = f"{parts[0]}{parts[1][:4]}"
            for prefix, sub in zip(['IEC', 'UEC', 'WEC'], ['I', 'U', 'W']):
                sat_path = os.path.join(base_dir, date_str, 'cloud_img', sub,
                                        f"{prefix}{timestamp_no_ss}_GH4.jpg")
                if not os.path.exists(sat_path):
                    missing_reasons.append(f"Satellite {prefix}: missing")
        if missing_reasons:
            print(f"  Discarding: {date_str}, {label_ts_str} -- {'; '.join(missing_reasons)}")
            return False
        return True

    info_df = pd.read_excel(info_excel_path)
    dates = info_df['filename'].astype(str).tolist()
    print(f"Step 1: Loaded {len(dates)} dates from infomation.xlsx")

    all_positive, all_negative = [], []
    for date_str in dates:
        try:
            formatted_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
        except ValueError:
            continue
        label_path = os.path.join(label_base_dir, f"{formatted_date}.xlsx")
        if not os.path.exists(label_path):
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

    classes_list = ['雷暴', '短时强降水', '大风']
    potential_data = []
    for date, ts, lbls in all_positive:
        sample_classes = ','.join([classes_list[j] for j in range(3) if lbls[j] == 1])
        potential_data.append({'Type': 'Positive', 'Date': date,
                               'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                               'Classes': sample_classes, 'Labels': str(list(lbls))})
    for date, ts in all_negative:
        potential_data.append({'Type': 'Negative', 'Date': date,
                               'Timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                               'Classes': '', 'Labels': '[0, 0, 0]'})
    if potential_data:
        pd.DataFrame(potential_data).to_excel(
            os.path.join(output_dir, 'potential_samples.xlsx'), index=False)

    return all_positive, all_negative


# ============================================================================
# 4. 数据加载函数
# ============================================================================

def find_nearest_ts(base_dir, yyyymmdd, target_ts_str, max_delta=10):
    target_dt = datetime.strptime(target_ts_str, '%Y%m%d_%H%M%S')
    available_files = glob.glob(os.path.join(base_dir, yyyymmdd, 'radar_img', '*_50kM.jpg'))
    available_ts = []
    for f in available_files:
        parts = os.path.basename(f).split('_')
        if len(parts) >= 2:
            ts = parts[0] + '_' + parts[1]
            try:
                available_ts.append((datetime.strptime(ts, '%Y%m%d_%H%M%S'), ts))
            except ValueError:
                continue
    nearest, min_delta = None, None
    for dt, ts in available_ts:
        delta = abs((dt - target_dt).total_seconds()) / 60.0
        if delta <= max_delta and (min_delta is None or delta < min_delta):
            min_delta = delta
            nearest = ts
    return nearest


def load_radar_images(base_dir, dates_timestamps, img_size=(400, 400), num_angles=15):
    radar_data, all_timestamps = {}, {}
    pattern = re.compile(r'(\d{8}_\d{6})_(\d+(\.\d+)?)_50kM\.jpg', re.IGNORECASE)
    for date_str in dates_timestamps:
        yyyymmdd = date_str.replace('-', '')
        date_dir = os.path.join(base_dir, yyyymmdd, 'radar_img')
        if not os.path.exists(date_dir):
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
                img = np.array(Image.open(os.path.join(date_dir, filename)).convert('L')) / 255.0
                date_radar[timestamp][int_angle] = img[:, :, np.newaxis]
        sorted_ts = sorted(date_radar.keys())
        radar_data[date_str] = np.array([date_radar[ts] for ts in sorted_ts])
        all_timestamps[date_str] = sorted_ts
        print(f"Radar {date_str}: {len(sorted_ts)} timestamps")
    return radar_data, all_timestamps


def load_satellite_images(base_dir, dates_timestamps, img_size=(200, 200)):
    satellite_data, all_timestamps = {}, {}
    pattern = re.compile(r'(IEC|UEC|WEC)(\d{8}\d{4})_GH4\.jpg', re.IGNORECASE)
    channel_map = {'IEC': 0, 'UEC': 1, 'WEC': 2}
    subdirs = {'IEC': 'I', 'UEC': 'U', 'WEC': 'W'}
    for date_str in dates_timestamps:
        date_dir_str = date_str.replace('-', '')
        date_dir = os.path.join(base_dir, date_dir_str, 'cloud_img')
        if not os.path.exists(date_dir):
            continue
        date_satellite_dict, date_ts_list = {}, []
        for channel_prefix, subdir in subdirs.items():
            sub_dir = os.path.join(date_dir, subdir)
            if not os.path.exists(sub_dir):
                continue
            for filename in sorted(os.listdir(sub_dir)):
                match = pattern.match(filename)
                if not match or match.group(1) != channel_prefix:
                    continue
                try:
                    dt = datetime.strptime(match.group(2), '%Y%m%d%H%M')
                except ValueError:
                    continue
                full_ts = dt.strftime('%Y%m%d_%H%M%S')
                if full_ts not in dates_timestamps.get(date_str, []):
                    continue
                try:
                    img = Image.open(os.path.join(sub_dir, filename)).convert('L')
                    img = img.resize((img_size[1], img_size[0]), Image.Resampling.LANCZOS)
                    img_array = np.array(img) / 255.0
                except Exception:
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
            satellite_data[date_str] = np.array([date_satellite_dict[ts] for ts in sorted_ts])
            all_timestamps[date_str] = sorted_ts
            print(f"Satellite {date_str}: {len(sorted_ts)} timestamps")
    return satellite_data, all_timestamps


def load_awos_data(base_dir, dates_timestamps):
    awos_data, all_timestamps = {}, {}
    features = ['10分风向', '10分风速', '气压', '温度', '湿度',
                '24h_变温度', '3h_变温度', '24h_变气压', '3h_变气压']
    scaler = StandardScaler()
    for date_str in dates_timestamps:
        yyyymmdd = date_str.replace('-', '')
        awos_path = os.path.join(base_dir, yyyymmdd, 'AWS', f"{yyyymmdd}_AWS.csv")
        if not os.path.exists(awos_path):
            continue
        try:
            df = pd.read_csv(awos_path, encoding='gb18030', sep=',')
        except Exception:
            continue
        required_cols = features + ['datetime']
        if any(c not in df.columns for c in required_cols):
            continue
        df = df.dropna(subset=required_cols)
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
        df_filtered = df[df['timestamp'].isin(required_ts)]
        if df_filtered.empty:
            continue
        awos_values = scaler.fit_transform(df_filtered[features].values.astype(np.float32))
        awos_data[date_str] = awos_values
        all_timestamps[date_str] = df_filtered['timestamp'].tolist()
        print(f"AWOS {date_str}: {len(df_filtered)} rows")
    return awos_data, all_timestamps


def load_labels(label_base_dir, dates_timestamps):
    labels_data, all_timestamps = {}, {}
    for date_str in dates_timestamps:
        label_path = os.path.join(label_base_dir, f"{date_str}.xlsx")
        if not os.path.exists(label_path):
            continue
        df = pd.read_excel(label_path)
        df['timestamp'] = pd.to_datetime(df['时间(LT)']).apply(
            lambda dt: dt.strftime('%Y%m%d_%H%M%S'))
        required_ts = set(dates_timestamps[date_str])
        df_filtered = df[df['timestamp'].isin(required_ts)]
        if df_filtered.empty:
            continue
        labels_data[date_str] = df_filtered[['雷暴', '短时强降水', '大风']].values.astype(np.float32)
        all_timestamps[date_str] = df_filtered['timestamp'].tolist()
        print(f"Labels {date_str}: {len(df_filtered)} rows")
    return labels_data, all_timestamps


# ============================================================================
# 5. 样本构建（UNet 光流替换 TV-L1）
# ============================================================================

def create_samples(radar_data, satellite_data, awos_data, labels_data,
                   samples_list, all_timestamps,
                   optical_flow_model=None, time_steps=6, forecast_steps=3):
    samples = {'radar': [], 'satellite': [], 'awos': [], 'optical_flow': [],
               'labels': [], 'indices': [], 'timestamps': []}
    skipped_count = 0
    flow_device = next(optical_flow_model.parameters()).device if optical_flow_model else 'cpu'

    for idx, (date, label_ts, substituted) in enumerate(samples_list):
        try:
            input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i) for i in range(time_steps)]
            input_str_list = [ts.strftime('%Y%m%d_%H%M%S') for ts in input_ts_list]
            label_str = label_ts.strftime('%Y%m%d_%H%M%S')
            for i, ts_str in enumerate(input_str_list):
                if ts_str in substituted:
                    input_str_list[i] = substituted[ts_str]
            if label_str in substituted:
                label_str = substituted[label_str]

            date_radar = radar_data.get(date, np.array([]))
            date_satellite = satellite_data.get(date, np.array([]))
            date_awos = awos_data.get(date, np.array([]))
            date_labels = labels_data.get(date, np.array([]))
            date_ts = all_timestamps.get(date, [])
            if not date_ts or date_radar.size == 0:
                raise ValueError("No data")

            input_indices = [date_ts.index(ts_str) for ts_str in input_str_list if ts_str in date_ts]
            label_idx = date_ts.index(label_str) if label_str in date_ts else -1
            if len(input_indices) != time_steps or label_idx == -1:
                raise ValueError(f"Incomplete: input {len(input_indices)}/{time_steps}")
            if max(input_indices) >= len(date_radar) or label_idx >= len(date_labels):
                raise IndexError("Index out of range")

            # ★ UNet optical flow (replaces TV-L1)
            radar_3rd = date_radar[input_indices, 2, :, :, 0]
            if optical_flow_model is not None:
                optical_flow = compute_unet_optical_flow(optical_flow_model, radar_3rd, device=flow_device)
            else:
                optical_flow = np.zeros((time_steps - 1, radar_3rd.shape[1], radar_3rd.shape[2], 2), dtype=np.float32)
            if len(optical_flow) < time_steps - 1:
                pad = time_steps - 1 - len(optical_flow)
                optical_flow = np.concatenate([optical_flow, np.tile(optical_flow[-1:], (pad, 1, 1, 1))], axis=0)

            samples['radar'].append(date_radar[input_indices])
            samples['satellite'].append(date_satellite[input_indices])
            samples['awos'].append(date_awos[input_indices])
            samples['optical_flow'].append(optical_flow)
            samples['labels'].append(date_labels[label_idx])
            samples['timestamps'].append(label_str)
            samples['indices'].append(idx)
        except Exception as e:
            print(f"Warning: sample {idx} on {date} skipped: {e}")
            skipped_count += 1

    for key in samples:
        samples[key] = np.array(samples[key])
    print(f"Created {len(samples['indices'])} samples, skipped {skipped_count}")
    return samples


# ============================================================================
# 6. 数据增强
# ============================================================================

def augment_rare(samples, rare_multiplier=3):
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
        if lbl[1] or lbl[2]:
            for _ in range(rare_multiplier - 1):
                radar = samples['radar'][i]
                aug_radar = np.zeros_like(radar)
                for t in range(radar.shape[0]):
                    for a in range(radar.shape[1]):
                        img_tensor = torch.from_numpy(radar[t, a]).permute(2, 0, 1)
                        if img_tensor.shape[0] == 1:
                            img_tensor = img_tensor.repeat(3, 1, 1)
                        aug_tensor = transform(img_tensor)
                        if aug_tensor.shape[0] == 3:
                            aug_tensor = aug_tensor.mean(dim=0, keepdim=True)
                        aug_radar[t, a] = aug_tensor.permute(1, 2, 0).numpy()
                satellite = samples['satellite'][i]
                aug_satellite = np.zeros_like(satellite)
                for t in range(satellite.shape[0]):
                    img_tensor = torch.from_numpy(satellite[t]).permute(2, 0, 1)
                    aug_satellite[t] = transform(img_tensor).permute(1, 2, 0).numpy()
                augmented['radar'].append(aug_radar)
                augmented['satellite'].append(aug_satellite)
                augmented['awos'].append(samples['awos'][i] + np.random.normal(0, 0.01, samples['awos'][i].shape))
                augmented['optical_flow'].append(samples['optical_flow'][i])
                augmented['labels'].append(samples['labels'][i])
                augmented['indices'].append(samples['indices'][i])
                augmented['timestamps'].append(samples['timestamps'][i])
    for k in augmented:
        augmented[k] = np.array(augmented[k])
    print(f"After augmentation: {len(augmented['labels'])} (was {len(samples['labels'])})")
    return augmented


# ============================================================================
# 7. 主数据加载（含 UNet 预训练）
# ============================================================================

def load_data():
    global save_path

    if os.path.exists(save_path):
        print(f"Loading cached dataset from {save_path}")
        loaded = np.load(save_path, allow_pickle=True)
        samples = {key: loaded[key] for key in loaded.files}
        print(f"Loaded {len(samples['indices'])} samples")
        return samples

    all_positive, all_negative = sample_selection(info_excel, label_dir)
    print(f"Candidate positive: {len(all_positive)}, negative: {len(all_negative)}")
    save_sample_names(all_positive, all_negative, output_dir)

    all_samples_list = [(date, target_ts, {}) for date, target_ts, _ in all_positive] + \
                       [(date, target_ts, {}) for date, target_ts in all_negative]
    dates_timestamps = {}
    for date, label_ts, _ in all_samples_list:
        input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i) for i in range(time_steps)]
        required_str = [ts.strftime('%Y%m%d_%H%M%S') for ts in input_ts_list + [label_ts]]
        dates_timestamps.setdefault(date, []).extend(required_str)
        dates_timestamps[date] = sorted(set(dates_timestamps[date]))

    radar_data, radar_ts = load_radar_images(base_dir, dates_timestamps)

    def extract_radar_features(date, label_ts, radar_data, radar_ts, time_steps=6):
        try:
            input_ts_list = [label_ts - timedelta(minutes=90 - 10 * i) for i in range(time_steps)]
            input_str_list = [ts.strftime('%Y%m%d_%H%M%S') for ts in input_ts_list]
            date_ts_list = radar_ts.get(date, [])
            if not date_ts_list:
                return None
            input_indices = [date_ts_list.index(ts_str) for ts_str in input_str_list if ts_str in date_ts_list]
            if len(input_indices) != time_steps:
                return None
            radar_seq = radar_data[date][input_indices]
            features = []
            for t in range(time_steps):
                slice_t = radar_seq[t].flatten()
                features.extend([np.mean(slice_t), np.var(slice_t)])
            return np.array(features)
        except Exception:
            return None

    pos_features, pos_keys = [], []
    for date, label_ts, _ in all_positive:
        feat = extract_radar_features(date, label_ts, radar_data, radar_ts)
        if feat is not None:
            pos_features.append(feat)
            pos_keys.append((date, label_ts))
    neg_features, neg_keys = [], []
    for date, target_ts in all_negative:
        feat = extract_radar_features(date, target_ts, radar_data, radar_ts)
        if feat is not None:
            neg_features.append(feat)
            neg_keys.append((date, target_ts))

    pos_features = np.array(pos_features)
    neg_features = np.array(neg_features)
    print(f"Extracted features -- positive: {len(pos_features)}, negative: {len(neg_features)}")

    if len(pos_features) > 0 and len(neg_features) > 0:
        similarities = cosine_similarity(neg_features, pos_features).mean(axis=1)
        sorted_indices = np.argsort(similarities)[::-1]
        num_delete = int(len(neg_features) * 0.3)
        all_negative = [neg_keys[i] for i in sorted_indices[num_delete:]]
        print(f"Hard negative filtering: {len(neg_keys)} -> {len(all_negative)}")

    manual_neg_path = os.path.join(base_dir, 'negative_samples0.xlsx')
    if os.path.exists(manual_neg_path):
        try:
            df_manual = pd.read_excel(manual_neg_path)
            all_negative = []
            for _, row in df_manual.iterrows():
                dt = datetime.strptime(row['Timestamp'], '%Y-%m-%d %H:%M:%S')
                all_negative.append((dt.strftime('%Y-%m-%d'), dt))
            print(f"Using manual negatives: {len(all_negative)}")
        except Exception as e:
            print(f"Warning: failed to read manual negatives: {e}")

    del radar_data, radar_ts
    gc.collect()

    rare_positive = [c for c in all_positive if c[2][1] == 1 or c[2][2] == 1]
    other_positive = [c for c in all_positive if c not in rare_positive]
    print(f"Rare positive: {len(rare_positive)}, Other positive: {len(other_positive)}")

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
                radar_pattern = os.path.join(base_dir, yyyymmdd, 'radar_img',
                                             f'{ts_str[:8]}_{ts_str[9:15]}_*_50kM.jpg')
                if len(glob.glob(radar_pattern)) == 0:
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

    target_negative = int(len(valid_positive) * negative_ratio)
    valid_negative = []
    candidates_neg = list(all_negative)
    random.shuffle(candidates_neg)
    for candidate in candidates_neg:
        if len(valid_negative) >= target_negative:
            break
        valid_negative.append(candidate)

    samples_list = [(date, label_ts, substituted_timestamps.get(
        (date, label_ts.strftime('%Y%m%d_%H%M%S')), {}))
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

    # ★ Pre-train / load UNet flow model
    unet_model_path = os.path.join(base_dir, 'unet_flow_pretrained.pth')
    print("\n" + "=" * 60)
    print("UNet Optical Flow Model Setup")
    print("=" * 60)
    optical_flow_model = get_unet_flow_model(
        base_dir=base_dir, dates=list(dates_timestamps.keys()),
        model_path=unet_model_path, device=str(device), force_retrain=False)
    print("=" * 60 + "\n")

    radar_data, radar_ts = load_radar_images(base_dir, dates_timestamps)
    satellite_data, satellite_ts = load_satellite_images(base_dir, dates_timestamps)
    awos_data, awos_ts = load_awos_data(base_dir, dates_timestamps)
    labels_data, labels_ts = load_labels(label_dir, dates_timestamps)

    samples = create_samples(radar_data, satellite_data, awos_data, labels_data,
                             samples_list, radar_ts, optical_flow_model=optical_flow_model)
    samples = augment_rare(samples, rare_multiplier=3)

    rare_count = sum(1 for lbl in samples['labels'] if any(lbl[1:]))
    print(f"Final stats -- Rare: {rare_count}, Wind: {sum(1 for l in samples['labels'] if l[2])}, "
          f"Rain: {sum(1 for l in samples['labels'] if l[1])}")

    if len(samples['indices']) > 0:
        print(f"Saving dataset to {save_path}")
        np.savez(save_path, **samples)
    return samples


# ============================================================================
# 8. 模型架构
# ============================================================================

class TwoDCNN(nn.Module):
    def __init__(self, input_channels, feature_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        h = w = 400 if input_channels in [15, 2] else 200
        h, w = h // 4, w // 4
        self.fc = nn.Linear(64 * h * w, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        return self.fc(x.reshape(x.size(0), -1))


class ConvLSTM2d(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, num_layers=1, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_size + hidden_size, 4 * hidden_size,
                              kernel_size=self.kernel_size, padding=self.padding, bias=True)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        B, T, C, H, W = input_tensor.size()
        if hidden_state is None:
            h_t = torch.zeros(self.num_layers, B, self.hidden_size, H, W, device=input_tensor.device)
            c_t = torch.zeros(self.num_layers, B, self.hidden_size, H, W, device=input_tensor.device)
        else:
            h_t, c_t = hidden_state
        h_new, c_new = torch.zeros_like(h_t), torch.zeros_like(c_t)
        layer_output = []
        for layer in range(self.num_layers):
            hl, cl = h_t[layer].clone(), c_t[layer].clone()
            out_inner = []
            for t in range(T):
                gates = self.conv(torch.cat((input_tensor[:, t], hl), dim=1))
                i_g, f_g, c_g, o_g = gates.chunk(4, 1)
                cl = torch.sigmoid(f_g) * cl + torch.sigmoid(i_g) * torch.tanh(c_g)
                hl = torch.sigmoid(o_g) * torch.tanh(cl)
                out_inner.append(hl.clone())
            layer_output.append(torch.stack(out_inner, dim=1))
            h_new[layer], c_new[layer] = hl.clone(), cl.clone()
        return torch.stack(layer_output, dim=0)[-1], (h_new, c_new)


class WeatherModel(nn.Module):
    def __init__(self, time_steps=6, use_optical_flow=False):
        super().__init__()
        self.time_steps = time_steps
        self.use_optical_flow = use_optical_flow
        self.radar_cnn = TwoDCNN(15, 64)
        self.satellite_cnn = TwoDCNN(3, 64)
        self.conv_lstm = ConvLSTM2d(64, 64, kernel_size=3, num_layers=1, batch_first=True)
        self.fc_awos = nn.Linear(9, 16)
        input_dim = 64 + 64 + 16
        if use_optical_flow:
            self.flow_cnn = TwoDCNN(2, 64)
            input_dim += 64
        self.fc = nn.Linear(input_dim, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, radar_data, satellite_data, awos_data, optical_flow=None):
        B = radar_data.size(0)
        rf = torch.stack([self.radar_cnn(radar_data[:, t].squeeze(-1)) for t in range(self.time_steps)], dim=1)
        sf = torch.stack([self.satellite_cnn(satellite_data[:, t].permute(0, 3, 1, 2)) for t in range(self.time_steps)], dim=1)
        _, (hn, _) = self.conv_lstm(rf.unsqueeze(-1).unsqueeze(-1))
        radar_out = hn[-1].view(B, -1)
        _, (hn, _) = self.conv_lstm(sf.unsqueeze(-1).unsqueeze(-1))
        sat_out = hn[-1].view(B, -1)
        awos_out = self.relu(self.fc_awos(awos_data[:, -1]))
        if self.use_optical_flow and optical_flow is not None:
            ff = torch.stack([self.flow_cnn(optical_flow[:, t].permute(0, 3, 1, 2) if optical_flow[:, t].size(1) != 2 else optical_flow[:, t])
                              for t in range(min(optical_flow.size(1), self.time_steps))], dim=1)
            _, (hn, _) = self.conv_lstm(ff.unsqueeze(-1).unsqueeze(-1))
            features = torch.cat([radar_out, sat_out, awos_out, hn[-1].view(B, -1)], dim=1)
        else:
            features = torch.cat([radar_out, sat_out, awos_out], dim=1)
        return self.sigmoid(self.fc(features))


# ============================================================================
# 9. 损失函数
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, device=None):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
        self.gamma = gamma
        self.device = device

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs.to(self.device), targets.to(self.device), reduction='none')
        pt = torch.exp(-BCE_loss)
        return (self.alpha[None, :] * (1 - pt) ** self.gamma * BCE_loss).mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha, gamma, focal_weight=0.7, dice_weight=0.3, smooth=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def dice_loss(self, inputs, targets):
        intersection = (inputs * targets).sum(dim=0)
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=0) + targets.sum(dim=0) + self.smooth)
        return 1 - dice.mean()

    def forward(self, inputs, targets):
        return self.focal_weight * self.focal(inputs, targets) + self.dice_weight * self.dice_loss(inputs, targets)


# ============================================================================
# 10. 评估指标（★ 修正 TS 为多标签二分类版本）
# ============================================================================

def compute_ts_score(y_true, y_pred, class_idx=None):
    """★ Corrected TS for multi-label: per-class binary confusion matrix."""
    n_classes = y_true.shape[1]
    ts_scores = []
    for i in range(n_classes):
        if class_idx is not None and i != class_idx:
            continue
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        if cm.shape[0] > 1 and cm.shape[1] > 1:
            tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
        else:
            tp = fp = fn = 0
        ts_scores.append(tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0)
    return np.mean(ts_scores) if ts_scores else 0.0


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics['accuracy'] = np.mean([accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
    metrics['f1_macro'] = sk_f1_score(y_true, y_pred, average='macro', zero_division=0)
    aucs = [roc_auc_score(y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1])
            if y_true[:, i].sum() > 0 and len(np.unique(y_true[:, i])) > 1]
    metrics['roc_auc'] = np.mean(aucs) if aucs else 0.0
    metrics['ts'] = compute_ts_score(y_true, y_pred)
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
# 11. 训练函数
# ============================================================================

def train_model(model, train_loader, val_loader, timestamp_list, model_name='model',
                lr=0.002, alpha=None, gamma=5, num_epochs=100, patience=20,
                trial_id=None, is_final=False, focal_weight=0.7):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_dir, 'output.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_labels = np.concatenate([batch[4].numpy() for batch in train_loader])
    label_dist = np.mean(train_labels, axis=0)
    if alpha is None:
        alpha = [1 / max(d, 1e-6) for d in label_dist]
        alpha = [a / sum(alpha) for a in alpha]
    print(f"Dynamic alpha: {alpha}")
    criterion = CombinedLoss(alpha=alpha, gamma=gamma, focal_weight=focal_weight)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    suffix = '_final' if is_final else f'_trial_{trial_id}' if trial_id is not None else ''
    model_file = os.path.join(output_dir, f'model_{model_name}{suffix}.pth')
    metrics_file = os.path.join(output_dir, f'metrics_{model_name}{suffix}.csv')
    error_file = os.path.join(output_dir, f'errors_{model_name}{suffix}.csv')

    if os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} already exists, skipping")
        m = pd.read_csv(metrics_file).to_dict('records')[0]
        e = pd.read_csv(error_file).to_dict('records') if os.path.exists(error_file) else []
        return m, e

    best_ts, best_metrics, errors, patience_counter = 0, {}, [], 0
    epoch_records, train_ll, val_ll, lr_h = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            radar_data, satellite_data, awos_data, optical_flow, labels, indices, _ = [x.to(device) for x in batch]
            optimizer.zero_grad()
            loss = criterion(model(radar_data, satellite_data, awos_data, optical_flow), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()
        train_ll.append(train_loss)
        lr_h.append(optimizer.param_groups[0]['lr'])

        model.eval()
        val_loss = 0
        y_true, y_pred, y_prob, y_errors = [], [], [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                radar_data, satellite_data, awos_data, optical_flow, labels, indices, batch_indices = \
                    [x.to(device) for x in batch]
                outputs = model(radar_data, satellite_data, awos_data, optical_flow)
                val_loss += criterion(outputs, labels).item()
                preds = (outputs > 0.5).float()
                y_errors.extend([(idx.item(), pred.tolist(), label.tolist(), timestamp_list[batch_idx.item()])
                                 for idx, pred, label, batch_idx in zip(indices, preds, labels, batch_indices)])
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(outputs.cpu().numpy())
        val_loss /= len(val_loader)
        val_ll.append(val_loss)
        yt, yp, ypr = np.array(y_true), np.array(y_pred), np.array(y_prob)
        metrics = compute_metrics(yt, yp, ypr)
        metrics['val_loss'] = val_loss
        print(f"Epoch {epoch+1}: TS={metrics['ts']:.4f}, POD={metrics['pod']:.4f}, FAR={metrics['far']:.4f}")
        epoch_records.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                              'accuracy': metrics['accuracy'], 'f1_macro': metrics['f1_macro'],
                              'roc_auc': metrics['roc_auc'], 'ts': metrics['ts'],
                              'pod': metrics['pod'], 'far': metrics['far'], 'csi': metrics['csi']})
        if metrics['ts'] > best_ts:
            best_ts, best_metrics, errors = metrics['ts'], metrics.copy(), y_errors
            torch.save({'state_dict': model.state_dict(), 'train_loss': train_ll,
                        'val_loss': val_ll, 'lr_history': lr_h}, model_file)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    pd.DataFrame(epoch_records).to_csv(metrics_file, index=False)
    pd.DataFrame(errors, columns=['index', 'prediction', 'label', 'timestamp']).to_csv(error_file, index=False)
    print(f"Best metrics for {model_name}: {best_metrics}")
    return best_metrics, errors


# ============================================================================
# 12. 超参数调优
# ============================================================================

def hyperparameter_tune(train_loader, val_loader, timestamp_list, model_name='model'):
    param_grid = {
        'lr': [0.0001, 0.0005, 0.001],
        'alpha': [[0.8, 0.9, 0.95], [0.85, 0.95, 0.98], [0.85, 0.95, 1.0]],
        'gamma': [3, 4, 5],
        'num_epochs': [50], 'focal_weight': [0.6, 0.7, 0.8], 'patience': [50]
    }
    default_params = {'lr': 0.0005, 'alpha': [0.8, 0.95, 1.0], 'gamma': 3,
                      'focal_weight': 0.7, 'num_epochs': 50, 'patience': 50}
    results = []
    params_file = os.path.join(output_dir, f'trial_params_{model_name}.csv')
    saved_params = {}
    if os.path.exists(params_file):
        for _, row in pd.read_csv(params_file).iterrows():
            saved_params[int(row['trial'])] = {k: row[k] for k in default_params}
            saved_params[int(row['trial'])]['alpha'] = eval(row['alpha'])

    all_combinations = list(itertools.product(*param_grid.values()))
    for trial_idx, combo in enumerate(all_combinations):
        trial_id = trial_idx + 1
        metrics_file = os.path.join(output_dir, f'metrics_{model_name}_trial_{trial_id}.csv')
        if os.path.exists(metrics_file):
            mdf = pd.read_csv(metrics_file)
            results.append({'trial': trial_id, **saved_params.get(trial_id, {}),
                            'accuracy': mdf['accuracy'].max(), 'f1_macro': mdf['f1_macro'].max(),
                            'roc_auc': mdf['roc_auc'].max(), 'ts': mdf['ts'].max()})
            continue
        params = dict(zip(param_grid.keys(), combo))
        print(f"\nTrial {trial_id}/{len(all_combinations)}: {params}")
        pd.DataFrame([{'trial': trial_id, **{k: str(v) for k, v in params.items()}}]).to_csv(
            params_file, mode='a', header=not os.path.exists(params_file), index=False)
        model = WeatherModel(use_optical_flow=(model_name == 'flow')).to(device)
        model.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, (nn.Conv2d, nn.Linear)) else None)
        try:
            bm, _ = train_model(model, train_loader, val_loader, timestamp_list, model_name,
                                lr=params['lr'], alpha=params['alpha'], gamma=params['gamma'],
                                num_epochs=params['num_epochs'], patience=params['patience'],
                                trial_id=trial_id, focal_weight=params['focal_weight'])
            results.append({'trial': trial_id, **params, **bm})
        except Exception as e:
            print(f"Trial {trial_id} failed: {e}")

    results_file = os.path.join(output_dir, f'hyperparam_results_{model_name}.csv')
    pd.DataFrame(results).to_csv(results_file, index=False)
    if results:
        best = max(results, key=lambda x: x['ts'])
        print(f"\nBest trial for {model_name}: {best}")
        return best
    return default_params


# ============================================================================
# 13. 分析与可视化
# ============================================================================

def predict_fn(model, data):
    model.eval()
    with torch.no_grad():
        radar, satellite, awos = data[0].to(device), data[1].to(device), data[2].to(device)
        flow = data[3].to(device) if len(data) == 4 and model.use_optical_flow else None
        outputs = model(radar, satellite, awos, flow) if flow is not None else model(radar, satellite, awos)
    return outputs.cpu().numpy()


def analyze_features_and_errors(model_flow, model_no_flow, val_data, val_labels,
                                val_timestamps, checkpoint_flow, checkpoint_no_flow):
    # Loss/LR curves
    for ckpt, name in [(checkpoint_flow, 'flow'), (checkpoint_no_flow, 'no_flow')]:
        tl, vl = ckpt.get('train_loss', []), ckpt.get('val_loss', [])
        if tl and vl:
            epochs = np.arange(1, len(tl) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, tl, label='Train'); plt.plot(epochs, vl, label='Val')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Loss Curve ({name})')
            plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{name}_loss_curve.png')); plt.close()

    val_labels_np = np.array(val_labels)
    classes = ['thunderstorm', 'heavy_rain', 'strong_wind']
    val_data_flow = [v.clone().detach().cpu() if isinstance(v, torch.Tensor) else v for v in val_data]
    val_data_no_flow = val_data_flow[:3]

    preds_f = predict_fn(model_flow, val_data_flow)
    preds_nf = predict_fn(model_no_flow, val_data_no_flow)
    preds_f_bin = (preds_f > 0.5).astype(int)
    preds_nf_bin = (preds_nf > 0.5).astype(int)

    metrics_flow = compute_metrics(val_labels_np, preds_f_bin, preds_f)
    metrics_no_flow = compute_metrics(val_labels_np, preds_nf_bin, preds_nf)

    print(f"\n{'='*60}\nModel Performance Comparison\n{'='*60}")
    for name, m in [('With UNet Flow', metrics_flow), ('Without Flow', metrics_no_flow)]:
        print(f"\n{name}:")
        for k in ['accuracy', 'f1_macro', 'roc_auc', 'ts', 'pod', 'far', 'csi']:
            print(f"  {k}: {m[k]:.4f}")

    # ★ Feature Importance (Permutation)
    def custom_permutation_importance(model, data_list, y_true, features, n_repeats=5):
        base_prob = predict_fn(model, data_list)
        base_score = compute_ts_score(y_true, (base_prob > 0.5).astype(int))
        importances = np.zeros((len(features), n_repeats))
        for i in range(len(features)):
            for r in range(n_repeats):
                shuf = [d.clone() for d in data_list]
                idx = torch.randperm(shuf[i].shape[0])
                shuf[i] = shuf[i][idx]
                shuf_prob = predict_fn(model, shuf)
                importances[i, r] = base_score - compute_ts_score(y_true, (shuf_prob > 0.5).astype(int))
        return {'importances_mean': np.mean(importances, axis=1), 'importances_std': np.std(importances, axis=1)}

    try:
        for model, data, features, label in [
            (model_flow, val_data_flow, ['Radar', 'Satellite', 'AWOS', 'Optical Flow'], 'flow'),
            (model_no_flow, val_data_no_flow, ['Radar', 'Satellite', 'AWOS'], 'no_flow')
        ]:
            imp = custom_permutation_importance(model, data, val_labels_np, features)
            print(f"\nFeature Importance ({label}):")
            for i, (m, s) in enumerate(zip(imp['importances_mean'], imp['importances_std'])):
                print(f"  {features[i]}: {m:.4f} +/- {s:.4f}")
            sorted_idx = np.argsort(imp['importances_mean'])
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(sorted_idx)), imp['importances_mean'][sorted_idx], xerr=imp['importances_std'][sorted_idx])
            plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
            plt.xlabel('Importance (Mean TS Decrease)')
            plt.title(f'Feature Importance -- {label}')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'feature_importance_{label}.png')); plt.close()
            pd.DataFrame({'feature': features, 'mean': imp['importances_mean'],
                          'std': imp['importances_std']}).to_csv(
                os.path.join(output_dir, f'feature_importance_{label}.csv'), index=False)
    except Exception as e:
        print(f"Feature importance failed: {e}")

    # Confusion matrices
    for mode, preds_bin in [('flow', preds_f_bin), ('no_flow', preds_nf_bin)]:
        for i, cls in enumerate(classes):
            cm = confusion_matrix(val_labels_np[:, i], preds_bin[:, i])
            plt.figure(figsize=(6, 4))
            plt.imshow(cm, cmap='Blues'); plt.colorbar()
            plt.title(f'CM ({mode} - {cls})')
            for x in range(2):
                for y in range(2):
                    plt.text(y, x, str(cm[x, y]), ha='center', color='white' if cm[x, y] > cm.max()/2 else 'black')
            plt.savefig(os.path.join(output_dir, f'cm_{mode}_{cls}.png')); plt.close()

    # ROC curves
    for mode, preds in [('flow', preds_f), ('no_flow', preds_nf)]:
        plt.figure(figsize=(8, 6))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(val_labels_np[:, i], preds[:, i])
            plt.plot(fpr, tpr, label=f'{cls} (AUC={auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--'); plt.xlim([0, 1]); plt.ylim([0, 1.05])
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC ({mode})')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'roc_curve_{mode}.png')); plt.close()

    # PR curves
    for mode, preds in [('flow', preds_f), ('no_flow', preds_nf)]:
        plt.figure(figsize=(8, 6))
        for i, cls in enumerate(classes):
            pr, rc, _ = precision_recall_curve(val_labels_np[:, i], preds[:, i])
            plt.step(rc, pr, where='post', label=f'{cls} (AP={average_precision_score(val_labels_np[:, i], preds[:, i]):.2f})')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR Curve ({mode})')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'pr_curve_{mode}.png')); plt.close()

    # Per-class comparison bar plots
    for metric_name in ['pod', 'far', 'csi', 'ts']:
        plt.figure(figsize=(8, 6))
        x = np.arange(len(classes))
        w = 0.35
        flow_vals, no_flow_vals = [], []
        for i in range(len(classes)):
            for preds_bin, vals in [(preds_f_bin, flow_vals), (preds_nf_bin, no_flow_vals)]:
                cm = confusion_matrix(val_labels_np[:, i], preds_bin[:, i])
                tp = cm[1, 1] if cm.shape[0] > 1 else 0
                fp = cm[0, 1] if cm.shape[0] > 1 else 0
                fn = cm[1, 0] if cm.shape[0] > 1 else 0
                if metric_name in ('pod',):
                    vals.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                elif metric_name in ('far',):
                    vals.append(fp / (tp + fp) if (tp + fp) > 0 else 0)
                else:
                    vals.append(tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0)
        plt.bar(x - w/2, flow_vals, w, label='With UNet Flow')
        plt.bar(x + w/2, no_flow_vals, w, label='Without Flow')
        plt.xlabel('Class'); plt.ylabel(metric_name.upper())
        plt.title(f'Per-Class {metric_name.upper()} Comparison')
        plt.xticks(x, classes); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'comparison_{metric_name}.png')); plt.close()

    print(f"\nAnalysis complete. Figures saved to: {output_dir}")


# ============================================================================
# 14. 辅助函数
# ============================================================================

def save_sample_names(all_positive, all_negative, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'positive_samples.txt'), 'w', encoding='utf-8') as f:
        f.write("Positive samples\n" + "=" * 60 + "\n")
        for i, (date, ts, labels) in enumerate(all_positive, 1):
            names = [n for n, v in zip(['雷暴', '短时强降水', '大风'], labels) if v]
            f.write(f"{i:3d}. {date} {ts.strftime('%Y-%m-%d %H:%M:%S')} [{', '.join(names)}]\n")
    with open(os.path.join(out_dir, 'negative_samples.txt'), 'w', encoding='utf-8') as f:
        f.write("Negative samples\n" + "=" * 60 + "\n")
        for i, (date, ts) in enumerate(all_negative, 1):
            f.write(f"{i:3d}. {date} {ts.strftime('%Y-%m-%d %H:%M:%S')}\n")


def run_analysis():
    samples = load_data()
    n = len(samples['radar'])
    split = int(0.8 * n)
    val_data = [torch.FloatTensor(samples[k][split:]) for k in ['radar', 'satellite', 'awos', 'optical_flow']]
    val_labels = samples['labels'][split:]
    val_timestamps = samples['timestamps'][split:]

    ckpt_f = torch.load(os.path.join(output_dir, 'model_flow_final.pth'), map_location=device)
    model_flow = WeatherModel(use_optical_flow=True).to(device)
    model_flow.load_state_dict(ckpt_f['state_dict'] if isinstance(ckpt_f, dict) else ckpt_f)
    ckpt_nf = torch.load(os.path.join(output_dir, 'model_no_flow_final.pth'), map_location=device)
    model_no_flow = WeatherModel(use_optical_flow=False).to(device)
    model_no_flow.load_state_dict(ckpt_nf['state_dict'] if isinstance(ckpt_nf, dict) else ckpt_nf)

    analyze_features_and_errors(model_flow, model_no_flow, val_data, val_labels,
                                val_timestamps, ckpt_f, ckpt_nf)


# ============================================================================
# 15. 主函数
# ============================================================================

def main():
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    samples = load_data()
    print(f"Loaded {len(samples['radar'])} samples")

    radar_tensor = torch.FloatTensor(samples['radar'])
    satellite_tensor = torch.FloatTensor(samples['satellite'])
    awos_tensor = torch.FloatTensor(samples['awos'])
    flow_tensor = torch.FloatTensor(samples['optical_flow'])
    label_tensor = torch.FloatTensor(samples['labels'])
    indices = torch.LongTensor(samples['indices'])
    timestamp_list = samples['timestamps']

    label_array = label_tensor.numpy()
    print(f"Label counts -- 雷暴: {label_array[:,0].sum():.0f}, "
          f"短时强降水: {label_array[:,1].sum():.0f}, 大风: {label_array[:,2].sum():.0f}")

    train_idx, val_idx = train_test_split(range(len(label_tensor)), test_size=0.2, random_state=42)

    dataset = TensorDataset(radar_tensor, satellite_tensor, awos_tensor, flow_tensor,
                            label_tensor, indices, torch.arange(len(radar_tensor)))
    num_workers = min(8, multiprocessing.cpu_count())
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print("\n" + "=" * 60)
    print("Hyperparameter Tuning -- With UNet Flow")
    print("=" * 60)
    best_flow = hyperparameter_tune(train_loader, val_loader, timestamp_list, model_name='flow')

    print("\n" + "=" * 60)
    print("Hyperparameter Tuning -- Without Flow")
    print("=" * 60)
    best_no_flow = hyperparameter_tune(train_loader, val_loader, timestamp_list, model_name='no_flow')

    # Train final models
    model_flow = WeatherModel(use_optical_flow=True).to(device)
    model_flow.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, (nn.Conv2d, nn.Linear)) else None)
    alpha_f = ast.literal_eval(best_flow['alpha']) if isinstance(best_flow['alpha'], str) else best_flow['alpha']
    metrics_flow, _ = train_model(model_flow, train_loader, val_loader, timestamp_list, 'flow',
                                  lr=best_flow['lr'], alpha=alpha_f, gamma=best_flow['gamma'],
                                  num_epochs=best_flow['num_epochs'], patience=best_flow['patience'],
                                  trial_id=0, is_final=True, focal_weight=best_flow['focal_weight'])

    model_no_flow = WeatherModel(use_optical_flow=False).to(device)
    model_no_flow.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, (nn.Conv2d, nn.Linear)) else None)
    alpha_nf = ast.literal_eval(best_no_flow['alpha']) if isinstance(best_no_flow['alpha'], str) else best_no_flow['alpha']
    metrics_no_flow, _ = train_model(model_no_flow, train_loader, val_loader, timestamp_list, 'no_flow',
                                     lr=best_no_flow['lr'], alpha=alpha_nf, gamma=best_no_flow['gamma'],
                                     num_epochs=best_no_flow['num_epochs'], patience=best_no_flow['patience'],
                                     trial_id=0, is_final=True, focal_weight=best_no_flow['focal_weight'])

    comparison = pd.DataFrame({
        'Model': ['NoFlow', 'UNetFlow'],
        'Accuracy': [metrics_no_flow['accuracy'], metrics_flow['accuracy']],
        'F1_macro': [metrics_no_flow['f1_macro'], metrics_flow['f1_macro']],
        'ROC_AUC': [metrics_no_flow['roc_auc'], metrics_flow['roc_auc']],
        'TS': [metrics_no_flow['ts'], metrics_flow['ts']],
        'POD': [metrics_no_flow['pod'], metrics_flow['pod']],
        'FAR': [metrics_no_flow['far'], metrics_flow['far']],
        'CSI': [metrics_no_flow['csi'], metrics_flow['csi']],
    })
    comparison.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    print("\n" + "=" * 60 + "\nFINAL COMPARISON\n" + "=" * 60)
    print(comparison.to_string(index=False))

    print("\n" + "=" * 60 + "\nRunning Analysis\n" + "=" * 60)
    run_analysis()
    print(f"\nAll done! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
