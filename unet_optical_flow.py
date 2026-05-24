"""
UNet-based Optical Flow for Weather Radar Nowcasting.
Replaces TV-L1 optical flow with a learned deep optical flow estimator.
Includes self-supervised pre-training on synthetic flow data generated from radar images.

Architecture: 4-level UNet with skip connections
Input:  2 consecutive radar frames (400, 400, 2)
Output: Dense optical flow field (400, 400, 2) — (u, v) displacement
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import glob
import random
import re
import gc

# ──────────────────────────────────────────────────────────────
# 1. UNet Architecture for Optical Flow
# ──────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU"""
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
    """Upsample → concat skip → DoubleConv"""
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
    Input:  (B, 2, H, W) — two consecutive frames concatenated
    Output: (B, 2, H, W) — (u, v) flow field
    """
    def __init__(self, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock(2, base_ch)       # H/2
        self.enc2 = EncoderBlock(base_ch, base_ch * 2)    # H/4
        self.enc3 = EncoderBlock(base_ch * 2, base_ch * 4)  # H/8
        self.enc4 = EncoderBlock(base_ch * 4, base_ch * 8)  # H/16

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)

        # Decoder
        self.dec4 = DecoderBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.dec3 = DecoderBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec1 = DecoderBlock(base_ch * 2, base_ch, base_ch)

        # Output head
        self.out_conv = nn.Conv2d(base_ch, 2, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            x: (B, 2, H, W) stacked frame_t and frame_{t+1}
        Returns:
            flow: (B, 2, H, W) predicted optical flow
        """
        d1, s1 = self.enc1(x)
        d2, s2 = self.enc2(d1)
        d3, s3 = self.enc3(d2)
        d4, s4 = self.enc4(d3)

        bn = self.bottleneck(d4)

        u4 = self.dec4(bn, s4)
        u3 = self.dec3(u4, s3)
        u2 = self.dec2(u3, s2)
        u1 = self.dec1(u2, s1)

        flow = self.out_conv(u1)
        return flow

    def predict_flow(self, img1, img2):
        """
        Convenience method for inference.
        Args:
            img1, img2: (H, W) numpy arrays or tensors
        Returns:
            flow: (H, W, 2) numpy array
        """
        was_numpy = isinstance(img1, np.ndarray)
        if was_numpy:
            img1 = torch.from_numpy(img1).float()
            img2 = torch.from_numpy(img2).float()
        if img1.dim() == 2:
            img1 = img1.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            img2 = img2.unsqueeze(0).unsqueeze(0)
        pair = torch.cat([img1, img2], dim=1)  # (1, 2, H, W)
        with torch.no_grad():
            flow = self.forward(pair)  # (1, 2, H, W)
        if was_numpy:
            flow = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        return flow


# ──────────────────────────────────────────────────────────────
# 2. Synthetic Flow Data Generator (Self-Supervised Pre-training)
# ──────────────────────────────────────────────────────────────

def generate_random_flow_field(h, w, max_displacement=20.0):
    """
    Generate a smooth random flow field using low-frequency sine waves.
    Returns flow of shape (h, w, 2).
    """
    # Create grid of coordinates
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Use multiple sine waves at different frequencies for smooth, diverse flows
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

    # Add a uniform translation component
    u += random.uniform(-max_displacement * 0.3, max_displacement * 0.3)
    v += random.uniform(-max_displacement * 0.3, max_displacement * 0.3)

    return np.stack([u, v], axis=-1).astype(np.float32)


def warp_image(img, flow):
    """
    Warp image using backward flow with bilinear sampling.
    img:  (H, W) numpy array
    flow: (H, W, 2) numpy array (u, v displacement from img1 to img2)
    Returns warped image.
    """
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Target coordinates in source image (backward warp)
    src_x = xx - flow[:, :, 0]
    src_y = yy - flow[:, :, 1]

    # Clip to valid range
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)

    # Bilinear interpolation
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
    """
    Generate a batch of synthetic training data.
    Args:
        images: list of (H, W) numpy arrays (radar images)
        batch_size: number of pairs to generate
        img_size: target image size
        max_displacement: max flow magnitude in pixels
    Returns:
        img1_batch: (B, 1, H, W) tensor
        img2_batch: (B, 1, H, W) tensor
        flow_batch: (B, 2, H, W) tensor
    """
    img1_list = []
    img2_list = []
    flow_list = []

    for _ in range(batch_size):
        # Pick random base image
        idx = random.randint(0, len(images) - 1)
        img = images[idx].astype(np.float32)

        if img.shape != img_size:
            img = np.array(Image.fromarray(img).resize(
                (img_size[1], img_size[0]), Image.Resampling.LANCZOS))

        # Generate random flow
        flow = generate_random_flow_field(img_size[0], img_size[1], max_displacement)

        # Warp image
        warped = warp_image(img, flow)

        img1_list.append(img)
        img2_list.append(warped)
        flow_list.append(flow)

    img1_batch = torch.from_numpy(np.stack(img1_list)).unsqueeze(1).to(device)
    img2_batch = torch.from_numpy(np.stack(img2_list)).unsqueeze(1).to(device)
    flow_batch = torch.from_numpy(np.stack(flow_list)).permute(0, 3, 1, 2).to(device)

    return img1_batch, img2_batch, flow_batch


# ──────────────────────────────────────────────────────────────
# 3. Loss Functions
# ──────────────────────────────────────────────────────────────

class EPEWithSmoothnessLoss(nn.Module):
    """
    End Point Error + Edge-aware smoothness loss.
    """
    def __init__(self, smoothness_weight=0.1):
        super().__init__()
        self.smoothness_weight = smoothness_weight

    def forward(self, pred_flow, gt_flow, img1):
        """
        Args:
            pred_flow: (B, 2, H, W) predicted flow
            gt_flow:   (B, 2, H, W) ground truth flow
            img1:      (B, 1, H, W) first frame (for edge-aware smoothness)
        """
        # EPE: L2 distance
        epe = torch.mean((pred_flow - gt_flow) ** 2)

        # Edge-aware smoothness loss
        flow_grad_x = torch.abs(pred_flow[:, :, :, 1:] - pred_flow[:, :, :, :-1])
        flow_grad_y = torch.abs(pred_flow[:, :, 1:, :] - pred_flow[:, :, :-1, :])

        img_grad_x = torch.abs(img1[:, :, :, 1:] - img1[:, :, :, :-1])
        img_grad_y = torch.abs(img1[:, :, 1:, :] - img1[:, :, :-1, :])

        # Edge weights: penalize less where image gradients are high (edges)
        weight_x = torch.exp(-img_grad_x * 5.0)
        weight_y = torch.exp(-img_grad_y * 5.0)

        smoothness = (weight_x * flow_grad_x).mean() + (weight_y * flow_grad_y).mean()

        return epe + self.smoothness_weight * smoothness, epe.item(), smoothness.item()


# ──────────────────────────────────────────────────────────────
# 4. Pre-training Loop
# ──────────────────────────────────────────────────────────────

def pretrain_unet_flow(radar_dir, base_date_dir, dates, save_path,
                       img_size=(400, 400), num_epochs=50, batch_size=16,
                       max_displacement=20.0, device='cuda'):
    """
    Pre-train UNet optical flow on synthetic flow data generated from real radar images.

    Args:
        radar_dir: path template for radar images (e.g. '{base_dir}/{date}/radar_img')
        base_date_dir: base directory containing date folders
        dates: list of date strings to load images from
        save_path: where to save the pretrained model
        img_size: (H, W) target size
        num_epochs: training epochs
        batch_size: batch size
        max_displacement: max flow magnitude for synthetic data
        device: 'cuda' or 'cpu'
    """
    print(f"=== UNet Optical Flow Pre-training ===")
    print(f"Loading radar images for pre-training...")

    # Load all available radar images (3rd elevation, single channel)
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
                # Use 3rd elevation angle (closest to angle 3.0)
                if abs(angle - 3.0) > 0.5:
                    continue
                img_path = os.path.join(date_dir, filename)
                img = np.array(Image.open(img_path).convert('L')) / 255.0
                if img.shape != img_size:
                    img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(
                        (img_size[1], img_size[0]), Image.Resampling.LANCZOS)) / 255.0
                date_images.append(img.astype(np.float32))

        if date_images:
            # Take evenly spaced samples to avoid temporal redundancy
            step = max(1, len(date_images) // 20)
            all_images.extend(date_images[::step])
            print(f"  {date_str}: loaded {len(date_images)} images, kept {len(date_images[::step])}")

    if len(all_images) < 100:
        print(f"WARNING: Only {len(all_images)} images available for pre-training. Need more data.")

    print(f"Total images for synthetic generation: {len(all_images)}")

    # Create model
    model = UNetOpticalFlow(base_ch=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = EPEWithSmoothnessLoss(smoothness_weight=0.1)

    # Training loop
    steps_per_epoch = 200  # Generate new synthetic data each step
    model.train()

    best_loss = float('inf')
    for epoch in range(num_epochs):
        epoch_epe = 0.0
        epoch_smooth = 0.0

        for step in range(steps_per_epoch):
            img1, img2, gt_flow = generate_synthetic_batch(
                all_images, batch_size, img_size, max_displacement, device)

            # Predict flow
            pair = torch.cat([img1, img2], dim=1)  # (B, 2, H, W)
            pred_flow = model(pair)

            # Compute loss
            loss, epe, smooth = criterion(pred_flow, gt_flow, img1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_epe += epe
            epoch_smooth += smooth

        scheduler.step()
        avg_epe = epoch_epe / steps_per_epoch
        avg_smooth = epoch_smooth / steps_per_epoch
        avg_loss = avg_epe + 0.1 * avg_smooth

        print(f"Epoch {epoch+1}/{num_epochs} | EPE: {avg_epe:.4f} | "
              f"Smooth: {avg_smooth:.4f} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model to {save_path}")

    print(f"Pre-training complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {save_path}")
    return model


# ──────────────────────────────────────────────────────────────
# 5. UNet Optical Flow Computer (replaces TV-L1 in create_samples)
# ──────────────────────────────────────────────────────────────

def compute_unet_optical_flow(model, radar_sequence, device='cuda'):
    """
    Compute optical flow between consecutive frames in a radar sequence using UNet.

    Args:
        model: pre-trained UNetOpticalFlow model
        radar_sequence: (T, H, W) numpy array of radar frames (3rd elevation, normalized 0-1)
        device: device to run on

    Returns:
        optical_flow: (T-1, H, W, 2) numpy array of flow fields
    """
    model.eval()
    model.to(device)
    T = radar_sequence.shape[0]
    optical_flow = []

    for t in range(T - 1):
        img1 = torch.from_numpy(radar_sequence[t]).float().unsqueeze(0).unsqueeze(0).to(device)
        img2 = torch.from_numpy(radar_sequence[t + 1]).float().unsqueeze(0).unsqueeze(0).to(device)

        pair = torch.cat([img1, img2], dim=1)
        with torch.no_grad():
            flow = model(pair)  # (1, 2, H, W)
        flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        optical_flow.append(flow_np)

    return np.array(optical_flow, dtype=np.float32)


def compute_unet_optical_flow_batch(model, radar_batch, device='cuda'):
    """
    Batch version for computing optical flow on multiple samples at once.
    Args:
        model: pre-trained UNetOpticalFlow
        radar_batch: (N, T, H, W) numpy array (batch of radar sequences at 3rd elevation)
        device: device
    Returns:
        optical_flow_batch: (N, T-1, H, W, 2) numpy array
    """
    model.eval()
    model.to(device)
    N, T, H, W = radar_batch.shape
    optical_flow_batch = np.zeros((N, T-1, H, W, 2), dtype=np.float32)

    for t in range(T - 1):
        img1 = torch.from_numpy(radar_batch[:, t]).float().unsqueeze(1).to(device)
        img2 = torch.from_numpy(radar_batch[:, t+1]).float().unsqueeze(1).to(device)
        pair = torch.cat([img1, img2], dim=1)  # (N, 2, H, W)
        with torch.no_grad():
            flow = model(pair)  # (N, 2, H, W)
        optical_flow_batch[:, t] = flow.permute(0, 2, 3, 1).cpu().numpy()

    return optical_flow_batch


# ──────────────────────────────────────────────────────────────
# 6. Integration Helper: Load or Train UNet Flow Model
# ──────────────────────────────────────────────────────────────

def get_unet_flow_model(base_dir, dates, model_path=None, device='cuda',
                         force_retrain=False):
    """
    Get a pre-trained UNet optical flow model.
    Loads from disk if available, otherwise pre-trains from scratch.

    Args:
        base_dir: base directory for training data
        dates: list of date strings
        model_path: path to saved model (default: base_dir/unet_flow_pretrained.pth)
        device: 'cuda' or 'cpu'
        force_retrain: if True, re-train even if saved model exists

    Returns:
        model: pre-trained UNetOpticalFlow
    """
    if model_path is None:
        model_path = os.path.join(base_dir, 'unet_flow_pretrained.pth')

    model = UNetOpticalFlow(base_ch=64).to(device)

    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading pre-trained UNet optical flow from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Pre-training UNet optical flow from scratch...")
        model = pretrain_unet_flow(
            radar_dir=None,  # not used directly, we use base_date_dir
            base_date_dir=base_dir,
            dates=dates,
            save_path=model_path,
            device=device,
            num_epochs=50,
            batch_size=16,
        )

    model.eval()
    return model


if __name__ == "__main__":
    # Example: pre-train on some radar data
    # This is just for testing — adjust paths for your setup
    import sys
    if len(sys.argv) > 1:
        base = sys.argv[1]
    else:
        base = 'D:/CMA-HKQX-2024/dataset-for-training'

    # Get available dates
    info_path = os.path.join(base, 'infomation.xlsx')
    import pandas as pd
    info = pd.read_excel(info_path)
    dates = info['filename'].astype(str).tolist()

    save = os.path.join(base, 'unet_flow_pretrained.pth')
    pretrain_unet_flow(None, base, dates, save, device='cuda')
