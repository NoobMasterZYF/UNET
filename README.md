# UNet Optical Flow for Weather Nowcasting

将 TV-L1 光流替换为 U-Net 深度光流，用于 CMA-HKQX 天气短临预测。

## 文件说明

- `unet_optical_flow.py` — U-Net 光流模型、合成数据生成、预训练、推理
- `weather_model_unet_flow.py` — 完整训练流程（数据加载 → 调参 → 训练 → 分析）

## 快速开始

```bash
# 1. 预训练 U-Net 光流（首次运行自动执行，约20-30分钟 GPU）
python unet_optical_flow.py D:/CMA-HKQX-2024/dataset-for-training

# 2. 完整训练 + 对比分析
python weather_model_unet_flow.py
```

## 模型对比

| 方案 | 光流方法 | 特点 |
|------|---------|------|
| No Flow | — | 仅 Radar + Satellite + AWOS |
| TV-L1 Flow | 变分优化 | 假设亮度恒定，雷达回波不满足 |
| **UNet Flow** | 深度学习 | 数据驱动，自监督预训练 |

## 特征重要性

使用 Permutation Importance 量化各模态（Radar/Satellite/AWOS/OpticalFlow）对 TS 的贡献。
