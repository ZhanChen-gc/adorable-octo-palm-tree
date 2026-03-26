#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NIfTI 可视化工具 - 支持原始图像与分割叠加，暗色主题，交互式滑块。
用法：
    python view_nii.py 原始图像.nii.gz [--seg 分割图像.nii.gz] [--axis 0/1/2] [--cmap_img gray] [--cmap_seg jet]
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import argparse

def load_nii(filepath):
    """加载 NIfTI 文件，返回 numpy 数组"""
    img = nib.load(filepath)
    return img.get_fdata()

def get_slice(data, idx, axis):
    """根据轴获取切片，并转置为显示方向（x向右，y向上）"""
    if axis == 0:
        slice_data = data[idx, :, :]
    elif axis == 1:
        slice_data = data[:, idx, :]
    else:
        slice_data = data[:, :, idx]
    # 转置使图像方向符合视觉习惯
    return slice_data.T

def main():
    parser = argparse.ArgumentParser(description='NIfTI 可视化工具')
    parser.add_argument('image', help='原始图像文件 (.nii 或 .nii.gz)')
    parser.add_argument('--seg', help='分割结果文件（可选）', default=None)
    parser.add_argument('-a', '--axis', type=int, default=2, choices=[0,1,2],
                        help='切片轴: 0=x, 1=y, 2=z (默认 z)')
    parser.add_argument('--cmap_img', default='gray', help='原始图像颜色映射 (默认 gray)')
    parser.add_argument('--cmap_seg', default='jet', help='分割颜色映射 (默认 jet)')
    args = parser.parse_args()

    # 加载数据
    img_data = load_nii(args.image)
    seg_data = load_nii(args.seg) if args.seg else None

    # 获取切片数量
    n_slices = img_data.shape[args.axis]
    init_slice = n_slices // 2

    # 设置暗色背景样式
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
    plt.subplots_adjust(bottom=0.2)
    ax.set_facecolor('black')

    # 初始显示原始图像
    img_slice = get_slice(img_data, init_slice, args.axis)
    im = ax.imshow(img_slice, cmap=args.cmap_img, origin='lower', interpolation='none')
    ax.set_title(f'Slice {init_slice}', color='white', fontsize=12)
    ax.axis('off')

    # 如果有分割，叠加显示
    if seg_data is not None:
        seg_slice = get_slice(seg_data, init_slice, args.axis)
        # 使用固定的 vmin/vmax 使颜色对比鲜明
        seg_overlay = ax.imshow(seg_slice, cmap=args.cmap_seg, alpha=0.7,
                                vmin=0, vmax=seg_data.max(), origin='lower', interpolation='none')
        # 添加颜色条
        cbar = plt.colorbar(seg_overlay, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Class', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')

    # 添加滑块
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgray')
    slider = Slider(ax_slider, 'Slice', 0, n_slices-1, valinit=init_slice, valstep=1)

    def update(val):
        idx = int(slider.val)
        img_slice = get_slice(img_data, idx, args.axis)
        im.set_data(img_slice)
        if seg_data is not None:
            seg_slice = get_slice(seg_data, idx, args.axis)
            seg_overlay.set_data(seg_slice)
        ax.set_title(f'Slice {idx}', color='white')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

if __name__ == '__main__':
    main()