"""中间过程可视化（调试与精度报告用，NFR-5 可解释性）。

只依赖 Pillow。所有输出为 PNG：
``01_gray`` / ``02_binary`` / ``03_cleaned`` / ``04_skeleton`` /
``05_profile_overlay``（提取实体叠在原图上）/ ``06_truth_vs_extracted``。
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

from . import geometry as G


def _to_image(mask):
    return Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), 'L').convert('RGB')


def save_mask(mask, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), 'L').save(path)
    return path


def _px_polylines(ents, axis_row, max_step=0.3):
    """实体（px 坐标 (z_px, x_px)）→ 图像坐标折线 [(col,row), ...]。"""
    out = []
    for e in ents:
        pts = G.entity_polyline(e, max_step=max_step)
        out.append([(p[0], axis_row - p[1]) for p in pts])
    return out


def _mm_polylines(ents_mm, calib, max_step_mm=0.1):
    """实体（mm）→ 图像坐标折线，用标定的逆变换。"""
    out = []
    for e in ents_mm:
        pts = G.entity_polyline(e, max_step=max_step_mm)
        seg = []
        for (z, x) in pts:
            z_px = z / (calib.scale * calib.z_sign) + calib.z_ref_px
            seg.append((z_px, calib.axis_row_used - x / calib.scale))
        out.append(seg)
    return out


def overlay(ink, *, ents_px=None, axis_row=0, ents_mm=None, calib=None,
            truth_mm=None, extra_lines=None, path=None, max_step=0.3,
            upscale=1.0):
    """把提取结果（红）与真值（蓝）叠在墨迹（灰）上。"""
    img = _to_image(ink)
    drw = ImageDraw.Draw(img)
    h, w = ink.shape
    if axis_row:
        drw.line([(0, axis_row), (w - 1, axis_row)], fill=(120, 120, 120), width=1)

    def draw(polys, color, width=1):
        for seg in polys:
            if len(seg) >= 2:
                drw.line([(float(a), float(b)) for a, b in seg], fill=color, width=width)

    if ents_px is not None:
        draw(_px_polylines(ents_px, axis_row, max_step), (255, 40, 40), 1)
    if ents_mm is not None and calib is not None:
        draw(_mm_polylines(ents_mm, calib), (255, 40, 40), 1)
    if truth_mm is not None and calib is not None:
        draw(_mm_polylines(truth_mm, calib), (40, 90, 255), 1)
    if extra_lines:
        for seg, color in extra_lines:
            draw([seg], color, 1)

    if upscale and upscale != 1.0:
        img = img.resize((int(w * upscale), int(h * upscale)), Image.NEAREST)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        img.save(path)
    return img


def stage_sheet(stages, path, upscale=1.0):
    """把若干二值阶段图横向拼成一张对照图。"""
    imgs = [_to_image(m) for m in stages]
    if not imgs:
        return None
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs) + 4 * (len(imgs) - 1)
    sheet = Image.new('RGB', (w, h), (255, 255, 255))
    x = 0
    for im in imgs:
        sheet.paste(im, (x, 0))
        x += im.width + 4
    if upscale != 1.0:
        sheet = sheet.resize((int(sheet.width * upscale), int(sheet.height * upscale)),
                             Image.NEAREST)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sheet.save(path)
    return path
