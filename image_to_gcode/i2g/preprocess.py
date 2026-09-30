"""① 预处理：灰度 → 二值(Otsu) → 去噪 → 形态学闭运算 → 小连通域剔除。

立项文档 §3.2① 的偏离说明（实测驱动，已写入精度报告）
------------------------------------------------------
文档写"中值滤波 3×3"。**二值图的 3×3 中值滤波 = 多数表决滤波**，
它会抹掉线宽 1 px 的笔画：1 px 笔画上每个墨迹像素的 3×3 邻域内
最多只有 3 个墨迹像素（自身 + 沿线两个），不到 9 的多数 5，于是整条线被清零。
测试矩阵里 ``line_width=1`` 是明确要求覆盖的工况，因此本实现改为：

1. ``remove_isolated``  —— 只剔除 8 邻域内**完全没有**墨迹邻居的孤立点
   （真正的椒盐噪点；1 px 笔画上任何像素都至少有一个邻居，安全）；
2. ``remove_small_components`` —— 连通域面积阈值，剔除剩余噪团/文字/箭头。

``median_filter3`` 仍然保留为可选项（``--denoise median``），并有单元测试
证明它会摧毁 1 px 笔画 —— 保留它是为了可复现地展示这个取舍，而不是因为它可用。

只做纯函数，不读文件（NFR-4）。
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .errors import ImageReadError

NEIGHBORS_8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


# =============================================================================
# 读图 / 灰度 / 二值
# =============================================================================

def load_gray(path):
    """读图为 uint8 灰度数组，并记录是否做了 16bit→8bit 降位。"""
    try:
        img = Image.open(path)
        img.load()
    except FileNotFoundError as e:
        raise ImageReadError(f'图像不存在: {path}') from e
    except Exception as e:                                    # noqa: BLE001
        raise ImageReadError(f'图像无法解码: {path} ({type(e).__name__}: {e})') from e
    if img.mode in ('I;16', 'I', 'F'):
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-9:
            raise ImageReadError(f'图像灰度恒定（{lo}），没有任何可提取的信息: {path}')
        arr = (arr - lo) / (hi - lo) * 255.0
        img = Image.fromarray(arr.astype(np.uint8), 'L')
    else:
        img = img.convert('L')
    a = np.asarray(img, dtype=np.uint8)
    if a.size == 0:
        raise ImageReadError(f'图像为空: {path}')
    return a


def otsu_threshold(gray):
    """Otsu 大津法：最大化类间方差。返回 0..255 的整数阈值。"""
    hist = np.bincount(np.asarray(gray, dtype=np.uint8).ravel(), minlength=256)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 127
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / np.maximum(denom, 1e-12), 0.0)
    return int(np.argmax(sigma_b))


def binarize(gray, threshold=None, polarity='auto'):
    """二值化，返回 ``(ink, threshold)``，``ink`` 为 bool 且 True = 墨迹。

    ``polarity``: ``'dark'`` 深色为墨 / ``'light'`` 浅色为墨 / ``'auto'`` 自动。
    线框图的墨迹占比天然远小于背景，故 auto 按"少数派即墨"判定。
    """
    t = otsu_threshold(gray) if threshold is None else int(threshold)
    ink = np.asarray(gray, dtype=np.uint8) <= t
    pol = polarity
    if pol == 'auto':
        pol = 'dark' if ink.mean() <= 0.5 else 'light'
    if pol == 'light':
        ink = ~ink
    return ink.astype(bool), t


# =============================================================================
# 去噪
# =============================================================================

def _count_neighbors(mask, offsets):
    h, w = mask.shape
    P = np.pad(mask.astype(np.uint8), 1)
    acc = np.zeros((h, w), np.uint8)
    for dy, dx in offsets:
        acc += P[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return acc


def remove_isolated(mask):
    """剔除 8 邻域内没有任何墨迹邻居的孤立像素（椒盐噪点）。

    对 1 px 笔画安全：笔画上任何像素都至少有一个邻居。
    """
    return mask & (_count_neighbors(mask, NEIGHBORS_8) > 0)


def median_filter3(mask):
    """二值图 3×3 中值滤波（= 多数表决）。

    ⚠️ 会摧毁线宽 <= 1 px 的笔画（见模块 docstring）。默认不启用。
    """
    return _count_neighbors(mask, NEIGHBORS_8) >= 5


def _dilate(mask, radius=1):
    h, w = mask.shape
    P = np.pad(mask, radius)
    out = np.zeros((h, w), bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius + 1e-9:
                continue                       # 近似圆盘结构元
            out |= P[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return out


def _erode(mask, radius=1):
    return ~_dilate(~mask, radius)


def closing(mask, radius=1):
    """形态学闭运算：先膨胀后腐蚀，用于修补断线。

    对 1 px 笔画近似幂等（膨胀到 3 px 再腐蚀回来），不会改变线宽。
    """
    if radius <= 0:
        return mask
    return _erode(_dilate(mask, radius), radius)


# =============================================================================
# 连通域
# =============================================================================

def label_components(mask, connectivity=8):
    """连通域标记（稀疏 BFS，只遍历墨迹像素）。

    返回 ``(labels_shape_like_mask, sizes)``：``labels`` 为 int32，-1 = 背景。
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    n = len(ys)
    labels = np.full((h, w), -1, np.int32)
    if n == 0:
        return labels, np.zeros(0, np.int64)
    idx = np.full((h, w), -1, np.int32)
    idx[ys, xs] = np.arange(n, dtype=np.int32)
    comp = np.full(n, -1, np.int32)
    offs = NEIGHBORS_8 if connectivity == 8 else NEIGHBORS_4
    n_comp = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        comp[s] = n_comp
        stack = [s]
        while stack:
            p = stack.pop()
            y, x = int(ys[p]), int(xs[p])
            for dy, dx in offs:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    q = idx[yy, xx]
                    if q >= 0 and comp[q] < 0:
                        comp[q] = n_comp
                        stack.append(int(q))
        n_comp += 1
    labels[ys, xs] = comp
    sizes = np.bincount(comp, minlength=n_comp).astype(np.int64)
    return labels, sizes


def remove_small_components(mask, min_area=40):
    """剔除面积小于 min_area 的连通域（文字、箭头、噪团）。"""
    if min_area <= 1:
        return mask
    labels, sizes = label_components(mask)
    if len(sizes) == 0:
        return mask
    keep = sizes >= min_area
    if not keep.any():
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    out[ys, xs] = keep[labels[ys, xs]]
    return out


def crop_to_ink(mask, margin=8):
    """裁剪到墨迹包围盒（+margin），返回 ``(cropped, row0, col0)``。

    大幅降低后续细化/连通域的开销（NFR-6：2000×1500 图 <= 10 s）。
    """
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return mask, 0, 0
    r0 = max(0, int(rows[0]) - margin)
    r1 = min(mask.shape[0], int(rows[-1]) + 1 + margin)
    c0 = max(0, int(cols[0]) - margin)
    c1 = min(mask.shape[1], int(cols[-1]) + 1 + margin)
    return mask[r0:r1, c0:c1].copy(), r0, c0


# =============================================================================
# 流水线入口
# =============================================================================

def run(gray, *, denoise='isolated', min_area=40, close_radius=1,
        threshold=None, polarity='auto', margin=8):
    """预处理全流程。

    返回 dict: ``ink``（裁后二值图）、``gray01``（裁后归一化覆盖率 0..1，
    供亚像素中线精化用）、``threshold``、``row0``、``col0``、
    ``n_components``、``dropped_area``、``stats``。
    """
    gray01_full = np.asarray(gray, dtype=np.float64) / 255.0
    ink, t = binarize(gray, threshold=threshold, polarity=polarity)
    raw_ink = int(ink.sum())
    stats = {'threshold': t, 'ink_px_raw': raw_ink,
             'image_size': [int(gray.shape[1]), int(gray.shape[0])]}

    if denoise == 'median':
        ink = median_filter3(ink)
    elif denoise == 'isolated':
        ink = remove_isolated(ink)
    elif denoise in ('none', None):
        pass
    else:
        raise ValueError(f'未知去噪方式: {denoise}')
    stats['ink_px_denoised'] = int(ink.sum())

    if close_radius and close_radius > 0:
        ink = closing(ink, int(close_radius))
        stats['ink_px_closed'] = int(ink.sum())

    labels, sizes = label_components(ink)
    stats['n_components'] = int(len(sizes))
    ink2 = remove_small_components(ink, min_area=min_area)
    stats['dropped_area_px'] = int(ink.sum()) - int(ink2.sum())
    stats['min_area'] = int(min_area)

    cropped, r0, c0 = crop_to_ink(ink2, margin=margin)
    gray01 = gray01_full[r0:r0 + cropped.shape[0], c0:c0 + cropped.shape[1]]
    if polarity == 'light' or (polarity == 'auto' and ink.mean() > 0.5):
        gray01 = 1.0 - gray01
    stats['cropped_size'] = [int(cropped.shape[1]), int(cropped.shape[0])]
    stats['ink_px_final'] = int(cropped.sum())
    return {'ink': cropped, 'gray01': gray01, 'threshold': t, 'row0': r0, 'col0': c0,
            'n_components': int(len(sizes)), 'stats': stats}
