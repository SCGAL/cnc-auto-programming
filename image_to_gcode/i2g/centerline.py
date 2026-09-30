"""② 基准定位：倾斜角估计 + 旋转校正。

立项文档 §3.2② 建议用 PCA 主轴估计倾斜角。**实测量级下 PCA 不够用**：
本项目的倾斜预算是 0.2 mm/80 mm ≈ 0.14°（见下方"为什么角度精度是硬约束"），
而整条轮廓的长宽比只有约 4:1，主轴方向受凹肩、倒角影响会偏离图样水平线
0.5° 以上。因此改用更直接、更契合"车削件有大量水平外圆"这一先验的方法：

    **投影直方图能量最大化**：把墨迹点绕质心旋转候选角 φ，对行的坐标做 1 px
    直方图，取 Σh² 最大者。水平外圆对齐到同一行时直方图最尖锐，Σh² 取极大。
    粗到细三级搜索（0.1° → 0.02° → 0.005°），可达约 0.01° 的角度分辨力。

为什么角度精度是硬约束
----------------------
旋转误差 δ（弧度）会让距旋转中心 d 的点产生约 d·δ 的位移。
轮廓半长 d ≈ 40 mm，δ = 0.1° = 1.75e-3 rad → 位移 0.07 mm；δ = 0.3° → 0.21 mm
——**单是倾斜残差就能吃掉整个 0.2 mm 预算**。这就是立项文档 R3
"倾斜未校正彻底 → 水平面判定全错"的量化版本。

（注：即使倾斜残留 0.1°，后续 snap 的角度吸附还会把水平外圆拉回正，
所以本模块的主要职责是让"哪个是水平面"这件事判对，而不是做到 0 误差。）
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

DEFAULT_STEPS = (0.1, 0.02, 0.005)


def _row_hist_energy(pts_rel, phi_rad, bin_px=1.0):
    """候选角 φ 下，行坐标直方图的能量 Σh²（**三角核/线性插值分箱**）。

    点变换与 ``PIL.Image.rotate(φ, expand=True)`` 保持一致：
        x' =  x·cosφ + y·sinφ
        y' = −x·sinφ + y·cosφ

    ★ 为什么必须用线性插值分箱而不是硬分箱（``floor``）
    ---------------------------------------------------
    实测（10 px/mm、线宽 2 px、**skew_deg=0 的图**）硬分箱的能量曲线是一段
    **平坦台阶**：−0.100°~−0.025° 全为 237568，真正的 0° 只有 233854，
    而 −0.175° 处冒出一个高 12（+0.005%）的伪尖峰 —— 因为硬分箱下
    "某条水平线正好落进同一行 bin"是个阶跃事件，目标函数不光滑，
    搜索必然锁死在伪极值上。后果很严重：估计出 −0.195° 并强行旋转，
    690 px 宽的外圆两端因此错开 2.3 px，直接造成 0.1~0.2 mm 的系统偏差
    （实测 OD1/OD2/OD3 高度差变成 119/39 px，真值应为 120/40）。

    线性插值分箱（每点按权重摊到相邻两个 bin）使能量对 φ 连续可微，
    倾斜导致的"抹平"从第一刻起就被惩罚，峰值精确落在真实角度上。
    """
    c, s = math.cos(phi_rad), math.sin(phi_rad)
    y = -pts_rel[:, 0] * s + pts_rel[:, 1] * c
    t = (y - y.min()) / bin_px
    b0 = np.floor(t).astype(np.int64)
    frac = t - b0
    nb = int(b0.max()) + 2
    h = (np.bincount(b0, weights=(1.0 - frac), minlength=nb)
         + np.bincount(b0 + 1, weights=frac, minlength=nb))
    return float((h * h).sum())


def estimate_skew(ink, *, max_deg=3.0, steps=DEFAULT_STEPS, bin_px=1.0,
                  max_points=8000):
    """估计"把图形转正所需施加的旋转角"（度，正值 = PIL 逆时针）。

    返回 dict：``theta_deg``、``contrast``（最优 / 中位能量，越大越可信）、
    ``n_evals``、``n_points``、``score``。
    """
    ys, xs = np.nonzero(ink)
    n = len(ys)
    if n < 16:
        return {'theta_deg': 0.0, 'contrast': 1.0, 'n_evals': 0,
                'n_points': n, 'score': 0.0, 'reliable': False}
    if n > max_points:
        sel = np.linspace(0, n - 1, max_points).astype(np.int64)
        ys, xs = ys[sel], xs[sel]
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    pts_rel = pts - pts.mean(axis=0)

    scores_all = []
    n_eval = 0

    def scan(center, half, step):
        nonlocal n_eval
        best_a, best_s = center, -1.0
        k = int(math.ceil(half / step))
        for i in range(-k, k + 1):
            a = center + i * step
            s = _row_hist_energy(pts_rel, math.radians(a), bin_px)
            scores_all.append(s)
            n_eval += 1
            if s > best_s:
                best_a, best_s = a, s
        return best_a, best_s

    a, _ = scan(0.0, max_deg, steps[0])
    for st in steps[1:]:
        a, s = scan(a, st * 5.0, st)
    best_score = s
    med = float(np.median(scores_all)) if scores_all else 1.0
    contrast = float(best_score / med) if med > 0 else 1.0
    return {'theta_deg': float(a), 'contrast': contrast, 'n_evals': n_eval,
            'n_points': int(len(pts)), 'score': float(best_score),
            'reliable': contrast >= 1.02 and abs(a) <= max_deg + 1e-9}


def rotate_pair(ink, gray01, theta_deg, *, threshold=0.5):
    """把二值图与归一化覆盖率图按同一角度旋转并裁回墨迹范围。

    返回 ``(ink_rot, gray01_rot, row0, col0)``。覆盖率图用双三次插值并钳到 [0,1]，
    亚像素中线精化依赖它，必须与二值图严格同一变换。
    """
    if abs(theta_deg) < 1e-9:
        return ink, (gray01 if gray01 is not None else None), 0, 0
    im = Image.fromarray(np.where(ink, 255, 0).astype(np.uint8), 'L')
    rot = im.rotate(float(theta_deg), resample=Image.BICUBIC, expand=True, fillcolor=0)
    a = np.asarray(rot, dtype=np.uint8) >= int(round(255 * threshold))
    g = None
    if gray01 is not None:
        gi = Image.fromarray(np.clip(gray01 * 255.0, 0, 255).astype(np.uint8), 'L')
        gr = gi.rotate(float(theta_deg), resample=Image.BICUBIC, expand=True, fillcolor=0)
        g = np.clip(np.asarray(gr, dtype=np.float64) / 255.0, 0.0, 1.0)
    rows = np.where(a.any(axis=1))[0]
    cols = np.where(a.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return a, g, 0, 0
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    return (a[r0:r1, c0:c1].copy(),
            (g[r0:r1, c0:c1].copy() if g is not None else None), r0, c0)


def run(ink, gray01=None, *, enabled=True, max_deg=3.0, steps=DEFAULT_STEPS,
        min_contrast=1.02, extra_deg=0.0):
    """倾斜校正全流程。

    ``extra_deg`` 是外层迭代（见 ``pipeline``）反哺的残余倾角修正量：
    投影直方图法的角度分辨力约 0.05~0.1°，而**只需 0.1° 残差就能让矢量化多切出
    伪圆弧**（实测 skew=+2° 时 arc 数从 2 变 4、端点误差 0.05→1.01 mm）。
    故第一遍跑完后用"已提取的最长水平外圆实际倾角"回推修正量，再重跑一遍。

    返回 dict：``ink``（转正后的图）、``gray01``、``theta_deg``、``row0``、``col0``、
    ``contrast``、``reliable``、``warnings``。
    """
    info = {'theta_deg': 0.0, 'contrast': 1.0, 'row0': 0, 'col0': 0,
            'reliable': True, 'warnings': [], 'enabled': bool(enabled),
            'extra_deg': float(extra_deg)}
    if not enabled:
        info['warnings'].append('倾斜校正被显式关闭（--no-deskew）')
        return {'ink': ink, 'gray01': gray01, **info}

    est = estimate_skew(ink, max_deg=max_deg, steps=steps)
    info.update({k: est[k] for k in ('theta_deg', 'contrast', 'reliable')})
    info['n_evals'] = est['n_evals']
    info['theta_hist_deg'] = est['theta_deg']
    if not est['reliable']:
        info['warnings'].append(
            f'倾斜角估计可信度低（对比度 {est["contrast"]:.3f} < {min_contrast}）：'
            f'图中缺少足够长的水平参考线，已按估计值 {est["theta_deg"]:+.3f}° 校正，'
            f'结果可能整体旋转偏差')
    theta = est['theta_deg'] + float(extra_deg)
    info['theta_deg'] = theta
    rot, grot, r0, c0 = rotate_pair(ink, gray01, theta)
    info['row0'], info['col0'] = r0, c0
    if abs(theta) > 0.3:
        info['warnings'].append(f'倾斜角 {theta:+.3f}° 已被校正')
    return {'ink': rot, 'gray01': grot, **info}


def residual_tilt_deg(points):
    """对已提取的有序点列再估一次倾斜，用于校正后的回检（R3）。

    取点列的**最长的近水平段**所在方向作为整体倾斜的代理指标：这里用
    全局最小二乘拟合的稳健做法 —— 只对 |dy/dx| < 0.3 的长段投票。
    """
    P = np.asarray(points, dtype=float)
    if len(P) < 8:
        return 0.0
    d = np.diff(P, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    m = seg > 1.0
    if not m.any():
        return 0.0
    dx, dy = d[m, 0], d[m, 1]
    horiz = np.abs(dy) <= 0.3 * np.abs(dx) + 1e-9
    if not horiz.any():
        return 0.0
    w = seg[m][horiz]
    ang = np.degrees(np.arctan2(dy[horiz], dx[horiz]))
    return float(np.average(ang, weights=w))
