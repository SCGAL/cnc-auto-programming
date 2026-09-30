"""⑥ 标定：像素 → 毫米。

三种方式（立项文档 §3.2⑥）
--------------------------
===============  ==============================  ============================
方式             参数                            说明
===============  ==============================  ============================
A 显式比例       ``--mm-per-px 0.1``             用户已知比例
B 已知外径 ⭐     ``--ref-diameter 40``           轮廓最大高度 = 半径
C 两特征点       ``--ref-points`` + ``--ref-mm``  最灵活
===============  ==============================  ============================

**强制要求**：没有任何标定参数时**必须报错退出**，并打印提取到的轮廓包围盒
（像素）供用户推算 —— 绝不默认假设 1 px = 1 mm（立项文档 §3.2⑥）。

流水线顺序说明（与文档的一处偏离）
----------------------------------
文档把"几何吸附(⑤)"排在"标定(⑥)"之前，但吸附表格里的容差全部以 **mm** 给出
（共线 0.15 mm、端点 0.2 mm、闭合 0.5 mm）。单位未定就无法判断容差，
故本实现的顺序是 **矢量化(px) → 标定 → 吸附(mm)**。

坐标映射
--------
``X_mm = x_px · scale``，``Z_mm = (z_px − z_ref)·scale·z_sign``，``z_sign`` 默认 +1。
其中 ``x_px`` 已是以轴线为 0 的"像素高"，``z_ref`` 取全轮廓的**最大 z_px**，
即右端面所在列 —— 车削件 Z 原点取在右端面是行业惯例。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import geometry as G
from .errors import CalibrationError


@dataclass
class Calibration:
    scale: float                      # mm / px
    mode: str
    z_ref_px: float = 0.0
    z_sign: float = 1.0
    detail: dict = field(default_factory=dict)
    axis_row_used: float = 0.0        # 轴线行号，仅用于可视化反算

    def to_mm(self, ent):
        s, z0, sg = self.scale, self.z_ref_px, self.z_sign

        def P(p):
            return ((p[0] - z0) * s * sg, p[1] * s)

        out = {'type': ent['type'], 'start': P(ent['start']), 'end': P(ent['end'])}
        if ent['type'] == 'arc':
            out['center'] = P(ent['center'])
            out['radius'] = ent['radius'] * s
            out['ccw'] = bool(ent['ccw'])
        return out

    def to_px(self, ent_mm):
        """``to_mm`` 的逆变换（可视化与报告复现用）。"""
        s, z0, sg = self.scale, self.z_ref_px, self.z_sign

        def P(p):
            return (p[0] / (s * sg) + z0, p[1] / s)

        out = {'type': ent_mm['type'], 'start': P(ent_mm['start']),
               'end': P(ent_mm['end'])}
        if ent_mm['type'] == 'arc':
            out['center'] = P(ent_mm['center'])
            out['radius'] = ent_mm['radius'] / s
            out['ccw'] = bool(ent_mm['ccw'])
        return out


def outline_bbox_px(ents, hint_fmt='Z {zmin:.3f}~{zmax:.3f} mm-px, X 0~{xmax:.3f} px'):
    """轮廓包围盒（px），标定缺失时打印给用户推算用。"""
    if not ents:
        return None
    zs = [p[0] for e in ents for p in (e['start'], e['end'])]
    lo, hi = G.entities_x_extremes(ents)
    return {'z_px': [min(zs), max(zs)], 'x_px': [lo, hi],
            'height_px': hi - lo, 'width_px': max(zs) - min(zs)}


def _require(ents):
    if not ents:
        raise CalibrationError('没有实体可供标定：轮廓提取结果为空')


def from_mm_per_px(ents, mm_per_px, *, z_sign=1.0, z_ref_px=None):
    _require(ents)
    if not (mm_per_px > 0):
        raise CalibrationError(f'--mm-per-px 必须为正，收到 {mm_per_px}')
    z0 = _max_z(ents) if z_ref_px is None else float(z_ref_px)
    lo, hi = G.entities_x_extremes(ents)
    return Calibration(scale=float(mm_per_px), mode='mm-per-px', z_ref_px=z0,
                       z_sign=float(z_sign),
                       detail={'source': 'explicit', 'x_max_px': hi,
                               'x_min_px': lo})


def from_ref_diameter(ents, diameter, *, z_sign=1.0, x_entity_max=None):
    """方式 B：轮廓最大高度 = 半径 → scale = (D/2) / h_px。

    ``x_entity_max`` 允许调用方显式给出"直径对应的最高点"（默认取全轮廓最高）。
    """
    _require(ents)
    if not (diameter > 0):
        raise CalibrationError(f'--ref-diameter 必须为正，收到 {diameter}')
    _, hi = G.entities_x_extremes(ents)
    if x_entity_max is not None:
        hi = float(x_entity_max)
    if hi <= 1e-9:
        raise CalibrationError(
            f'轮廓最大高度仅 {hi:.4f} px，无法由外径推算比例；'
            f'请检查图像是否为空或标定参数是否正确')
    z0 = _max_z(ents)
    scale = (diameter / 2.0) / hi
    return Calibration(scale=scale, mode='ref-diameter', z_ref_px=z0,
                       z_sign=float(z_sign),
                       detail={'source': 'ref-diameter', 'diameter': float(diameter),
                               'h_px': hi, 'pixels_per_mm': 1.0 / scale})


def from_ref_points(ents, p1, p2, ref_mm, *, z_sign=1.0):
    """方式 C：两个已知特征点的实际距离。"""
    _require(ents)
    d_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if d_px <= 1e-9:
        raise CalibrationError('两个参考特征点在像素上重合，无法标定')
    if not (ref_mm > 0):
        raise CalibrationError(f'--ref-mm 必须为正，收到 {ref_mm}')
    z0 = _max_z(ents)
    return Calibration(scale=float(ref_mm) / d_px, mode='ref-points', z_ref_px=z0,
                       z_sign=float(z_sign),
                       detail={'source': 'ref-points', 'p1': list(p1), 'p2': list(p2),
                               'dist_px': d_px, 'ref_mm': float(ref_mm),
                               'pixels_per_mm': d_px / float(ref_mm)})


def _max_z(ents):
    return max(p[0] for e in ents for p in (e['start'], e['end']))


def resolve(ents, *, mm_per_px=None, ref_diameter=None, ref_points=None,
            ref_mm=None, z_sign=1.0):
    """按 CLI 参数选择标定方式；一个都没有就**报错退出**。

    报错信息里带上提取到的轮廓包围盒（px），用户可直接据此推算 ``--mm-per-px``。
    """
    if mm_per_px is not None:
        return from_mm_per_px(ents, mm_per_px, z_sign=z_sign)
    if ref_diameter is not None:
        return from_ref_diameter(ents, ref_diameter, z_sign=z_sign)
    if ref_points is not None and ref_mm is not None:
        (p1, p2) = ref_points
        return from_ref_points(ents, p1, p2, ref_mm, z_sign=z_sign)
    if ref_points is not None or ref_mm is not None:
        raise CalibrationError('--ref-points 与 --ref-mm 必须同时给出')
    bb = outline_bbox_px(ents)
    msg = ['缺少标定参数 —— 拒绝默认假设 1 px = 1 mm。请任选一种方式给出比例：',
           '  --mm-per-px <毫米/像素>      已知比例',
           '  --ref-diameter <毫米>        已知外径（推荐）：取轮廓最大高度 = 半径',
           '  --ref-points x1,y1 x2,y2 --ref-mm <毫米>   已知两特征点距离']
    if bb:
        msg.append('供推算的轮廓包围盒（像素）：'
                   f'Z 跨度 {bb["width_px"]:.1f} px, '
                   f'X 高度 {bb["height_px"]:.1f} px '
                   f'(Z {bb["z_px"][0]:.1f}~{bb["z_px"][1]:.1f}, '
                   f'X {bb["x_px"][0]:.1f}~{bb["x_px"][1]:.1f})')
        msg.append('  例：若该件外径为 Ø40，则 --ref-diameter 40；'
                   f'若已知长度 80 mm，则 --mm-per-px ≈ {80.0 / max(1e-9, bb["width_px"]):.6f}')
    raise CalibrationError('\n'.join(msg))
