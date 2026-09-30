"""流水线编排：图像 → 实体 → G 代码 + 全链路诊断。

    ① preprocess → ② centerline → ③ profile → ④ vectorize
    → ⑥ calibrate → ⑤ snap → ⑦ emit

（⑤⑥ 的相对顺序与文档不同，理由见 ``calibrate`` 模块 docstring：
吸附容差以 mm 给出，必须先标定。）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import calibrate, centerline, emit, preprocess, profile, snap, vectorize, viz
from .errors import I2GError


@dataclass
class Params:
    # ---- 预处理
    denoise: str = 'isolated'
    min_area: int = 40
    close_radius: int = 1
    threshold: int | None = None
    polarity: str = 'auto'
    # ---- 基准与轮廓
    ink: str = 'line'
    deskew: bool = True
    deskew_max_deg: float = 3.0
    deskew_refine_iters: int = 2          # 用已提取几何反哺倾斜的外层迭代次数
    deskew_refine_tol_deg: float = 0.03   # 残余倾角小于此值即停止迭代
    subpixel: bool = True                # 沿法线做覆盖率加权质心，取亚像素中线
    # ---- 矢量化
    vector: vectorize.VectorizeParams = field(default_factory=vectorize.VectorizeParams)
    # ---- 标定
    mm_per_px: float | None = None
    ref_diameter: float | None = None
    ref_points: list | None = None
    ref_mm: float | None = None
    z_sign: float = 1.0
    # ---- 吸附
    angle_tol_deg: float = 3.0
    collinear_tol_mm: float = 0.15
    close_tol_mm: float = 0.5
    # ---- 输出
    gcode: dict = field(default_factory=dict)
    viz_dir: str | None = None
    viz_prefix: str = 'stage'
    source_name: str = 'image'


def model_score(pts_px, ents_px):
    """向量模型对轮廓点列的自洽度（像素）：轮廓点到实体链的均方距离，越小越好。

    这是**不需要真值**的质量观测量，用于判断"再旋转一次"到底有没有帮助。
    实测：外层倾斜迭代在真正有倾斜的图上收益显著，但在本来就干净的图上，
    观测量里那点 0.05° 级噪声会让它多转一次、多一次旋转重采样，反而把
    端点误差从 0.11 mm 拖到 0.53 mm。故必须"改进了才采纳"。
    """
    from . import geometry as G
    P = np.asarray(pts_px, dtype=float)
    if len(P) == 0:
        return float('inf')
    poly = G.entities_polyline(ents_px, max_step=0.25)
    Q = np.asarray(poly, dtype=float)
    if len(Q) < 2:
        return float('inf')
    tot, n = 0.0, 0
    for i in range(0, len(P), 512):
        blk = P[i:i + 512]
        d = np.hypot(blk[:, None, 0] - Q[None, :, 0],
                     blk[:, None, 1] - Q[None, :, 1]).min(axis=1)
        tot += float((d ** 2).sum())
        n += len(blk)
    return float(np.sqrt(tot / max(1, n)))


def run_from_gray(gray, params: Params):
    """跑完整流水线，并对倾斜校正做外层迭代（采纳采用"改进了才要"的准则）。

    ★ 为什么要迭代
    --------------
    投影直方图法的角度分辨力约 0.05~0.1°，而**只要 0.1° 的残余倾斜**（690 px 宽上
    造成约 1.2 px 的纵向漂移）就足以让"水平外圆"在区域生长/曲率切分里显出弯曲，
    切出伪圆弧 —— 实测 skew=+2° 时弧数从 2 变 4、端点误差从 0.05 mm 涨到 1.01 mm。

    做法：第一遍跑完后，用**已提取的最长水平外圆的实际倾角**（吸附前的原始值）
    作为观测量，反向旋转重跑；再用 :func:`model_score` 比较两遍的自洽度，
    **只有确实变好才采纳**（否则保留第一遍）。

    每遍都从**原始预处理结果**一次性旋转，不做二次旋转，避免重复重采样。
    返回 ``res`` dict。
    """
    res = _run_once(gray, params, extra_deg=0.0)
    trace = [{'iter': 0, 'extra_deg': 0.0,
              'theta_deg': round(res['stats']['centerline']['theta_deg'], 5),
              'raw_tilt_deg': _round(res['stats']['snap'].get('raw_horizontal_tilt_deg')),
              'model_score_px': round(res['stats']['vectorize'].get('model_score_px', 0), 4),
              'accepted': True}]
    if not params.deskew:
        res['stats']['deskew_refine'] = trace
        return res

    best = res
    extra = 0.0
    for it in range(1, max(1, int(params.deskew_refine_iters)) + 1):
        tilt = best['stats']['snap'].get('raw_horizontal_tilt_deg')
        if tilt is None or abs(tilt) <= params.deskew_refine_tol_deg:
            break
        # 提取几何的 (Z,X) 平面内倾角 → PIL 旋转需施加 −tilt
        trial = extra - float(tilt)
        cand = _run_once(gray, params, extra_deg=trial)
        new_tilt = cand['stats']['snap'].get('raw_horizontal_tilt_deg')
        # ★ 采纳准则 = "残余倾角确实变小"。不能用"模型自洽度"：过分割的模型
        #   总能拟合得更好（实测角度修正后自洽度反而从 0.124 变 0.203，
        #   把正确修正误判成退步）。倾角观测量本身已在干净图上验证为 0.000°，
        #   不会无故触发。
        improved = new_tilt is not None and abs(new_tilt) < abs(tilt) * 0.8
        trace.append({'iter': it, 'extra_deg': round(trial, 5),
                      'theta_deg': round(cand['stats']['centerline']['theta_deg'], 5),
                      'raw_tilt_deg': _round(new_tilt),
                      'prev_tilt_deg': _round(tilt),
                      'model_score_px': round(cand['stats']['vectorize'].get('model_score_px', 0), 4),
                      'accepted': bool(improved)})
        if not improved:
            break
        extra, best = trial, cand
    best['stats']['deskew_refine'] = trace
    return best


def _round(v):
    try:
        return round(float(v), 5)
    except Exception:                                        # noqa: BLE001
        return None


def _run_once(gray, params: Params, *, extra_deg=0.0):
    """单趟流水线。``extra_deg`` 为叠加在直方图估计之上的残余倾角修正量。"""
    warnings: list[str] = []
    stats: dict = {}

    # ① 预处理
    pre = preprocess.run(gray, denoise=params.denoise, min_area=params.min_area,
                         close_radius=params.close_radius,
                         threshold=params.threshold, polarity=params.polarity)
    ink0 = pre['ink']
    stats['preprocess'] = pre['stats']
    stats['preprocess']['n_components_raw'] = pre['n_components']
    if not ink0.any():
        from .errors import EmptyContourError
        raise EmptyContourError(
            '预处理后图像中没有剩余墨迹：可能整幅被 Otsu 判为同类，'
            '或所有连通域都被面积阈值 --min-area 剔除。'
            f'（阈值 {pre["threshold"]}，剔除面积 {pre["stats"]["dropped_area_px"]} px）')

    # ② 倾斜校正
    cl = centerline.run(ink0, pre.get('gray01'), enabled=params.deskew,
                        max_deg=params.deskew_max_deg, extra_deg=extra_deg)
    ink = cl['ink']
    warnings += cl['warnings']
    stats['centerline'] = {'theta_deg': cl['theta_deg'], 'contrast': cl['contrast'],
                           'reliable': cl['reliable'],
                           'theta_hist_deg': cl.get('theta_hist_deg'),
                           'extra_deg': cl.get('extra_deg')}
    if not ink.any():
        from .errors import EmptyContourError
        raise EmptyContourError('倾斜校正后图像为空（旋转结果异常）')

    # ③ 轮廓提取（含亚像素中线精化）
    pr = profile.extract(ink, mode=params.ink, gray01=cl.get('gray01'),
                         subpixel=params.subpixel)
    pts_px = pr['points']
    axis_row = pr['axis_row']
    stats['profile'] = dict(pr['diag'])

    # ④ 矢量化（像素单位）
    ents_px, vdiag = vectorize.vectorize(pts_px, params.vector)
    snap.check_non_empty(ents_px)          # 实体数为 0 → 明确报错
    stats['vectorize'] = {k: v for k, v in vdiag.items() if k != 'segments'}
    stats['vectorize']['model_score_px'] = model_score(pts_px, ents_px)
    stats['vectorize_segments'] = vdiag['segments']

    # ⑥ 标定（像素 → 毫米）
    cal = calibrate.resolve(ents_px, mm_per_px=params.mm_per_px,
                            ref_diameter=params.ref_diameter,
                            ref_points=params.ref_points, ref_mm=params.ref_mm,
                            z_sign=params.z_sign)
    cal.axis_row_used = float(axis_row)
    stats['calibration'] = {'mode': cal.mode, 'mm_per_px': cal.scale,
                            'pixels_per_mm': 1.0 / cal.scale,
                            'z_ref_px': cal.z_ref_px, 'detail': cal.detail}
    ents_mm = [cal.to_mm(e) for e in ents_px]
    # 内部拟合参数（px）按比例搬到 mm，供吸附使用
    for src, dst in zip(ents_px, ents_mm):
        if src['type'] == 'line':
            dst['_point'] = ((src['_point'][0] - cal.z_ref_px) * cal.scale * cal.z_sign,
                             src['_point'][1] * cal.scale)
            dst['_dirn'] = (src['_dirn'][0] * cal.z_sign, src['_dirn'][1])
            dst['_rms'] = src.get('_rms', 0.0) * cal.scale
            dst['_n_pts'] = src.get('_n_pts', 0)

    # ⑤ 吸附（毫米单位）
    ents_mm, sinfo = snap.run(
        ents_mm, angle_tol_deg=params.angle_tol_deg,
        collinear_tol_mm=params.collinear_tol_mm, close_tol_mm=params.close_tol_mm)
    warnings += sinfo.pop('warnings', [])
    stats['snap'] = sinfo

    # ⑦ 输出 G 代码
    out = emit.emit(ents_mm, source_name=params.source_name, **params.gcode)
    stats['emit'] = {'n_entities': out['n_entities'],
                     'n_dupes_removed': out['n_dupes_removed'],
                     'n_moves': out['n_moves'], 'safe_x': out['safe_x'],
                     'generator': out['generator_path']}

    if params.viz_dir:
        _dump_viz(params, gray, ink0, ink, ents_px, ents_mm, cal, axis_row)

    return {'entities_mm': ents_mm, 'entities_px': ents_px, 'calibration': cal,
            'stages': {'ink_raw': ink0, 'ink': ink, 'axis_row': axis_row},
            'profile_points': pts_px, 'stats': stats, 'warnings': warnings,
            'gcode': out['gcode'], 'gcode_report': out['report'],
            'gcode_ordered': out['ordered']}


def _dump_viz(params, gray, ink0, ink, ents_px, ents_mm, cal, axis_row):
    d = params.viz_dir
    p = lambda n: os.path.join(d, f'{params.viz_prefix}_{n}.png')      # noqa: E731
    viz.save_mask(gray > preprocess.otsu_threshold(gray), p('01_binary'))
    viz.save_mask(ink0, p('02_cleaned'))
    viz.save_mask(ink, p('03_deskewed'))
    viz.overlay(ink, ents_px=ents_px, axis_row=axis_row, path=p('04_profile_px'))
    viz.overlay(ink, ents_mm=ents_mm, calib=cal, path=p('05_profile_mm'))
    return d


def run_from_image(path, params: Params):
    """读图 → 流水线。``params.source_name`` 缺省用文件名。"""
    gray = preprocess.load_gray(path)
    if not params.source_name or params.source_name == 'image':
        params.source_name = os.path.basename(path)
    return run_from_gray(gray, params)


def run_from_mask(mask, params: Params):
    """直接喂 bool 二值图（跳过文件 I/O；无覆盖率信息，亚像素精化自动退化为关）。"""
    gray = np.where(np.asarray(mask, dtype=bool), 0, 255).astype(np.uint8)
    return run_from_gray(gray, params)


def run_from_coverage(cov01, params: Params):
    """喂归一化覆盖率图（0..1 的 float 数组，抗锯齿渲染结果）。"""
    gray = np.clip(np.round(np.asarray(cov01, dtype=np.float64) * 255.0),
                   0, 255).astype(np.uint8)
    return run_from_gray(gray, params)


def run_from_png(path, params: Params):
    """读任意灰度/彩色 PNG → 流水线。"""
    return run_from_image(path, params)
