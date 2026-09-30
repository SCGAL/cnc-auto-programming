"""⑤ 几何吸附与清理（单位：**毫米**）。

立项文档 §3.2⑤ 的四项规则
--------------------------
==============  ==============================================  ============
操作            规则                                            本实现位置
==============  ==============================================  ============
角度吸附        与 0°/90°/45° 相差 < 3° → 吸附                  本模块
共线/共弧合并    相邻同类型段偏差 < 0.15 mm → 合并                本模块 + vectorize
端点合并        相邻端点距离 < 0.2 mm → 合并                      ``apply_junctions``
闭合检查        首尾端点距离 > 0.5 mm → **报警**（不静默补线）    本模块
==============  ==============================================  ============

关于"端点合并"的实现位置
------------------------
本实现让端点在**矢量化阶段**就取"相邻实体的几何交点"（``apply_junctions``），
于是相邻端点**恒等**、间隙恒为 0 —— 比"事后把两个接近的点挪到中点"更严格：
既消除了间隙，又拿到了亚像素端点。本模块因此只需要做"间隙超限则报警"。

关于"闭合"的语义（与文档口径的必要澄清）
----------------------------------------
车削上半部外轮廓是一条**开路径**：首尾两端都落在轴线上、彼此相距整个零件长度
（本项目的合成件是 67~80 mm）。若按"多边形首尾闭合"判定，合格零件会被一律判失败。
文档 §3.2⑤ 讲的显然是 DXF 版的教训（``chain()`` 的断点检查），即**链条连续性**。
故本实现把"轮廓不闭合"定义为：

    **相邻实体之间的连接间隙 > 0.5 mm → 报错退出**（``--close-tol`` 可配）

这与 DXF 版 ``JOIN_TOL = 0.05`` 的串联检查同源，且能真正抓住"提取断链"这类故障。
"""

from __future__ import annotations

import math

import numpy as np

from . import geometry as G
from .errors import (ContourDirectionError, DiscontinuousContourError,
                     EmptyContourError)

SNAP_ANGLES_DEG = (0.0, 45.0, 90.0, 135.0)


def check_non_empty(ents):
    """实体数为 0 必须明确报错（约束 5）。"""
    if not ents:
        raise EmptyContourError(
            '识别到的实体数为 0 —— 图像中没有可用的直线/圆弧轮廓。'
            '请检查：是否为干净线框图、是否整幅都是背景、'
            'Otsu 阈值是否把墨迹判成了背景（可试 --polarity dark/light）')
    return ents


def entity_length(e):
    """实体长度（mm）。"""
    if e['type'] == 'line':
        return G.dist(e['start'], e['end'])
    return abs(G.arc_sweep(e['center'], e['start'], e['end'], e['ccw'])) * e['radius']


def absorb_short_segments(ents, min_len_mm=0.8, hard_len_mm=0.35, protect_ends=True):
    """吸收过短的中间段，让相邻两段直接求交。

    细化在 90° 拐角处会把骨架"抹"出一小段（实测右端面与 Ø40 外圆的拐角被切成
    0.43~0.63 mm 的伪段）。它既不是真特征，也不算断裂，但会让顶点集合多出一个点，
    把端点误差顶到 1.16 mm。删掉它、让端面与外圆按**几何交点**相连，拐角就精确
    落回真值顶点（实测 (0,20) 误差 0.43 → 0.03 mm）。

    ★ 为什么不能只看长度（这是本项目最容易误伤的地方）
    --------------------------------------------------
    单纯按长度删会误杀真特征：肩面被曲率切分切成不足阈值的短段时会被删掉，
    两条圆角重新合成一条弧。故分两级判据：

    * ``L < hard_len_mm`` —— 明显是噪声，直接吸收；
    * ``hard_len_mm ≤ L < min_len_mm`` —— 必须先过**几何可行性闸门**：
      只有当左右两段**真的存在几何交点**、且该交点就在这条短段附近时才吸收。
      折角伪段的两侧是端面与外圆，必然相交；而被肩面隔开的两条圆角，其拟合圆的
      圆心距略大于 R1+R2，求交为空 —— 恰好把两类分开。

    实测（``px10_w2_s+1_r2``）：拐角伪弧段长 0.625 mm，刚好越过旧的 0.6 mm 单阈值，
    导致段数被顶到 9/9 却挤掉了真正的肩面，端点误差 1.16 mm。加闸门后该伪段被吸收。

    仅在**两端都不是首尾实体**时生效：首尾段是轮廓的入口/出口，删掉会改变 Z 基准。
    """
    if min_len_mm <= 0:
        return [dict(e) for e in ents], 0
    out = [dict(e) for e in ents]
    lo = 1 if protect_ends else 0
    changed = 0
    i = lo
    while i < len(out) - lo:
        L = entity_length(out[i])
        if L >= min_len_mm:
            i += 1
            continue
        if L >= hard_len_mm:
            cands = G.junction_candidates(out[i - 1], out[i + 1])
            if not cands:
                i += 1                      # 左右无几何交点 → 真特征，保留
                continue
            mid = ((out[i]['start'][0] + out[i]['end'][0]) / 2.0,
                   (out[i]['start'][1] + out[i]['end'][1]) / 2.0)
            if min(G.dist(c, mid) for c in cands) > 2.5 * L + hard_len_mm:
                i += 1                      # 交点离得太远 → 不是折角伪段
                continue
        del out[i]
        changed += 1
    return out, changed


def drop_degenerate(ents, min_len_mm=0.02):
    """剔除长度近零的退化实体（会让生成器连不上而报断点）。"""
    return [e for e in ents if entity_length(e) >= min_len_mm]


def _ensure_line_basis(ents):
    """保证每条 line 都有 ``_dirn`` / ``_point``（外部构造的实体可能只给了端点）。"""
    out = []
    for e in ents:
        f = dict(e)
        if f['type'] == 'line' and (f.get('_dirn') is None or f.get('_point') is None):
            dx = f['end'][0] - f['start'][0]
            dy = f['end'][1] - f['start'][1]
            L = math.hypot(dx, dy)
            f['_dirn'] = (dx / L, dy / L) if L > 1e-12 else (1.0, 0.0)
            f['_point'] = ((f['start'][0] + f['end'][0]) / 2.0,
                           (f['start'][1] + f['end'][1]) / 2.0)
        out.append(f)
    return out


def snap_angles(ents, tol_deg=3.0, angles=SNAP_ANGLES_DEG):
    """直线方向吸附到 0°/45°/90°/135°。

    绕直线自身质心旋转 —— 这样**位置（截距）保持不变**，只改方向：
    水平外圆的 X 高度、端面的 Z 位置都不会被吸附改变。
    """
    if tol_deg <= 0:
        # 早退也不能让实体缺 _dirn/_point —— 否则 merge_collinear_lines 会 KeyError。
        # （评审实测：snap.run(bare_entities, angle_tol_deg=0.0) 直接崩）
        return _ensure_line_basis(ents), 0
    out, n = [], 0
    for e in ents:
        if e['type'] != 'line':
            out.append(dict(e))
            continue
        f = dict(e)
        d = f.get('_dirn')
        if d is None:
            # 外部构造的实体可能只给了端点 —— 从端点反推方向，而不是跳过
            # （跳过会让后续 merge_collinear_lines 直接 KeyError）
            dx = e['end'][0] - e['start'][0]
            dy = e['end'][1] - e['start'][1]
            L = math.hypot(dx, dy)
            d = (dx / L, dy / L) if L > 1e-12 else (1.0, 0.0)
            f['_dirn'] = d
            f['_point'] = ((e['start'][0] + e['end'][0]) / 2.0,
                           (e['start'][1] + e['end'][1]) / 2.0)
        ang = math.degrees(math.atan2(d[1], d[0])) % 180.0
        best, bd = None, tol_deg
        for a in angles:
            dd = min(abs(ang - a), 180.0 - abs(ang - a))
            if dd <= bd:
                best, bd = a, dd
        if best is not None:
            r = math.radians(best)
            nd = (math.cos(r), math.sin(r))
            # ★ 方向必须与原有"起点→终点"朝向一致：θ 只确定到 mod 180°，
            #   直接取 (cosθ, sinθ) 会把竖直端面等线段整个掉头
            #   （实测把 4 mm 的左端面塌成 0.19 mm）。
            d0 = (e['end'][0] - e['start'][0], e['end'][1] - e['start'][1])
            if nd[0] * d0[0] + nd[1] * d0[1] < 0:
                nd = (-nd[0], -nd[1])
            # 保持线段中点在原处：绕质心旋转
            c = f['_point']
            half = G.dist(e['start'], e['end']) / 2.0
            f['_dirn'] = nd
            f['_point'] = tuple(c)
            f['start'] = (c[0] - nd[0] * half, c[1] - nd[1] * half)
            f['end'] = (c[0] + nd[0] * half, c[1] + nd[1] * half)
            f['_snapped_deg'] = best
            n += 1
        out.append(f)
    return out, n


def merge_collinear_lines(ents, tol_mm=0.15, angle_tol_deg=1.5):
    """相邻共线直线合并（单位 mm）。

    矢量化阶段已按像素容差合并过一次；此处按文档要求的毫米容差再兜一次底。
    """
    out = [dict(e) for e in ents]
    changed = 0
    i = 0
    while i < len(out) - 1:
        a, b = out[i], out[i + 1]
        if a['type'] != 'line' or b['type'] != 'line':
            i += 1
            continue
        da = np.asarray(a['_dirn'], float)
        db = np.asarray(b['_dirn'], float)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(da @ db))))))
        if ang > angle_tol_deg:
            i += 1
            continue
        n = np.array([-da[1], da[0]])
        off = abs(float((np.asarray(b['_point'], float) -
                         np.asarray(a['_point'], float)) @ n))
        if off > tol_mm:
            i += 1
            continue
        p0, p1 = a['start'], b['end']
        out[i] = {'type': 'line', 'start': p0, 'end': p1,
                  '_point': ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0),
                  '_dirn': (p1[0] - p0[0], p1[1] - p0[1]),
                  '_rms': max(a.get('_rms', 0.0), b.get('_rms', 0.0)),
                  '_n_pts': a.get('_n_pts', 0) + b.get('_n_pts', 0),
                  '_merged': True}
        L = G.dist(p0, p1)
        if L > 1e-12:
            out[i]['_dirn'] = ((p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L)
        del out[i + 1]
        changed += 1
    return out, changed


def merge_coincident_arcs(ents, tol_center_mm=0.15, tol_radius_mm=0.1):
    """相邻同圆心同半径圆弧合并（mm 口径兜底）。"""
    out = [dict(e) for e in ents]
    changed = 0
    i = 0
    while i < len(out) - 1:
        a, b = out[i], out[i + 1]
        if a['type'] != 'arc' or b['type'] != 'arc':
            i += 1
            continue
        if (G.dist(a['center'], b['center']) <= tol_center_mm
                and abs(a['radius'] - b['radius']) <= tol_radius_mm):
            out[i] = {**a, 'end': b['end'], '_merged': True}
            del out[i + 1]
            changed += 1
            continue
        i += 1
    return out, changed


def chain_gap_max(ents):
    gaps = G.chain_gaps(ents)
    return max((g for _, g in gaps), default=0.0)


def verify_continuity(ents, close_tol_mm=0.5):
    """闭合（连续性）检查：任一段间间隙 > close_tol_mm → 报错，绝不静默补线。"""
    gaps = G.chain_gaps(ents)
    bad = [(i, g) for i, g in gaps if g > close_tol_mm]
    if bad:
        detail = '；'.join(
            f'第 {i} 段与第 {i+1} 段之间间隙 {g:.4f} mm '
            f'(Z{ents[i-1]["end"][0]:.3f}X{ents[i-1]["end"][1]:.3f} → '
            f'Z{ents[i]["start"][0]:.3f}X{ents[i]["start"][1]:.3f})'
            for i, g in bad[:5])
        raise DiscontinuousContourError(
            f'轮廓不闭合：{len(bad)} 处连接间隙超过 {close_tol_mm} mm 容差。{detail}\n'
            f'提示：多为提取断链（笔画中断、噪点切断、小连通域被误删）。'
            f'可试 --no-denoise / 减小 --min-area / 增大闭运算半径 --close-radius。')
    return chain_gap_max(ents)


def horizontal_tilt_deg(ents, min_len_mm=2.0, max_deg=45.0):
    """回检（R3）：最长**近水平**直线的实际倾角，用于外层倾斜迭代与报告。

    ★ 必须取直线**自身的拟合方向** ``_dirn``，不能用两端点夹角
    ---------------------------------------------------------
    端点是通过 :func:`~i2g.geometry.junction_point` 与邻段求交得到的；在圆弧相切处，
    切点的 X 与该直线的高度并不严格一致（相差 |R−d|），于是端点夹角会混入邻段偏差。
    实测：完全无倾斜的合成图上，端点法报出 +0.080° 的伪倾角，并在按此修正后
    涨到 +0.247°（越修越歪）。改用最小二乘方向后该偏差消失（同一张图报 0.002°）。

    只统计与水平夹角不超过 ``max_deg`` 的段：车削件的竖直端面往往是最长的单段
    （本项目合成件右端面 20 mm），不过滤会把 90° 当成"倾斜未校正"。
    """
    best, best_len = 0.0, 0.0
    for e in ents:
        if e['type'] != 'line':
            continue
        L = G.dist(e['start'], e['end'])
        if L < min_len_mm:
            continue
        d = e.get('_dirn')
        if d is not None:
            ang = math.degrees(math.atan2(d[1], d[0]))
        else:
            ang = math.degrees(math.atan2(e['end'][1] - e['start'][1],
                                          e['end'][0] - e['start'][0]))
        ang = (ang + 90.0) % 180.0 - 90.0
        if abs(ang) > max_deg:
            continue
        if L > best_len:
            best_len, best = L, ang
    return best, best_len


def snap_axis_endpoints(ents, rel_tol=0.05):
    """把首/末实体落在轴线上的端点 X 钉到 0。

    ★ 为什么必须钉（这是一项被长期漏掉的系统性偏差）
    ------------------------------------------------
    车削上半部外轮廓的**两个端点必然落在轴线 X=0 上** —— 这是本项目的领域先验，
    也是合成真值与 ``check_truth`` 的前提。但骨架在这两点上的 X 有系统性抬升：
    实测 +0.17 px（w=1）/+0.44 px（w=2）/+1.16 px（w=3）@10 px/mm，
    即线宽 3 px 时 **0.14 mm，占 0.2 mm 预算的 70%**，且随像素密度同步增长。

    为什么现有机制修不了它：
      * 沿法线的亚像素精化修不了 X —— 竖端面的法线是水平的，只能修 Z；
      * 盖帽质量守恒只修正了"轴线基准行"，端点自身仍停在骨架末端；
      * 外径标定把轮廓最大 X 钉死，于是这个偏差被"藏"在标定里，任何自检都看不见。

    安全边界：只对**近竖直**的直线端点生效（|ΔX| ≥ |ΔZ|），且仅当该点 X 已经很小
    （``|X| ≤ rel_tol × 轮廓最大 X``）时才钉 —— 若图样本身不到轴线，不会被误伤。
    """
    if not ents:
        return [dict(e) for e in ents], 0
    out = [dict(e) for e in ents]
    max_x = max(abs(p[1]) for e in out for p in (e['start'], e['end']))
    tol = max(1e-9, rel_tol * max_x)
    n = 0
    for idx, key in ((0, 'start'), (len(out) - 1, 'end')):
        e = out[idx]
        if e['type'] != 'line':
            continue
        p = e[key]
        if abs(p[1]) > tol:
            continue
        other = e['end'] if key == 'start' else e['start']
        if abs(other[1] - p[1]) < abs(other[0] - p[0]):
            continue                       # 更像水平段，不动
        f = dict(e)
        f[key] = (p[0], 0.0)
        dx = f['end'][0] - f['start'][0]
        dy = f['end'][1] - f['start'][1]
        L = math.hypot(dx, dy)
        if L < 1e-12:
            continue
        f['_dirn'] = (dx / L, dy / L)
        f['_point'] = ((f['start'][0] + f['end'][0]) / 2.0,
                       (f['start'][1] + f['end'][1]) / 2.0)
        f['_axis_pinned'] = True
        out[idx] = f
        n += 1
    return out, n


def verify_z_monotonic(ents, tol_mm=0.2):
    """刀路 Z 必须**单调不增**（车削上半部外轮廓的铁律）。

    ★ 这是全项目最便宜的护栏，原来漏掉了
    ------------------------------------
    车削上半部外轮廓沿刀路 Z 只会从右端面（Z=0）向负方向走，永不回头。
    没有这条检查时，一条"来回切两遍"的折返轮廓可以：``verify_continuity`` 通过
    （折返链是严格连续的）、退出码 0、8 项契约校验过 7 项、C4 重复段也没抓到
    （反向副本的端点差约 0.1 mm，3 位小数去重去不掉）—— 即**静默输出一份会把零件
    切两遍的程序**。评审用 ``--ink fill`` 喂线框图复现了这一点。

    Z 回调超过 ``tol_mm`` 即报错（不静默修正）。
    """
    bad = []
    for k, e in enumerate(ents):
        z0, z1 = e['start'][0], e['end'][0]
        if z1 > z0 + tol_mm:
            bad.append((k, z0, z1))
    if bad:
        k, z0, z1 = bad[0]
        raise ContourDirectionError(
            f'刀路 Z 非单调：第 {k} 段 Z 从 {z0:.3f} 回调到 {z1:.3f} '
            f'（容差 {tol_mm} mm），共 {len(bad)} 处。\n'
            f'车削上半部外轮廓沿刀路 Z 只会向负方向单调前进；回调意味着'
            f'轮廓折返（例如多视图混入、标注线干扰、或 --ink 模式选错）。'
            f'拒绝输出可能把零件切两遍的程序。')
    return max((e['end'][0] - e['start'][0] for e in ents), default=0.0)


def run(ents, *, angle_tol_deg=3.0, collinear_tol_mm=0.15,
        arc_center_tol_mm=0.15, arc_radius_tol_mm=0.1,
        close_tol_mm=0.5, tilt_warn_deg=0.3, min_len_mm=0.02,
        absorb_len_mm=0.8, absorb_hard_len_mm=0.35,
        axis_rel_tol=0.05, z_mono_tol_mm=0.2):
    """吸附全流程。返回 ``(entities, info)``。"""
    check_non_empty(ents)
    info = {'n_in': len(ents)}

    ents = [dict(e) for e in ents]
    # ★ 先记录**吸附前**的最长近水平外圆实际倾角：吸附会把 3° 内的一律拉到 0°，
    #   吸附后再测就永远是 0，反馈信号就没了。该值是外层倾斜迭代的观测量。
    raw_tilt, raw_tilt_len = horizontal_tilt_deg(ents)
    info['raw_horizontal_tilt_deg'] = raw_tilt
    info['raw_horizontal_len_mm'] = raw_tilt_len

    ents, n_snap = snap_angles(ents, tol_deg=angle_tol_deg)
    info['n_angle_snapped'] = n_snap

    ents, n_lin = merge_collinear_lines(ents, tol_mm=collinear_tol_mm)
    ents, n_arc = merge_coincident_arcs(ents, tol_center_mm=arc_center_tol_mm,
                                        tol_radius_mm=arc_radius_tol_mm)
    info['n_lines_merged'] = n_lin
    info['n_arcs_merged'] = n_arc

    ents, n_absorb = absorb_short_segments(ents, min_len_mm=absorb_len_mm,
                                           hard_len_mm=absorb_hard_len_mm)
    info['n_absorbed'] = n_absorb
    info['absorb_len_mm'] = absorb_len_mm

    # ★ 结合点重算闸门与最终连续性容差必须**解耦**：
    #   `absorb_short_segments` 会把短段并进邻段，**相邻两段的端点会各自被位移**，
    #   因此它造成的最大端点间距是 2 × `absorb_len_mm`（默认 0.8 → 1.6 mm）；
    #   而 `apply_junctions` 的语义是"原始两端点间距 > join_gap_tol 即判为真断口、跳过重算"。
    #   若此处沿用 `close_tol_mm`（默认 0.5），被 absorb 推开的端点永远得不到重算，
    #   紧接着 `verify_continuity(close_tol_mm)` 直接报"轮廓不闭合"（实测间隙 0.51~0.94 mm）。
    #   安全性：本闸门只决定"要不要尝试重算结合点"；`apply_junctions` 内部仍要求两条原语
    #   存在**真实交点**才会落点，真断口不会被凭空接上；最终连续性仍按 `close_tol_mm` 严格判定。
    join_gap_tol = max(close_tol_mm, 2.0 * absorb_len_mm)

    from .vectorize import apply_junctions
    ents = apply_junctions(ents, join_gap_tol=join_gap_tol)

    ents = drop_degenerate(ents, min_len_mm=min_len_mm)
    if not ents:
        raise EmptyContourError('吸附清理后实体数为 0（全部被判为退化实体）')
    ents = apply_junctions(ents, join_gap_tol=join_gap_tol)

    # 轴线端点按领域先验钉到 X=0（消除骨架端点的系统性 X 抬升）
    ents, n_pin = snap_axis_endpoints(ents, rel_tol=axis_rel_tol)
    info['n_axis_pinned'] = n_pin
    if n_pin:
        ents = apply_junctions(ents, join_gap_tol=join_gap_tol)

    info['n_out'] = len(ents)
    info['n_lines'] = sum(1 for e in ents if e['type'] == 'line')
    info['n_arcs'] = sum(1 for e in ents if e['type'] == 'arc')
    info['chain_gap_max_mm'] = verify_continuity(ents, close_tol_mm=close_tol_mm)
    info['z_nonmonotonic_max_mm'] = verify_z_monotonic(ents, tol_mm=z_mono_tol_mm)
    info['close_tol_mm'] = close_tol_mm
    info['join_gap_tol_mm'] = join_gap_tol
    tilt, tilt_len = horizontal_tilt_deg(ents)
    info['longest_horizontal_tilt_deg'] = tilt
    info['longest_horizontal_len_mm'] = tilt_len
    if tilt_len >= 2.0 and abs(tilt) > tilt_warn_deg:
        info.setdefault('warnings', []).append(
            f'最长水平外圆倾角 {tilt:+.3f}°，超过 {tilt_warn_deg}° —— '
            f'倾斜校正可能不彻底（R3）')
    return ents, info
