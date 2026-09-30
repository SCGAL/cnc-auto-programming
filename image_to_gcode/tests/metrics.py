"""6 项精度指标比对框架（M1 交付物 2/2）。

指标定义（与立项文档 §6.2 一致）
--------------------------------
=================  ==========================================  ==========
指标                定义                                        目标
=================  ==========================================  ==========
``endpoint_err``   提取顶点与真值顶点的双向最近距离之最大值      ≤ 0.2 mm
``center_err``     匹配上的圆弧圆心距离之最大值                 ≤ 0.2 mm
``radius_err``     匹配上的圆弧半径之差之最大值                 ≤ 0.2 mm
``type_acc``       直线/圆弧类型判别正确的段数 / 真值段数        ≥ 0.90
``arc_recall``     真值圆弧中被识别为圆弧的比例                 ≥ 0.90
``hausdorff``      两条轮廓点列的豪斯多夫距离                   ≤ 0.30 mm
=================  ==========================================  ==========

半径门槛口径
------------
用户简报写"圆心/半径误差 ≤ 0.2 mm"，立项文档 §6.2 指标表写 ``radius_err ≤ 0.1``。
二选一时按**用户简报**取 0.2 mm 作硬门槛；文档更严的 0.1 mm 作为
:data:`STRICT_RADIUS_TARGET` 单独披露（见 ``strict_radius_ok``），不隐藏差异。

关于 Z 方向"规范自由度"（gauge freedom）—— 方法学说明
-----------------------------------------------------
车削件的图样不给定绝对 Z 基准：Z 原点由编程者按习惯（通常是右端面）设定。
因此 **Z 方向的整体平移不是误差**，是坐标规范。本框架因此：

1. 先按约定把两者的右端面对齐（``align='convention'``）；
2. 再允许沿 Z 做**只平移不旋转**的 ICP 精对齐（``align='icp'``，默认），
   并在结果里如实报告所用的总平移量 ``gauge_shift_mm``。

X 方向**没有**这种自由度（X=0 是物理轴线），故 X 不做对齐。
同时额外给出 ``endpoint_err_convention``（只用约定对齐、零 ICP 平移），
以便区分"形状误差"与"Z 基准约定差异"。

本模块只做度量，不含任何提取算法。
"""

from __future__ import annotations

import math

import numpy as np

from i2g import geometry as G

# ------------------------------------------------------------------ 目标值

# name -> (比较方向, 阈值, 单位)
TARGETS = {
    'endpoint_err': ('<=', 0.20, 'mm'),
    'center_err':   ('<=', 0.20, 'mm'),
    'radius_err':   ('<=', 0.20, 'mm'),
    'type_acc':     ('>=', 0.90, ''),
    'arc_recall':   ('>=', 0.90, ''),
    'hausdorff':    ('<=', 0.30, 'mm'),
}
STRICT_RADIUS_TARGET = 0.10          # 立项文档 §6.2 的更严档（仅披露，不作硬门槛）

# 实体配对的最大中点距离（mm）。见 ``match_entities`` 的说明：无上限会让
# "张冠李戴的圆弧配对"在前四项指标上同时满分 —— 那是个自洽闭环漏洞。
DEFAULT_MAX_PAIR_DIST = 1.0

METRIC_ORDER = ['endpoint_err', 'center_err', 'radius_err',
                'type_acc', 'arc_recall', 'hausdorff']


# ------------------------------------------------------------------ 基础工具

def _as_array(pts):
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def _nn_dist(A, B, chunk=512):
    """A 中每个点到 B 中最近点的距离。分块计算控制内存。"""
    A, B = _as_array(A), _as_array(B)
    if len(A) == 0 or len(B) == 0:
        return np.full(len(A), np.inf)
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        blk = A[i:i + chunk]
        d = np.hypot(blk[:, None, 0] - B[None, :, 0], blk[:, None, 1] - B[None, :, 1])
        out[i:i + chunk] = d.min(axis=1)
    return out


def _nn_index(A, B, chunk=512):
    """A 中每个点在 B 中的最近点下标。"""
    A, B = _as_array(A), _as_array(B)
    out = np.zeros(len(A), dtype=int)
    for i in range(0, len(A), chunk):
        blk = A[i:i + chunk]
        d = np.hypot(blk[:, None, 0] - B[None, :, 0], blk[:, None, 1] - B[None, :, 1])
        out[i:i + chunk] = d.argmin(axis=1)
    return out


def entity_midpoint(ent):
    """实体中点（圆弧取**弧上**中点，不是弦中点）。"""
    if ent['type'] == 'line':
        return ((ent['start'][0] + ent['end'][0]) / 2.0,
                (ent['start'][1] + ent['end'][1]) / 2.0)
    return G.arc_midpoint(tuple(ent['center']), tuple(ent['start']),
                          tuple(ent['end']), bool(ent['ccw']))


def entity_vertices(ents):
    """实体链的顶点序列（结合点取两端点中点）+ 首末点。"""
    if not ents:
        return []
    vs = [tuple(ents[0]['start'])]
    for i in range(1, len(ents)):
        a, b = ents[i - 1]['end'], ents[i]['start']
        vs.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))
    vs.append(tuple(ents[-1]['end']))
    return vs


def profile_points(ents, step=0.05):
    """实体链 → 等弧长重采样的轮廓点列（用于豪斯多夫）。"""
    pts = G.resample_polyline(G.entities_polyline(ents, max_step=step), step)
    return _as_array(pts)


# ------------------------------------------------------------------ Z 对齐

def _shift(pts, dz):
    P = _as_array(pts).copy()
    P[:, 0] += dz
    return P


def estimate_gauge_shift(P_true, P_ext, init=0.0, iters=5, max_step=2.0):
    """只沿 Z 平移的 1-D ICP，返回使两条轮廓最贴合的 ΔZ。

    ``P_true`` 为真值点列，``P_ext`` 为提取点列，返回 ΔZ 表示
    "提取点列沿 +Z 平移 ΔZ 后与真值对齐"。
    """
    dz = float(init)
    if len(P_true) == 0 or len(P_ext) == 0:
        return dz
    zt = _as_array(P_true)[:, 0]
    ze = _as_array(P_ext)[:, 0]
    for _ in range(iters):
        S = _shift(P_ext, dz)
        idx = _nn_index(S, P_true)
        # 当前对应关系下"从原始位置算起"的最佳总位移
        best = float(np.median(zt[idx] - ze))
        step = max(-max_step, min(max_step, best - dz))   # ★ 增量而非绝对值
        dz += step
        if abs(step) < 1e-5:
            break
    return dz


def convention_shift(P_true, P_ext):
    """按"右端面对齐"约定求 ΔZ（车削件 Z 原点取在右端面）。"""
    if len(P_true) == 0 or len(P_ext) == 0:
        return 0.0
    return float(_as_array(P_true)[:, 0].max() - _as_array(P_ext)[:, 0].max())


# ------------------------------------------------------------------ 实体匹配

def match_entities(truth, ext, dz=0.0, max_pairs=None, max_dist=None):
    """按中点距离做一对一的贪心最近匹配。

    返回 ``(pairs, unmatched_truth, unmatched_ext)``；
    ``pairs`` 元素为 ``(i_truth, j_ext, 中点距离)``。

    ★ ``max_dist`` 是**必需的安全闸门**（默认 1.0 mm）
    -------------------------------------------------
    无上限的贪心匹配存在自洽闭环漏洞：只要把 A 配到 B′、B 配到 A′，而 A′/B′ 恰好带着
    对方的圆心，那么 ``center_err``/``radius_err``/``type_acc``/``arc_recall`` 会**同时满分**，
    而几何其实错了 8.57 mm（评审给出的反例：两条同半径 R5 圆角的圆心互换，
    配对距离 8.33 / 14.45 mm，前四项指标全绿，只有豪斯多夫距离抓到）。
    加上配对距离上限后，这种错配会直接变成"未匹配"，进而使圆弧类指标判失败。
    """
    if max_dist is None:
        max_dist = DEFAULT_MAX_PAIR_DIST
    if not truth or not ext:
        return [], list(range(len(truth))), list(range(len(ext)))
    tm = _as_array([entity_midpoint(e) for e in truth])
    em = _shift([entity_midpoint(e) for e in ext], dz)
    cand = []
    for i in range(len(tm)):
        d = np.hypot(em[:, 0] - tm[i, 0], em[:, 1] - tm[i, 1])
        for j in range(len(em)):
            if float(d[j]) <= max_dist:
                cand.append((float(d[j]), i, j))
    cand.sort()
    used_t, used_e, pairs = set(), set(), []
    for d, i, j in cand:
        if i in used_t or j in used_e:
            continue
        used_t.add(i)
        used_e.add(j)
        pairs.append((i, j, d))
        if max_pairs and len(pairs) >= max_pairs:
            break
    pairs.sort()
    return (pairs,
            [i for i in range(len(truth)) if i not in used_t],
            [j for j in range(len(ext)) if j not in used_e])


# ------------------------------------------------------------------ 主入口

def compare(truth_ents, ext_ents, *, step=0.05, align='icp',
            strict_radius_target=STRICT_RADIUS_TARGET):
    """比对真值实体链与提取实体链，返回 6 项指标 + 诊断量。

    ``align``:
      ``'icp'``        约定对齐 + Z 向平移 ICP 精对齐（默认）
      ``'convention'`` 只按右端面约定对齐
      ``'none'``       完全不对齐（用于诊断）
    """
    res = {
        'ok': False, 'failure': None,
        'n_truth': len(truth_ents or []), 'n_ext': len(ext_ents or []),
        'n_truth_arcs': sum(1 for e in (truth_ents or []) if e['type'] == 'arc'),
        'n_ext_arcs': sum(1 for e in (ext_ents or []) if e['type'] == 'arc'),
        'gauge_shift_mm': 0.0, 'gauge_shift_icp_mm': 0.0,
        'align': align, 'step_mm': step,
    }
    for k in METRIC_ORDER:
        res[k] = float('nan')
    res['endpoint_rms'] = float('nan')
    res['center_err_rms'] = float('nan')
    res['radius_err_rms'] = float('nan')
    res['endpoint_err_convention'] = float('nan')
    res['n_matched'] = 0
    res['n_unmatched_truth'] = res['n_truth']
    res['n_spurious_ext'] = res['n_ext']

    if not truth_ents:
        res['failure'] = '真值实体为空（测试框架自身异常）'
        return res
    if not ext_ents:
        res['failure'] = '提取实体数为 0'
        return res

    P_true = profile_points(truth_ents, step)
    P_ext = profile_points(ext_ents, step)

    dz_conv = convention_shift(P_true, P_ext)
    if align == 'none':
        dz, dz_icp = 0.0, 0.0
    elif align == 'convention':
        dz, dz_icp = dz_conv, 0.0
    else:
        dz = estimate_gauge_shift(P_true, P_ext, init=dz_conv)
        dz_icp = dz - dz_conv
    res['gauge_shift_mm'] = dz
    res['gauge_shift_icp_mm'] = dz_icp

    # ---- 顶点/端点误差（双向）
    vt = _as_array(entity_vertices(truth_ents))
    ve = _shift(entity_vertices(ext_ents), dz)
    fwd = _nn_dist(vt, ve)
    bwd = _nn_dist(ve, vt)
    res['endpoint_err'] = float(max(fwd.max(), bwd.max()))
    res['endpoint_rms'] = float(np.sqrt(np.concatenate([fwd ** 2, bwd ** 2]).mean()))
    ve0 = _shift(entity_vertices(ext_ents), dz_conv)
    res['endpoint_err_convention'] = float(max(
        float(_nn_dist(vt, ve0).max()), float(_nn_dist(ve0, vt).max())))

    # ---- 实体匹配
    pairs, un_t, un_e = match_entities(truth_ents, ext_ents, dz)
    res['n_matched'] = len(pairs)
    res['n_unmatched_truth'] = len(un_t)
    res['n_spurious_ext'] = len(un_e)
    res['matched_pairs'] = [(i, j, round(d, 4)) for i, j, d in pairs]

    type_ok = 0
    arc_hit = 0
    c_errs, r_errs = [], []
    pair_detail = []
    for i, j, d in pairs:
        t, e = truth_ents[i], ext_ents[j]
        same = (t['type'] == e['type'])
        type_ok += int(same)
        if t['type'] == 'arc':
            if e['type'] == 'arc':
                arc_hit += 1
                ce = G.dist(t['center'], (e['center'][0] + dz, e['center'][1]))
                re = abs(t['radius'] - e['radius'])
                c_errs.append(ce)
                r_errs.append(re)
                pair_detail.append({'truth': i, 'ext': j, 'kind': 'arc',
                                    'center_err': round(ce, 4), 'radius_err': round(re, 4),
                                    'dist_mm': round(d, 4)})
            else:
                pair_detail.append({'truth': i, 'ext': j, 'kind': 'arc→line MISS',
                                    'dist_mm': round(d, 4)})
        else:
            pair_detail.append({'truth': i, 'ext': j,
                                'kind': 'line' if same else 'line→arc MISS',
                                'dist_mm': round(d, 4)})

    res['pair_detail'] = pair_detail
    res['type_acc'] = type_ok / float(res['n_truth'])
    res['arc_recall'] = (arc_hit / float(res['n_truth_arcs'])) if res['n_truth_arcs'] else 1.0
    # ★ 有真值圆弧、却一段都没能配对成"圆弧↔圆弧"时，圆心/半径误差**判为 nan（失败）**，
    #   而不是返回 0.0。返回 0.0 等于"空集拿满分"：把两条圆弧降级成弦线后，
    #   评审实测 center_err/radius_err 仍双双 PASS，只有 type_acc/arc_recall/hausdorff
    #   抓到 —— 而这恰恰让"圆心误差 95.8% 达标""半径误差 98.6% 达标"这类招牌数字
    #   失去了意义。nan 与阈值比较恒为 False，verdict 会正确判 FAIL。
    if res['n_truth_arcs'] and not c_errs:
        res['center_err'] = float('nan')
        res['radius_err'] = float('nan')
        res['center_err_rms'] = float('nan')
        res['radius_err_rms'] = float('nan')
        res['arc_metrics_na_reason'] = '真值含圆弧但无任何圆弧被正确配对'
    else:
        res['center_err'] = float(max(c_errs)) if c_errs else 0.0
        res['radius_err'] = float(max(r_errs)) if r_errs else 0.0
        res['center_err_rms'] = (float(math.sqrt(sum(v * v for v in c_errs) / len(c_errs)))
                                 if c_errs else 0.0)
        res['radius_err_rms'] = (float(math.sqrt(sum(v * v for v in r_errs) / len(r_errs)))
                                 if r_errs else 0.0)
    res['strict_radius_ok'] = bool(res['radius_err'] == res['radius_err']
                                   and res['radius_err'] <= strict_radius_target)

    # ---- 豪斯多夫
    S = _shift(P_ext, dz)
    d_te = _nn_dist(P_true, S)
    d_et = _nn_dist(S, P_true)
    res['hausdorff'] = float(max(d_te.max(), d_et.max()))
    res['hausdorff_truth_to_ext'] = float(d_te.max())
    res['hausdorff_ext_to_truth'] = float(d_et.max())

    res['ok'] = True
    return res


def verdict(res, targets=None):
    """按目标值逐项判定，返回 ``(all_pass, rows)``；``rows`` 为
    ``[(指标, 实测, 目标串, 是否通过), ...]``。"""
    targets = targets or TARGETS
    rows, all_pass = [], True
    if not res.get('ok'):
        for k in METRIC_ORDER:
            rows.append((k, float('nan'), _target_str(k, targets), False))
        return False, rows
    for k in METRIC_ORDER:
        op, thr, unit = targets[k]
        v = res[k]
        ok = (v <= thr) if op == '<=' else (v >= thr)
        all_pass = all_pass and ok
        rows.append((k, v, _target_str(k, targets), ok))
    return all_pass, rows


def _target_str(k, targets=None):
    targets = targets or TARGETS
    op, thr, unit = targets[k]
    if k in ('type_acc', 'arc_recall'):
        return f'{op} {thr:.0%}'
    return f'{op} {thr:g} {unit}'


def summary_line(res):
    """一行摘要，用于矩阵表格。"""
    if not res.get('ok'):
        return f"FAIL ({res.get('failure')})"
    return (f"ep={res['endpoint_err']:.3f} c={res['center_err']:.3f} "
            f"r={res['radius_err']:.3f} t={res['type_acc']:.2f} "
            f"rec={res['arc_recall']:.2f} h={res['hausdorff']:.3f}")


def aggregate(results):
    """把一批比对结果聚合成"最差值"统计（回归门槛用最差值，不用均值）。"""
    good = [r for r in results if r.get('ok')]
    agg = {
        'n_total': len(results),
        'n_ok': len(good),
        'n_failed': len(results) - len(good),
        'failures': [r.get('failure') for r in results if not r.get('ok')],
    }
    for k in METRIC_ORDER:
        if k in ('type_acc', 'arc_recall'):
            agg[k + '_worst'] = min((r[k] for r in good), default=float('nan'))
        else:
            agg[k + '_worst'] = max((r[k] for r in good), default=float('nan'))
    return agg
