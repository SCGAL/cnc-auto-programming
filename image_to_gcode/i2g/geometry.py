"""共享几何核：直线/圆弧的采样、拟合、求交与极值。

本模块被两侧共用，是本项目的"度量衡"：
  1. ``tests/synth.py``      —— 把真值几何光栅化成测试图（真值侧）
  2. ``i2g/vectorize.py``、``snap.py``、``emit.py`` —— 流水线侧
共用同一份采样与求交代码，保证"真值"与"提取结果"之间的比较不存在实现口径差异
（否则测量误差会混进被测误差里）。

坐标系约定（全项目统一，与 ``dxf_to_gcode_v2.py`` 一致）:
    点 = (Z, X)，单位 mm，X 为**半径值**，X >= 0 为零件上半部。
    圆弧用 ``ccw`` 布尔量表示 ZX 平面内的旋向（True = 逆时针）。
    平面内 Z 为横轴（向右为正）、X 为纵轴（向上为正）。

本模块只含**无状态纯函数**，不含任何图像算法（NFR-4 可测性）。
"""

from __future__ import annotations

import math

import numpy as np

TAU = 2.0 * math.pi


# =============================================================================
# 基本量
# =============================================================================

def dist(p, q):
    """两点距离。"""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def angle_of(center, p):
    """点 p 相对圆心 center 的极角（弧度）。"""
    return math.atan2(p[1] - center[1], p[0] - center[0])


def arc_sweep(center, start, end, ccw):
    """由圆心 + 两端点 + 旋向反推有向包角，取值 (0, 2π]。"""
    a1, a2 = angle_of(center, start), angle_of(center, end)
    s = (a2 - a1) % TAU if ccw else (a1 - a2) % TAU
    return s if s > 1e-12 else TAU


def arc_midpoint(center, start, end, ccw):
    """**实际被切削的**圆弧中点（用于校验旋向，抓 270° 大圆弧）。"""
    a1 = angle_of(center, start)
    sw = arc_sweep(center, start, end, ccw)
    if not ccw:
        sw = -sw
    am = a1 + sw / 2.0
    r = dist(center, start)
    return (center[0] + r * math.cos(am), center[1] + r * math.sin(am))


# =============================================================================
# 采样
# =============================================================================

def sample_arc(center, radius, start, end, ccw, max_step=0.2):
    """把圆弧离散成折线点列（含首末点）。max_step 为最大弦长。"""
    a1 = angle_of(center, start)
    sweep = arc_sweep(center, start, end, ccw)
    if not ccw:
        sweep = -sweep
    arc_len = abs(sweep) * radius
    n = max(2, int(math.ceil(arc_len / max(1e-9, max_step))))
    return [(center[0] + radius * math.cos(a1 + sweep * i / n),
             center[1] + radius * math.sin(a1 + sweep * i / n))
            for i in range(n + 1)]


def densify_segment(p, q, max_step):
    """把直线段按 max_step 加密成点列（含首末点）。

    ⚠️ 必须加密：光栅化用"沿折线打圆盘"的笔刷模型，若只给端点，
    直线段上就只会落下几个孤立圆斑，整条外圆画不出来。
    """
    L = dist(p, q)
    if not max_step or max_step <= 0 or L <= max_step:
        return [tuple(p), tuple(q)]
    n = int(math.ceil(L / max_step))
    return [(p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n)
            for i in range(n + 1)]


def entity_polyline(ent, max_step=0.2):
    """单个实体 → 折线点列（含首末点，相邻点间距 <= max_step）。"""
    if ent['type'] == 'line':
        return densify_segment(tuple(ent['start']), tuple(ent['end']), max_step)
    return sample_arc(tuple(ent['center']), float(ent['radius']),
                      tuple(ent['start']), tuple(ent['end']), bool(ent['ccw']),
                      max_step=max_step)


def entities_polyline(ents, max_step=0.2):
    """实体链 → 单条折线点列（相邻实体末首点重合时自动去重）。"""
    pts = []
    for e in ents:
        seg = entity_polyline(e, max_step=max_step)
        if pts and dist(pts[-1], seg[0]) < 1e-9:
            pts.extend(seg[1:])
        else:
            pts.extend(seg)
    return pts


def polyline_length(pts):
    return sum(dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def resample_polyline(pts, step):
    """按弧长等距重采样（含首末点）。用于豪斯多夫距离的公平比较。"""
    if len(pts) < 2:
        return list(pts)
    total = polyline_length(pts)
    if total <= 0:
        return [pts[0], pts[-1]]
    n = max(2, int(math.ceil(total / max(1e-9, step))) + 1)
    out = [pts[0]]
    target = total / (n - 1)
    acc, i = 0.0, 0
    for k in range(1, n - 1):
        want = target * k
        while i < len(pts) - 2 and acc + dist(pts[i], pts[i + 1]) < want:
            acc += dist(pts[i], pts[i + 1])
            i += 1
        seg = dist(pts[i], pts[i + 1])
        f = 0.0 if seg < 1e-12 else (want - acc) / seg
        out.append((pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f,
                    pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f))
    out.append(pts[-1])
    return out


def points_to_array(pts):
    return np.asarray(pts, dtype=float).reshape(-1, 2)


# =============================================================================
# 直线拟合（总体最小二乘 / PCA）
# =============================================================================

def fit_line(pts):
    """总体最小二乘直线拟合。

    返回 dict: ``point`` 质心, ``dirn`` 单位方向, ``normal`` 单位法向,
    ``rms`` 垂直残差均方根, ``resid`` 带符号垂直残差数组。
    用 PCA 而非 ``y = kx + b``：竖直线段（端面）不会爆掉。
    """
    P = points_to_array(pts)
    c = P.mean(axis=0)
    if len(P) < 2:
        return {'point': tuple(c), 'dirn': (1.0, 0.0), 'normal': (0.0, 1.0),
                'rms': 0.0, 'resid': np.zeros(len(P))}
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    resid = (P - c) @ n
    rms = float(np.sqrt(float((resid ** 2).mean())))
    return {'point': tuple(c), 'dirn': (float(d[0]), float(d[1])),
            'normal': (float(n[0]), float(n[1])), 'rms': rms, 'resid': resid}


def line_signed_distances(point, normal, pts):
    P = points_to_array(pts)
    return (P - np.asarray(point, float)) @ np.asarray(normal, float)


def line_angle_deg(dirn):
    """直线方向角，归一化到 [0, 180)。"""
    a = math.degrees(math.atan2(dirn[1], dirn[0])) % 180.0
    return a


# =============================================================================
# 圆拟合
# =============================================================================

def fit_circle_kasa(pts):
    """Kåsa 代数圆拟合（线性最小二乘，闭式解）—— 用作几何精修的初值。

    拟合 x² + y² + Dx + Ey + F = 0，圆心 (-D/2, -E/2)，r = √(cx²+cy²−F)。
    返回 ``(center, radius)``；退化时返回 ``None``。
    """
    P = points_to_array(pts)
    if len(P) < 3:
        return None
    x, y = P[:, 0], P[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    D, E, F = (float(v) for v in sol)
    cx, cy = -D / 2.0, -E / 2.0
    r2 = cx * cx + cy * cy - F
    if not np.isfinite(r2) or r2 <= 1e-12:
        return None
    return ((cx, cy), math.sqrt(r2))


def circle_residuals(center, radius, pts):
    """几何残差 = |p−c| − r（带符号：正 = 点在圆外）。"""
    P = points_to_array(pts)
    d = np.hypot(P[:, 0] - center[0], P[:, 1] - center[1])
    return d - radius


def fit_circle_geometric(pts, center, radius, iters=40, tol=1e-11):
    """以 (center, radius) 为初值做 Gauss-Newton 几何距离精修。

    最小化 Σ(|p_i − c| − r)²。带步长限幅，避免初值差时发散。
    """
    P = points_to_array(pts)
    c = np.asarray(center, dtype=float)
    r = float(radius)
    if len(P) < 3:
        return (tuple(c), r)
    eye3 = np.eye(3)
    for _ in range(iters):
        d = P - c
        dl = np.hypot(d[:, 0], d[:, 1])
        if float(dl.min()) < 1e-9:
            break
        res = dl - r
        J = np.column_stack([-d[:, 0] / dl, -d[:, 1] / dl, -np.ones(len(P))])
        JtJ = J.T @ J
        Jtr = J.T @ res
        try:
            dx = np.linalg.solve(JtJ + 1e-14 * eye3, -Jtr)
        except np.linalg.LinAlgError:
            break
        step = float(np.abs(dx).max())
        if step > 1e6:
            break
        c = c + dx[:2]
        r = r + float(dx[2])
        if step < tol:
            break
    return (tuple(c), float(r))


# =============================================================================
# 求交（用于把"拟合出的相邻实体"接成严格连续的端点）
# =============================================================================

def line_line_intersect(a_point, a_dirn, b_point, b_dirn):
    """两直线交点；平行返回 None。"""
    ax, ay = a_point
    adx, ady = a_dirn
    bx, by = b_point
    bdx, bdy = b_dirn
    den = adx * bdy - ady * bdx
    if abs(den) < 1e-12:
        return None
    t = ((bx - ax) * bdy - (by - ay) * bdx) / den
    return (ax + adx * t, ay + ady * t)


def line_circle_intersect(point, dirn, center, radius):
    """直线与圆交点列表（0/1/2 个）。"""
    px, py = point
    dx, dy = dirn
    n = math.hypot(dx, dy)
    if n < 1e-12:
        return []
    dx, dy = dx / n, dy / n
    fx, fy = px - center[0], py - center[1]
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - c
    if disc < -1e-12:
        return []
    disc = max(0.0, disc)
    s = math.sqrt(disc)
    ts = [-b - s] if s < 1e-12 else [-b - s, -b + s]
    return [(px + dx * t, py + dy * t) for t in ts]


def circle_circle_intersect(c1, r1, c2, r2):
    """两圆交点列表（0/1/2 个）。"""
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    d = math.hypot(dx, dy)
    if d < 1e-12 or d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9:
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h2 = r1 * r1 - a * a
    h = math.sqrt(max(0.0, h2))
    mx, my = c1[0] + a * dx / d, c1[1] + a * dy / d
    if h < 1e-12:
        return [(mx, my)]
    ox, oy = -dy / d * h, dx / d * h
    return [(mx + ox, my + oy), (mx - ox, my - oy)]


def _entity_primitive(ent):
    """取出实体的几何描述：line → (point, dirn)；arc → (center, radius)。"""
    if ent['type'] == 'line':
        d = ent.get('_dirn')
        if d is None:
            p, q = ent['start'], ent['end']
            dx, dy = q[0] - p[0], q[1] - p[1]
            n = math.hypot(dx, dy)
            d = (dx / n, dy / n) if n > 1e-12 else (1.0, 0.0)
        return ('line', ent.get('_point', ent['start']), d)
    return ('arc', tuple(ent['center']), float(ent['radius']))


def _tangent_point(line_point, line_dirn, center, radius, tol_frac=0.03):
    """直线与圆若**近似相切**，返回切点（落在圆上）；否则 None。

    ★ 为什么必须有这一支
    --------------------
    直线与圆的交点在**相切**处是病态的：设圆心到直线距离 d、半径 R，
    交点到切点的偏移 ≈ ``sqrt(2R·|R−d|)``。R=50 px 时，
    ``|R−d| = 0.04 px`` 就会让交点漂 2 px（0.2 mm）—— 实测本该在 (-20, 20)
    的圆角起点被算到 (-20.285, 20)，单点误差 0.285 mm，直接顶穿预算。

    相切时正确做法是取"圆心到直线的垂足方向上的圆周点"：它同时几乎落在
    直线上（偏差恰为 |R−d|）与圆上，条件数从 sqrt 恶化回到 O(1)。

    容差取 ``tol_frac·R``：误判成相切时引入的误差上界就是该容差
    （0.03·R，R=50 px 时为 1.5 px = 0.15 mm），仍在预算内。
    """
    c = np.asarray(center, float)
    p = np.asarray(line_point, float)
    d = np.asarray(line_dirn, float)
    nrm = math.hypot(d[0], d[1])
    if nrm < 1e-12 or radius <= 0:
        return None
    d = d / nrm
    n = np.array([-d[1], d[0]])
    signed = float((c - p) @ n)
    if abs(abs(signed) - radius) > tol_frac * radius:
        return None
    foot = c - n * signed
    v = foot - c
    nv = float(np.hypot(v[0], v[1]))
    if nv < 1e-12:
        return None
    return (float(c[0] + v[0] / nv * radius), float(c[1] + v[1] / nv * radius))


def junction_candidates(ent_a, ent_b):
    """相邻两实体**真实存在**的几何交点候选列表（不存在时返回空列表）。

    与 :func:`junction_point` 的区别：不做"退化取中点"的兜底。相切/相离的
    两个原语（例如被肩面隔开的两条圆弧，圆心距略大于 R1+R2）会返回空 ——
    这一信息被 :func:`~i2g.snap.absorb_short_segments` 用来判断"短段该不该删"。
    """
    ka, avp, avd = _entity_primitive(ent_a)
    kb, bvp, bvd = _entity_primitive(ent_b)
    if ka == 'line' and kb == 'line':
        p = line_line_intersect(avp, avd, bvp, bvd)
        return [p] if p is not None else []
    if ka == 'line' and kb == 'arc':
        t = _tangent_point(avp, avd, bvp, bvd)
        return [t] if t is not None else line_circle_intersect(avp, avd, bvp, bvd)
    if ka == 'arc' and kb == 'line':
        t = _tangent_point(bvp, bvd, avp, avd)
        return [t] if t is not None else line_circle_intersect(bvp, bvd, avp, avd)
    return circle_circle_intersect(avp, avd, bvp, bvd)


def junction_point(ent_a, ent_b, hint=None):
    """求相邻实体 A 末 → B 首 的实际结合点。

    拟合出的两条直线/圆弧一般不会精确交于同一点，直接取拟合端点会留下微缝隙
    （并被"轮廓不闭合"检查抓住）。此处取两者的**几何交点**，且在两实体各自的
    参数范围内选最靠近 ``hint`` 的那一个；直线-圆弧近似相切时改用切点（见
    :func:`_tangent_point`，否则交点条件数会让误差放大一个量级）。
    无交点时退化为两端点的中点。
    """
    pa, qa = tuple(ent_a['start']), tuple(ent_a['end'])
    pb, qb = tuple(ent_b['start']), tuple(ent_b['end'])
    if hint is None:
        hint = ((qa[0] + pb[0]) / 2.0, (qa[1] + pb[1]) / 2.0)

    cands = [c for c in junction_candidates(ent_a, ent_b) if c is not None]
    # 只在"两实体端点附近"的候选里挑，避免选中延长线上的另一侧交点
    gate = max(0.5, 4.0 * (dist(qa, hint) + dist(pb, hint)) + 0.5)
    near = [c for c in cands if dist(c, hint) <= gate]
    pool = near or cands
    if not pool:
        return ((qa[0] + pb[0]) / 2.0, (qa[1] + pb[1]) / 2.0)
    return min(pool, key=lambda c: dist(c, hint))


# =============================================================================
# 实体的坐标极值（标定用：稳健地取"轮廓最高点"，不受单像素毛刺影响）
# =============================================================================

def entity_x_extremes(ent, max_step=0.05):
    """实体在其几何范围内的 X（第二坐标）最小/最大值。

    圆弧用**解析解**：X 的极值只可能出现在两端点或 sin 取 ±1 处（即 90°/270°），
    把这些角度在弧的参数区间内的解一并纳入。采样求极值会有 ~1e-4 的系统偏差，
    而标定要拿这个值当"轮廓最大高度"，偏差会整体放大到所有尺寸上。
    """
    if ent['type'] == 'line':
        xs = (ent['start'][1], ent['end'][1])
        return min(xs), max(xs)

    center = tuple(ent['center'])
    r = float(ent['radius'])
    a1 = angle_of(center, ent['start'])
    sweep = arc_sweep(center, ent['start'], ent['end'], ent['ccw'])
    if not ent['ccw']:
        sweep = -sweep
    cand = [ent['start'][1], ent['end'][1]]
    if abs(sweep) > 1e-12:
        for target in (math.pi / 2.0, -math.pi / 2.0):
            for k in (-2, -1, 0, 1, 2):
                t = (target + TAU * k - a1) / sweep
                if -1e-12 <= t <= 1.0 + 1e-12:
                    cand.append(center[1] + r * math.sin(target))
                    break
    return min(cand), max(cand)


def entities_x_extremes(ents, max_step=0.05):
    lo = min(entity_x_extremes(e, max_step)[0] for e in ents)
    hi = max(entity_x_extremes(e, max_step)[1] for e in ents)
    return lo, hi


# =============================================================================
# 输出清洗
# =============================================================================

def strip_private(ent):
    """剥掉下划线开头的内部拟合参数，得到与 dxf_to_gcode_v2 契约一致的实体。"""
    return {k: v for k, v in ent.items() if not k.startswith('_')}


def flip_entity(ent):
    """反转实体方向（圆弧同时反转旋向）。"""
    f = dict(ent)
    f['start'], f['end'] = ent['end'], ent['start']
    if ent['type'] == 'arc':
        f['ccw'] = not ent['ccw']
    if '_point' in f:
        f['_point'] = f['start']
    if '_dirn' in f:
        f['_dirn'] = (-f['_dirn'][0], -f['_dirn'][1])
    return f


def chain_gaps(ents):
    """相邻实体间的连接间隙列表 [(i, gap_mm), ...]。"""
    return [(i, dist(ents[i - 1]['end'], ents[i]['start'])) for i in range(1, len(ents))]
