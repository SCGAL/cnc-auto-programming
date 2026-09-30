"""③ 轮廓提取：细化取**线条中线** → 图论最长路跟踪 → 有序点列。

为什么必须做"线条中线提取"
--------------------------
输入是**线框图**（笔画），不是实心轮廓。一条宽 w 的笔画覆盖的是
``[X_true − w/2, X_true + w/2]``。若直接描笔画的外边界，会引入 **+w/2 的系统偏差**：
在 10 px/mm、线宽 3 px 时是 0.15 mm —— 单这一项就吃掉 0.2 mm 预算的 75%。
立项文档把"轮廓中线提取"列为 P1 优化项（§5.2），但按精度预算它是**必需项**，
故本版提前到核心实现。

做法：Zhang-Suen 细化（自实现，纯 numpy，确定性）把笔画压成 1 px 骨架，
再用"两遍 BFS 求图论直径"取出最长路径 —— 骨架上的毛刺（噪声、角点伪枝）
自然被剔除，不需要额外的剪枝规则。

两种输入约定
------------
``mode='line'``（默认）线框笔画图 → 细化取中线；
``mode='fill'``          实心填充轮廓图 → Moore 邻域边界跟踪 → 取上半链。

坐标约定
--------
输出点为 ``(z_px, x_px)``，与最终 (Z, X) 同向：
``z_px = col``（向右为 +Z）、``x_px = axis_row − row``（向上为 +X，轴线上为 0）。
注意像素中心约定：像素 row 的中心在连续坐标 row+0.5，轴线与点都用同一约定，
故差值里 0.5 相互抵消 —— 这也是"轴线取骨架最低行、无需再加 0.5"的原因。
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .errors import EmptyContourError
from .preprocess import NEIGHBORS_8

_SQ2 = 1.4142135623730951


# =============================================================================
# Zhang-Suen 细化
# =============================================================================

def thin_zhang_suen(mask, max_iter=200):
    """Zhang-Suen 细化：把笔画压成 1 px 宽骨架。

    经典两子迭代并行细化，保留连通性与端点（B >= 2 的门限使端点不被删）。
    返回 bool 骨架。对 1 px 笔画近似幂等。
    """
    img = mask.astype(np.uint8)
    for _ in range(max_iter):
        removed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2 = P[0:-2, 1:-1]
            p3 = P[0:-2, 2:]
            p4 = P[1:-1, 2:]
            p5 = P[2:, 2:]
            p6 = P[2:, 1:-1]
            p7 = P[2:, 0:-2]
            p8 = P[1:-1, 0:-2]
            p9 = P[0:-2, 0:-2]
            B = (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9).astype(np.uint8)
            seq = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            A = np.zeros_like(B)
            for i in range(8):
                A += ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
            if step == 0:
                c1, c2 = p2 * p4 * p6, p4 * p6 * p8
            else:
                c1, c2 = p2 * p4 * p8, p2 * p6 * p8
            cond = ((img == 1) & (B >= 2) & (B <= 6) & (A == 1)
                    & (c1 == 0) & (c2 == 0))
            if cond.any():
                img = img.copy()
                img[cond] = 0
                removed = True
        if not removed:
            break
    return img.astype(bool)


# =============================================================================
# 骨架图与最长路
# =============================================================================

def _build_graph(skel):
    ys, xs = np.nonzero(skel)
    n = len(ys)
    pos = np.column_stack([ys, xs]).astype(np.int64)
    key = {}
    for i in range(n):
        key[(int(ys[i]), int(xs[i]))] = i
    adj = [[] for _ in range(n)]
    for i in range(n):
        y, x = int(ys[i]), int(xs[i])
        for dy, dx in NEIGHBORS_8:
            j = key.get((y + dy, x + dx))
            if j is not None:
                adj[j].append(i if i != j else j)
    for i in range(n):
        adj[i] = sorted(set(adj[i]) - {i})
    return pos, adj


def _bfs(start, adj):
    n = len(adj)
    dist = np.full(n, -1, np.int64)
    par = np.full(n, -1, np.int64)
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                par[v] = u
                q.append(v)
    far = int(np.argmax(dist))
    return far, dist, par


def longest_path(skel):
    """取骨架的图论最长路（两遍 BFS，对树精确）。

    返回 ``(path_yx, diag)``；``path_yx`` 是有序的 (row, col) 列表。
    """
    pos, adj = _build_graph(skel)
    n = len(pos)
    if n == 0:
        return [], {'n_skeleton_px': 0, 'n_endpoints': 0, 'n_branch': 0,
                    'n_components': 0, 'path_len_px': 0, 'path_fraction': 0.0}
    deg = np.array([len(a) for a in adj])
    u, _, _ = _bfs(0, adj)
    v, dist, par = _bfs(u, adj)
    path = []
    cur = v
    while cur >= 0:
        path.append(cur)
        cur = int(par[cur])
    path.reverse()

    # 连通域计数（用已知的 dist 覆盖集之外的节点继续 BFS）
    seen = dist >= 0
    n_comp = 1
    while not seen.all():
        s = int(np.argmin(seen))
        _, d2, _ = _bfs(s, adj)
        seen |= (d2 >= 0)
        n_comp += 1

    plen = 0.0
    for i in range(1, len(path)):
        a, b = pos[path[i - 1]], pos[path[i]]
        plen += _SQ2 if (a[0] != b[0] and a[1] != b[1]) else 1.0
    diag = {
        'n_skeleton_px': n,
        'n_endpoints': int((deg == 1).sum()),
        'n_branch': int((deg >= 3).sum()),
        'n_components': int(n_comp),
        'path_len_px': float(plen),
        'path_fraction': float(len(path) / n),
    }
    return [(int(pos[i][0]), int(pos[i][1])) for i in path], diag


# =============================================================================
# Moore 邻域边界跟踪（fill 模式）
# =============================================================================

# 顺时针，从东开始
_MOORE = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))


def trace_boundary_moore(mask):
    """Moore 邻域边界跟踪，返回有序闭合边界 ``[(row, col), ...]``。

    起点取栅格序第一个墨迹像素（其西邻必为背景），用 Jacob 停止准则。
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    h, w = mask.shape
    start = (int(ys[0]), int(xs[0]))
    b = (start[0], start[1] - 1)                     # 进入方向：自西向东
    contour = [start]
    cur, back = start, b
    for _ in range(4 * int(mask.sum()) + 16):
        # 从 back 位置出发顺时针找第一个墨迹像素
        bi = _MOORE.index((back[0] - cur[0], back[1] - cur[1])) if \
            (back[0] - cur[0], back[1] - cur[1]) in _MOORE else 4
        nxt = None
        for k in range(1, 9):
            d = _MOORE[(bi + k) % 8]
            yy, xx = cur[0] + d[0], cur[1] + d[1]
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx]:
                nxt = (yy, xx)
                prev = _MOORE[(bi + k - 1) % 8]
                back = (cur[0] + prev[0], cur[1] + prev[1])
                break
        if nxt is None:
            break
        if nxt == start:
            break
        contour.append(nxt)
        cur = nxt
    return contour


# =============================================================================
# 上半部轮廓筛选
# =============================================================================

def split_upper_chain(contour, axis_row, band=0):
    """从闭合边界里去掉贴着轴线的那条底边，返回剩余的上半链。

    车削件上半部轮廓 + 沿轴线的封底边构成闭合边界；底边是 y 接近 axis_row 的
    一段**连续**的子序列。取该连续段之外的部分即为轮廓。

    ⚠️ ``band`` 必须取 0（只认恰好落在轴线那一行上的点）。早期用 ``band=2``，
    结果把**竖端面靠近轴线的 3 px 也一起当成封底边删掉**：填充模式下轮廓两端
    各短 3 px、轴线整体偏高 0.30 mm（实测端点误差 0.33 mm）。底边本身是一条
    水平线，本来就在同一行上，不需要容差带。
    """
    n = len(contour)
    if n == 0:
        return []
    low = np.array([1 if p[0] >= axis_row - band else 0 for p in contour])
    if low.all():
        return list(contour)
    if not low.any():
        return list(contour)
    # 找最长的 low 连续段（环形）
    best_len, best_start, cur_len, cur_start = 0, 0, 0, 0
    ext = np.concatenate([low, low])
    for i in range(2 * n):
        if ext[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    if best_len >= n:
        return list(contour)
    keep = [contour[i % n] for i in range(best_start + best_len, best_start + n)]
    return keep


def _orient(pts):
    """把点列方向统一为：从 Z 较大的一端（右端面）开始。"""
    if len(pts) < 2:
        return pts
    if pts[0][0] < pts[-1][0]:
        return list(reversed(pts))
    return list(pts)


# =============================================================================
# 亚像素中线精化
# =============================================================================

def sample_bilinear(img, rows, cols):
    """双线性采样（越界按边缘像素钳制）。"""
    h, w = img.shape
    r = np.clip(np.asarray(rows, float), 0.0, h - 1.0)
    c = np.clip(np.asarray(cols, float), 0.0, w - 1.0)
    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    fr = r - r0
    fc = c - c0
    return (img[r0, c0] * (1 - fr) * (1 - fc) + img[r0, c1] * (1 - fr) * fc
            + img[r1, c0] * fr * (1 - fc) + img[r1, c1] * fr * fc)


def refine_subpixel(gray01, path, half_span=3.0, step=0.2):
    """沿骨架法线做**覆盖率加权质心**，把中线推到亚像素。

    ★ 为什么必须做（实测依据）
    --------------------------
    数字骨架（Zhang-Suen）在**曲笔画**上有系统性形变：沿一条 R=50 px 的圆弧笔画，
    骨架点相对真圆心的半径中位值偏内 0.26 px，且沿弧的取边不一致（极差 1.4 px）。
    这个形变是**相关的**而非随机的，会整体带偏圆拟合 —— 实测即使在**完全正确**的
    真值圆弧点集上做 Kåsa+几何精修，仍得到 R=52.42 px（真值 50，偏 +2.42 px），
    且加 0.4 px 噪声几乎不改变结果（→ 系统性偏差，不是噪声）。

    笔画是"以中线为轴、两侧对称"的覆盖带，故沿法线做覆盖率加权质心可以精确
    复原中线位置（该估计量对对称覆盖带无偏），与笔画宽度、是否抗锯齿都无关。
    这才是 0.2 mm 精度成立的前提。
    """
    P = np.asarray(path, dtype=float)
    n = len(P)
    if gray01 is None or n < 2:
        return P
    tang = np.zeros_like(P)
    tang[1:-1] = P[2:] - P[:-2]
    tang[0] = P[1] - P[0]
    tang[-1] = P[-1] - P[-2]
    mag = np.hypot(tang[:, 0], tang[:, 1])
    ok = mag > 1e-9
    tang[ok] /= mag[ok, None]
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    offs = np.arange(-half_span, half_span + 1e-9, step)
    out = P.copy()
    for i in range(n):
        rr = P[i, 0] + normal[i, 0] * offs
        cc = P[i, 1] + normal[i, 1] * offs
        wgt = np.clip(sample_bilinear(gray01, rr, cc), 0.0, None)
        s = float(wgt.sum())
        if s > 1e-6:
            t = float((wgt * offs).sum() / s)
            out[i, 0] = P[i, 0] + normal[i, 0] * t
            out[i, 1] = P[i, 1] + normal[i, 1] * t
    return out


def estimate_stroke_width(gray01, path_len):
    """由覆盖率总量与中线长度反解笔画宽度。

    笔画面积 ``S = L·w + π(w/2)²``（两端各一个半圆盖帽，合计一个整圆）。
    解 ``(π/4)w² + L·w − S = 0``。
    """
    S = float(np.clip(gray01, 0.0, 1.0).sum())
    L = float(path_len)
    if L <= 1e-9:
        return 1.0
    disc = L * L + math.pi * S
    return max(0.2, (-L + math.sqrt(disc)) / (math.pi / 2.0))


def _mass_below(row_mass, a):
    """行质量向量（行 i 覆盖连续区间 [i, i+1)）在连续行坐标 a **以下**的总质量。"""
    n = len(row_mass)
    i = int(math.floor(a))
    full = float(row_mass[i + 1:].sum()) if i + 1 <= n else 0.0
    if 0 <= i < n:
        full += (i + 1 - a) * float(row_mass[i])
    return full


def axis_row_from_cap_mass(gray01, w, search=6.0, step=0.005):
    """用**圆头盖帽的质量守恒**定轴线行（亚像素）。

    ★ 为什么不能沿用骨架端点
    ------------------------
    轮廓两端都落在轴线 X=0 上，端面是竖直笔画，其**法线是水平的** —— 沿法线的
    覆盖率加权质心只能修正 Z 方向，修不了 X 方向，骨架端点的行号仍带量化与内缩
    误差（实测两端一致地偏 0.5 px，直接造成 0.33% 的比例误差与 0.2 mm 级端点误差）。

    改用面积守恒：两端各有一个半径 ρ=w/2 的圆头盖帽，盖帽圆心正好在 X=0 上，
    故"轴线以下"的总覆盖面积恰为 ``2·(πρ²/2) = πρ²``。从底部向上累加行覆盖质量，
    达到 πρ² 处的连续行坐标就是轴线。该式只依赖面积，对是否抗锯齿、线宽奇偶都不敏感。
    """
    m = np.clip(gray01, 0.0, 1.0).sum(axis=1)
    n = len(m)
    if n == 0 or m.sum() <= 0:
        return None
    target = math.pi * (w / 2.0) ** 2
    lo = max(0.0, n - w - search - 2.0)
    hi = float(n)
    best, best_err = None, None
    k = int(round((hi - lo) / step)) + 1
    for j in range(k):
        a = lo + j * step
        e = abs(_mass_below(m, a) - target)
        if best_err is None or e < best_err:
            best, best_err = a, e
    if best is None:
        return None
    # 连续行坐标 a ↔ 索引坐标 a − 0.5（像素 i 覆盖 [i, i+1)，中心在 i+0.5）
    return best - 0.5


def bottom_edge_subpixel(gray01):
    """填充区域的**下边界亚像素位置**（连续行坐标）。

    填充模式下沿轴线的封底边就是 X=0 本身，故下边界即轴线。
    逐列取"最后一个有覆盖的行 r + 该行的覆盖率"：对一条位于连续行 B 的水平边界，
    行 r 的覆盖率恰为 ``clamp(B−r, 0, 1)``，故 ``floor(B) + (B − floor(B)) = B`` —— 精确无偏。
    再对所有列取中位数抗噪。
    """
    g = np.clip(np.asarray(gray01, dtype=float), 0.0, 1.0)
    nz = g > 1e-6
    any_col = nz.any(axis=0)
    cols = np.nonzero(any_col)[0]
    if len(cols) == 0:
        return None
    vals = []
    for c in cols:
        rr = np.nonzero(nz[:, c])[0]
        if len(rr) == 0:
            continue
        r_last = int(rr[-1])
        vals.append(r_last + float(g[r_last, c]))
    if not vals:
        return None
    return float(np.median(vals))


# =============================================================================
# 主入口
# =============================================================================

def extract(ink, *, mode='line', gray01=None, subpixel=True, axis_mode='cap_mass'):
    """提取上半部外轮廓的有序点列。

    返回 dict:
      ``points``  : (N, 2) float 数组，列为 (z_px, x_px)，已按加工方向排序
      ``axis_row``: 轴线所在行（索引坐标，亚像素）
      ``diag``    : 诊断信息（骨架规模、端点/分支数、亚像素精化量、轴线来源等）
    """
    if mode not in ('line', 'fill'):
        raise ValueError(f"未知的输入约定 mode={mode!r}（只支持 'line' / 'fill'）")

    if mode == 'line':
        skel = thin_zhang_suen(ink)
        path, diag = longest_path(skel)
        if len(path) < 2:
            raise EmptyContourError(
                '细化后未取出可用骨架路径：图像中没有可识别的连续轮廓'
                f'（骨架 {diag.get("n_skeleton_px", 0)} px）')
        raw = np.array([[p[0], p[1]] for p in path], dtype=np.float64)
        if subpixel and gray01 is not None:
            refined = refine_subpixel(gray01, raw)
            diag['subpixel'] = True
            d = np.hypot(refined[:, 0] - raw[:, 0], refined[:, 1] - raw[:, 1])
            diag['subpixel_shift_px'] = float(np.median(d))
            diag['subpixel_shift_max_px'] = float(d.max())
        else:
            refined = raw
            diag['subpixel'] = False

        ends = (float(refined[0, 0]), float(refined[-1, 0]))
        diag['axis_row_ends'] = list(ends)
        diag['axis_row_end_spread'] = abs(ends[0] - ends[1])
        axis_row = (ends[0] + ends[1]) / 2.0
        diag['axis_source'] = 'skeleton_ends'
        if axis_mode == 'cap_mass' and gray01 is not None:
            w = estimate_stroke_width(gray01, diag.get('path_len_px', 0.0))
            diag['stroke_width_px'] = round(w, 4)
            a2 = axis_row_from_cap_mass(gray01, w)
            if a2 is not None and abs(a2 - axis_row) <= max(2.0, w):
                diag['axis_row_skeleton_ends'] = axis_row
                diag['axis_source'] = 'cap_mass'
                axis_row = float(a2)
            elif a2 is not None:
                diag['axis_cap_mass_rejected'] = float(a2)
        rows = refined[:, 0]
        cols = refined[:, 1]
    elif mode == 'fill':
        ys, xs = np.nonzero(ink)
        if len(ys) == 0:
            raise EmptyContourError('填充模式：图中没有任何前景像素')
        bottom_row_int = int(ys.max())      # 离散底行，用于分离封底边
        axis_row = float(bottom_row_int)    # 亚像素轴线（下面尝试精化）
        axis_source = 'fill_bottom_integer'
        if gray01 is not None:
            b = bottom_edge_subpixel(gray01)
            if b is not None and abs((b - 0.5) - axis_row) <= 2.0:
                axis_row = b - 0.5          # 连续行坐标 → 索引坐标
                axis_source = 'fill_bottom_subpixel'
        contour = trace_boundary_moore(ink)
        if len(contour) < 4:
            raise EmptyContourError('填充模式：边界跟踪失败（边界点不足）')
        # ⚠️ 分离封底边必须用**离散底行**，不能用上一步的亚像素轴线：
        #    亚像素轴线可能落在最后一行之内（如 207.75 → +0.5 取整成 208），
        #    而轮廓点最大行号只有 207，判定会全部落空、封底边一条都不删。
        upper = split_upper_chain(contour, bottom_row_int)
        if len(upper) < 2:
            raise EmptyContourError('填充模式：未能分离出上半部轮廓链')
        rows = np.array([p[0] for p in upper], dtype=np.float64)
        cols = np.array([p[1] for p in upper], dtype=np.float64)
        diag = {'n_skeleton_px': int(ink.sum()), 'n_boundary_px': len(contour),
                'n_upper_px': len(upper), 'mode': 'fill',
                'axis_source': axis_source,
                'bottom_row_int': bottom_row_int}

    pts = np.column_stack([cols, axis_row - rows])
    diag['axis_row'] = float(axis_row)
    diag['h_px_raw'] = float(pts[:, 1].max())
    diag['n_profile_pts'] = int(len(pts))
    pts = np.asarray(_orient([tuple(p) for p in pts]), dtype=np.float64)
    return {'points': pts, 'axis_row': axis_row, 'diag': diag}
