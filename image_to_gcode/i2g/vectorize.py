"""④ 矢量化 —— 本项目最难的一段。

流程（立项文档 §3.2④）
----------------------
1. **DP 简化**（Ramer-Douglas-Peucker）取出候选分段点
2. 逐段做**直线最小二乘**与**圆拟合**，比较残差 SSE 判别类型
3. 相邻同类型段**合并并复拟合**（DP 会把一条圆弧切成若干弦，必须合回来）
4. 圆弧：**Kåsa 代数拟合**取初值 → （必要时）**RANSAC 去外点** → **几何距离精修**
5. 端点不由拟合端点直接取，而取**相邻实体的几何交点** —— 这是亚像素精度的关键

关于端点：为什么不用拟合端点
----------------------------
拟合端点只能到"该段点列的末端"，而离散点列的末端本身有 ±0.5 px 量化误差。
相邻两条拟合直线/圆弧的**交点**则由两条亚像素精度的曲线共同决定，
误差远小于单段端点。同时它天然保证链条严格连续（这正是"轮廓不闭合"检查要求的）。

关于 RANSAC 的可复现性（NFR-3）
-------------------------------
RANSAC 用固定种子、每次调用新建生成器，因此同输入必然同输出（有测试守着）。
但**不要以为"干净图不会触发 RANSAC"**：触发条件是**核心区**上 Kåsa 的几何残差
RMS > ``ransac_trigger_rms``，而干净图的骨架本身带阶梯形变，实测基准例 9 段中仍有
4 段触发（评审用 `used_ransac` 逐段核过）。诊断里报告的 ``circle_rms`` 是
**内点子集**上的值（≈0.003），与触发判据用的核心区 RMS 不是同一个量，容易误读。
可复现性靠的是固定种子 + 无全局状态，不是靠"不触发"。

关于预平滑
----------
立项文档建议拟合前做滑动平均（窗口 5-9）。**默认关闭**：骨架点的阶梯抖动
幅度 <= 0.5 px，最小二乘拟合会把它平均掉，不需要预平滑；而滑动平均会侵蚀
45° 倒角与圆角切点（把尖角抹圆），反而制造系统误差。``smooth_window`` 保留为
可配置项，测试矩阵里两种都跑，用实测数据说话。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from . import geometry as G
from .errors import GeometryError

TAU = 2.0 * math.pi


@dataclass
class VectorizeParams:
    """矢量化参数（单位：像素）。"""
    # ---- 分段方式
    #  'grow' 区域生长 + 逐点模型选择（默认，实测唯一能做到 arc_recall >= 90% 的方案）
    #  'dp'   Douglas-Peucker 分段后逐段判别（立项文档 §3.2④ 的方案，保留作对照）
    segment_method: str = 'grow'
    grow_rms_factor: float = 4.0         # 生长容差 = factor × p90 局部噪声
    grow_tol_min: float = 0.9
    grow_tol_max: float = 3.0
    grow_min_pts: int = 6
    grow_window: int = 15                # 只看"最新 W 点"的最大偏差，避免被长段稀释
    refine_boundaries: bool = True       # 生长后用"联合残差最小"精修段边界
    refine_search: int = 30              # 边界搜索半径（点）
    refine_iters: int = 2
    # ---- 曲率突变切分（解决"相切过渡"靠残差发现不了的问题）
    #    以下三个默认值由 tests/make_probe.py 的 24 组网格探针实测选定
    #    （split_iters × curv_split_tol × refine_search × refine_iters，固定 12 例）：
    #    当前组合 8/12，而旧默认（split_iters=1, curv_split_tol=0.10）只有 5/12。
    #    探针可复现：python -m tests.make_probe
    split_iters: int = 2
    curv_window: int = 12                # 局部转角估计的半窗（点）
    curv_smooth: int = 5                 # 转角序列的滑动平均窗
    curv_split_tol: float = 0.06         # 转角偏离区段中位值的阈值（弧度）
    split_keep_ratio: float = 0.85       # 切开后两段代价之和须低于整体的这个比例才收刀
    # ---- DP（segment_method='dp' 时使用）
    dp_eps: float = 1.5
    # ---- 判别
    arc_sse_ratio: float = 0.3           # SSE_circle < ratio * SSE_line 才判圆弧
    min_arc_radius: float = 3.0          # 半径下限，防噪声过拟合
    min_arc_chord_ratio: float = 1.01    # 弧长/弦长 > ratio，排除近似直线的微弧
    #   取值 1.01（对应约 26° 弧）而非 1.02：切分可能在弧内部切开一刀，得到的
    #   碎片只跨约 30°，其 弧长/弦长 仅 1.013 —— 用 1.02 会把**真弧碎片**踢出
    #   "圆弧"类、判成直线，于是它无法与同类弧合并回去，留下一个伪顶点。
    #   实测（px10_w1_s+0_r15）该伪顶点落在 21 mm 长的圆弧中部，端点误差直接顶到 9.4 mm。
    #   1.01 仍远高于"近直线伪弧"的水平（实测竖直端面被误判时包角仅 0.24°）
    min_sweep_deg: float = 12.0          # ★ 包角下限：防"超大半径近直线"被过拟合成圆弧
    max_sweep_deg: float = 200.0         # 单段包角上限，超过视为不合理
    min_seg_pts: int = 5                 # 少于这么多点强制判直线
    trim: int = 1                        # 拟合时两端各裁掉几个点（下限）
    trim_frac: float = 0.18              # ★ 拟合核心区：两端各裁掉区段长度的这个比例
    inlier_px: float = 1.0               # 内点阈值下限（px）
    inlier_noise_factor: float = 3.5     # 内点阈值 = factor × 噪声
    # 内点扩展复拟合：默认**关闭**。相切过渡处偏差是二次的，
    # 1 px 的内点阈值等于放回约 10 px 邻段（sqrt(2·R·1) = 10 px @R=50），
    # 会把刚刚靠核心区去掉的偏差又装回来（实测 R 从 5.00 劣化到 5.22）。
    extend_inliers: bool = False
    smooth_window: int = 0               # 0 = 不平滑（见模块 docstring）
    # ---- RANSAC
    ransac_thresh: float = 1.5
    ransac_iters: int = 200
    ransac_trigger_rms: float = 0.8
    ransac_seed: int = 12345
    # ---- 合并（DP 路线用；生长路线天然已是最大区段）
    line_merge_angle_deg: float = 1.5
    line_merge_offset: float = 1.0
    arc_merge_center: float = 3.0
    arc_merge_radius_rel: float = 0.08
    arc_merge_radius_abs: float = 1.5
    merge_verify_rms: float = 1.6
    # ---- 用每个区段自己的模型回收被吃进去的中间段（针对主失效模式）
    recover_mid_segments: bool = True
    recover_tol_px: float = 0.25
    recover_noise_factor: float = 1.0
    recover_passes: int = 2
    recover_min_run_pts: int = 6        # 坏段最短长度（噪声段挡门）
    recover_run_gain: float = 2.0       # 坏段均值残差 / 容差 的下限
    # ---- 模型交叉点精修（**默认关闭**，实测有害，见该函数 docstring）
    crossover_refine: bool = False
    crossover_win_frac: float = 0.30
    crossover_win_max: int = 40
    crossover_passes: int = 3
    # ---- 贪心合并化简（压掉过度切分）
    merge_rms_max: float = 1.10
    merge_noise_factor: float = 3.5

    def as_dict(self):
        return asdict(self)


# =============================================================================
# 预处理点列
# =============================================================================

def smooth_polyline(pts, window=3):
    """对称滑动平均（首末点保持不动）。"""
    P = np.asarray(pts, dtype=float)
    n = len(P)
    if window is None or window < 3 or n < window:
        return P
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = P.copy()
    for i in range(half, n - half):
        out[i] = P[i - half:i + half + 1].mean(axis=0)
    return out


def dedupe_consecutive(pts):
    """去掉完全重合的相邻点（细化/跟踪可能产生）。"""
    P = np.asarray(pts, dtype=float)
    if len(P) < 2:
        return P
    keep = np.ones(len(P), bool)
    keep[1:] = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])) > 1e-12
    return P[keep]


# =============================================================================
# Douglas-Peucker
# =============================================================================

def dp_simplify(pts, eps):
    """返回被保留的顶点**下标**（含首末）。迭代实现，避免递归深度问题。"""
    P = np.asarray(pts, dtype=float)
    n = len(P)
    if n < 3 or eps <= 0:
        return np.array([0, n - 1]) if n >= 2 else np.arange(n)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        seg = P[i + 1:j]
        d = b - a
        L = math.hypot(d[0], d[1])
        if L < 1e-12:
            dist = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            dist = np.abs((seg[:, 0] - a[0]) * d[1] - (seg[:, 1] - a[1]) * d[0]) / L
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return np.nonzero(keep)[0]


# =============================================================================
# 区域生长（默认分段方式）
# =============================================================================

def estimate_point_noise(P, window=17, percentile=90.0):
    """估计轮廓点列的局部噪声水平（px）。

    做法：滑动窗口内做直线拟合，取各窗口垂直残差 RMS 的 **p90**（不是中位数）。
    取 p90 而非中位数很重要：骨架在水平/垂直段上残差近乎 0，而在 45° 斜段上
    因栅格阶梯会到 0.3~0.4 px。用中位数会被大量轴对齐段拉低到 0.05，
    导致容差过紧、把 45° 倒角切碎；p90 能把这个"斜段惩罚"纳入进来。
    生长容差由它自适应导出，从而 noise=0 与 noise=2% 的图都不需要手工调参。
    """
    P = np.asarray(P, dtype=float)
    n = len(P)
    if n < window + 2:
        return 0.3
    rms = []
    step = max(1, window // 2)
    for i in range(0, n - window, step):
        f = G.fit_line(P[i:i + window])
        rms.append(f['rms'])
    if not rms:
        return 0.3
    return max(0.05, float(np.percentile(rms, percentile)))


def _fit_both(pts):
    """同时做直线与圆拟合，返回 ``(line_fit, circle_fit, rms_line, rms_circle)``。"""
    lf = G.fit_line(pts)
    sse_line = _sse(pts, 'line', (lf['point'], lf['dirn']))
    rms_line = math.sqrt(sse_line / max(1, len(pts)))
    kasa = G.fit_circle_kasa(pts)
    rms_circ = float('inf')
    if kasa is not None and 1e-6 < kasa[1] < 1e9:
        rms_circ = math.sqrt(_sse(pts, 'circle', kasa) / max(1, len(pts)))
    return lf, kasa, rms_line, rms_circ


def _tail_dev(pts, window):
    """当前区段"最新 window 个点"在其最佳模型下的最大偏差。

    ★ 为什么不用全段 RMS：RMS 会被长段稀释。实测中段长 214 点的直线段
    吸收掉 27 px 圆弧后，总 RMS 仍只有 0.32 px（< 0.55 容差），生长不收敛，
    段边界偏离真实过渡点 2.76 mm。只看尾部窗口就能立刻发现偏差。
    """
    lf, kasa, rl, rc = _fit_both(pts)
    if rc < rl and kasa is not None:
        resid = np.abs(G.circle_residuals(kasa[0], kasa[1], pts))
    else:
        n = (-lf['dirn'][1], lf['dirn'][0])
        resid = np.abs(G.line_signed_distances(lf['point'], n, pts))
    tail = resid[-min(window, len(pts)):]
    return float(tail.max()), float(resid.max())


def _model_cost(pts):
    """区段的模型代价 = 较优模型的 RMS²（按点数归一，避免长段受罚）。"""
    _, _, rl, rc = _fit_both(pts)
    r = min(rl, rc)
    return r * r


def refine_boundaries(P, regions, params: VectorizeParams):
    """精修段边界：让相邻两段的**联合模型代价**最小。

    生长阶段用宽松容差保证不早断，代价是边界会"多吃"邻段若干点
    （直线段吞掉一段圆弧）。此处对每个内部边界做局部搜索，
    最小化 ``cost(左) + cost(右)``，把边界推回真实过渡点。
    这是"宽松生长 + 精确回找"的两段式：生长负责不漏段，精修负责准。
    """
    if not params.refine_boundaries or len(regions) < 2:
        return regions
    m = max(3, int(params.grow_min_pts))
    bounds = [r[0] for r in regions] + [regions[-1][1]]
    for _ in range(max(1, int(params.refine_iters))):
        for k in range(1, len(bounds) - 1):
            i, j = bounds[k - 1], bounds[k + 1]
            lo = max(i + m, bounds[k] - params.refine_search)
            hi = min(j - m, bounds[k] + params.refine_search)
            if hi <= lo:
                continue
            best_b, best_c = bounds[k], float('inf')
            for b in range(lo, hi + 1):
                c = _model_cost(P[i:b + 1]) + _model_cost(P[b:j + 1])
                if c < best_c:
                    best_c, best_b = c, b
            bounds[k] = best_b
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def grow_regions(P, params: VectorizeParams, tol=None):
    """区域生长分段：向后逐点扩展，只要"直线或圆弧中至少一个"仍拟合得住就继续。

    为什么不能只用 DP
    ------------------
    立项文档 §3.2④ 的路线是"DP 简化 → 每段判别类型 → 相邻同类型段合并"。实测
    （10 px/mm、线宽 2 px、R=5 mm 的 90° 圆角）该路线 **arc_recall = 0**：
    DP 以 eps=1.5 px 的容差会把 90° 圆弧切成 4 段弦，单段弦的矢高只有
    ``r(1−cos(θ/2)) ≈ 0.96 px``，与骨架点的阶梯噪声（0.3~0.4 px）同量级，
    于是 ``SSE_circle`` 与 ``SSE_line`` 几乎相等（实测比值 0.97），判别必然失败。
    把 eps 调小只会切得更碎（每段点数不足，判别更糟）——DP 的容差**同时**承担
    "抗噪"和"识别圆弧"两个互相冲突的职责，单靠它无法两全。

    本实现的取舍：DP 仍保留（``segment_method='dp'``，见 ``dp_simplify``），
    默认改用"宽松生长 + 边界精修"：

    * 生长阶段只看尾部窗口的最大偏差（见 :func:`_tail_dev`），任一模型跟得上就继续；
    * 到达模型切换点或角点时两侧模型都跟不上，自然断开；
    * 再用 :func:`refine_boundaries` 把边界推回真实过渡点；
    * 类型最后由断点处的一次 SSE 比值判定。
    """
    P = np.asarray(P, dtype=float)
    n = len(P)
    noise = estimate_point_noise(P)
    if tol is None:
        tol = min(params.grow_tol_max, max(params.grow_tol_min,
                                           params.grow_rms_factor * noise))
    m = max(2, int(params.grow_min_pts))
    regions = []
    i = 0
    while i < n - 1:
        j = min(i + m - 1, n - 1)
        while j + 1 < n:
            sub = P[i:j + 2]
            tail_dev, all_dev = _tail_dev(sub, params.grow_window)
            if tail_dev > tol or all_dev > 3.0 * tol:
                break
            j += 1
        regions.append((i, j))
        i = j
    if not regions:
        regions = [(0, n - 1)]
    regions = refine_boundaries(P, regions, params)
    return regions, tol


# =============================================================================
# 单段拟合
# =============================================================================

def _sse(pts, kind, fit):
    if kind == 'line':
        point, dirn = fit
        n = (-dirn[1], dirn[0])
        r = G.line_signed_distances(point, n, pts)
    else:
        c, rr = fit
        r = G.circle_residuals(c, rr, pts)
    return float((r ** 2).sum())


def ransac_circle(pts, thresh=1.5, iters=200, seed=12345, min_inliers_ratio=0.5):
    """RANSAC 圆拟合（固定种子 → 可复现）。返回 ``(center, radius, inlier_mask)``。"""
    P = np.asarray(pts, dtype=float)
    n = len(P)
    if n < 5:
        return None
    rng = np.random.default_rng(seed)
    best_mask, best = None, None
    for _ in range(int(iters)):
        idx = rng.choice(n, size=3, replace=False)
        f = G.fit_circle_kasa(P[idx])
        if f is None:
            continue
        c, r = f
        if not (1e-6 < r < 1e9):
            continue
        d = np.abs(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) - r)
        m = d <= thresh
        if best_mask is None or int(m.sum()) > int(best_mask.sum()):
            best_mask, best = m, (c, r)
    if best_mask is None or best_mask.sum() < max(4, int(min_inliers_ratio * n)):
        return None
    f = G.fit_circle_kasa(P[best_mask])
    if f is None:
        return None
    c, r = G.fit_circle_geometric(P[best_mask], f[0], f[1])
    return c, r, best_mask


def _resid_of(pts, kind, fit):
    if kind == 'line':
        point, dirn = fit
        return np.abs(G.line_signed_distances(point, (-dirn[1], dirn[0]), pts))
    c, r = fit
    return np.abs(G.circle_residuals(c, r, pts))


def fit_segment(pts, params: VectorizeParams, noise=0.35):
    """对一段点列同时做直线与圆拟合，判别类型。

    返回 ``(kind, fit, info)``。``kind`` 为 ``'line'`` 或 ``'arc'``。

    ★ 两步稳健拟合（本项目精度能达标的关键之一）
    ---------------------------------------------
    区段边界总会略微"多吃"邻段（相切过渡处尤其明显：肩面与 R=50 px 圆弧相切时，
    多吃 12 px 肩面只让圆残差涨 0.14 px，边界精修找不到它）。但**端点是由相邻
    原语的几何交点决定的**，所以真正要紧的是"拟合出的原语准确"，而不是边界精确。

    于是：先在**核心区**（两端各裁掉 ``trim_frac``）上做拟合 —— 核心区几乎总是
    干净的单类型片段；再用该模型在全区间上挑内点（|残差| <= 阈值）并**在内点上
    复拟合**。吸收进来的邻段点会被自动排除，且不需要边界本身精确。
    """
    P = np.asarray(pts, dtype=float)
    n = len(P)
    info = {'n_pts': int(n)}

    trim_n = max(int(params.trim), int(round(params.trim_frac * n)))
    trim_n = min(trim_n, max(0, (n - 4) // 2))
    core = P[trim_n:n - trim_n] if (n - 2 * trim_n) >= 4 else P
    info['n_core_pts'] = int(len(core))

    lf = G.fit_line(core)
    sse_line = _sse(core, 'line', (lf['point'], lf['dirn']))
    info['sse_line'] = sse_line
    info['line_rms'] = math.sqrt(sse_line / max(1, len(core)))

    arc_len = G.polyline_length(P)
    chord = G.dist(P[0], P[-1])
    info['arc_len'] = arc_len
    info['chord'] = chord
    info['len_ratio'] = (arc_len / chord) if chord > 1e-9 else 1.0

    if len(core) < params.min_seg_pts:
        return 'line', (lf, None), {**info, 'reason': '点数不足'}

    kasa = G.fit_circle_kasa(core)
    if kasa is None:
        return 'line', (lf, None), {**info, 'reason': '圆拟合退化'}
    c, r = kasa
    rms0 = math.sqrt(_sse(core, 'circle', (c, r)) / max(1, len(core)))
    used_ransac = False
    fit_pts = core
    if rms0 > params.ransac_trigger_rms:
        got = ransac_circle(core, thresh=params.ransac_thresh,
                            iters=params.ransac_iters, seed=params.ransac_seed)
        if got is not None:
            c2, r2, mask = got
            if _sse(core[mask], 'circle', (c2, r2)) < _sse(core, 'circle', (c, r)):
                c, r = c2, r2
                fit_pts = core[mask]
                used_ransac = True
                info['n_inliers_ransac'] = int(mask.sum())
    c, r = G.fit_circle_geometric(fit_pts, c, r)
    sse_circ = _sse(fit_pts, 'circle', (c, r))
    info['sse_circle'] = sse_circ
    info['circle_rms'] = math.sqrt(sse_circ / max(1, len(fit_pts)))
    info['radius'] = r
    info['center'] = c
    info['used_ransac'] = used_ransac

    # ---- 分类判据（全部在核心区上比较，避免被邻段污染）
    if r < params.min_arc_radius:
        return 'line', (lf, (c, r)), {**info, 'reason': f'半径 {r:.2f} < 下限'}
    if info['len_ratio'] <= params.min_arc_chord_ratio:
        return 'line', (lf, (c, r)), {**info, 'reason': '弧长/弦长过小（近似直线）'}
    if sse_line <= 1e-12:
        return 'line', (lf, (c, r)), {**info, 'reason': '直线残差为零'}
    if sse_circ >= params.arc_sse_ratio * sse_line:
        return 'line', (lf, (c, r)), {
            **info, 'reason': f'SSE_circle/SSE_line = {sse_circ / sse_line:.3f} '
                              f'>= {params.arc_sse_ratio}'}

    ccw = _ccw_from_points(fit_pts, c)
    sweep = _sweep_deg(P[0], P[-1], c, r, ccw)
    info['sweep_deg'] = sweep
    # ★ 包角下限：圆比直线多一个自由度，对"近直线的长段"总能拟合出
    #   一个超大半径圆并让 SSE 略微更小。实测 205 点的竖直端面会被拟合成
    #   R=47097 px、包角 0.24° 的"圆弧"，SSE 比值 0.166 < 0.3 通过比值检验。
    #   包角下限把这个退化解直接挡掉。
    if sweep < params.min_sweep_deg:
        return 'line', (lf, (c, r)), {
            **info, 'reason': f'包角 {sweep:.2f}° < 下限 {params.min_sweep_deg}°'}
    if sweep > params.max_sweep_deg:
        return 'line', (lf, (c, r)), {**info, 'reason': f'包角 {sweep:.1f}° 过大'}

    # ---- 内点扩展复拟合（默认关闭，见参数注释）
    if not params.extend_inliers:
        return 'arc', (lf, (c, r)), info
    tol_in = max(params.inlier_px, params.inlier_noise_factor * float(noise or 0.35))
    mask = _resid_of(P, 'circle', (c, r)) <= tol_in
    info['n_ext_inliers'] = int(mask.sum())
    info['inlier_tol_px'] = round(tol_in, 3)
    if int(mask.sum()) >= max(params.min_seg_pts, int(0.4 * n)):
        c2, r2 = G.fit_circle_geometric(P[mask], c, r)
        if r2 >= params.min_arc_radius:
            ccw = _ccw_from_points(P[mask], c2)
            info['radius_before_ext'] = r
            info['center_before_ext'] = c
            c, r, fit_pts = c2, r2, P[mask]
            info['circle_rms'] = math.sqrt(_sse(fit_pts, 'circle', (c, r))
                                           / max(1, len(fit_pts)))
            info['radius'] = r
            info['center'] = c
            info['sweep_deg'] = _sweep_deg(P[0], P[-1], c, r, ccw)
    return 'arc', (lf, (c, r)), info


def fit_line_refined(pts, params: VectorizeParams, noise=0.35):
    """直线的"核心拟合 + 内点扩展复拟合"，与圆弧同口径。"""
    P = np.asarray(pts, dtype=float)
    n = len(P)
    trim_n = max(int(params.trim), int(round(params.trim_frac * n)))
    trim_n = min(trim_n, max(0, (n - 4) // 2))
    core = P[trim_n:n - trim_n] if (n - 2 * trim_n) >= 4 else P
    lf = G.fit_line(core)
    if not params.extend_inliers:
        return lf, int(len(core))
    tol_in = max(params.inlier_px, params.inlier_noise_factor * float(noise or 0.35))
    mask = _resid_of(P, 'line', (lf['point'], lf['dirn'])) <= tol_in
    if int(mask.sum()) >= max(params.min_seg_pts, int(0.4 * n)):
        lf = G.fit_line(P[mask])
        return lf, int(mask.sum())
    return lf, int(len(core))


def _ccw_from_points(pts, center):
    """由点列顺序与圆心的叉积符号判定旋向。"""
    P = np.asarray(pts, dtype=float)
    v = P - np.asarray(center, dtype=float)
    cross = v[:-1, 0] * v[1:, 1] - v[:-1, 1] * v[1:, 0]
    return bool(cross.sum() > 0)


def _sweep_deg(p0, p1, c, r, ccw):
    if r <= 1e-9:
        return 0.0
    a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
    a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
    s = (a1 - a0) % TAU if ccw else (a0 - a1) % TAU
    return math.degrees(s if s > 1e-12 else TAU)


# =============================================================================
# 合并
# =============================================================================

def _make_entity(kind, pts, fit, info):
    lf, cf = fit
    if kind == 'line':
        return {'type': 'line', 'start': tuple(pts[0]), 'end': tuple(pts[-1]),
                '_point': lf['point'], '_dirn': lf['dirn'], '_rms': lf['rms'],
                '_n_pts': len(pts)}
    c, r = cf
    P = np.asarray(pts, dtype=float)
    ccw = _ccw_from_points(P, c)
    return {'type': 'arc', 'start': tuple(P[0]), 'end': tuple(P[-1]),
            'center': tuple(c), 'radius': float(r), 'ccw': ccw,
            '_n_pts': len(pts), '_sweep_deg': _sweep_deg(P[0], P[-1], c, r, ccw),
            '_rms': info.get('circle_rms', 0.0),
            '_used_ransac': info.get('used_ransac', False)}


def _segments_from_breakpoints(P, idx):
    return [(int(idx[i]), int(idx[i + 1])) for i in range(len(idx) - 1)]


def _merge_lines(ents, slices, params):
    """相邻共线直线合并（角度接近 且 共享顶点偏离合并直线 <= 容差）。"""
    i = 0
    while i < len(ents) - 1:
        a, b = ents[i], ents[i + 1]
        if a['type'] != 'line' or b['type'] != 'line':
            i += 1
            continue
        da = np.asarray(a['_dirn'], float)
        db = np.asarray(b['_dirn'], float)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(da @ db))))))
        if ang > params.line_merge_angle_deg:
            i += 1
            continue
        shared = np.asarray(a['end'], float)
        off = abs(float((shared - np.asarray(a['_point'], float)) @
                        np.array([-a['_dirn'][1], a['_dirn'][0]])))
        if off > params.line_merge_offset:
            i += 1
            continue
        merged = np.vstack([slices[i], slices[i + 1][1:]])
        lf = G.fit_line(merged)
        sse = _sse(merged, 'line', (lf['point'], lf['dirn']))
        rms = math.sqrt(sse / max(1, len(merged)))
        if rms > params.merge_verify_rms:
            i += 1
            continue
        ents[i] = {**a, 'end': b['end'], '_point': lf['point'], '_dirn': lf['dirn'],
                   '_rms': lf['rms'], '_n_pts': len(merged)}
        slices[i] = merged
        del ents[i + 1]
        del slices[i + 1]
    return ents, slices


def _merge_arcs(ents, slices, params):
    """相邻同圆心同半径圆弧合并（DP 会把一条圆弧切成多条弦）。"""
    i = 0
    while i < len(ents) - 1:
        a, b = ents[i], ents[i + 1]
        if a['type'] != 'arc' or b['type'] != 'arc':
            i += 1
            continue
        dc = G.dist(a['center'], b['center'])
        dr = abs(a['radius'] - b['radius'])
        tol_c = max(params.arc_merge_center, params.arc_merge_radius_rel * a['radius'])
        tol_r = max(params.arc_merge_radius_abs, params.arc_merge_radius_rel * a['radius'])
        if dc > tol_c or dr > tol_r:
            i += 1
            continue
        merged = np.vstack([slices[i], slices[i + 1][1:]])
        f = G.fit_circle_kasa(merged)
        if f is None:
            i += 1
            continue
        c, r = G.fit_circle_geometric(merged, f[0], f[1])
        rms = math.sqrt(_sse(merged, 'circle', (c, r)) / max(1, len(merged)))
        if rms > params.merge_verify_rms:
            i += 1
            continue
        ccw = _ccw_from_points(merged, c)
        ents[i] = {'type': 'arc', 'start': tuple(merged[0]), 'end': tuple(merged[-1]),
                   'center': (float(c[0]), float(c[1])), 'radius': float(r), 'ccw': ccw,
                   '_n_pts': len(merged), '_rms': rms,
                   '_sweep_deg': _sweep_deg(merged[0], merged[-1], c, r, ccw)}
        slices[i] = merged
        del ents[i + 1]
        del slices[i + 1]
    return ents, slices


# =============================================================================
# 端点：取相邻实体的几何交点
# =============================================================================

def project_onto_line(p, point, dirn):
    d = np.asarray(dirn, float)
    v = np.asarray(p, float) - np.asarray(point, float)
    t = float(v @ d)
    return (float(point[0] + d[0] * t), float(point[1] + d[1] * t))


def project_onto_circle(p, center, radius):
    v = np.asarray(p, float) - np.asarray(center, float)
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-12:
        return (float(center[0] + radius), float(center[1]))
    return (float(center[0] + v[0] / n * radius), float(center[1] + v[1] / n * radius))


def project_onto_entity(ent, p):
    """把点投影到实体自身的几何原语上。"""
    if ent['type'] == 'line':
        d = ent.get('_dirn')
        pt = ent.get('_point')
        if d is None or pt is None:
            a, b = ent['start'], ent['end']
            dx, dy = b[0] - a[0], b[1] - a[1]
            n = math.hypot(dx, dy)
            d = (dx / n, dy / n) if n > 1e-12 else (1.0, 0.0)
            pt = a
        return project_onto_line(p, pt, d)
    return project_onto_circle(p, ent['center'], ent['radius'])


def apply_junctions(ents, balance_iters=8, join_gap_tol=2.0):
    """把相邻实体的结合点替换为几何交点，保证严格连续 + 亚像素端点。

    ★ **绝不静默补线**：``join_gap_tol`` 是"原始两端点本来就够近"的闸门。
    ----------------------------------------------------------------
    若两原语没有真实交点，且原始两端点相距超过 ``join_gap_tol``，说明这是**真断口**
    （笔画中断、噪点切断、小连通域被误删），此时**保留原样、留出间隙**，
    交给下游的 :func:`~i2g.snap.verify_continuity` 报"轮廓不闭合"。
    早期版本无条件做交替投影，结果把 8 mm 的断口也"接"上了 —— 那是静默降级，
    违反"轮廓不闭合必须明确报错"的硬约束（有专门测试守着）。

    ★ 退化情形：两原语真的不相交（但端点够近）
    ------------------------------------------
    肩面隔开的两条圆角，其拟合圆的圆心距可能略大于 R1+R2（实测 30.07 vs 29.99 px），
    **真的没有交点**。此时若退回"两端点取中点"，该点会整体脱离其中一条圆的圆周
    （实测残差 0.196 mm，会被 v2 生成器的"圆弧圆心-端点半径最大残差"自检抓出来）。
    改为**交替投影**：在两条原语之间来回投影若干次，收敛到"到两条原语距离之和最小"
    的折中点，误差被平摊而非全压在某一条上。

    有真实交点（含相切）时不做交替投影 —— 否则相切点会被投影来回拉扯，反而离开圆周。
    """
    out = [dict(e) for e in ents]
    if not out:
        return out
    if len(out) == 1:
        out[0]['start'] = project_onto_entity(out[0], out[0]['start'])
        out[0]['end'] = project_onto_entity(out[0], out[0]['end'])
        return out
    for i in range(1, len(out)):
        a, b = out[i - 1], out[i]
        hint = ((a['end'][0] + b['start'][0]) / 2.0,
                (a['end'][1] + b['start'][1]) / 2.0)
        raw_gap = G.dist(a['end'], b['start'])
        if raw_gap > join_gap_tol:
            continue                           # 真断口 —— 保留间隙，交给连续性检查
        cands = G.junction_candidates(a, b)
        p = None
        if cands:
            cand = G.junction_point(a, b, hint)
            if G.dist(cand, hint) <= max(6.0 * join_gap_tol, 3.0 * raw_gap + join_gap_tol):
                p = cand
        if p is None:
            # 没有可用交点（两原语不相交，或交点离端点太远）→ **交替投影**取折中点。
            #   为什么必须兜底而不是原样保留：原样保留会让该端点整体脱离自身圆周
            #   （实测半径残差 0.229 mm，会被 v2 生成器的"圆弧圆心-端点半径最大残差"
            #   自检抓出来）。交替投影把误差平摊到两侧，残差有界且链条仍然严格连续。
            #   「真断口」已在上面被 raw_gap 闸门挡住，所以这里兜底不会掩盖断裂。
            p = hint
            for _ in range(max(1, int(balance_iters))):
                p = project_onto_entity(a, p)
                p = project_onto_entity(b, p)
        a['end'] = p
        b['start'] = p
    # 首尾端点不属于任何结合点，投影到自身原语上
    out[0]['start'] = project_onto_entity(out[0], out[0]['start'])
    out[-1]['end'] = project_onto_entity(out[-1], out[-1]['end'])
    return out


# =============================================================================
# 主入口
# =============================================================================

def signed_turning_angle(P, k, smooth=0):
    """逐点有符号局部转角（弧度）：前 k 点方向 → 后 k 点方向的夹角。

    直线段为 ~0，半径 R 的圆弧段为 ~``2k/R``。这是**唯一能发现"相切过渡"**的
    局部信号：肩面与两侧圆弧相切时，残差对"多算了一小截肩面"几乎无感
    （12 px 肩面偏离 50 px 圆弧的距离只有 0.14 px），但转角会从 ``2k/R`` 骤降到 0。
    """
    P = np.asarray(P, dtype=float)
    n = len(P)
    th = np.zeros(n)
    if n <= 2 * k + 1:
        return th
    a = P[k:n - k] - P[0:n - 2 * k]
    b = P[2 * k:n] - P[k:n - k]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
    th[k:n - k] = np.arctan2(cross, dot)
    if smooth and smooth >= 3:
        w = int(smooth) | 1
        h = w // 2
        out = th.copy()
        for i in range(h, n - h):
            out[i] = th[i - h:i + h + 1].mean()
        th = out
    return th


def split_by_curvature(P, regions, params: VectorizeParams):
    """在"转角跳变最大处"切开区段（符号无关）。

    ★ 判据为什么用"跳变"而不是"偏离中位值"
    ----------------------------------------
    肩面两侧的圆角**曲率符号相反**（一凸一凹）。若某区段同时跨越两条弧，
    转角的分布是 ±|κ| 的双峰，中位值≈0，于是"偏离中位值最大"的点落在**弧内部**，
    切出来的还是弧 —— 实测 R2/R15 两例肩面都被整体吞掉、type_acc 掉到 8/9、
    端点误差 1.06~1.25 mm。改用 ``θ[i+w] − θ[i−w]`` 的**跳变量**：弧内部该值≈0，
    而在"弧↔直线"过渡处必然最大，与曲率正负无关。

    与 :func:`refine_boundaries` 互补：边界精修只能挪动已有边界，
    若相切过渡处根本没生成边界，就无从可挪。本函数负责**新增**边界。
    """
    if params.curv_split_tol <= 0 or len(P) < 3 * params.curv_window:
        return regions, False
    k = max(2, int(params.curv_window))
    # ★ 跳变跨度必须取满窗口（w = k），不能取 k/2。
    #   转角是用 2k 点窗口估计的、又叠了 ``curv_smooth`` 点平滑，相切过渡因此被抹开到
    #   约 ±k 点；若跳变只跨 ±k/2，就只采到过渡的一小部分。实测（px10_w1_s+0_r15，
    #   R=150 px 圆弧接 19 点肩面）：真实转角落差 0.16 rad，但 ±k/2 跨度只测到约 0.08，
    #   压在阈值 0.06 边缘 —— 结果切分器**只切了直线段（伪切分），从未切那颗含肩面的
    #   弧区段**，肩面永远丢不出来，端点误差 1.02 mm。跨度取满后落差被完整采到。
    w = max(2, k)
    th = signed_turning_angle(P, k, params.curv_smooth)
    m = max(3, int(params.grow_min_pts))
    npts = len(P)
    out, changed = [], False
    for (i, j) in regions:
        if j - i < 2 * k + 2 * m:
            out.append((i, j))
            continue
        # ★ 搜索范围不能退成 ``[i+k+w, j−k−w]``。
        #   真值过渡点可能就落在区段最末的 ``k+w`` 点之内（实测 px10_w1_s+0_r15：
        #   肩面是区段最后的 19 点，过渡在索引 749，而旧上限是 770−12−12=746
        #   —— 切分器**根本看不见它**，于是只切直线段、肩面永远丢不出来）。
        #   改为走到 θ 的有效边界 [i+k, j−k]，并让跳变窗在边界处**自适应收缩**，
        #   这样既覆盖到末端过渡，又不会读到窗口不足处人为置 0 的 θ。
        lo_valid = max(i, k)
        hi_valid = min(j, npts - 1 - k)
        lo = max(lo_valid + 2, i + m)
        hi = min(hi_valid - 2, j - m)
        if hi <= lo:
            out.append((i, j))
            continue
        best_val, split = 0.0, None
        for t in range(lo, hi + 1):
            wt = min(w, t - lo_valid, hi_valid - t)
            if wt < 2:
                continue
            dv = abs(float(th[t + wt] - th[t - wt]))
            if dv > best_val:
                best_val, split = dv, t
        if split is None or best_val <= params.curv_split_tol:
            # 判据二：转角**偏离区段中位值**（对单条弧里的过渡更灵敏）
            seg = th[lo_valid:hi_valid + 1]
            if len(seg) >= 3:
                med = float(np.median(seg))
                p2 = int(np.argmax(np.abs(seg - med)))
                if abs(float(seg[p2]) - med) > params.curv_split_tol:
                    split = lo_valid + p2
        if split is None or not (i + m <= split <= j - m):
            out.append((i, j))
            continue
        # ★ 切分收益判据（局部的、不受长段稀释的"模型复杂度是否值得"比较）
        #   只按"转角跳变够大"就切，会把**同一条圆弧内部**也切开：实测 px10_w2_s+0_r15
        #   在 R=150 px 的弧内切出一段跨 20° 的碎片，其 弧长/弦长 仅 1.005，会被
        #   误判成直线，既合并不回去、又多出一个落在 21 mm 长弧中部的伪顶点，
        #   端点误差直接顶到 3.23 mm。
        #   这里比较"用一个模型拟合整体"与"分两段各用一个模型"的代价：
        #   切完若总代价没有显著下降，说明这一刀没有换来模型复杂度上的收益 —— 收回。
        cost_union = _model_cost(P[i:j + 1])
        cost_split = (_model_cost(P[i:split + 1]) + _model_cost(P[split:j + 1]))
        if cost_split >= params.split_keep_ratio * cost_union:
            out.append((i, j))
            continue
        out.append((i, split))
        out.append((split, j))
        changed = True
    return out, changed


def simplify_regions(P, regions, params: VectorizeParams, noise):
    """贪心合并化简：能并回一个原语的就并回去，压掉过度切分。

    ★ 为什么必须有这一步（实测依据）
    --------------------------------
    曲率切分为了不漏掉"相切过渡"，判据偏灵敏，代价是会把直线段/斜段切碎。
    DoD 子集 72 例的失败**绝大多数是过度切分**：提取出 11~22 段而真值 9 段，
    豪斯多夫距离常常正常（0.09~0.45 px），但多出来的顶点把端点误差顶到 6~19 mm
    （顶点集合不匹配，属于度量口径上的硬失败）。

    合并判据是标准的"模型复杂度是否值得"：试着删掉每个内部边界，若**合并后**
    的最佳模型残差仍不超过 ``merge_rms_max``，就认为该边界是伪边界并删掉。

    ★ 必须加"同类型"前置条件
    ------------------------
    只看残差会把**真肩面并进圆弧**：肩面与圆弧相切，20 px 的相切肩面并进 R=50 px
    圆弧后残差只有约 1.1 px，恰好擦过阈值 —— 实测那样做会让全部用例退化到 8/9 段、
    端点误差 1.8 mm。加上"两侧区段各自的最佳模型类型必须相同"后，
    "直线↔圆弧"过渡天然不可合并，只合并被切碎的同类片段（线并线、弧并弧）。
    """
    if len(regions) < 2:
        return regions
    m = max(3, int(params.grow_min_pts))
    tol = max(params.merge_rms_max, params.merge_noise_factor * float(noise or 0.35))
    nz = float(noise or 0.35)
    kinds = []
    for (i, j) in regions:
        k, _f, _info = fit_segment(P[i:j + 1], params, nz)
        kinds.append(k)
    regs = list(regions)
    for _ in range(max(1, len(regions))):
        best = None
        for k in range(1, len(regs)):
            if kinds[k - 1] != kinds[k]:
                continue                     # ★ 禁止跨类型合并
            i, j = regs[k - 1][0], regs[k][1]
            if j - i + 1 < 2 * m:
                continue
            cost = _model_cost(P[i:j + 1])
            if cost <= tol * tol and (best is None or cost < best[0]):
                best = (cost, k, i, j)
        if best is None:
            break
        _c, k, i, j = best
        kinds[k - 1:k + 1] = [kinds[k - 1]]
        regs[k - 1:k + 1] = [(i, j)]
    return regs


def _fit_model_on(P, i, j, params, noise):
    """在 ``P[i:j+1]`` 上拟合"较优的那个模型"，返回 ``(kind, fit)``。

    ⚠️ 只按 RMS 取较优是**不够的**：直线段的 Kåsa 圆拟合 RMS 往往略小于直线拟合 RMS
    （圆多一个自由度），会把竖直端面判成"R=47097 px、包角 0.2° 的圆弧"。
    必须沿用与最终分类完全相同的护栏（半径下限 / 包角下限 / 弧长弦长比）。
    """
    sub = P[i:j + 1]
    kind, fit, _info = fit_segment(sub, params, noise)
    if kind == 'arc':
        return 'arc', tuple(fit[1])
    lf = fit[0]
    return 'line', (lf['point'], lf['dirn'])


def _model_resid_1pt(p, kind, fit):
    """单点到模型的绝对残差。"""
    if kind == 'line':
        pt, d = fit
        n = (-d[1], d[0])
        return abs((p[0] - pt[0]) * n[0] + (p[1] - pt[1]) * n[1])
    c, r = fit
    return abs(math.hypot(p[0] - c[0], p[1] - c[1]) - r)


def refine_boundaries_by_crossover(P, regions, params: VectorizeParams,
                                   passes: int = 3):
    """用**两侧原语残差的交叉点**把边界钉到相切/相交过渡点上。

    ★ 为什么这一招能解决主失效模式
    ------------------------------
    DoD 子集里 41 例失败中有 27 例属于"形状对但端点错位"：段数正确（9/9）、
    类型全对、豪斯多夫只有 0.18~0.24，但某个端点偏 1.1~1.2 mm —— 全是**边界放错**。

    在相切过渡处，基于残差的边界精修是**平的**（R=150 px 的圆弧多吃 12 px 肩面，
    残差只涨 0.48 px），基于曲率窗口的切分又受窗口宽度限制（±k 点 ≈ ±1.2 mm @10px/mm）。

    本函数改用两个原语**自己的几何**来定位：在边界左右各留 ``w`` 点的隔离带后，
    分别拟合左、右原语（此时的拟合是干净的），然后在 ``[b−w, b+w]`` 上取使
    ``|resid_left − resid_right|`` 最小的点。

    * 相切过渡：切点处两残差**同时为 0**，交叉点即切点，精确无窗口偏差；
    * 一般相交（角点）：交叉点即两原语的交点；
    * 两原语真的不相交（肩面隔开的两条圆角）：交叉点是让两侧误差相等的"平衡点"。

    拟合用的隔离带同时解决了"边界已经吃进邻段导致拟合被带偏"的问题 ——
    边界越准，隔离带越短，迭代收敛。

    ⚠️ **实测结论：默认关闭（``crossover_refine=False``）**
    ------------------------------------------------------
    对照实验（16 例固定集，落盘于 ``tests/report/probe_crossover.json``）：::

        关闭              9/16   最差端点 2.12 mm
        开启(0.30/40/3)   9/16   最差端点 3.04 mm   ← 更差
        开启(0.20/20/2)   8/16
        开启(0.15/15/2)   8/16
        开启(0.40/40/1)   8/16

    **开启是有害的**，故默认关闭；代码保留，供后续在有更严格判据时复用。

    失效原因：判据 ``|resid_left − resid_right|`` 最小的位置，在
    **短肩面夹于两条相切圆弧之间**时会落在**肩面正中** —— 因为在肩面内部，
    到左弧的距离由小变大、到右弧的距离由大变小，两者"相等"处正是肩面中点，
    而正确边界在肩面**两端**。该判据只在"单个区段内混合了两种原语"时方向正确，
    不适用于"两个已正确分开的区段之间"。
    """
    if not params.crossover_refine or len(regions) < 2:
        return list(regions)
    m = max(3, int(params.grow_min_pts))
    regs = [(int(i), int(j)) for (i, j) in regions]
    for _ in range(max(0, int(passes))):
        moved = False
        for k in range(1, len(regs)):
            i0, j0 = regs[k - 1]
            i1, j1 = regs[k]
            b = i1
            ln, rn = j0 - i0 + 1, j1 - i1 + 1
            w = int(round(params.crossover_win_frac * min(ln, rn)))
            w = max(m, min(w, params.crossover_win_max,
                           (ln - m) // 2, (rn - m) // 2))
            if w < m:
                continue
            li, lj = i0, b - w                 # 左侧隔离带之外的核心
            ri, rj = b + w, j1
            if lj - li + 1 < m or rj - ri + 1 < m:
                continue
            kl, fl = _fit_model_on(P, li, lj, params, noise)
            kr, fr = _fit_model_on(P, ri, rj, params, noise)
            lo, hi = b - w, b + w
            best, best_v = b, None
            for t in range(lo, hi + 1):
                dl = _model_resid_1pt(P[t], kl, fl)
                dr = _model_resid_1pt(P[t], kr, fr)
                v = abs(dl - dr)
                if best_v is None or v < best_v:
                    best_v, best = v, t
            if best == b:
                continue
            if not (i0 + m <= best <= j1 - m):
                continue
            regs[k - 1] = (i0, best)
            regs[k] = (best, j1)
            moved = True
        if not moved:
            break
    return regs


def segment_points(P, params: VectorizeParams):
    """默认分段：宽松生长 → （边界精修 ↔ 曲率切分）→ 贪心合并化简。

    返回 ``(regions, info)``。
    """
    regions, tol = grow_regions(P, params)
    n_grow = len(regions)
    n_split = 0
    for _ in range(max(0, int(params.split_iters))):
        before = len(regions)
        regions, changed = split_by_curvature(P, regions, params)
        n_split += len(regions) - before
        regions = refine_boundaries(P, regions, params)
        if not changed:
            break
    n_before_simplify = len(regions)
    regions = simplify_regions(P, regions, params, estimate_point_noise(P))
    n_after_simplify = len(regions)
    regions = recover_trimmed_subregions(P, regions, params,
                                         estimate_point_noise(P),
                                         passes=params.recover_passes)
    n_after_recover = len(regions)
    regions = refine_boundaries(P, regions, params)
    regions = refine_boundaries_by_crossover(P, regions, params,
                                             passes=params.crossover_passes)
    return regions, {'grow_tol_px': tol, 'n_regions_grown': n_grow,
                     'n_regions_split': n_split,
                     'n_regions_before_simplify': n_before_simplify,
                     'n_regions_after_simplify': n_after_simplify,
                     'n_regions_after_recover': n_after_recover,
                     'n_regions_final': len(regions)}


def recover_trimmed_subregions(P, regions, params: VectorizeParams, noise,
                               passes: int = 2):
    """用每个区段**自己的拟合模型**把"吃进来的中间段"退回去，切成新区段。

    ★ 这一招针对 DoD 里的主失效模式
    -------------------------------
    实测（``px10_w2_s+1_r2``）：段数 9/9、类型全对、圆弧拟合也很准
    （c/R 误差仅 0.05 mm），但端点误差 1.16 mm —— 因为 **2 mm 的肩面被两条圆弧
    区段各吃掉一半，根本没成为独立区段**，两条圆弧只好用"平衡点"直接相连，
    而那个点离真值两端各差 0.87~1.16 mm。

    关键观察：圆弧区段的**核心区拟合是准的**（核心区离边界足够远，不受污染）。
    于是可以用它自己的模型去筛全区间：残差超过 ``tol`` 的点就是"不属于这条弧"的点。
    把它们里面**足够长的连续段**退出来，就恢复了被吃掉的肩面。

    为什么"退出来"就够用：**结合点是由拟合出的原语算的，不是由区段端点算的**。
    只要肩面的**直线**被准确拟合（哪怕只有 10 个点），它与两侧圆弧的切点就能被
    ``geometry._tangent_point`` 精确算出，区段端点本身偏一点无所谓。

    容差 ``tol`` 与噪声挂钩：相切过渡处偏差是二次的，``tol`` 对应"弧能解释到多远"
    ``d ≈ sqrt(2·R·tol)``。R=20 px / tol=0.6 px 时 d≈4.9 px，两侧各退 4.9 px 后，
    2 mm(20 px) 的肩面仍能留下约 10 px ≥ ``min_seg_pts``，足够把那条直线拟合出来。
    """
    if not params.recover_mid_segments or len(regions) < 1:
        return list(regions)
    m = max(3, int(params.grow_min_pts))
    tol = max(params.recover_tol_px, params.recover_noise_factor * float(noise or 0.35))
    regs = [(int(i), int(j)) for (i, j) in regions]
    for _ in range(max(0, int(passes))):
        out, changed = [], False
        for (i, j) in regs:
            n = j - i + 1
            if n < 3 * m:
                out.append((i, j))
                continue
            kind, fit = _fit_model_on(P, i, j, params, noise)
            if kind != 'arc':
                # 只对圆弧做：直线的"切向延续"残差筛不出来（共线点残差本就为 0）
                out.append((i, j))
                continue
            bad = _model_resid_all(P[i:j + 1], kind, fit) > tol
            min_run = max(m, int(params.recover_min_run_pts))
            runs, k = [], m
            while k < n:
                if not bad[k]:
                    k += 1
                    continue
                s = k
                while k < n and bad[k]:
                    k += 1
                # ★ 只认"成规模的系统性偏离"：噪声只会产生零散短段。
                #   实测（px10_w2_s+0_n0.01_r5）只把容差从 0.66 收到 0.33 px，
                #   端点误差就从 0.125 mm 涨到 0.608 mm —— 因为随机噪声也能凑出
                #   若干超容差点并连续成 6 点以上的段。加上"长度 ≥ min_run 且
                #   段内均值残差 ≥ gain×tol"两道门，噪声段被挡掉、真肩面仍能通过
                #   （真肩面的残差从 0 单调涨到 ~2.5 px，均值远高于 2×容差）。
                if k - s < min_run:
                    continue
                seg_resid = _model_resid_all(P[i + s:i + k], kind, fit)
                if float(seg_resid.mean()) < params.recover_run_gain * tol:
                    continue
                runs.append((s, k - 1))
            if not runs:
                out.append((i, j))
                continue
            pos = 0
            for (s, e) in runs:
                if s > pos:
                    out.append((i + pos, i + s - 1))     # 弧能解释的部分
                out.append((i + s, i + e))               # 弧解释不了 → 独立成段
                pos = e + 1
            if pos < n:
                out.append((i + pos, j))
            changed = True
        regs = _normalize_regions(out, m)
        if not changed:
            break
    return regs


def _model_resid_all(P, kind, fit):
    if kind == 'line':
        pt, d = fit
        return np.abs(G.line_signed_distances(pt, (-d[1], d[0]), P))
    c, r = fit
    return np.abs(G.circle_residuals(c, r, P))


def _normalize_regions(regions, min_pts):
    """把区段列表整理成"首尾相接、无重叠、每段不少于 min_pts 点"的规范形式。

    ⚠️ 重叠判定必须用**严格小于**：本项目的区段是**共享边界点**的
    （A 结束于 b，B 起始于 b），若用 ``i <= prev_end`` 判定重叠，
    相邻区段会被全部合并成一段。
    """
    if not regions:
        return []
    regions = sorted((int(i), int(j)) for (i, j) in regions if j > i)
    merged = []
    for (i, j) in regions:
        if merged and i < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], j))
        else:
            merged.append((i, j))
    # 过短的段并入前一段
    out = []
    for (i, j) in merged:
        if out and (j - i + 1) < min_pts:
            out[-1] = (out[-1][0], j)
        else:
            out.append((i, j))
    return out


def vectorize(points, params: VectorizeParams | None = None):
    """有序点列（px）→ 实体链（px）。

    返回 ``(entities, diag)``。
    """
    params = params or VectorizeParams()
    P = dedupe_consecutive(points)
    if params.smooth_window and params.smooth_window >= 3:
        P = smooth_polyline(P, params.smooth_window)
    if len(P) < 3:
        raise GeometryError(f'矢量化需要至少 3 个轮廓点，收到 {len(P)}')

    noise = estimate_point_noise(P)
    if params.segment_method == 'grow':
        segs, sinfo = segment_points(P, params)
        tol = sinfo['grow_tol_px']
        n_vertices = len(segs) + 1
    elif params.segment_method == 'dp':
        idx = dp_simplify(P, params.dp_eps)
        segs = _segments_from_breakpoints(P, idx)
        tol, n_vertices, sinfo = None, len(idx), {}
    else:
        raise GeometryError(f"未知分段方式 segment_method={params.segment_method!r}")

    diag = {'n_points': int(len(P)), 'segment_method': params.segment_method,
            'n_dp_vertices': int(n_vertices), 'dp_eps': params.dp_eps,
            'noise_estimate_px': round(noise, 4),
            'grow_tol_px': (round(tol, 4) if tol is not None else None),
            **sinfo,
            'segments': []}

    ents, slices = [], []
    for (i, j) in segs:
        if j <= i:
            continue
        sub = P[i:j + 1]
        kind, fit, info = fit_segment(sub, params, noise=noise)
        if kind == 'line':
            lf2, n_in = fit_line_refined(sub, params, noise=noise)
            fit = (lf2, fit[1])
            info['n_ext_inliers'] = n_in
        ents.append(_make_entity(kind, sub, fit, info))
        slices.append(sub)
        diag['segments'].append({'type': kind, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                  for k, v in info.items()
                                                  if k in ('n_pts', 'n_core_pts', 'line_rms',
                                                           'circle_rms', 'radius',
                                                           'len_ratio', 'sweep_deg',
                                                           'n_ext_inliers', 'n_inliers_ransac',
                                                           'inlier_tol_px',
                                                           'reason', 'used_ransac')}})

    before = len(ents)
    ents, slices = _merge_lines(ents, slices, params)
    ents, slices = _merge_arcs(ents, slices, params)
    diag['n_segments_before_merge'] = before
    diag['n_segments_after_merge'] = len(ents)
    diag['n_arcs'] = sum(1 for e in ents if e['type'] == 'arc')

    ents = apply_junctions(ents)
    diag['continuity_gap_max'] = max(
        (g for _, g in G.chain_gaps(ents)), default=0.0)
    return ents, diag
