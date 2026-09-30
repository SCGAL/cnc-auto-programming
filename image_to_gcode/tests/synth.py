"""合成真值测试图生成器 + 真值几何定义（M1 交付物 1/2）。

设计思路
--------
真值几何由本程序自己定义 → 光栅化成图像 → 跑流水线 → 与真值比对。
因为真值完全已知，误差可以精确到毫米，不必依赖任何外部标注。

真值几何
--------
车削件上半部外轮廓，工件坐标 (Z, X)，单 mm，X 为半径值。沿加工方向（Z 从 0 递减）：

    序号  类型   说明
    E1    line   右端面（车削 Z 原点），X: 0 → R1
    E2    line   大端外圆，X = R1
    E3    arc    凸圆角 R=rf（切 E2 与肩面）
    E4    line   肩面（竖直，长 FACE_MIN）
    E5    arc    凹圆角 R=rf（切肩面与小端外圆）
    E6    line   小端外圆，X = R2
    E7    line   45° 倒角
    E8    line   末段外圆，X = R3
    E9    line   左端面，X: R3 → 0

⚠️ 与《立项与可行性分析》§6.1 示例真值的两处差异（已在精度报告中记录）：

1. **Z 非单调 / 外圆重叠**：文档示例为 ``('line',(0,20),(-35,20))`` 紧接
   ``('arc',(-35,20),(-30,15),center=(-35,15),r=5)``，即 Z 从 −35 **回退**到 −30，
   使 Ø40 段(Z 0~−35) 与 Ø30 段(Z −30~−60) 在 Z 区间上重叠 —— 同一轴向位置
   出现两个直径，物理上不成立。本实现改为 Z 严格单调递减。
2. **"一条圆角连接两段外圆"在几何上不可能**：文档示例的圆心 (-35,15) 到
   外圆 X=20 的距离为 5 ≠ r，实际是 90° 折角而非相切；更本质地，两条**平行**
   直线不存在公共内切圆，故"圆角直接连接两段直径"无解。真实过渡必然经过
   肩面 / 锥面 / 倒角。本实现按 **凸圆弧 + 肩面 + 凹圆弧** 建模，并由
   :func:`check_truth` 逐条校验真实相切性与切点位置。

用两条 90° 弧（一凸一凹）而不是一条，是为了同时检验圆弧识别对**两种旋向**的判别。

光栅化约定
----------
* 用**圆盘笔刷的闵可夫斯基和**画线（沿折线密集打点 + 实心圆），
  而不是 ``ImageDraw.line(width=w)``：后者对水平/斜线覆盖率各向异性，
  会污染"真值"本身。圆盘笔刷是各向同性的标准笔画模型。
* 4 倍超采样 + BOX 均值降采样得到灰度覆盖率，再按 0.5 覆盖率二值化。
* 倾斜：在 (Z, X) 平面内旋转几何后再映射（等价于旋转位图，但少一次重采样）。
* 噪声：对**二值图**做固定种子的椒盐翻转（NFR-3 可复现）。

用法::

    python -m tests.synth --list
    python -m tests.synth --build tests/synth          # 生成标准矩阵
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from i2g import geometry as G  # noqa: E402

DEFAULT_SEED = 20260928


# =============================================================================
# 真值几何
# =============================================================================

R_BASE = 20.0       # 大端基准半径（fillet_r 较大时按 3·rf 放大，见下）
L1 = 25.0           # 大端外圆长度（到肩面）
L2 = 20.0           # 肩面之后小端外圆的长度
L3 = 15.0           # 末段长度
CHAMFER = 5.0       # 45° 倒角的轴向长度（= 径向高度）
FACE_MIN = 2.0      # 肩面直线段的最小长度


def build_profile(fillet_r: float = 5.0):
    """构造 Z 单调、相切自洽的车削上半部外轮廓真值。

    几何构成（沿加工方向，Z 从 0 递减）::

        E1 右端面        竖直，X: 0 → R1
        E2 大端外圆      水平，X = R1
        E3 凸圆角 R=rf   切 E2 于 (z1+rf, R1)、切肩面于 (z1, R1−rf)
        E4 肩面          竖直，X: R1−rf → R2+rf（长 = FACE_MIN）
        E5 凹圆角 R=rf   切肩面于 (z1, R2+rf)、切小端外圆于 (z1−rf, R2)
        E6 小端外圆      水平，X = R2
        E7 45° 倒角      ΔZ = ΔX = c
        E8 末段外圆      水平，X = R3 = R2 − c
        E9 左端面        竖直，X: R3 → 0

    为什么不是"一条圆角直接连接两段外圆"：两条平行直线不存在公共内切圆，
    该构型在几何上不可能（立项文档 §6.1 的示例真值就栽在这里，见模块 docstring）。
    真实车削件的直径过渡必然经过肩面（或锥面/倒角），故此处按肩面过渡建模。

    ``fillet_r`` 同时用于凸圆角与凹圆角：一条 90° 凸弧 + 一条 90° 凹弧，
    可同时检验圆弧识别对**两种旋向**的判别能力。

    返回 ``(entities, geom)``。
    """
    rf = float(fillet_r)
    if rf <= 0:
        raise ValueError(f'fillet_r 必须为正，收到 {fillet_r}')
    R1 = max(R_BASE, 3.0 * rf)          # 保证肩面能让两条 90° 弧都放得下
    face = rf + FACE_MIN                 # 肩面总长（>= rf，否则凹圆角放不下）
    R2 = R1 - rf - face                  # = R1 − 2rf − 2
    if R2 <= 1.0:
        raise ValueError(f'fillet_r={rf} 过大：小端半径仅 {R2:.3f} mm')
    c = min(CHAMFER, R2 / 2.0)
    R3 = R2 - c

    z0 = 0.0
    z1 = -L1                              # 肩面所在 Z（大端外圆的名义终点）
    z2 = z1 - rf - L2                     # 小端外圆终点
    z3 = z2 - c                           # 倒角终点
    z4 = z3 - L3                          # 左端面

    ents = [
        {'type': 'line', 'start': (z0, 0.0), 'end': (z0, R1)},
        {'type': 'line', 'start': (z0, R1), 'end': (z1 + rf, R1)},
        {'type': 'arc', 'start': (z1 + rf, R1), 'end': (z1, R1 - rf),
         'center': (z1 + rf, R1 - rf), 'radius': rf, 'ccw': True},
        {'type': 'line', 'start': (z1, R1 - rf), 'end': (z1, R2 + rf)},
        {'type': 'arc', 'start': (z1, R2 + rf), 'end': (z1 - rf, R2),
         'center': (z1 - rf, R2 + rf), 'radius': rf, 'ccw': False},
        {'type': 'line', 'start': (z1 - rf, R2), 'end': (z2, R2)},
        {'type': 'line', 'start': (z2, R2), 'end': (z3, R3)},
        {'type': 'line', 'start': (z3, R3), 'end': (z4, R3)},
        {'type': 'line', 'start': (z4, R3), 'end': (z4, 0.0)},
    ]
    geom = {
        'fillet_r': rf, 'r_max': R1, 'ref_diameter': 2.0 * R1,
        'r_shoulder_top': R1 - rf, 'r_small': R2, 'r_last': R3,
        'face_mm': face, 'chamfer': c, 'z_end': z4,
        'z_vertices': [z0, z1, z2, z3, z4],
        'length_mm': abs(z4),
        'n_arcs': 2,
    }
    return ents, geom


def check_truth(ents, tol=1e-9):
    """真值自校验 —— 真值本身错了，整个测量就废了，所以必须先验证真值。

    检查：链条连续、Z 单调递减、X >= 0、圆弧端点确实在圆上、圆角与两侧外圆相切。
    返回问题字符串列表（空 = 通过）。
    """
    bad = []
    for i, e in enumerate(ents):
        if e['type'] == 'arc':
            r = e['radius']
            for tag, p in (('start', e['start']), ('end', e['end'])):
                err = abs(G.dist(e['center'], p) - r)
                if err > 1e-9:
                    bad.append(f'E{i} 圆弧 {tag} 不在圆上: |center-p|-r = {err:.3e}')
    for i in range(1, len(ents)):
        gap = G.dist(ents[i - 1]['end'], ents[i]['start'])
        if gap > tol:
            bad.append(f'E{i-1}→E{i} 不连续: 间隙 {gap:.3e} mm')
    zs = [p[0] for e in ents for p in (e['start'], e['end'])]
    for i in range(1, len(zs)):
        if zs[i] > zs[i - 1] + tol:
            bad.append(f'Z 非单调递减: {zs[i-1]:.4f} → {zs[i]:.4f}')
            break
    xs = [p[1] for e in ents for p in (e['start'], e['end'])]
    if min(xs) < -tol:
        bad.append(f'X 出现负值（轴线以下）: {min(xs):.4f}')
    if abs(min(xs)) > tol:
        bad.append(f'轮廓未落到中心线 X=0: 最小 X = {min(xs):.4f}')
    # 圆弧与相邻直线必须真相切：圆心到直线距离 == 半径，且切点落在线段范围内
    for i, e in enumerate(ents):
        if e['type'] != 'arc':
            continue
        for j in (i - 1, i + 1):
            if not (0 <= j < len(ents)) or ents[j]['type'] != 'line':
                continue
            d = ents[j]
            p, q = tuple(d['start']), tuple(d['end'])
            vx, vy = q[0] - p[0], q[1] - p[1]
            n = math.hypot(vx, vy)
            if n < 1e-12:
                continue
            ux, uy = vx / n, vy / n
            t = (e['center'][0] - p[0]) * ux + (e['center'][1] - p[1]) * uy
            proj = (p[0] + ux * t, p[1] + uy * t)
            dd = G.dist(e['center'], proj)
            if abs(dd - e['radius']) > 1e-6:
                bad.append(f'E{i} 与 E{j} 不相切: 圆心到线距离 {dd:.6f} vs R {e["radius"]:.6f}')
            elif not (-1e-9 <= t <= n + 1e-9):
                bad.append(f'E{i} 与 E{j} 的切点落在 E{j} 线段之外 '
                           f'(t={t:.4f}, 线段长 {n:.4f}) —— 属"与延长线相切"的伪相切')
    return bad


def truth_vertices(ents, tol=1e-7):
    """真值顶点（相邻实体结合点）序列；首末点也计入。"""
    vs = [tuple(ents[0]['start'])]
    for i in range(1, len(ents)):
        a, b = ents[i - 1]['end'], ents[i]['start']
        vs.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))
    vs.append(tuple(ents[-1]['end']))
    return vs


def truth_polyline(ents, step=0.05):
    return G.entities_polyline(ents, max_step=step)


# =============================================================================
# 测试用例规格
# =============================================================================

@dataclass(frozen=True)
class Spec:
    """一个合成用例的参数。对应立项文档 §6.1 的 MATRIX 五个维度。"""
    px_per_mm: float = 10.0
    line_width: float = 2.0
    skew_deg: float = 0.0
    noise: float = 0.0
    fillet_r: float = 5.0
    seed: int = DEFAULT_SEED
    margin_px: int = 24
    supersample: int = 4
    tag: str = ''

    @property
    def name(self) -> str:
        return (self.tag or
                f'px{self.px_per_mm:g}_w{self.line_width:g}_'
                f's{self.skew_deg:+g}_n{self.noise:g}_r{self.fillet_r:g}')

    def to_json(self):
        return asdict(self)

    @staticmethod
    def from_json(d):
        return Spec(**d)


@dataclass
class Case:
    """渲染好的合成用例：图像 + 真值 + 映射元数据。"""
    spec: Spec
    gray: np.ndarray                    # float64 覆盖率 0..1（抗锯齿灰度，真值图）
    entities: list                      # 真值实体（未旋转的工件坐标，mm）
    geom: dict
    meta: dict = field(default_factory=dict)

    @property
    def name(self):
        return self.spec.name

    @property
    def mask(self):
        """0.5 覆盖率二值化结果（供需要二值图的测试使用）。"""
        return self.gray >= 0.5


def matrix(px_per_mm=(5, 10, 20), line_width=(1, 2, 3), skew_deg=(0, 1, 2, -2),
           noise=(0.0, 0.01, 0.02), fillet_r=(2, 5, 15), **over):
    """立项文档 §6.1 的五维参数矩阵（笛卡尔积）。"""
    out = []
    for a in px_per_mm:
        for b in line_width:
            for c in skew_deg:
                for d in noise:
                    for e in fillet_r:
                        out.append(Spec(px_per_mm=a, line_width=b, skew_deg=c,
                                        noise=d, fillet_r=e, **over))
    return out


def dod_cases(**over):
    """DoD 门槛子集：px_per_mm >= 10 且 noise = 0 的全部组合（72 例）。"""
    return matrix(px_per_mm=(10, 20), noise=(0.0,), **over)


# =============================================================================
# 光栅化
# =============================================================================

def _rotate_mm(pts, theta_deg):
    """在 (Z, X) 平面内绕点集包围盒中心旋转（等价于旋转位图，且无重采样损失）。"""
    if abs(theta_deg) < 1e-12:
        return [tuple(p) for p in pts]
    P = np.asarray(pts, dtype=float)
    c = (P.min(axis=0) + P.max(axis=0)) / 2.0
    th = math.radians(theta_deg)
    ct, st = math.cos(th), math.sin(th)
    Q = P - c
    return [(float(c[0] + q[0] * ct - q[1] * st),
             float(c[1] + q[0] * st + q[1] * ct)) for q in Q]


def render(spec: Spec) -> Case:
    """按 spec 渲染一张合成真值图。"""
    ents, geom = build_profile(spec.fillet_r)
    bad = check_truth(ents)
    if bad:
        raise AssertionError('真值几何自校验失败:\n  ' + '\n  '.join(bad))

    s = float(spec.px_per_mm)
    ss = int(spec.supersample)
    step_mm = min(0.25 / s, 0.5 * spec.line_width / s)

    pts = _rotate_mm(G.entities_polyline(ents, max_step=step_mm), spec.skew_deg)
    P = np.asarray(pts, dtype=float)
    u_min, u_max = float(P[:, 0].min()), float(P[:, 0].max())
    v_min, v_max = float(P[:, 1].min()), float(P[:, 1].max())
    m = int(spec.margin_px)

    W = int(math.ceil((u_max - u_min) * s)) + 2 * m + 1
    H = int(math.ceil((v_max - v_min) * s)) + 2 * m + 1

    # 映射: u = u_min + (col - m)/s ; v = v_max - (row - m)/s
    cols_ss = m * ss + (P[:, 0] - u_min) * s * ss
    rows_ss = m * ss + (v_max - P[:, 1]) * s * ss

    canvas = Image.new('L', (W * ss, H * ss), 0)
    drw = ImageDraw.Draw(canvas)
    rad = max(0.5, spec.line_width * ss / 2.0)
    for cx, cy in zip(cols_ss, rows_ss):
        drw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=255)

    gray = canvas.resize((W, H), Image.BOX)
    cov = np.asarray(gray, dtype=np.float64) / 255.0

    # 椒盐噪声作用在**覆盖率**上：翻转像素即 cov → 1 − cov。
    # 这样噪声模型对抗锯齿图与二值图是同一个定义。
    if spec.noise > 0:
        rng = np.random.default_rng(spec.seed)
        n_flip = int(round(float(spec.noise) * cov.size))
        if n_flip > 0:
            idx = rng.choice(cov.size, size=min(n_flip, cov.size), replace=False)
            flat = cov.reshape(-1)
            flat[idx] = 1.0 - flat[idx]

    mask = cov >= 0.5
    meta = {
        'px_per_mm': s,
        'mm_per_px': 1.0 / s,
        'supersample': ss,
        'margin_px': m,
        'image_size': [W, H],
        'rotation_deg': float(spec.skew_deg),
        'u_min': u_min, 'v_max': v_max,
        'mapping': 'u = u_min + (col - margin)/s ; v = v_max - (row - margin)/s',
        'ink_px': int(mask.sum()),
        'antialiased': True,
        'truth_bbox_mm': {'z': [min(p[0] for p in G.entities_polyline(ents, 0.05)),
                               max(p[0] for p in G.entities_polyline(ents, 0.05))],
                          'x': [0.0, geom['r_max']]},
    }
    return Case(spec=spec, gray=cov, entities=ents, geom=geom, meta=meta)


def render_fill(spec: Spec, *, close_along_axis=True) -> Case:
    """渲染**实心填充轮廓图**（``--ink fill`` 的兼容输入）。

    真值轮廓 + 沿轴线的封底边构成闭合多边形，整体填充。
    用于验证 ``profile.extract(mode='fill')`` 这条代码路径 ——
    它不在主测试矩阵里，必须有独立测试守着，否则很容易长期处于"没人跑过"的状态。
    """
    ents, geom = build_profile(spec.fillet_r)
    bad = check_truth(ents)
    if bad:
        raise AssertionError('真值几何自校验失败:\n  ' + '\n  '.join(bad))
    poly = G.entities_polyline(ents, max_step=min(0.1 / spec.px_per_mm, 0.02))
    closed = list(poly)
    if close_along_axis:
        closed = closed + [(poly[-1][0], 0.0), (poly[0][0], 0.0)]

    s, ss, m = float(spec.px_per_mm), int(spec.supersample), int(spec.margin_px)
    P = np.asarray(closed, dtype=float)
    u_min, u_max = float(P[:, 0].min()), float(P[:, 0].max())
    v_min, v_max = float(P[:, 1].min()), float(P[:, 1].max())
    W = int(math.ceil((u_max - u_min) * s)) + 2 * m + 1
    H = int(math.ceil((v_max - v_min) * s)) + 2 * m + 1

    canvas = Image.new('L', (W * ss, H * ss), 0)
    ImageDraw.Draw(canvas).polygon(
        [(m * ss + (u - u_min) * s * ss, m * ss + (v_max - v) * s * ss)
         for (u, v) in closed], fill=255)
    cov = np.asarray(canvas.resize((W, H), Image.BOX), dtype=np.float64) / 255.0

    meta = {'px_per_mm': s, 'mm_per_px': 1.0 / s, 'supersample': ss,
            'margin_px': m, 'image_size': [W, H], 'rotation_deg': 0.0,
            'u_min': u_min, 'v_max': v_max, 'antialiased': True, 'ink': 'fill',
            'truth_bbox_mm': {'z': [u_min, u_max], 'x': [0.0, geom['r_max']]}}
    return Case(spec=spec, gray=cov, entities=ents, geom=geom, meta=meta)


# =============================================================================
# 落盘 / 读回
# =============================================================================

def save_case(case: Case, outdir: str):
    """落盘为**抗锯齿灰度 PNG**（与 CAD 导出/截图一致），而不是硬二值图。

    硬二值化会丢掉覆盖率里的亚像素信息，实测会让数字骨架在曲笔画上产生
    +2.4 px 的系统性半径偏差（R=52.4 vs 真值 50 @10px/mm）—— 那是测试集
    自身失真，不是算法误差。
    """
    os.makedirs(outdir, exist_ok=True)
    img_path = os.path.join(outdir, case.name + '.png')
    Image.fromarray(np.clip(np.round(case.gray * 255.0), 0, 255).astype(np.uint8),
                    'L').save(img_path)
    truth = {
        'name': case.name,
        'spec': case.spec.to_json(),
        'entities': case.entities,
        'geom': case.geom,
        'meta': case.meta,
    }
    with open(os.path.join(outdir, case.name + '.truth.json'), 'w', encoding='utf-8') as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)
    return img_path


def load_case(path: str) -> Case:
    """从 PNG 或 .truth.json 读回用例。"""
    if path.endswith('.json'):
        truth_path = path
        img_path = path[:-len('.truth.json')] + '.png'
    else:
        img_path = path
        truth_path = path[:-len('.png')] + '.truth.json'
    with open(truth_path, encoding='utf-8') as f:
        t = json.load(f)
    gray = np.asarray(Image.open(img_path).convert('L'), dtype=np.float64) / 255.0
    ents = [{k: (tuple(v) if isinstance(v, list) and k in ('start', 'end', 'center') else v)
             for k, v in e.items()} for e in t['entities']]
    return Case(spec=Spec.from_json(t['spec']), gray=gray, entities=ents,
                geom=t['geom'], meta=t['meta'])


def ensure_case(spec: Spec, cachedir: str) -> Case:
    """命中缓存则读回，否则渲染并落盘。加速测试矩阵重复运行。"""
    name = spec.name
    jpath = os.path.join(cachedir, name + '.truth.json')
    if os.path.exists(jpath):
        return load_case(jpath)
    case = render(spec)
    save_case(case, cachedir)
    return case


# =============================================================================
# CLI
# =============================================================================

def _main():
    ap = argparse.ArgumentParser(description='合成真值测试图生成器')
    ap.add_argument('--build', metavar='DIR', help='生成标准矩阵到 DIR')
    ap.add_argument('--dod', metavar='DIR', help='只生成 DoD 门槛子集 (px>=10, noise=0)')
    ap.add_argument('--list', action='store_true', help='列出标准矩阵参数')
    ap.add_argument('--profile', action='store_true', help='打印真值几何并自校验')
    a = ap.parse_args()

    if a.list:
        cases = matrix()
        print(f'标准矩阵: {len(cases)} 例 '
              f'(px_per_mm x line_width x skew x noise x fillet_r)')
        print(f'  px_per_mm : 5, 10, 20')
        print(f'  line_width: 1, 2, 3')
        print(f'  skew_deg  : 0, 1, 2, -2')
        print(f'  noise     : 0.0, 0.01, 0.02')
        print(f'  fillet_r  : 2, 5, 15')
        print(f'DoD 门槛子集 (px>=10, noise=0): {len(dod_cases())} 例')
        return 0

    if a.profile:
        for fr in (2, 5, 15):
            ents, geom = build_profile(fr)
            bad = check_truth(ents)
            print(f'fillet_r={fr:g}: Z 跨 {geom["length_mm"]:.3f} mm, '
                  f'R1={geom["r_max"]:.3f}(Ø{geom["ref_diameter"]:g}), '
                  f'肩顶 R={geom["r_shoulder_top"]:.3f}, 小端 R={geom["r_small"]:.3f}, '
                  f'末段 R={geom["r_last"]:.3f}, 倒角 {geom["chamfer"]:.3f} mm, '
                  f'肩面 {geom["face_mm"]:.3f} mm, 圆弧 {geom["n_arcs"]} 段 -> '
                  f'{"OK" if not bad else "FAIL: " + "; ".join(bad)}')
        return 0

    todo = None
    if a.dod:
        todo, outdir = dod_cases(), a.dod
    elif a.build:
        todo, outdir = matrix(), a.build
    if todo is None:
        ap.print_help()
        return 1

    for i, spec in enumerate(todo):
        ensure_case(spec, outdir)
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(todo)} ...')
    print(f'已生成 {len(todo)} 例 → {outdir}')
    return 0


if __name__ == '__main__':
    sys.exit(_main())
