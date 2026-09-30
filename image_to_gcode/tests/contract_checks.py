"""G 代码契约校验器（对标 DXF 版 ``tests/run_checks.py`` 的 9 项断言）。

DXF 版的 9 项断言是针对"已知 DXF 真值"硬编码的（bulge.nc / shaft.nc …）。
本模块把它**参数化**：真值由合成用例给出，校验项与判据完全沿用 DXF 版口径，
从而可以套在任意一张合成图生成的 .nc 上。

校验项
------
============  ==========================================================
C1a           圆弧圆心正确性：由「起点 + I/K」反推圆心，与真值比对
C1b           圆弧半径一致性：圆心到两端点距离 == R
C2a           旋向正确性：计算**实际被切削的圆弧中点**与真值中点比对
C2b           包角正确性：实际包角与真值包角一致
C3            脉冲当量量化：所有坐标必须是 pulse 的整数倍
C4            无重复几何段：同一段几何不被切削两次
C5            安全退刀：任何改变 Z 的 G00 执行前 X 必须已抬到安全高度
C6            G91 增量模式：增量累加后必须复现 G90 的绝对位置
============  ==========================================================

坐标约定与 DXF 版一致：``(Z, X)``，X 为半径值。``--ik fanuc`` 时 I→X、K→Z。
"""

from __future__ import annotations

import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from i2g import geometry as G            # noqa: E402  (仅用于角度计算)

SAFE_CLEARANCE = 2.0
WORD = re.compile(r'([XZIK])\s*(-?\d+(?:\.\d+)?)')


def parse(path, pulse=0.01):
    """解析 .nc，返回 (指令列表, 是否增量模式)。"""
    cmds = []
    pos = (0.0, 0.0)
    g91 = False
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if line.startswith('G91'):
                g91 = True
                continue
            m = re.match(r'^(G0[0-3])\b', line)
            if not m:
                continue
            g = m.group(1)
            vals = {k: float(v) for k, v in WORD.findall(line)}
            if g91:
                nz = pos[0] + vals.get('Z', 0.0)
                nx = pos[1] + vals.get('X', 0.0)
            else:
                nz = vals.get('Z', pos[0])
                nx = vals.get('X', pos[1])
            cmds.append({'g': g, 'start': pos, 'end': (nz, nx), 'raw': line,
                         'I': vals.get('I'), 'K': vals.get('K')})
            pos = (nz, nx)
    return cmds, g91


def arc_center(start, cmd, ik='fanuc'):
    if cmd['I'] is None or cmd['K'] is None:
        return None
    if ik == 'fanuc':
        return (start[0] + cmd['K'], start[1] + cmd['I'])
    return (start[0] + cmd['I'], start[1] + cmd['K'])


def arc_mid_sweep(center, start, end, ccw):
    a1 = math.atan2(start[1] - center[1], start[0] - center[0])
    a2 = math.atan2(end[1] - center[1], end[0] - center[0])
    sw = (a2 - a1) % (2 * math.pi) if ccw else -((a1 - a2) % (2 * math.pi))
    am = a1 + sw / 2.0
    r = math.hypot(start[0] - center[0], start[1] - center[1])
    return (center[0] + r * math.cos(am), center[1] + r * math.sin(am)), abs(sw)


def _near(p, q, tol):
    return abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol


class Report:
    def __init__(self, label):
        self.label = label
        self.rows = []

    def add(self, cid, name, ok, detail):
        self.rows.append((cid, name, bool(ok), detail))

    def dump(self):
        out = [f"\n{'='*78}", f"  G 代码契约校验: {self.label}", '=' * 78]
        for cid, name, ok, detail in self.rows:
            out.append(f"  [{'PASS' if ok else 'FAIL'}] {cid:<4s} {name:<26s} {detail}")
        n_ok = sum(1 for r in self.rows if r[2])
        out.append('-' * 78)
        out.append(f"  结果: {n_ok}/{len(self.rows)} 通过")
        return '\n'.join(out)

    @property
    def n_ok(self):
        return sum(1 for r in self.rows if r[2])

    @property
    def n_total(self):
        return len(self.rows)


def check(nc_path, truth_entities, *, ik='fanuc', pulse=0.01,
          label=None, gcode_abs=None, tol_center=0.2, tol_radius=0.2,
          tol_mid=0.2):
    """对 ``nc_path`` 跑 9 项契约断言。

    ``truth_entities`` 为真值实体列表（mm，与生成 .nc 时同一坐标系的 (Z,X)）。
    ``gcode_abs`` 可选：同一轮廓的 G90 版本文本，用于 C6。

    容差口径说明：DXF 版把 ``tol_center`` 设成 0.01 mm，因为它比对的是
    **精确矢量几何**（生成器有无缺陷是二值的）。本模块比对的是
    **图像提取出的几何**，它天然带有识别误差（DoD 预算 0.2 mm），
    故默认容差取 0.2 mm —— 校验的仍是"G 代码是否忠实编码了输入几何"，
    而不是"提取精度是否达标"（后者由 ``tests/metrics.py`` 的 6 项指标负责）。
    """
    rep = Report(label or os.path.basename(nc_path))
    cmds, is_g91 = parse(nc_path, pulse)
    if not cmds:
        rep.add('C1a', '圆弧圆心', False, '未解析到任何运动指令')
        return rep

    truth_arcs = [e for e in truth_entities if e['type'] == 'arc']
    out_arcs = [c for c in cmds if c['g'] in ('G02', 'G03')]

    # ---- C1a/C1b：圆弧圆心与半径一致性
    if not truth_arcs:
        rep.add('C1a', '圆弧圆心', True, '真值无圆弧，跳过')
        rep.add('C1b', '圆弧半径一致性', True, '真值无圆弧，跳过')
    elif not out_arcs:
        rep.add('C1a', '圆弧圆心', False, f'真值有 {len(truth_arcs)} 段圆弧，输出 0 段')
        rep.add('C1b', '圆弧半径一致性', False, '无圆弧输出')
    else:
        worst, bad = 0.0, []
        for c in out_arcs:
            ctr = arc_center(c['start'], c, ik)
            if ctr is None:
                bad.append('缺 I/K')
                continue
            for truth in truth_arcs:
                d = G.dist(ctr, truth['center'])
                if d < 1.0:                       # 就近配对
                    worst = max(worst, d)
                    break
            else:
                bad.append(f'圆心({ctr[0]:.2f},{ctr[1]:.2f}) 找不到对应真值弧')
        rep.add('C1a', '圆弧圆心', worst <= tol_center and not bad,
                f'最大圆心偏差 {worst:.4f} mm (容差 {tol_center}); 异常 {bad[:2] or "无"}')

        bad2, worst2 = [], 0.0
        for c in out_arcs:
            ctr = arc_center(c['start'], c, ik)
            if ctr is None:
                bad2.append('缺 I/K')
                continue
            r1 = math.hypot(c['start'][0] - ctr[0], c['start'][1] - ctr[1])
            r2 = math.hypot(c['end'][0] - ctr[0], c['end'][1] - ctr[1])
            worst2 = max(worst2, abs(r1 - r2))
            if abs(r1 - r2) > max(tol_radius, 2 * pulse + 1e-9):
                bad2.append(f'R起{r1:.3f}/R终{r2:.3f}')
        rep.add('C1b', '圆弧半径一致性', not bad2,
                f'圆心到两端点半径最大差 {worst2:.4f} mm; 异常 {bad2[:2] or "无"}')

    # ---- C2a/C2b：旋向与实际切削中点 / 包角
    if not truth_arcs or not out_arcs:
        rep.add('C2a', '圆弧旋向/中点', True, '无圆弧可比对')
        rep.add('C2b', '圆弧包角', True, '无圆弧可比对')
    else:
        worst_mid, worst_sw, bad3 = 0.0, 0.0, []
        for c in out_arcs:
            ctr = arc_center(c['start'], c, ik)
            if ctr is None:
                continue
            ccw = c['g'] == 'G03'
            mid, sw = arc_mid_sweep(ctr, c['start'], c['end'], ccw)
            best = None
            for truth in truth_arcs:
                tmid = G.arc_midpoint(tuple(truth['center']), tuple(truth['start']),
                                      tuple(truth['end']), bool(truth['ccw']))
                d = G.dist(mid, tmid)
                if best is None or d < best[0]:
                    best = (d, tmid, truth)
            if best is None:
                continue
            worst_mid = max(worst_mid, best[0])
            tsw = G.arc_sweep(tuple(best[2]['center']), tuple(best[2]['start']),
                              tuple(best[2]['end']), bool(best[2]['ccw']))
            worst_sw = max(worst_sw, abs(sw - tsw))
            if best[0] > tol_mid:
                bad3.append(f'中点偏 {best[0]:.3f}')
        rep.add('C2a', '圆弧旋向/中点', worst_mid <= tol_mid and not bad3,
                f'实际切削中点最大偏差 {worst_mid:.4f} mm (容差 {tol_mid}); '
                f'异常 {bad3[:2] or "无"}')
        rep.add('C2b', '圆弧包角', worst_sw <= math.radians(1.0),
                f'包角最大偏差 {math.degrees(worst_sw):.3f}° (容差 1°)')

    # ---- C3：脉冲当量量化
    off = []
    for c in cmds:
        for v in (c['end'][0], c['end'][1]):
            k = v / pulse
            if abs(k - round(k)) > 1e-6:
                off.append(f'{v:.4f}')
    rep.add('C3', '脉冲当量量化', not off,
            f'pulse={pulse}; 不在网格上的坐标 {sorted(set(off))[:5] or "无"}')

    # ---- C4：重复几何段
    seen, dup = set(), 0
    for c in cmds:
        if c['g'] != 'G01':
            continue
        key = tuple(sorted((tuple(round(v, 3) for v in c['start']),
                            tuple(round(v, 3) for v in c['end']))))
        if key in seen:
            dup += 1
        seen.add(key)
    rep.add('C4', '重复几何段', dup == 0, f'重复切削段 {dup} 处')

    # ---- C5：安全退刀
    xs = [c['end'][1] for c in cmds if c['g'] != 'G00']
    safe = max(xs) + SAFE_CLEARANCE if xs else 0.0
    bad5 = []
    for c in cmds:
        if c['g'] != 'G00':
            continue
        dz = abs(c['end'][0] - c['start'][0])
        if dz > 1e-9 and c['end'][1] < safe - 1e-9:
            bad5.append(f'Z {c["start"][0]:.1f}→{c["end"][0]:.1f} @X={c["end"][1]:.1f}')
    rep.add('C5', '安全退刀', not bad5,
            f'安全X={safe:.1f}; 危险快速移动 {len(bad5)} 处 {bad5[:2]}')

    # ---- C6：G91 增量模式
    if gcode_abs is None:
        rep.add('C6', 'G91 增量模式', True, '未提供 G90 对照，跳过')
    else:
        tmp = nc_path + '.g90tmp'
        with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
            f.write(gcode_abs + '\n')
        try:
            g90, _ = parse(tmp, pulse)
            ok = (is_g91 and len(g90) == len(cmds)
                  and all(_near(d['end'], s['end'], 0.02)
                          for d, s in zip(cmds, g90)))
            e91 = cmds[-1]['end']
            e90 = g90[-1]['end'] if g90 else (0, 0)
            rep.add('C6', 'G91 增量模式', ok,
                    f'G91末端=({e91[0]:.3f},{e91[1]:.3f}) vs '
                    f'G90末端=({e90[0]:.3f},{e90[1]:.3f}), 指令数 {len(cmds)}/{len(g90)}')
        finally:
            os.remove(tmp)
    return rep
