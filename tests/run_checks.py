"""G 代码契约校验器
用法:
    python tests/run_checks.py --dir tests/out_v1 --ik legacy --label v1
    python tests/run_checks.py --dir tests/out_v2 --ik fanuc  --label v2

校验项:
    C1 圆弧圆心正确性（由起点 + I/K 反推圆心，与真值比对）
    C2 圆弧旋向正确性（计算实际切削圆弧中点，与真值比对 → 抓 270° 大圆弧）
    C3 脉冲当量量化（所有坐标必须是 pulse 的整数倍）
    C4 无重复几何段（同一段被切两遍）
    C5 安全退刀（G00 改变 Z 时必须先抬到安全 X）
    C6 G91 增量模式（累加后必须复现 G90 的绝对位置）
"""
import argparse
import math
import os
import re
import sys

SAFE_CLEARANCE = 2.0

WORD = re.compile(r'([XZIK])\s*(-?\d+(?:\.\d+)?)')


def parse(path, pulse=0.01):
    """解析 .nc，返回 (指令列表, 累计绝对位置列表)。"""
    cmds = []
    pos = (0.0, 0.0)          # (Z, X)
    g91 = False
    for raw in open(path, encoding='utf-8'):
        line = raw.strip()
        if line.startswith('G91'):
            g91 = True
            continue
        m = re.match(r'^(G0[0-3])\b', line)
        if not m:
            continue
        g = m.group(1)
        vals = {k: float(v) for k, v in WORD.findall(line)}
        tgt = dict(vals)
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


def arc_center(start, cmd, ik):
    """由起点 + I/K 反推圆心。"""
    if cmd['I'] is None or cmd['K'] is None:
        return None
    if ik == 'fanuc':          # I → X 轴, K → Z 轴
        return (start[0] + cmd['K'], start[1] + cmd['I'])
    return (start[0] + cmd['I'], start[1] + cmd['K'])   # legacy: I → Z, K → X


def arc_mid(center, start, end, ccw):
    a1 = math.atan2(start[1] - center[1], start[0] - center[0])
    a2 = math.atan2(end[1] - center[1], end[0] - center[0])
    sw = (a2 - a1) % (2 * math.pi) if ccw else -((a1 - a2) % (2 * math.pi))
    am = a1 + sw / 2.0
    r = math.hypot(start[0] - center[0], start[1] - center[1])
    return (center[0] + r * math.cos(am), center[1] + r * math.sin(am)), abs(sw)


def near(p, q, tol):
    return abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol


class Report:
    def __init__(self, label):
        self.label = label
        self.rows = []

    def add(self, cid, name, ok, detail):
        self.rows.append((cid, name, ok, detail))

    def dump(self):
        print(f"\n{'='*78}\n  校验对象: {self.label}\n{'='*78}")
        for cid, name, ok, detail in self.rows:
            print(f"  [{'PASS' if ok else 'FAIL'}] {cid} {name:<28} {detail}")
        n_ok = sum(1 for r in self.rows if r[2])
        print(f"{'-'*78}\n  结果: {n_ok}/{len(self.rows)} 通过")
        return n_ok, len(self.rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--ik', choices=['fanuc', 'legacy'], default='fanuc')
    ap.add_argument('--label', default='?')
    ap.add_argument('--pulse', type=float, default=0.01)
    a = ap.parse_args()

    D = a.dir
    rep = Report(a.label)

    def load(name):
        p = os.path.join(D, name)
        return parse(p, a.pulse) if os.path.exists(p) else (None, False)

    # ---------------- C1/C2  bulge 圆弧 ----------------
    cmds, _ = load('bulge.nc')
    if cmds is None:
        rep.add('C1', 'bulge 圆弧圆心', False, '缺少 bulge.nc')
        rep.add('C2', 'bulge 圆弧旋向', False, '缺少 bulge.nc')
    else:
        arcs = [c for c in cmds if c['g'] in ('G02', 'G03')]
        if not arcs:
            rep.add('C1', 'bulge 圆弧圆心', False, '无圆弧输出')
            rep.add('C2', 'bulge 圆弧旋向', False, '无圆弧输出')
        else:
            c = arcs[0]
            ctr = arc_center(c['start'], c, a.ik)
            truth = (-10.0, 20.0)
            ok1 = ctr is not None and near(ctr, truth, 0.01)
            rep.add('C1', 'bulge 圆弧圆心', ok1,
                    f'输出圆心=({ctr[0]:.3f},{ctr[1]:.3f}) 真值=({truth[0]},{truth[1]})')
            ccw = c['g'] == 'G03'
            mid, sw = arc_mid(ctr, c['start'], c['end'], ccw)
            exp = (-2.9289, 12.9289)
            ok2 = near(mid, exp, 0.05) and abs(math.degrees(sw) - 90) < 1
            rep.add('C2', 'bulge 圆弧旋向/包角', ok2,
                    f'包角={math.degrees(sw):.1f}° 中点=({mid[0]:.3f},{mid[1]:.3f}) 期望90°/({exp[0]:.3f},{exp[1]:.3f})')

    # ---------------- C1/C2  车削圆角 ----------------
    cmds, _ = load('shaft.nc')
    if cmds is None:
        rep.add('C1', 'shaft 圆角圆心', False, '缺少 shaft.nc')
        rep.add('C2', 'shaft 圆角旋向', False, '缺少 shaft.nc')
    else:
        arcs = [c for c in cmds if c['g'] in ('G02', 'G03')]
        if not arcs:
            rep.add('C1', 'shaft 圆角圆心', False, '无圆弧输出')
            rep.add('C2', 'shaft 圆角旋向', False, '无圆弧输出')
        else:
            c = arcs[0]
            ctr = arc_center(c['start'], c, a.ik)
            truth = (35.0, 15.0)
            ok1 = ctr is not None and near(ctr, truth, 0.01)
            rep.add('C1', 'shaft 圆角圆心', ok1,
                    f'输出圆心=({ctr[0]:.3f},{ctr[1]:.3f}) 真值={truth}')
            ccw = c['g'] == 'G03'
            mid, sw = arc_mid(ctr, c['start'], c['end'], ccw)
            exp = (31.4635, 18.5355)
            ok2 = near(mid, exp, 0.05) and abs(math.degrees(sw) - 90) < 1
            rep.add('C2', 'shaft 圆角旋向/包角', ok2,
                    f'包角={math.degrees(sw):.1f}° 中点=({mid[0]:.3f},{mid[1]:.3f}) 期望90°/({exp[0]:.3f},{exp[1]:.3f})')

    # ---------------- C1  整圆圆心 + 半径一致性 ----------------
    cmds, _ = load('circle.nc')
    if cmds is None:
        rep.add('C1', '整圆圆心/半径', False, '缺少 circle.nc')
    else:
        arcs = [c for c in cmds if c['g'] in ('G02', 'G03')]
        truth = (-45.0, 0.0)
        R = 3.0
        bad = []
        for c in arcs:
            ctr = arc_center(c['start'], c, a.ik)
            if ctr is None:
                bad.append('无I/K')
                continue
            r1 = math.hypot(c['start'][0] - ctr[0], c['start'][1] - ctr[1])
            r2 = math.hypot(c['end'][0] - ctr[0], c['end'][1] - ctr[1])
            if not near(ctr, truth, 0.01) or abs(r1 - R) > 0.01 or abs(r2 - R) > 0.01:
                bad.append(f'圆心({ctr[0]:.2f},{ctr[1]:.2f}) R起{r1:.2f}/R终{r2:.2f}')
        ok = len(arcs) == 2 and not bad
        rep.add('C1', '整圆圆心/半径', ok,
                f'共{len(arcs)}段(期望2); 应=圆心({truth[0]},{truth[1]}) R{R}; 异常: {bad or "无"}')

    # ---------------- C3  脉冲当量量化 ----------------
    cmds, _ = load('pulse.nc')
    if cmds is None:
        rep.add('C3', '脉冲当量量化', False, '缺少 pulse.nc')
    else:
        off = []
        for c in cmds:
            for v in (c['end'][0], c['end'][1]):
                k = v / a.pulse
                if abs(k - round(k)) > 1e-6:
                    off.append(f'{v:.4f}')
        rep.add('C3', '脉冲当量量化', not off,
                f'pulse=0.1, 不在网格上的坐标: {sorted(set(off))[:5] or "无"}')

    # ---------------- C4  重复几何段 ----------------
    cmds, _ = load('shaft.nc')
    if cmds is None:
        rep.add('C4', '重复几何段', False, '缺少 shaft.nc')
    else:
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

    # ---------------- C5  安全退刀 ----------------
    if cmds is None:
        rep.add('C5', '安全退刀', False, '缺少 shaft.nc')
    else:
        xs = [c['end'][1] for c in cmds if c['g'] != 'G00']
        safe = max(xs) + SAFE_CLEARANCE if xs else 0.0
        bad = []
        for c in cmds:
            if c['g'] != 'G00':
                continue
            dz = abs(c['end'][0] - c['start'][0])
            if dz > 1e-9 and c['end'][1] < safe - 1e-9:
                bad.append(f'Z {c["start"][0]:.1f}→{c["end"][0]:.1f} @X={c["end"][1]:.1f}')
        rep.add('C5', '安全退刀', not bad,
                f'安全X={safe:.1f}; 危险快速移动 {len(bad)} 处 {bad[:3]}')

    # ---------------- C6  G91 增量模式 ----------------
    g90, _ = load('shaft.nc')
    g91, is_g91 = load('shaft_g91.nc')
    if g90 is None or g91 is None:
        rep.add('C6', 'G91 增量模式', False, '缺少 shaft.nc 或 shaft_g91.nc')
    else:
        ok = is_g91 and len(g90) == len(g91) and all(
            near(d['end'], s['end'], 0.02) for d, s in zip(g91, g90))
        e91 = g91[-1]['end'] if g91 else (0, 0)
        e90 = g90[-1]['end'] if g90 else (0, 0)
        rep.add('C6', 'G91 增量模式', ok,
                f'G91末端=({e91[0]:.3f},{e91[1]:.3f}) vs G90末端=({e90[0]:.3f},{e90[1]:.3f})')

    n_ok, n = rep.dump()
    return 0 if n_ok == n else 1


if __name__ == '__main__':
    sys.exit(main())
