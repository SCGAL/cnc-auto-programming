"""
DXF → CNC 车削 G 代码转换器  v2
================================
修复 v1 (dxf_to_gcode.py) 中已实测确认的缺陷，并加入自校验模块。

 F1  bulge 圆弧圆心算到弦的另一侧 → 输出 270° 大圆弧
     实测: 真值圆心 (-10,20)、r10、90°；v1 输出圆心 (0,10)
 F2  整圆拆分的两段半圆都标成 CW → 其中一段切到圆的另一侧
 F3  Z 镜像后未反转圆弧旋向 → 圆弧绕 ≥180° 长边切削
 F4  I/K 按"游标当前位置"计算而非圆弧自身起点 → 排序跳段/不排序时圆心错位
 F5  --mode g91 只改头部注释，坐标仍为绝对值（实机按增量执行 → 撞刀）
 F6  --pulse 只写进注释，坐标从未量化
 F7  同 Z 位置自动合成垂直线段 → 与已存在端面线段重复，刀具来回空切
 F8  快速定位不抬刀（无安全 X）→ 实机撞刀
 F9  弧长统计一律按劣弧，与实际输出旋向矛盾
 F10 文档声明 --safe-z 但从未实现

新增: 轮廓连续性自检 / 圆弧参数一致性自检 / 量化残差报告 /
      I/K 轴对应可配置(Fanuc G18) / 直径编程 / 安全退刀 / 程序头 / ASCII 注释

坐标系约定（内部统一）:
    实体坐标 = (Z, X)，Z = DXF.x，X = DXF.y（半径值）
    圆弧一律用 ccw 布尔量表示 ZX 平面内的旋向（True = 逆时针）

用法: python dxf_to_gcode_v2.py <dxf> --turning --check
"""

from __future__ import annotations

import argparse
import math
import os
import sys

try:
    import ezdxf
except ImportError:
    print("请先安装 ezdxf: pip install ezdxf")
    sys.exit(1)

JOIN_TOL = 0.05          # 轮廓连接容差 (mm)


# =============================================================================
# 几何基元
# =============================================================================

def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def bulge_arc(p1, p2, bulge):
    """由弦两端点 + bulge 求圆心/半径/旋向。

    bulge b = tan(θ/4)，θ 为有向包角（>0 逆时针）
    半径 r = L / (2·|sin(θ/2)|)
    圆心到弦中点 h = r·cos(θ/2)，沿弦的左法线
    统一公式: center = mid + n·h     （h 带符号即可自动处理 θ>180°）
    """
    sweep = 4.0 * math.atan(bulge)
    ccw = sweep > 0.0
    chord = dist(p1, p2)
    if chord < 1e-12 or abs(math.sin(sweep / 2.0)) < 1e-12:
        return None
    r = chord / (2.0 * abs(math.sin(sweep / 2.0)))
    mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    ux, uy = (p2[0] - p1[0]) / chord, (p2[1] - p1[1]) / chord
    nx, ny = -uy, ux                     # 弦的左法线
    h = r * math.cos(sweep / 2.0)        # 带符号
    return {'center': (mx + nx * h, my + ny * h), 'radius': r, 'ccw': ccw}


def arc_sweep(center, start, end, ccw):
    """由圆心 + 两端点 + 旋向反推包角 (0, 2π]。"""
    a1 = math.atan2(start[1] - center[1], start[0] - center[0])
    a2 = math.atan2(end[1] - center[1], end[0] - center[0])
    sweep = (a2 - a1) % (2 * math.pi) if ccw else (a1 - a2) % (2 * math.pi)
    return sweep if sweep > 1e-12 else 2 * math.pi


def arc_midpoint(center, start, end, ccw):
    """实际被切削的圆弧中点（用于校验旋向是否正确）。"""
    a1 = math.atan2(start[1] - center[1], start[0] - center[0])
    sw = arc_sweep(center, start, end, ccw)
    sw = sw if ccw else -sw
    am = a1 + sw / 2.0
    r = dist(center, start)
    return (center[0] + r * math.cos(am), center[1] + r * math.sin(am))


# =============================================================================
# DXF 提取
# =============================================================================

def extract_entities(path):
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    out = []

    code = doc.header.get('$INSUNITS', 0)
    unit_note = {0: 'unspecified', 1: 'inch', 4: 'mm', 5: 'cm', 6: 'm'}.get(code, f'code{code}')

    for e in msp.query('LINE'):
        p1, p2 = (e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)
        if dist(p1, p2) < 1e-9:
            continue
        out.append({'type': 'line', 'start': p1, 'end': p2})

    for e in msp.query('ARC'):
        c = (e.dxf.center.x, e.dxf.center.y)
        r = e.dxf.radius
        sa, ea = math.radians(e.dxf.start_angle), math.radians(e.dxf.end_angle)
        p1 = (c[0] + r * math.cos(sa), c[1] + r * math.sin(sa))
        p2 = (c[0] + r * math.cos(ea), c[1] + r * math.sin(ea))
        out.append({'type': 'arc', 'start': p1, 'end': p2,
                    'center': c, 'radius': r, 'ccw': True})   # DXF ARC 恒逆时针

    for e in msp.query('LWPOLYLINE'):
        pts = list(e.get_points('xyb'))
        for i in range(len(pts) - 1):
            x1, y1, b = pts[i]
            x2, y2, _ = pts[i + 1]
            p1, p2 = (x1, y1), (x2, y2)
            if dist(p1, p2) < 1e-12:
                continue
            if abs(b) < 1e-12:
                out.append({'type': 'line', 'start': p1, 'end': p2})
            else:
                g = bulge_arc(p1, p2, b)
                if g is None:
                    out.append({'type': 'line', 'start': p1, 'end': p2})
                else:
                    out.append({'type': 'arc', 'start': p1, 'end': p2,
                                'center': g['center'], 'radius': g['radius'],
                                'ccw': g['ccw']})

    for e in msp.query('CIRCLE'):
        c = (e.dxf.center.x, e.dxf.center.y)
        r = e.dxf.radius
        for sa, ea in ((0.0, math.pi), (math.pi, 2 * math.pi)):
            p1 = (c[0] + r * math.cos(sa), c[1] + r * math.sin(sa))
            p2 = (c[0] + r * math.cos(ea), c[1] + r * math.sin(ea))
            # 两段半圆都必须是逆时针（v1 在此处误标 CW）
            out.append({'type': 'arc', 'start': p1, 'end': p2,
                        'center': c, 'radius': r, 'ccw': True})

    return out, unit_note


# =============================================================================
# 变换 / 去重 / 串联
# =============================================================================

def transform(ents, mirror_z=False, x_min=None, z_offset=0.0, x_diameter=False,
              unit_scale=1.0):
    out = []
    for ent in ents:
        e = dict(ent)
        if unit_scale != 1.0:
            e['start'] = (e['start'][0] * unit_scale, e['start'][1] * unit_scale)
            e['end'] = (e['end'][0] * unit_scale, e['end'][1] * unit_scale)
            if e['type'] == 'arc':
                e['center'] = (e['center'][0] * unit_scale, e['center'][1] * unit_scale)
                e['radius'] = e['radius'] * unit_scale
        if mirror_z:
            e['start'] = (-e['start'][0], e['start'][1])
            e['end'] = (-e['end'][0], e['end'][1])
            if e['type'] == 'arc':
                e['center'] = (-e['center'][0], e['center'][1])
                e['ccw'] = not e['ccw']              # ★ 镜像必须反转旋向
        if z_offset:
            e['start'] = (e['start'][0] + z_offset, e['start'][1])
            e['end'] = (e['end'][0] + z_offset, e['end'][1])
            if e['type'] == 'arc':
                e['center'] = (e['center'][0] + z_offset, e['center'][1])
        if x_min is not None:
            if e['start'][1] < x_min - 0.01 or e['end'][1] < x_min - 0.01:
                continue
        if e['type'] == 'line' and abs(e['start'][1]) < 0.01 and abs(e['end'][1]) < 0.01:
            continue                                  # 中心线
        out.append(e)                                 # X 直径换算推迟到输出阶段
    return out


def dedupe(ents):
    """移除重复几何段。

    直线忽略方向判重（反向的同一线段就是同一段）。
    圆弧必须保留方向：整圆拆出的两段半圆端点集合、圆心、半径、旋向全都相同，
    唯一区别就是"哪个端点作为起点"——若把端点排序后再判重会把下半圆误删。
    因此圆弧用有序 (start, end, center, ccw) 作键。
    """
    seen, out = set(), []
    for e in ents:
        r6 = lambda t: tuple(round(v, 6) for v in t)
        if e['type'] == 'arc':
            key = ('arc', r6(e['start']), r6(e['end']), r6(e['center']), bool(e['ccw']))
        else:
            key = ('line',) + tuple(sorted((r6(e['start']), r6(e['end']))))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def flip(ent):
    f = dict(ent)
    f['start'], f['end'] = ent['end'], ent['start']
    if ent['type'] == 'arc':
        f['ccw'] = not ent['ccw']
    return f


def chain(ents, start_at='z0'):
    """最近邻串联。返回 (有序实体, 断点列表)。"""
    if not ents:
        return [], []
    remaining = list(ents)

    def znear(e):
        return min(abs(e['start'][0]), abs(e['end'][0]))

    def zmax(e):
        return max(e['start'][0], e['end'][0])

    if start_at == 'z0':
        idx = min(range(len(remaining)), key=lambda i: znear(remaining[i]))
    elif start_at == 'max_z':
        idx = max(range(len(remaining)), key=lambda i: zmax(remaining[i]))
    else:
        idx = 0

    first = remaining.pop(idx)
    if abs(first['end'][0]) < abs(first['start'][0]):
        first = flip(first)                       # 从靠近端面的一端进刀

    ordered, breaks = [first], []
    cursor = first['end']

    while remaining:
        bi, bd, bf = 0, float('inf'), False
        for i, ent in enumerate(remaining):
            d = dist(cursor, ent['start'])
            if d < bd:
                bi, bd, bf = i, d, False
            d = dist(cursor, ent['end'])
            if d < bd:
                bi, bd, bf = i, d, True
        chosen = remaining.pop(bi)
        if bf:
            chosen = flip(chosen)
        if bd > JOIN_TOL:
            breaks.append((cursor, chosen['start'], bd))
        ordered.append(chosen)
        cursor = chosen['end']

    return ordered, breaks


# =============================================================================
# G 代码生成
# =============================================================================

def generate(ordered, args, safe_x, unit_note, x_diameter):
    q = lambda v: round(v / args.pulse) * args.pulse
    cn = args.cn_comments
    lines = []
    state = {'pos': None}      # 内部统一用半径制 (Z, X)；直径换算只在格式化时做
    xscale = 2.0 if x_diameter else 1.0

    def C(en, zh):
        lines.append(f"({zh if cn else en})")

    def do_move(z, x, code):
        z, x = q(z), q(x)
        pos = state['pos']
        if args.mode == 'g91':
            if pos is None:
                dz, dx = z, x
            else:
                dz, dx = q(z - pos[0]), q(x - pos[1])
            s = f"X{dx * xscale:.3f} Z{dz:.3f}"
        else:
            s = f"X{x * xscale:.3f} Z{z:.3f}"
        state['pos'] = (z, x)
        lines.append(f"{code} {s}")

    def need(z, x):
        p = state['pos']
        return (p is None or abs(p[0] - q(z)) > 1e-9 or abs(p[1] - q(x)) > 1e-9)

    def goto(z, x, code="G00"):
        """仅在确实需要位置变化时才输出，避免大量无位移的空指令。"""
        if need(z, x):
            do_move(z, x, code)

    C("DXF to G-Code Converter v2", "DXF 转 G 代码 v2")
    C(f"Source: {os.path.basename(args.input)}", f"源文件: {os.path.basename(args.input)}")
    C(f"Pulse equivalent: {args.pulse} mm", f"脉冲当量: {args.pulse} mm")
    C(f"DXF units: {unit_note}", f"DXF 单位: {unit_note}")
    if unit_note != 'mm':
        C(f"WARNING: DXF $INSUNITS is '{unit_note}', not mm."
          f" Add --unit-scale (e.g. 25.4 for inch) if geometry is wrong.",
          f"警告: DXF 单位标记为 '{unit_note}' 而非毫米; 若整体尺寸不对请用 --unit-scale")
    C(f"Unit scale applied: {args.unit_scale}", f"单位缩放: {args.unit_scale}")
    C(f"X programming: {'diameter' if x_diameter else 'radius'}",
      f"X 编程: {'直径' if x_diameter else '半径'}")
    C(f"Safe X: {safe_x:.3f}", f"安全退刀 X: {safe_x:.3f}")
    C(f"I/K: {'Fanuc G18 (I=X,K=Z)' if args.ik == 'fanuc' else 'legacy (I=Z,K=X)'}",
      f"I/K 约定: {'Fanuc G18 (I=X,K=Z)' if args.ik == 'fanuc' else '旧约定 (I=Z,K=X)'}")
    if not x_diameter:
        C("WARNING: X emitted as RADIUS; add --x-diameter for diameter programming",
          "警告: X 按半径输出; 若机床为直径编程请加 --x-diameter")
    if args.mode == 'g91':
        C("WARNING: incremental mode assumes machine starts at X0 Z0",
          "警告: 增量模式假定机床起始位置为 X0 Z0")

    lines.append("G21 (mm)")
    lines.append("G18 (ZX plane)")
    if args.header == 'full':
        lines.append("G54")
        if args.tool is not None:
            lines.append(f"T{args.tool:02d} M06")
        if args.spindle:
            lines.append(f"S{int(args.spindle)} M03")
        lines.append("M08")
    lines.append("G90 (absolute)" if args.mode == 'g90' else "G91 (incremental)")
    if args.mode == 'g90':
        lines.append(f"G94 F{args.feed:g}")
    lines.append("")

    # 起点: 先径向退到安全 X
    goto(ordered[0]['start'][0], safe_x)

    for i, ent in enumerate(ordered):
        if i > 0:
            gap = dist(ordered[i - 1]['end'], ent['start'])
            if gap > JOIN_TOL:                    # F8: 断点必须先抬刀再走 Z
                goto(ordered[i - 1]['end'][0], safe_x)
                goto(ent['start'][0], safe_x)
        goto(ent['start'][0], ent['start'][1])

        if ent['type'] == 'line':
            do_move(ent['end'][0], ent['end'][1], "G01")
            continue

        # F4: I/K 由圆弧自身起点 + 量化后圆心计算（内部半径制）
        base = (q(ent['start'][0]), q(ent['start'][1]))
        cz, cx = q(ent['center'][0]), q(ent['center'][1])
        z_off, x_off = cz - base[0], cx - base[1]
        # Fanuc 直径编程: X 字按直径给出, 但 I 仍为半径方向偏移量 → 此处不乘 2
        ik = (f"I{x_off:.3f} K{z_off:.3f}" if args.ik == 'fanuc'
              else f"I{z_off:.3f} K{x_off:.3f}")
        gc = "G03" if ent['ccw'] else "G02"

        ez, ex = q(ent['end'][0]), q(ent['end'][1])
        pos = state['pos']
        if args.mode == 'g91':
            dz, dx = q(ez - pos[0]), q(ex - pos[1])
            s = f"X{dx * xscale:.3f} Z{dz:.3f}"
        else:
            s = f"X{ex * xscale:.3f} Z{ez:.3f}"
        state['pos'] = (ez, ex)
        lines.append(f"{gc} {s} {ik}" + (f" (R{ent['radius']:.3f})" if cn else ""))

    goto(ordered[-1]['end'][0], safe_x)
    if args.header == 'full':
        lines.append("M09")
        lines.append("M05")
    lines.append("M30")
    return "\n".join(lines)


# =============================================================================
# 自校验
# =============================================================================

def validate(ordered, breaks, args, n_raw):
    r = ["=" * 64, "  轮廓自校验报告", "=" * 64]
    n_line = sum(1 for e in ordered if e['type'] == 'line')
    n_arc = len(ordered) - n_line
    r.append(f"  实体: 直线 {n_line} 段 / 圆弧 {n_arc} 段 / 合计 {len(ordered)} 段")
    r.append(f"  去重前实体数: {n_raw}（移除重复几何 {n_raw - len(ordered)} 段）")

    if breaks:
        r.append(f"  [!] 轮廓断点 {len(breaks)} 处:")
        for p, qq, d in breaks[:6]:
            r.append(f"      Z{p[0]:.3f}X{p[1]:.3f} → Z{qq[0]:.3f}X{qq[1]:.3f}  间隙 {d:.3f} mm")
    else:
        r.append("  [OK] 轮廓连续，无断点")

    worst, bad = 0.0, 0
    for e in ordered:
        if e['type'] != 'arc':
            continue
        for p in (e['start'], e['end']):
            resid = abs(dist(e['center'], p) - e['radius'])
            worst = max(worst, resid)
            if resid > max(1e-3, args.pulse):
                bad += 1
    r.append(f"  圆弧圆心-端点半径最大残差: {worst:.6f} mm"
             + ("  [OK]" if bad == 0 else f"  [!] {bad} 处超差"))

    dev = max((abs(v - round(v / args.pulse) * args.pulse)
               for e in ordered for p in (e['start'], e['end']) for v in p), default=0.0)
    r.append(f"  量化前对脉冲网格最大偏差: {dev:.4f} mm (脉冲当量 {args.pulse})")

    total = sum(dist(e['start'], e['end']) if e['type'] == 'line'
                else e['radius'] * arc_sweep(e['center'], e['start'], e['end'], e['ccw'])
                for e in ordered)
    zs = [p[0] for e in ordered for p in (e['start'], e['end'])]
    xs = [p[1] for e in ordered for p in (e['start'], e['end'])]
    r.append(f"  加工路径总长: {total:.2f} mm")
    r.append(f"  程序坐标范围: Z {min(zs):.3f} ~ {max(zs):.3f}   X {min(xs):.3f} ~ {max(xs):.3f}")
    r.append("=" * 64)
    return "\n".join(r)


# =============================================================================
# 主程序
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="DXF → CNC 车削 G 代码转换器 v2")
    p.add_argument("input")
    p.add_argument("--pulse", type=float, default=0.01)
    p.add_argument("--feed", type=float, default=100.0)
    p.add_argument("--mode", choices=["g90", "g91"], default="g90")
    p.add_argument("--output", default=None)
    p.add_argument("--turning", action="store_true", help="车削模式: Z 镜像 + 仅保留 X≥0")
    p.add_argument("--z-flip", action="store_true", help="镜像 Z")
    p.add_argument("--z-offset", type=float, default=0.0)
    p.add_argument("--x-min", type=float, default=None)
    p.add_argument("--x-diameter", action="store_true", help="X 按直径输出")
    p.add_argument("--safe-x", type=float, default=None, help="安全退刀 X (默认 = 最大X + 2)")
    p.add_argument("--ik", choices=["fanuc", "legacy"], default="fanuc")
    p.add_argument("--header", choices=["basic", "full"], default="basic")
    p.add_argument("--tool", type=int, default=None)
    p.add_argument("--spindle", type=float, default=None)
    p.add_argument("--start", choices=["z0", "max_z", "first"], default="z0")
    p.add_argument("--no-sort", action="store_true")
    p.add_argument("--cn-comments", action="store_true")
    p.add_argument("--unit-scale", type=float, default=1.0,
                   help="坐标缩放（英寸图纸用 25.4）")
    p.add_argument("--check", action="store_true")
    p.add_argument("--report", default=None)
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}")
        sys.exit(1)

    ents, unit_note = extract_entities(args.input)
    if not ents:
        print("错误: 未找到 LINE/ARC/LWPOLYLINE/CIRCLE 实体")
        sys.exit(1)

    mirrored = args.z_flip or args.turning
    x_min = args.x_min if args.x_min is not None else (0.0 if args.turning else None)
    ents = transform(ents, mirrored, x_min, args.z_offset, args.x_diameter,
                     args.unit_scale)
    if not ents:
        print("错误: 过滤后无实体")
        sys.exit(1)

    n_raw = len(ents)
    ents = dedupe(ents)
    ordered, breaks = (ents, []) if args.no_sort else chain(ents, args.start)

    xs = [p[1] for e in ordered for p in (e['start'], e['end'])]
    safe_x = args.safe_x if args.safe_x is not None else max(xs) + 2.0

    gcode = generate(ordered, args, safe_x, unit_note, args.x_diameter)
    out = args.output or (os.path.splitext(os.path.basename(args.input))[0] + "_v2.nc")
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(gcode + "\n")

    n_move = sum(1 for l in gcode.splitlines() if l[:3] in ("G00", "G01", "G02", "G03"))
    print(f"已生成: {out}")
    print(f"运动指令 {n_move} 条 | 实体 {len(ordered)} 段(去重前 {n_raw}) | 安全退刀 X={safe_x:.3f}")

    if args.check or args.report:
        rep = validate(ordered, breaks, args, n_raw)
        print("\n" + rep)
        if args.report:
            with open(args.report, 'w', encoding='utf-8') as f:
                f.write(rep + "\n")


if __name__ == "__main__":
    main()
