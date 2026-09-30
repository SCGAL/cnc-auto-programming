"""
DXF → CNC车削G代码转换器
==========================
读取DXF文件中的线段和圆弧特征，自动排序并输出加工G代码。
坐标映射: DXF的X→Z轴(轴向), Y→X轴(径向/半径值)

用法: python dxf_to_gcode.py <dxf文件路径> [选项]
选项:
  --pulse N      脉冲当量(mm), 默认0.01
  --offset-z N   Z轴偏移(mm), 默认0
  --offset-x N   X轴偏移(mm), 默认0
  --mode g90|g91 坐标模式, 默认g90
  --feed N       进给速度(mm/min), 默认100
  --rapid N      快速定位速度, 默认2000
  --safe-z N     安全Z高度, 默认2.0
  --output NAME  输出文件名, 默认根据输入自动生成
  --no-sort      不自动排序(保留DXF原始顺序)

示例: python dxf_to_gcode.py part.dxf --pulse 0.01 --mode g90
"""

import math
import sys
import os
import argparse

try:
    import ezdxf
except ImportError:
    print("请先安装 ezdxf: pip install ezdxf")
    sys.exit(1)


# =============================================================================
# DXF 实体提取
# =============================================================================

def extract_entities(dxf_path):
    """从DXF文件提取所有线段和圆弧实体,返回列表.

    每个实体是一个字典:
      LINE:   {'type':'line', 'start':(z,x), 'end':(z,x)}
      ARC:    {'type':'arc', 'start':(z,x), 'end':(z,x), 'center':(cz,cx),
               'radius':r, 'cw':True/False}
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    entities = []

    # 提取LINE
    for e in msp.query('LINE'):
        sx, sy = e.dxf.start.x, e.dxf.start.y
        ex, ey = e.dxf.end.x, e.dxf.end.y
        # 跳过零长度线段
        if abs(ex - sx) < 1e-9 and abs(ey - sy) < 1e-9:
            continue
        entities.append({
            'type': 'line',
            'start': (sx, sy),  # (Z, X)
            'end': (ex, ey),
        })

    # 提取ARC
    for e in msp.query('ARC'):
        cx, cy = e.dxf.center.x, e.dxf.center.y
        r = e.dxf.radius
        sa = math.radians(e.dxf.start_angle)
        ea = math.radians(e.dxf.end_angle)
        sx = cx + r * math.cos(sa)
        sy = cy + r * math.sin(sa)
        ex = cx + r * math.cos(ea)
        ey = cy + r * math.sin(ea)
        # DXF的ARC永远逆时针(CCW)绘制 → cw=False → 输出G03
        # 若排序时实体被翻转(start/end交换), cw会自动取反 → 输出G02
        cw = False
        entities.append({
            'type': 'arc',
            'start': (sx, sy),
            'end': (ex, ey),
            'center': (cx, cy),
            'radius': r,
            'cw': cw,
        })

    # 提取LWPOLYLINE (拆分为线段和圆弧段)
    for e in msp.query('LWPOLYLINE'):
        pts = list(e.get_points('xyb'))
        if len(pts) < 2:
            continue
        bulge = e.get_points('xyb')  # 需要重新获取
        pts_xyb = list(e.get_points('xyb'))

        i = 0
        while i < len(pts_xyb) - 1:
            x1, y1, b = pts_xyb[i]
            x2, y2, _ = pts_xyb[i + 1]

            if abs(b) < 1e-9:
                # 直线段
                if abs(x2 - x1) > 1e-9 or abs(y2 - y1) > 1e-9:
                    entities.append({
                        'type': 'line',
                        'start': (x1, y1),
                        'end': (x2, y2),
                    })
            else:
                # 圆弧段 (bulge = tan(sweep_angle/4))
                chord_len = math.hypot(x2 - x1, y2 - y1)
                if chord_len < 1e-9:
                    i += 1
                    continue
                sweep = 4 * math.atan(b)
                r = chord_len / (2 * abs(math.sin(sweep / 2)))
                # 弦中点
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                # 弦方向单位向量
                dx, dy = x2 - x1, y2 - y1
                cl = math.hypot(dx, dy)
                ux, uy = dx / cl, dy / cl
                # 法向量 (逆时针旋转90度)
                nx, ny = -uy, ux
                # 圆心偏移距离
                h = r * math.cos(sweep / 2)
                if b > 0:
                    cx, cy = mx - nx * h, my - ny * h
                else:
                    cx, cy = mx + nx * h, my + ny * h
                cw = b < 0  # bulge>0 = CCW, bulge<0 = CW
                entities.append({
                    'type': 'arc',
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'center': (cx, cy),
                    'radius': abs(r),
                    'cw': cw,
                })
            i += 1

    # 提取CIRCLE (完整圆拆分为两个半圆)
    for e in msp.query('CIRCLE'):
        cx, cy = e.dxf.center.x, e.dxf.center.y
        r = e.dxf.radius
        # 拆为两个半圆弧
        for half in [(0, math.pi, True), (math.pi, 2 * math.pi, True)]:
            sa, ea, cw = half
            sx = cx + r * math.cos(sa)
            sy = cy + r * math.sin(sa)
            ex = cx + r * math.cos(ea)
            ey = cy + r * math.sin(ea)
            entities.append({
                'type': 'arc',
                'start': (sx, sy),
                'end': (ex, ey),
                'center': (cx, cy),
                'radius': r,
                'cw': cw,
            })

    return entities


# =============================================================================
# 实体过滤 — 车削外轮廓提取
# =============================================================================

def clip_entity_at_x(ent, which, x_min):
    """裁剪线段到X=x_min处,保留X≥x_min的半段.
    which='start' 或 'end' 标记哪一端在x_min以下需要裁剪.
    坐标系: (Z, X) = (entity[0], entity[1])"""
    if ent['type'] != 'line':
        return None  # 圆弧暂不裁剪
    z1, x1 = ent['start']
    z2, x2 = ent['end']
    if abs(x2 - x1) < 1e-9:
        return None  # 水平线两端X相同却穿过x_min → 不可能,检查容差
    t = (x_min - x1) / (x2 - x1)
    if t < 0 or t > 1:
        return None
    z_new = z1 + t * (z2 - z1)
    if which == 'start':
        ent['start'] = (z_new, x_min)
    else:
        ent['end'] = (z_new, x_min)
    if distance(ent['start'], ent['end']) < 0.001:
        return None
    return ent


def filter_profile(entities, z_flip=False, x_min=None):
    """过滤实体，只保留外轮廓。

    z_flip: 翻转Z方向 (DXF中Z负方向→G代码Z正方向)
    x_min:  最小X值过滤 (车削时只保留X≥0的上半部轮廓)
    同时过滤水平/垂直的短辅助线 (可能是中心线/标注线)。
    """
    filtered = []
    for ent in entities:
        e = dict(ent)

        # 翻转Z
        if z_flip:
            e['start'] = (-e['start'][0], e['start'][1])
            e['end'] = (-e['end'][0], e['end'][1])
            if e['type'] == 'arc':
                e['center'] = (-e['center'][0], e['center'][1])

        # X_min 过滤: 严格模式 — 两端都在x_min以上才保留
        if x_min is not None:
            x1, x2 = e['start'][1], e['end'][1]
            if x1 < x_min - 0.01 or x2 < x_min - 0.01:
                continue

        # 过滤中心线/辅助线: 水平线Y≈0 或 很短的线
        if e['type'] == 'line':
            dx = e['end'][0] - e['start'][0]
            dy = e['end'][1] - e['start'][1]
            length = math.hypot(dx, dy)
            # X≈0的水平线 = 中心线, 跳过
            if abs(e['start'][1]) < 0.01 and abs(e['end'][1]) < 0.01:
                continue

        filtered.append(e)

    # 添加合成垂直线段: 同一Z位置存在多个端点X值时,连接相邻X值
    if x_min is not None and filtered:
        from collections import defaultdict
        eps_by_z = defaultdict(set)
        for ent in filtered:
            eps_by_z[round(ent['start'][0], 3)].add(round(ent['start'][1], 3))
            eps_by_z[round(ent['end'][0], 3)].add(round(ent['end'][1], 3))
        for z, xs in eps_by_z.items():
            x_sorted = sorted(xs)
            for i in range(len(x_sorted) - 1):
                x_a, x_b = x_sorted[i], x_sorted[i + 1]
                if x_b - x_a > 0.01:
                    filtered.append({
                        'type': 'line',
                        'start': (z, x_a),
                        'end': (z, x_b),
                    })

    return filtered

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def sort_entities(entities, start_from_right=False):
    """将实体按端点最近邻贪心排序,形成连续路径.
    start_from_right: 从最大Z值端点开始 (车削外轮廓从右端向左端加工)."""
    if len(entities) <= 1:
        return entities

    remaining = list(entities)

    # 选择起始实体: 从右端点(最大Z)开始
    if start_from_right:
        best_start = 0
        best_z = -float('inf')
        best_is_end = False
        for i, ent in enumerate(remaining):
            if ent['start'][0] > best_z:
                best_z = ent['start'][0]
                best_start = i
                best_is_end = False
            if ent['end'][0] > best_z:
                best_z = ent['end'][0]
                best_start = i
                best_is_end = True
        # 将最佳起始实体放到第一位
        remaining[0], remaining[best_start] = remaining[best_start], remaining[0]
        # 若最右端点在实体的end,翻转使起点为最右端
        if best_is_end:
            remaining[0] = flip_entity(remaining[0])

    sorted_ents = [remaining.pop(0)]
    cursor = sorted_ents[0]['end']

    while remaining:
        best_idx = 0
        best_dist = float('inf')
        best_flipped = False

        for i, ent in enumerate(remaining):
            d = distance(cursor, ent['start'])
            if d < best_dist:
                best_dist, best_idx, best_flipped = d, i, False
            d = distance(cursor, ent['end'])
            if d < best_dist:
                best_dist, best_idx, best_flipped = d, i, True

        chosen = remaining.pop(best_idx)
        if best_flipped:
            chosen = flip_entity(chosen)

        # 如果间隙 > 5mm, 插入快速定位段
        if best_dist > 5.0:
            sorted_ents.append({
                'type': 'rapid',
                'start': cursor,
                'end': chosen['start'],
            })

        sorted_ents.append(chosen)
        cursor = chosen['end']

    return sorted_ents


def flip_entity(ent):
    """反转实体方向 (交换起点/终点, 圆弧反转CW/CCW)."""
    flipped = dict(ent)
    flipped['start'], flipped['end'] = ent['end'], ent['start']
    if ent['type'] == 'arc':
        flipped['cw'] = not ent.get('cw', False)
    return flipped


# =============================================================================
# G代码生成
# =============================================================================

def format_coord(z, x, pulse):
    """将毫米坐标转为脉冲数(整数), 返回格式化的坐标字符串."""
    z_steps = int(round(z / pulse))
    x_steps = int(round(x / pulse))
    # 输出仍用毫米值, 保留合理精度
    return f"Z{z:.3f} X{x:.3f}"


def generate_gcode(entities, args):
    """根据实体列表生成G代码字符串."""
    pulse = args.pulse
    abs_mode = args.mode == 'g90'

    lines = []
    lines.append(f"(DXF to G-Code Converter)")
    lines.append(f"(Source: {os.path.basename(args.input)})")
    lines.append(f"(Pulse Equivalent: {pulse} mm)")
    lines.append(f"G21 (mm mode)")
    lines.append(f"G90 (absolute coordinates)" if abs_mode else f"G91 (incremental)")
    lines.append(f"F{args.feed}")
    lines.append("")

    current_z, current_x = args.offset_z, args.offset_x

    for i, ent in enumerate(entities):
        sz = ent['start'][0] + args.offset_z
        sx = ent['start'][1] + args.offset_x
        ez = ent['end'][0] + args.offset_z
        ex = ent['end'][1] + args.offset_x

        # 如果是第一个实体或快速定位, 先G00到起点
        if i == 0 or ent['type'] == 'rapid':
            if ent['type'] == 'rapid':
                sz_r = ent['start'][0] + args.offset_z
                sx_r = ent['start'][1] + args.offset_x
                ez_r = ent['end'][0] + args.offset_z
                ex_r = ent['end'][1] + args.offset_x
                lines.append(f"G00 Z{ez_r:.3f} X{ex_r:.3f} (快速移动到下一段起点)")
                current_z, current_x = ez_r, ex_r
                continue
            else:
                lines.append(f"G00 {format_coord(sz, sx, pulse)} (快速定位到起点)")
                current_z, current_x = sz, sx

        # 直线插补
        if ent['type'] == 'line':
            lines.append(f"G01 {format_coord(ez, ex, pulse)}")
            current_z, current_x = ez, ex

        # 圆弧插补
        elif ent['type'] == 'arc':
            cz = ent['center'][0] + args.offset_z
            cx_center = ent['center'][1] + args.offset_x
            i = cz - current_z  # I: 圆心Z增量
            k = cx_center - current_x  # K: 圆心X增量
            gc = "G02" if ent['cw'] else "G03"
            lines.append(
                f"{gc} {format_coord(ez, ex, pulse)} "
                f"I{i:.3f} K{k:.3f} "
                f"(R{ent['radius']:.3f})"
            )
            current_z, current_x = ez, ex

    lines.append("")
    lines.append("M30 (程序结束)")
    return "\n".join(lines)


# =============================================================================
# 统计信息
# =============================================================================

def print_summary(entities, sorted_ents):
    """打印实体统计信息."""
    n_line = sum(1 for e in entities if e['type'] == 'line')
    n_arc = sum(1 for e in entities if e['type'] == 'arc')
    total_len = 0.0
    for e in entities:
        if e['type'] == 'line':
            total_len += distance(e['start'], e['end'])
        elif e['type'] == 'arc':
            r = e['radius']
            sx, sy = e['start']
            ex, ey = e['end']
            # 计算弧长
            chord = distance(e['start'], e['end'])
            if r > 0 and chord / (2 * r) <= 1:
                angle = 2 * math.asin(chord / (2 * r))
                total_len += r * angle

    print(f"实体统计:")
    print(f"  直线段: {n_line} 段")
    print(f"  圆弧段: {n_arc} 段")
    print(f"  总实体数: {len(entities)} 段")
    print(f"  加工路径总长: {total_len:.1f} mm")
    rapid_count = sum(1 for e in sorted_ents if e['type'] == 'rapid')
    if rapid_count:
        print(f"  快速定位桥接: {rapid_count} 次")


# =============================================================================
# 主程序
# =============================================================================

def gui_select_and_run():
    """无命令行参数时弹出文件选择窗口并设置参数."""
    import tkinter as tk
    from tkinter import filedialog, ttk

    root = tk.Tk()
    root.title("DXF → G代码转换器")
    root.geometry("520x440")
    root.resizable(False, False)

    # 样式
    bg = "#F5F6F8"
    fg = "#1D3259"
    root.configure(bg=bg)

    result = {"file": None, "confirmed": False}

    # 标题
    tk.Label(root, text="DXF → CNC车削G代码转换器",
             font=("Microsoft YaHei", 14, "bold"), fg=fg, bg=bg).pack(pady=(20, 5))
    tk.Label(root, text="选择DXF文件并设置加工参数",
             font=("Microsoft YaHei", 10), fg="#888888", bg=bg).pack(pady=(0, 15))

    # 文件选择行
    file_frame = tk.Frame(root, bg=bg)
    file_frame.pack(fill="x", padx=30, pady=(0, 15))
    tk.Label(file_frame, text="DXF文件:", font=("Microsoft YaHei", 11),
             fg="#333333", bg=bg).pack(side="left")
    file_var = tk.StringVar()
    file_entry = tk.Entry(file_frame, textvariable=file_var, font=("Consolas", 10),
                          width=36, state="readonly", readonlybackground="#FFFFFF")
    file_entry.pack(side="left", padx=(8, 0))

    def choose_file():
        path = filedialog.askopenfilename(
            title="选择DXF文件",
            filetypes=[("DXF文件", "*.dxf"), ("所有文件", "*.*")],
        )
        if path:
            file_var.set(path)
            # 自动设置输出文件名
            base = os.path.splitext(os.path.basename(path))[0]
            out_var.set(f"{base}_gcode.nc")

    tk.Button(file_frame, text="浏览...", command=choose_file,
              font=("Microsoft YaHei", 10), bg="#1D3259", fg="#FFFFFF",
              relief="flat", padx=12, cursor="hand2").pack(side="left", padx=(6, 0))

    # 参数设置
    param_frame = tk.LabelFrame(root, text="加工参数", font=("Microsoft YaHei", 11, "bold"),
                                fg=fg, bg=bg, padx=15, pady=10)
    param_frame.pack(fill="x", padx=30, pady=(0, 10))

    def make_row(parent, label, default, row):
        tk.Label(parent, text=label, font=("Microsoft YaHei", 10),
                 fg="#555555", bg=bg).grid(row=row, column=0, sticky="e", padx=(0, 8), pady=4)
        var = tk.StringVar(value=str(default))
        entry = tk.Entry(parent, textvariable=var, font=("Consolas", 10), width=10,
                         justify="center")
        entry.grid(row=row, column=1, sticky="w", pady=4)
        return var

    pulse_var = make_row(param_frame, "脉冲当量(mm):", "0.01", 0)
    feed_var = make_row(param_frame, "进给速度(mm/min):", "100", 1)
    oz_var = make_row(param_frame, "Z轴偏移(mm):", "0", 2)
    ox_var = make_row(param_frame, "X轴偏移(mm):", "0", 3)

    # 坐标模式
    tk.Label(param_frame, text="坐标模式:", font=("Microsoft YaHei", 10),
             fg="#555555", bg=bg).grid(row=0, column=2, sticky="e", padx=(30, 8), pady=4)
    mode_var = tk.StringVar(value="G90")
    mode_combo = ttk.Combobox(param_frame, textvariable=mode_var, values=["G90", "G91"],
                              state="readonly", width=8, font=("Consolas", 10))
    mode_combo.grid(row=0, column=3, sticky="w", pady=4)

    # 排序选项
    sort_var = tk.BooleanVar(value=True)
    tk.Checkbutton(param_frame, text="自动排序(推荐)", variable=sort_var,
                   font=("Microsoft YaHei", 10), fg="#555555", bg=bg,
                   selectcolor=bg).grid(row=1, column=2, columnspan=2, sticky="w", padx=(30, 0))

    # 车削模式
    turning_var = tk.BooleanVar(value=True)
    tk.Checkbutton(param_frame, text="车削模式(外轮廓)", variable=turning_var,
                   font=("Microsoft YaHei", 10), fg="#555555", bg=bg,
                   selectcolor=bg).grid(row=2, column=2, columnspan=2, sticky="w", padx=(30, 0))

    # 输出文件
    out_frame = tk.Frame(root, bg=bg)
    out_frame.pack(fill="x", padx=30, pady=(5, 15))
    tk.Label(out_frame, text="输出文件:", font=("Microsoft YaHei", 11),
             fg="#333333", bg=bg).pack(side="left")
    out_var = tk.StringVar(value="output_gcode.nc")
    tk.Entry(out_frame, textvariable=out_var, font=("Consolas", 10),
             width=42).pack(side="left", padx=(8, 0))

    # 按钮
    btn_frame = tk.Frame(root, bg=bg)
    btn_frame.pack(pady=(0, 20))

    def on_go():
        if not file_var.get():
            from tkinter import messagebox
            messagebox.showwarning("提示", "请先选择DXF文件")
            return
        result["file"] = file_var.get()
        result["confirmed"] = True
        root.destroy()

    tk.Button(btn_frame, text="开始转换", command=on_go,
              font=("Microsoft YaHei", 12, "bold"), bg="#1D3259", fg="#FFFFFF",
              relief="flat", padx=40, pady=6, cursor="hand2").pack(side="left", padx=(0, 15))
    tk.Button(btn_frame, text="取消", command=root.destroy,
              font=("Microsoft YaHei", 12), bg="#CCCCCC", fg="#333333",
              relief="flat", padx=40, pady=6, cursor="hand2").pack(side="left")

    root.mainloop()

    if not result["confirmed"]:
        sys.exit(0)

    # 构建参数对象
    class Args:
        pass
    args = Args()
    args.input = result["file"]
    args.pulse = float(pulse_var.get())
    args.feed = float(feed_var.get())
    args.offset_z = float(oz_var.get())
    args.offset_x = float(ox_var.get())
    args.mode = mode_var.get().lower()
    args.rapid = 2000.0
    args.output = out_var.get() if out_var.get() else None
    args.no_sort = not sort_var.get()
    args.turning = turning_var.get()
    if args.turning:
        args.z_flip = True
        args.x_min = 0.0
    else:
        args.z_flip = False
        args.x_min = None
    return args


def main():
    # 如果有命令行参数, 走原来的命令行模式
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="DXF → CNC车削G代码转换器",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  python dxf_to_gcode.py part.dxf
  python dxf_to_gcode.py part.dxf --pulse 0.01 --mode g90 --offset-z 2.0
  python dxf_to_gcode.py part.dxf --no-sort --output my_gcode.nc
            """,
        )
        parser.add_argument("input", help="DXF文件路径")
        parser.add_argument("--pulse", type=float, default=0.01,
                            help="脉冲当量(mm), 默认0.01")
        parser.add_argument("--offset-z", type=float, default=0.0,
                            help="Z轴偏移(mm), 默认0")
        parser.add_argument("--offset-x", type=float, default=0.0,
                            help="X轴偏移(mm), 默认0")
        parser.add_argument("--mode", choices=["g90", "g91"], default="g90",
                            help="坐标模式, 默认g90")
        parser.add_argument("--feed", type=float, default=100.0,
                            help="进给速度(mm/min), 默认100")
        parser.add_argument("--rapid", type=float, default=2000.0,
                            help="快速定位速度, 默认2000")
        parser.add_argument("--output", type=str, default=None,
                            help="输出文件名, 默认基于输入文件名生成")
        parser.add_argument("--no-sort", action="store_true",
                            help="不自动排序(保留DXF内部顺序)")
        parser.add_argument("--turning", action="store_true",
                            help="车削模式: 自动翻转Z+过滤X≥0外轮廓")
        parser.add_argument("--z-flip", action="store_true",
                            help="翻转Z轴方向")
        parser.add_argument("--x-min", type=float, default=None,
                            help="最小X值过滤(如 --x-min 0 只保留上半部)")
        args = parser.parse_args()
        # turning模式 = z-flip + x-min=0
        if args.turning:
            if not args.z_flip:
                args.z_flip = True
            if args.x_min is None:
                args.x_min = 0.0
    else:
        # 无参数: 弹出GUI窗口选择文件
        args = gui_select_and_run()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}")
        sys.exit(1)

    print(f"读取DXF文件: {args.input}")
    entities = extract_entities(args.input)

    # 应用过滤 (车削模式 / Z翻转 / X过滤)
    if args.z_flip or args.x_min is not None:
        entities = filter_profile(entities, z_flip=args.z_flip, x_min=args.x_min)
        print(f"(已过滤: Z翻转={'是' if args.z_flip else '否'}, X≥{args.x_min if args.x_min is not None else '无限制'})")

    if not entities:
        print("错误: DXF文件中未找到线段/圆弧/多段线实体.")
        sys.exit(1)

    # 排序形成连续路径
    if args.no_sort:
        sorted_ents = entities
        print("(跳过自动排序, 保留DXF原始顺序)")
    else:
        start_right = (args.x_min is not None)  # 车削外轮廓从右端开始
        sorted_ents = sort_entities(entities, start_from_right=start_right)
        print("(已按最近邻贪心排序为连续加工路径)")

    print_summary(entities, sorted_ents)

    # 生成G代码
    gcode = generate_gcode(sorted_ents, args)

    # 输出文件
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = f"{base}_gcode.nc"

    # 确保输出到桌面或当前目录
    if not os.path.dirname(out_path):
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.input)) or ".", out_path)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(gcode)

    print(f"\nG代码已生成: {out_path}")
    n_cmds = len([l for l in gcode.split(chr(10)) if l.startswith(('G0', 'G1', 'G2', 'G3'))])
    print(f"共 {n_cmds} 条运动指令")


if __name__ == "__main__":
    main()
