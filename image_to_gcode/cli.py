#!/usr/bin/env python3
"""图像识别版数控自动编程工具 —— 命令行入口。

    python cli.py part.png --ref-diameter 40 --turning -o out.nc
    python cli.py part.png --mm-per-px 0.1 --turning -o out.nc --check
    python cli.py part.png --ref-diameter 40 --turning --viz out/viz --report out/report.txt

退出码
------
0  成功
2  业务错误（缺标定 / 实体数为 0 / 轮廓不闭合 / 图像无法读取 …）—— 明确报错，不静默输出
1  意外异常（含完整 traceback）

**边界声明**：本工具只做「车削上半部外轮廓 → 精加工轮廓路径」。
不做粗车循环 G71/G70、螺纹、切槽、钻孔、多刀具、刀补、内孔、多视图、
GD&T 识别、透视校正、刀具干涉仿真。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from i2g import __version__, emit, pipeline, vectorize       # noqa: E402
from i2g.errors import I2GError                              # noqa: E402

SCOPE_NOTE = ('本版仅支持「车削上半部外轮廓 + 精加工轮廓路径」。'
              '粗车循环/螺纹/切槽/钻孔/多刀具/刀补/GD&T/透视校正均不在范围内。')


def _point(s):
    try:
        a, b = s.split(',')
        return (float(a), float(b))
    except Exception:                                        # noqa: BLE001
        raise argparse.ArgumentTypeError(f'点格式应为 "z,x"，收到 {s!r}')


def build_parser():
    p = argparse.ArgumentParser(
        prog='cli.py',
        description='图像识别版数控自动编程工具：零件轮廓图 → 车削 G 代码',
        epilog=SCOPE_NOTE,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('input', help='输入零件轮廓图（PNG/JPG/BMP）')
    p.add_argument('-o', '--output', default=None,
                   help='输出 .nc 路径（缺省 = 输入同名 + .nc）')
    p.add_argument('--turning', action='store_true',
                   help='车削模式（本版必须显式给出；用于确认你接受"只做车削上半部外轮廓"这条边界）')

    g = p.add_argument_group('标定（三选一，必须给；否则报错退出）')
    g.add_argument('--mm-per-px', type=float, default=None, help='显式比例：毫米/像素')
    g.add_argument('--ref-diameter', type=float, default=None,
                   help='已知外径 mm（推荐）：取轮廓最大高度 = 半径')
    g.add_argument('--ref-points', type=_point, nargs=2, default=None,
                   metavar=('Z1,X1', 'Z2,X2'),
                   help='两个参考特征点（像素坐标，空格分隔）')
    g.add_argument('--ref-mm', type=float, default=None, help='两参考点实际距离 mm')
    g.add_argument('--z-direction', type=float, choices=(1.0, -1.0), default=1.0,
                   help='Z 轴方向：图像列增大方向为 +Z 时取 1')

    g = p.add_argument_group('输入约定与预处理')
    g.add_argument('--ink', choices=('line', 'fill'), default='line',
                   help="line=线框笔画图（取线条中线）；fill=实心填充轮廓图")
    g.add_argument('--denoise', choices=('isolated', 'median', 'none'),
                   default='isolated',
                   help='isolated=只去孤立点（默认，对 1 px 笔画安全）；'
                        'median=3x3 中值（会摧毁 1 px 笔画，仅作对照）')
    g.add_argument('--min-area', type=int, default=40, help='连通域最小面积（像素）')
    g.add_argument('--close-radius', type=int, default=1, help='形态学闭运算半径（0=关闭）')
    g.add_argument('--threshold', type=int, default=None, help='二值化阈值（缺省 Otsu）')
    g.add_argument('--polarity', choices=('auto', 'dark', 'light'), default='auto',
                   help='墨迹的明暗极性')

    g = p.add_argument_group('倾斜校正')
    g.add_argument('--no-deskew', action='store_true', help='关闭倾斜校正')
    g.add_argument('--deskew-max-deg', type=float, default=3.0, help='倾斜角搜索范围')

    g = p.add_argument_group('矢量化')
    g.add_argument('--dp-eps', type=float,
                   default=vectorize.VectorizeParams.dp_eps,
                   help='Douglas-Peucker 容差（像素）')
    g.add_argument('--arc-sse-ratio', type=float,
                   default=vectorize.VectorizeParams.arc_sse_ratio,
                   help='SSE_circle < ratio·SSE_line 才判圆弧')
    g.add_argument('--arc-chord-ratio', type=float,
                   default=vectorize.VectorizeParams.min_arc_chord_ratio,
                   help='弧长/弦长下限（排除近似直线的微弧）')
    g.add_argument('--min-arc-radius', type=float,
                   default=vectorize.VectorizeParams.min_arc_radius,
                   help='圆弧半径下限（像素），防噪声过拟合')
    g.add_argument('--max-sweep-deg', type=float,
                   default=vectorize.VectorizeParams.max_sweep_deg,
                   help='单段圆弧包角上限')
    g.add_argument('--smooth-window', type=int,
                   default=vectorize.VectorizeParams.smooth_window,
                   help='拟合前滑动平均窗口（0=不平滑，见 vectorize docstring）')
    g.add_argument('--ransac-thresh', type=float,
                   default=vectorize.VectorizeParams.ransac_thresh,
                   help='RANSAC 内点阈值（像素）')
    g.add_argument('--ransac-iters', type=int,
                   default=vectorize.VectorizeParams.ransac_iters)

    g = p.add_argument_group('几何吸附')
    g.add_argument('--angle-tol', type=float, default=3.0, help='角度吸附容差（度）')
    g.add_argument('--collinear-tol', type=float, default=0.15,
                   help='共线/共弧合并容差（mm）')
    g.add_argument('--close-tol', type=float, default=0.5,
                   help='轮廓连接间隙容差（mm），超过即报「轮廓不闭合」')

    g = p.add_argument_group('G 代码输出（透传给 dxf_to_gcode_v2.generate）')
    g.add_argument('--pulse', type=float, default=0.01, help='脉冲当量 mm')
    g.add_argument('--feed', type=float, default=100.0)
    g.add_argument('--mode', choices=('g90', 'g91'), default='g90')
    g.add_argument('--ik', choices=('fanuc', 'legacy'), default='fanuc',
                   help='圆弧 I/K 轴对应：fanuc = I→X、K→Z')
    g.add_argument('--x-diameter', action='store_true', help='X 按直径输出')
    g.add_argument('--header', choices=('basic', 'full'), default='basic')
    g.add_argument('--tool', type=int, default=None)
    g.add_argument('--spindle', type=float, default=None)
    g.add_argument('--safe-x', type=float, default=None, help='安全退刀 X（缺省 = 最大X + 2）')
    g.add_argument('--no-cn-comments', action='store_true', help='注释用英文')

    g = p.add_argument_group('报告与调试')
    g.add_argument('--check', action='store_true', help='打印 G 代码自校验报告')
    g.add_argument('--report', default=None, help='把自校验报告写入文件')
    g.add_argument('--viz', default=None, metavar='DIR', help='导出中间过程图片')
    g.add_argument('--dump-json', default=None, metavar='FILE',
                   help='把实体/标定/统计导出为 JSON（供精度报告使用）')
    g.add_argument('--quiet', action='store_true')
    g.add_argument('--version', action='version', version=f'image_to_gcode {__version__}')
    return p


def build_params(a):
    vp = vectorize.VectorizeParams(
        dp_eps=a.dp_eps, arc_sse_ratio=a.arc_sse_ratio,
        min_arc_chord_ratio=a.arc_chord_ratio, min_arc_radius=a.min_arc_radius,
        max_sweep_deg=a.max_sweep_deg, smooth_window=a.smooth_window,
        ransac_thresh=a.ransac_thresh, ransac_iters=a.ransac_iters)
    return pipeline.Params(
        denoise=a.denoise, min_area=a.min_area, close_radius=a.close_radius,
        threshold=a.threshold, polarity=a.polarity,
        ink=a.ink, deskew=not a.no_deskew, deskew_max_deg=a.deskew_max_deg,
        vector=vp,
        mm_per_px=a.mm_per_px, ref_diameter=a.ref_diameter,
        ref_points=[list(p) for p in a.ref_points] if a.ref_points else None,
        ref_mm=a.ref_mm, z_sign=a.z_direction,
        angle_tol_deg=a.angle_tol, collinear_tol_mm=a.collinear_tol,
        close_tol_mm=a.close_tol,
        gcode=dict(pulse=a.pulse, feed=a.feed, mode=a.mode, ik=a.ik,
                   header=a.header, x_diameter=a.x_diameter,
                   cn_comments=not a.no_cn_comments, tool=a.tool,
                   spindle=a.spindle, safe_x=a.safe_x),
        viz_dir=a.viz,
        viz_prefix=os.path.splitext(os.path.basename(a.input))[0],
        source_name=os.path.basename(a.input))


def _jsonable(o):
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    try:
        import numpy as np
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:                                        # noqa: BLE001
        pass
    return str(o)


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)

    if not a.turning:
        print('错误: 本版仅支持车削上半部外轮廓，必须显式给出 --turning。\n'
              f'{SCOPE_NOTE}', file=sys.stderr)
        return 2
    if not os.path.exists(a.input):
        print(f'错误: 输入文件不存在: {a.input}', file=sys.stderr)
        return 2

    try:
        res = pipeline.run_from_image(a.input, build_params(a))
    except I2GError as e:
        print(f'错误: {e}', file=sys.stderr)
        return getattr(e, 'exit_code', 2)
    except Exception:                                        # noqa: BLE001
        print('未预期异常：', file=sys.stderr)
        traceback.print_exc()
        return 1

    out = a.output or (os.path.splitext(a.input)[0] + '.nc')
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(res['gcode'] + '\n')

    if not a.quiet:
        st = res['stats']
        cal = st['calibration']
        pi = st['preprocess']
        print(f'已生成: {out}')
        print(f'  图像      : {pi["image_size"][0]}×{pi["image_size"][1]} px，'
              f'Otsu={pi["threshold"]}，墨迹 {pi["ink_px_raw"]} px '
              f'(连通域 {pi["n_components_raw"]}，剔除 {pi["dropped_area_px"]} px)')
        print(f'  倾斜校正  : {st["centerline"]["theta_deg"]:+.3f}° '
              f'(可信度 {st["centerline"]["contrast"]:.3f})')
        print(f'  轮廓点列  : {st["profile"]["n_profile_pts"]} 点，'
              f'轴线行 {st["profile"]["axis_row"]:.2f}')
        print(f'  矢量化    : {st["vectorize"]["n_segments_before_merge"]} 段 → '
              f'{st["vectorize"]["n_segments_after_merge"]} 段 '
              f'(圆弧 {st["vectorize"]["n_arcs"]}，'
              f'dp_eps={st["vectorize"]["dp_eps"]})')
        print(f'  标定      : {cal["mode"]}，1 px = {cal["mm_per_px"]:.6f} mm '
              f'({cal["pixels_per_mm"]:.3f} px/mm)')
        print(f'  吸附      : 角度吸附 {st["snap"]["n_angle_snapped"]} 段，'
              f'合并直线 {st["snap"]["n_lines_merged"]} / 圆弧 {st["snap"]["n_arcs_merged"]}，'
              f'最大连接间隙 {st["snap"]["chain_gap_max_mm"]:.6f} mm')
        print(f'  实体      : 直线 {st["snap"]["n_lines"]} 段 / '
              f'圆弧 {st["snap"]["n_arcs"]} 段 / 合计 {st["snap"]["n_out"]} 段')
        print(f'  运动指令  : {st["emit"]["n_moves"]} 条，安全退刀 X={st["emit"]["safe_x"]:.3f}')
        for w in res['warnings']:
            print(f'  [警告] {w}')

    if a.check or a.report:
        if not a.quiet:
            print('\n' + res['gcode_report'])
        if a.report:
            os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
            with open(a.report, 'w', encoding='utf-8') as f:
                f.write(res['gcode_report'] + '\n')

    if a.dump_json:
        payload = {
            'input': a.input, 'output': out, 'version': __version__,
            'calibration': {'mode': res['calibration'].mode,
                            'mm_per_px': res['calibration'].scale,
                            'z_ref_px': res['calibration'].z_ref_px,
                            'z_sign': res['calibration'].z_sign,
                            'axis_row': res['calibration'].axis_row_used,
                            'detail': res['calibration'].detail},
            'entities_mm': [{k: v for k, v in e.items() if not k.startswith('_')}
                            for e in res['entities_mm']],
            'stats': res['stats'], 'warnings': res['warnings'],
        }
        os.makedirs(os.path.dirname(os.path.abspath(a.dump_json)), exist_ok=True)
        with open(a.dump_json, 'w', encoding='utf-8') as f:
            json.dump(_jsonable(payload), f, ensure_ascii=False, indent=1)

    return 0


if __name__ == '__main__':
    sys.exit(main())
