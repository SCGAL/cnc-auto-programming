"""生成测试用 DXF 文件，用于验证 dxf_to_gcode.py

三个测试件:
  1. shaft.dxf  —— 带圆角的车削阶梯轴外轮廓（LINE + ARC），含中心线干扰
  2. bulge.dxf  —— 单个 LWPOLYLINE bulge 圆弧（圆心/半径已知，用于数值校验）
  3. circle.dxf —— 整圆 CIRCLE（测试圆拆分逻辑）
"""
import math
import os

import ezdxf

OUT = os.path.dirname(os.path.abspath(__file__))


def make_shaft():
    """车削阶梯轴外轮廓（上半部，y=半径值），Z 从左到右 0 → -80。
    圆角: R5，圆心 (-35,15)，切点 (-30,15) 与 (-35,20)
    """
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    segs = [
        ((0, 0), (0, 20)),        # 右端面  Z=0  R0→R20
        ((0, 20), (-35, 20)),     # 外圆 Ø40
        ((-30, 15), (-60, 15)),   # 台阶面到外圆 Ø30（圆角切点起）
        ((-60, 15), (-65, 10)),   # 45° 倒角
        ((-65, 10), (-80, 10)),   # 外圆 Ø20
        ((-80, 10), (-80, 0)),    # 左端面
    ]
    for (z1, x1), (z2, x2) in segs:
        msp.add_line((z1, x1), (z2, x2))

    # 圆角圆弧：DXF ARC 恒为 CCW，start=0°(点 -30,15) → end=90°(点 -35,20)
    msp.add_arc(center=(-35, 15), radius=5, start_angle=0, end_angle=90)

    # 干扰项：中心线（X≈0 水平线，应被过滤）
    msp.add_line((5, 0), (-85, 0))

    path = os.path.join(OUT, "shaft.dxf")
    doc.saveas(path)
    return path


def make_bulge():
    """LWPOLYLINE bulge 圆弧：真值圆心 (-10,20)，半径 10，CCW 90°。

    起点 (-10,10) → 终点 (0,20)，bulge = tan(90°/4) = tan(22.5°) ≈ 0.414214
    """
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    pts = [(-10, 10, 0.414213562373095), (0, 20, 0.0)]  # 起点带 bulge
    msp.add_lwpolyline(pts, format="xyb")

    path = os.path.join(OUT, "bulge.dxf")
    doc.saveas(path)
    return path


def make_pulse():
    """脉冲当量测试：坐标故意不在 0.1 网格上（Z=12.3456, X=7.7777）"""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_line((12.3456, 7.7777), (12.3456, 0.0))
    msp.add_line((12.3456, 7.7777), (-30.1234, 7.7777))
    msp.add_line((-30.1234, 7.7777), (-30.1234, 0.0))

    path = os.path.join(OUT, "pulse.dxf")
    doc.saveas(path)
    return path


def make_circle():
    """整圆：圆心 (-45,0) 半径 3"""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_circle(center=(-45, 0), radius=3)
    msp.add_line((0, 0), (0, 10))

    path = os.path.join(OUT, "circle.dxf")
    doc.saveas(path)
    return path


if __name__ == "__main__":
    for fn in (make_shaft, make_bulge, make_circle, make_pulse):
        p = fn()
        print("生成:", p)
