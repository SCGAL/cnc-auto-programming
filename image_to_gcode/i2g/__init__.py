"""图像识别版数控自动编程工具 —— 核心包。

把"零件轮廓图（PNG/JPG）"转换为"车削精加工轮廓路径 G 代码"。

流水线（对应立项文档 §3.1）::

    preprocess  灰度 → Otsu 二值 → 去噪 → 形态学闭运算 → 小连通域剔除
    centerline  倾斜角估计（投影直方图能量） → 旋转校正 → 中心线定位
    profile     细化取**线条中线** → 图论最长路跟踪 → 有序点列
    vectorize   DP 简化 → 直线/圆弧判别 → Kåsa 初值 → RANSAC 去外点 → 几何精修
    calibrate   像素 → 毫米（显式比例 / 已知外径 / 两特征点）
    snap        角度吸附 → 共线/共弧合并 → 端点合并 → 连续性检查
    emit        转 v2 实体格式 → 调用 ../dxf_to_gcode_v2.py 的 generate()

硬边界（本版不做，见 README"后续版本"）：
    粗车循环 G71/G70、螺纹、切槽、钻孔、多刀具、刀补 G41/G42、
    内孔、多视图、GD&T 识别、透视校正、刀具干涉仿真。

依赖：numpy + Pillow（必需）。opencv-python 完全不需要。
"""

__version__ = '1.0.0'
__all__ = ['errors', 'geometry', 'preprocess', 'centerline', 'profile',
           'vectorize', 'calibrate', 'snap', 'emit', 'viz', 'pipeline']
