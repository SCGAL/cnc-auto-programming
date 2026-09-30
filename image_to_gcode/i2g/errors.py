"""错误类型。

立项文档 NFR-7 / 用户约束 5：**禁止静默降级**。以下三种情况必须明确报错退出，
不允许"输出一段看起来像 G 代码的垃圾"：

* 标定缺失（没有 --mm-per-px / --ref-diameter / --ref-points）
* 识别到的实体数为 0
* 轮廓断裂（相邻实体端点间隙超限）

所有错误的 ``exit_code`` 统一为 2，与"意外异常"（1）区分。
"""

from __future__ import annotations


class I2GError(Exception):
    """本项目业务错误的基类。"""
    exit_code = 2


class ImageReadError(I2GError):
    """图像不存在 / 无法解码 / 尺寸异常。"""


class CalibrationError(I2GError):
    """标定参数缺失或不可用。"""


class EmptyContourError(I2GError):
    """没有从图中提取到任何轮廓 / 实体数为 0。"""


class DiscontinuousContourError(I2GError):
    """轮廓断裂：相邻实体端点间隙超过容差。"""


class ContourDirectionError(I2GError):
    """刀路方向非法：Z 沿刀路不再单调（轮廓折返 / 多视图混入 / 标注线干扰）。"""


class GeometryError(I2GError):
    """几何退化：拟合失败、半径为负、自适应参数不合理等。"""
