"""image_to_gcode 的 pytest 入口与路径兜底。

项目根（含 ``i2g/`` 与 ``tests/``）加入 sys.path，使
``pytest`` 从任意目录调用都能 import 到包。
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
