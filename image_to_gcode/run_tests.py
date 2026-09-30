"""零配置测试入口。

    python run_tests.py                    # 跑 tests/ 下全部
    python run_tests.py tests/test_units.py -k rasterizer

优先使用项目内 ``_deps/`` 里的 pytest（``pip install --target _deps pytest``），
其次使用全局 pytest。这样测试框架本身也不污染全局环境（与 NFR-2 的
"依赖最小化"同一取向），同时测试代码仍按标准 pytest 组织。
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DEPS = os.path.join(ROOT, '_deps')

for p in (DEPS, ROOT):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    import pytest
except ImportError:                                  # pragma: no cover
    print('未找到 pytest。请任选其一：\n'
          '  python -m pip install --target _deps pytest '
          '-i https://pypi.tuna.tsinghua.edu.cn/simple\n'
          '  python -m pip install pytest', file=sys.stderr)
    raise SystemExit(2)

if __name__ == '__main__':
    args = sys.argv[1:] or [os.path.join(ROOT, 'tests')]
    raise SystemExit(pytest.main(['-q', '-ra', *args]))
