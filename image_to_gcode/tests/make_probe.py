"""参数探针：验证"某类失败不是调参能解决的"。

    python -m tests.make_probe                 # 默认网格
    python -m tests.make_probe --quick         # 只跑 2 组，约 1 分钟

产出 ``tests/report/probe.json``：每个参数组合下、固定用例集上的通过数与最差端点误差。
报告《精度验证报告》§4.2 引用它来支撑"结构性限制而非调参问题"这一结论 ——
**这个结论必须有可复现的生成脚本**，不能靠一份来源不明的 JSON。

网格维度（在 `--quick` 之外）：
    split_iters     曲率切分迭代次数        (1, 2)
    curv_split_tol  曲率跳变阈值（弧度）    (0.06, 0.10, 0.16)
    refine_search   边界精修搜索半径（点）  (30, 60)
    refine_iters    边界精修迭代次数        (2, 6)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from i2g import vectorize                          # noqa: E402
from tests import synth                            # noqa: E402
from tests.run_matrix import REPORT_DIR, run_case  # noqa: E402

# 覆盖三类失效模式的小样本：R5（基准）、R2（小圆角）、R15（大圆角）× 线宽 × 倾斜
PROBE_SPECS = [
    synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=0.0, fillet_r=5.0),
    synth.Spec(px_per_mm=10.0, line_width=1, skew_deg=0.0, fillet_r=5.0),
    synth.Spec(px_per_mm=10.0, line_width=3, skew_deg=0.0, fillet_r=5.0),
    synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=2.0, fillet_r=5.0),
    synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=-2.0, fillet_r=5.0),
    synth.Spec(px_per_mm=20.0, line_width=3, skew_deg=1.0, fillet_r=5.0),
    synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=0.0, fillet_r=2.0),
    synth.Spec(px_per_mm=20.0, line_width=1, skew_deg=-2.0, fillet_r=2.0),
    synth.Spec(px_per_mm=20.0, line_width=2, skew_deg=2.0, fillet_r=2.0),
    synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=0.0, fillet_r=15.0),
    synth.Spec(px_per_mm=20.0, line_width=2, skew_deg=0.0, fillet_r=15.0),
    synth.Spec(px_per_mm=10.0, line_width=1, skew_deg=1.0, fillet_r=15.0),
]

GRID_FULL = list(itertools.product((1, 2), (0.06, 0.10, 0.16), (30, 60), (2, 6)))
GRID_QUICK = [(1, 0.10, 30, 2), (2, 0.06, 60, 6)]


def run_grid(grid, specs):
    out = []
    for si, ct, rs, ri in grid:
        ov = {'vector': vectorize.VectorizeParams(
            split_iters=si, curv_split_tol=ct, refine_search=rs, refine_iters=ri)}
        npass, worst, n_err = 0, 0.0, 0
        for s in specs:
            r, _ = run_case(s, params_overrides=ov)
            if r.get('error'):
                n_err += 1
                continue
            if r.get('all_pass'):
                npass += 1
            ep = (r.get('metrics') or {}).get('endpoint_err')
            if ep is not None and ep == ep:
                worst = max(worst, ep)
        rec = {'split_iters': si, 'curv_split_tol': ct, 'refine_search': rs,
               'refine_iters': ri, 'n_pass': npass, 'n_total': len(specs),
               'n_error': n_err, 'worst_endpoint_err': round(worst, 4)}
        out.append(rec)
        print(f"  split_iters={si} curv_split_tol={ct} refine_search={rs} "
              f"refine_iters={ri}: {npass}/{len(specs)}  worst_ep={worst:.3f}",
              flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='矢量化参数探针')
    ap.add_argument('--quick', action='store_true', help='只跑 2 组')
    a = ap.parse_args(argv)
    grid = GRID_QUICK if a.quick else GRID_FULL
    print(f'探针网格 {len(grid)} 组 × {len(PROBE_SPECS)} 例 ...', flush=True)
    recs = run_grid(grid, PROBE_SPECS)
    os.makedirs(REPORT_DIR, exist_ok=True)
    payload = {
        'n_grid': len(grid), 'n_specs': len(PROBE_SPECS),
        'grid_dims': {'split_iters': sorted({r['split_iters'] for r in recs}),
                      'curv_split_tol': sorted({r['curv_split_tol'] for r in recs}),
                      'refine_search': sorted({r['refine_search'] for r in recs}),
                      'refine_iters': sorted({r['refine_iters'] for r in recs})},
        'specs': [s.name for s in PROBE_SPECS],
        'records': recs,
    }
    p = os.path.join(REPORT_DIR, 'probe.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    npasses = {r['n_pass'] for r in recs}
    print(f'→ {p}')
    print(f'通过数取值集合 = {sorted(npasses)}'
          + ('  ← 各组合完全相同，支持"结构性限制而非调参问题"'
             if len(npasses) == 1 else '  ← 组合间有差异，说明仍有调参空间'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
