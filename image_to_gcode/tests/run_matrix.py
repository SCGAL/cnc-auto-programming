"""测试矩阵跑批：合成用例 → 流水线 → 6 项指标 → JSON / 表格 / 失败案例。

    python -m tests.run_matrix --subset quick      # 小样本冒烟
    python -m tests.run_matrix --subset dod        # DoD 门槛子集（72 例）
    python -m tests.run_matrix --subset full       # 立项文档 §6.1 全矩阵（324 例）
    python -m tests.run_matrix --grid-resolution   # 精度-分辨率关系曲线数据

输出（默认 ``tests/report/``）::

    matrix_<subset>.json     逐例明细
    matrix_<subset>.md       结果表
    failures/*.png           失败案例可视化（墨迹 + 提取实体 + 真值）
    failures.json            失败原因清单
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from i2g import pipeline, viz                       # noqa: E402
from i2g.errors import I2GError                     # noqa: E402
from tests import metrics, synth                    # noqa: E402

REPORT_DIR = os.path.join(_HERE, 'report')
CACHE_DIR = os.path.join(_HERE, 'synth')


# =============================================================================
# 单例
# =============================================================================

def run_case(spec, *, params_overrides=None, cachedir=CACHE_DIR, viz_dir=None,
             keep_case=False):
    """跑一个合成用例，返回结果 dict。任何异常都转成 ``ok=False`` 记录，不中断跑批。"""
    out = {'case': spec.name, 'spec': spec.to_json(), 'ok': False,
           'error': None, 'seconds': None, 'metrics': None, 'verdict': None,
           'stats': None, 'warnings': []}
    try:
        case = synth.ensure_case(spec, cachedir)
    except Exception as e:                                   # noqa: BLE001
        out['error'] = f'渲染失败: {type(e).__name__}: {e}'
        return out, None

    kw = {'ref_diameter': float(case.geom['ref_diameter'])}
    kw.update(params_overrides or {})
    t0 = time.time()
    try:
        res = pipeline.run_from_coverage(case.gray, pipeline.Params(**kw))
    except I2GError as e:
        out['error'] = f'{type(e).__name__}: {e}'
        out['seconds'] = time.time() - t0
        return out, case
    except Exception as e:                                   # noqa: BLE001
        out['error'] = f'未预期异常 {type(e).__name__}: {e}'
        out['traceback'] = traceback.format_exc()
        out['seconds'] = time.time() - t0
        return out, case
    out['seconds'] = time.time() - t0

    m = metrics.compare(case.entities, res['entities_mm'])
    ok, rows = metrics.verdict(m)
    out['ok'] = bool(m['ok'])
    out['metrics'] = {k: (None if m[k] != m[k] else float(m[k]))
                      for k in metrics.METRIC_ORDER}
    out['metrics']['endpoint_rms'] = _f(m.get('endpoint_rms'))
    out['metrics']['center_err_rms'] = _f(m.get('center_err_rms'))
    out['metrics']['radius_err_rms'] = _f(m.get('radius_err_rms'))
    out['metrics']['strict_radius_ok'] = bool(m.get('strict_radius_ok', False))
    out['metrics']['gauge_shift_mm'] = _f(m.get('gauge_shift_mm'))
    out['metrics']['endpoint_err_convention'] = _f(m.get('endpoint_err_convention'))
    out['verdict'] = [{'metric': r[0], 'value': _f(r[1]), 'target': r[2], 'pass': r[3]}
                      for r in rows]
    out['all_pass'] = bool(ok)
    out['n_entities'] = m['n_ext']
    out['n_truth'] = m['n_truth']
    out['pair_detail'] = m.get('pair_detail')

    st = res['stats']
    out['stats'] = {
        'mm_per_px': _f(st['calibration']['mm_per_px']),
        'pixels_per_mm': _f(st['calibration']['pixels_per_mm']),
        'scale_err_pct': _f(100.0 * (st['calibration']['mm_per_px']
                                     * case.spec.px_per_mm - 1.0)),
        'theta_deg': _f(st['centerline']['theta_deg']),
        'skew_contrast': _f(st['centerline']['contrast']),
        'noise_estimate_px': _f(st['vectorize'].get('noise_estimate_px')),
        'grow_tol_px': _f(st['vectorize'].get('grow_tol_px')),
        'n_segments': st['vectorize']['n_segments_after_merge'],
        'n_arcs_detected': st['vectorize']['n_arcs'],
        'axis_source': st['profile'].get('axis_source'),
        'stroke_width_px': _f(st['profile'].get('stroke_width_px')),
        'subpixel_shift_px': _f(st['profile'].get('subpixel_shift_px')),
        'axis_end_spread_px': _f(st['profile'].get('axis_row_end_spread')),
        'n_angle_snapped': st['snap']['n_angle_snapped'],
        'n_absorbed': st['snap'].get('n_absorbed'),
        'chain_gap_max_mm': _f(st['snap']['chain_gap_max_mm']),
        'n_moves': st['emit']['n_moves'],
        'safe_x': _f(st['emit']['safe_x']),
    }
    out['warnings'] = res['warnings']
    out['entities_mm'] = [{k: v for k, v in e.items() if not k.startswith('_')}
                          for e in res['entities_mm']]
    out['gcode'] = res['gcode']

    if not ok and viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
        try:
            viz.overlay(res['stages']['ink'], ents_mm=res['entities_mm'],
                        calib=res['calibration'], truth_mm=case.entities,
                        path=os.path.join(viz_dir, spec.name + '.png'),
                        upscale=2.0)
        except Exception:                                    # noqa: BLE001
            pass
    return out, case


def _f(v):
    try:
        return None if v is None or v != v else float(v)
    except Exception:                                        # noqa: BLE001
        return None


# =============================================================================
# 批量
# =============================================================================

def run_specs(specs, *, params_overrides=None, cachedir=CACHE_DIR,
              viz_dir=None, verbose=True):
    results = []
    t0 = time.time()
    for i, spec in enumerate(specs):
        r, _ = run_case(spec, params_overrides=params_overrides,
                        cachedir=cachedir, viz_dir=viz_dir)
        results.append(r)
        if verbose and ((i + 1) % 10 == 0 or i + 1 == len(specs)):
            npass = sum(1 for x in results if x.get('all_pass'))
            print(f'  {i+1}/{len(specs)}  通过 {npass}  '
                  f'({time.time()-t0:.1f}s)', flush=True)
    return results


def aggregate(results):
    good = [r for r in results if r.get('metrics')]
    agg = {'n_total': len(results), 'n_with_metrics': len(good),
           'n_all_pass': sum(1 for r in results if r.get('all_pass')),
           'n_error': sum(1 for r in results if r.get('error')),
           'n_metric_fail': sum(1 for r in results
                                if not r.get('error') and not r.get('all_pass'))}
    per = {}
    for k in metrics.METRIC_ORDER:
        vals = [r['metrics'][k] for r in good if r['metrics'].get(k) is not None]
        if not vals:
            per[k] = None
            continue
        if k in ('type_acc', 'arc_recall'):
            per[k] = {'worst': min(vals), 'mean': float(np.mean(vals)),
                      'target': metrics.TARGETS[k][1], 'dir': 'min'}
        else:
            per[k] = {'worst': max(vals), 'mean': float(np.mean(vals)),
                      'target': metrics.TARGETS[k][1], 'dir': 'max'}
    agg['per_metric'] = per
    # 每项指标独立的达标率（比"全部通过"更细）
    agg['metric_pass_rate'] = {}
    for k in metrics.METRIC_ORDER:
        op, thr, _u = metrics.TARGETS[k]
        vals = [r['metrics'][k] for r in good if r['metrics'].get(k) is not None]
        n_ok = sum(1 for v in vals if (v <= thr if op == '<=' else v >= thr))
        agg['metric_pass_rate'][k] = {'n_pass': n_ok, 'n': len(vals),
                                      'rate': (n_ok / len(vals)) if vals else None}
    return agg


# =============================================================================
# 表格
# =============================================================================

def _cell(v, nd=3, star=''):
    """None / NaN → '—'（空集不拿满分时，圆弧两项会返回 NaN，表格必须能吃下）。"""
    if v is None:
        return '—'
    try:
        if v != v:
            return '—'
    except Exception:                                        # noqa: BLE001
        return '—'
    return f'{v:.{nd}f}{star}'


def markdown_table(results, max_rows=None):
    hdr = ('| 用例 | px/mm | 线宽 | 倾斜 | 噪声 | R圆角 | 端点(mm) | 圆心(mm) | 半径(mm) | '
           '类型 | 圆弧召回 | 豪斯多夫(mm) | 结论 |')
    sep = '|' + '---|' * 13
    lines = [hdr, sep]
    rows = results if max_rows is None else results[:max_rows]
    for r in rows:
        s = r['spec'] or {}
        if r.get('error'):
            lines.append(f'| `{r["case"]}` | {s.get("px_per_mm", 0):g} | '
                         f'{s.get("line_width", 0):g} | '
                         f'{s.get("skew_deg", 0):+g}° | {s.get("noise", 0):g} | '
                         f'{s.get("fillet_r", 0):g} | '
                         f'— | — | — | — | — | — | ❌ 报错 |')
            continue
        mm = r['metrics'] or {}
        v = '✅' if r.get('all_pass') else '❌'
        star = '' if mm.get('strict_radius_ok') else ' *'
        lines.append(
            f'| `{r["case"]}` | {s.get("px_per_mm", 0):g} | {s.get("line_width", 0):g} | '
            f'{s.get("skew_deg", 0):+g}° | {s.get("noise", 0):g} | '
            f'{s.get("fillet_r", 0):g} | '
            f'{_cell(mm.get("endpoint_err"))} | {_cell(mm.get("center_err"))} | '
            f'{_cell(mm.get("radius_err"), star=star)} | '
            f'{_cell(mm.get("type_acc"), 2)} | {_cell(mm.get("arc_recall"), 2)} | '
            f'{_cell(mm.get("hausdorff"))} | {v} |')
    return '\n'.join(lines)


def aggregate_markdown(agg):
    L = ['| 指标 | 目标 | 实测最差 | 实测均值 | 独立达标率 |',
         '|---|---|---|---|---|']
    for k in metrics.METRIC_ORDER:
        p = agg['per_metric'].get(k)
        rate = agg['metric_pass_rate'].get(k, {})
        if not p:
            L.append(f'| `{k}` | — | — | — | — |')
            continue
        tgt = metrics._target_str(k)
        fmt = '{:.3f}' if k not in ('type_acc', 'arc_recall') else '{:.3f}'
        L.append(f'| `{k}` | {tgt} | {fmt.format(p["worst"])} | '
                 f'{fmt.format(p["mean"])} | '
                 f'{rate.get("n_pass", 0)}/{rate.get("n", 0)} '
                 f'({(rate.get("rate") or 0)*100:.1f}%) |')
    return '\n'.join(L)


# =============================================================================
# CLI
# =============================================================================

def _resolution_grid():
    """精度-分辨率关系：固定其它维度，只扫 px_per_mm。"""
    return [synth.Spec(px_per_mm=p, line_width=2, skew_deg=0.0, noise=0.0,
                       fillet_r=5.0) for p in (3, 4, 5, 6, 8, 10, 12, 15, 20, 25)]


def main(argv=None):
    ap = argparse.ArgumentParser(description='合成测试矩阵跑批 + 精度指标')
    ap.add_argument('--subset', choices=('quick', 'dod', 'full', 'resolution'),
                    default='quick')
    ap.add_argument('--out', default=REPORT_DIR)
    ap.add_argument('--cachedir', default=CACHE_DIR)
    ap.add_argument('--viz-failures', action='store_true', default=True)
    ap.add_argument('--no-viz', dest='viz_failures', action='store_false')
    a = ap.parse_args(argv)

    if a.subset == 'quick':
        specs = [
            synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=0.0, noise=0.0, fillet_r=5.0),
            synth.Spec(px_per_mm=10.0, line_width=1, skew_deg=0.0, noise=0.0, fillet_r=5.0),
            synth.Spec(px_per_mm=10.0, line_width=3, skew_deg=0.0, noise=0.0, fillet_r=5.0),
            synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=2.0, noise=0.0, fillet_r=5.0),
            synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=-2.0, noise=0.0, fillet_r=5.0),
            synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=0.0, noise=0.0, fillet_r=2.0),
            synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=0.0, noise=0.0, fillet_r=15.0),
            synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=0.0, noise=0.02, fillet_r=5.0),
            synth.Spec(px_per_mm=20.0, line_width=2, skew_deg=0.0, noise=0.0, fillet_r=5.0),
            synth.Spec(px_per_mm=5.0, line_width=2, skew_deg=0.0, noise=0.0, fillet_r=5.0),
        ]
    elif a.subset == 'dod':
        specs = synth.dod_cases()
    elif a.subset == 'full':
        specs = synth.matrix()
    else:
        specs = _resolution_grid()

    viz_dir = os.path.join(a.out, 'failures') if a.viz_failures else None
    os.makedirs(a.out, exist_ok=True)
    print(f'跑批 subset={a.subset} 共 {len(specs)} 例 ...')
    results = run_specs(specs, cachedir=a.cachedir, viz_dir=viz_dir)

    agg = aggregate(results)
    payload = {'subset': a.subset, 'n': len(specs), 'aggregate': agg,
               'results': results}
    with open(os.path.join(a.out, f'matrix_{a.subset}.json'), 'w',
              encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, f'matrix_{a.subset}.md'), 'w',
              encoding='utf-8') as f:
        f.write(f'# 测试矩阵结果 · subset={a.subset}（{len(specs)} 例）\n\n')
        f.write('## 汇总\n\n' + aggregate_markdown(agg) + '\n\n')
        f.write(f"- 全部通过: {agg['n_all_pass']}/{agg['n_total']}\n")
        f.write(f"- 报错: {agg['n_error']}\n")
        f.write(f"- 指标不达标: {agg['n_metric_fail']}\n\n")
        f.write('## 逐例明细\n\n' + markdown_table(results) + '\n')

    print(f"\n全部通过 {agg['n_all_pass']}/{agg['n_total']}，"
          f"报错 {agg['n_error']}，指标不达标 {agg['n_metric_fail']}")
    for k in metrics.METRIC_ORDER:
        p = agg['per_metric'].get(k)
        if p:
            print(f'  {k:14s} 最差 {p["worst"]:.4f}  目标 {metrics._target_str(k)}')
    print(f'→ {os.path.join(a.out, f"matrix_{a.subset}.md")}')
    return 0 if agg['n_all_pass'] == agg['n_total'] else 1


if __name__ == '__main__':
    sys.exit(main())
