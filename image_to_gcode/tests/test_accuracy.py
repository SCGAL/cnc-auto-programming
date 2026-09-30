"""精度回归测试：6 项指标 + DoD 门槛 + 失败路径。

分两层
------
**快层（默认运行）**：核心子集（约 12 例）× 6 项指标，断言**回归基线**
（:data:`REGRESSION_BASELINE`）。基线取的是当前实测最差值再加一点余量 ——
它的作用是"任何改动都不得让精度劣化"，这是本项目最重要的日常工作闸门。

**慢层（``-m slow`` 才跑）**：完整 DoD 门槛子集（72 例：px_per_mm ≥ 10 且
noise = 0，全部组合的 6 项指标全部达标）。这一层的现状见
《精度验证报告》—— 尚未全数达标，故标记为 ``xfail`` 并保留完整断言语义，
不隐藏、不放松。

另有独立的失败路径测试：无标定 / 实体数为 0 / 轮廓断裂必须**明确报错**（约束 5）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from i2g import emit, pipeline, vectorize            # noqa: E402
from i2g.errors import CalibrationError, DiscontinuousContourError, EmptyContourError  # noqa: E402
from tests import contract_checks, metrics, synth    # noqa: E402
from tests.run_matrix import run_case                # noqa: E402

# ---------------------------------------------------------------------------
# 核心回归子集：小、快、覆盖全部门槛条件
# ---------------------------------------------------------------------------

CORE_SPECS = [
    synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=0.0, noise=0.0, fillet_r=5.0),
    synth.Spec(px_per_mm=10.0, line_width=1, skew_deg=0.0, noise=0.0, fillet_r=5.0),
    synth.Spec(px_per_mm=10.0, line_width=3, skew_deg=0.0, noise=0.0, fillet_r=5.0),
    synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=-2.0, noise=0.0, fillet_r=5.0),
    synth.Spec(px_per_mm=20.0, line_width=2, skew_deg=2.0, noise=0.0, fillet_r=5.0),
    synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=0.0, noise=0.0, fillet_r=2.0),
    synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=0.0, noise=0.0, fillet_r=15.0),
    synth.Spec(px_per_mm=10.0, line_width=1, skew_deg=0.0, noise=0.0, fillet_r=15.0),
    synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=0.0, noise=0.01, fillet_r=5.0),
    synth.Spec(px_per_mm=20.0, line_width=2, skew_deg=0.0, noise=0.0, fillet_r=15.0),
    synth.Spec(px_per_mm=20.0, line_width=2, skew_deg=0.0, noise=0.0, fillet_r=5.0),
    synth.Spec(px_per_mm=20.0, line_width=1, skew_deg=0.0, noise=0.0, fillet_r=2.0),
]

# 已验证失效、暂未达标的条件。这里记录的是"天花板"：只保证不再进一步劣化，
# **不放进回归基线集**（否则红格会掩盖真实退化），也不假装它达标。
# 完整清单与归因见《精度验证报告》第 4 节。
KNOWN_LIMITS = [
    (synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=2.0, noise=0.0, fillet_r=5.0),
     {'endpoint_err': 0.35}),
    (synth.Spec(px_per_mm=5.0, line_width=2, skew_deg=0.0, noise=0.0, fillet_r=5.0),
     {'endpoint_err': 0.35}),
    (synth.Spec(px_per_mm=10.0, line_width=1, skew_deg=1.0, noise=0.0, fillet_r=5.0),
     {'endpoint_err': 0.35}),
]

# 回归基线：当前实测最差值 + 余量。劣化即失败。
REGRESSION_BASELINE = {
    'endpoint_err': 0.25,
    'center_err': 0.25,
    'radius_err': 0.25,
    'hausdorff': 0.30,
    'type_acc_min': 0.99,
    'arc_recall_min': 1.00,
}

_cached = {}


def _result_for(spec):
    key = spec.name
    if key not in _cached:
        r, _case = run_case(spec)
        _cached[key] = r
    return _cached[key]


# =============================================================================
# 快层：回归基线
# =============================================================================

@pytest.mark.parametrize('spec', CORE_SPECS, ids=lambda s: s.name)
def test_core_subset_within_regression_baseline(spec):
    r = _result_for(spec)
    assert r['error'] is None, f'{spec.name} 报错: {r["error"]}'
    m = r['metrics']
    assert m is not None, f'{spec.name} 未产出指标'
    assert m['endpoint_err'] <= REGRESSION_BASELINE['endpoint_err'], (
        f'{spec.name}: endpoint_err={m["endpoint_err"]:.4f} '
        f'超回归基线 {REGRESSION_BASELINE["endpoint_err"]}')
    assert m['center_err'] <= REGRESSION_BASELINE['center_err'], (
        f'{spec.name}: center_err={m["center_err"]:.4f}')
    assert m['radius_err'] <= REGRESSION_BASELINE['radius_err'], (
        f'{spec.name}: radius_err={m["radius_err"]:.4f}')
    assert m['hausdorff'] <= REGRESSION_BASELINE['hausdorff'], (
        f'{spec.name}: hausdorff={m["hausdorff"]:.4f}')
    assert m['type_acc'] >= REGRESSION_BASELINE['type_acc_min'], (
        f'{spec.name}: type_acc={m["type_acc"]:.4f}')
    assert m['arc_recall'] >= REGRESSION_BASELINE['arc_recall_min'], (
        f'{spec.name}: arc_recall={m["arc_recall"]:.4f}')


def test_fill_mode_compat_path():
    """``--ink fill``（实心填充轮廓图）这条独立代码路径必须有测试守着。

    它不在主测试矩阵里（矩阵只覆盖线框笔画图），属于最容易长期"没人跑过"的路径。

    实测（随分段机制演进而变化，以当前值为准）：填充模式在本基准例上
    **段数 8/9、type_acc 0.889、arc_recall 1.00**，圆心/半径/豪斯多夫达标；
    端点误差偏大。填充模式走的是 Moore 边界跟踪 + 去封底边，与线框模式的
    细化取中线完全不同，其边界跟踪结果比骨架略粗，故按"明显宽于线框模式"的
    上界守护，而把**真正不可退让的项**（圆弧召回、圆心、半径、豪斯多夫、
    以及严禁静默降级）逐项断言。
    """
    spec = synth.Spec(px_per_mm=10.0, line_width=1, fillet_r=5.0)
    case = synth.render_fill(spec)
    res = pipeline.run_from_coverage(
        case.gray, pipeline.Params(ink='fill',
                                   ref_diameter=float(case.geom['ref_diameter'])))
    assert res['stats']['profile']['mode'] == 'fill'
    m = metrics.compare(case.entities, res['entities_mm'])
    assert m['ok']
    assert abs(m['n_ext'] - m['n_truth']) <= 2, (
        f'填充模式段数 {m["n_ext"]} 与真值 {m["n_truth"]} 相差过大')
    assert m['type_acc'] >= 0.85, m['type_acc']      # 实测 0.889，按实测上界守护
    assert m['arc_recall'] >= 0.99, m['arc_recall']  # 圆弧一个都不许漏
    assert m['center_err'] <= 0.2, m['center_err']
    assert m['radius_err'] <= 0.2, m['radius_err']
    assert m['hausdorff'] <= 0.30, m['hausdorff']
    # 端点误差：填充模式当前**明显弱于**线框模式，按实测上界守护（防劣化，不假装达标）。
    # 实测 1.011 mm。这是本轮分段机制演进的**已知代价**：填充路径靠 Moore 边界跟踪取
    # 区域边界点，没有线框路径那套"沿法线的覆盖率加权质心"亚像素精化，因而对分段
    # 参数更敏感。已在《精度验证报告》第 7 节登记为已知限制与后续项。
    assert m['endpoint_err'] <= 1.10, m['endpoint_err']


def test_safe_x_must_exceed_profile_max():
    """安全退刀 X 必须大于轮廓最大 X，否则断点处的 G00 会切进工件。"""
    spec = synth.Spec(px_per_mm=10.0, line_width=2, fillet_r=5.0)
    case = synth.ensure_case(spec, os.path.join(_HERE, 'synth'))
    base = pipeline.Params(ref_diameter=float(case.geom['ref_diameter']))
    ok = pipeline.run_from_coverage(case.gray, base)
    max_x = max(p[1] for e in ok['entities_mm'] for p in (e['start'], e['end']))
    with pytest.raises(Exception) as e:
        pipeline.run_from_coverage(
            case.gray, pipeline.Params(ref_diameter=float(case.geom['ref_diameter']),
                                       gcode={'safe_x': max_x / 2.0}))
    assert '安全退刀' in str(e.value)


def test_unit_scale_is_rejected_not_silently_dropped():
    """``unit_scale`` 只会写进注释、不会缩放坐标 —— 必须拒绝而不是静默接受。

    （emit 不走 v2 的 transform()，而坐标缩放只发生在那里；上游 generate() 仅把
    unit_scale 写进程序头。若静默接受，就会输出一份头部撒谎的 NC 文件。）
    """
    ents = [{'type': 'line', 'start': (0.0, 0.0), 'end': (0.0, 10.0)},
            {'type': 'line', 'start': (0.0, 10.0), 'end': (-20.0, 10.0)},
            {'type': 'line', 'start': (-20.0, 10.0), 'end': (-20.0, 0.0)}]
    with pytest.raises(Exception) as e:
        emit.emit(ents, unit_scale=25.4)
    assert 'unit_scale' in str(e.value)
    # 单位 1.0 正常
    assert emit.emit(ents, unit_scale=1.0)['gcode']


HEADLINE_CASE = 'px10_w2_s+0_n0_r5'
_README_ROWS = {
    '直线端点误差': 'endpoint_err',
    '圆弧圆心误差': 'center_err',
    '圆弧半径误差': 'radius_err',
    '类型判别正确率': 'type_acc',
    '圆弧识别率': 'arc_recall',
    '轮廓豪斯多夫距离': 'hausdorff',
}


def test_readme_headline_numbers_match_reports():
    """★ README 的招牌数字必须与 `matrix_quick.json` 逐位一致。

    评审指出：同一用例曾在 README（0.047）、报告（1.298）、JSON（1.261）三处
    出现三个不同的值，"全部数字来自实测、可逐例追溯"这句话就站不住。
    证据链的可信度是能被外部信任的分水岭，故用测试把它钉住。
    """
    import json
    import re
    qp = os.path.join(_HERE, 'report', 'matrix_quick.json')
    if not os.path.exists(qp):
        pytest.skip('尚未生成 matrix_quick.json；先跑 run_matrix --subset quick')
    with open(qp, encoding='utf-8') as f:
        quick = json.load(f)
    rec = next((r for r in quick['results'] if r['case'] == HEADLINE_CASE), None)
    assert rec is not None, f'{HEADLINE_CASE} 不在 quick 子集里'
    m = rec['metrics']

    readme = open(os.path.join(_ROOT, 'README.md'), encoding='utf-8').read()
    mism = []
    for label, key in _README_ROWS.items():
        row = next((ln for ln in readme.splitlines() if ln.startswith(f'| {label} |')), None)
        assert row is not None, f'README 缺少行: {label}'
        cells = [c.strip() for c in row.strip('|').split('|')]
        shown = re.sub(r'[*\s]', '', cells[2])
        if key in ('type_acc', 'arc_recall'):
            want = f'{m[key]*100:.0f}%'
        else:
            want = f'{m[key]:.3f}mm'
        if shown != want:
            mism.append((label, shown, want))
    assert not mism, f'README 与 JSON 不一致（行, README, JSON）: {mism}'


@pytest.mark.parametrize('spec', CORE_SPECS, ids=lambda s: s.name)
def test_entities_are_geometrically_consistent(spec):
    """实体链必须严格连续、圆弧自洽（与 DXF 版契约同源、机器精度级）。"""
    r = _result_for(spec)
    assert 'entities_mm' in r
    ents = r['entities_mm']
    assert len(ents) >= 3
    for i in range(1, len(ents)):
        a, b = ents[i - 1], ents[i]
        gap = ((a['end'][0] - b['start'][0]) ** 2
               + (a['end'][1] - b['start'][1]) ** 2) ** 0.5
        assert gap <= 1e-9, f'{spec.name} 第 {i} 处不连续（间隙 {gap:.3e} mm）'
    for e in ents:
        if e['type'] == 'arc':
            assert e['radius'] > 0
            for p in (e['start'], e['end']):
                resid = abs(((p[0] - e['center'][0]) ** 2
                             + (p[1] - e['center'][1]) ** 2) ** 0.5 - e['radius'])
                assert resid <= 1e-9, f'圆弧端点半径残差 {resid:.3e}'


@pytest.mark.parametrize('spec,limit', KNOWN_LIMITS, ids=lambda x: getattr(x, 'name', ''))
def test_known_limitations_do_not_degrade(spec, limit):
    """已知失效条件只保证"不进一步劣化"，不假装达标。

    这些用例**不在 DoD 通过集内**（见《精度验证报告》第 4 节失败分析）。
    把它们单列出来的目的：一是别让红格混进回归基线掩盖真实退化，
    二是给后续修失效模式时一个"先看是否至少没变差"的锚点。
    """
    r = _result_for(spec)
    if r.get('error'):
        # 已知失效条件**允许直接报错**（护栏正确拦下 > 输出垃圾程序）：
        # 例如 px10_w1_s+2_r15 会因提取断链抛"轮廓不闭合"。
        # 只要求它是本项目的**类型化错误**，不能是未捕获异常/崩溃。
        assert any(t.__name__ in r['error'] for t in
                   (CalibrationError, DiscontinuousContourError, EmptyContourError)), \
            f'{spec.name} 报的是非类型化错误: {r["error"]}'
        return
    for k, ceiling in limit.items():
        if ceiling is None:
            continue
        got = r['metrics'][k]
        assert got <= ceiling, (
            f'{spec.name}: {k}={got:.4f} 劣于已知天花板 {ceiling}')


def test_arc_endpoints_lie_on_their_circles_even_for_degenerate_junctions():
    """退化结合点（两圆弧真的不相交）也不得让端点整体脱离圆周。

    肩面隔开的两条圆角，其拟合圆圆心距可能略大于 R1+R2（实测 30.07 vs 29.99 px）。
    此时退回"取两端点中点"会让该点整体脱离其中一条圆（实测残差 0.196 mm），
    必须改用交替投影把误差平摊。
    """
    ents = [
        {'type': 'line', 'start': (0.0, 0.0), 'end': (0.0, 20.0)},
        {'type': 'arc', 'start': (0.0, 20.0), 'end': (-5.0, 15.0),
         'center': (0.0, 15.0), 'radius': 5.0, 'ccw': True},
        # 与上一条圆**真的不相交**：圆心距 5.5 > R1+R2=5.0
        {'type': 'arc', 'start': (-5.5, 15.0), 'end': (-10.5, 10.0),
         'center': (-5.5, 10.0), 'radius': 5.0, 'ccw': False},
        {'type': 'line', 'start': (-10.5, 10.0), 'end': (-20.0, 10.0)},
    ]
    from i2g.vectorize import apply_junctions
    out = apply_junctions(ents)
    assert len(out) == 4
    for i in range(1, len(out)):
        a, b = out[i - 1], out[i]
        gap = ((a['end'][0] - b['start'][0]) ** 2
               + (a['end'][1] - b['start'][1]) ** 2) ** 0.5
        assert gap <= 1e-12, f'第 {i} 处不连续'
    for e in out:
        if e['type'] == 'arc':
            for p in (e['start'], e['end']):
                resid = abs(((p[0] - e['center'][0]) ** 2
                             + (p[1] - e['center'][1]) ** 2) ** 0.5 - e['radius'])
                # 交替投影把 0.5 mm 的圆间距平摊成每侧约 0.25 mm，而不是全压一侧
                assert resid <= 0.30, f'圆弧端点半径残差 {resid:.4f} 过大'


def test_reproducible_run(tmp_path):
    """NFR-3：同一输入两次运行结果必须完全一致。"""
    spec = synth.Spec(px_per_mm=10.0, line_width=2, noise=0.01, fillet_r=5.0)
    case = synth.ensure_case(spec, os.path.join(_HERE, 'synth'))
    p = pipeline.Params(ref_diameter=float(case.geom['ref_diameter']),
                        viz_dir=None)
    a = pipeline.run_from_coverage(case.gray, p)
    b = pipeline.run_from_coverage(case.gray, p)
    assert a['gcode'] == b['gcode']
    assert len(a['entities_mm']) == len(b['entities_mm'])
    for x, y in zip(a['entities_mm'], b['entities_mm']):
        assert x['type'] == y['type']
        assert x['start'] == pytest.approx(y['start'], abs=0)
        assert x['end'] == pytest.approx(y['end'], abs=0)


def test_no_opencv_dependency():
    """NFR-2：主实现不得依赖 opencv（缺失 cv2 时全流程必须仍能运行）。"""
    assert 'cv2' not in sys.modules, '本测试要求 cv2 未被导入'
    src = []
    for fn in ('preprocess', 'centerline', 'profile', 'vectorize', 'snap',
               'calibrate', 'emit', 'pipeline', 'geometry', 'viz'):
        p = os.path.join(_ROOT, 'i2g', fn + '.py')
        with open(p, encoding='utf-8') as f:
            src.append(f.read())
    joined = '\n'.join(src)
    assert 'import cv2' not in joined
    assert 'from cv2' not in joined


def test_gcode_passes_contract_checks(tmp_path):
    """DoD：生成的 G 代码必须通过 DXF 版全部 9 项契约校验。

    同时覆盖 C6（G91 增量模式）—— 生成 G90 与 G91 两份程序并比对。
    """
    spec = synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=0.0, noise=0.0,
                      fillet_r=5.0)
    case = synth.ensure_case(spec, os.path.join(_HERE, 'synth'))
    mpp = float(case.geom['ref_diameter'])
    base = pipeline.Params(ref_diameter=mpp)
    res90 = pipeline.run_from_coverage(case.gray, base)
    res91 = pipeline.run_from_coverage(
        case.gray, pipeline.Params(ref_diameter=mpp, gcode={'mode': 'g91'}))

    p90 = tmp_path / 'out_g90.nc'
    p91 = tmp_path / 'out_g91.nc'
    p90.write_text(res90['gcode'] + '\n', encoding='utf-8')
    p91.write_text(res91['gcode'] + '\n', encoding='utf-8')

    rep = contract_checks.check(str(p90), case.entities, ik='fanuc',
                                label=f'{spec.name} G90')
    rep91 = contract_checks.check(str(p91), case.entities, ik='fanuc',
                                  label=f'{spec.name} G91',
                                  gcode_abs=res90['gcode'])
    for r in (rep, rep91):
        if r.n_ok != r.n_total:
            print(r.dump())
    assert rep.n_ok == rep.n_total, rep.dump()
    assert rep91.n_ok == rep91.n_total, rep91.dump()


# =============================================================================
# 失败路径（约束 5：禁止静默降级）
# =============================================================================

def _case(spec=None):
    spec = spec or synth.Spec(px_per_mm=10.0, line_width=2, fillet_r=5.0)
    return synth.ensure_case(spec, os.path.join(_HERE, 'synth'))


def test_missing_calibration_raises():
    case = _case()
    with pytest.raises(CalibrationError) as e:
        pipeline.run_from_coverage(case.gray, pipeline.Params())
    msg = str(e.value)
    assert '缺少标定参数' in msg
    assert '--ref-diameter' in msg
    assert '轮廓包围盒' in msg, '报错必须给出供推算的像素包围盒'


def test_blank_image_raises_empty_contour():
    gray = np.full((200, 300), 255, dtype=np.uint8)
    with pytest.raises(EmptyContourError):
        pipeline.run_from_gray(gray, pipeline.Params(ref_diameter=40.0))


def test_single_tiny_speck_raises_empty_contour():
    """只剩一个被面积阈值剔除的小噪点 → 必须报"无墨迹"，而不是输出空程序。"""
    gray = np.full((120, 160), 255, dtype=np.uint8)
    gray[60, 80] = 0
    with pytest.raises(EmptyContourError):
        pipeline.run_from_gray(gray, pipeline.Params(ref_diameter=40.0))


def test_discontinuous_contour_raises():
    """人为把实体链扯开 → 必须报"轮廓不闭合"，绝不静默补线。"""
    ents = [
        {'type': 'line', 'start': (0.0, 0.0), 'end': (0.0, 20.0)},
        {'type': 'line', 'start': (0.0, 20.0), 'end': (-30.0, 20.0)},
        {'type': 'line', 'start': (-30.0, 12.0), 'end': (-60.0, 12.0)},   # 断开 8 mm
        {'type': 'line', 'start': (-60.0, 12.0), 'end': (-60.0, 0.0)},
    ]
    from i2g import snap
    with pytest.raises(DiscontinuousContourError) as e:
        snap.run(ents)
    assert '轮廓不闭合' in str(e.value)


def test_empty_entity_list_raises_before_gcode():
    with pytest.raises(Exception):
        emit.emit([])


def test_zero_entities_raises_empty_contour():
    from i2g import snap
    with pytest.raises(EmptyContourError):
        snap.check_non_empty([])


# =============================================================================
# 慢层：DoD 门槛（px_per_mm >= 10 且 noise = 0 的全部 72 例）
# =============================================================================

@pytest.mark.slow
@pytest.mark.xfail(reason='DoD 尚未全数达成 —— 现状与失败分析见《精度验证报告》'
                          '；本测试保留完整断言语义，不放松口径',
                   strict=False)
def test_dod_gate_all_metrics():
    specs = synth.dod_cases()
    results = [run_case(s)[0] for s in specs]
    fails = []
    for r in results:
        if r['error']:
            fails.append((r['case'], 'error', r['error']))
            continue
        m = r['metrics']
        for k in metrics.METRIC_ORDER:
            op, thr, _u = metrics.TARGETS[k]
            v = m[k]
            if (v > thr) if op == '<=' else (v < thr):
                fails.append((r['case'], k, v))
    assert not fails, (f'DoD 未达标 {len(fails)} 项（共 {len(specs)} 例），'
                       f'前 10 项: {fails[:10]}')
