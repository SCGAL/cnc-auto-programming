"""单元测试：几何核、真值生成器、精度比对框架。

⚠️ 本文件的第一部分是 **M1 阶段的"验证框架自验证"**：
把真值喂给比对框架必须得到零误差；注入已知误差必须按比例响应；
Z 方向整体平移不得被当成误差。
如果这部分不过，后面流水线报出的任何精度数字都没有意义。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from i2g import geometry as G
from tests import metrics, synth


# =============================================================================
# 第一部分：真值几何自校验
# =============================================================================

@pytest.mark.parametrize('fillet_r', [2, 5, 15])
def test_truth_profile_passes_self_check(fillet_r):
    """真值几何必须自身无瑕：连续、Z 单调、X≥0、圆弧端点落在圆上、与邻边真相切。"""
    ents, geom = synth.build_profile(fillet_r)
    assert synth.check_truth(ents) == []


@pytest.mark.parametrize('fillet_r', [2, 5, 15])
def test_truth_profile_topology(fillet_r):
    ents, geom = synth.build_profile(fillet_r)
    assert len(ents) == 9
    assert sum(1 for e in ents if e['type'] == 'arc') == 2
    assert geom['face_mm'] >= fillet_r - 1e-9, '肩面必须不短于圆角半径'
    assert geom['r_small'] > 0
    # 首末都落在轴线上（车削上半部外轮廓的两端）
    assert ents[0]['start'][1] == 0.0
    assert ents[-1]['end'][1] == 0.0


def test_truth_rejects_impossible_fillet():
    """立项文档 §6.1 那种"一条圆角直接连接两段外圆"不可能相切，必须被自校验抓住。"""
    bad = [
        {'type': 'line', 'start': (0.0, 20.0), 'end': (-35.0, 20.0)},
        {'type': 'arc', 'start': (-35.0, 20.0), 'end': (-30.0, 15.0),
         'center': (-35.0, 15.0), 'radius': 5.0, 'ccw': False},
        {'type': 'line', 'start': (-30.0, 15.0), 'end': (-60.0, 15.0)},
    ]
    problems = synth.check_truth(bad)
    joined = '\n'.join(problems)
    assert '不相切' in joined, problems
    assert any('Z 非单调' in p or '不相切' in p for p in problems)


def test_truth_checker_rejects_non_tangent_arc():
    """连续、Z 单调、落到轴线的链条里，若圆弧与邻边其实是 90° 折角，必须判失败。

    这正是立项文档 §6.1 示例真值的错误模式：圆角与相邻直线只是端点相接，
    圆心并不在距离 r 处 —— 看着像相切，实际是折角。
    """
    ents = [
        {'type': 'line', 'start': (0.0, 0.0), 'end': (0.0, 20.0)},
        {'type': 'line', 'start': (0.0, 20.0), 'end': (-10.0, 20.0)},
        {'type': 'arc', 'start': (-10.0, 20.0), 'end': (-15.0, 15.0),
         'center': (-15.0, 20.0), 'radius': 5.0, 'ccw': False},
    ]
    problems = synth.check_truth(ents)
    assert len(problems) == 1, problems
    assert '不相切' in problems[0]
    # 该链条除相切性外一切正常（连续性 / Z 单调 / 落到轴线 / 端点在圆上）
    assert 'Z 非单调' not in problems[0]


def test_truth_checker_accepts_true_tangency():
    """真正的相切（切点即结合点）必须通过。"""
    ents = [
        {'type': 'line', 'start': (0.0, 0.0), 'end': (0.0, 20.0)},
        {'type': 'line', 'start': (0.0, 20.0), 'end': (-10.0, 20.0)},
        {'type': 'arc', 'start': (-10.0, 20.0), 'end': (-15.0, 15.0),
         'center': (-10.0, 15.0), 'radius': 5.0, 'ccw': False},
        {'type': 'line', 'start': (-15.0, 15.0), 'end': (-15.0, 0.0)},
    ]
    assert synth.check_truth(ents) == []


# =============================================================================
# 第二部分：光栅化正确性（真值图像本身必须可信）
# =============================================================================

def _x_of_row(case, row):
    """像素行 → X（mm）。

    ⚠️ 半像素约定：像素 (row, col) 覆盖 [row,row+1)×[col,col+1)，
    其**中心**在 row+0.5。真值点被光栅化到连续坐标 col_cont = margin + (u−u_min)·s，
    故 col_cont = col_index + 0.5。漏掉这 0.5 px 就是 0.05 mm 的系统偏差 @10px/mm，
    已占 0.2 mm 预算的 25% —— 故此约定在流水线里也必须一致。
    """
    m = case.meta
    return m['v_max'] - (row + 0.5 - m['margin_px']) / m['px_per_mm']


def _z_of_col(case, col):
    m = case.meta
    return m['u_min'] + (col + 0.5 - m['margin_px']) / m['px_per_mm']


def _col_of_z(case, z_mm):
    m = case.meta
    return int(round(m['margin_px'] + (z_mm - m['u_min']) * m['px_per_mm'] - 0.5))


def _top_run_center(case, col):
    rows = np.where(case.mask[:, col])[0]
    if len(rows) == 0:
        return None
    runs, cur = [], [rows[0]]
    for r in rows[1:]:
        if r == cur[-1] + 1:
            cur.append(r)
        else:
            runs.append(cur)
            cur = [r]
    runs.append(cur)
    top = runs[0]
    return sum(top) / len(top)


@pytest.mark.parametrize('px_per_mm', [5, 10, 20])
@pytest.mark.parametrize('fillet_r', [2, 5, 15])
def test_rasterizer_centerline_lands_on_truth(px_per_mm, fillet_r):
    """笔画中线的位置必须落在真值几何上（容差远小于 1 px）。

    这条测试是"合成真值可信"的核心：若光栅化本身带系统偏移，
    后面测到的就不是算法的误差，而是画图误差。
    """
    case = synth.render(synth.Spec(px_per_mm=px_per_mm, line_width=3,
                                   fillet_r=fillet_r))
    r1 = case.geom['r_max']
    z_mid = -8.0                      # 大端外圆中段，避开圆角与肩面
    col = _col_of_z(case, z_mid)
    row = _top_run_center(case, col)
    assert row is not None
    x_meas = _x_of_row(case, row)
    # 中线应等于 r_max（笔画对称，中线就是真值线）
    tol_mm = 0.6 / px_per_mm
    assert abs(x_meas - r1) <= tol_mm, f'{x_meas:.4f} vs {r1} (tol {tol_mm:.4f})'


@pytest.mark.parametrize('line_width', [1, 2, 3])
def test_rasterizer_produces_a_connected_stroke(line_width):
    """墨迹必须是一条**连续笔画**，而不是几个孤立圆斑。

    专门防"直线段未加密 → 圆盘笔刷只落在端点"这类光栅化缺陷：
    若只画端点，像素数会比 轮廓长 × 线宽 少一个数量级（实测会掉到 0.23×）。

    上界放宽到 2.5×：一条恰好落在像素边界上的 1 px 线，会把相邻两行各覆盖 50%，
    二值化后**两行都算墨迹**，这是栅格化的正常现象（笔画中线仍在正确位置，
    由 test_rasterizer_centerline_lands_on_truth 单独把关），不是缺陷。
    """
    case = synth.render(synth.Spec(px_per_mm=10.0, line_width=line_width))
    length_px = G.polyline_length(G.entities_polyline(case.entities, 0.01)) \
        * case.spec.px_per_mm
    expect = length_px * line_width
    got = int(case.mask.sum())
    assert 0.6 * expect <= got <= 2.5 * expect, f'ink={got}, 期望≈{expect:.0f}'


def test_rasterizer_covers_every_column_between_extremes():
    """轮廓在 Z 上连续投影 → 两端墨迹列之间的每一列都必须有墨。"""
    case = synth.render(synth.Spec(px_per_mm=10.0, line_width=2))
    cols = np.where(case.mask.any(axis=0))[0]
    missing = [c for c in range(cols.min(), cols.max() + 1) if not case.mask[:, c].any()]
    assert missing == [], f'{len(missing)} 列缺墨，例如 {missing[:10]}'


@pytest.mark.parametrize('line_width', [1, 2, 3])
def test_rasterizer_stroke_width_is_isotropic(line_width):
    """圆盘笔刷应为各向同性：水平段厚度应等于设定线宽（±1 px）。"""
    case = synth.render(synth.Spec(px_per_mm=10.0, line_width=line_width))
    col = _col_of_z(case, -8.0)
    rows = np.where(case.mask[:, col])[0]
    thickness = len(rows)
    assert abs(thickness - line_width) <= 1, f'实测厚度 {thickness}, 期望 {line_width}'


def test_rasterizer_min_x_is_half_linewidth_below_axis():
    """轴线处的墨迹应低于 X=0 半个线宽（笔画有宽度，这是物理事实不是缺陷）。"""
    case = synth.render(synth.Spec(px_per_mm=10.0, line_width=2))
    rows = np.where(case.mask.any(axis=1))[0]
    x_min = _x_of_row(case, rows.max() + 1)       # 最下一行墨迹的下沿
    assert -0.25 <= x_min <= 0.05, x_min


def test_render_is_deterministic():
    """NFR-3：同参数多次渲染必须逐位一致。"""
    spec = synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=1.0,
                      noise=0.01, fillet_r=5.0)
    a = synth.render(spec).mask
    b = synth.render(spec).mask
    assert np.array_equal(a, b)


def test_render_noise_is_reproducible_and_off_by_default():
    clean = synth.render(synth.Spec(noise=0.0)).mask
    noisy = synth.render(synth.Spec(noise=0.02)).mask
    assert not np.array_equal(clean, noisy)
    assert np.array_equal(noisy, synth.render(synth.Spec(noise=0.02)).mask)


def test_case_roundtrip(tmp_path):
    case = synth.render(synth.Spec(px_per_mm=10.0, line_width=2, skew_deg=-2.0))
    synth.save_case(case, str(tmp_path))
    back = synth.load_case(str(tmp_path / (case.name + '.truth.json')))
    assert np.array_equal(back.mask, case.mask)
    assert back.spec == case.spec
    assert back.entities == case.entities


def test_matrix_sizes():
    assert len(synth.matrix()) == 324
    assert len(synth.dod_cases()) == 72
    names = [s.name for s in synth.dod_cases()]
    assert len(set(names)) == len(names), '用例名必须唯一（缓存键）'


# =============================================================================
# 第三部分：精度比对框架自验证（喂真值必须得零误差）
# =============================================================================

def _truth():
    return synth.build_profile(5.0)[0]


def test_metrics_perfect_self_match_is_zero():
    """把真值当作"提取结果"喂回去，6 项指标必须全为零/满分。"""
    t = _truth()
    r = metrics.compare(t, [dict(e) for e in t])
    assert r['ok']
    assert r['endpoint_err'] < 1e-9
    assert r['hausdorff'] < 1e-9
    assert r['center_err'] < 1e-9
    assert r['radius_err'] < 1e-9
    assert r['type_acc'] == 1.0
    assert r['arc_recall'] == 1.0
    ok, _ = metrics.verdict(r)
    assert ok


def _shift_ents(ents, dz=0.0, dx=0.0):
    out = []
    for e in ents:
        f = dict(e)
        f['start'] = (e['start'][0] + dz, e['start'][1] + dx)
        f['end'] = (e['end'][0] + dz, e['end'][1] + dx)
        if e['type'] == 'arc':
            f['center'] = (e['center'][0] + dz, e['center'][1] + dx)
        out.append(f)
    return out


def test_metrics_z_translation_is_gauge_not_error():
    """Z 整体平移 3 mm 是坐标规范自由度，不得算作误差。"""
    t = _truth()
    r = metrics.compare(t, _shift_ents(t, dz=3.0))
    assert r['ok']
    assert r['endpoint_err'] < 1e-6, r['endpoint_err']
    assert r['hausdorff'] < 1e-6, r['hausdorff']
    assert abs(r['center_err']) < 1e-6
    # 不对齐时应当暴露出这 3 mm
    r0 = metrics.compare(t, _shift_ents(t, dz=3.0), align='none')
    assert abs(r0['gauge_shift_mm']) < 1e-12
    assert r0['hausdorff'] > 2.5, r0['hausdorff']


def test_metrics_x_translation_is_error():
    """X 方向没有规范自由度（X=0 是物理轴线），平移必须被计为误差。"""
    t = _truth()
    r = metrics.compare(t, _shift_ents(t, dx=0.35))
    assert abs(r['endpoint_err'] - 0.35) < 0.02, r['endpoint_err']
    assert abs(r['hausdorff'] - 0.35) < 0.02, r['hausdorff']
    ok, _ = metrics.verdict(r)
    assert not ok


def test_metrics_detects_radius_error_linearly():
    t = _truth()
    ext = [dict(e) for e in t]
    ext[2]['radius'] = t[2]['radius'] + 0.05
    r = metrics.compare(t, ext)
    assert abs(r['radius_err'] - 0.05) < 1e-9, r['radius_err']
    assert r['center_err'] < 1e-9
    assert r['strict_radius_ok'] is True      # 0.05 < 0.1 严档


def test_metrics_detects_center_error():
    t = _truth()
    ext = [dict(e) for e in t]
    for idx in (2,):
        c = t[idx]['center']
        ext[idx]['center'] = (c[0] + 0.12, c[1] - 0.16)
    r = metrics.compare(t, ext)
    assert abs(r['center_err'] - 0.2) < 0.01, r['center_err']
    assert r['radius_err'] == 0.0


def test_metrics_detects_arc_downgraded_to_line():
    """真值圆弧被识别成折线 —— 必须计入 type_acc 与 arc_recall 的下降。"""
    t = _truth()
    ext = [dict(e) for e in t]
    ext[2] = {'type': 'line', 'start': t[2]['start'], 'end': t[2]['end']}
    r = metrics.compare(t, ext)
    assert r['arc_recall'] == 0.5
    assert abs(r['type_acc'] - 8.0 / 9.0) < 1e-9
    ok, _ = metrics.verdict(r)
    assert not ok


def test_metrics_reports_empty_extraction_as_failure():
    r = metrics.compare(_truth(), [])
    assert not r['ok']
    assert '实体数为 0' in r['failure']
    ok, rows = metrics.verdict(r)
    assert not ok
    assert all(not row[3] for row in rows)


def test_metrics_detects_spurious_extra_segment():
    t = _truth()
    ext = [dict(e) for e in t] + [{'type': 'line', 'start': (-90.0, 0.0),
                                   'end': (-90.0, 5.0)}]
    r = metrics.compare(t, ext)
    assert r['n_spurious_ext'] == 1
    assert r['n_matched'] == 9


def test_metrics_endpoint_error_uses_worst_case():
    """单点毛刺必须被最大误差抓到，不能被平均掩盖。"""
    t = _truth()
    ext = [dict(e) for e in t]
    p = ext[-1]['end']
    ext[-1] = dict(ext[-1])
    ext[-1]['end'] = (p[0] - 0.19, p[1])
    r = metrics.compare(t, ext)
    assert 0.15 < r['endpoint_err'] < 0.25, r['endpoint_err']
    assert r['endpoint_rms'] < r['endpoint_err']


def test_aggregate_uses_worst_case():
    t = _truth()
    good = metrics.compare(t, [dict(e) for e in t])
    bad = metrics.compare(t, _shift_ents(t, dx=0.5))
    agg = metrics.aggregate([good, bad])
    assert agg['n_total'] == 2 and agg['n_ok'] == 2
    assert abs(agg['endpoint_err_worst'] - 0.5) < 0.02
    assert agg['type_acc_worst'] == 1.0


# =============================================================================
# 第四部分：几何核单元测试
# =============================================================================

def test_arc_sweep_and_midpoint_ccw():
    c = (0.0, 0.0)
    p1, p2 = (1.0, 0.0), (0.0, 1.0)
    assert abs(G.arc_sweep(c, p1, p2, True) - math.pi / 2) < 1e-12
    assert abs(G.arc_sweep(c, p1, p2, False) - 3 * math.pi / 2) < 1e-12
    mid = G.arc_midpoint(c, p1, p2, True)
    assert abs(mid[0] - math.cos(math.pi / 4)) < 1e-12
    assert abs(mid[1] - math.sin(math.pi / 4)) < 1e-12
    # 反向时必须走另一侧（270°）
    mid2 = G.arc_midpoint(c, p1, p2, False)
    assert mid2[1] < 0


def test_kasa_recovers_exact_circle():
    ang = np.linspace(0.4, 2.2, 40)
    pts = [(3.0 + 7.5 * math.cos(a), -2.0 + 7.5 * math.sin(a)) for a in ang]
    c, r = G.fit_circle_kasa(pts)
    assert abs(c[0] - 3.0) < 1e-9 and abs(c[1] + 2.0) < 1e-9
    assert abs(r - 7.5) < 1e-9


def test_geometric_refinement_beats_kasa_on_noisy_arc():
    rng = np.random.default_rng(7)
    ang = np.linspace(0.0, math.pi, 120)
    truth_c, truth_r = (-4.0, 11.0), 6.0
    pts = np.array([(truth_c[0] + truth_r * math.cos(a), truth_c[1] + truth_r * math.sin(a))
                    for a in ang])
    pts = pts + rng.normal(0, 0.35, pts.shape)
    c0, r0 = G.fit_circle_kasa(pts)
    e0 = math.hypot(c0[0] - truth_c[0], c0[1] - truth_c[1]) + abs(r0 - truth_r)
    c1, r1 = G.fit_circle_geometric(pts, c0, r0)
    e1 = math.hypot(c1[0] - truth_c[0], c1[1] - truth_c[1]) + abs(r1 - truth_r)
    assert e1 <= e0 + 1e-9, (e0, e1)


def test_fit_line_handles_vertical_segment():
    pts = [(5.0, y) for y in np.linspace(-3, 9, 25)]
    f = G.fit_line(pts)
    assert f['rms'] < 1e-9
    assert abs(abs(f['dirn'][1]) - 1.0) < 1e-9      # 方向沿 X（竖直）
    assert abs(f['point'][0] - 5.0) < 1e-9


def test_junction_point_line_line():
    """结合点必须取自拟合出的**直线**交点，而不是两个原始端点的中点。"""
    a = {'type': 'line', 'start': (0.0, 0.0), 'end': (10.0, 0.0),
         '_point': (5.0, 0.0), '_dirn': (1.0, 0.0)}
    b = {'type': 'line', 'start': (10.15, 0.02), 'end': (9.95, 5.0),
         '_point': (10.0, 2.5), '_dirn': (0.0, 1.0)}
    p = G.junction_point(a, b)
    assert abs(p[0] - 10.0) < 1e-6 and abs(p[1]) < 1e-6


def test_junction_point_line_arc_picks_correct_side():
    arc = {'type': 'arc', 'start': (0.0, 5.0), 'end': (5.0, 0.0),
           'center': (0.0, 0.0), 'radius': 5.0, 'ccw': False}
    line = {'type': 'line', 'start': (-3.0, 0.0), 'end': (3.0, 0.0),
            '_point': (0.0, 0.0), '_dirn': (1.0, 0.0)}
    p = G.junction_point(line, arc, hint=(5.0, 0.2))
    assert abs(p[0] - 5.0) < 1e-6 and abs(p[1]) < 1e-6


def test_metrics_rejects_mismatched_arc_pairing():
    """★ 度量框架的反例守护：错配的圆弧配对**不得**靠"误差恰好抵消"蒙混过关。

    无上限的贪心匹配 + 在同一配对集合上算误差 = 自洽闭环：
    两条同半径 R5 圆角的圆心互换后，提取的"该弧圆心"恰好等于真值的"该弧圆心"，
    于是 center_err / radius_err / type_acc / arc_recall 会**同时满分**，
    而几何其实错了 8.57 mm（只有豪斯多夫距离抓到）。
    加了配对距离上限后，这种错配必须被判为"未匹配"。
    """
    t = _truth()
    ext = [dict(e) for e in t]
    # 交换两条圆弧的圆心（同半径，故半径误差仍为 0）
    c2, c4 = t[2]['center'], t[4]['center']
    ext[2] = dict(ext[2], center=c4, start=(c4[0] + 5.0, c4[1]), end=(c4[0], c4[1] - 5.0))
    ext[4] = dict(ext[4], center=c2, start=(c2[0], c2[1] - 5.0),
                  end=(c2[0] - 5.0, c2[1]))
    r = metrics.compare(t, ext)
    # 形状被破坏 → 豪斯多夫必须抓到
    assert r['hausdorff'] > 1.0, r['hausdorff']
    # 配对距离上限必须拦住错配：至少有一项圆弧指标被判失败
    arc_failed = (not (r['center_err'] <= 0.2)
                  or not (r['radius_err'] <= 0.2)
                  or not (r['type_acc'] >= 0.9))
    assert arc_failed, {
        'center_err': r['center_err'], 'radius_err': r['radius_err'],
        'type_acc': r['type_acc'], 'worst_pair_dist': r.get('worst_pair_dist')}
    ok, _ = metrics.verdict(r)
    assert not ok


def test_metrics_no_matched_arc_is_a_failure_not_a_free_pass():
    """★ 空集不得拿满分：真值有圆弧却一段都没配上时，圆心/半径误差判 nan（失败）。"""
    t = _truth()
    ext = [dict(e) for e in t]
    for i in (2, 4):                      # 两条圆弧降级为弦线
        ext[i] = {'type': 'line', 'start': t[i]['start'], 'end': t[i]['end']}
    r = metrics.compare(t, ext)
    assert r['arc_recall'] == 0.0
    assert r['center_err'] != r['center_err'], 'center_err 应为 nan'
    assert r['radius_err'] != r['radius_err'], 'radius_err 应为 nan'
    ok, rows = metrics.verdict(r)
    assert not ok
    for name, val, _tgt, good in rows:
        if name in ('center_err', 'radius_err'):
            assert not good, f'{name} 不该判通过'


def test_metrics_pair_distance_cap_is_env_independent():
    """配对距离上限是显式的、有默认值的，不会随图尺寸静默失效。"""
    assert metrics.DEFAULT_MAX_PAIR_DIST > 0
    t = _truth()
    far = [{'type': 'line', 'start': (100.0, 100.0), 'end': (110.0, 100.0)}]
    pairs, un_t, un_e = metrics.match_entities(t, far, dz=0.0)
    assert pairs == [] and len(un_t) == len(t) and un_e == [0]


def test_axis_endpoints_are_pinned_to_the_axis():
    """车削上半部外轮廓的首末点必在轴线上；骨架端点的系统性 X 抬升必须被钉掉。

    实测该项在 w=3 @10px/mm 下曾达 0.14 mm（占 0.2 mm 预算 70%）。
    """
    from i2g import snap
    ents = [
        {'type': 'line', 'start': (0.0, 0.14), 'end': (0.0, 20.0)},
        {'type': 'line', 'start': (0.0, 20.0), 'end': (-30.0, 20.0)},
        {'type': 'line', 'start': (-30.0, 20.0), 'end': (-30.0, 0.09)},
    ]
    out, n = snap.snap_axis_endpoints(ents)
    assert n == 2
    assert out[0]['start'][1] == 0.0
    assert out[-1]['end'][1] == 0.0
    # 首实体仍是竖直端面（方向未被破坏）
    assert abs(out[0]['_dirn'][1] - 1.0) < 1e-9


def test_axis_endpoint_pinning_does_not_touch_far_from_axis():
    """轮廓本身不到轴线时不得被误钉。"""
    from i2g import snap
    ents = [
        {'type': 'line', 'start': (0.0, 8.0), 'end': (0.0, 20.0)},
        {'type': 'line', 'start': (0.0, 20.0), 'end': (-30.0, 20.0)},
    ]
    out, n = snap.snap_axis_endpoints(ents)
    assert n == 0 and out[0]['start'][1] == 8.0


def test_z_monotonicity_guard_rejects_folded_contour():
    """★ Z 折返必须报错 —— 否则会静默输出"把零件切两遍"的程序。

    评审反例：折返链在 verify_continuity 下是"严格连续"的，退出码 0、
    8 项契约校验过 7 项、C4 重复段也抓不到（反向副本端点差 0.1 mm）。
    """
    from i2g import snap
    from i2g.errors import ContourDirectionError
    folded = [
        {'type': 'line', 'start': (0.0, 0.0), 'end': (0.0, 20.0)},
        {'type': 'line', 'start': (0.0, 20.0), 'end': (-30.0, 20.0)},
        {'type': 'line', 'start': (-30.0, 20.0), 'end': (-30.0, 4.0)},
        # Z 回调（折返）
        {'type': 'line', 'start': (-30.0, 4.0), 'end': (-10.0, 10.0)},
        {'type': 'line', 'start': (-10.0, 10.0), 'end': (-10.0, 0.0)},
    ]
    with pytest.raises(ContourDirectionError) as e:
        snap.verify_z_monotonic(folded)
    assert 'Z 非单调' in str(e.value)
    # 单调件必须通过
    good = [
        {'type': 'line', 'start': (0.0, 0.0), 'end': (0.0, 20.0)},
        {'type': 'line', 'start': (0.0, 20.0), 'end': (-30.0, 20.0)},
        {'type': 'line', 'start': (-30.0, 20.0), 'end': (-30.0, 0.0)},
    ]
    assert snap.verify_z_monotonic(good) <= 1e-12


def test_resample_polyline_preserves_endpoints_and_spacing():
    pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    rs = G.resample_polyline(pts, 0.5)
    assert rs[0] == (0.0, 0.0)
    assert abs(rs[-1][0] - 10.0) < 1e-9 and abs(rs[-1][1] - 10.0) < 1e-9
    for i in range(1, len(rs)):
        assert abs(G.dist(rs[i - 1], rs[i]) - 0.5) < 1e-6


def test_entity_x_extremes_for_arc_uses_geometry_not_endpoints():
    """半圆的 X 最大/最小值必须由几何求出，不能只看两端点。

    注意坐标含义：本项目的点是 (Z, X)，X 是**第二**坐标。
    圆心原点、两端 (±3,0)、ccw=False 的弧走上半周，X ∈ [0, 3]。
    """
    arc = {'type': 'arc', 'start': (-3.0, 0.0), 'end': (3.0, 0.0),
           'center': (0.0, 0.0), 'radius': 3.0, 'ccw': False}
    lo, hi = G.entity_x_extremes(arc)
    assert abs(hi - 3.0) < 1e-6
    assert abs(lo - 0.0) < 1e-6
    # 反向的半周则完全在轴线以下 —— 用来证明极值确实取自几何而非端点
    arc_below = dict(arc, ccw=True)
    lo2, hi2 = G.entity_x_extremes(arc_below)
    assert hi2 < 1e-9 and abs(lo2 + 3.0) < 1e-6


def test_junction_point_unpacks_entity_primitive_correctly():
    """junction_point 的返回值契约：3 元组 (kind, point/center, dirn/radius)。"""
    line = {'type': 'line', 'start': (0.0, 0.0), 'end': (4.0, 0.0)}
    kind, p, d = G._entity_primitive(line)
    assert kind == 'line' and p == line['start'] and d == (1.0, 0.0)
    arc = {'type': 'arc', 'start': (0.0, 5.0), 'end': (5.0, 0.0),
           'center': (0.0, 0.0), 'radius': 5.0, 'ccw': False}
    kind2, c2, r2 = G._entity_primitive(arc)
    assert kind2 == 'arc' and c2 == (0.0, 0.0) and r2 == 5.0
