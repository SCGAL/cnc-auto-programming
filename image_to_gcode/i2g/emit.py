"""⑦ 输出：转 v2 实体格式 → 调用 ``../dxf_to_gcode_v2.py`` 的 ``generate()``。

**这是本项目最重要的架构决策**（立项文档 §3.2⑦）：复用已被 9/9 契约校验过的
代码生成器，把风险隔离在"图像 → 几何"这一段。本模块**不重写任何 G 代码逻辑**。

两处必须踩准的接口细节
----------------------
1. ``generate(ordered, args, safe_x, unit_note, x_diameter)`` 需要的是一个
   ``argparse.Namespace``，字段有 ``pulse/mode/feed/ik/header/x_diameter/
   cn_comments/tool/spindle/input/unit_scale``。这里用 ``SimpleNamespace`` 构造。
2. **不能再走 v2 的 ``transform()``**。v2 CLI 的 ``--turning`` 会做
   ``(-x, y)`` 的 Z 镜像；而图像流水线**直接**产出已镜像好的 (Z, X)
   （右端面 Z=0，向左 Z 递减，X >= 0 恒成立）。再过一次 ``transform()``
   就是二次镜像，零件会翻到另一边。因此这里只做实体格式转换与 ``generate()`` 调用。

实体格式（与 v2 契约一致）::

    {'type': 'line', 'start': (z, x), 'end': (z, x)}
    {'type': 'arc',  'start': (z, x), 'end': (z, x),
     'center': (cz, cx), 'radius': r, 'ccw': bool}

坐标为 (Z, X)，X 为**半径值**。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

from .errors import GeometryError

_HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(_HERE)                 # .../image_to_gcode
PROJECT_PARENT = os.path.dirname(PKG_ROOT)        # .../数控

_GENERATOR_NAME = 'dxf_to_gcode_v2'
_MOD = None


class GCodeGeneratorUnavailable(GeometryError):
    """找不到或无法加载上游 G 代码生成器。"""


def generator_candidates():
    return [
        os.path.join(PROJECT_PARENT, 'dxf_to_gcode_v2.py'),                    # ../dxf_to_gcode_v2.py
        os.path.join(PROJECT_PARENT, '..', '数控', 'dxf_to_gcode_v2.py'),      # 若本项目与 数控 平级
        os.path.join(PKG_ROOT, 'dxf_to_gcode_v2.py'),
    ]


def generator_path():
    for p in generator_candidates():
        p = os.path.abspath(p)
        if os.path.isfile(p):
            return p
    raise GCodeGeneratorUnavailable(
        '找不到上游 G 代码生成器 dxf_to_gcode_v2.py。已查找：\n  '
        + '\n  '.join(os.path.abspath(p) for p in generator_candidates()))


def load_generator():
    """按文件路径加载 ``dxf_to_gcode_v2`` 并缓存。

    该模块顶层 ``import ezdxf``，缺失时会 ``sys.exit(1)``；这里先探测依赖，
    再捕获 ``SystemExit``，把"依赖缺失"变成可读的错误而不是进程猝死。
    """
    global _MOD
    if _MOD is not None:
        return _MOD
    path = generator_path()
    if importlib.util.find_spec('ezdxf') is None:
        raise GCodeGeneratorUnavailable(
            f'复用 G 代码生成器需要 ezdxf（{os.path.basename(path)} 顶层 import ezdxf）。\n'
            '安装：python -m pip install --target _deps ezdxf '
            '-i https://pypi.tuna.tsinghua.edu.cn/simple')
    spec = importlib.util.spec_from_file_location('i2g_' + _GENERATOR_NAME, path)
    if spec is None or spec.loader is None:
        raise GCodeGeneratorUnavailable(f'无法加载生成器模块: {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit as e:                                   # pragma: no cover
        raise GCodeGeneratorUnavailable(
            f'加载 {path} 时其顶层代码退出（exit code {e.code}），通常仍是依赖缺失') from e
    for attr in ('generate', 'validate'):
        if not hasattr(mod, attr):
            raise GCodeGeneratorUnavailable(f'{path} 未提供 {attr}()')
    _MOD = mod
    return mod


def make_args(*, source_name='image', pulse=0.01, feed=100.0, mode='g90',
              ik='fanuc', header='basic', x_diameter=False, cn_comments=True,
              tool=None, spindle=None, unit_scale=1.0):
    """构造 v2 ``generate()`` 需要的参数对象。

    ⚠️ ``unit_scale`` 恒为 1.0。原因：本模块**不调用** v2 的 ``transform()``
    （那会二次镜像 Z），而 ``unit_scale`` 的坐标缩放只在 ``transform()`` 里发生；
    v2 的 ``generate()`` 只把 ``args.unit_scale`` 写进程序头注释。
    若允许传非 1.0，就会生成一份**头部声称已缩放、坐标其实没缩放**的程序 ——
    正是上游 F5/F6 类缺陷的翻版（评审实测确认：unit_scale=25.4 时坐标逐字未变）。
    图像流水线已在毫米域完成标定，本就无需单位缩放，故直接拒绝。
    """
    if pulse <= 0:
        raise GeometryError(f'--pulse 必须为正，收到 {pulse}')
    if abs(float(unit_scale) - 1.0) > 1e-12:
        raise GeometryError(
            f'unit_scale={unit_scale} 不受支持：图像流水线已在毫米域完成标定，'
            f'且 emit() 不走 v2 的 transform()，unit_scale 只会写进注释而不会缩放坐标。'
            f'若图纸单位不是毫米，请改用 --mm-per-px / --ref-diameter 完成标定。')
    return SimpleNamespace(
        input=source_name, pulse=float(pulse), feed=float(feed), mode=mode,
        ik=ik, header=header, x_diameter=bool(x_diameter),
        cn_comments=bool(cn_comments), tool=tool, spindle=spindle,
        unit_scale=1.0)


def to_v2_entities(ents_mm):
    """剥掉 ``_`` 开头的内部拟合参数，得到与 v2 契约完全一致的实体。"""
    out = []
    for e in ents_mm:
        d = {k: v for k, v in e.items() if not k.startswith('_')}
        if d.get('type') not in ('line', 'arc'):
            raise GeometryError(f'未知实体类型: {d.get("type")!r}')
        for key in ('start', 'end'):
            if key not in d:
                raise GeometryError(f'{d["type"]} 实体缺少 {key}')
        if d['type'] == 'arc':
            for key in ('center', 'radius', 'ccw'):
                if key not in d:
                    raise GeometryError(f'arc 实体缺少 {key}')
            if d['radius'] <= 0:
                raise GeometryError(f'arc 半径非正: {d["radius"]}')
        out.append(d)
    return out


def safe_x_default(ents_mm, clearance=2.0):
    xs = [p[1] for e in ents_mm for p in (e['start'], e['end'])]
    if not xs:
        raise GeometryError('实体列表为空，无法确定安全退刀 X')
    return max(xs) + clearance


def emit(ents_mm, *, source_name='image', pulse=0.01, feed=100.0, mode='g90',
         ik='fanuc', header='basic', x_diameter=False, cn_comments=True,
         tool=None, spindle=None, safe_x=None, unit_scale=1.0,
         start_at='z0'):
    """生成 G 代码。

    返回 dict: ``gcode``、``args``、``ordered``、``safe_x``、``report``、
    ``generator_path``、``n_moves``。
    """
    mod = load_generator()
    if not ents_mm:
        raise GeometryError('实体列表为空，拒绝生成 G 代码（禁止静默输出空程序）')
    ents = to_v2_entities(ents_mm)
    args = make_args(source_name=source_name, pulse=pulse, feed=feed, mode=mode,
                     ik=ik, header=header, x_diameter=x_diameter,
                     cn_comments=cn_comments, tool=tool, spindle=spindle,
                     unit_scale=unit_scale)
    sx = safe_x_default(ents) if safe_x is None else float(safe_x)
    max_x = max(p[1] for e in ents for p in (e['start'], e['end']))
    if sx <= max_x + 1e-9:
        # 安全退刀 X 是全流程唯一没有校验的机床安全参数：若低于轮廓最大 X，
        # 一旦链上有断点，generate() 会在该 X 高度上沿 Z 快速移动 → 直接扎进实体。
        # 评审实测：--safe-x 5（轮廓最大 X = 20）时 rc=0 且契约校验 C5 仍 8/8 全过。
        raise GeometryError(
            f'安全退刀 X 必须大于轮廓最大 X：给定 safe_x={sx:.3f}，'
            f'而轮廓最大 X={max_x:.3f}。否则断点处的快速移动会切进工件。')

    deduped = mod.dedupe(ents)
    ordered, breaks = (deduped, []) if start_at == 'none' else mod.chain(deduped, start_at)
    gcode = mod.generate(ordered, args, sx, 'mm', bool(x_diameter))
    report = mod.validate(ordered, breaks, args, len(ents))
    n_moves = sum(1 for line in gcode.splitlines()
                  if line[:3] in ('G00', 'G01', 'G02', 'G03'))
    return {'gcode': gcode, 'ordered': ordered, 'breaks': breaks, 'args': args,
            'safe_x': sx, 'report': report, 'n_moves': n_moves,
            'n_entities': len(ordered), 'n_dupes_removed': len(ents) - len(deduped),
            'generator_path': generator_path()}
