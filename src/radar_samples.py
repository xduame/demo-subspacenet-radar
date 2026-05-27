"""
RadarSamples - 雷达信号源,扩展 SubspaceNet 的 Samples 类
=========================================================
只重写 signal_creation() 方法,从雷达辐射源库生成 PDW + I/Q 信号。
不重写 samples_creation(),不做任何门控/检测处理。
"""

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from src.signal_creation import Samples
from src.system_model import SystemModelParams


# ===================================================================
# 雷达辐射源库加载(从 Excel 解析,缓存为模块级字典)
# ===================================================================
_LIB_CACHE = None


def _is_na(value):
    if pd is not None:
        return pd.isna(value)
    return value is None or value == ''


def _row_value(row, index):
    try:
        return row[index]
    except (IndexError, KeyError):
        return None


def _column_index(cell_ref):
    letters = re.match(r'[A-Z]+', cell_ref).group(0)
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - ord('A') + 1
    return index - 1


def _load_shared_strings(xlsx_file, ns):
    if 'xl/sharedStrings.xml' not in xlsx_file.namelist():
        return []
    root = ET.fromstring(xlsx_file.read('xl/sharedStrings.xml'))
    return [
        ''.join(text.text or '' for text in item.findall('.//m:t', ns))
        for item in root.findall('m:si', ns)
    ]


def _worksheet_path(xlsx_file, sheet):
    ns = {
        'm': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    }
    workbook = ET.fromstring(xlsx_file.read('xl/workbook.xml'))
    sheets = workbook.findall('m:sheets/m:sheet', ns)
    target = next((s for s in sheets if s.attrib.get('name') == sheet), sheets[0])
    rel_id = target.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
    rels = ET.fromstring(xlsx_file.read('xl/_rels/workbook.xml.rels'))
    for rel in rels:
        if rel.attrib['Id'] == rel_id:
            return 'xl/' + rel.attrib['Target']
    raise ValueError(f'Worksheet {sheet} not found in workbook relationships.')


def _read_excel_rows_stdlib(xlsx_path, sheet):
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(xlsx_path) as xlsx_file:
        shared_strings = _load_shared_strings(xlsx_file, ns)
        worksheet = ET.fromstring(xlsx_file.read(_worksheet_path(xlsx_file, sheet)))

    rows = []
    for row in worksheet.findall('.//m:sheetData/m:row', ns):
        values = []
        for cell in row.findall('m:c', ns):
            cell_ref = cell.attrib.get('r', 'A1')
            column = _column_index(cell_ref)
            while len(values) <= column:
                values.append(None)
            value_node = cell.find('m:v', ns)
            value = None if value_node is None else value_node.text
            if cell.attrib.get('t') == 's' and value is not None:
                value = shared_strings[int(value)]
            elif cell.attrib.get('t') == 'inlineStr':
                value = ''.join(
                    text.text or '' for text in cell.findall('.//m:t', ns)
                )
            values[column] = value
        rows.append(values)
    return rows


def _parse_range(s):
    """把 '5-10' 这种字符串解析成 (5.0, 10.0)"""
    if _is_na(s) or not isinstance(s, str):
        return None
    m = re.match(r'\s*([-\d.]+)\s*-\s*([-\d.]+)', s.strip())
    return (float(m.group(1)), float(m.group(2))) if m else None


def load_emitter_library(xlsx_path, sheet='空中平台追踪'):
    """从 Excel 加载辐射源库 - 5 型号 × 6 模式 = 30 个工作模式。"""
    global _LIB_CACHE
    if _LIB_CACHE is not None:
        return _LIB_CACHE

    if pd is not None:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
        rows = (row for _, row in df.iterrows())
    else:
        rows = _read_excel_rows_stdlib(xlsx_path, sheet)

    library, current_model = {}, None
    valid_modes = {'VS', 'HRWS', 'MRWS', 'TASS', 'TAST', 'STT'}

    for row in rows:
        c0 = _row_value(row, 0)
        if _is_na(c0) or not isinstance(c0, str):
            continue
        if re.match(r'^型号\d+$', c0):
            current_model = c0
            library[current_model] = {}
        elif c0 in valid_modes and current_model is not None:
            library[current_model][c0] = dict(
                duration_s=_parse_range(_row_value(row, 1)),
                pri_us=_parse_range(_row_value(row, 2)),
                pri_mod=str(_row_value(row, 3)).strip() if not _is_na(_row_value(row, 3)) else '固定',
                duty=_parse_range(_row_value(row, 4)),
                rf_mhz=_parse_range(_row_value(row, 6)),
                rf_mod=str(_row_value(row, 7)).strip() if not _is_na(_row_value(row, 7)) else '固定',
                bw_mhz=_parse_range(_row_value(row, 9)),
            )
    _LIB_CACHE = library
    return library


# ===================================================================
# PDW 生成器
# ===================================================================
def _gen_pdw(params, n_pulses, rng):
    """根据辐射源参数生成 n_pulses 个脉冲的 PDW 序列。"""
    pri_lo, pri_hi = params['pri_us']
    pri_mod = params['pri_mod']
    if pri_mod == '固定':
        pri = np.full(n_pulses, rng.uniform(pri_lo, pri_hi))
    elif pri_mod == '组变':
        pri = np.repeat(rng.uniform(pri_lo, pri_hi, n_pulses // 16 + 1), 16)[:n_pulses]
    elif pri_mod == '组参':
        pri = np.empty(n_pulses)
        for i in range(0, n_pulses, 32):
            s = rng.uniform(pri_lo, pri_hi, 4)
            seg = np.tile(s, 8 + 1)[:min(32, n_pulses - i)]
            pri[i:i + len(seg)] = seg
    elif pri_mod == '参差':
        s = rng.uniform(pri_lo, pri_hi, 8)
        pri = np.tile(s, n_pulses // 8 + 1)[:n_pulses]
    else:
        pri = np.full(n_pulses, (pri_lo + pri_hi) / 2)

    toa = np.concatenate([[0.0], np.cumsum(pri[:-1])])
    duty = rng.uniform(*params['duty'], n_pulses)
    pw = duty * pri

    rf_lo, rf_hi = params['rf_mhz']
    if params['rf_mod'] == '脉组捷变':
        rf = np.repeat(rng.uniform(rf_lo, rf_hi, n_pulses // 32 + 1), 32)[:n_pulses]
    else:
        rf = np.full(n_pulses, rng.uniform(rf_lo, rf_hi))

    bw = np.full(n_pulses, rng.uniform(*params['bw_mhz'])) if params['bw_mhz'] else np.full(n_pulses, 0.0)
    return toa, pri, pw, rf, bw


# ===================================================================
# PDW → 基带 I/Q 波形合成(LFM 调制)
# ===================================================================
def _synthesize_iq(toa, pw, rf, bw, T, fs_mhz, rf_center_mhz):
    """把 PDW 序列合成为长度 T 的基带复信号。"""
    sig = np.zeros(T, dtype=complex)
    duration_us = T / fs_mhz
    for i in range(len(toa)):
        if toa[i] >= duration_us:
            break
        f0 = rf[i] - rf_center_mhz
        k = bw[i] / pw[i] if pw[i] > 0 else 0
        i0 = int(toa[i] * fs_mhz)
        i1 = min(int((toa[i] + pw[i]) * fs_mhz), T)
        if i1 <= i0:
            continue
        tau = np.arange(i1 - i0) / fs_mhz
        phase = 2 * np.pi * (f0 * tau + 0.5 * k * tau ** 2)
        sig[i0:i1] += np.exp(1j * phase)
    return sig


# ===================================================================
# RadarSamples 主类
# ===================================================================
class RadarSamples(Samples):
    """雷达信号源 - 替代默认的高斯信号。"""

    def __init__(self, system_model_params: SystemModelParams):
        super().__init__(system_model_params)
        self.lib_path = getattr(system_model_params, 'lib_path',
                                'data/data_manual.xlsx')
        self.library = load_emitter_library(self.lib_path)
        self.fs_mhz = getattr(system_model_params, 'fs_mhz', 200)
        self.rf_center_mhz = getattr(system_model_params, 'rf_center_mhz', 9000)
        self.modes_pool = getattr(system_model_params, 'modes_pool',
                                  ['VS', 'MRWS', 'TASS', 'TAST'])

    def signal_creation(self, signal_mean=0, signal_variance=1):
        """生成 M 个雷达辐射源的基带 I/Q 信号矩阵 S[M, T]。"""
        T = self.params.T
        M = self.params.M
        amplitude = 10 ** (self.params.snr / 10)
        rng = np.random.default_rng()
        models = list(self.library.keys())

        # 随机选 M 个 (型号, 模式) 组合
        chosen = []
        for _ in range(M):
            mdl = rng.choice(models)
            md = rng.choice(self.modes_pool)
            chosen.append((mdl, md))

        S = np.zeros((M, T), dtype=complex)
        approx_pulses = max(50, T // 20)

        if self.params.signal_nature == 'non-coherent':
            for m, (mdl, md) in enumerate(chosen):
                toa, pri, pw, rf, bw = _gen_pdw(self.library[mdl][md],
                                                approx_pulses, rng)
                sig = _synthesize_iq(toa, pw, rf, bw, T,
                                     self.fs_mhz, self.rf_center_mhz)
                S[m] = sig
        else:  # coherent: 所有源用同一个信号
            mdl, md = chosen[0]
            toa, pri, pw, rf, bw = _gen_pdw(self.library[mdl][md],
                                            approx_pulses, rng)
            sig = _synthesize_iq(toa, pw, rf, bw, T,
                                 self.fs_mhz, self.rf_center_mhz)
            for m in range(M):
                S[m] = sig

        S = amplitude * (np.sqrt(2) / 2) * np.sqrt(signal_variance) * S + signal_mean
        return S
