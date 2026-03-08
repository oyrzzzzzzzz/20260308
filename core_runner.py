# core_runner.py
# -*- coding: utf-8 -*-

import os
import glob
import json
import time
import math
import cmath
import random
import warnings
import traceback
import multiprocessing
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np

# xlwings optional (CLTC)
try:
    import xlwings as xw
except ImportError:
    xw = None

warnings.filterwarnings('ignore')

from core_kernels import (
    LUT_V_AXIS_SW_DEFAULT, LUT_T_AXIS_SW_DEFAULT, LUT_T_AXIS_COND_DEFAULT,
    LUT_I_AXIS_SW_MOS_DEFAULT, LUT_I_AXIS_SW_DIODE_DEFAULT,
    LUT_I_AXIS_COND_MOS_DEFAULT, LUT_I_AXIS_COND_SR_DEFAULT, LUT_I_AXIS_COND_DIODE_DEFAULT,
    TABLE_E_ON_DEFAULT, TABLE_E_OFF_DEFAULT, TABLE_E_RR_DEFAULT,
    TABLE_V_DROP_DEFAULT, TABLE_V_SR_DEFAULT, TABLE_V_DIODE_DEFAULT,
    worker_simulation_task
)

# =============================================================================
# Helpers: MOS neg-branch -> SR extrap if missing
# =============================================================================
def _build_sr_from_mos_linear_extrap_first2(i_axis_pos, v_drop_table):
    """
    用 MOS 正向导通曲线前两点(i[0],i[1])线性外推到负电流，
    取 abs 作为 SR(反向导通) Vf。返回 SR 的 (I_axis>=0, Vdrop(T,I)).
    """
    i_axis_pos = np.array(i_axis_pos, dtype=np.float64)
    v_drop_table = np.array(v_drop_table, dtype=np.float64)

    if i_axis_pos.size < 2:
        return None, None
    if v_drop_table.ndim != 2 or v_drop_table.shape[1] != i_axis_pos.size:
        return None, None

    I1 = float(i_axis_pos[0])
    I2 = float(i_axis_pos[1])
    if abs(I2 - I1) < 1e-12:
        return None, None

    sr_i = i_axis_pos.copy()
    Nt, Ni = v_drop_table.shape
    sr_v = np.zeros((Nt, Ni), dtype=np.float64)

    for ti in range(Nt):
        V1 = float(v_drop_table[ti, 0])
        V2 = float(v_drop_table[ti, 1])
        a = (V2 - V1) / (I2 - I1)
        b = V1 - a * I1
        Vneg = a * (-sr_i) + b
        sr_v[ti, :] = np.abs(Vneg)

    return sr_i, sr_v


def _strip_ns(root):
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]


def _txt_to_arr(text):
    if not text or not text.strip():
        return np.array([], dtype=np.float64)
    return np.array([float(x) for x in text.split()], dtype=np.float64)


def _parse_table_data(node, target_scale_base=1.0):
    """
    Read plexim-like Energy tables that may have scale attribute.
    target_scale_base=0.001 for mJ target.
    """
    xml_scale = float(node.get('scale', 1.0))
    conversion_factor = xml_scale / target_scale_base

    mat_temps = []
    for t_node in node.findall(".//Temperature"):
        v_nodes = t_node.findall('Voltage')
        mat_volts = []
        if v_nodes:
            for v_node in v_nodes:
                arr = _txt_to_arr(v_node.text)
                mat_volts.append(arr * conversion_factor)
        elif t_node.text and t_node.text.strip():
            single_row = _txt_to_arr(t_node.text)
            mat_volts.append(single_row * conversion_factor)
        mat_temps.append(np.array(mat_volts))
    return np.array(mat_temps) if mat_temps else None


def load_data_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    _strip_ns(root)

    new_data = {}
    new_data['IS_MOSFET'] = False

    new_data['R_TH'] = None
    new_data['HAS_THERMAL'] = False
    new_data['FOSTER_R'] = np.array([], dtype=np.float64)
    new_data['FOSTER_TAU'] = np.array([], dtype=np.float64)

    def _set_axis_if_better(key, candidate_arr):
        """
        轴选择策略：
        - 空 -> 用 candidate
        - candidate 更长 -> 用 candidate（避免被占位轴(单点/0轴)覆盖真实轴）
        """
        if candidate_arr is None:
            return
        cand = np.array(candidate_arr, dtype=np.float64)
        if cand.size == 0:
            return
        if key not in new_data:
            new_data[key] = cand
            return
        old = np.array(new_data.get(key), dtype=np.float64)
        if old.size == 0 or cand.size > old.size:
            new_data[key] = cand
            return

    for semi in root.findall(".//SemiconductorData"):
        s_type = (semi.get('type') or '')

        target_type = None
        if 'MOSFET' in s_type:
            target_type = 'IGBT'   # keep key names for switch channel
            new_data['IS_MOSFET'] = True
        elif 'IGBT' in s_type:
            target_type = 'IGBT'
        elif 'Diode' in s_type:
            target_type = 'Diode'
        if target_type is None:
            continue

        ton_node = semi.find('TurnOnLoss') or semi.find('TurnOn')
        toff_node = semi.find('TurnOffLoss') or semi.find('TurnOff')
        cond_node = semi.find('ConductionLoss')

        # --- TurnOn / Eon ---
        if ton_node is not None:
            v_node = ton_node.find('VoltageAxis')
            if v_node is not None:
                _set_axis_if_better('V_AXIS_SW', np.abs(_txt_to_arr(v_node.text)))

            t_axis_node = ton_node.find('TemperatureAxis')
            if t_axis_node is not None:
                _set_axis_if_better('T_AXIS_SW', _txt_to_arr(t_axis_node.text))

            if target_type == 'IGBT':
                i_axis_node = ton_node.find('CurrentAxis')
                if i_axis_node is not None:
                    new_data['IGBT_I_SW'] = _txt_to_arr(i_axis_node.text)

            e_node = ton_node.find('Energy')
            if e_node is not None:
                arr = _parse_table_data(e_node, target_scale_base=0.001)  # mJ
                if arr is not None:
                    new_data['IGBT_E_ON'] = arr

        # --- TurnOff / Eoff or Err ---
        if toff_node is not None:
            # NOTE:
            # Diode 模型常见 TurnOnLoss 是“占位单点轴(25°C/0V/0A)”，
            # 真实 RR 数据在 TurnOffLoss。这里必须允许 TurnOffLoss 的轴覆盖占位轴。
            v_node = toff_node.find('VoltageAxis')
            if v_node is not None:
                v_arr = np.abs(_txt_to_arr(v_node.text))
                if target_type == 'Diode':
                    new_data['V_AXIS_SW'] = v_arr
                else:
                    _set_axis_if_better('V_AXIS_SW', v_arr)

            t_axis_node = toff_node.find('TemperatureAxis')
            if t_axis_node is not None:
                t_arr = _txt_to_arr(t_axis_node.text)
                if target_type == 'Diode':
                    new_data['T_AXIS_SW'] = t_arr
                else:
                    _set_axis_if_better('T_AXIS_SW', t_arr)

            current_axis_off_node = toff_node.find('CurrentAxis')
            current_axis_off = _txt_to_arr(current_axis_off_node.text) if current_axis_off_node is not None else np.array([], dtype=np.float64)

            if target_type == 'IGBT':
                if 'IGBT_I_SW' not in new_data:
                    new_data['IGBT_I_SW'] = current_axis_off
            elif target_type == 'Diode':
                new_data['DIODE_I_SW'] = current_axis_off

            # 旧逻辑是“仅当 V_AXIS_SW 不存在才设置”，会导致被 TurnOnLoss 占位轴锁死。
            # 这里保留兼容：如果上面没拿到 axis，这里也不再额外覆盖。

            e_node = toff_node.find('Energy')
            if e_node is not None:
                arr = _parse_table_data(e_node, target_scale_base=0.001)
                if arr is not None:
                    if target_type == 'IGBT':
                        new_data['IGBT_E_OFF'] = arr
                    elif target_type == 'Diode':
                        new_data['DIODE_E_RR'] = arr

        # --- ConductionLoss / Vdrop ---
        if cond_node is not None:
            t_axis_node = cond_node.find('TemperatureAxis')
            if t_axis_node is not None:
                new_data['T_AXIS_COND'] = _txt_to_arr(t_axis_node.text)

            mat = []
            v_drop_node = cond_node.find('VoltageDrop')
            parent_node = v_drop_node if v_drop_node is not None else cond_node
            scale = float(parent_node.get('scale', 1.0))

            for t_node in parent_node.findall('Temperature'):
                row = _txt_to_arr(t_node.text)
                mat.append(row * scale)

            if target_type == 'IGBT':
                raw_i_axis_node = cond_node.find('CurrentAxis')
                raw_i_axis = _txt_to_arr(raw_i_axis_node.text) if raw_i_axis_node is not None else np.array([], dtype=np.float64)
                raw_v_drop = np.array(mat, dtype=np.float64)

                if new_data.get('IS_MOSFET', False) and np.any(raw_i_axis < 0):
                    # split: +I -> MOS forward; -I -> SR
                    mask_pos = raw_i_axis >= 0
                    new_data['IGBT_I_COND'] = raw_i_axis[mask_pos]
                    new_data['IGBT_V_DROP'] = raw_v_drop[:, mask_pos]

                    mask_neg = raw_i_axis <= 0
                    sr_i = np.abs(raw_i_axis[mask_neg])
                    sr_v = np.abs(raw_v_drop[:, mask_neg])
                    new_data['SR_I_COND'] = sr_i[::-1]
                    new_data['SR_V_DROP'] = sr_v[:, ::-1]
                    print(f"[XML Parser] MOS Split: FWD len={len(new_data['IGBT_I_COND'])}, SR len={len(new_data['SR_I_COND'])}")
                else:
                    # no neg axis: forward only, optionally extrapolate SR from first2
                    new_data['IGBT_I_COND'] = raw_i_axis
                    new_data['IGBT_V_DROP'] = raw_v_drop

                    if new_data.get('IS_MOSFET', False):
                        if ('SR_I_COND' not in new_data) and ('SR_V_DROP' not in new_data):
                            if raw_i_axis.size >= 2 and raw_v_drop.shape[1] == raw_i_axis.size:
                                sr_i, sr_v = _build_sr_from_mos_linear_extrap_first2(raw_i_axis, raw_v_drop)
                                if sr_i is not None:
                                    new_data['SR_I_COND'] = sr_i
                                    new_data['SR_V_DROP'] = sr_v
                                    print(f"[XML Parser] MOS Extrap(SR): len={len(sr_i)} using i[0],i[1]")

            elif target_type == 'Diode':
                i_axis_node = cond_node.find('CurrentAxis')
                new_data['DIODE_I_COND'] = _txt_to_arr(i_axis_node.text) if i_axis_node is not None else np.array([], dtype=np.float64)
                new_data['DIODE_V_DROP'] = np.array(mat, dtype=np.float64)

    # --- ThermalModel (Foster) ---
    thermal = root.find(".//ThermalModel")
    if thermal is not None:
        branch = thermal.find(".//Branch")
        btype = (branch.get('type') or '').lower() if branch is not None else ''
        if branch is not None and (('foster' in btype) or (btype == '')):
            r_list = []
            tau_list = []
            for el in branch.findall(".//RTauElement"):
                try:
                    r_list.append(float(el.get('R')))
                    tau_list.append(float(el.get('Tau')))
                except:
                    pass
            if len(r_list) > 0:
                rth = float(np.sum(r_list))
                if rth > 1e-6:
                    new_data['FOSTER_R'] = np.array(r_list, dtype=np.float64)
                    new_data['FOSTER_TAU'] = np.array(tau_list, dtype=np.float64)
                    new_data['R_TH'] = rth
                    new_data['HAS_THERMAL'] = True

    return new_data


def _align_3d_voltage(table_3d, v_axis_src, v_axis_tgt):
    """Align 3D voltage axis: table[T, V, I]."""
    if table_3d is None:
        return None
    if not hasattr(table_3d, 'ndim') or table_3d.ndim != 3:
        return table_3d

    vt = np.array(v_axis_tgt, dtype=np.float64).copy()
    if vt.size == 0:
        return table_3d
    vs = np.array(v_axis_src, dtype=np.float64).copy() if v_axis_src is not None else vt.copy()
    if vs.size == 0:
        vs = vt.copy()

    vs = np.abs(vs); vt = np.abs(vt)
    vs.sort(); vt.sort()

    Tn, Vn_src, In = table_3d.shape
    Vn_tgt = vt.size

    # Hardening: if axis length mismatches table dimension, rebuild a usable vs to avoid np.interp crash.
    # Typical root cause: Diode XML 的 TurnOnLoss 占位轴(单点)覆盖了 TurnOffLoss 的真实电压轴。
    if vs.size != Vn_src:
        if Vn_src == 1:
            vs = np.array([vt[0] if vt.size > 0 else 0.0], dtype=np.float64)
        else:
            if vt.size >= 2:
                vmin = float(vt[0]); vmax = float(vt[-1])
                if abs(vmax - vmin) < 1e-12:
                    vmax = vmin + 1e-9
                vs = np.linspace(vmin, vmax, Vn_src, dtype=np.float64)
            else:
                vs = np.linspace(0.0, 1e-9, Vn_src, dtype=np.float64)

    if Vn_src == Vn_tgt:
        return table_3d
    if Vn_src == 1:
        return np.repeat(table_3d, Vn_tgt, axis=1)
    if Vn_tgt == 1:
        idx = int(np.argmin(np.abs(vs - vt[0])))
        return table_3d[:, idx:idx + 1, :]

    out = np.zeros((Tn, Vn_tgt, In), dtype=np.float64)
    for ti in range(Tn):
        for ii in range(In):
            out[ti, :, ii] = np.interp(vt, vs, table_3d[ti, :, ii],
                                       left=table_3d[ti, 0, ii],
                                       right=table_3d[ti, -1, ii])
    return out


def _align_3d_temperature(table_3d, t_axis_src, t_axis_tgt):
    """Align 3D temperature axis: table[T, V, I] along axis 0 (T)."""
    if table_3d is None:
        return None
    if not hasattr(table_3d, 'ndim') or table_3d.ndim != 3:
        return table_3d

    tt = np.array(t_axis_tgt, dtype=np.float64)
    if tt.size == 0:
        return table_3d

    ts = np.array(t_axis_src, dtype=np.float64) if t_axis_src is not None else tt.copy()
    if ts.size == 0:
        ts = tt.copy()

    Tn_src, Vn, In = table_3d.shape
    Tn_tgt = tt.size

    # Hardening: mismatch length
    if ts.size != Tn_src:
        if Tn_src == 1:
            # single point in table, just pick first of target or default 25
            ts = np.array([tt[0] if tt.size > 0 else 25.0], dtype=np.float64)
        else:
            # spread
            start_t = tt[0] if tt.size > 0 else 25.0
            end_t = tt[-1] if tt.size > 0 else 150.0
            if abs(end_t - start_t) < 1e-9:
                end_t = start_t + 10.0
            ts = np.linspace(start_t, end_t, Tn_src, dtype=np.float64)

    if Tn_src == Tn_tgt:
        # Check if identical (optional optimization)
        # return table_3d
        pass

    if Tn_src == 1:
        # Broadcast single temp to all target temps
        return np.repeat(table_3d, Tn_tgt, axis=0)

    # Interpolate along axis 0
    out = np.zeros((Tn_tgt, Vn, In), dtype=np.float64)
    for vi in range(Vn):
        for ii in range(In):
            out[:, vi, ii] = np.interp(tt, ts, table_3d[:, vi, ii])
    return out


def _align_2d_temperature(table_2d, t_axis_src, t_axis_tgt):
    """Align 2D conduction table: table[T, I]."""
    if table_2d is None:
        return None
    if not hasattr(table_2d, 'ndim') or table_2d.ndim != 2:
        return table_2d

    tt = np.array(t_axis_tgt, dtype=np.float64)
    if tt.size == 0:
        return table_2d

    ts = np.array(t_axis_src, dtype=np.float64) if t_axis_src is not None else tt.copy()
    if ts.size == 0:
        ts = tt.copy()

    Tn_src, In = table_2d.shape
    Tn_tgt = tt.size

    if Tn_src == Tn_tgt:
        return table_2d

    if Tn_src == 1:
        return np.repeat(table_2d, Tn_tgt, axis=0)

    out = np.zeros((Tn_tgt, In), dtype=np.float64)
    for ii in range(In):
        out[:, ii] = np.interp(tt, ts, table_2d[:, ii])
    return out


# =============================================================================
# Main logic
# =============================================================================
def run_simulation_logic(cfg, logger_func):
    logger_func("=" * 60)
    logger_func("开始运行集成仿真 (Integrated Simulation)")
    logger_func(f"模式: {'TOP' if cfg['TARGET_IDX'] == 1 else 'BOTTOM'} Switch")
    logger_func(f"并行核数: {cfg['CPU_CORES']}")
    logger_func("=" * 60)

    start_time = time.time()

    # Defaults
    my_v_axis_sw = LUT_V_AXIS_SW_DEFAULT
    my_t_sw = LUT_T_AXIS_SW_DEFAULT
    my_t_cond = LUT_T_AXIS_COND_DEFAULT

    my_i_sw_mos = LUT_I_AXIS_SW_MOS_DEFAULT
    my_e_on = TABLE_E_ON_DEFAULT
    my_e_off = TABLE_E_OFF_DEFAULT

    my_i_cond_mos = LUT_I_AXIS_COND_MOS_DEFAULT
    my_v_drop_mos = TABLE_V_DROP_DEFAULT

    # NEW: SR conduction from MOS XML
    my_i_cond_sr = LUT_I_AXIS_COND_SR_DEFAULT
    my_v_drop_sr = TABLE_V_SR_DEFAULT

    # NEW: Deadtime diode (from diode XML when MOS deadtime enabled, or always in IGBT+Diode)
    my_i_sw_diode_dt = LUT_I_AXIS_SW_DIODE_DEFAULT
    my_e_rr_dt = TABLE_E_RR_DEFAULT
    my_i_cond_diode_dt = LUT_I_AXIS_COND_DIODE_DEFAULT
    my_v_diode_dt = TABLE_V_DIODE_DEFAULT

    local_rth_igbt = float(cfg['RTH_IGBT'])
    local_rth_diode = float(cfg['RTH_DIODE'])
    is_mosfet_global = False

    def _override_rth_if_valid(vals, current_rth, name):
        r = vals.get('R_TH', None)
        has_thermal = vals.get('HAS_THERMAL', False)
        if (not has_thermal) or (r is None):
            logger_func(f" -> {name} 未解析到有效 ThermalModel，保持默认 Rth={current_rth:.6f} K/W")
            return current_rth
        try:
            r = float(r)
        except:
            logger_func(f" -> {name} Rth 解析异常，保持默认 Rth={current_rth:.6f} K/W")
            return current_rth
        if r <= 1e-6:
            logger_func(f" -> {name} Rth={r:.6g} 视为无效(<=1e-6)，保持默认 Rth={current_rth:.6f} K/W")
            return current_rth
        logger_func(f" -> {name} ThermalModel 有效，使用 XML Rth={r:.6f} K/W")
        return r

    # XML paths
    xml_switch_path = os.path.join(cfg['DIR_XML'], cfg['FILE_XML_IGBT'])
    xml_diode_path = os.path.join(cfg['DIR_XML'], cfg['FILE_XML_DIODE'])

    # ------------------------
    # Load switch XML (IGBT or MOS)
    # ------------------------
    logger_func(f"[XML] 正在加载 Switch: {os.path.basename(xml_switch_path)}")
    try:
        vals_sw = load_data_from_xml(xml_switch_path)
        if vals_sw.get('IS_MOSFET', False):
            is_mosfet_global = True
            logger_func(" -> 检测到 MOSFET 类型。")

        if 'V_AXIS_SW' in vals_sw:
            my_v_axis_sw = np.array(vals_sw['V_AXIS_SW'], dtype=np.float64)
        if 'T_AXIS_SW' in vals_sw:
            my_t_sw = np.array(vals_sw['T_AXIS_SW'], dtype=np.float64)
        if 'T_AXIS_COND' in vals_sw:
            my_t_cond = np.array(vals_sw['T_AXIS_COND'], dtype=np.float64)

        if 'IGBT_I_SW' in vals_sw:
            my_i_sw_mos = np.array(vals_sw['IGBT_I_SW'], dtype=np.float64)
        if 'IGBT_E_ON' in vals_sw:
            v_src = vals_sw.get('V_AXIS_SW', my_v_axis_sw)
            my_e_on = _align_3d_voltage(np.array(vals_sw['IGBT_E_ON'], dtype=np.float64), v_src, my_v_axis_sw)
        if 'IGBT_E_OFF' in vals_sw:
            v_src = vals_sw.get('V_AXIS_SW', my_v_axis_sw)
            my_e_off = _align_3d_voltage(np.array(vals_sw['IGBT_E_OFF'], dtype=np.float64), v_src, my_v_axis_sw)

        if 'IGBT_I_COND' in vals_sw:
            my_i_cond_mos = np.array(vals_sw['IGBT_I_COND'], dtype=np.float64)
        if 'IGBT_V_DROP' in vals_sw:
            # align conduction table temperature to main axis
            t_src_cond = vals_sw.get('T_AXIS_COND', my_t_cond)
            my_v_drop_mos = _align_2d_temperature(np.array(vals_sw['IGBT_V_DROP'], dtype=np.float64), t_src_cond, my_t_cond)

        # SR conduction from MOS XML (if MOS)
        if is_mosfet_global and ('SR_I_COND' in vals_sw) and ('SR_V_DROP' in vals_sw):
            my_i_cond_sr = np.array(vals_sw['SR_I_COND'], dtype=np.float64)
            # align SR temperature
            t_src_cond = vals_sw.get('T_AXIS_COND', my_t_cond)
            my_v_drop_sr = _align_2d_temperature(np.array(vals_sw['SR_V_DROP'], dtype=np.float64), t_src_cond, my_t_cond)
            logger_func(" -> MOS 同步整流(SR)导通：使用 MOS XML 负电流段(拆分/外推)")

        local_rth_igbt = _override_rth_if_valid(vals_sw, local_rth_igbt, "Switch(IGBT/MOS)")

    except Exception as e:
        logger_func(f"!!! Switch XML 解析失败: {e}")
        traceback.print_exc()
        logger_func("将使用硬编码默认数据。")
        vals_sw = {}

    # ------------------------
    # Decide diode data source
    # ------------------------
    enable_deadtime_cfg = bool(cfg.get('ENABLE_DEADTIME', False))
    if is_mosfet_global:
        # MOS mode:
        # - SR conduction from MOS XML (already loaded)
        # - deadtime diode (cond + RR) from diode XML ONLY when enable_deadtime
        if enable_deadtime_cfg:
            logger_func(f"[XML] MOS 死区已启用，加载 Deadtime Diode XML: {os.path.basename(xml_diode_path)}")
            try:
                vals_diode = load_data_from_xml(xml_diode_path)

                if 'DIODE_I_SW' in vals_diode:
                    my_i_sw_diode_dt = np.array(vals_diode['DIODE_I_SW'], dtype=np.float64)
                if 'DIODE_E_RR' in vals_diode:
                    v_src = vals_diode.get('V_AXIS_SW', my_v_axis_sw)
                    tmp_rr = _align_3d_voltage(np.array(vals_diode['DIODE_E_RR'], dtype=np.float64), v_src, my_v_axis_sw)
                    # NEW: align temperature axis to IGBT
                    t_src = vals_diode.get('T_AXIS_SW', my_t_sw)
                    my_e_rr_dt = _align_3d_temperature(tmp_rr, t_src, my_t_sw)

                if 'DIODE_I_COND' in vals_diode:
                    my_i_cond_diode_dt = np.array(vals_diode['DIODE_I_COND'], dtype=np.float64)
                if 'DIODE_V_DROP' in vals_diode:
                    # align diode conduction temperature
                    t_src_cond = vals_diode.get('T_AXIS_COND', my_t_cond)
                    my_v_diode_dt = _align_2d_temperature(np.array(vals_diode['DIODE_V_DROP'], dtype=np.float64), t_src_cond, my_t_cond)

                # thermal logic
                if bool(cfg.get('RTH_DIODE_FOLLOW_MOS', False)):
                    local_rth_diode = local_rth_igbt
                    logger_func(" -> MOS Coupled: Diode Rth 强制跟随 MOS (Single-Die 模型)")
                else:
                    local_rth_diode = _override_rth_if_valid(vals_diode, local_rth_diode, "Diode")
                    logger_func(f" -> MOS Decoupled: Diode 使用独立热阻 = {local_rth_diode:.6f} K/W")

            except Exception as e:
                logger_func(f"!!! Deadtime Diode XML 解析失败: {e}")
                traceback.print_exc()
                logger_func(" -> 将无法使用 diode XML 做死区损耗（RR/死区续流导通可能为0）。")
                my_e_rr_dt = np.zeros_like(my_e_on)
        else:
            logger_func("[MOS] 死区未启用：不加载二极管XML，RR 默认置零。")
            my_e_rr_dt = np.zeros_like(my_e_on)

        # safety: SR tables must exist; else fallback to MOS forward
        if my_i_cond_sr.size == 0 or my_v_drop_sr.size == 0:
            logger_func("!!! 警告：MOS 未获得 SR 导通数据(SR_I_COND/SR_V_DROP)。将回退使用 MOS 正向导通表近似。")
            my_i_cond_sr = my_i_cond_mos
            my_v_drop_sr = my_v_drop_mos

        # If deadtime enabled but diode cond missing, fallback to SR for freewheel
        if enable_deadtime_cfg and (my_i_cond_diode_dt.size == 0 or my_v_diode_dt.size == 0):
            logger_func("!!! 警告：Deadtime Diode 导通表为空。将回退使用 SR 导通表近似死区二极管导通。")
            my_i_cond_diode_dt = my_i_cond_sr
            my_v_diode_dt = my_v_drop_sr

    else:
        # IGBT + Diode mode: always load diode XML, and deadtime is ignored in kernels
        logger_func(f"[XML] IGBT+Diode 模式，加载 Diode XML: {os.path.basename(xml_diode_path)}")
        try:
            vals_diode = load_data_from_xml(xml_diode_path)

            if 'DIODE_I_SW' in vals_diode:
                my_i_sw_diode_dt = np.array(vals_diode['DIODE_I_SW'], dtype=np.float64)
            if 'DIODE_E_RR' in vals_diode:
                v_src = vals_diode.get('V_AXIS_SW', my_v_axis_sw)
                tmp_rr = _align_3d_voltage(np.array(vals_diode['DIODE_E_RR'], dtype=np.float64), v_src, my_v_axis_sw)
                # align RR temperature to main switch axis
                t_src_sw = vals_diode.get('T_AXIS_SW', my_t_sw)
                my_e_rr_dt = _align_3d_temperature(tmp_rr, t_src_sw, my_t_sw)

            if 'DIODE_I_COND' in vals_diode:
                my_i_cond_diode_dt = np.array(vals_diode['DIODE_I_COND'], dtype=np.float64)
            if 'DIODE_V_DROP' in vals_diode:
                # align diode conduction temperature
                t_src_cond = vals_diode.get('T_AXIS_COND', my_t_cond)
                my_v_diode_dt = _align_2d_temperature(np.array(vals_diode['DIODE_V_DROP'], dtype=np.float64), t_src_cond, my_t_cond)

            if bool(cfg.get('RTH_DIODE_FOLLOW_MOS', False)):
                local_rth_diode = local_rth_igbt
                logger_func(" -> IGBT Coupled: Diode Rth 强制跟随 Switch (Co-Pack/Module Coupled)")
            else:
                local_rth_diode = _override_rth_if_valid(vals_diode, local_rth_diode, "Diode")
        except Exception as e:
            logger_func(f"!!! Diode XML 解析失败: {e}")
            traceback.print_exc()
            logger_func("将使用硬编码默认数据。")

        # IGBT mode doesn't use SR; give dummy (unused)
        my_i_cond_sr = my_i_cond_diode_dt
        my_v_drop_sr = my_v_diode_dt

    if local_rth_igbt <= 1e-6:
        logger_func("!!! 警告：Rth(IGBT/MOS) <=0，临时使用 0.0001 K/W")
        local_rth_igbt = 0.0001
    if local_rth_diode <= 1e-6:
        logger_func("!!! 警告：Rth(Diode) <=0，临时使用 0.0001 K/W")
        local_rth_diode = 0.0001

    cfg['RTH_IGBT'] = float(local_rth_igbt)
    cfg['RTH_DIODE'] = float(local_rth_diode)

    logger_func(f" -> V_axis={my_v_axis_sw}")
    logger_func(f" -> Rth(IGBT/MOS)={cfg['RTH_IGBT']:.4f}, Rth(Diode)={cfg['RTH_DIODE']:.4f}")
    
    # Deadtime Debug Info
    dt_en = bool(cfg.get('ENABLE_DEADTIME', False))
    dt_comp = bool(cfg.get('ENABLE_DT_COMP', False))
    td_val = float(cfg.get('TD', 0.0))
    ith_val = float(cfg.get('I_TH', 0.0))
    logger_func(f" -> MOS模式={is_mosfet_global}")
    logger_func(f" -> Deadtime Config: Enabled={dt_en}, Comp={dt_comp}, Td={td_val*1e6:.2f}us, I_th={ith_val}A")
    
    # lut bundle: append is_mosfet flag
    lut_bundle = (
        np.array(my_v_axis_sw, dtype=np.float64),
        np.array(my_t_sw, dtype=np.float64),
        np.array(my_t_cond, dtype=np.float64),

        np.array(my_i_sw_mos, dtype=np.float64),
        np.array(my_e_on, dtype=np.float64),
        np.array(my_e_off, dtype=np.float64),

        np.array(my_i_cond_mos, dtype=np.float64),
        np.array(my_v_drop_mos, dtype=np.float64),

        np.array(my_i_cond_sr, dtype=np.float64),
        np.array(my_v_drop_sr, dtype=np.float64),

        np.array(my_i_sw_diode_dt, dtype=np.float64),
        np.array(my_e_rr_dt, dtype=np.float64),

        np.array(my_i_cond_diode_dt, dtype=np.float64),
        np.array(my_v_diode_dt, dtype=np.float64),

        is_mosfet_global
    )

    # ------------------------
    # Read maps
    # ------------------------
    path_v = os.path.join(cfg['DIR_MAP'], cfg['FILE_CSV_V'])
    path_i = os.path.join(cfg['DIR_MAP'], cfg['FILE_CSV_I'])
    path_pf = os.path.join(cfg['DIR_MAP'], cfg['FILE_CSV_PF'])
    path_mask = os.path.join(cfg['DIR_MAP'], cfg['FILE_MASK'])

    logger_func("读取 Map CSV 文件...")
    try:
        df_v = pd.read_csv(path_v, header=None)
        df_i = pd.read_csv(path_i, header=None)
        df_p = pd.read_csv(path_pf, header=None)

        df_mask = None
        if cfg.get('ENABLE_MASK', False) and os.path.exists(path_mask):
            logger_func(f"读取 Mask 文件: {os.path.basename(path_mask)}")
            df_mask = pd.read_excel(path_mask, sheet_name=0, header=None)

    except Exception as e:
        logger_func(f"错误: 无法读取 Map 文件: {e}")
        return

    # tasks
    r_start, r_end = cfg['ROW_START'] - 1, cfg['ROW_END']
    c_start, c_end = cfg['COL_START'] - 1, cfg['COL_END']

    freq_row_idx = 0
    torque_col_idx = 0

    tasks = []
    mask_skipped_count = 0
    error_skipped_count = 0

    logger_func(f"正在生成任务 (范围 R:{r_start+1}~{r_end}, C:{c_start+1}~{c_end})...")

    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            if cfg.get('ENABLE_MASK', False) and df_mask is not None:
                try:
                    if r < df_mask.shape[0] and c < df_mask.shape[1]:
                        mask_val = df_mask.iloc[r, c]
                        if pd.isna(mask_val) or float(mask_val) == 0:
                            mask_skipped_count += 1
                            continue
                except:
                    pass
            try:
                freq_val = float(df_v.iloc[freq_row_idx, c])
                t_val = float(df_v.iloc[r, torque_col_idx])
                v_val = float(df_v.iloc[r, c])
                i_val = float(df_i.iloc[r, c])
                pf_val = float(df_p.iloc[r, c])
                
                tasks.append((r, c, freq_val, i_val, v_val, pf_val, t_val, cfg, True, lut_bundle))
            except Exception as e:
                if error_skipped_count == 0:
                    logger_func(f"DEBUG: 数据读取错误 (r={r}, c={c}) -> {e}")
                error_skipped_count += 1

    total_tasks = len(tasks)
    logger_func(f"任务报告: 有效={total_tasks}, Mask跳过={mask_skipped_count}, 数据错误跳过={error_skipped_count}")

    if total_tasks <= 0:
        logger_func("没有生成有效任务！请检查范围设置。")
        return

    # Canary
    logger_func(">>> 正在运行 Canary 测试 (单点串行检查)...")
    try:
        t_start_c = time.time()
        res_test = worker_simulation_task(tasks[0])
        t_end_c = time.time()

        (rid, cid, i_on, i_off, i_cond, i_tot, d_sw, d_cond, d_tot,
         t_i, t_d, ef, pdc, p_loss_single, sim_irms, sim_vrms) = res_test

        if i_tot == -1:
            logger_func("!!! Canary 测试失败，仿真中止。")
            return

        # Calculate Total Loss based on scale setting
        scale_val = float(cfg.get('LOSS_SCALE', 1.0))
        p_loss_total_display = p_loss_single * scale_val

        logger_func(f"   [Simulation Results]")
        logger_func(f"   Sim V_line_rms  : {sim_vrms:.2f} V")
        logger_func(f"   Sim I_phase_rms : {sim_irms:.2f} A")
        logger_func(f"   Input P_DC      : {pdc:.4f} W")
        logger_func(f"   Efficiency      : {ef:.4f} %")
        logger_func(f"   Single Dev Loss : {p_loss_single:.4f} W")
        logger_func(f"   Total Loss (x{int(scale_val)})  : {p_loss_total_display:.4f} W")
        logger_func(f"   Tj Switch       : {t_i:.2f} C")
        logger_func(f"   Tj Diode        : {t_d:.2f} C")
        logger_func(f"   ----------------------------------")
        logger_func(f"   [单管损耗/模组损耗]")
        logger_func(f"   Switch-ON       : {i_on:.4f} W")
        logger_func(f"   Switch-OFF      : {i_off:.4f} W")
        logger_func(f"   Cond (Fwd)      : {i_cond:.4f} W")
        logger_func(f"   Diode/SR Cond   : {d_cond:.4f} W")
        logger_func(f"   Reverse Recov   : {d_sw:.4f} W")
        logger_func(f"   ----------------------------------")
        logger_func(f"   Time: {t_end_c - t_start_c:.2f}s")
        logger_func(">>> Canary 测试通过，开始并行计算...")

    except Exception as e:
        logger_func(f"!!! Canary 崩溃: {e}")
        traceback.print_exc()
        return

    # parallel compute
    random.shuffle(tasks)

    res_igbt_on = pd.DataFrame(0.0, index=df_v.index, columns=df_v.columns)
    res_igbt_off = pd.DataFrame(0.0, index=df_v.index, columns=df_v.columns)
    res_igbt_cond = pd.DataFrame(0.0, index=df_v.index, columns=df_v.columns)
    res_igbt_tot = pd.DataFrame(0.0, index=df_v.index, columns=df_v.columns)
    res_igbt_tj = pd.DataFrame(0.0, index=df_v.index, columns=df_v.columns)

    res_diode_sw = pd.DataFrame(0.0, index=df_v.index, columns=df_v.columns)
    res_diode_cond = pd.DataFrame(0.0, index=df_v.index, columns=df_v.columns)
    res_diode_tot = pd.DataFrame(0.0, index=df_v.index, columns=df_v.columns)
    res_diode_tj = pd.DataFrame(0.0, index=df_v.index, columns=df_v.columns)

    res_sys_eff = pd.DataFrame(0.0, index=df_v.index, columns=df_v.columns)
    res_sys_pdc = pd.DataFrame(0.0, index=df_v.index, columns=df_v.columns)
    res_single_loss = pd.DataFrame(0.0, index=df_v.index, columns=df_v.columns)

    logger_func(f">>> 启动并行池 (Pool={cfg['CPU_CORES']})...")
    chunk = max(1, total_tasks // (int(cfg['CPU_CORES']) * 16))

    with multiprocessing.Pool(processes=int(cfg['CPU_CORES'])) as pool:
        results = pool.map(worker_simulation_task, tasks, chunksize=chunk)

    logger_func("计算完成，正在填充结果矩阵...")
    for item in results:
        (r, c, i_on, i_off, i_cond, i_tot, d_sw, d_cond, d_tot, t_i, t_d, ef, pdc, p_loss_single, s_irms, s_vrms) = item
        if i_tot != -1:
            res_igbt_on.iloc[r, c] = i_on
            res_igbt_off.iloc[r, c] = i_off
            res_igbt_cond.iloc[r, c] = i_cond
            res_igbt_tot.iloc[r, c] = i_tot
            res_igbt_tj.iloc[r, c] = t_i

            res_diode_sw.iloc[r, c] = d_sw
            res_diode_cond.iloc[r, c] = d_cond
            res_diode_tot.iloc[r, c] = d_tot
            res_diode_tj.iloc[r, c] = t_d

            res_sys_eff.iloc[r, c] = ef
            res_sys_pdc.iloc[r, c] = pdc
            res_single_loss.iloc[r, c] = p_loss_single

    # save outputs
    all_res_dfs = [
        (res_igbt_on, 'Output_IGBT_Sw_On.csv'),
        (res_igbt_off, 'Output_IGBT_Sw_Off.csv'),
        (res_igbt_cond, 'Output_IGBT_Conduction.csv'),
        (res_igbt_tot, 'Output_IGBT_Total.csv'),
        (res_igbt_tj, 'Output_IGBT_Tj.csv'),
        (res_diode_sw, 'Output_Diode_Switching.csv'),
        (res_diode_cond, 'Output_Diode_Conduction.csv'),
        (res_diode_tot, 'Output_Diode_Total.csv'),
        (res_diode_tj, 'Output_Diode_Tj.csv'),
        (res_sys_eff, 'Output_System_Efficiency.csv'),
        (res_sys_pdc, 'Output_System_P_DC.csv'),
        (res_single_loss, 'Output_Single_Device_Total_Loss.csv')
    ]

    logger_func("保存 CSV 文件...")
    os.makedirs(cfg['DIR_OUT'], exist_ok=True)

    for df, fname in all_res_dfs:
        df.iloc[freq_row_idx, :] = df_v.iloc[freq_row_idx, :]
        df.iloc[:, torque_col_idx] = df_v.iloc[:, torque_col_idx]
        df.iloc[freq_row_idx, torque_col_idx] = 0.0
        full_path = os.path.join(cfg['DIR_OUT'], fname)
        df.to_csv(full_path, index=False, header=False)

    # CLTC automation
    if cfg.get('ENABLE_CLTC', False) and (xw is not None):
        cltc_path = os.path.join(cfg['DIR_MAP'], cfg['FILE_CLTC_TEMP'])
        logger_func(f"[CLTC] 启动 Excel 自动化: {os.path.basename(cltc_path)}")
        if os.path.exists(cltc_path):
            try:
                app = xw.App(visible=True, add_book=False)
                app.display_alerts = False
                wb = app.books.open(cltc_path)
                ws = wb.sheets[0]

                r_end_idx = cfg['ROW_END']
                c_end_idx = cfg['COL_END']
                data_to_paste = res_sys_eff.iloc[cfg['ROW_START'] - 1:r_end_idx, cfg['COL_START'] - 1:c_end_idx]
                ws.range((19, 3)).options(index=False, header=False).value = data_to_paste.values

                time.sleep(2.0)
                result_val = ws.range((7, 16)).value
                logger_func(f"   >>> CLTC 计算结果: {result_val}")

                wb.save()
                wb.close()
                app.quit()
            except Exception as e:
                logger_func(f"CLTC 自动化错误: {e}")
        else:
            logger_func("找不到 CLTC 模板文件，跳过。")

    logger_func(f"全部完成! 总耗时: {time.time() - start_time:.2f}s")
    logger_func("=" * 60)


# =============================================================================
# CLI template
# =============================================================================
def _print_logger(msg):
    print(msg)


def _default_cfg_template():
    return {
        "DIR_XML": r".\xml",
        "FILE_XML_IGBT": "Your_MOS_or_IGBT.xml",
        "FILE_XML_DIODE": "Your_Diode.xml",

        "DIR_MAP": r".\map",
        "FILE_CSV_V": "Map_V.csv",
        "FILE_CSV_I": "Map_I.csv",
        "FILE_CSV_PF": "Map_PF.csv",
        "ENABLE_MASK": False,
        "FILE_MASK": "Mask.xlsx",

        "DIR_OUT": r".\out",

        "VDC": 630.0,
        "FSW": 10000.0,
        "TC": 65.0,
        "MOTOR_R": 20e-3,
        "MOTOR_L": 0.5e-3,
        "TARGET_IDX": 0,
        "RTH_IGBT": 0.1,
        "RTH_DIODE": 0.1,

        "ROW_START": 2,
        "ROW_END": 10,
        "COL_START": 2,
        "COL_END": 10,

        "CPU_CORES": max(1, multiprocessing.cpu_count() - 1),

        "ENABLE_CLTC": False,
        "FILE_CLTC_TEMP": "CLTC_Template.xlsx",

        # Deadtime (MOS only)
        "ENABLE_DEADTIME": False,
        "ENABLE_DT_COMP": False,
        "TD": 0.0,  # seconds (0 => disabled)
        "I_TH": 5.0, # Current threshold for deadtime (A)
        "I_RMS_TH_DEADTIME": 0.0, # RMS threshold to globally disable deadtime
        "FREQ_THRESHOLD": 150.0,
        "HI_DT": 1e-8,
        "HI_STARTUP": 100,
        "HI_ITER": 50,
        "HI_STAT": 24,
        "LO_DT": 1e-8,
        "LO_STARTUP": 10,
        "LO_ITER": 10,
        "LO_STAT": 6
    }


if __name__ == "__main__":
    multiprocessing.freeze_support()
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        os.makedirs(cfg.get("DIR_OUT", ".\\out"), exist_ok=True)
        run_simulation_logic(cfg, _print_logger)
    else:
        print("未找到 config.json。你可以在 core_runner.py 同目录创建 config.json，然后再运行。")
        print("下面是 config.json 模板（复制出去改路径/文件名即可）：\n")
        print(json.dumps(_default_cfg_template(), indent=2, ensure_ascii=False))
