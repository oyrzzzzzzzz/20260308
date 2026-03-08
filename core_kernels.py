# -*- coding: utf-8 -*-core_kernels.py
"""
Created on Thu Dec 25 11:23:32 2025
"""

# core_kernels.py
# -*- coding: utf-8 -*-
"""
Deadtime/SR split for MOS mode:
- SR conduction uses MOS XML negative-current branch (SR_I_COND / SR_V_DROP)
- Deadtime diode conduction + RR uses Diode XML (DIODE_*_DT)

IGBT+Diode mode:
- Deadtime disabled (forced) and losses always from IGBT XML + Diode XML.
"""

import math
import cmath
import numpy as np
from numba import njit
import traceback

# =============================================================================
# Defaults (empty => safe-0 in lookups)
# =============================================================================
LUT_V_AXIS_SW_DEFAULT = np.array([], dtype=np.float64)
LUT_T_AXIS_SW_DEFAULT = np.array([], dtype=np.float64)
LUT_T_AXIS_COND_DEFAULT = np.array([], dtype=np.float64)

LUT_I_AXIS_SW_MOS_DEFAULT = np.array([], dtype=np.float64)
LUT_I_AXIS_SW_DIODE_DEFAULT = np.array([], dtype=np.float64)

LUT_I_AXIS_COND_MOS_DEFAULT = np.array([], dtype=np.float64)       # MOS forward cond axis
LUT_I_AXIS_COND_SR_DEFAULT = np.array([], dtype=np.float64)        # MOS SR (neg branch mapped to +I axis)
LUT_I_AXIS_COND_DIODE_DEFAULT = np.array([], dtype=np.float64)     # deadtime diode cond axis (from diode xml)

TABLE_E_ON_DEFAULT = np.zeros((0, 0, 0), dtype=np.float64)
TABLE_E_OFF_DEFAULT = np.zeros((0, 0, 0), dtype=np.float64)
TABLE_E_RR_DEFAULT = np.zeros((0, 0, 0), dtype=np.float64)

TABLE_V_DROP_DEFAULT = np.zeros((0, 0), dtype=np.float64)          # MOS forward Vdrop(T,I)
TABLE_V_SR_DEFAULT = np.zeros((0, 0), dtype=np.float64)            # SR Vdrop(T,I) from MOS neg branch
TABLE_V_DIODE_DEFAULT = np.zeros((0, 0), dtype=np.float64)         # deadtime diode Vdrop(T,I) from diode xml


# =============================================================================
# Lookup helpers (safe for empty tables)
# =============================================================================
@njit(fastmath=False, cache=False)
def _is_invalid_2d(x_axis, y_axis, table_2d):
    if table_2d.size == 0:
        return True
    if x_axis.size < 1 or y_axis.size < 1:
        return True
    if table_2d.ndim != 2:
        return True
    if table_2d.shape[0] != y_axis.size:
        return True
    if table_2d.shape[1] != x_axis.size:
        return True
    return False


@njit(fastmath=False, cache=False)
def _is_invalid_3d(i_axis, t_axis, v_axis, table_3d):
    if table_3d.size == 0:
        return True
    if i_axis.size < 1 or t_axis.size < 1 or v_axis.size < 1:
        return True
    if table_3d.ndim != 3:
        return True
    if table_3d.shape[0] != t_axis.size:
        return True
    if table_3d.shape[1] != v_axis.size:
        return True
    if table_3d.shape[2] != i_axis.size:
        return True
    return False


@njit(fastmath=False, cache=False)
def lookup_2d_jit(x_val, y_val, x_axis, y_axis, table_data):
    """2D bilinear interpolation: X=current, Y=temperature, table[y,x]. Safe-0 if invalid."""
    if _is_invalid_2d(x_axis, y_axis, table_data):
        return 0.0

    x = x_val
    y = y_val

    # handle single-point axis
    if x_axis.size == 1 and y_axis.size == 1:
        v = table_data[0, 0]
        return 0.0 if v < 0 else v
    if x_axis.size == 1:
        # interpolate only in y
        iy = np.searchsorted(y_axis, y) - 1
        if iy < 0:
            iy = 0
        if iy >= y_axis.size - 1:
            iy = y_axis.size - 2
        y1 = y_axis[iy]
        y2 = y_axis[iy + 1]
        q1 = table_data[iy, 0]
        q2 = table_data[iy + 1, 0]
        dy = y2 - y1
        fy = 0.0 if dy == 0.0 else (y - y1) / dy
        v = q1 * (1.0 - fy) + q2 * fy
        return 0.0 if v < 0 else v
    if y_axis.size == 1:
        # interpolate only in x
        ix = np.searchsorted(x_axis, x) - 1
        if ix < 0:
            ix = 0
        if ix >= x_axis.size - 1:
            ix = x_axis.size - 2
        x1 = x_axis[ix]
        x2 = x_axis[ix + 1]
        q1 = table_data[0, ix]
        q2 = table_data[0, ix + 1]
        dx = x2 - x1
        fx = 0.0 if dx == 0.0 else (x - x1) / dx
        v = q1 * (1.0 - fx) + q2 * fx
        return 0.0 if v < 0 else v

    ix = np.searchsorted(x_axis, x) - 1
    iy = np.searchsorted(y_axis, y) - 1

    if ix < 0:
        ix = 0
    if iy < 0:
        iy = 0
    if ix >= x_axis.size - 1:
        ix = x_axis.size - 2
    if iy >= y_axis.size - 1:
        iy = y_axis.size - 2

    x1 = x_axis[ix]
    x2 = x_axis[ix + 1]
    y1 = y_axis[iy]
    y2 = y_axis[iy + 1]

    q11 = table_data[iy, ix]
    q21 = table_data[iy, ix + 1]
    q12 = table_data[iy + 1, ix]
    q22 = table_data[iy + 1, ix + 1]

    dx = x2 - x1
    dy = y2 - y1
    fx = 0.0 if dx == 0.0 else (x - x1) / dx
    fy = 0.0 if dy == 0.0 else (y - y1) / dy

    r1 = q11 * (1.0 - fx) + q21 * fx
    r2 = q12 * (1.0 - fx) + q22 * fx
    v = r1 * (1.0 - fy) + r2 * fy

    return 0.0 if v < 0 else v


@njit(fastmath=False, cache=False)
def lookup_3d_jit(i_val, t_val, v_val, i_axis, t_axis, v_axis, table_3d):
    """3D trilinear interpolation: table[t,v,i]. Safe-0 if invalid."""
    if _is_invalid_3d(i_axis, t_axis, v_axis, table_3d):
        return 0.0

    i = i_val
    t = t_val
    v = v_val

    # indices & fractions (support single-point axis)
    if v_axis.size < 2:
        iv0 = 0
        iv1 = 0
        fv = 0.0
    else:
        iv0 = np.searchsorted(v_axis, v) - 1
        if iv0 < 0:
            iv0 = 0
        if iv0 >= v_axis.size - 1:
            iv0 = v_axis.size - 2
        iv1 = iv0 + 1
        v1 = v_axis[iv0]
        v2 = v_axis[iv1]
        dv = v2 - v1
        fv = 0.0 if dv == 0.0 else (v - v1) / dv

    if t_axis.size < 2:
        it0 = 0
        it1 = 0
        ft = 0.0
    else:
        it0 = np.searchsorted(t_axis, t) - 1
        if it0 < 0:
            it0 = 0
        if it0 >= t_axis.size - 1:
            it0 = t_axis.size - 2
        it1 = it0 + 1
        t1 = t_axis[it0]
        t2 = t_axis[it1]
        dt = t2 - t1
        ft = 0.0 if dt == 0.0 else (t - t1) / dt

    if i_axis.size < 2:
        ii0 = 0
        ii1 = 0
        fi = 0.0
    else:
        ii0 = np.searchsorted(i_axis, i) - 1
        if ii0 < 0:
            ii0 = 0
        if ii0 >= i_axis.size - 1:
            ii0 = i_axis.size - 2
        ii1 = ii0 + 1
        i1 = i_axis[ii0]
        i2 = i_axis[ii1]
        di = i2 - i1
        fi = 0.0 if di == 0.0 else (i - i1) / di

    # corners
    v_tl_vl_il = table_3d[it0, iv0, ii0]
    v_tl_vl_ih = table_3d[it0, iv0, ii1]
    val_tl_vl = v_tl_vl_il * (1.0 - fi) + v_tl_vl_ih * fi

    v_tl_vh_il = table_3d[it0, iv1, ii0]
    v_tl_vh_ih = table_3d[it0, iv1, ii1]
    val_tl_vh = v_tl_vh_il * (1.0 - fi) + v_tl_vh_ih * fi

    v_th_vl_il = table_3d[it1, iv0, ii0]
    v_th_vl_ih = table_3d[it1, iv0, ii1]
    val_th_vl = v_th_vl_il * (1.0 - fi) + v_th_vl_ih * fi

    v_th_vh_il = table_3d[it1, iv1, ii0]
    v_th_vh_ih = table_3d[it1, iv1, ii1]
    val_th_vh = v_th_vh_il * (1.0 - fi) + v_th_vh_ih * fi

    val_tl = val_tl_vl * (1.0 - fv) + val_tl_vh * fv
    val_th = val_th_vl * (1.0 - fv) + val_th_vh * fv
    out = val_tl * (1.0 - ft) + val_th * ft
    return 0.0 if out < 0 else out


# =============================================================================
# Loss primitives
# =============================================================================
@njit(fastmath=False, cache=False)
def calculate_E_ON_arg(v_bus, i_instant, T_degC, i_axis, t_axis, v_axis, table):
    i_abs = abs(i_instant)
    if i_abs < 1e-5:
        return 0.0
    e_mJ = lookup_3d_jit(i_abs, T_degC, v_bus, i_axis, t_axis, v_axis, table)
    return e_mJ * 1e-3  # J


@njit(fastmath=False, cache=False)
def calculate_E_OFF_arg(v_bus, i_instant, T_degC, i_axis, t_axis, v_axis, table):
    i_abs = abs(i_instant)
    if i_abs < 1e-5:
        return 0.0
    e_mJ = lookup_3d_jit(i_abs, T_degC, v_bus, i_axis, t_axis, v_axis, table)
    return e_mJ * 1e-3


@njit(fastmath=False, cache=False)
def calculate_E_RR_arg(v_bus, i_instant, T_degC, i_axis, t_axis, v_axis, table):
    i_abs = abs(i_instant)
    if i_abs < 1e-5:
        return 0.0
    e_mJ = lookup_3d_jit(i_abs, T_degC, v_bus, i_axis, t_axis, v_axis, table)
    return e_mJ * 1e-3


@njit(fastmath=False, cache=False)
def calculate_Cond_P(i_instant, T_degC, i_axis, t_axis, table_vdrop):
    i_abs = abs(i_instant)
    if i_abs < 1e-12:
        return 0.0
    v = lookup_2d_jit(i_abs, T_degC, i_axis, t_axis, table_vdrop)
    return v * i_abs


# =============================================================================
# SVPWM (C-style) with optional deadtime compensation (Td_comp can be 0)
# =============================================================================
@njit(fastmath=False, cache=False)
def run_c_style_svpwm_logic(Valpha, Vbeta, Vdc, Ts, ia, ib, ic, Td_comp, I_th):
    PI = 3.141592653589793
    Vm = math.sqrt(Valpha * Valpha + Vbeta * Vbeta)
    angle = math.atan2(Vbeta, Valpha)
    if angle < 0.0:
        angle += 2.0 * PI

    sector = int(angle / (PI / 3.0))
    if sector > 5:
        sector = 5
    Theta_rel = angle - sector * (PI / 3.0)

    # note: keep your original K form
    K = Vm * Ts / (math.sqrt(3.0) / 2.0) / (2.0 * Vdc / 3.0)

    if sector % 2 != 0:
        T1 = K * math.sin(Theta_rel)
        T2 = K * math.sin(PI / 3.0 - Theta_rel)
    else:
        T1 = K * math.sin(PI / 3.0 - Theta_rel)
        T2 = K * math.sin(Theta_rel)

    if T1 + T2 > Ts:
        ratio = Ts / (T1 + T2)
        T1 *= ratio
        T2 *= ratio

    T0 = Ts - T1 - T2

    cmpa = 0.0
    cmpb = 0.0
    cmpc = 0.0

    if sector == 0:
        cmpa = T0 / 4.0
        cmpb = T0 / 4.0 + T1 / 2.0
        cmpc = T0 / 4.0 + T1 / 2.0 + T2 / 2.0
    elif sector == 1:
        cmpb = T0 / 4.0
        cmpa = T0 / 4.0 + T1 / 2.0
        cmpc = T0 / 4.0 + T1 / 2.0 + T2 / 2.0
    elif sector == 2:
        cmpb = T0 / 4.0
        cmpc = T0 / 4.0 + T1 / 2.0
        cmpa = T0 / 4.0 + T1 / 2.0 + T2 / 2.0
    elif sector == 3:
        cmpc = T0 / 4.0
        cmpb = T0 / 4.0 + T1 / 2.0
        cmpa = T0 / 4.0 + T1 / 2.0 + T2 / 2.0
    elif sector == 4:
        cmpc = T0 / 4.0
        cmpa = T0 / 4.0 + T1 / 2.0
        cmpb = T0 / 4.0 + T1 / 2.0 + T2 / 2.0
    else:
        cmpa = T0 / 4.0
        cmpc = T0 / 4.0 + T1 / 2.0
        cmpb = T0 / 4.0 + T1 / 2.0 + T2 / 2.0

    # deadtime compensation (Td_comp may be 0)
    # Applied only if abs(i) > I_th
    cmpa_comp = cmpa
    cmpb_comp = cmpb
    cmpc_comp = cmpc

    if ia > I_th:
        cmpa_comp = cmpa - 0.5 * Td_comp
    elif ia < -I_th:
        cmpa_comp = cmpa + 0.5 * Td_comp

    if ib > I_th:
        cmpb_comp = cmpb - 0.5 * Td_comp
    elif ib < -I_th:
        cmpb_comp = cmpb + 0.5 * Td_comp

    if ic > I_th:
        cmpc_comp = cmpc - 0.5 * Td_comp
    elif ic < -I_th:
        cmpc_comp = cmpc + 0.5 * Td_comp

    return cmpa_comp, cmpb_comp, cmpc_comp


# =============================================================================
# Main simulation kernel (stateful) with optional deadtime
# =============================================================================
@njit(fastmath=False, cache=False)
def simulation_kernel_stateful(
        t_start, ia_init, ib_init, ic_init,
        run_cycles, sim_dt, T_fund, w, V_phase_max, Vdc, Ts, fsw,
        E_mag, E_angle, R_load, L_load,
        T_junction_igbt, T_junction_diode,
        ANALYZE_LAST_N_CYCLES, target_device_idx,
        calc_rms_flag, is_mosfet_mode,
        enable_deadtime_flag,     # 0/1 (MOS only; IGBT forced 0 in runner)
        Td_dead,                  # actual deadtime inserted in gating
        Td_comp,                  # deadtime compensation applied in SVPWM compare (can be 0)
        I_th,                     # Current threshold for deadtime disabling
        LUT_V_AXIS_SW, LUT_T_AXIS_SW, LUT_T_AXIS_COND,
        LUT_I_AXIS_SW_MOS, TABLE_E_ON, TABLE_E_OFF,
        LUT_I_AXIS_COND_MOS, TABLE_V_DROP,           # MOS forward conduction
        LUT_I_AXIS_COND_SR, TABLE_V_SR,              # MOS SR conduction
        LUT_I_AXIS_SW_DIODE_DT, TABLE_E_RR_DT,       # deadtime diode RR (from diode xml when enabled)
        LUT_I_AXIS_COND_DIODE_DT, TABLE_V_DIODE_DT   # deadtime diode conduction
):
    PI = 3.141592653589793

    total_steps = int(run_cycles * T_fund / sim_dt)
    ia = ia_init
    ib = ib_init
    ic = ic_init

    acc_sw_on_igbt = 0.0
    acc_sw_off_igbt = 0.0
    acc_sw_rr_diode = 0.0
    acc_cond_igbt = 0.0     # device forward channel conduction energy (J)
    acc_cond_diode = 0.0    # diode-related conduction energy (J) (includes SR or DT diode depending on path)
    acc_input_energy = 0.0

    acc_i_sq_sum = 0.0
    acc_v_sq_sum = 0.0
    stat_sample_count = 0

    current_Valpha = 0.0
    current_Vbeta = 0.0

    max_ia_abs = 0.0

    run_duration = run_cycles * T_fund
    stat_relative_start = run_duration - (ANALYZE_LAST_N_CYCLES * T_fund)
    if stat_relative_start < 0.0:
        stat_relative_start = 0.0

    t_abs = t_start
    current_pwm_idx = -1
    active_da = 0.0
    active_db = 0.0
    active_dc = 0.0

    # deadtime state machine
    t_fell_top_a = -1.0
    t_fell_bot_a = -1.0
    t_fell_top_b = -1.0
    t_fell_bot_b = -1.0
    t_fell_top_c = -1.0
    t_fell_bot_c = -1.0

    gate_top_a = 0
    gate_bot_a = 0
    gate_top_b = 0
    gate_bot_b = 0
    gate_top_c = 0
    gate_bot_c = 0

    prev_gate_top_a = 0
    prev_gate_bot_a = 0
    prev_gate_top_b = 0
    prev_gate_bot_b = 0
    prev_gate_top_c = 0
    prev_gate_bot_c = 0

    # Effective deadtime latched per cycle
    eff_td_a = 0.0
    eff_td_b = 0.0
    eff_td_c = 0.0

    pwm_phase_offset = w * (Ts / 2.0)

    for k in range(total_steps):
        t_rel = k * sim_dt
        t_abs = t_start + t_rel

        # --- SVPWM update per switching interval
        now_pwm_idx = int(t_abs * fsw)
        if now_pwm_idx > current_pwm_idx:
            theta = w * t_abs + pwm_phase_offset
            current_Valpha = V_phase_max * math.cos(theta)
            current_Vbeta = V_phase_max * math.sin(theta)

            c_a, c_b, c_c = run_c_style_svpwm_logic(current_Valpha, current_Vbeta, Vdc, Ts, ia, ib, ic, Td_comp, I_th)
            
            # Latch deadtime decision for this cycle to match Compensation decision
            eff_td_a = Td_dead if abs(ia) > I_th else 0.0
            eff_td_b = Td_dead if abs(ib) > I_th else 0.0
            eff_td_c = Td_dead if abs(ic) > I_th else 0.0

            active_da = 1.0 - 2.0 * c_a / Ts
            active_db = 1.0 - 2.0 * c_b / Ts
            active_dc = 1.0 - 2.0 * c_c / Ts
            current_pwm_idx = now_pwm_idx

        # carrier triangle 0..1..0
        t_mod = t_abs % Ts
        if t_mod < (Ts / 2.0):
            carrier = t_mod / (Ts / 2.0)
        else:
            carrier = 2.0 - t_mod / (Ts / 2.0)

        ideal_top_a = 1 if active_da >= carrier else 0
        ideal_top_b = 1 if active_db >= carrier else 0
        ideal_top_c = 1 if active_dc >= carrier else 0

        if enable_deadtime_flag == 0 or Td_dead <= 0.0:
            # ideal complementary
            gate_top_a = ideal_top_a
            gate_bot_a = 1 - ideal_top_a
            gate_top_b = ideal_top_b
            gate_bot_b = 1 - ideal_top_b
            gate_top_c = ideal_top_c
            gate_bot_c = 1 - ideal_top_c
        else:
            # edge-trigger deadtime insertion (your "final" logic)
            # Use latched effective deadtime
            td_a = eff_td_a
            td_b = eff_td_b
            td_c = eff_td_c

            # Phase A
            if ideal_top_a == 1:
                if (prev_gate_bot_a == 0) and (t_abs - t_fell_bot_a >= td_a):
                    gate_top_a = 1
                else:
                    gate_top_a = -1
            else:
                gate_top_a = 0

            if ideal_top_a == 0:
                if (prev_gate_top_a == 0) and (t_abs - t_fell_top_a >= td_a):
                    gate_bot_a = 1
                else:
                    gate_bot_a = -1
            else:
                gate_bot_a = 0

            # Phase B
            if ideal_top_b == 1:
                if (prev_gate_bot_b == 0) and (t_abs - t_fell_bot_b >= td_b):
                    gate_top_b = 1
                else:
                    gate_top_b = -1
            else:
                gate_top_b = 0

            if ideal_top_b == 0:
                if (prev_gate_top_b == 0) and (t_abs - t_fell_top_b >= td_b):
                    gate_bot_b = 1
                else:
                    gate_bot_b = -1
            else:
                gate_bot_b = 0

            # Phase C
            if ideal_top_c == 1:
                if (prev_gate_bot_c == 0) and (t_abs - t_fell_bot_c >= td_c):
                    gate_top_c = 1
                else:
                    gate_top_c = -1
            else:
                gate_top_c = 0

            if ideal_top_c == 0:
                if (prev_gate_top_c == 0) and (t_abs - t_fell_top_c >= td_c):
                    gate_bot_c = 1
                else:
                    gate_bot_c = -1
            else:
                gate_bot_c = 0

        # falling edge timestamps
        if prev_gate_top_a == 1 and gate_top_a == 0:
            t_fell_top_a = t_abs
        if prev_gate_bot_a == 1 and gate_bot_a == 0:
            t_fell_bot_a = t_abs
        if prev_gate_top_b == 1 and gate_top_b == 0:
            t_fell_top_b = t_abs
        if prev_gate_bot_b == 1 and gate_bot_b == 0:
            t_fell_bot_b = t_abs
        if prev_gate_top_c == 1 and gate_top_c == 0:
            t_fell_top_c = t_abs
        if prev_gate_bot_c == 1 and gate_bot_c == 0:
            t_fell_bot_c = t_abs

        # phase voltages with freewheel during deadtime/off
        if gate_top_a == 1:
            va = Vdc
        elif gate_bot_a == 1:
            va = 0.0
        else:
            if ia > 0.0:
                va = 0.0
            elif ia < 0.0:
                va = Vdc
            else:
                va = 0.5 * Vdc

        if gate_top_b == 1:
            vb = Vdc
        elif gate_bot_b == 1:
            vb = 0.0
        else:
            if ib > 0.0:
                vb = 0.0
            elif ib < 0.0:
                vb = Vdc
            else:
                vb = 0.5 * Vdc

        if gate_top_c == 1:
            vc = Vdc
        elif gate_bot_c == 1:
            vc = 0.0
        else:
            if ic > 0.0:
                vc = 0.0
            elif ic < 0.0:
                vc = Vdc
            else:
                vc = 0.5 * Vdc

        # stats window
        if t_rel >= stat_relative_start:
            if calc_rms_flag > 0:
                acc_i_sq_sum += (ia * ia + ib * ib + ic * ic)

                # physical line-line rms based on ref vector (kept as your previous)
                v_ref_a = current_Valpha
                v_ref_b = -0.5 * current_Valpha + 0.866 * current_Vbeta
                v_ref_c = -0.5 * current_Valpha - 0.866 * current_Vbeta
                v_ab = v_ref_a - v_ref_b
                v_bc = v_ref_b - v_ref_c
                v_ca = v_ref_c - v_ref_a
                acc_v_sq_sum += (v_ab * v_ab + v_bc * v_bc + v_ca * v_ca)
                stat_sample_count += 1
            dead_a = (gate_top_a == -1) or (gate_bot_a == -1)
            # =========================
            # Loss accounting (Phase A only, keep your speed pattern)
            # =========================
            if target_device_idx == 0:
                # Bottom device monitored
                # switching on/off for bottom gate (use MOS/IGBT Eon/Eoff tables)
                if gate_bot_a != prev_gate_bot_a:
                    if gate_bot_a == 1:
                        # ON event (for bottom, your original condition used ia < 0)
                        if ia < 0.0:
                            acc_sw_on_igbt += calculate_E_ON_arg(Vdc, ia, T_junction_igbt, LUT_I_AXIS_SW_MOS, LUT_T_AXIS_SW, LUT_V_AXIS_SW, TABLE_E_ON)
                    elif prev_gate_bot_a == 1 and gate_bot_a == 0:
                        # OFF event
                        if ia < 0.0:
                            acc_sw_off_igbt += calculate_E_OFF_arg(Vdc, ia, T_junction_igbt, LUT_I_AXIS_SW_MOS, LUT_T_AXIS_SW, LUT_V_AXIS_SW, TABLE_E_OFF)

                # RR: top turns ON while current freewheeled through bottom diode path (ia > 0)
                # Use deadtime diode RR table (from diode xml when enabled; else likely zeros)
                if gate_top_a == 1 and prev_gate_top_a != 1 and ia > 0.0:
                    acc_sw_rr_diode += calculate_E_RR_arg(Vdc, ia, T_junction_diode, LUT_I_AXIS_SW_DIODE_DT, LUT_T_AXIS_SW, LUT_V_AXIS_SW, TABLE_E_RR_DT)

                # conduction split:
                if gate_bot_a == 1:
                    if ia < 0.0:
                        # MOS forward channel (bottom)
                        acc_cond_igbt += calculate_Cond_P(ia, T_junction_igbt, LUT_I_AXIS_COND_MOS, LUT_T_AXIS_COND, TABLE_V_DROP) * sim_dt
                    else:
                        # SR conduction (MOS neg branch) when gate ON but current opposite
                        # If IGBT mode: this is actually the Diode conducting (Free-wheeling while gate ON but current opposed? No, IGBT cannot conduct reverse.)
                        # Wait, for IGBT: if gate is ON but ia > 0 (current flows out), current must flow through Diode (anti-parallel).
                        # So it's Diode conduction.
                        p_sr = calculate_Cond_P(ia, T_junction_diode, LUT_I_AXIS_COND_SR, LUT_T_AXIS_COND, TABLE_V_SR) * sim_dt
                        if is_mosfet_mode == 1:
                            acc_cond_igbt += p_sr
                        else:
                            acc_cond_diode += p_sr
                else:
                    # deadtime / off freewheel diode conduction (ia > 0 uses bottom diode)
                    if dead_a and ia > 0.0:
                        acc_cond_diode += calculate_Cond_P(ia, T_junction_diode, LUT_I_AXIS_COND_DIODE_DT, LUT_T_AXIS_COND, TABLE_V_DIODE_DT) * sim_dt

            else:
                # Top device monitored
                if gate_top_a != prev_gate_top_a:
                    if gate_top_a == 1:
                        if ia > 0.0:
                            acc_sw_on_igbt += calculate_E_ON_arg(Vdc, ia, T_junction_igbt, LUT_I_AXIS_SW_MOS, LUT_T_AXIS_SW, LUT_V_AXIS_SW, TABLE_E_ON)
                    elif prev_gate_top_a == 1 and gate_top_a == 0:
                        if ia > 0.0:
                            acc_sw_off_igbt += calculate_E_OFF_arg(Vdc, ia, T_junction_igbt, LUT_I_AXIS_SW_MOS, LUT_T_AXIS_SW, LUT_V_AXIS_SW, TABLE_E_OFF)

                # RR: bottom turns ON while current freewheeled through top diode path (ia < 0)
                if gate_bot_a == 1 and prev_gate_bot_a != 1 and ia < 0.0:
                    acc_sw_rr_diode += calculate_E_RR_arg(Vdc, ia, T_junction_diode, LUT_I_AXIS_SW_DIODE_DT, LUT_T_AXIS_SW, LUT_V_AXIS_SW, TABLE_E_RR_DT)

                if gate_top_a == 1:
                    if ia > 0.0:
                        acc_cond_igbt += calculate_Cond_P(ia, T_junction_igbt, LUT_I_AXIS_COND_MOS, LUT_T_AXIS_COND, TABLE_V_DROP) * sim_dt
                    else:
                        # SR conduction when gate ON but current opposite
                        p_sr = calculate_Cond_P(ia, T_junction_diode, LUT_I_AXIS_COND_SR, LUT_T_AXIS_COND, TABLE_V_SR) * sim_dt
                        if is_mosfet_mode == 1:
                            acc_cond_igbt += p_sr
                        else:
                            acc_cond_diode += p_sr
                else:
                    if dead_a and ia < 0.0:
                        acc_cond_diode += calculate_Cond_P(ia, T_junction_diode, LUT_I_AXIS_COND_DIODE_DT, LUT_T_AXIS_COND, TABLE_V_DIODE_DT) * sim_dt

            # DC input power integration
            threshold_v = 0.5 * Vdc
            idc = 0.0
            if va > threshold_v:
                idc += ia
            if vb > threshold_v:
                idc += ib
            if vc > threshold_v:
                idc += ic
            acc_input_energy += (Vdc * idc) * sim_dt

        # update history
        prev_gate_top_a = gate_top_a
        prev_gate_bot_a = gate_bot_a
        prev_gate_top_b = gate_top_b
        prev_gate_bot_b = gate_bot_b
        prev_gate_top_c = gate_top_c
        prev_gate_bot_c = gate_bot_c

        # circuit update
        vn = (va + vb + vc) / 3.0
        theta_e = w * t_abs + E_angle
        ea = E_mag * math.cos(theta_e)
        eb = E_mag * math.cos(theta_e - 2.0 * PI / 3.0)
        ec = E_mag * math.cos(theta_e + 2.0 * PI / 3.0)

        ia += (va - vn - R_load * ia - ea) / L_load * sim_dt
        ib += (vb - vn - R_load * ib - eb) / L_load * sim_dt
        ic += (vc - vn - R_load * ic - ec) / L_load * sim_dt

    return (acc_sw_on_igbt, acc_sw_off_igbt, acc_sw_rr_diode,
            acc_cond_igbt, acc_cond_diode, acc_input_energy,
            acc_i_sq_sum, acc_v_sq_sum, stat_sample_count,
            t_abs, ia, ib, ic)


# =============================================================================
# Worker for multiprocessing
# =============================================================================
def worker_simulation_task(params):
    """
    params = (r, c, f_out, i_rms, v_line_rms, pf, torque, cfg, calc_rms, lut_bundle)
    lut_bundle = (... arrays ..., is_mosfet_flag)
    """
    try:
        r_idx, c_idx, f_out, i_rms, v_line_rms, pf, torque_val, cfg, calc_rms, lut_bundle = params
        (*base_params, is_mosfet) = lut_bundle
        (LUT_V_AXIS_SW, LUT_T_AXIS_SW, LUT_T_AXIS_COND,
         LUT_I_AXIS_SW_MOS, TABLE_E_ON, TABLE_E_OFF,
         LUT_I_AXIS_COND_MOS, TABLE_V_DROP,
         LUT_I_AXIS_COND_SR, TABLE_V_SR,
         LUT_I_AXIS_SW_DIODE_DT, TABLE_E_RR_DT,
         LUT_I_AXIS_COND_DIODE_DT, TABLE_V_DIODE_DT) = tuple(base_params)

        f_out = float(f_out)
        i_rms = float(i_rms)
        v_line_rms = float(v_line_rms)
        pf = float(pf)
        torque_val = float(torque_val)

        if i_rms < 0.1 or v_line_rms < 0.1 or f_out < 0.1:
            return (r_idx, c_idx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    cfg['TC'], cfg['TC'], 0.0, 0.0, 0.0, 0.0, 0.0)

        Vdc_sys = float(cfg['VDC'])
        fsw = float(cfg['FSW'])
        w = 2.0 * math.pi * f_out
        rms_flag_int = 1 if calc_rms else 0

        # speed tiers
        if f_out < float(cfg['FREQ_THRESHOLD']):
            current_startup = int(cfg['LO_STARTUP'])
            current_iter = int(cfg['LO_ITER'])
            current_stat = int(cfg['LO_STAT'])
            sim_dt = float(cfg['LO_DT'])
        else:
            current_startup = int(cfg['HI_STARTUP'])
            current_iter = int(cfg['HI_ITER'])
            current_stat = int(cfg['HI_STAT'])
            sim_dt = float(cfg['HI_DT'])

        T_fund = 1.0 / f_out
        stat_duration = current_stat * T_fund
        if stat_duration <= 0.0:
            stat_duration = 1e-9

        # modulation and phasors
        m = (v_line_rms * math.sqrt(2.0)) / Vdc_sys
        if m > 1.15:
            m = 1.15
        if i_rms < 1e-4:
            i_rms = 1e-4

        v_phase_rms = v_line_rms / math.sqrt(3.0)
        v_ref_vec_rms = complex(v_phase_rms, 0.0)
        theta = math.acos(min(max(abs(pf), -1.0), 1.0))
        i_real = i_rms * math.cos(theta)
        i_imag = -i_rms * math.sin(theta)
        if torque_val < 0.0:
            i_real = -i_real

        i_ref_vec_rms = complex(i_real, i_imag)
        z_motor = complex(float(cfg['MOTOR_R']), w * float(cfg['MOTOR_L']))
        e_vec_rms = v_ref_vec_rms - i_ref_vec_rms * z_motor
        E_angle = cmath.phase(e_vec_rms)
        E_mag_peak = abs(e_vec_rms) * math.sqrt(2.0)
        V_phase_pk_pwm = m * Vdc_sys / math.sqrt(3.0)

        # deadtime effective only in MOS mode
        enable_deadtime = 1 if (is_mosfet and bool(cfg.get('ENABLE_DEADTIME', False))) else 0

        # Check if CSV RMS current is below threshold
        i_rms_th_dt = float(cfg.get('I_RMS_TH_DEADTIME', 0.0))
        if enable_deadtime == 1 and i_rms_th_dt > 0.0 and i_rms < i_rms_th_dt:
             enable_deadtime = 0

        Td = float(cfg.get('TD', 0.0))
        if enable_deadtime == 0:
            Td_dead = 0.0
            Td_comp = 0.0
        else:
            Td_dead = Td if Td > 0.0 else 0.0
            Td_comp = Td_dead if bool(cfg.get('ENABLE_DT_COMP', False)) else 0.0

        I_th = float(cfg.get('I_TH', 0.0))

        # #region agent log
        if r_idx == 0 and c_idx == 0: # Log only first task
            try:
                with objmode():
                    with open(r"c:\Users\HASEE\Desktop\CLTC计算脚本\.cursor\debug.log", "a") as f:
                        log_entry = {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H4",
                            "location": "core_kernels.py:worker_simulation_task",
                            "message": "Worker config check",
                            "data": {
                                "enable_deadtime": enable_deadtime,
                                "i_rms": i_rms,
                                "Td": Td,
                                "I_th": I_th
                            },
                            "timestamp": time.time() * 1000
                        }
                        f.write(json.dumps(log_entry) + "\n")
            except:
                pass
        # #endregion

        # thermal iteration
        Tvj_igbt = float(cfg['TC'])
        Tvj_diode = float(cfg['TC'])
        t_curr = 0.0
        ia_curr = 0.0
        ib_curr = 0.0
        ic_curr = 0.0

        final_losses = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        final_rms = (0.0, 0.0)

        MAX_THERMAL_ITERS = 30
        for it in range(MAX_THERMAL_ITERS):
            cycles = current_startup if it == 0 else current_iter

            res = simulation_kernel_stateful(
                t_curr, ia_curr, ib_curr, ic_curr,
                cycles, sim_dt, T_fund, w, V_phase_pk_pwm, Vdc_sys, 1.0 / fsw, fsw,
                E_mag_peak, E_angle, float(cfg['MOTOR_R']), float(cfg['MOTOR_L']),
                Tvj_igbt, Tvj_diode,
                current_stat, int(cfg['TARGET_IDX']),
                rms_flag_int, 1 if is_mosfet else 0,
                enable_deadtime, Td_dead, Td_comp, I_th,
                LUT_V_AXIS_SW, LUT_T_AXIS_SW, LUT_T_AXIS_COND,
                LUT_I_AXIS_SW_MOS, TABLE_E_ON, TABLE_E_OFF,
                LUT_I_AXIS_COND_MOS, TABLE_V_DROP,
                LUT_I_AXIS_COND_SR, TABLE_V_SR,
                LUT_I_AXIS_SW_DIODE_DT, TABLE_E_RR_DT,
                LUT_I_AXIS_COND_DIODE_DT, TABLE_V_DIODE_DT
            )

            (j_sw_on, j_sw_off, j_sw_rr, j_cond_fwd, j_cond_diode_like, j_input,
             acc_i_sq, acc_v_sq, n_samples, t_curr, ia_curr, ib_curr, ic_curr) = res
            p_sw_on = j_sw_on / stat_duration
            p_sw_off = j_sw_off / stat_duration
            p_sw_rr = j_sw_rr / stat_duration
            p_cond_fwd = j_cond_fwd / stat_duration
            p_cond_diode_like = j_cond_diode_like / stat_duration
            p_in_avg = j_input / stat_duration

            p_igbt_total = p_sw_on + p_sw_off + p_cond_fwd
            p_diode_total = p_sw_rr + p_cond_diode_like

            if bool(cfg.get('RTH_DIODE_FOLLOW_MOS', False)):
                p_total = p_igbt_total + p_diode_total
                new_Tvj = float(cfg['TC']) + (p_total * float(cfg['RTH_IGBT']))
                new_Tvj_igbt = new_Tvj
                new_Tvj_diode = new_Tvj
            else:
                new_Tvj_igbt = float(cfg['TC']) + (p_igbt_total * float(cfg['RTH_IGBT']))
                new_Tvj_diode = float(cfg['TC']) + (p_diode_total * float(cfg['RTH_DIODE']))

            diff_igbt = abs(new_Tvj_igbt - Tvj_igbt)
            diff_diode = abs(new_Tvj_diode - Tvj_diode)

            if calc_rms and n_samples > 0:
                sim_i_rms = math.sqrt(acc_i_sq / (3.0 * n_samples))
                sim_v_rms = math.sqrt(acc_v_sq / (3.0 * n_samples))
            else:
                sim_i_rms = 0.0
                sim_v_rms = 0.0

            final_losses = (p_sw_on, p_sw_off, p_sw_rr, p_cond_fwd, p_cond_diode_like, p_in_avg)
            final_rms = (sim_i_rms, sim_v_rms)

            Tvj_igbt = new_Tvj_igbt
            Tvj_diode = new_Tvj_diode

            if diff_igbt < 0.01 and diff_diode < 0.01:
                break

        p_on, p_off, p_rr, p_ci, p_cd, p_in = final_losses
        sim_i_rms, sim_v_rms = final_rms
        scale = cfg.get("LOSS_SCALE", 1)
        out_igbt_on = p_on
        out_igbt_off = p_off
        out_igbt_cond = p_ci
        out_igbt_tot = p_on + p_off + p_ci

        out_diode_sw = p_rr
        out_diode_cond = p_cd
        out_diode_tot = p_rr + p_cd

        p_loss_inv = (out_igbt_tot + out_diode_tot)*scale
        p_loss_single = out_igbt_tot + out_diode_tot

        eff = 0.0
        if p_in > 0.0:
            if p_in > 1.0:
                eff = ((p_in - p_loss_inv) / p_in) * 100.0
        else:
            p_rec = abs(p_in)
            if p_rec > 1.0:
                eff = ((p_rec - p_loss_inv) / p_rec) * 100.0

        return (r_idx, c_idx, out_igbt_on, out_igbt_off, out_igbt_cond, out_igbt_tot,
                out_diode_sw, out_diode_cond, out_diode_tot, Tvj_igbt, Tvj_diode,
                eff, p_in, p_loss_single, sim_i_rms, sim_v_rms)

    except Exception:
        traceback.print_exc()
        return (params[0], params[1], -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
