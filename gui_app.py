# gui_app.py
# -*- coding: utf-8 -*-

import os
import glob
import json
import threading
import traceback
import multiprocessing
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from core_runner import run_simulation_logic


class SimulationConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HPD 仿真配置中心 (MOS SR + Deadtime Diode)")
        self.root.geometry("920x920")

        # Paths
        self.var_map_dir = tk.StringVar()
        self.var_xml_dir = tk.StringVar()
        self.var_out_dir = tk.StringVar()
        self.var_csv_v = tk.StringVar()
        self.var_csv_i = tk.StringVar()
        self.var_csv_pf = tk.StringVar()
        self.var_xml_igbt = tk.StringVar()
        self.var_xml_diode = tk.StringVar()
        self.var_mask_file = tk.StringVar(value="choose_map_point.xlsx")
        self.var_cltc_template = tk.StringVar(value="CLTC_Calculator_Template.xlsx")

        # Basic params
        self.var_vdc = tk.DoubleVar(value=630.0)
        self.var_tc = tk.DoubleVar(value=65.0)
        self.var_fsw = tk.DoubleVar(value=10000.0)
        self.var_motor_r = tk.DoubleVar(value=20e-3)
        self.var_motor_l = tk.DoubleVar(value=0.5e-3)
        self.var_target = tk.StringVar(value="Bottom (下管)")
        self.var_rth_igbt = tk.DoubleVar(value=0.1)
        self.var_rth_diode = tk.DoubleVar(value=0.1)

        # Thermal & Efficiency controls
        self.var_rth_diode_follow_mos = tk.BooleanVar(value=False)
        self.var_loss_scale_mode = tk.StringVar(value="Single Device (x6)")

        # Range/parallel
        self.var_row_start = tk.IntVar(value=55)
        self.var_row_end = tk.IntVar(value=114)
        self.var_col_start = tk.IntVar(value=2)
        self.var_col_end = tk.IntVar(value=31)
        self.var_enable_mask = tk.BooleanVar(value=True)
        self.var_enable_cltc = tk.BooleanVar(value=True)
        self.var_cpu = tk.IntVar(value=max(1, multiprocessing.cpu_count() - 1))

        # Speed tiers
        self.var_freq_threshold = tk.DoubleVar(value=150.0)
        self.var_hi_dt = tk.DoubleVar(value=1e-8)
        self.var_hi_startup = tk.IntVar(value=100)
        self.var_hi_iter = tk.IntVar(value=50)
        self.var_hi_stat = tk.IntVar(value=24)
        self.var_lo_dt = tk.DoubleVar(value=1e-8)
        self.var_lo_startup = tk.IntVar(value=10)
        self.var_lo_iter = tk.IntVar(value=10)
        self.var_lo_stat = tk.IntVar(value=6)

        # Deadtime controls (MOS only)
        self.var_enable_deadtime = tk.BooleanVar(value=False)
        self.var_enable_dt_comp = tk.BooleanVar(value=False)
        self.var_td_us = tk.DoubleVar(value=0.0)  # microseconds in GUI
        self.var_i_th = tk.DoubleVar(value=5.0)    # Current threshold (A)
        self.var_i_rms_th_dt = tk.DoubleVar(value=0.0) # RMS Current threshold to disable DT (A)

        self.create_widgets()

    def create_widgets(self):
        paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(paned)
        paned.add(top_frame, weight=3)

        canvas = tk.Canvas(top_frame)
        scrollbar = ttk.Scrollbar(top_frame, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        bottom_frame = ttk.LabelFrame(paned, text="运行日志 (Console Log)", padding=5)
        paned.add(bottom_frame, weight=1)

        self.log_text = scrolledtext.ScrolledText(bottom_frame, state='disabled', height=10, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        pad_opts = {'padx': 10, 'pady': 5}
        self.create_path_section(scroll_frame)

        notebook = ttk.Notebook(scroll_frame)
        notebook.pack(fill=tk.BOTH, expand=True, **pad_opts)

        tab_files = ttk.Frame(notebook, padding=10)
        notebook.add(tab_files, text="1. 文件映射")
        self.setup_tab_files(tab_files)

        tab_physics = ttk.Frame(notebook, padding=10)
        notebook.add(tab_physics, text="2. 物理参数")
        self.setup_tab_physics(tab_physics)

        tab_adv = ttk.Frame(notebook, padding=10)
        notebook.add(tab_adv, text="3. 高级设置")
        self.setup_tab_adv(tab_adv)

        btn_frame = ttk.Frame(scroll_frame)
        btn_frame.pack(fill=tk.X, pady=20, padx=10)

        ttk.Button(btn_frame, text="加载配置 (Load)", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="保存配置 (Save)", command=self.save_config).pack(side=tk.LEFT, padx=5)

        style = ttk.Style()
        style.configure("Accent.TButton", foreground="green", font=("Arial", 10, "bold"))
        self.btn_run = ttk.Button(btn_frame, text="启动仿真 (Start Simulation)", command=self.on_start_click, style="Accent.TButton")
        self.btn_run.pack(side=tk.RIGHT, padx=5)

    def create_path_section(self, parent):
        grp = ttk.LabelFrame(parent, text="全局路径设置", padding="10")
        grp.pack(fill=tk.X, padx=10, pady=5)

        def row(idx, txt, var, cmd):
            ttk.Label(grp, text=txt, foreground="blue" if idx == 0 else "black").grid(row=idx, column=0, sticky="w")
            ttk.Entry(grp, textvariable=var, width=55).grid(row=idx, column=1, padx=5, pady=2)
            ttk.Button(grp, text="浏览...", command=cmd).grid(row=idx, column=2)

        row(0, "CSV/Excel Map 路径:", self.var_map_dir, self.select_map_dir)
        row(1, "XML 器件模型路径:", self.var_xml_dir, self.select_xml_dir)
        row(2, "仿真结果保存路径:", self.var_out_dir, self.select_out_dir)

    def setup_tab_files(self, parent):
        grid = {'sticky': 'w', 'pady': 3}
        ttk.Label(parent, text="[CSV Map 文件] (从上方路径自动扫描)", foreground="blue", font=("Arial", 9, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 5), sticky="w")

        ttk.Label(parent, text="电压 Map:").grid(row=1, column=0, **grid)
        self.cb_csv_v_widget = ttk.Combobox(parent, textvariable=self.var_csv_v, width=50)
        self.cb_csv_v_widget.grid(row=1, column=1, **grid)

        ttk.Label(parent, text="电流 Map:").grid(row=2, column=0, **grid)
        self.cb_csv_i_widget = ttk.Combobox(parent, textvariable=self.var_csv_i, width=50)
        self.cb_csv_i_widget.grid(row=2, column=1, **grid)

        ttk.Label(parent, text="PF Map:").grid(row=3, column=0, **grid)
        self.cb_csv_pf_widget = ttk.Combobox(parent, textvariable=self.var_csv_pf, width=50)
        self.cb_csv_pf_widget.grid(row=3, column=1, **grid)

        ttk.Separator(parent).grid(row=4, column=0, columnspan=2, sticky='ew', pady=10)

        ttk.Label(parent, text="[XML 器件模型(开关损耗0A对应Esw需为0))]", foreground="darkred", font=("Arial", 9, "bold")).grid(
            row=5, column=0, columnspan=2, pady=(0, 5), sticky="w")

        ttk.Label(parent, text="Switch XML (IGBT/MOS):").grid(row=6, column=0, **grid)
        self.cb_xml_igbt_widget = ttk.Combobox(parent, textvariable=self.var_xml_igbt, width=50)
        self.cb_xml_igbt_widget.grid(row=6, column=1, **grid)

        ttk.Label(parent, text="Diode XML:").grid(row=7, column=0, **grid)
        self.cb_xml_diode_widget = ttk.Combobox(parent, textvariable=self.var_xml_diode, width=50)
        self.cb_xml_diode_widget.grid(row=7, column=1, **grid)

        ttk.Separator(parent).grid(row=8, column=0, columnspan=2, sticky='ew', pady=10)

        ttk.Label(parent, text="[辅助 Excel]", foreground="green", font=("Arial", 9, "bold")).grid(
            row=9, column=0, columnspan=2, pady=(0, 5), sticky="w")

        ttk.Label(parent, text="Mask 文件:").grid(row=10, column=0, **grid)
        self.cb_mask_widget = ttk.Combobox(parent, textvariable=self.var_mask_file, width=50)
        self.cb_mask_widget.grid(row=10, column=1, **grid)

        ttk.Label(parent, text="CLTC 模板:").grid(row=11, column=0, **grid)
        self.cb_cltc_widget = ttk.Combobox(parent, textvariable=self.var_cltc_template, width=50)
        self.cb_cltc_widget.grid(row=11, column=1, **grid)

    def setup_tab_physics(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        def add(idx, txt, var, unit):
            ttk.Label(frame, text=txt).grid(row=idx, column=0, sticky="w", pady=5)
            ttk.Entry(frame, textvariable=var, width=15).grid(row=idx, column=1, sticky="w", padx=10)
            ttk.Label(frame, text=unit).grid(row=idx, column=2, sticky="w")

        add(0, "母线电压 (Vdc):", self.var_vdc, "V")
        add(1, "冷却液温度 (Tc):", self.var_tc, "°C")
        add(2, "开关频率 (Fsw):", self.var_fsw, "Hz")
        ttk.Separator(frame).grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)
        add(4, "电机电阻 (R):", self.var_motor_r, "Ohm")
        add(5, "电机电感 (L):", self.var_motor_l, "H")
        ttk.Separator(frame).grid(row=6, column=0, columnspan=3, sticky='ew', pady=10)
        add(7, "默认 Rth (Switch):", self.var_rth_igbt, "K/W")
        add(8, "默认 Rth (Diode):", self.var_rth_diode, "K/W")

        ttk.Checkbutton(frame, text="Diode Rth/Loss 耦合 MOS (Single-Die)", variable=self.var_rth_diode_follow_mos).grid(row=9, column=0, columnspan=2, sticky="w")

        ttk.Label(frame, text="监控对象:").grid(row=10, column=0, sticky="w", pady=10)
        ttk.Combobox(frame, textvariable=self.var_target, values=["Bottom (下管)", "Top (上管)"], state="readonly").grid(row=10, column=1)

        ttk.Label(frame, text="效率计算模式:").grid(row=11, column=0, sticky="w", pady=10)
        ttk.Combobox(frame, textvariable=self.var_loss_scale_mode, values=["Single Device (x6)", "Module Level (x1)"], state="readonly").grid(row=11, column=1)

    def setup_tab_adv(self, parent):
        grp_range = ttk.LabelFrame(parent, text="范围与并行", padding=10)
        grp_range.pack(fill=tk.X, pady=5)

        ttk.Label(grp_range, text="Row Start-End:").grid(row=0, column=0)
        ttk.Entry(grp_range, textvariable=self.var_row_start, width=5).grid(row=0, column=1)
        ttk.Entry(grp_range, textvariable=self.var_row_end, width=5).grid(row=0, column=2, padx=5)

        ttk.Label(grp_range, text="Col Start-End:").grid(row=0, column=3, padx=10)
        ttk.Entry(grp_range, textvariable=self.var_col_start, width=5).grid(row=0, column=4)
        ttk.Entry(grp_range, textvariable=self.var_col_end, width=5).grid(row=0, column=5, padx=5)

        ttk.Label(grp_range, text="CPU Cores:").grid(row=0, column=6, padx=10)
        ttk.Spinbox(grp_range, from_=1, to=64, textvariable=self.var_cpu, width=4).grid(row=0, column=7)

        grp_feat = ttk.LabelFrame(parent, text="功能开关", padding=10)
        grp_feat.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(grp_feat, text="启用 Mask 筛选", variable=self.var_enable_mask).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(grp_feat, text="启用 CLTC 自动填表", variable=self.var_enable_cltc).pack(side=tk.LEFT, padx=10)

        grp_spd = ttk.LabelFrame(parent, text="速度优化 (Low/High Freq)", padding=10)
        grp_spd.pack(fill=tk.X, pady=5)

        f_frame = ttk.Frame(grp_spd)
        f_frame.pack(fill=tk.X)
        ttk.Label(f_frame, text="阈值 (Hz):").pack(side=tk.LEFT)
        ttk.Entry(f_frame, textvariable=self.var_freq_threshold, width=8).pack(side=tk.LEFT, padx=5)

        mat = ttk.Frame(grp_spd)
        mat.pack(fill=tk.X, pady=5)
        ttk.Label(mat, text="参数").grid(row=0, column=0)
        ttk.Label(mat, text="Low Freq", foreground="blue").grid(row=0, column=1, padx=10)
        ttk.Label(mat, text="High Freq", foreground="red").grid(row=0, column=2, padx=10)

        def add_spd(r, txt, v1, v2):
            ttk.Label(mat, text=txt).grid(row=r, column=0, sticky="w")
            ttk.Entry(mat, textvariable=v1, width=10).grid(row=r, column=1, padx=10)
            ttk.Entry(mat, textvariable=v2, width=10).grid(row=r, column=2, padx=10)

        add_spd(1, "dt (s):", self.var_lo_dt, self.var_hi_dt)
        add_spd(2, "Startup:", self.var_lo_startup, self.var_hi_startup)
        add_spd(3, "Iter:", self.var_lo_iter, self.var_hi_iter)
        add_spd(4, "Stat:", self.var_lo_stat, self.var_hi_stat)

        grp_dt = ttk.LabelFrame(parent, text="死区时间 (仅 MOS 模式生效)", padding=10)
        grp_dt.pack(fill=tk.X, pady=5)

        row = ttk.Frame(grp_dt)
        row.pack(fill=tk.X)
        ttk.Checkbutton(row, text="启用死区 (Enable Deadtime)", variable=self.var_enable_deadtime).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(row, text="启用补偿 (Enable DT Compensation)", variable=self.var_enable_dt_comp).pack(side=tk.LEFT, padx=8)
        ttk.Label(row, text="Td (us):").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.var_td_us, width=8).pack(side=tk.LEFT)
        ttk.Label(row, text="I_th (A):").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.var_i_th, width=8).pack(side=tk.LEFT)
        
        ttk.Label(row, text="RMS< (A) Off:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(row, textvariable=self.var_i_rms_th_dt, width=8).pack(side=tk.LEFT)

        ttk.Label(grp_dt, text="说明：\n"
                               "1. I_th: 瞬时电流绝对值 < I_th 时，单点死区失效(理想互补)。\n"
                               "2. RMS< Off: 当工况点电流 I_rms < 此值时，该工况点完全关闭死区(全程理想互补)。\n"
                               "3. 仅 MOS 模式有效，死区续流使用二极管XML。",
                  foreground="gray").pack(anchor="w", pady=(6, 0))

    def select_map_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.var_map_dir.set(p)
            self.scan_dir(p)

    def select_xml_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.var_xml_dir.set(p)
            self.scan_xml(p)

    def select_out_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.var_out_dir.set(p)

    def scan_dir(self, p):
        try:
            csvs = [os.path.basename(x) for x in glob.glob(os.path.join(p, "*.csv"))]
            xls = [os.path.basename(x) for x in glob.glob(os.path.join(p, "*.xlsx"))]

            for cb in [self.cb_csv_v_widget, self.cb_csv_i_widget, self.cb_csv_pf_widget]:
                cb['values'] = csvs
            for cb in [self.cb_mask_widget, self.cb_cltc_widget]:
                cb['values'] = xls

            for f in csvs:
                fl = f.lower()
                if "vol" in fl or "volt" in fl:
                    self.var_csv_v.set(f)
                if "cur" in fl or "curr" in fl:
                    self.var_csv_i.set(f)
                if "pf" in fl or "factor" in fl:
                    self.var_csv_pf.set(f)
        except:
            pass

    def scan_xml(self, p):
        try:
            xmls = [os.path.basename(x) for x in glob.glob(os.path.join(p, "*.xml"))]
            for cb in [self.cb_xml_igbt_widget, self.cb_xml_diode_widget]:
                cb['values'] = xmls

            for f in xmls:
                fl = f.lower()
                if "mos" in fl or "igbt" in fl:
                    self.var_xml_igbt.set(f)
                if "diode" in fl or "body" in fl:
                    self.var_xml_diode.set(f)
        except:
            pass

    def get_cfg(self):
        # Td logic: if deadtime unchecked => TD forced 0
        td_s = float(self.var_td_us.get()) * 1e-6
        enable_dead = bool(self.var_enable_deadtime.get())
        if not enable_dead:
            td_s = 0.0

        return {
            "DIR_MAP": self.var_map_dir.get(),
            "DIR_XML": self.var_xml_dir.get(),
            "DIR_OUT": self.var_out_dir.get(),
            "FILE_XML_IGBT": self.var_xml_igbt.get(),
            "FILE_XML_DIODE": self.var_xml_diode.get(),
            "FILE_CSV_V": self.var_csv_v.get(),
            "FILE_CSV_I": self.var_csv_i.get(),
            "FILE_CSV_PF": self.var_csv_pf.get(),
            "FILE_MASK": self.var_mask_file.get(),
            "FILE_CLTC_TEMP": self.var_cltc_template.get(),
            "VDC": float(self.var_vdc.get()),
            "TC": float(self.var_tc.get()),
            "FSW": float(self.var_fsw.get()),
            "MOTOR_R": float(self.var_motor_r.get()),
            "MOTOR_L": float(self.var_motor_l.get()),
            "TARGET_IDX": 0 if "Bottom" in self.var_target.get() else 1,
            "RTH_IGBT": float(self.var_rth_igbt.get()),
            "RTH_DIODE": float(self.var_rth_diode.get()),
            "RTH_DIODE_FOLLOW_MOS": bool(self.var_rth_diode_follow_mos.get()),
            "LOSS_SCALE": 6.0 if "x6" in self.var_loss_scale_mode.get() else 1.0,
            "ROW_START": int(self.var_row_start.get()),
            "ROW_END": int(self.var_row_end.get()),
            "COL_START": int(self.var_col_start.get()),
            "COL_END": int(self.var_col_end.get()),
            "ENABLE_MASK": bool(self.var_enable_mask.get()),
            "ENABLE_CLTC": bool(self.var_enable_cltc.get()),
            "CPU_CORES": int(self.var_cpu.get()),
            "FREQ_THRESHOLD": float(self.var_freq_threshold.get()),
            "HI_DT": float(self.var_hi_dt.get()),
            "HI_STARTUP": int(self.var_hi_startup.get()),
            "HI_ITER": int(self.var_hi_iter.get()),
            "HI_STAT": int(self.var_hi_stat.get()),
            "LO_DT": float(self.var_lo_dt.get()),
            "LO_STARTUP": int(self.var_lo_startup.get()),
            "LO_ITER": int(self.var_lo_iter.get()),
            "LO_STAT": int(self.var_lo_stat.get()),

            # Deadtime (MOS only)
            "ENABLE_DEADTIME": enable_dead,
            "ENABLE_DT_COMP": bool(self.var_enable_dt_comp.get()),
            "TD": float(td_s),
            "I_TH": float(self.var_i_th.get()),
            "I_RMS_TH_DEADTIME": float(self.var_i_rms_th_dt.get()),
        }

    def log_print(self, msg):
        self.log_text.after(0, lambda: self._append_log(msg))

    def _append_log(self, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, str(msg) + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

    def on_start_click(self):
        cfg = self.get_cfg()
        if not cfg['DIR_MAP'] or not cfg['DIR_XML'] or not cfg['DIR_OUT']:
            messagebox.showwarning("警告", "请先完善路径设置！")
            return

        self.btn_run.config(state="disabled")
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')

        t = threading.Thread(target=self.run_sim_thread, args=(cfg,))
        t.daemon = True
        t.start()

    def run_sim_thread(self, cfg):
        try:
            run_simulation_logic(cfg, self.log_print)
            self.log_print(">>> 仿真任务结束。")
        except Exception as e:
            self.log_print(f"!!! 致命错误: {e}")
            traceback.print_exc()
        finally:
            self.btn_run.config(state="normal")
            messagebox.showinfo("完成", "仿真流程已结束，请检查日志或输出目录。")

    def save_config(self):
        f = filedialog.asksaveasfilename(defaultextension=".json")
        if f:
            with open(f, 'w', encoding='utf-8') as fp:
                json.dump(self.get_cfg(), fp, indent=4, ensure_ascii=False)
            messagebox.showinfo("OK", "配置已保存")

    def load_config(self):
        f = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not f:
            return

        with open(f, 'r', encoding='utf-8') as fp:
            cfg = json.load(fp)

        self.var_map_dir.set(cfg.get("DIR_MAP", ""))
        self.var_xml_dir.set(cfg.get("DIR_XML", ""))
        self.var_out_dir.set(cfg.get("DIR_OUT", ""))

        if self.var_map_dir.get():
            self.scan_dir(self.var_map_dir.get())
        if self.var_xml_dir.get():
            self.scan_xml(self.var_xml_dir.get())

        self.var_csv_v.set(cfg.get("FILE_CSV_V", ""))
        self.var_csv_i.set(cfg.get("FILE_CSV_I", ""))
        self.var_csv_pf.set(cfg.get("FILE_CSV_PF", ""))
        self.var_xml_igbt.set(cfg.get("FILE_XML_IGBT", ""))
        self.var_xml_diode.set(cfg.get("FILE_XML_DIODE", ""))
        self.var_mask_file.set(cfg.get("FILE_MASK", "choose_map_point.xlsx"))
        self.var_cltc_template.set(cfg.get("FILE_CLTC_TEMP", "CLTC_Calculator_Template.xlsx"))

        self.var_vdc.set(cfg.get("VDC", 630.0))
        self.var_tc.set(cfg.get("TC", 65.0))
        self.var_fsw.set(cfg.get("FSW", 10000.0))
        self.var_motor_r.set(cfg.get("MOTOR_R", 20e-3))
        self.var_motor_l.set(cfg.get("MOTOR_L", 0.5e-3))
        self.var_rth_igbt.set(cfg.get("RTH_IGBT", 0.1))
        self.var_rth_diode.set(cfg.get("RTH_DIODE", 0.1))
        self.var_rth_diode_follow_mos.set(cfg.get("RTH_DIODE_FOLLOW_MOS", False))
        
        scale = float(cfg.get("LOSS_SCALE", 1.0))
        self.var_loss_scale_mode.set("Single Device (x6)" if scale > 3.0 else "Module Level (x1)")

        self.var_row_start.set(cfg.get("ROW_START", 55))
        self.var_row_end.set(cfg.get("ROW_END", 114))
        self.var_col_start.set(cfg.get("COL_START", 2))
        self.var_col_end.set(cfg.get("COL_END", 31))
        self.var_enable_mask.set(cfg.get("ENABLE_MASK", True))
        self.var_enable_cltc.set(cfg.get("ENABLE_CLTC", True))
        self.var_cpu.set(cfg.get("CPU_CORES", max(1, multiprocessing.cpu_count() - 1)))

        self.var_freq_threshold.set(cfg.get("FREQ_THRESHOLD", 150.0))
        self.var_hi_dt.set(cfg.get("HI_DT", 1e-8))
        self.var_hi_startup.set(cfg.get("HI_STARTUP", 100))
        self.var_hi_iter.set(cfg.get("HI_ITER", 50))
        self.var_hi_stat.set(cfg.get("HI_STAT", 24))
        self.var_lo_dt.set(cfg.get("LO_DT", 1e-8))
        self.var_lo_startup.set(cfg.get("LO_STARTUP", 10))
        self.var_lo_iter.set(cfg.get("LO_ITER", 10))
        self.var_lo_stat.set(cfg.get("LO_STAT", 6))

        # target
        target_idx = int(cfg.get("TARGET_IDX", 0))
        self.var_target.set("Top (上管)" if target_idx == 1 else "Bottom (下管)")

        # deadtime
        self.var_enable_deadtime.set(bool(cfg.get("ENABLE_DEADTIME", False)))
        self.var_enable_dt_comp.set(bool(cfg.get("ENABLE_DT_COMP", False)))
        td_s = float(cfg.get("TD", 0.0))
        self.var_td_us.set(td_s * 1e6)
        self.var_i_th.set(float(cfg.get("I_TH", 5.0)))
        self.var_i_rms_th_dt.set(float(cfg.get("I_RMS_TH_DEADTIME", 0.0)))

        messagebox.showinfo("OK", "配置已加载")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = SimulationConfigGUI(root)
    root.mainloop()
