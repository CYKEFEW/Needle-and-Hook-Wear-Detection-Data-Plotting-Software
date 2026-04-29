import os
import queue
import threading
import traceback
from typing import Dict, Optional, Tuple

import matplotlib
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, ttk

from analysis import analyze_monitor_data, interpolate_invalid_samples, summary_dict, summary_lines
from db_io import DEFAULT_DB_PATH, export_optimized_db, export_wrap_angle_db, load_monitor_db
from models import AnalysisParams, AnalysisResult, MonitorData, OptimizationParams, OptimizedData, PlotOptions
from optimization import infer_wrap_angle_rad, optimize_monitor_data, recompute_mu_for_wrap_angle
from plotting import (
    downsample_for_plot,
    export_monitor_plots,
    plot_labels,
    plot_mu_axis,
    plot_tension_axis,
    validate_plot_options,
)


matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


class MonitorAnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("针钩监测数据分析与绘图")
        self.root.geometry("1560x980")

        self._queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self._worker_alive = False
        self._opt_worker_alive = False
        self._last_data: Optional[MonitorData] = None
        self._last_result: Optional[AnalysisResult] = None
        self._last_params: Optional[AnalysisParams] = None
        self._opt_source_data: Optional[MonitorData] = None
        self._opt_data: Optional[OptimizedData] = None
        self._opt_params: Optional[OptimizationParams] = None
        self._opt_signature: Optional[Tuple[str, str]] = None
        self._wrap_source_data: Optional[MonitorData] = None
        self._wrap_mu: Optional[np.ndarray] = None
        self._wrap_angle: Optional[float] = None
        self._wrap_signature: Optional[Tuple[str, str]] = None

        default_db = DEFAULT_DB_PATH if os.path.exists(DEFAULT_DB_PATH) else ""
        self.db_path_var = tk.StringVar(value=default_db)
        self.table_name_var = tk.StringVar(value="Data")
        self.status_var = tk.StringVar(value="选择数据库后点击“分析并绘图”。")
        self.export_lang_var = tk.StringVar(value="中文")
        self.plot_preview_lang_var = tk.StringVar(value="中文")
        self.plot_preview_mode_var = tk.StringVar(value="张力")
        self._param_vars: Dict[str, tk.Variable] = {}
        self._plot_limit_vars: Dict[str, tk.StringVar] = {}
        self._plot_bool_vars: Dict[str, tk.BooleanVar] = {}
        self._export_bool_vars: Dict[str, tk.BooleanVar] = {}

        self.opt_db_path_var = tk.StringVar(value=default_db)
        self.opt_table_name_var = tk.StringVar(value="Data")
        self.opt_output_path_var = tk.StringVar(value="")
        self.opt_wrap_angle_var = tk.StringVar(value="")
        self.opt_hampel_enabled_var = tk.BooleanVar(value=True)
        self.opt_smooth_enabled_var = tk.BooleanVar(value=True)
        self.opt_hampel_strength_var = tk.DoubleVar(value=50.0)
        self.opt_smooth_strength_var = tk.DoubleVar(value=50.0)
        self.opt_hampel_strength_text_var = tk.StringVar(value="50%")
        self.opt_smooth_strength_text_var = tk.StringVar(value="50%")
        self.opt_hampel_window_var = tk.StringVar(value="1.0")
        self.opt_hampel_sigma_var = tk.StringVar(value="3.0")
        self.opt_smooth_window_var = tk.StringVar(value="0.5")
        self.opt_sample_rate_var = tk.StringVar(value="0")
        self.opt_preview_mode_var = tk.StringVar(value="高张力")
        self.opt_status_var = tk.StringVar(value="选择数据库后可反推包角、生成优化预览并导出。")

        self.opt_preview_lang_var = tk.StringVar(value="中文")
        self.wrap_db_path_var = tk.StringVar(value=default_db)
        self.wrap_table_name_var = tk.StringVar(value="Data")
        self.wrap_output_path_var = tk.StringVar(value="")
        self.wrap_angle_var = tk.StringVar(value="")
        self.wrap_status_var = tk.StringVar(value="选择数据库并输入新包角后，可预览并导出重算 μ 的数据库。")

        self.wrap_preview_lang_var = tk.StringVar(value="中文")
        self._build_ui()
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_mousewheel, add="+")
        self.root.after(120, self._poll_queue)

    def _build_ui(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        plot_tab = ttk.Frame(self.notebook)
        opt_tab = ttk.Frame(self.notebook)
        wrap_tab = ttk.Frame(self.notebook)
        self.notebook.add(plot_tab, text="绘图区")
        self.notebook.add(opt_tab, text="数据优化区")

        self._build_plot_tab(plot_tab)
        self.notebook.add(wrap_tab, text="包角重算区")
        self._build_optimization_tab(opt_tab)
        self._build_wrap_angle_tab(wrap_tab)

    def _build_plot_tab(self, parent: ttk.Frame) -> None:
        shell = ttk.Frame(parent)
        shell.pack(fill=tk.BOTH, expand=True)

        self.viewport_canvas = tk.Canvas(shell, highlightthickness=0)
        self.viewport_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(shell, orient=tk.VERTICAL, command=self.viewport_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.viewport_canvas.configure(yscrollcommand=scrollbar.set)

        self.viewport_body = ttk.Frame(self.viewport_canvas, padding=10)
        self.viewport_window = self.viewport_canvas.create_window((0, 0), window=self.viewport_body, anchor="nw")
        self.viewport_body.bind("<Configure>", self._on_body_configure)
        self.viewport_canvas.bind("<Configure>", self._on_canvas_configure)
        self.viewport_body.columnconfigure(0, weight=0)
        self.viewport_body.columnconfigure(1, weight=1)

        left = ttk.Frame(self.viewport_body)
        right = ttk.Frame(self.viewport_body)
        left.grid(row=0, column=0, sticky="nw", padx=(0, 14))
        right.grid(row=0, column=1, sticky="nsew")

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _on_body_configure(self, _event=None) -> None:
        self.viewport_canvas.configure(scrollregion=self.viewport_canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.viewport_canvas.itemconfigure(self.viewport_window, width=event.width)

    def _on_mousewheel(self, event) -> None:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if widget is not None and str(widget).startswith(str(getattr(self, "opt_preview_canvas", ""))):
            return
        delta = 0
        if getattr(event, "delta", 0):
            delta = -int(event.delta / 120)
        elif getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        if delta != 0 and hasattr(self, "viewport_canvas"):
            self.viewport_canvas.yview_scroll(delta, "units")

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        file_box = ttk.LabelFrame(parent, text="数据源", padding=10)
        file_box.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(file_box, text="数据库").grid(row=0, column=0, sticky="w")
        ttk.Entry(file_box, textvariable=self.db_path_var, width=42).grid(row=0, column=1, sticky="ew", padx=(8, 6))
        ttk.Button(file_box, text="浏览...", command=self._browse_db).grid(row=0, column=2, sticky="ew")

        ttk.Label(file_box, text="表名").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(file_box, textvariable=self.table_name_var, width=14).grid(row=1, column=1, sticky="w", pady=(6, 0), padx=(8, 0))
        file_box.columnconfigure(1, weight=1)

        calc_box = ttk.LabelFrame(parent, text="判据参数", padding=10)
        calc_box.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        param_items = [
            ("稳态窗口 Wss (s)", "stable_win_s", 1200.0),
            ("稳态最短保持 Whold (s)", "stable_hold_s", 3600.0),
            ("稳态标准差阈值 σmax", "stable_sigma_max", 0.03),
            ("稳态总漂移阈值 Δμmax", "stable_slope_max", 0.015),
            ("稳态有效比例 qmin", "stable_valid_min", 0.9),
            ("失效阈值增量 δ", "fail_delta", 0.25),
            ("持续超阈值 Wpersist (s)", "fail_hold_s", 300.0),
            ("最大容忍中断 Wbreak (s)", "fail_break_s", 1.0),
            ("采样率 fs (Hz, 0=自动推断)", "sample_rate_hz", 0.0),
            ("绘图最大点数", "max_plot_points", 40000),
        ]
        for i, (label, key, default) in enumerate(param_items):
            row = i // 2
            col = (i % 2) * 2
            var = tk.StringVar(value=str(default))
            self._param_vars[key] = var
            ttk.Label(calc_box, text=label).grid(row=row, column=col, sticky="w", pady=2, padx=(0, 6))
            ttk.Entry(calc_box, textvariable=var, width=12).grid(row=row, column=col + 1, sticky="ew", pady=2, padx=(0, 12))
        calc_box.columnconfigure(1, weight=1)
        calc_box.columnconfigure(3, weight=1)

        plot_box = ttk.LabelFrame(parent, text="绘图显示", padding=10)
        plot_box.pack(fill=tk.X, pady=(0, 10))
        limit_items = [
            ("张力Y下限", "tension_y_min"),
            ("张力Y上限", "tension_y_max"),
            ("高张力Y下限", "high_tension_y_min"),
            ("高张力Y上限", "high_tension_y_max"),
            ("低张力Y下限", "low_tension_y_min"),
            ("低张力Y上限", "low_tension_y_max"),
            ("μY下限", "mu_y_min"),
            ("μY上限", "mu_y_max"),
        ]
        for i, (label, key) in enumerate(limit_items):
            row = i // 2
            col = (i % 2) * 2
            var = tk.StringVar(value="Auto")
            self._plot_limit_vars[key] = var
            ttk.Label(plot_box, text=label).grid(row=row, column=col, sticky="w", pady=2, padx=(0, 6))
            entry = ttk.Entry(plot_box, textvariable=var, width=12)
            entry.grid(row=row, column=col + 1, sticky="ew", pady=2, padx=(0, 12))
            entry.bind("<Return>", lambda _event: self._refresh_current_plot())
            entry.bind("<FocusOut>", lambda _event: self._refresh_current_plot())
        for i, (key, text) in enumerate(
            [
                ("show_mu", "μ"),
                ("show_stable_segments", "连续稳定段"),
                ("show_mu_ss", "稳定段基线"),
                ("show_tlife", "tlife"),
                ("show_mu_th", "超限阈值"),
            ],
            start=0,
        ):
            var = tk.BooleanVar(value=True)
            self._plot_bool_vars[key] = var
            ttk.Checkbutton(plot_box, text=text, variable=var, command=self._refresh_current_plot).grid(
                row=4 + i // 3, column=i % 3, sticky="w", pady=(6 if i < 3 else 2, 2), padx=(0, 12)
            )
        plot_box.columnconfigure(1, weight=1)
        plot_box.columnconfigure(3, weight=1)

        action_box = ttk.LabelFrame(parent, text="操作与导出", padding=10)
        action_box.pack(fill=tk.X, pady=(0, 10))
        self.analyze_btn = ttk.Button(action_box, text="分析并绘图", command=self._run_analysis_async)
        self.analyze_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=(0, 6))
        ttk.Button(action_box, text="清空结果", command=self._clear_result).grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=(0, 6))
        ttk.Label(action_box, text="导出语言").grid(row=0, column=2, sticky="e", pady=(0, 6))
        ttk.Combobox(action_box, textvariable=self.export_lang_var, values=("中文", "English"), state="readonly", width=9).grid(
            row=0, column=3, sticky="ew", padx=(8, 8), pady=(0, 6)
        )
        ttk.Button(action_box, text="导出图片...", command=self._export_plots).grid(row=0, column=4, sticky="ew", pady=(0, 6))

        ttk.Label(action_box, text="导出内容").grid(row=1, column=0, sticky="w", pady=(2, 0))
        export_items = [
            ("tension", "张力", True),
            ("high", "高张力", False),
            ("low", "低张力", False),
            ("avg", "平均张力", False),
            ("mu", "摩擦系数", True),
        ]
        for i, (key, text, default) in enumerate(export_items):
            var = tk.BooleanVar(value=default)
            self._export_bool_vars[key] = var
            ttk.Checkbutton(action_box, text=text, variable=var).grid(
                row=1, column=i + 1, sticky="w", padx=(0, 10), pady=(2, 0)
            )
        action_box.columnconfigure(3, weight=1)

        result_box = ttk.LabelFrame(parent, text="结果摘要", padding=10)
        result_box.pack(fill=tk.BOTH, expand=True)
        self.summary_text = tk.Text(result_box, width=44, height=9, wrap=tk.WORD)
        summary_scroll = ttk.Scrollbar(result_box, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_text.configure(state=tk.DISABLED)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, anchor="w")
        ttk.Label(top, text="预览语言").pack(side=tk.RIGHT, padx=(8, 0))
        self.plot_preview_lang_combo = ttk.Combobox(
            top,
            textvariable=self.plot_preview_lang_var,
            values=("English", "中文"),
            state="readonly",
            width=10,
        )
        self.plot_preview_lang_combo.pack(side=tk.RIGHT)
        self.plot_preview_lang_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_current_plot())
        ttk.Label(top, text="预览曲线").pack(side=tk.RIGHT, padx=(18, 0))
        self.plot_preview_selector = ttk.Combobox(
            top,
            textvariable=self.plot_preview_mode_var,
            values=("张力", "高张力", "低张力", "平均张力", "μ"),
            state="readonly",
            width=8,
        )
        self.plot_preview_selector.pack(side=tk.RIGHT)
        self.plot_preview_selector.bind("<<ComboboxSelected>>", lambda _event: self._refresh_current_plot())

        preview_figsize = tuple(float(v) for v in matplotlib.rcParams["figure.figsize"])
        preview_dpi = float(matplotlib.rcParams["figure.dpi"])
        self.figure = Figure(figsize=preview_figsize, dpi=preview_dpi)
        self.ax_plot_preview = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        plot_widget = self.canvas.get_tk_widget()
        plot_widget.configure(width=int(preview_figsize[0] * preview_dpi), height=int(preview_figsize[1] * preview_dpi))
        plot_widget.pack(anchor="n", pady=(8, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X)
        self._draw_placeholder()

    def _build_optimization_tab(self, parent: ttk.Frame) -> None:
        shell = ttk.Frame(parent, padding=10)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(0, weight=1)

        controls = ttk.Frame(shell)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        preview = ttk.Frame(shell)
        preview.grid(row=0, column=1, sticky="nsew")

        source_box = ttk.LabelFrame(controls, text="优化数据源", padding=10)
        source_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(source_box, text="源数据库").grid(row=0, column=0, sticky="w")
        ttk.Entry(source_box, textvariable=self.opt_db_path_var, width=48).grid(row=1, column=0, sticky="ew", pady=(4, 6))
        ttk.Button(source_box, text="浏览...", command=self._browse_opt_db).grid(row=1, column=1, padx=(8, 0), sticky="ew")
        ttk.Label(source_box, text="表名").grid(row=2, column=0, sticky="w")
        ttk.Entry(source_box, textvariable=self.opt_table_name_var, width=18).grid(row=2, column=0, sticky="w", pady=(4, 0))
        source_box.columnconfigure(0, weight=1)

        param_box = ttk.LabelFrame(controls, text="优化参数", padding=10)
        param_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(param_box, text="包角 θ (rad)").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(param_box, textvariable=self.opt_wrap_angle_var, width=18).grid(row=0, column=1, sticky="ew", pady=2, padx=(8, 0))
        ttk.Button(param_box, text="自动反推", command=self._infer_wrap_angle_async).grid(row=0, column=2, padx=(8, 0))
        ttk.Checkbutton(param_box, text="自动去除毛刺", variable=self.opt_hampel_enabled_var).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(8, 2)
        )
        ttk.Label(param_box, text="去毛刺窗口 (s)").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(param_box, textvariable=self.opt_hampel_window_var, width=18).grid(row=2, column=1, sticky="ew", pady=2, padx=(8, 0))
        ttk.Label(param_box, text="去毛刺阈值 σ").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(param_box, textvariable=self.opt_hampel_sigma_var, width=18).grid(row=3, column=1, sticky="ew", pady=2, padx=(8, 0))
        ttk.Checkbutton(param_box, text="自动平滑曲线", variable=self.opt_smooth_enabled_var).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(8, 2)
        )
        ttk.Label(param_box, text="平滑窗口 (s)").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Entry(param_box, textvariable=self.opt_smooth_window_var, width=18).grid(row=5, column=1, sticky="ew", pady=2, padx=(8, 0))
        ttk.Label(param_box, text="采样率 fs (Hz, 0=自动)").grid(row=6, column=0, sticky="w", pady=2)
        ttk.Entry(param_box, textvariable=self.opt_sample_rate_var, width=18).grid(row=6, column=1, sticky="ew", pady=2, padx=(8, 0))
        ttk.Label(param_box, text="去毛刺力度").grid(row=7, column=0, sticky="w", pady=(8, 2))
        ttk.Scale(
            param_box,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.opt_hampel_strength_var,
            command=lambda _value: self._update_strength_text(
                self.opt_hampel_strength_var, self.opt_hampel_strength_text_var
            ),
        ).grid(row=7, column=1, sticky="ew", pady=(8, 2), padx=(8, 0))
        ttk.Label(param_box, textvariable=self.opt_hampel_strength_text_var, width=6).grid(row=7, column=2, sticky="e", pady=(8, 2))
        ttk.Label(param_box, text="平滑力度").grid(row=8, column=0, sticky="w", pady=2)
        ttk.Scale(
            param_box,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.opt_smooth_strength_var,
            command=lambda _value: self._update_strength_text(
                self.opt_smooth_strength_var, self.opt_smooth_strength_text_var
            ),
        ).grid(row=8, column=1, sticky="ew", pady=2, padx=(8, 0))
        ttk.Label(param_box, textvariable=self.opt_smooth_strength_text_var, width=6).grid(row=8, column=2, sticky="e", pady=2)
        param_box.columnconfigure(1, weight=1)

        export_box = ttk.LabelFrame(controls, text="导出", padding=10)
        export_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(export_box, text="输出数据库").grid(row=0, column=0, sticky="w")
        ttk.Entry(export_box, textvariable=self.opt_output_path_var, width=48).grid(row=1, column=0, sticky="ew", pady=(4, 6))
        ttk.Button(export_box, text="另存为...", command=self._browse_opt_output).grid(row=1, column=1, padx=(8, 0))
        self.opt_preview_btn = ttk.Button(export_box, text="生成优化预览", command=self._run_optimization_preview_async)
        self.opt_preview_btn.grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.opt_export_btn = ttk.Button(export_box, text="导出优化后的db", command=self._export_optimized_db)
        self.opt_export_btn.grid(row=2, column=1, sticky="e", pady=(4, 0))
        export_box.columnconfigure(0, weight=1)

        status_box = ttk.LabelFrame(controls, text="优化摘要", padding=10)
        status_box.pack(fill=tk.BOTH, expand=True)
        ttk.Label(status_box, textvariable=self.opt_status_var, wraplength=360, justify=tk.LEFT).pack(fill=tk.X)

        preview_switch = ttk.Frame(preview)
        preview_switch.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(preview_switch, text="预览曲线").pack(side=tk.LEFT)
        self.opt_preview_selector = ttk.Combobox(
            preview_switch,
            textvariable=self.opt_preview_mode_var,
            values=("高张力", "低张力", "平均张力", "μ"),
            state="readonly",
            width=10,
        )
        self.opt_preview_selector.pack(side=tk.LEFT, padx=(8, 0))
        self.opt_preview_selector.bind("<<ComboboxSelected>>", lambda _event: self._refresh_optimization_preview_plot())
        ttk.Label(preview_switch, text="预览语言").pack(side=tk.LEFT, padx=(18, 0))
        self.opt_preview_lang_combo = ttk.Combobox(
            preview_switch,
            textvariable=self.opt_preview_lang_var,
            values=("English", "中文"),
            state="readonly",
            width=10,
        )
        self.opt_preview_lang_combo.pack(side=tk.LEFT, padx=(8, 0))
        self.opt_preview_lang_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_optimization_preview_plot())

        self.opt_figure = Figure(figsize=(10.8, 7.2), dpi=100)
        self.opt_ax_preview = self.opt_figure.add_subplot(111)
        self.opt_preview_canvas = FigureCanvasTkAgg(self.opt_figure, master=preview)
        self.opt_preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.opt_toolbar = NavigationToolbar2Tk(self.opt_preview_canvas, preview, pack_toolbar=False)
        self.opt_toolbar.update()
        self.opt_toolbar.pack(fill=tk.X)
        self._draw_optimization_placeholder()

    def _build_wrap_angle_tab(self, parent: ttk.Frame) -> None:
        shell = ttk.Frame(parent, padding=10)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(0, weight=1)

        controls = ttk.Frame(shell)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        preview = ttk.Frame(shell)
        preview.grid(row=0, column=1, sticky="nsew")

        source_box = ttk.LabelFrame(controls, text="包角重算数据源", padding=10)
        source_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(source_box, text="源数据库").grid(row=0, column=0, sticky="w")
        ttk.Entry(source_box, textvariable=self.wrap_db_path_var, width=48).grid(row=1, column=0, sticky="ew", pady=(4, 6))
        ttk.Button(source_box, text="浏览...", command=self._browse_wrap_db).grid(row=1, column=1, padx=(8, 0), sticky="ew")
        ttk.Label(source_box, text="表名").grid(row=2, column=0, sticky="w")
        ttk.Entry(source_box, textvariable=self.wrap_table_name_var, width=18).grid(row=2, column=0, sticky="w", pady=(4, 0))
        source_box.columnconfigure(0, weight=1)

        param_box = ttk.LabelFrame(controls, text="包角参数", padding=10)
        param_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(param_box, text="新包角 θ (rad)").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(param_box, textvariable=self.wrap_angle_var, width=18).grid(row=0, column=1, sticky="ew", pady=2, padx=(8, 0))
        ttk.Button(param_box, text="自动反推当前包角", command=self._infer_wrap_recalc_angle_async).grid(row=0, column=2, padx=(8, 0))
        param_box.columnconfigure(1, weight=1)

        export_box = ttk.LabelFrame(controls, text="预览与导出", padding=10)
        export_box.pack(fill=tk.X, pady=(0, 10))
        self.wrap_preview_btn = ttk.Button(export_box, text="生成 μ 预览", command=self._run_wrap_preview_async)
        self.wrap_preview_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.wrap_export_btn = ttk.Button(export_box, text="导出包角重算db", command=self._export_wrap_angle_db)
        self.wrap_export_btn.grid(row=0, column=1, sticky="ew")
        export_box.columnconfigure(0, weight=1)
        export_box.columnconfigure(1, weight=1)

        status_box = ttk.LabelFrame(controls, text="包角重算摘要", padding=10)
        status_box.pack(fill=tk.BOTH, expand=True)
        ttk.Label(status_box, textvariable=self.wrap_status_var, wraplength=360, justify=tk.LEFT).pack(fill=tk.X)

        preview_switch = ttk.Frame(preview)
        preview_switch.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(preview_switch, text="预览语言").pack(side=tk.LEFT)
        self.wrap_preview_lang_combo = ttk.Combobox(
            preview_switch,
            textvariable=self.wrap_preview_lang_var,
            values=("English", "中文"),
            state="readonly",
            width=10,
        )
        self.wrap_preview_lang_combo.pack(side=tk.LEFT, padx=(8, 0))
        self.wrap_preview_lang_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_wrap_preview_plot())

        self.wrap_figure = Figure(figsize=(10.8, 7.2), dpi=100)
        self.wrap_ax_mu = self.wrap_figure.add_subplot(111)
        self.wrap_canvas = FigureCanvasTkAgg(self.wrap_figure, master=preview)
        self.wrap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.wrap_toolbar = NavigationToolbar2Tk(self.wrap_canvas, preview, pack_toolbar=False)
        self.wrap_toolbar.update()
        self.wrap_toolbar.pack(fill=tk.X)
        self._draw_wrap_placeholder()

    def _add_entry(self, parent: ttk.Widget, row: int, label: str, key: str, default) -> int:
        var = tk.StringVar(value=str(default))
        self._param_vars[key] = var
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=18).grid(row=row, column=1, sticky="ew", pady=2, padx=(8, 0))
        return row + 1

    def _add_plot_limit(self, parent: ttk.Widget, row: int, label: str, key: str) -> None:
        var = tk.StringVar(value="Auto")
        self._plot_limit_vars[key] = var
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=var, width=18)
        entry.grid(row=row, column=1, sticky="ew", pady=2, padx=(8, 0))
        entry.bind("<Return>", lambda _event: self._refresh_current_plot())
        entry.bind("<FocusOut>", lambda _event: self._refresh_current_plot())

    def _update_strength_text(self, value_var: tk.Variable, text_var: tk.StringVar) -> None:
        text_var.set(f"{int(round(float(value_var.get())))}%")

    def _preview_lang_code(self, var: tk.StringVar) -> str:
        return "en" if var.get().strip().lower().startswith("english") else "zh"

    def _browse_db(self) -> None:
        path = filedialog.askopenfilename(title="选择监测数据库", filetypes=[("SQLite DB", "*.db *.sqlite *.sqlite3"), ("所有文件", "*.*")])
        if path:
            self.db_path_var.set(path)
            self.opt_db_path_var.set(path)
            self.wrap_db_path_var.set(path)

    def _browse_opt_db(self) -> None:
        path = filedialog.askopenfilename(title="选择待优化数据库", filetypes=[("SQLite DB", "*.db *.sqlite *.sqlite3"), ("所有文件", "*.*")])
        if path:
            self.opt_db_path_var.set(path)
            self._clear_optimization_preview_state()
            if not self.opt_output_path_var.get().strip():
                base, ext = os.path.splitext(path)
                self.opt_output_path_var.set(base + "_optimized" + (ext or ".db"))

    def _browse_opt_output(self) -> None:
        default_path = self._default_optimized_output_path()
        path = filedialog.asksaveasfilename(
            title="保存优化后的数据库",
            defaultextension=".db",
            initialdir=os.path.dirname(default_path),
            initialfile=os.path.basename(default_path),
            filetypes=[("SQLite DB", "*.db *.sqlite *.sqlite3"), ("所有文件", "*.*")],
        )
        if path:
            self.opt_output_path_var.set(path)

    def _browse_wrap_db(self) -> None:
        path = filedialog.askopenfilename(title="选择待重算包角数据库", filetypes=[("SQLite DB", "*.db *.sqlite *.sqlite3"), ("所有文件", "*.*")])
        if path:
            self.wrap_db_path_var.set(path)
            self._clear_wrap_preview_state()
            if not self.wrap_output_path_var.get().strip():
                self.wrap_output_path_var.set(self._default_wrap_output_path(path))

    def _browse_wrap_output(self) -> None:
        default_path = self._default_wrap_output_path()
        path = filedialog.asksaveasfilename(
            title="保存包角重算后的数据库",
            defaultextension=".db",
            initialdir=os.path.dirname(default_path),
            initialfile=os.path.basename(default_path),
            filetypes=[("SQLite DB", "*.db *.sqlite *.sqlite3"), ("所有文件", "*.*")],
        )
        if path:
            self.wrap_output_path_var.set(path)

    def _get_params(self) -> AnalysisParams:
        def f(name: str) -> float:
            return float(str(self._param_vars[name].get()).strip())

        return AnalysisParams(
            stable_win_s=f("stable_win_s"),
            stable_hold_s=f("stable_hold_s"),
            stable_sigma_max=f("stable_sigma_max"),
            stable_slope_max=f("stable_slope_max"),
            stable_valid_min=f("stable_valid_min"),
            fail_delta=f("fail_delta"),
            fail_hold_s=f("fail_hold_s"),
            fail_break_s=f("fail_break_s"),
            sample_rate_hz=f("sample_rate_hz"),
            max_plot_points=int(float(str(self._param_vars["max_plot_points"].get()).strip())),
        )

    def _get_plot_options(self) -> PlotOptions:
        def f(name: str) -> Optional[float]:
            text = str(self._plot_limit_vars[name].get()).strip()
            if text == "" or text.lower() == "auto":
                return None
            return float(text)

        options = PlotOptions(
            tension_y_min=f("tension_y_min"),
            tension_y_max=f("tension_y_max"),
            high_tension_y_min=f("high_tension_y_min"),
            high_tension_y_max=f("high_tension_y_max"),
            low_tension_y_min=f("low_tension_y_min"),
            low_tension_y_max=f("low_tension_y_max"),
            mu_y_min=f("mu_y_min"),
            mu_y_max=f("mu_y_max"),
            show_mu=bool(self._plot_bool_vars["show_mu"].get()),
            show_stable_segments=bool(self._plot_bool_vars["show_stable_segments"].get()),
            show_mu_ss=bool(self._plot_bool_vars["show_mu_ss"].get()),
            show_tlife=bool(self._plot_bool_vars["show_tlife"].get()),
            show_mu_th=bool(self._plot_bool_vars["show_mu_th"].get()),
        )
        validate_plot_options(options)
        return options

    def _get_optimization_params(self) -> OptimizationParams:
        wrap_text = self.opt_wrap_angle_var.get().strip()
        if not wrap_text:
            raise ValueError("请先自动反推或手动输入包角 θ。")
        return OptimizationParams(
            wrap_angle_rad=float(wrap_text),
            enable_hampel=bool(self.opt_hampel_enabled_var.get()),
            enable_smoothing=bool(self.opt_smooth_enabled_var.get()),
            hampel_window_s=float(self.opt_hampel_window_var.get().strip()),
            hampel_sigma=float(self.opt_hampel_sigma_var.get().strip()),
            smooth_window_s=float(self.opt_smooth_window_var.get().strip()),
            sample_rate_hz=float(self.opt_sample_rate_var.get().strip()),
            hampel_strength_pct=float(self.opt_hampel_strength_var.get()),
            smooth_strength_pct=float(self.opt_smooth_strength_var.get()),
        ).normalized()

    def _current_optimization_signature(self) -> Tuple[str, str]:
        db_path = self.opt_db_path_var.get().strip() or self.db_path_var.get().strip()
        table_name = self.opt_table_name_var.get().strip() or "Data"
        return (os.path.abspath(db_path), table_name)

    def _clear_optimization_preview_state(self) -> None:
        self._opt_source_data = None
        self._opt_data = None
        self._opt_params = None
        self._opt_signature = None

    def _current_wrap_signature(self) -> Tuple[str, str]:
        db_path = self.wrap_db_path_var.get().strip() or self.db_path_var.get().strip()
        table_name = self.wrap_table_name_var.get().strip() or "Data"
        return (os.path.abspath(db_path), table_name)

    def _clear_wrap_preview_state(self) -> None:
        self._wrap_source_data = None
        self._wrap_mu = None
        self._wrap_angle = None
        self._wrap_signature = None

    def _default_optimized_output_path(self) -> str:
        source_path = ""
        if self._opt_source_data is not None:
            source_path = self._opt_source_data.db_path
        if not source_path:
            source_path = self.opt_db_path_var.get().strip() or self.db_path_var.get().strip()
        if not source_path:
            source_path = os.path.abspath("optimized.db")
        base, ext = os.path.splitext(os.path.abspath(source_path))
        return base + "_optimized" + (ext or ".db")

    def _default_wrap_output_path(self, source_path: str = "") -> str:
        if self._wrap_source_data is not None:
            source_path = self._wrap_source_data.db_path
        if not source_path:
            source_path = self.wrap_db_path_var.get().strip() or self.db_path_var.get().strip()
        if not source_path:
            source_path = os.path.abspath("wrap_angle.db")
        base, ext = os.path.splitext(os.path.abspath(source_path))
        return base + "_wrap_angle" + (ext or ".db")

    def _run_analysis_async(self) -> None:
        if self._worker_alive:
            return
        db_path = self.db_path_var.get().strip()
        table_name = self.table_name_var.get().strip() or "Data"
        if not db_path:
            messagebox.showwarning("提示", "请先选择数据库文件。")
            return
        try:
            params = self._get_params()
            self._get_plot_options()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        self._worker_alive = True
        self.analyze_btn.configure(state=tk.DISABLED)
        self.status_var.set("正在从数据库提取数据并绘制图表，请稍候...")
        threading.Thread(target=self._analysis_worker, args=(db_path, table_name, params), daemon=True).start()

    def _analysis_worker(self, db_path: str, table_name: str, params: AnalysisParams) -> None:
        try:
            data = load_monitor_db(db_path, table_name=table_name)
            result = analyze_monitor_data(data, params)
            self._queue.put(("analysis_ok", (data, result, params.normalized())))
        except Exception:
            self._queue.put(("analysis_err", traceback.format_exc()))

    def _infer_wrap_angle_async(self) -> None:
        if self._opt_worker_alive:
            return
        db_path = self.opt_db_path_var.get().strip() or self.db_path_var.get().strip()
        table_name = self.opt_table_name_var.get().strip() or "Data"
        if not db_path:
            messagebox.showwarning("提示", "请先选择待优化数据库。")
            return
        self._opt_worker_alive = True
        self.opt_status_var.set("正在读取数据库并反推包角...")
        threading.Thread(target=self._infer_wrap_angle_worker, args=(db_path, table_name), daemon=True).start()

    def _infer_wrap_angle_worker(self, db_path: str, table_name: str) -> None:
        try:
            data = load_monitor_db(db_path, table_name=table_name)
            theta = infer_wrap_angle_rad(data)
            self._queue.put(("theta_ok", (data, theta)))
        except Exception:
            self._queue.put(("opt_err", traceback.format_exc()))

    def _run_optimization_preview_async(self) -> None:
        if self._opt_worker_alive:
            return
        db_path = self.opt_db_path_var.get().strip() or self.db_path_var.get().strip()
        table_name = self.opt_table_name_var.get().strip() or "Data"
        if not db_path:
            messagebox.showwarning("提示", "请先选择待优化数据库。")
            return
        try:
            params = self._get_optimization_params()
        except Exception as exc:
            messagebox.showerror("优化参数错误", str(exc))
            return
        self._opt_worker_alive = True
        self.opt_preview_btn.configure(state=tk.DISABLED)
        self.opt_export_btn.configure(state=tk.DISABLED)
        self.opt_status_var.set("正在生成优化前/优化后对比预览...")
        threading.Thread(target=self._optimization_preview_worker, args=(db_path, table_name, params), daemon=True).start()

    def _optimization_preview_worker(self, db_path: str, table_name: str, params: OptimizationParams) -> None:
        try:
            data = load_monitor_db(db_path, table_name=table_name)
            optimized = optimize_monitor_data(data, params)
            self._queue.put(("opt_preview_ok", (data, optimized, params)))
        except Exception:
            self._queue.put(("opt_err", traceback.format_exc()))

    def _infer_wrap_recalc_angle_async(self) -> None:
        if self._opt_worker_alive:
            return
        db_path = self.wrap_db_path_var.get().strip() or self.db_path_var.get().strip()
        table_name = self.wrap_table_name_var.get().strip() or "Data"
        if not db_path:
            messagebox.showwarning("提示", "请先选择待重算包角数据库。")
            return
        self._opt_worker_alive = True
        self.wrap_status_var.set("正在读取数据库并反推当前包角...")
        threading.Thread(target=self._infer_wrap_recalc_angle_worker, args=(db_path, table_name), daemon=True).start()

    def _infer_wrap_recalc_angle_worker(self, db_path: str, table_name: str) -> None:
        try:
            data = load_monitor_db(db_path, table_name=table_name)
            theta = infer_wrap_angle_rad(data)
            self._queue.put(("wrap_theta_ok", (data, theta)))
        except Exception:
            self._queue.put(("wrap_err", traceback.format_exc()))

    def _run_wrap_preview_async(self) -> None:
        if self._opt_worker_alive:
            return
        db_path = self.wrap_db_path_var.get().strip() or self.db_path_var.get().strip()
        table_name = self.wrap_table_name_var.get().strip() or "Data"
        if not db_path:
            messagebox.showwarning("提示", "请先选择待重算包角数据库。")
            return
        try:
            theta = float(self.wrap_angle_var.get().strip())
        except Exception as exc:
            messagebox.showerror("包角参数错误", str(exc))
            return
        self._opt_worker_alive = True
        self.wrap_preview_btn.configure(state=tk.DISABLED)
        self.wrap_export_btn.configure(state=tk.DISABLED)
        self.wrap_status_var.set("正在重算 μ 并生成预览...")
        threading.Thread(target=self._wrap_preview_worker, args=(db_path, table_name, theta), daemon=True).start()

    def _wrap_preview_worker(self, db_path: str, table_name: str, theta: float) -> None:
        try:
            data = load_monitor_db(db_path, table_name=table_name)
            mu_new = recompute_mu_for_wrap_angle(data, theta)
            self._queue.put(("wrap_preview_ok", (data, mu_new, theta)))
        except Exception:
            self._queue.put(("wrap_err", traceback.format_exc()))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind.startswith("analysis"):
                    self._worker_alive = False
                    self.analyze_btn.configure(state=tk.NORMAL)
                if kind in {"theta_ok", "opt_preview_ok", "opt_err"}:
                    self._opt_worker_alive = False
                    self.opt_preview_btn.configure(state=tk.NORMAL)
                    self.opt_export_btn.configure(state=tk.NORMAL)
                if kind in {"wrap_theta_ok", "wrap_preview_ok", "wrap_err"}:
                    self._opt_worker_alive = False
                    self.wrap_preview_btn.configure(state=tk.NORMAL)
                    self.wrap_export_btn.configure(state=tk.NORMAL)

                if kind == "analysis_ok":
                    data, result, params = payload
                    self._last_data = data
                    self._last_result = result
                    self._last_params = params
                    self._update_summary(data, result)
                    self._update_plots(data, result)
                    self.status_var.set(f"分析完成: {os.path.basename(data.db_path)} | rows={data.row_count} | fs={result.fs_hz:.3f} Hz")
                elif kind == "analysis_err":
                    self.status_var.set("分析失败")
                    messagebox.showerror("分析失败", str(payload))
                elif kind == "theta_ok":
                    data, theta = payload
                    self._opt_source_data = data
                    self.opt_wrap_angle_var.set(f"{float(theta):.16g}")
                    self.opt_status_var.set(f"包角反推完成: θ={float(theta):.12g} rad | rows={data.row_count}")
                elif kind == "opt_preview_ok":
                    data, optimized, params = payload
                    self._opt_source_data = data
                    self._opt_data = optimized
                    self._opt_params = params
                    self._opt_signature = (os.path.abspath(data.db_path), data.table_name)
                    self.opt_preview_mode_var.set("高张力")
                    self._draw_optimization_preview(data, optimized)
                    self.opt_status_var.set(
                        f"预览完成: rows={data.row_count} | θ={params.wrap_angle_rad:.12g} rad | "
                        f"高张力毛刺={optimized.high_spike_count} | 低张力毛刺={optimized.low_spike_count}"
                    )
                elif kind == "opt_err":
                    self.opt_status_var.set("优化失败")
                    messagebox.showerror("优化失败", str(payload))
                if kind == "wrap_theta_ok":
                    data, theta = payload
                    self._wrap_source_data = data
                    self.wrap_angle_var.set(f"{float(theta):.16g}")
                    self.wrap_status_var.set(f"当前包角反推完成: θ={float(theta):.12g} rad | rows={data.row_count}")
                elif kind == "wrap_preview_ok":
                    data, mu_new, theta = payload
                    self._wrap_source_data = data
                    self._wrap_mu = mu_new
                    self._wrap_angle = float(theta)
                    self._wrap_signature = (os.path.abspath(data.db_path), data.table_name)
                    self._draw_wrap_preview(data, mu_new, float(theta))
                    null_count = int(np.count_nonzero(~np.isfinite(mu_new)))
                    self.wrap_status_var.set(
                        f"预览完成: rows={data.row_count} | 新包角 θ={float(theta):.12g} rad | NULL={null_count}"
                    )
                elif kind == "wrap_err":
                    self.wrap_status_var.set("包角重算失败")
                    messagebox.showerror("包角重算失败", str(payload))
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _update_summary(self, data: MonitorData, result: AnalysisResult) -> None:
        lines = summary_lines(summary_dict(data, result))
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.configure(state=tk.DISABLED)

    def _update_plots(self, data: MonitorData, result: AnalysisResult) -> None:
        labels = plot_labels(self._preview_lang_code(self.plot_preview_lang_var))
        max_points = (self._last_params or self._get_params().normalized()).max_plot_points
        options = self._get_plot_options()
        plot_tension_axis(self.ax_tension, data, result, max_points, labels, options)
        plot_mu_axis(self.ax_mu, data, result, max_points, labels, options)
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.18, hspace=0.92)
        self.canvas.draw_idle()
        self.root.after_idle(self._on_body_configure)

    def _refresh_current_plot(self) -> None:
        if self._last_data is None or self._last_result is None:
            self._draw_placeholder()
            return
        try:
            self._update_plots(self._last_data, self._last_result)
        except Exception as exc:
            messagebox.showerror("绘图参数错误", str(exc))

    def _export_plots(self) -> None:
        if self._last_data is None or self._last_result is None or self._last_params is None:
            messagebox.showwarning("提示", "请先完成分析，再导出图片。")
            return
        out_dir = filedialog.askdirectory(title="选择图片导出目录")
        if not out_dir:
            return
        lang = "en" if self.export_lang_var.get().strip().lower().startswith("english") else "zh"
        export_items = [key for key, var in self._export_bool_vars.items() if bool(var.get())]
        if not export_items:
            messagebox.showwarning("提示", "请至少选择一项导出内容。")
            return
        try:
            outputs = export_monitor_plots(
                self._last_data,
                self._last_result,
                self._last_params,
                out_dir,
                lang=lang,
                options=self._get_plot_options(),
                export_items=export_items,
            )
        except Exception:
            messagebox.showerror("导出失败", traceback.format_exc())
            return
        self.status_var.set(f"图片已导出: {os.path.basename(out_dir)} | lang={lang}")
        messagebox.showinfo("导出完成", "\n".join(outputs.values()))

    def _export_optimized_db(self) -> None:
        if self._opt_source_data is None or self._opt_data is None:
            self._run_optimization_preview_async()
            messagebox.showwarning("提示", "请先生成优化预览，确认后再导出。")
            return
        previous_output_path = self.opt_output_path_var.get().strip()
        self.opt_output_path_var.set("")
        self._browse_opt_output()
        output_path = self.opt_output_path_var.get().strip()
        if not output_path:
            self.opt_output_path_var.set(previous_output_path)
            return
        try:
            params = self._get_optimization_params()
            signature = self._current_optimization_signature()
            if self._opt_params != params or self._opt_signature != signature:
                messagebox.showwarning("提示", "优化参数或数据源已变化，请重新生成优化预览后再导出。")
                return
            result = export_optimized_db(self._opt_source_data.db_path, output_path, self._opt_source_data.table_name, self._opt_data, params)
        except Exception:
            messagebox.showerror("导出失败", traceback.format_exc())
            return
        self.opt_status_var.set(
            f"导出完成: {result.output_path}\nrows={result.row_count} | θ={result.wrap_angle_rad:.12g} rad | "
            f"高张力毛刺={result.high_spike_count} | 低张力毛刺={result.low_spike_count}"
        )
        messagebox.showinfo("导出完成", result.output_path)

    def _export_wrap_angle_db(self) -> None:
        if self._wrap_source_data is None or self._wrap_mu is None or self._wrap_angle is None:
            self._run_wrap_preview_async()
            messagebox.showwarning("提示", "请先生成包角重算预览，确认后再导出。")
            return

        previous_output_path = self.wrap_output_path_var.get().strip()
        self.wrap_output_path_var.set("")
        self._browse_wrap_output()
        output_path = self.wrap_output_path_var.get().strip()
        if not output_path:
            self.wrap_output_path_var.set(previous_output_path)
            return

        try:
            theta = float(self.wrap_angle_var.get().strip())
            signature = self._current_wrap_signature()
            if self._wrap_angle != theta or self._wrap_signature != signature:
                messagebox.showwarning("提示", "包角参数或数据源已变化，请重新生成预览后再导出。")
                return
            result = export_wrap_angle_db(
                self._wrap_source_data.db_path,
                output_path,
                self._wrap_source_data.table_name,
                self._wrap_mu,
                theta,
            )
        except Exception:
            messagebox.showerror("导出失败", traceback.format_exc())
            return

        self.wrap_status_var.set(
            f"导出完成: {result.output_path}\nrows={result.row_count} | θ={result.wrap_angle_rad:.12g} rad | NULL={result.nan_count}"
        )
        messagebox.showinfo("导出完成", result.output_path)

    def _clear_result(self) -> None:
        self._last_data = None
        self._last_result = None
        self._last_params = None
        self.status_var.set("结果已清空。")
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.configure(state=tk.DISABLED)
        self._draw_placeholder()

    def _draw_placeholder(self) -> None:
        labels = plot_labels(self._preview_lang_code(self.plot_preview_lang_var))
        self.ax_tension.clear()
        self.ax_mu.clear()
        self.ax_tension.set_title(labels["title_closed_t"])
        self.ax_mu.set_title(labels["title_mu"])
        self.ax_tension.set_ylabel(labels["tension"])
        self.ax_mu.set_ylabel(labels["mu"])
        self.ax_mu.set_xlabel(labels["t"])
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.18, hspace=0.92)
        self.canvas.draw_idle()

    def _draw_optimization_placeholder(self) -> None:
        self.opt_ax_preview.clear()
        self.opt_ax_preview.set_title("优化前/后高张力侧对比")
        self.opt_ax_preview.set_ylabel("T_high (N)")
        self.opt_ax_preview.set_xlabel("时间 t (s)")
        self.opt_figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)
        self.opt_preview_canvas.draw_idle()

    def _refresh_optimization_preview_plot(self) -> None:
        if self._opt_source_data is None or self._opt_data is None:
            self._draw_optimization_placeholder()
            return
        self._draw_optimization_preview(self._opt_source_data, self._opt_data)

    def _draw_optimization_preview(self, data: MonitorData, optimized: OptimizedData) -> None:
        mode = self.opt_preview_mode_var.get().strip() or "高张力"
        max_points = 40000
        q_from_db = np.asarray(data.quality_flag, dtype=int) != 0
        display_high = interpolate_invalid_samples(data.t_high_N, q_from_db)
        display_low = interpolate_invalid_samples(data.t_low_N, q_from_db)
        display_avg = (display_high + display_low) / 2.0
        display_mu = interpolate_invalid_samples(data.mu, q_from_db)

        tx, raw_high = downsample_for_plot(data.t_s, display_high, max_points)
        _, opt_high = downsample_for_plot(data.t_s, optimized.t_high_N, max_points)
        _, raw_low = downsample_for_plot(data.t_s, display_low, max_points)
        _, opt_low = downsample_for_plot(data.t_s, optimized.t_low_N, max_points)
        _, raw_avg = downsample_for_plot(data.t_s, display_avg, max_points)
        _, opt_avg = downsample_for_plot(data.t_s, optimized.t_avg_N, max_points)
        _, raw_mu = downsample_for_plot(data.t_s, display_mu, max_points)
        _, opt_mu = downsample_for_plot(data.t_s, optimized.mu, max_points)

        series = {
            "高张力": ("优化前/后高张力侧对比", "T_high (N)", raw_high, opt_high),
            "低张力": ("优化前/后低张力侧对比", "T_low (N)", raw_low, opt_low),
            "平均张力": ("优化前/后平均张力对比", "T_avg (N)", raw_avg, opt_avg),
            "μ": ("优化前/后 μ 对比", "μ", raw_mu, opt_mu),
        }
        title, ylabel, raw_y, opt_y = series.get(mode, series["高张力"])

        self.opt_ax_preview.clear()
        self.opt_ax_preview.plot(tx, raw_y, label="优化前", color="tab:blue", alpha=0.55)
        self.opt_ax_preview.plot(tx, opt_y, label="优化后", color="tab:red", linewidth=1.0)
        self.opt_ax_preview.set_title(title)
        self.opt_ax_preview.set_ylabel(ylabel)
        self.opt_ax_preview.set_xlabel("时间 t (s)")
        self.opt_ax_preview.legend(loc="upper right")
        self.opt_figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)
        self.opt_preview_canvas.draw_idle()

    def _draw_wrap_placeholder(self) -> None:
        self.wrap_ax_mu.clear()
        self.wrap_ax_mu.set_title("包角重算：μ vs Time")
        self.wrap_ax_mu.set_xlabel("Time t (s)")
        self.wrap_ax_mu.set_ylabel("Friction coefficient μ")
        self.wrap_figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)
        self.wrap_canvas.draw_idle()

    def _draw_wrap_preview(self, data: MonitorData, mu_new: np.ndarray, theta: float) -> None:
        max_points = 40000
        tx, raw_mu = downsample_for_plot(data.t_s, data.mu, max_points)
        _, new_mu = downsample_for_plot(data.t_s, mu_new, max_points)

        self.wrap_ax_mu.clear()
        self.wrap_ax_mu.plot(tx, raw_mu, label="原 μ", color="tab:blue", alpha=0.55)
        self.wrap_ax_mu.plot(tx, new_mu, label=f"新 μ (θ={theta:.6g} rad)", color="tab:red", linewidth=1.0)
        self.wrap_ax_mu.set_title("包角重算：μ vs Time")
        self.wrap_ax_mu.set_xlabel("Time t (s)")
        self.wrap_ax_mu.set_ylabel("Friction coefficient μ")
        self.wrap_ax_mu.legend(loc="upper right")
        self.wrap_figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)
        self.wrap_canvas.draw_idle()

    def _optimization_preview_text(self, mode: str) -> Dict[str, str]:
        is_en = self._preview_lang_code(self.opt_preview_lang_var) == "en"
        if "low" in mode.lower() or "低" in mode:
            key = "low"
        elif "avg" in mode.lower() or "平均" in mode:
            key = "avg"
        elif "μ" in mode or "mu" in mode.lower():
            key = "mu"
        else:
            key = "high"
        labels = {
            "high": ("Before/After Optimization: T_high", "优化前/后高张力侧对比", "T_high (N)"),
            "low": ("Before/After Optimization: T_low", "优化前/后低张力侧对比", "T_low (N)"),
            "avg": ("Before/After Optimization: T_avg", "优化前/后平均张力对比", "T_avg (N)"),
            "mu": ("Before/After Optimization: μ", "优化前/后 μ 对比", "μ"),
        }[key]
        return {
            "key": key,
            "title": labels[0] if is_en else labels[1],
            "ylabel": labels[2],
            "xlabel": "Time t (s)" if is_en else "时间 t (s)",
            "before": "Before" if is_en else "优化前",
            "after": "After" if is_en else "优化后",
        }

    def _draw_optimization_placeholder(self) -> None:
        text = self._optimization_preview_text(self.opt_preview_mode_var.get().strip())
        self.opt_ax_preview.clear()
        self.opt_ax_preview.set_title(text["title"])
        self.opt_ax_preview.set_ylabel(text["ylabel"])
        self.opt_ax_preview.set_xlabel(text["xlabel"])
        self.opt_figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)
        self.opt_preview_canvas.draw_idle()

    def _refresh_optimization_preview_plot(self) -> None:
        if self._opt_source_data is None or self._opt_data is None:
            self._draw_optimization_placeholder()
            return
        self._draw_optimization_preview(self._opt_source_data, self._opt_data)

    def _draw_optimization_preview(self, data: MonitorData, optimized: OptimizedData) -> None:
        text = self._optimization_preview_text(self.opt_preview_mode_var.get().strip())
        max_points = 40000
        q_from_db = np.asarray(data.quality_flag, dtype=int) != 0
        display_high = interpolate_invalid_samples(data.t_high_N, q_from_db)
        display_low = interpolate_invalid_samples(data.t_low_N, q_from_db)
        display_avg = (display_high + display_low) / 2.0
        display_mu = interpolate_invalid_samples(data.mu, q_from_db)

        tx, raw_high = downsample_for_plot(data.t_s, display_high, max_points)
        _, opt_high = downsample_for_plot(data.t_s, optimized.t_high_N, max_points)
        _, raw_low = downsample_for_plot(data.t_s, display_low, max_points)
        _, opt_low = downsample_for_plot(data.t_s, optimized.t_low_N, max_points)
        _, raw_avg = downsample_for_plot(data.t_s, display_avg, max_points)
        _, opt_avg = downsample_for_plot(data.t_s, optimized.t_avg_N, max_points)
        _, raw_mu = downsample_for_plot(data.t_s, display_mu, max_points)
        _, opt_mu = downsample_for_plot(data.t_s, optimized.mu, max_points)

        series = {
            "high": (raw_high, opt_high),
            "low": (raw_low, opt_low),
            "avg": (raw_avg, opt_avg),
            "mu": (raw_mu, opt_mu),
        }
        raw_y, opt_y = series[text["key"]]
        self.opt_ax_preview.clear()
        self.opt_ax_preview.plot(tx, raw_y, label=text["before"], color="tab:blue", alpha=0.55)
        self.opt_ax_preview.plot(tx, opt_y, label=text["after"], color="tab:red", linewidth=1.0)
        self.opt_ax_preview.set_title(text["title"])
        self.opt_ax_preview.set_ylabel(text["ylabel"])
        self.opt_ax_preview.set_xlabel(text["xlabel"])
        self.opt_ax_preview.legend(loc="upper right")
        self.opt_figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)
        self.opt_preview_canvas.draw_idle()

    def _wrap_preview_text(self) -> Dict[str, str]:
        is_en = self._preview_lang_code(self.wrap_preview_lang_var) == "en"
        return {
            "title": "Wrap Angle Recalculation: μ vs Time" if is_en else "包角重算：μ-时间",
            "xlabel": "Time t (s)" if is_en else "时间 t (s)",
            "ylabel": "Friction coefficient μ" if is_en else "摩擦系数 μ",
            "old": "Original μ" if is_en else "原 μ",
            "new": "New μ" if is_en else "新 μ",
        }

    def _draw_wrap_placeholder(self) -> None:
        text = self._wrap_preview_text()
        self.wrap_ax_mu.clear()
        self.wrap_ax_mu.set_title(text["title"])
        self.wrap_ax_mu.set_xlabel(text["xlabel"])
        self.wrap_ax_mu.set_ylabel(text["ylabel"])
        self.wrap_figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)
        self.wrap_canvas.draw_idle()

    def _refresh_wrap_preview_plot(self) -> None:
        if self._wrap_source_data is None or self._wrap_mu is None or self._wrap_angle is None:
            self._draw_wrap_placeholder()
            return
        self._draw_wrap_preview(self._wrap_source_data, self._wrap_mu, self._wrap_angle)

    def _draw_wrap_preview(self, data: MonitorData, mu_new: np.ndarray, theta: float) -> None:
        text = self._wrap_preview_text()
        max_points = 40000
        tx, raw_mu = downsample_for_plot(data.t_s, data.mu, max_points)
        _, new_mu = downsample_for_plot(data.t_s, mu_new, max_points)

        self.wrap_ax_mu.clear()
        self.wrap_ax_mu.plot(tx, raw_mu, label=text["old"], color="tab:blue", alpha=0.55)
        self.wrap_ax_mu.plot(tx, new_mu, label=f'{text["new"]} (θ={theta:.6g} rad)', color="tab:red", linewidth=1.0)
        self.wrap_ax_mu.set_title(text["title"])
        self.wrap_ax_mu.set_xlabel(text["xlabel"])
        self.wrap_ax_mu.set_ylabel(text["ylabel"])
        self.wrap_ax_mu.legend(loc="upper right")
        self.wrap_figure.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)
        self.wrap_canvas.draw_idle()

    def _plot_preview_kind(self) -> str:
        mode = self.plot_preview_mode_var.get().strip().lower()
        if "μ" in mode or "mu" in mode or "摩擦" in mode:
            return "mu"
        if "高" in mode or "high" in mode:
            return "high"
        if "低" in mode or "low" in mode:
            return "low"
        if "平均" in mode or "avg" in mode:
            return "avg"
        return "tension"

    def _apply_plot_preview_tension_limits(self, options: PlotOptions, kind: str) -> None:
        if kind == "high":
            lo = options.high_tension_y_min
            hi = options.high_tension_y_max
        elif kind == "low":
            lo = options.low_tension_y_min
            hi = options.low_tension_y_max
        else:
            lo = options.tension_y_min
            hi = options.tension_y_max
        if lo is None and hi is None:
            return
        current_lo, current_hi = self.ax_plot_preview.get_ylim()
        self.ax_plot_preview.set_ylim(current_lo if lo is None else lo, current_hi if hi is None else hi)

    def _plot_single_tension_preview(
        self,
        data: MonitorData,
        y: np.ndarray,
        label: str,
        title: str,
        color: str,
        max_points: int,
        labels: Dict[str, str],
        options: PlotOptions,
        kind: str,
    ) -> None:
        tx, yy = downsample_for_plot(data.t_s, y, max_points)
        self.ax_plot_preview.clear()
        self.ax_plot_preview.plot(tx, yy, label=label, color=color)
        self.ax_plot_preview.set_title(title)
        self.ax_plot_preview.set_ylabel(labels["tension"])
        self.ax_plot_preview.set_xlabel(labels["t"])
        self._apply_plot_preview_tension_limits(options, kind)
        self.ax_plot_preview.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1, frameon=False)

    def _update_plots(self, data: MonitorData, result: AnalysisResult) -> None:
        labels = plot_labels(self._preview_lang_code(self.plot_preview_lang_var))
        max_points = self._get_params().normalized().max_plot_points
        options = self._get_plot_options()
        kind = self._plot_preview_kind()
        self.ax_plot_preview.clear()
        if kind == "mu":
            plot_mu_axis(self.ax_plot_preview, data, result, max_points, labels, options)
            self.figure.tight_layout(rect=[0, 0.18, 1, 1])
        elif kind == "high":
            self._plot_single_tension_preview(data, result.display_t_high, labels["th"], labels["th"], "tab:blue", max_points, labels, options, kind)
            self.figure.tight_layout(rect=[0, 0.12, 1, 1])
        elif kind == "low":
            self._plot_single_tension_preview(data, result.display_t_low, labels["tl"], labels["tl"], "tab:orange", max_points, labels, options, kind)
            self.figure.tight_layout(rect=[0, 0.12, 1, 1])
        elif kind == "avg":
            self._plot_single_tension_preview(data, result.display_t_avg, labels["ta"], labels["ta"], "tab:green", max_points, labels, options, kind)
            self.figure.tight_layout(rect=[0, 0.12, 1, 1])
        else:
            plot_tension_axis(self.ax_plot_preview, data, result, max_points, labels, options)
            self.ax_plot_preview.set_xlabel(labels["t"])
            self.figure.tight_layout(rect=[0, 0.12, 1, 1])
        self.canvas.draw_idle()
        self.root.after_idle(self._on_body_configure)

    def _draw_placeholder(self) -> None:
        labels = plot_labels(self._preview_lang_code(self.plot_preview_lang_var))
        kind = self._plot_preview_kind()
        self.ax_plot_preview.clear()
        if kind == "mu":
            self.ax_plot_preview.set_title(labels["title_mu"])
            self.ax_plot_preview.set_ylabel(labels["mu"])
        elif kind == "high":
            self.ax_plot_preview.set_title(labels["th"])
            self.ax_plot_preview.set_ylabel(labels["tension"])
        elif kind == "low":
            self.ax_plot_preview.set_title(labels["tl"])
            self.ax_plot_preview.set_ylabel(labels["tension"])
        elif kind == "avg":
            self.ax_plot_preview.set_title(labels["ta"])
            self.ax_plot_preview.set_ylabel(labels["tension"])
        else:
            self.ax_plot_preview.set_title(labels["title_closed_t"])
            self.ax_plot_preview.set_ylabel(labels["tension"])
        self.ax_plot_preview.set_xlabel(labels["t"])
        self.figure.tight_layout()
        self.canvas.draw_idle()


def run_headless(db_path: str, table_name: str, params: AnalysisParams) -> Dict[str, object]:
    data = load_monitor_db(db_path, table_name=table_name)
    result = analyze_monitor_data(data, params)
    return summary_dict(data, result)
