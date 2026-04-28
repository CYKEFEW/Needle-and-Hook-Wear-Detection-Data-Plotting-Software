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
from db_io import DEFAULT_DB_PATH, export_optimized_db, load_monitor_db
from models import AnalysisParams, AnalysisResult, MonitorData, OptimizationParams, OptimizedData, PlotOptions
from optimization import infer_wrap_angle_rad, optimize_monitor_data
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

        default_db = DEFAULT_DB_PATH if os.path.exists(DEFAULT_DB_PATH) else ""
        self.db_path_var = tk.StringVar(value=default_db)
        self.table_name_var = tk.StringVar(value="Data")
        self.status_var = tk.StringVar(value="选择数据库后点击“分析并绘图”。")
        self.export_lang_var = tk.StringVar(value="中文")
        self._param_vars: Dict[str, tk.Variable] = {}
        self._plot_limit_vars: Dict[str, tk.StringVar] = {}
        self._plot_bool_vars: Dict[str, tk.BooleanVar] = {}

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
        self.notebook.add(plot_tab, text="绘图区")
        self.notebook.add(opt_tab, text="数据优化区")

        self._build_plot_tab(plot_tab)
        self._build_optimization_tab(opt_tab)

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
        ttk.Entry(file_box, textvariable=self.db_path_var, width=48).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 6))
        ttk.Button(file_box, text="浏览...", command=self._browse_db).grid(row=1, column=2, padx=(8, 0), sticky="ew")

        ttk.Label(file_box, text="表名").grid(row=2, column=0, sticky="w")
        ttk.Entry(file_box, textvariable=self.table_name_var, width=18).grid(row=2, column=1, sticky="w", pady=(4, 0))
        file_box.columnconfigure(0, weight=1)
        file_box.columnconfigure(1, weight=1)

        calc_box = ttk.LabelFrame(parent, text="判据参数", padding=10)
        calc_box.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        row = 0
        row = self._add_entry(calc_box, row, "稳态窗口 Wss (s)", "stable_win_s", 1200.0)
        row = self._add_entry(calc_box, row, "稳态最短保持 Whold (s)", "stable_hold_s", 3600.0)
        row = self._add_entry(calc_box, row, "稳态标准差阈值 σmax", "stable_sigma_max", 0.03)
        row = self._add_entry(calc_box, row, "稳态总漂移阈值 Δμmax", "stable_slope_max", 0.015)
        row = self._add_entry(calc_box, row, "稳态有效比例 qmin", "stable_valid_min", 0.9)
        row = self._add_entry(calc_box, row, "失效阈值增量 δ", "fail_delta", 0.25)
        row = self._add_entry(calc_box, row, "持续超阈值 Wpersist (s)", "fail_hold_s", 300.0)
        row = self._add_entry(calc_box, row, "最大容忍中断 Wbreak (s)", "fail_break_s", 1.0)
        row = self._add_entry(calc_box, row, "采样率 fs (Hz, 0=自动推断)", "sample_rate_hz", 0.0)
        self._add_entry(calc_box, row, "绘图最大点数", "max_plot_points", 40000)
        calc_box.columnconfigure(1, weight=1)

        plot_box = ttk.LabelFrame(parent, text="绘图显示", padding=10)
        plot_box.pack(fill=tk.X, pady=(0, 10))
        self._add_plot_limit(plot_box, 0, "张力Y下限", "tension_y_min")
        self._add_plot_limit(plot_box, 1, "张力Y上限", "tension_y_max")
        self._add_plot_limit(plot_box, 2, "μY下限", "mu_y_min")
        self._add_plot_limit(plot_box, 3, "μY上限", "mu_y_max")
        for i, (key, text) in enumerate(
            [
                ("show_mu", "μ"),
                ("show_stable_segments", "连续稳定段"),
                ("show_mu_ss", "稳定段基线"),
                ("show_tlife", "tlife"),
                ("show_mu_th", "超限阈值"),
            ],
            start=4,
        ):
            var = tk.BooleanVar(value=True)
            self._plot_bool_vars[key] = var
            ttk.Checkbutton(plot_box, text=text, variable=var, command=self._refresh_current_plot).grid(
                row=i, column=0, columnspan=2, sticky="w", pady=2
            )
        plot_box.columnconfigure(1, weight=1)

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X, pady=(0, 8))
        self.analyze_btn = ttk.Button(btn_row, text="分析并绘图", command=self._run_analysis_async)
        self.analyze_btn.pack(side=tk.LEFT)
        ttk.Button(btn_row, text="清空结果", command=self._clear_result).pack(side=tk.LEFT, padx=(8, 0))

        export_row = ttk.Frame(parent)
        export_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(export_row, text="导出语言").pack(side=tk.LEFT)
        ttk.Combobox(export_row, textvariable=self.export_lang_var, values=("中文", "English"), state="readonly", width=10).pack(
            side=tk.LEFT, padx=(8, 12)
        )
        ttk.Button(export_row, text="导出图片...", command=self._export_plots).pack(side=tk.LEFT)

        result_box = ttk.LabelFrame(parent, text="结果摘要", padding=10)
        result_box.pack(fill=tk.BOTH, expand=True)
        self.summary_text = tk.Text(result_box, width=44, height=14, wrap=tk.WORD)
        summary_scroll = ttk.Scrollbar(result_box, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_text.configure(state=tk.DISABLED)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, anchor="w")

        self.figure = Figure(figsize=(11.2, 10.8), dpi=100)
        self.ax_tension = self.figure.add_subplot(211)
        self.ax_mu = self.figure.add_subplot(212, sharex=self.ax_tension)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(8, 0))
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

        self.opt_figure = Figure(figsize=(10.8, 7.2), dpi=100)
        self.opt_ax_preview = self.opt_figure.add_subplot(111)
        self.opt_preview_canvas = FigureCanvasTkAgg(self.opt_figure, master=preview)
        self.opt_preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.opt_toolbar = NavigationToolbar2Tk(self.opt_preview_canvas, preview, pack_toolbar=False)
        self.opt_toolbar.update()
        self.opt_toolbar.pack(fill=tk.X)
        self._draw_optimization_placeholder()

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

    def _browse_db(self) -> None:
        path = filedialog.askopenfilename(title="选择监测数据库", filetypes=[("SQLite DB", "*.db *.sqlite *.sqlite3"), ("所有文件", "*.*")])
        if path:
            self.db_path_var.set(path)
            self.opt_db_path_var.set(path)

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
        labels = plot_labels("zh")
        max_points = (self._last_params or self._get_params().normalized()).max_plot_points
        options = self._get_plot_options()
        plot_tension_axis(self.ax_tension, data, result, max_points, labels, options)
        plot_mu_axis(self.ax_mu, data, result, max_points, labels, options)
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.18, hspace=0.92)
        self.canvas.draw_idle()
        self.root.after_idle(self._on_body_configure)

    def _refresh_current_plot(self) -> None:
        if self._last_data is None or self._last_result is None:
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
        try:
            outputs = export_monitor_plots(self._last_data, self._last_result, self._last_params, out_dir, lang=lang, options=self._get_plot_options())
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
        labels = plot_labels("zh")
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


def run_headless(db_path: str, table_name: str, params: AnalysisParams) -> Dict[str, object]:
    data = load_monitor_db(db_path, table_name=table_name)
    result = analyze_monitor_data(data, params)
    return summary_dict(data, result)
