import os
from typing import Dict, Iterable, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib.figure import Figure

from analysis import clip_stable_segments, interpolate_invalid_samples
from models import AnalysisParams, AnalysisResult, AxisRange, MonitorData, OptimizedData, PlotOptions


matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def downsample_for_plot(x: np.ndarray, y: np.ndarray, max_points: int = 200_000) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= max_points:
        return np.asarray(x), np.asarray(y)
    step = max(1, n // max_points)
    return np.asarray(x)[::step], np.asarray(y)[::step]


def plot_labels(lang: str = "zh") -> Dict[str, str]:
    if str(lang).strip().lower().startswith("en"):
        return {
            "t": "Time t (s)",
            "tension": "Tension (N)",
            "title_closed_t": "Closed-loop: Tension vs Time",
            "title_mu": "Closed-loop: μ vs Time (μss & μth)",
            "th": "T_high",
            "tl": "T_low",
            "ta": "T_avg",
            "mu": "Friction coefficient μ",
            "mu_f": "μ (filtered)",
            "mu_ss": "Baseline μss",
            "mu_th": "Threshold μth",
            "ss": "Stable segments",
            "tlife": "tlife",
        }
    return {
        "t": "时间 t (s)",
        "tension": "张力 (N)",
        "title_closed_t": "闭环：张力-时间",
        "title_mu": "闭环：摩擦系数 μ-时间（μss 与 μth）",
        "th": "高张力侧 T_high",
        "tl": "低张力侧 T_low",
        "ta": "平均张力 T_avg",
        "mu": "摩擦系数 μ",
        "mu_f": "μ（滤波）",
        "mu_ss": "稳定段基线 μss",
        "mu_th": "超限阈值 μth",
        "ss": "连续稳定段",
        "tlife": "tlife",
    }


def _axis_limit_pair(v_min: Optional[float], v_max: Optional[float], axis_name: str = "Y轴") -> Tuple[Optional[float], Optional[float]]:
    lo = None if v_min is None else float(v_min)
    hi = None if v_max is None else float(v_max)
    if lo is not None and hi is not None and hi <= lo:
        raise ValueError(f"{axis_name}上限必须大于下限，或使用 Auto 表示自适应。")
    return lo, hi


def _apply_y_limits(ax, y_min: Optional[float], y_max: Optional[float]) -> None:
    lo, hi = _axis_limit_pair(y_min, y_max, "Y轴")
    current_lo, current_hi = ax.get_ylim()
    ax.set_ylim(current_lo if lo is None else lo, current_hi if hi is None else hi)


def _apply_axis_range(ax, axis_range: Optional[AxisRange]) -> None:
    if axis_range is None:
        return
    x_lo, x_hi = _axis_limit_pair(axis_range.x_min, axis_range.x_max, "X轴")
    y_lo, y_hi = _axis_limit_pair(axis_range.y_min, axis_range.y_max, "Y轴")
    current_x_lo, current_x_hi = ax.get_xlim()
    current_y_lo, current_y_hi = ax.get_ylim()
    ax.set_xlim(current_x_lo if x_lo is None else x_lo, current_x_hi if x_hi is None else x_hi)
    ax.set_ylim(current_y_lo if y_lo is None else y_lo, current_y_hi if y_hi is None else y_hi)


def _range_with_y_fallback(axis_range: Optional[AxisRange], y_min: Optional[float], y_max: Optional[float]) -> AxisRange:
    if axis_range is None:
        return AxisRange(y_min=y_min, y_max=y_max)
    return axis_range


def validate_plot_options(options: PlotOptions) -> None:
    _axis_limit_pair(options.tension_y_min, options.tension_y_max, "张力Y轴")
    _axis_limit_pair(options.high_tension_y_min, options.high_tension_y_max, "高张力Y轴")
    _axis_limit_pair(options.low_tension_y_min, options.low_tension_y_max, "低张力Y轴")
    _axis_limit_pair(options.mu_y_min, options.mu_y_max, "μ Y轴")
    for key, axis_range in options.axis_ranges.items():
        _axis_limit_pair(axis_range.x_min, axis_range.x_max, f"{key} X轴")
        _axis_limit_pair(axis_range.y_min, axis_range.y_max, f"{key} Y轴")


def plot_tension_axis(
    ax,
    data: MonitorData,
    result: AnalysisResult,
    max_points: int,
    labels: Dict[str, str],
    options: Optional[PlotOptions] = None,
) -> None:
    options = options or PlotOptions()
    tx, hi = downsample_for_plot(data.t_s, result.display_t_high, max_points)
    _, lo = downsample_for_plot(data.t_s, result.display_t_low, max_points)
    _, avg = downsample_for_plot(data.t_s, result.display_t_avg, max_points)

    ax.clear()
    ax.plot(tx, hi, label=labels["th"], color="tab:blue")
    ax.plot(tx, lo, label=labels["tl"], color="tab:orange")
    ax.plot(tx, avg, label=labels["ta"], color="tab:green")
    ax.set_title(labels["title_closed_t"])
    ax.set_ylabel(labels["tension"])
    _apply_axis_range(ax, _range_with_y_fallback(options.axis_ranges.get("tension"), options.tension_y_min, options.tension_y_max))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False)


def plot_single_tension_axis(
    ax,
    data: MonitorData,
    y: np.ndarray,
    title: str,
    legend_label: str,
    color: str,
    max_points: int,
    labels: Dict[str, str],
    options: Optional[PlotOptions] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    axis_range: Optional[AxisRange] = None,
) -> None:
    options = options or PlotOptions()
    tx, yy = downsample_for_plot(data.t_s, y, max_points)

    ax.clear()
    ax.plot(tx, yy, label=legend_label, color=color)
    ax.set_title(title)
    ax.set_xlabel(labels["t"])
    ax.set_ylabel(labels["tension"])
    _apply_axis_range(ax, _range_with_y_fallback(axis_range, y_min, y_max))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1, frameon=False)


def plot_mu_axis(
    ax,
    data: MonitorData,
    result: AnalysisResult,
    max_points: int,
    labels: Dict[str, str],
    options: Optional[PlotOptions] = None,
) -> None:
    options = options or PlotOptions()
    tx, mu_eval = downsample_for_plot(data.t_s, result.display_mu_eval, max_points)
    stable_segs_clip = clip_stable_segments(data.t_s, result.stable_segments_idx, result.tlife_s, result.fs_hz)

    ax.clear()
    if options.show_mu:
        ax.plot(tx, mu_eval, label=labels["mu_f"], color="tab:blue", zorder=3)

    if options.show_mu_ss and result.mu_ss is not None and np.isfinite(result.mu_ss):
        ax.axhline(float(result.mu_ss), linestyle="--", label=f'{labels["mu_ss"]}={float(result.mu_ss):.4f}', color="tab:orange")
    if options.show_mu_th and result.mu_th is not None and np.isfinite(result.mu_th):
        ax.axhline(float(result.mu_th), linestyle="--", label=f'{labels["mu_th"]}={float(result.mu_th):.4f}', color="tab:red")

    if options.show_stable_segments and stable_segs_clip:
        first = True
        for k0, k1 in stable_segs_clip:
            t0 = float(data.t_s[k0])
            t1 = float(data.t_s[k1]) if (k1 < len(data.t_s)) else float(data.t_s[-1])
            if first:
                ax.axvspan(t0, t1, alpha=0.25, label=labels["ss"], color="tab:cyan", zorder=0, linewidth=0)
                first = False
            else:
                ax.axvspan(t0, t1, alpha=0.25, color="tab:cyan", zorder=0, linewidth=0)

    if options.show_tlife and result.tlife_s is not None:
        ax.axvline(float(result.tlife_s), linestyle="--", label=f'{labels["tlife"]}≈{float(result.tlife_s):.1f}s', color="tab:purple")

    ax.set_title(labels["title_mu"])
    ax.set_xlabel(labels["t"])
    ax.set_ylabel(labels["mu"])
    _apply_axis_range(ax, _range_with_y_fallback(options.axis_ranges.get("mu"), options.mu_y_min, options.mu_y_max))
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False)


def optimization_plot_labels(lang: str, key: str) -> Dict[str, str]:
    is_en = str(lang).strip().lower().startswith("en")
    labels = {
        "high": ("Before/After Optimization: T_high", "优化前/后高张力侧对比", "T_high (N)"),
        "low": ("Before/After Optimization: T_low", "优化前/后低张力侧对比", "T_low (N)"),
        "avg": ("Before/After Optimization: T_avg", "优化前/后平均张力对比", "T_avg (N)"),
        "mu": ("Before/After Optimization: μ", "优化前/后 μ 对比", "μ"),
    }.get(key, ("Before/After Optimization: T_high", "优化前/后高张力侧对比", "T_high (N)"))
    return {
        "title": labels[0] if is_en else labels[1],
        "ylabel": labels[2],
        "xlabel": "Time t (s)" if is_en else "时间 t (s)",
        "before": "Before" if is_en else "优化前",
        "after": "After" if is_en else "优化后",
    }


def optimization_comparison_series(data: MonitorData, optimized: OptimizedData, max_points: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
    return {
        "high": (tx, raw_high, opt_high),
        "low": (tx, raw_low, opt_low),
        "avg": (tx, raw_avg, opt_avg),
        "mu": (tx, raw_mu, opt_mu),
    }


def plot_optimization_comparison_axis(
    ax,
    data: MonitorData,
    optimized: OptimizedData,
    key: str,
    max_points: int,
    lang: str = "zh",
    axis_range: Optional[AxisRange] = None,
) -> None:
    key = key if key in {"high", "low", "avg", "mu"} else "high"
    text = optimization_plot_labels(lang, key)
    tx, raw_y, opt_y = optimization_comparison_series(data, optimized, max_points)[key]
    ax.clear()
    ax.plot(tx, raw_y, label=text["before"], color="tab:blue", alpha=0.55)
    ax.plot(tx, opt_y, label=text["after"], color="tab:red", linewidth=1.0)
    ax.set_title(text["title"])
    ax.set_ylabel(text["ylabel"])
    ax.set_xlabel(text["xlabel"])
    _apply_axis_range(ax, axis_range)
    ax.legend(loc="upper right")


def export_optimization_plots(
    data: MonitorData,
    optimized: OptimizedData,
    out_dir: str,
    lang: str = "zh",
    export_items: Optional[Iterable[str]] = None,
    axis_ranges: Optional[Dict[str, AxisRange]] = None,
    max_points: int = 40000,
) -> Dict[str, str]:
    selected = list(export_items or ("high", "low", "avg", "mu"))
    axis_ranges = axis_ranges or {}
    plot_dir = os.path.join(os.path.abspath(out_dir), "optimization_plots")
    os.makedirs(plot_dir, exist_ok=True)
    default_figsize = tuple(float(v) for v in matplotlib.rcParams["figure.figsize"])
    default_dpi = float(matplotlib.rcParams["figure.dpi"])
    outputs: Dict[str, str] = {}
    for key in selected:
        if key not in {"high", "low", "avg", "mu"}:
            continue
        fig = Figure(figsize=default_figsize, dpi=default_dpi)
        ax = fig.add_subplot(111)
        plot_optimization_comparison_axis(ax, data, optimized, key, max_points, lang, axis_ranges.get(key))
        fig.tight_layout(rect=[0, 0.02, 1, 1])
        path = os.path.join(plot_dir, f"optimization_{key}_{lang}.png")
        fig.savefig(path, dpi=180)
        fig.clear()
        outputs[key] = path
    return outputs


def export_monitor_plots(
    data: MonitorData,
    result: AnalysisResult,
    params: AnalysisParams,
    out_dir: str,
    lang: str = "zh",
    options: Optional[PlotOptions] = None,
    export_items: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    params = params.normalized()
    labels = plot_labels(lang)
    options = options or PlotOptions()
    selected = set(("tension", "mu") if export_items is None else export_items)
    plot_dir = os.path.join(os.path.abspath(out_dir), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    default_figsize = tuple(float(v) for v in matplotlib.rcParams["figure.figsize"])
    default_dpi = float(matplotlib.rcParams["figure.dpi"])
    outputs: Dict[str, str] = {}

    if "tension" in selected:
        fig_t = Figure(figsize=default_figsize, dpi=default_dpi)
        ax_t = fig_t.add_subplot(111)
        plot_tension_axis(ax_t, data, result, params.max_plot_points, labels, options)
        ax_t.set_xlabel(labels["t"])
        fig_t.tight_layout(rect=[0, 0.12, 1, 1])
        path = os.path.join(plot_dir, f"closed_tensions_{lang}.png")
        fig_t.savefig(path, dpi=180)
        fig_t.clear()
        outputs["closed_tension"] = path

    single_tension_specs = {
        "high": (
            result.display_t_high,
            labels["th"],
            labels["th"],
            "tab:blue",
            f"tension_high_{lang}.png",
            options.high_tension_y_min,
            options.high_tension_y_max,
            options.axis_ranges.get("high"),
        ),
        "low": (
            result.display_t_low,
            labels["tl"],
            labels["tl"],
            "tab:orange",
            f"tension_low_{lang}.png",
            options.low_tension_y_min,
            options.low_tension_y_max,
            options.axis_ranges.get("low"),
        ),
        "avg": (
            result.display_t_avg,
            labels["ta"],
            labels["ta"],
            "tab:green",
            f"tension_avg_{lang}.png",
            None,
            None,
            options.axis_ranges.get("avg"),
        ),
    }
    for key, (y, title, legend_label, color, filename, y_min, y_max, axis_range) in single_tension_specs.items():
        if key not in selected:
            continue
        fig_single = Figure(figsize=default_figsize, dpi=default_dpi)
        ax_single = fig_single.add_subplot(111)
        plot_single_tension_axis(
            ax_single,
            data,
            y,
            title,
            legend_label,
            color,
            params.max_plot_points,
            labels,
            options,
            y_min=y_min,
            y_max=y_max,
            axis_range=axis_range,
        )
        fig_single.tight_layout(rect=[0, 0.12, 1, 1])
        path = os.path.join(plot_dir, filename)
        fig_single.savefig(path, dpi=180)
        fig_single.clear()
        outputs[key] = path

    if "mu" in selected:
        fig_mu = Figure(figsize=default_figsize, dpi=default_dpi)
        ax_mu = fig_mu.add_subplot(111)
        plot_mu_axis(ax_mu, data, result, params.max_plot_points, labels, options)
        fig_mu.tight_layout(rect=[0, 0.18, 1, 1])
        path = os.path.join(plot_dir, f"mu_with_baseline_threshold_{lang}.png")
        fig_mu.savefig(path, dpi=180)
        fig_mu.clear()
        outputs["mu_time"] = path

    return outputs
