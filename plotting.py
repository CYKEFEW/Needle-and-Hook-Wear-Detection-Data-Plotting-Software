import os
from typing import Dict, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib.figure import Figure

from analysis import clip_stable_segments
from models import AnalysisParams, AnalysisResult, MonitorData, PlotOptions


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
            "mu_f": "μ",
            "mu_ss": "Baseline μss",
            "mu_th": "Threshold μth",
            "ss": "Stable segments",
            "tlife": "tlife",
        }
    return {
        "t": "时间 t (s)",
        "tension": "张力 (N)",
        "title_closed_t": "闭环：张力-时间",
        "title_mu": "闭环：摩擦系数-时间（μss 与 μth）",
        "th": "高张力侧 T_high",
        "tl": "低张力侧 T_low",
        "ta": "平均张力 T_avg",
        "mu": "摩擦系数 μ",
        "mu_f": "μ",
        "mu_ss": "稳定段基线 μss",
        "mu_th": "超限阈值 μth",
        "ss": "连续稳定段",
        "tlife": "tlife",
    }


def _axis_limit_pair(y_min: float, y_max: float) -> Tuple[Optional[float], Optional[float]]:
    lo = None if float(y_min) == -1.0 else float(y_min)
    hi = None if float(y_max) == -1.0 else float(y_max)
    if lo is not None and hi is not None and hi <= lo:
        raise ValueError("Y轴上限必须大于下限，或使用 -1 表示自适应。")
    return lo, hi


def _apply_y_limits(ax, y_min: float, y_max: float) -> None:
    lo, hi = _axis_limit_pair(y_min, y_max)
    current_lo, current_hi = ax.get_ylim()
    ax.set_ylim(current_lo if lo is None else lo, current_hi if hi is None else hi)


def validate_plot_options(options: PlotOptions) -> None:
    _axis_limit_pair(options.tension_y_min, options.tension_y_max)
    _axis_limit_pair(options.mu_y_min, options.mu_y_max)


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
    _apply_y_limits(ax, options.tension_y_min, options.tension_y_max)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)


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

    if options.show_mu:
        ax.plot(tx, mu_eval, label=labels["mu_f"], color="tab:blue", zorder=3)

    if options.show_mu_ss and result.mu_ss is not None and np.isfinite(result.mu_ss):
        ax.axhline(float(result.mu_ss), linestyle="--", label=f'{labels["mu_ss"]}={float(result.mu_ss):.4f}', color="tab:orange")
    if options.show_mu_th and result.mu_th is not None and np.isfinite(result.mu_th):
        ax.axhline(float(result.mu_th), linestyle="--", label=f'{labels["mu_th"]}={float(result.mu_th):.4f}', color="tab:red")
    if options.show_tlife and result.tlife_s is not None:
        ax.axvline(float(result.tlife_s), linestyle="--", label=f'{labels["tlife"]}≈{float(result.tlife_s):.1f}s', color="tab:purple")

    ax.set_title(labels["title_mu"])
    ax.set_xlabel(labels["t"])
    ax.set_ylabel(labels["mu"])
    _apply_y_limits(ax, options.mu_y_min, options.mu_y_max)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)


def export_monitor_plots(
    data: MonitorData,
    result: AnalysisResult,
    params: AnalysisParams,
    out_dir: str,
    lang: str = "zh",
    options: Optional[PlotOptions] = None,
) -> Dict[str, str]:
    params = params.normalized()
    labels = plot_labels(lang)
    options = options or PlotOptions()
    plot_dir = os.path.join(os.path.abspath(out_dir), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    default_figsize = tuple(float(v) for v in matplotlib.rcParams["figure.figsize"])
    default_dpi = float(matplotlib.rcParams["figure.dpi"])

    fig_t = Figure(figsize=default_figsize, dpi=default_dpi)
    ax_t = fig_t.add_subplot(111)
    plot_tension_axis(ax_t, data, result, params.max_plot_points, labels, options)
    ax_t.set_xlabel(labels["t"])
    fig_t.tight_layout(rect=[0, 0.10, 1, 1])
    p_tension = os.path.join(plot_dir, f"closed_tensions_{lang}.png")
    fig_t.savefig(p_tension, dpi=180)

    fig_mu = Figure(figsize=default_figsize, dpi=default_dpi)
    ax_mu = fig_mu.add_subplot(111)
    plot_mu_axis(ax_mu, data, result, params.max_plot_points, labels, options)
    fig_mu.tight_layout(rect=[0, 0.14, 1, 1])
    p_mu = os.path.join(plot_dir, f"mu_with_baseline_threshold_{lang}.png")
    fig_mu.savefig(p_mu, dpi=180)
    fig_t.clear()
    fig_mu.clear()

    return {
        "closed_tension": p_tension,
        "mu_time": p_mu,
    }
