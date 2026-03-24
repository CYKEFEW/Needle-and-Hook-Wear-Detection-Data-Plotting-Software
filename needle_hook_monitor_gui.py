import argparse
import json
import math
import os
import queue
import sqlite3
import threading
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    pd = None
    HAVE_PANDAS = False

try:
    from scipy import signal
    HAVE_SCIPY = True
except Exception:
    signal = None
    HAVE_SCIPY = False

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "needle_hook_wear_sim.db")


@dataclass
class AnalysisParams:
    theta_deg: float = 100.0
    interpolate_invalid_display: bool = True
    tmin_gate_N: float = 0.08
    ratio_clip_min: float = 0.3
    ratio_clip_max: float = 8.0
    use_hampel: bool = True
    hampel_win_s: float = 1.0
    hampel_nsig: float = 3.0
    notch_f0_hz: float = 0.0
    notch_q: float = 0.0
    lowpass_fc_hz: float = 2.5
    stable_win_s: float = 1200.0
    stable_hold_s: float = 3600.0
    stable_sigma_max: float = 0.03
    stable_slope_max: float = 0.015
    stable_valid_min: float = 0.9
    fail_delta: float = 0.25
    fail_hold_s: float = 300.0
    max_plot_points: int = 40000

    def normalized(self) -> "AnalysisParams":
        slope = float(self.stable_slope_max)
        if slope < 1e-3:
            slope *= max(1.0, float(self.stable_win_s))
        return AnalysisParams(
            theta_deg=max(1e-6, float(self.theta_deg)),
            interpolate_invalid_display=bool(self.interpolate_invalid_display),
            tmin_gate_N=max(0.0, float(self.tmin_gate_N)),
            ratio_clip_min=max(1e-9, float(self.ratio_clip_min)),
            ratio_clip_max=max(max(1e-9, float(self.ratio_clip_min)), float(self.ratio_clip_max)),
            use_hampel=bool(self.use_hampel),
            hampel_win_s=max(0.0, float(self.hampel_win_s)),
            hampel_nsig=max(0.1, float(self.hampel_nsig)),
            notch_f0_hz=max(0.0, float(self.notch_f0_hz)),
            notch_q=max(0.0, float(self.notch_q)),
            lowpass_fc_hz=max(0.0, float(self.lowpass_fc_hz)),
            stable_win_s=max(1.0, float(self.stable_win_s)),
            stable_hold_s=max(0.0, float(self.stable_hold_s)),
            stable_sigma_max=max(0.0, float(self.stable_sigma_max)),
            stable_slope_max=max(0.0, slope),
            stable_valid_min=min(1.0, max(0.0, float(self.stable_valid_min))),
            fail_delta=max(0.0, float(self.fail_delta)),
            fail_hold_s=max(0.0, float(self.fail_hold_s)),
            max_plot_points=max(1000, int(self.max_plot_points)),
        )


@dataclass
class MonitorData:
    db_path: str
    table_name: str
    row_count: int
    t_s: np.ndarray
    t_high_N: np.ndarray
    t_low_N: np.ndarray
    t_avg_N: np.ndarray
    f_fric_N: np.ndarray
    mu: Optional[np.ndarray]
    quality_flag: np.ndarray


@dataclass
class AnalysisResult:
    fs_hz: float
    mu_raw: np.ndarray
    mu_hampel: np.ndarray
    mu_eval: np.ndarray
    q_valid: np.ndarray
    stable_segments_idx: List[Tuple[int, int]]
    stable_window_idx: Optional[Tuple[int, int]]
    mu_ss: Optional[float]
    mu_th: Optional[float]
    tlife_s: Optional[float]
    tlife_idx: Optional[int]
    valid_ratio: float
    display_t_high: np.ndarray
    display_t_low: np.ndarray
    display_t_avg: np.ndarray
    display_mu_raw: np.ndarray
    display_mu_eval: np.ndarray


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    kernel = np.ones(int(win), dtype=float) / float(win)
    pad = int(win) // 2
    xpad = np.pad(np.asarray(x, dtype=float), (pad, pad), mode="reflect")
    return np.convolve(xpad, kernel, mode="valid")


def _hampel_filter_nan(x: np.ndarray, win: int, n_sigmas: float = 3.0) -> np.ndarray:
    if win <= 1 or (not HAVE_PANDAS):
        return np.asarray(x, dtype=float).copy()
    s = pd.Series(np.asarray(x, dtype=float))
    med = s.rolling(win, center=True, min_periods=max(3, win // 2)).median()
    mad = (s - med).abs().rolling(win, center=True, min_periods=max(3, win // 2)).median()
    sigma = 1.4826 * mad
    outlier = (s - med).abs() > (n_sigmas * sigma)
    y = s.astype(float).copy()
    y[outlier] = np.nan
    return y.to_numpy()


def _notch_freq_domain(x: np.ndarray, fs: float, f0: float, q: float) -> np.ndarray:
    if f0 <= 0 or f0 >= fs / 2:
        return x.copy()
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    bw = max(0.05, f0 / max(1.0, q))
    mask = (freqs > (f0 - bw)) & (freqs < (f0 + bw))
    X[mask] = 0
    return np.fft.irfft(X, n=len(x))


def _apply_notch(x: np.ndarray, fs: float, f0: float, q: float) -> np.ndarray:
    if f0 <= 0.0 or q <= 0.0 or f0 >= fs / 2.0:
        return x.copy()
    if HAVE_SCIPY:
        b, a = signal.iirnotch(w0=f0, Q=q, fs=fs)
        try:
            return signal.filtfilt(b, a, x, method="gust")
        except Exception:
            return signal.filtfilt(b, a, x)
    return _notch_freq_domain(x, fs, f0, q)


def _lowpass(x: np.ndarray, fs: float, fc: float, order: int = 3) -> np.ndarray:
    if fc <= 0.0:
        return x.copy()
    if HAVE_SCIPY:
        wn = min(0.99, fc / max(1e-9, (fs / 2.0)))
        b, a = signal.butter(order, wn, btype="low")
        try:
            return signal.filtfilt(b, a, x, method="gust")
        except Exception:
            return signal.filtfilt(b, a, x)
    win = int(max(1, round(fs / max(1e-6, fc))))
    win = min(win, 2001)
    return _moving_average(x, win)


def _interpolate_invalid_values(x: np.ndarray, valid_flag: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).copy()
    valid = np.asarray(valid_flag, dtype=bool)
    invalid_idx = np.flatnonzero(~valid)
    if invalid_idx.size == 0:
        return arr

    n = len(arr)
    finite_valid = valid & np.isfinite(arr)
    for idx in invalid_idx:
        left = idx - 1
        while left >= 0 and not finite_valid[left]:
            left -= 1
        right = idx + 1
        while right < n and not finite_valid[right]:
            right += 1

        if left >= 0 and right < n:
            arr[idx] = 0.5 * (arr[left] + arr[right])
        elif left >= 0:
            arr[idx] = arr[left]
        elif right < n:
            arr[idx] = arr[right]
    return arr


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, num=max_points, dtype=int)


def _downsample_for_plot(x: np.ndarray, y: np.ndarray, max_points: int = 200_000) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= max_points:
        return np.asarray(x), np.asarray(y)
    step = max(1, n // max_points)
    return np.asarray(x)[::step], np.asarray(y)[::step]


def _infer_fs_hz(t_s: np.ndarray) -> float:
    if len(t_s) < 2:
        return 1.0
    dt = np.diff(np.asarray(t_s, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 1.0
    return float(1.0 / np.median(dt))


def params_fs_hz(t_s: np.ndarray) -> float:
    return _infer_fs_hz(t_s)


def _compute_mu_from_tensions(t_high: np.ndarray, t_low: np.ndarray, params: AnalysisParams) -> Tuple[np.ndarray, np.ndarray]:
    theta = math.radians(params.theta_deg)
    eps = 1e-9
    q_gate = (t_high >= params.tmin_gate_N) & (t_low >= params.tmin_gate_N)
    ratio = (t_high + eps) / (t_low + eps)
    ratio = np.clip(ratio, params.ratio_clip_min, params.ratio_clip_max)
    mu = np.log(ratio) / max(theta, 1e-9)
    mu = np.asarray(mu, dtype=float)
    mu[~q_gate] = np.nan
    return mu, q_gate.astype(int)


def _clip_stable_segments(
    t_s: np.ndarray,
    stable_segments_idx: Sequence[Tuple[int, int]],
    tlife_s: Optional[float],
    fs_hz: float,
) -> List[Tuple[int, int]]:
    if tlife_s is None:
        return list(stable_segments_idx)

    clipped: List[Tuple[int, int]] = []
    t_cut = float(tlife_s)
    for k0, k1 in stable_segments_idx:
        t0 = float(t_s[k0])
        t1 = float(t_s[k1]) if (k1 < len(t_s)) else float(t_s[-1])
        if t0 >= t_cut:
            continue
        if t1 > t_cut:
            k1c = int(min(len(t_s) - 1, max(k0 + 1, round(t_cut * fs_hz))))
            clipped.append((k0, k1c))
        else:
            clipped.append((k0, k1))
    return clipped


def _plot_labels(lang: str = "zh") -> Dict[str, str]:
    if str(lang).strip().lower().startswith("en"):
        return {
            "t": "Time t (s)",
            "tension": "Tension (N)",
            "tavg": "Mean tension (N)",
            "mu": "Friction coefficient μ",
            "title_closed_t": "Closed-loop: Tension vs Time",
            "title_mu": "Closed-loop: μ vs Time (μss & μth)",
            "th": "T_high",
            "tl": "T_low",
            "ta": "T_avg",
            "mu_f": "μ (filtered)",
            "mu_ss": "Baseline μss",
            "mu_th": "Threshold μth",
            "ss": "Stable segments",
            "tlife": "tlife",
        }
    return {
        "t": "时间 t (s)",
        "tension": "张力 (N)",
        "tavg": "平均张力 (N)",
        "mu": "摩擦系数 μ",
        "title_closed_t": "闭环：张力-时间",
        "title_mu": "闭环：摩擦系数-时间（μss 与 μth）",
        "th": "高张力侧 T_high",
        "tl": "低张力侧 T_low",
        "ta": "平均张力 T_avg",
        "mu_f": "μ (滤波后)",
        "mu_ss": "稳定段基线 μss",
        "mu_th": "超限阈值 μth",
        "ss": "连续稳定段",
        "tlife": "tlife",
    }


def _plot_tension_axis(
    ax,
    data: MonitorData,
    result: AnalysisResult,
    max_points: int,
    labels: Dict[str, str],
) -> None:
    tx, hi = _downsample_for_plot(data.t_s, result.display_t_high, max_points)
    _, lo = _downsample_for_plot(data.t_s, result.display_t_low, max_points)
    _, avg = _downsample_for_plot(data.t_s, result.display_t_avg, max_points)

    ax.clear()
    ax.plot(tx, hi, label=labels["th"], color="tab:blue")
    ax.plot(tx, lo, label=labels["tl"], color="tab:orange")
    ax.plot(tx, avg, label=labels["ta"], color="tab:green")
    ax.set_title(labels["title_closed_t"])
    ax.set_ylabel(labels["tension"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)


def _plot_mu_axis(
    ax,
    data: MonitorData,
    result: AnalysisResult,
    max_points: int,
    labels: Dict[str, str],
) -> None:
    tx, mu_eval = _downsample_for_plot(data.t_s, result.display_mu_eval, max_points)
    stable_segs_clip = _clip_stable_segments(data.t_s, result.stable_segments_idx, result.tlife_s, result.fs_hz)

    ax.clear()
    ax.plot(tx, mu_eval, label=labels["mu_f"], color="tab:blue")

    if result.mu_ss is not None and np.isfinite(result.mu_ss):
        ax.axhline(float(result.mu_ss), linestyle="--", label=f'{labels["mu_ss"]}={float(result.mu_ss):.4f}', color="tab:orange")
    if result.mu_th is not None and np.isfinite(result.mu_th):
        ax.axhline(float(result.mu_th), linestyle="--", label=f'{labels["mu_th"]}={float(result.mu_th):.4f}', color="tab:red")

    if stable_segs_clip:
        first = True
        for k0, k1 in stable_segs_clip:
            t0 = float(data.t_s[k0])
            t1 = float(data.t_s[k1]) if (k1 < len(data.t_s)) else float(data.t_s[-1])
            if first:
                ax.axvspan(t0, t1, alpha=0.30, label=labels["ss"], color="tab:cyan", zorder=1, linewidth=0)
                first = False
            else:
                ax.axvspan(t0, t1, alpha=0.30, color="tab:cyan", zorder=1, linewidth=0)

    if result.tlife_s is not None:
        ax.axvline(float(result.tlife_s), linestyle="--", label=f'{labels["tlife"]}≈{float(result.tlife_s):.1f}s', color="tab:purple")

    ax.set_title(labels["title_mu"])
    ax.set_xlabel(labels["t"])
    ax.set_ylabel(labels["mu"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)


def export_monitor_plots(
    data: MonitorData,
    result: AnalysisResult,
    params: AnalysisParams,
    out_dir: str,
    lang: str = "zh",
) -> Dict[str, str]:
    params = params.normalized()
    labels = _plot_labels(lang)
    plot_dir = os.path.join(os.path.abspath(out_dir), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    default_figsize = tuple(float(v) for v in matplotlib.rcParams["figure.figsize"])
    default_dpi = float(matplotlib.rcParams["figure.dpi"])

    fig_t = Figure(figsize=default_figsize, dpi=default_dpi)
    ax_t = fig_t.add_subplot(111)
    _plot_tension_axis(ax_t, data, result, params.max_plot_points, labels)
    ax_t.set_xlabel(labels["t"])
    fig_t.tight_layout(rect=[0, 0.10, 1, 1])
    p_tension = os.path.join(plot_dir, f'closed_tensions_{lang}.png')
    fig_t.savefig(p_tension, dpi=180)

    fig_mu = Figure(figsize=default_figsize, dpi=default_dpi)
    ax_mu = fig_mu.add_subplot(111)
    _plot_mu_axis(ax_mu, data, result, params.max_plot_points, labels)
    fig_mu.tight_layout(rect=[0, 0.14, 1, 1])
    p_mu = os.path.join(plot_dir, f'mu_with_baseline_threshold_{lang}.png')
    fig_mu.savefig(p_mu, dpi=180)
    fig_t.clear()
    fig_mu.clear()

    return {
        "closed_tension": p_tension,
        "mu_time": p_mu,
    }


def _find_stable_segments(
    mu_f: np.ndarray,
    t: np.ndarray,
    q_valid: np.ndarray,
    params: AnalysisParams,
    end_idx: Optional[int] = None,
) -> List[Tuple[int, int]]:
    n_all = len(mu_f)
    if n_all == 0:
        return []
    n = n_all if end_idx is None else int(max(0, min(n_all, end_idx)))
    if n <= 0:
        return []

    win = int(round(params.stable_win_s * params_fs_hz(t)))
    win = max(win, 50)
    step = max(1, win // 10)
    stable_mask = np.zeros(n_all, dtype=bool)

    for k0 in range(0, max(1, n - win), step):
        k1 = k0 + win
        if k1 > n:
            break
        if float(np.mean(q_valid[k0:k1])) < params.stable_valid_min:
            continue

        seg = mu_f[k0:k1]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 10:
            continue
        if float(np.std(seg)) > params.stable_sigma_max:
            continue

        yy = mu_f[k0:k1]
        vv = np.asarray(q_valid[k0:k1], dtype=bool)
        mm = np.isfinite(yy) & vv
        if int(mm.sum()) < 10:
            mm = np.isfinite(yy)
        if int(mm.sum()) < 10:
            continue

        vals = yy[mm]
        nvals = int(len(vals))
        edge = max(5, int(round(0.10 * nvals)))
        edge = min(edge, max(5, nvals // 2))
        head = np.nanmedian(vals[:edge])
        tail = np.nanmedian(vals[-edge:])
        drift = float(abs(tail - head))
        if drift > params.stable_slope_max:
            continue
        stable_mask[k0:k1] = True

    segs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if stable_mask[i]:
            j = i + 1
            while j < n and stable_mask[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1

    if params.stable_hold_s > 0.0:
        fs_hz = params_fs_hz(t)
        filtered: List[Tuple[int, int]] = []
        for a, b in segs:
            dur_s = float(max(0.0, (b - a) / max(1e-9, fs_hz)))
            if dur_s + 1e-9 >= params.stable_hold_s:
                filtered.append((a, b))
        segs = filtered
    return segs


def _find_stable_baseline(
    mu_f: np.ndarray,
    t: np.ndarray,
    q_valid: np.ndarray,
    params: AnalysisParams,
    end_idx: Optional[int] = None,
) -> Tuple[Optional[Tuple[int, int]], Optional[float], List[Tuple[int, int]]]:
    segs = _find_stable_segments(mu_f, t, q_valid, params, end_idx=end_idx)
    if not segs:
        return None, None, []
    k0, k1 = segs[0]
    yy = mu_f[k0:k1]
    qq = np.asarray(q_valid[k0:k1], dtype=bool)
    mask = np.isfinite(yy) & qq
    if int(mask.sum()) < 5:
        mask = np.isfinite(yy)
    if int(mask.sum()) <= 0:
        return (k0, k1), None, segs
    mu_ss = float(np.nanmedian(yy[mask]))
    return (k0, k1), mu_ss, segs


def _find_failure_time(
    mu_f: np.ndarray,
    t: np.ndarray,
    q_valid: np.ndarray,
    mu_ss: Optional[float],
    params: AnalysisParams,
    start_idx: int = 0,
) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    if mu_ss is None or (not np.isfinite(mu_ss)):
        return None, None, None
    mu_th = float(mu_ss * (1.0 + params.fail_delta))
    fs_hz = params_fs_hz(t)
    dt = 1.0 / max(1e-9, fs_hz)
    hold_s = float(max(0.0, params.fail_hold_s))
    start_idx = int(max(0, min(len(mu_f) - 1, start_idx))) if len(mu_f) else 0

    run_start: Optional[int] = None
    for i in range(start_idx, len(mu_f)):
        if not bool(q_valid[i]):
            continue
        above = bool(np.isfinite(mu_f[i]) and (mu_f[i] > mu_th))
        if above:
            if run_start is None:
                run_start = i
            if float(t[i] - t[run_start] + dt) >= hold_s:
                return float(t[run_start]), int(run_start), mu_th
        else:
            run_start = None
    return None, None, mu_th


def _build_spans(mask: np.ndarray, t_s: np.ndarray) -> List[Tuple[float, float]]:
    spans: List[Tuple[float, float]] = []
    active = False
    start_idx = 0
    for i, flag in enumerate(mask):
        if bool(flag) and (not active):
            active = True
            start_idx = i
        elif (not bool(flag)) and active:
            active = False
            spans.append((float(t_s[start_idx]), float(t_s[max(start_idx, i - 1)])))
    if active and len(mask) > 0:
        spans.append((float(t_s[start_idx]), float(t_s[-1])))
    return spans


def _find_column(actual_columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lookup = {name.lower(): name for name in actual_columns}
    for alias in aliases:
        hit = lookup.get(alias.lower())
        if hit is not None:
            return hit
    return None


def load_monitor_db(db_path: str, table_name: str = "Data") -> MonitorData:
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(db_path)
    try:
        pragma_rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        if not pragma_rows:
            raise ValueError(f'表 "{table_name}" 不存在或不可读')
        actual_columns = [str(row[1]) for row in pragma_rows]

        col_time = _find_column(actual_columns, ["t_s", "time", "ts"])
        col_high = _find_column(actual_columns, ["t_high_n", "t_high", "high", "high_n"])
        col_low = _find_column(actual_columns, ["t_low_n", "t_low", "low", "low_n"])
        col_avg = _find_column(actual_columns, ["t_avg_n", "t_avg", "avg", "avg_n"])
        col_fric = _find_column(actual_columns, ["f_fric_n", "f_fric", "fric", "friction"])
        col_mu = _find_column(actual_columns, ["mu", "mu_filt", "mu_raw"])
        col_qf = _find_column(actual_columns, ["quality_flag", "q_valid", "质量标志"])

        missing = [name for name, col in [("t_s", col_time), ("t_high_N", col_high), ("t_low_N", col_low)] if col is None]
        if missing:
            raise ValueError("缺少必要列: " + ", ".join(missing))

        ordered = [col_time, col_high, col_low]
        optional = [col_avg, col_fric, col_mu, col_qf]
        select_cols = ordered + [col for col in optional if col is not None]
        sql = "SELECT " + ", ".join(f'"{c}"' for c in select_cols) + f' FROM "{table_name}"'

        rows = conn.execute(sql)
        t_s: List[float] = []
        t_high: List[float] = []
        t_low: List[float] = []
        t_avg: List[Optional[float]] = []
        f_fric: List[Optional[float]] = []
        mu_vals: List[Optional[float]] = []
        qf_vals: List[int] = []

        idx_map: Dict[str, int] = {name: i for i, name in enumerate(select_cols)}
        while True:
            chunk = rows.fetchmany(50000)
            if not chunk:
                break
            for row in chunk:
                t_s.append(float(row[idx_map[col_time]]))
                t_high.append(float(row[idx_map[col_high]]))
                t_low.append(float(row[idx_map[col_low]]))
                if col_avg is not None:
                    v = row[idx_map[col_avg]]
                    t_avg.append(None if v is None else float(v))
                if col_fric is not None:
                    v = row[idx_map[col_fric]]
                    f_fric.append(None if v is None else float(v))
                if col_mu is not None:
                    v = row[idx_map[col_mu]]
                    mu_vals.append(None if v is None else float(v))
                if col_qf is not None:
                    v = row[idx_map[col_qf]]
                    qf_vals.append(int(v) if v is not None else 0)

        t_s_arr = np.asarray(t_s, dtype=float)
        t_high_arr = np.asarray(t_high, dtype=float)
        t_low_arr = np.asarray(t_low, dtype=float)

        if col_avg is not None:
            t_avg_arr = np.asarray([np.nan if v is None else v for v in t_avg], dtype=float)
        else:
            t_avg_arr = (t_high_arr + t_low_arr) / 2.0

        if col_fric is not None:
            f_fric_arr = np.asarray([np.nan if v is None else v for v in f_fric], dtype=float)
        else:
            f_fric_arr = t_high_arr - t_low_arr

        mu_arr: Optional[np.ndarray]
        if col_mu is not None:
            mu_arr = np.asarray([np.nan if v is None else v for v in mu_vals], dtype=float)
        else:
            mu_arr = None

        if col_qf is not None:
            qf_arr = np.asarray(qf_vals, dtype=int)
        else:
            qf_arr = np.ones_like(t_s_arr, dtype=int)

        return MonitorData(
            db_path=os.path.abspath(db_path),
            table_name=table_name,
            row_count=int(len(t_s_arr)),
            t_s=t_s_arr,
            t_high_N=t_high_arr,
            t_low_N=t_low_arr,
            t_avg_N=t_avg_arr,
            f_fric_N=f_fric_arr,
            mu=mu_arr,
            quality_flag=qf_arr,
        )
    finally:
        conn.close()


def analyze_monitor_data(data: MonitorData, params: AnalysisParams) -> AnalysisResult:
    params = params.normalized()
    fs_hz = _infer_fs_hz(data.t_s)

    _, q_gate = _compute_mu_from_tensions(data.t_high_N, data.t_low_N, params)
    if data.mu is not None:
        mu_eval = np.asarray(data.mu, dtype=float).copy()
    else:
        mu_eval, _ = _compute_mu_from_tensions(data.t_high_N, data.t_low_N, params)
    mu_raw = mu_eval.copy()
    mu_hampel = mu_eval.copy()

    q_valid = (np.asarray(data.quality_flag, dtype=int) != 0) & (q_gate != 0) & np.isfinite(mu_eval)

    stable_idx0, mu_ss0, stable_segs0 = _find_stable_baseline(mu_eval, data.t_s, q_valid.astype(int), params, end_idx=None)
    start_fail_idx0 = int(stable_idx0[1]) if stable_idx0 is not None else 0
    tlife0_s, tlife0_idx, _ = _find_failure_time(mu_eval, data.t_s, q_valid.astype(int), mu_ss0, params, start_idx=start_fail_idx0)

    if tlife0_idx is not None:
        stable_idx, mu_ss, stable_segs = _find_stable_baseline(mu_eval, data.t_s, q_valid.astype(int), params, end_idx=tlife0_idx + 1)
        if mu_ss is None or (not np.isfinite(mu_ss)):
            stable_idx, mu_ss, stable_segs = stable_idx0, mu_ss0, stable_segs0
    else:
        stable_idx, mu_ss, stable_segs = stable_idx0, mu_ss0, stable_segs0

    start_fail_idx = int(stable_idx[1]) if stable_idx is not None else 0
    tlife_s, tlife_idx, mu_th = _find_failure_time(mu_eval, data.t_s, q_valid.astype(int), mu_ss, params, start_idx=start_fail_idx)

    display_t_high = np.asarray(data.t_high_N, dtype=float).copy()
    display_t_low = np.asarray(data.t_low_N, dtype=float).copy()
    display_t_avg = np.asarray(data.t_avg_N, dtype=float).copy()
    display_mu_raw = mu_raw.copy()
    display_mu_eval = mu_eval.copy()

    return AnalysisResult(
        fs_hz=fs_hz,
        mu_raw=mu_raw,
        mu_hampel=mu_hampel,
        mu_eval=mu_eval,
        q_valid=q_valid.astype(int),
        stable_segments_idx=stable_segs,
        stable_window_idx=stable_idx,
        mu_ss=None if mu_ss is None else float(mu_ss),
        mu_th=None if mu_th is None else float(mu_th),
        tlife_s=None if tlife_s is None else float(tlife_s),
        tlife_idx=tlife_idx,
        valid_ratio=float(np.mean(q_valid)) if len(q_valid) else 0.0,
        display_t_high=display_t_high,
        display_t_low=display_t_low,
        display_t_avg=display_t_avg,
        display_mu_raw=display_mu_raw,
        display_mu_eval=display_mu_eval,
    )


def summary_dict(data: MonitorData, result: AnalysisResult) -> Dict[str, object]:
    return {
        "db_path": data.db_path,
        "mu_ss": result.mu_ss,
        "mu_th": result.mu_th,
        "tlife_s": result.tlife_s,
    }


def _summary_value_text(value: Optional[float], digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "None"
    return f"{float(value):.{digits}f}"


def _summary_lines(payload: Dict[str, object]) -> List[str]:
    return [
        f"数据库路径: {payload['db_path']}",
        f"μss: {_summary_value_text(payload['mu_ss'])}",
        f"μth: {_summary_value_text(payload['mu_th'])}",
        f"tlife: {_summary_value_text(payload['tlife_s'], 3)}",
    ]


class MonitorAnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("针钩监测数据分析与绘图")
        self.root.geometry("1480x920")

        self._queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self._worker_alive = False
        self._last_data: Optional[MonitorData] = None
        self._last_result: Optional[AnalysisResult] = None

        self.db_path_var = tk.StringVar(value=DEFAULT_DB_PATH if os.path.exists(DEFAULT_DB_PATH) else "")
        self.table_name_var = tk.StringVar(value="Data")
        self.status_var = tk.StringVar(value="选择数据库后点击“分析并绘图”。")
        self._param_vars: Dict[str, tk.Variable] = {}

        self._build_ui()
        self.root.after(120, self._poll_queue)

    def _build_ui(self) -> None:
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer, padding=10)
        right = ttk.Frame(outer, padding=10)
        outer.add(left, weight=0)
        outer.add(right, weight=1)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        file_box = ttk.LabelFrame(parent, text="数据源", padding=10)
        file_box.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(file_box, text="数据库").grid(row=0, column=0, sticky="w")
        ttk.Entry(file_box, textvariable=self.db_path_var, width=48).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 6))
        ttk.Button(file_box, text="浏览...", command=self._browse_db).grid(row=1, column=2, padx=(8, 0), sticky="ew")

        ttk.Label(file_box, text="表名").grid(row=2, column=0, sticky="w")
        ttk.Entry(file_box, textvariable=self.table_name_var, width=18).grid(row=2, column=1, sticky="w", pady=(4, 0))
        ttk.Button(file_box, text="使用参考库", command=self._use_default_db).grid(row=2, column=2, padx=(8, 0), sticky="ew")
        file_box.columnconfigure(0, weight=1)
        file_box.columnconfigure(1, weight=1)

        calc_box = ttk.LabelFrame(parent, text="计算参数", padding=10)
        calc_box.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        row = 0
        row = self._add_entry(calc_box, row, "包角 θ (deg)", "theta_deg", 100.0)
        row = self._add_check(calc_box, row, "用高/低张力重算 μ", "recompute_mu", False)
        row = self._add_check(calc_box, row, "显示时插值无效点", "interpolate_invalid_display", True)
        row = self._add_entry(calc_box, row, "门控下限 T_min (N)", "tmin_gate_N", 0.08)
        row = self._add_entry(calc_box, row, "比值裁剪最小值", "ratio_clip_min", 0.3)
        row = self._add_entry(calc_box, row, "比值裁剪最大值", "ratio_clip_max", 8.0)
        row = self._add_check(calc_box, row, "启用 Hampel 去毛刺", "use_hampel", True)
        row = self._add_entry(calc_box, row, "Hampel 窗口 (s)", "hampel_win_s", 1.0)
        row = self._add_entry(calc_box, row, "Hampel 阈值 (σ)", "hampel_nsig", 3.0)
        row = self._add_entry(calc_box, row, "陷波中心频率 (Hz, 0=关)", "notch_f0_hz", 0.0)
        row = self._add_entry(calc_box, row, "陷波 Q (0=关)", "notch_q", 0.0)
        row = self._add_entry(calc_box, row, "低通截止频率 (Hz)", "lowpass_fc_hz", 2.5)
        row = self._add_entry(calc_box, row, "稳态窗口 Wss (s)", "stable_win_s", 1200.0)
        row = self._add_entry(calc_box, row, "稳态最短保持 Whold (s)", "stable_hold_s", 3600.0)
        row = self._add_entry(calc_box, row, "稳态标准差阈值 σmax", "stable_sigma_max", 0.03)
        row = self._add_entry(calc_box, row, "稳态总漂移阈值 Δμmax", "stable_slope_max", 0.015)
        row = self._add_entry(calc_box, row, "稳态有效比例 qmin", "stable_valid_min", 0.9)
        row = self._add_entry(calc_box, row, "失效阈值增量 δ", "fail_delta", 0.25)
        row = self._add_entry(calc_box, row, "持续超阈值 Wpersist (s)", "fail_hold_s", 300.0)
        self._add_entry(calc_box, row, "绘图最大点数", "max_plot_points", 40000)

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X, pady=(0, 10))
        self.analyze_btn = ttk.Button(btn_row, text="分析并绘图", command=self._run_analysis_async)
        self.analyze_btn.pack(side=tk.LEFT)
        ttk.Button(btn_row, text="清空结果", command=self._clear_result).pack(side=tk.LEFT, padx=(8, 0))

        result_box = ttk.LabelFrame(parent, text="结果摘要", padding=10)
        result_box.pack(fill=tk.BOTH, expand=True)
        self.summary_text = tk.Text(result_box, width=44, height=24, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        self.summary_text.configure(state=tk.DISABLED)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, anchor="w")

        self.figure = Figure(figsize=(11, 8), dpi=100)
        self.ax_tension = self.figure.add_subplot(211)
        self.ax_mu = self.figure.add_subplot(212, sharex=self.ax_tension)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X)
        self._draw_placeholder()

    def _draw_placeholder(self) -> None:
        self.ax_tension.clear()
        self.ax_mu.clear()
        self.ax_tension.set_title("闭环：张力-时间")
        self.ax_mu.set_title("闭环：摩擦系数-时间（μss 与 μth）")
        self.ax_tension.set_ylabel("张力 (N)")
        self.ax_mu.set_ylabel("摩擦系数 μ")
        self.ax_mu.set_xlabel("时间 (s)")
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.10, hspace=0.78)
        self.canvas.draw_idle()

    def _add_entry(self, parent: ttk.Widget, row: int, label: str, key: str, default) -> int:
        var = tk.StringVar(value=str(default))
        self._param_vars[key] = var
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=18).grid(row=row, column=1, sticky="ew", pady=2, padx=(8, 0))
        return row + 1

    def _add_check(self, parent: ttk.Widget, row: int, label: str, key: str, default: bool) -> int:
        var = tk.BooleanVar(value=bool(default))
        self._param_vars[key] = var
        ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        return row + 1

    def _browse_db(self) -> None:
        path = filedialog.askopenfilename(
            title="选择监测数据库",
            filetypes=[("SQLite DB", "*.db *.sqlite *.sqlite3"), ("所有文件", "*.*")],
        )
        if path:
            self.db_path_var.set(path)

    def _use_default_db(self) -> None:
        self.db_path_var.set(DEFAULT_DB_PATH)
        self.table_name_var.set("Data")

    def _get_params(self) -> AnalysisParams:
        def f(name: str) -> float:
            return float(str(self._param_vars[name].get()).strip())

        def b(name: str) -> bool:
            return bool(self._param_vars[name].get())

        return AnalysisParams(
            theta_deg=f("theta_deg"),
            recompute_mu=b("recompute_mu"),
            interpolate_invalid_display=b("interpolate_invalid_display"),
            tmin_gate_N=f("tmin_gate_N"),
            ratio_clip_min=f("ratio_clip_min"),
            ratio_clip_max=f("ratio_clip_max"),
            use_hampel=b("use_hampel"),
            hampel_win_s=f("hampel_win_s"),
            hampel_nsig=f("hampel_nsig"),
            notch_f0_hz=f("notch_f0_hz"),
            notch_q=f("notch_q"),
            lowpass_fc_hz=f("lowpass_fc_hz"),
            stable_win_s=f("stable_win_s"),
            stable_hold_s=f("stable_hold_s"),
            stable_sigma_max=f("stable_sigma_max"),
            stable_slope_max=f("stable_slope_max"),
            stable_valid_min=f("stable_valid_min"),
            fail_delta=f("fail_delta"),
            fail_hold_s=f("fail_hold_s"),
            max_plot_points=int(float(str(self._param_vars["max_plot_points"].get()).strip())),
        )

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
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        self._worker_alive = True
        self.analyze_btn.configure(state=tk.DISABLED)
        self.status_var.set("正在读取数据库并计算，请稍候...")
        worker = threading.Thread(target=self._analysis_worker, args=(db_path, table_name, params), daemon=True)
        worker.start()

    def _analysis_worker(self, db_path: str, table_name: str, params: AnalysisParams) -> None:
        try:
            data = load_monitor_db(db_path, table_name=table_name)
            result = analyze_monitor_data(data, params)
            self._queue.put(("ok", (data, result)))
        except Exception:
            self._queue.put(("err", traceback.format_exc()))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                self._worker_alive = False
                self.analyze_btn.configure(state=tk.NORMAL)
                if kind == "ok":
                    data, result = payload
                    self._last_data = data
                    self._last_result = result
                    self._update_summary(data, result)
                    self._update_plots(data, result)
                    self.status_var.set(
                        f"分析完成: {os.path.basename(data.db_path)} | rows={data.row_count} | fs={result.fs_hz:.3f} Hz"
                    )
                else:
                    self.status_var.set("分析失败")
                    messagebox.showerror("分析失败", str(payload))
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _update_summary(self, data: MonitorData, result: AnalysisResult) -> None:
        payload = summary_dict(data, result)
        lines = _summary_lines(payload)
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.configure(state=tk.DISABLED)

    def _update_plots(self, data: MonitorData, result: AnalysisResult) -> None:
        self.ax_tension.clear()
        self.ax_mu.clear()

        max_points = self._get_params().normalized().max_plot_points
        tx, hi = _downsample_for_plot(data.t_s, result.display_t_high, max_points)
        _, lo = _downsample_for_plot(data.t_s, result.display_t_low, max_points)
        _, avg = _downsample_for_plot(data.t_s, result.display_t_avg, max_points)
        _, mu_eval = _downsample_for_plot(data.t_s, result.display_mu_eval, max_points)

        stable_segs_clip: List[Tuple[int, int]] = []
        if result.tlife_s is None:
            stable_segs_clip = list(result.stable_segments_idx)
        else:
            t_cut = float(result.tlife_s)
            for k0, k1 in result.stable_segments_idx:
                t0 = float(data.t_s[k0])
                t1 = float(data.t_s[k1]) if (k1 < len(data.t_s)) else float(data.t_s[-1])
                if t0 >= t_cut:
                    continue
                if t1 > t_cut:
                    k1c = int(min(len(data.t_s) - 1, max(k0 + 1, round(t_cut * result.fs_hz))))
                    stable_segs_clip.append((k0, k1c))
                else:
                    stable_segs_clip.append((k0, k1))

        self.ax_tension.plot(tx, hi, label="高张力侧 T_high", color="tab:blue")
        self.ax_tension.plot(tx, lo, label="低张力侧 T_low", color="tab:orange")
        self.ax_tension.plot(tx, avg, label="平均张力 T_avg", color="tab:green")
        self.ax_tension.set_title("闭环：张力-时间")
        self.ax_tension.set_ylabel("张力 (N)")
        self.ax_tension.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)

        self.ax_mu.plot(tx, mu_eval, label="μ (滤波后)", color="tab:blue")
        if result.mu_ss is not None and np.isfinite(result.mu_ss):
            self.ax_mu.axhline(result.mu_ss, linestyle="--", label=f"稳定段基线 μss={result.mu_ss:.4f}", color="tab:orange")
        if result.mu_th is not None and np.isfinite(result.mu_th):
            self.ax_mu.axhline(result.mu_th, linestyle="--", label=f"超限阈值 μth={result.mu_th:.4f}", color="tab:red")
        if stable_segs_clip:
            first = True
            for k0, k1 in stable_segs_clip:
                t0 = float(data.t_s[k0])
                t1 = float(data.t_s[k1]) if (k1 < len(data.t_s)) else float(data.t_s[-1])
                if first:
                    self.ax_mu.axvspan(t0, t1, alpha=0.30, label="连续稳定段", color="tab:cyan", zorder=1, linewidth=0)
                    first = False
                else:
                    self.ax_mu.axvspan(t0, t1, alpha=0.30, color="tab:cyan", zorder=1, linewidth=0)
        if result.tlife_s is not None:
            self.ax_mu.axvline(float(result.tlife_s), linestyle="--", label=f"tlife≈{float(result.tlife_s):.1f}s", color="tab:purple")
        self.ax_mu.set_title("闭环：摩擦系数-时间（μss 与 μth）")
        self.ax_mu.set_xlabel("时间 (s)")
        self.ax_mu.set_ylabel("摩擦系数 μ")
        self.ax_mu.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.10, hspace=0.78)
        self.canvas.draw_idle()

    def _clear_result(self) -> None:
        self._last_data = None
        self._last_result = None
        self.status_var.set("结果已清空。")
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.configure(state=tk.DISABLED)
        self._draw_placeholder()


class MonitorAnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("针钩监测数据分析与绘图")
        self.root.geometry("1520x980")

        self._queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self._worker_alive = False
        self._last_data: Optional[MonitorData] = None
        self._last_result: Optional[AnalysisResult] = None
        self._last_params: Optional[AnalysisParams] = None

        self.db_path_var = tk.StringVar(value=DEFAULT_DB_PATH if os.path.exists(DEFAULT_DB_PATH) else "")
        self.table_name_var = tk.StringVar(value="Data")
        self.status_var = tk.StringVar(value="选择数据库后点击“分析并绘图”。")
        self.export_lang_var = tk.StringVar(value="中文")
        self._param_vars: Dict[str, tk.Variable] = {}

        self._build_ui()
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_mousewheel, add="+")
        self.root.after(120, self._poll_queue)

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root)
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
        delta = 0
        if getattr(event, "delta", 0):
            delta = -int(event.delta / 120)
        elif getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        if delta != 0:
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

        calc_box = ttk.LabelFrame(parent, text="计算参数", padding=10)
        calc_box.pack(fill=tk.BOTH, expand=False, pady=(0, 10))

        row = 0
        row = self._add_entry(calc_box, row, "包角 θ (deg)", "theta_deg", 100.0)
        row = self._add_check(calc_box, row, "显示插值值无效点", "interpolate_invalid_display", True)
        row = self._add_entry(calc_box, row, "门控下限 T_min (N)", "tmin_gate_N", 0.08)
        row = self._add_entry(calc_box, row, "比值裁剪最小值", "ratio_clip_min", 0.3)
        row = self._add_entry(calc_box, row, "比值裁剪最大值", "ratio_clip_max", 8.0)
        row = self._add_check(calc_box, row, "启用 Hampel 去毛刺", "use_hampel", True)
        row = self._add_entry(calc_box, row, "Hampel 窗口 (s)", "hampel_win_s", 1.0)
        row = self._add_entry(calc_box, row, "Hampel 阈值 (σ)", "hampel_nsig", 3.0)
        row = self._add_entry(calc_box, row, "陷波中心频率 (Hz, 0=关)", "notch_f0_hz", 0.0)
        row = self._add_entry(calc_box, row, "陷波 Q (0=关)", "notch_q", 0.0)
        row = self._add_entry(calc_box, row, "低通截止频率 (Hz)", "lowpass_fc_hz", 2.5)
        row = self._add_entry(calc_box, row, "稳态窗口 Wss (s)", "stable_win_s", 1200.0)
        row = self._add_entry(calc_box, row, "稳态最短保持 Whold (s)", "stable_hold_s", 3600.0)
        row = self._add_entry(calc_box, row, "稳态标准差阈值 σmax", "stable_sigma_max", 0.03)
        row = self._add_entry(calc_box, row, "稳态总漂移阈值 Δμmax", "stable_slope_max", 0.015)
        row = self._add_entry(calc_box, row, "稳态有效比例 qmin", "stable_valid_min", 0.9)
        row = self._add_entry(calc_box, row, "失效阈值增量 δ", "fail_delta", 0.25)
        row = self._add_entry(calc_box, row, "持续超阈值 Wpersist (s)", "fail_hold_s", 300.0)
        self._add_entry(calc_box, row, "绘图最大点数", "max_plot_points", 40000)
        calc_box.columnconfigure(1, weight=1)

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X, pady=(0, 8))
        self.analyze_btn = ttk.Button(btn_row, text="分析并绘图", command=self._run_analysis_async)
        self.analyze_btn.pack(side=tk.LEFT)
        ttk.Button(btn_row, text="清空结果", command=self._clear_result).pack(side=tk.LEFT, padx=(8, 0))

        export_row = ttk.Frame(parent)
        export_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(export_row, text="导出语言").pack(side=tk.LEFT)
        ttk.Combobox(
            export_row,
            textvariable=self.export_lang_var,
            values=("中文", "English"),
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=(8, 12))
        ttk.Button(export_row, text="导出图片...", command=self._export_plots).pack(side=tk.LEFT)

        result_box = ttk.LabelFrame(parent, text="结果摘要", padding=10)
        result_box.pack(fill=tk.BOTH, expand=True)
        self.summary_text = tk.Text(result_box, width=44, height=20, wrap=tk.WORD)
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

    def _draw_placeholder(self) -> None:
        labels = _plot_labels("zh")
        self.ax_tension.clear()
        self.ax_mu.clear()
        self.ax_tension.set_title(labels["title_closed_t"])
        self.ax_mu.set_title(labels["title_mu"])
        self.ax_tension.set_ylabel(labels["tension"])
        self.ax_mu.set_ylabel(labels["mu"])
        self.ax_mu.set_xlabel(labels["t"])
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.18, hspace=0.92)
        self.canvas.draw_idle()

    def _add_entry(self, parent: ttk.Widget, row: int, label: str, key: str, default) -> int:
        var = tk.StringVar(value=str(default))
        self._param_vars[key] = var
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=18).grid(row=row, column=1, sticky="ew", pady=2, padx=(8, 0))
        return row + 1

    def _add_check(self, parent: ttk.Widget, row: int, label: str, key: str, default: bool) -> int:
        var = tk.BooleanVar(value=bool(default))
        self._param_vars[key] = var
        ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        return row + 1

    def _browse_db(self) -> None:
        path = filedialog.askopenfilename(
            title="选择监测数据库",
            filetypes=[("SQLite DB", "*.db *.sqlite *.sqlite3"), ("所有文件", "*.*")],
        )
        if path:
            self.db_path_var.set(path)

    def _get_params(self) -> AnalysisParams:
        def f(name: str) -> float:
            return float(str(self._param_vars[name].get()).strip())

        def b(name: str) -> bool:
            return bool(self._param_vars[name].get())

        return AnalysisParams(
            theta_deg=f("theta_deg"),
            interpolate_invalid_display=b("interpolate_invalid_display"),
            tmin_gate_N=f("tmin_gate_N"),
            ratio_clip_min=f("ratio_clip_min"),
            ratio_clip_max=f("ratio_clip_max"),
            use_hampel=b("use_hampel"),
            hampel_win_s=f("hampel_win_s"),
            hampel_nsig=f("hampel_nsig"),
            notch_f0_hz=f("notch_f0_hz"),
            notch_q=f("notch_q"),
            lowpass_fc_hz=f("lowpass_fc_hz"),
            stable_win_s=f("stable_win_s"),
            stable_hold_s=f("stable_hold_s"),
            stable_sigma_max=f("stable_sigma_max"),
            stable_slope_max=f("stable_slope_max"),
            stable_valid_min=f("stable_valid_min"),
            fail_delta=f("fail_delta"),
            fail_hold_s=f("fail_hold_s"),
            max_plot_points=int(float(str(self._param_vars["max_plot_points"].get()).strip())),
        )

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
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        self._worker_alive = True
        self.analyze_btn.configure(state=tk.DISABLED)
        self.status_var.set("正在读取数据库并计算，请稍候...")
        worker = threading.Thread(target=self._analysis_worker, args=(db_path, table_name, params), daemon=True)
        worker.start()

    def _analysis_worker(self, db_path: str, table_name: str, params: AnalysisParams) -> None:
        try:
            data = load_monitor_db(db_path, table_name=table_name)
            result = analyze_monitor_data(data, params)
            self._queue.put(("ok", (data, result, params.normalized())))
        except Exception:
            self._queue.put(("err", traceback.format_exc()))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                self._worker_alive = False
                self.analyze_btn.configure(state=tk.NORMAL)
                if kind == "ok":
                    data, result, params = payload
                    self._last_data = data
                    self._last_result = result
                    self._last_params = params
                    self._update_summary(data, result)
                    self._update_plots(data, result)
                    self.status_var.set(
                        f"分析完成: {os.path.basename(data.db_path)} | rows={data.row_count} | fs={result.fs_hz:.3f} Hz"
                    )
                else:
                    self.status_var.set("分析失败")
                    messagebox.showerror("分析失败", str(payload))
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _update_summary(self, data: MonitorData, result: AnalysisResult) -> None:
        payload = summary_dict(data, result)
        lines = _summary_lines(payload)
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.configure(state=tk.DISABLED)

    def _update_plots(self, data: MonitorData, result: AnalysisResult) -> None:
        labels = _plot_labels("zh")
        max_points = (self._last_params or self._get_params().normalized()).max_plot_points
        _plot_tension_axis(self.ax_tension, data, result, max_points, labels)
        _plot_mu_axis(self.ax_mu, data, result, max_points, labels)
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.18, hspace=0.92)
        self.canvas.draw_idle()
        self.root.after_idle(self._on_body_configure)

    def _export_plots(self) -> None:
        if self._last_data is None or self._last_result is None or self._last_params is None:
            messagebox.showwarning("提示", "请先完成分析，再导出图片。")
            return

        out_dir = filedialog.askdirectory(title="选择图片导出目录")
        if not out_dir:
            return

        lang = "en" if self.export_lang_var.get().strip().lower().startswith("english") else "zh"
        try:
            outputs = export_monitor_plots(self._last_data, self._last_result, self._last_params, out_dir, lang=lang)
        except Exception:
            messagebox.showerror("导出失败", traceback.format_exc())
            return

        self.status_var.set(f"图片已导出: {os.path.basename(out_dir)} | lang={lang}")
        messagebox.showinfo("导出完成", "\n".join(outputs.values()))

    def _clear_result(self) -> None:
        self._last_data = None
        self._last_result = None
        self._last_params = None
        self.status_var.set("结果已清空。")
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.configure(state=tk.DISABLED)
        self._draw_placeholder()


class MonitorAnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("针钩监测数据分析与绘图")
        self.root.geometry("1520x980")

        self._queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self._worker_alive = False
        self._last_data: Optional[MonitorData] = None
        self._last_result: Optional[AnalysisResult] = None
        self._last_params: Optional[AnalysisParams] = None

        self.db_path_var = tk.StringVar(value=DEFAULT_DB_PATH if os.path.exists(DEFAULT_DB_PATH) else "")
        self.table_name_var = tk.StringVar(value="Data")
        self.status_var = tk.StringVar(value="选择数据库后点击“分析并绘图”。")
        self.export_lang_var = tk.StringVar(value="中文")
        self._param_vars: Dict[str, tk.Variable] = {}

        self._build_ui()
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_mousewheel, add="+")
        self.root.after(120, self._poll_queue)

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root)
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
        delta = 0
        if getattr(event, "delta", 0):
            delta = -int(event.delta / 120)
        elif getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        if delta != 0:
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
        self._add_entry(calc_box, row, "绘图最大点数", "max_plot_points", 40000)
        calc_box.columnconfigure(1, weight=1)

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X, pady=(0, 8))
        self.analyze_btn = ttk.Button(btn_row, text="分析并绘图", command=self._run_analysis_async)
        self.analyze_btn.pack(side=tk.LEFT)
        ttk.Button(btn_row, text="清空结果", command=self._clear_result).pack(side=tk.LEFT, padx=(8, 0))

        export_row = ttk.Frame(parent)
        export_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(export_row, text="导出语言").pack(side=tk.LEFT)
        ttk.Combobox(
            export_row,
            textvariable=self.export_lang_var,
            values=("中文", "English"),
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=(8, 12))
        ttk.Button(export_row, text="导出图片...", command=self._export_plots).pack(side=tk.LEFT)

        result_box = ttk.LabelFrame(parent, text="结果摘要", padding=10)
        result_box.pack(fill=tk.BOTH, expand=True)
        self.summary_text = tk.Text(result_box, width=44, height=20, wrap=tk.WORD)
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

    def _draw_placeholder(self) -> None:
        labels = _plot_labels("zh")
        self.ax_tension.clear()
        self.ax_mu.clear()
        self.ax_tension.set_title(labels["title_closed_t"])
        self.ax_mu.set_title(labels["title_mu"])
        self.ax_tension.set_ylabel(labels["tension"])
        self.ax_mu.set_ylabel(labels["mu"])
        self.ax_mu.set_xlabel(labels["t"])
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.18, hspace=0.92)
        self.canvas.draw_idle()

    def _add_entry(self, parent: ttk.Widget, row: int, label: str, key: str, default) -> int:
        var = tk.StringVar(value=str(default))
        self._param_vars[key] = var
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=18).grid(row=row, column=1, sticky="ew", pady=2, padx=(8, 0))
        return row + 1

    def _browse_db(self) -> None:
        path = filedialog.askopenfilename(
            title="选择监测数据库",
            filetypes=[("SQLite DB", "*.db *.sqlite *.sqlite3"), ("所有文件", "*.*")],
        )
        if path:
            self.db_path_var.set(path)

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
            max_plot_points=int(float(str(self._param_vars["max_plot_points"].get()).strip())),
        )

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
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        self._worker_alive = True
        self.analyze_btn.configure(state=tk.DISABLED)
        self.status_var.set("正在从数据库提取数据并绘制图表，请稍候...")
        worker = threading.Thread(target=self._analysis_worker, args=(db_path, table_name, params), daemon=True)
        worker.start()

    def _analysis_worker(self, db_path: str, table_name: str, params: AnalysisParams) -> None:
        try:
            data = load_monitor_db(db_path, table_name=table_name)
            result = analyze_monitor_data(data, params)
            self._queue.put(("ok", (data, result, params.normalized())))
        except Exception:
            self._queue.put(("err", traceback.format_exc()))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                self._worker_alive = False
                self.analyze_btn.configure(state=tk.NORMAL)
                if kind == "ok":
                    data, result, params = payload
                    self._last_data = data
                    self._last_result = result
                    self._last_params = params
                    self._update_summary(data, result)
                    self._update_plots(data, result)
                    self.status_var.set(
                        f"分析完成: {os.path.basename(data.db_path)} | rows={data.row_count} | fs={result.fs_hz:.3f} Hz"
                    )
                else:
                    self.status_var.set("分析失败")
                    messagebox.showerror("分析失败", str(payload))
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _update_summary(self, data: MonitorData, result: AnalysisResult) -> None:
        payload = summary_dict(data, result)
        lines = _summary_lines(payload)
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.configure(state=tk.DISABLED)

    def _update_plots(self, data: MonitorData, result: AnalysisResult) -> None:
        labels = _plot_labels("zh")
        max_points = (self._last_params or self._get_params().normalized()).max_plot_points
        _plot_tension_axis(self.ax_tension, data, result, max_points, labels)
        _plot_mu_axis(self.ax_mu, data, result, max_points, labels)
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.18, hspace=0.92)
        self.canvas.draw_idle()
        self.root.after_idle(self._on_body_configure)

    def _export_plots(self) -> None:
        if self._last_data is None or self._last_result is None or self._last_params is None:
            messagebox.showwarning("提示", "请先完成分析，再导出图片。")
            return

        out_dir = filedialog.askdirectory(title="选择图片导出目录")
        if not out_dir:
            return

        lang = "en" if self.export_lang_var.get().strip().lower().startswith("english") else "zh"
        try:
            outputs = export_monitor_plots(self._last_data, self._last_result, self._last_params, out_dir, lang=lang)
        except Exception:
            messagebox.showerror("导出失败", traceback.format_exc())
            return

        self.status_var.set(f"图片已导出: {os.path.basename(out_dir)} | lang={lang}")
        messagebox.showinfo("导出完成", "\n".join(outputs.values()))

    def _clear_result(self) -> None:
        self._last_data = None
        self._last_result = None
        self._last_params = None
        self.status_var.set("结果已清空。")
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.configure(state=tk.DISABLED)
        self._draw_placeholder()


def run_headless(db_path: str, table_name: str, params: AnalysisParams) -> Dict[str, object]:
    data = load_monitor_db(db_path, table_name=table_name)
    result = analyze_monitor_data(data, params)
    return summary_dict(data, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="针钩监测数据 GUI 绘图与寿命判据分析")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="sqlite 数据库路径")
    parser.add_argument("--table", default="Data", help="数据表名")
    parser.add_argument("--headless", action="store_true", help="仅分析并输出 JSON，不启动 GUI")
    args = parser.parse_args()

    if args.headless:
        payload = run_headless(args.db, args.table, AnalysisParams())
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    root = tk.Tk()
    app = MonitorAnalyzerApp(root)
    if args.db:
        app.db_path_var.set(args.db)
    if args.table:
        app.table_name_var.set(args.table)
    root.mainloop()


if __name__ == "__main__":
    main()
