from typing import Optional

import numpy as np

from analysis import resolve_fs_hz
from models import MonitorData, OptimizedData, OptimizationParams


def infer_wrap_angle_rad(data: MonitorData) -> float:
    high = np.asarray(data.t_high_N, dtype=float)
    low = np.asarray(data.t_low_N, dtype=float)
    mu = np.asarray(data.mu, dtype=float)
    quality = np.asarray(data.quality_flag, dtype=int) != 0
    mask = quality & np.isfinite(high) & np.isfinite(low) & np.isfinite(mu)
    mask &= (high > 0.0) & (low > 0.0) & (np.abs(mu) > 1e-12)
    theta = np.full_like(mu, np.nan, dtype=float)
    theta[mask] = np.log(high[mask] / low[mask]) / mu[mask]
    theta = theta[np.isfinite(theta) & (theta > 0.0)]
    if theta.size == 0:
        raise ValueError("无法从数据库有效行反推包角，请手动输入包角。")
    return float(np.nanmedian(theta))


def _window_samples(window_s: float, fs_hz: float, minimum: int = 1, odd: bool = True) -> int:
    n = int(round(max(0.0, float(window_s)) * max(1e-9, float(fs_hz))))
    n = max(minimum, n)
    if odd and n % 2 == 0:
        n += 1
    return n


def _fill_nan_linear(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    x = np.arange(len(arr))
    arr[~finite] = np.interp(x[~finite], x[finite], arr[finite])
    return arr


def hampel_filter(values: np.ndarray, window_samples: int, n_sigma: float) -> tuple[np.ndarray, int]:
    arr = _fill_nan_linear(values)
    out = arr.copy()
    half = max(1, int(window_samples) // 2)
    replaced = 0
    scale = 1.4826
    for i in range(len(arr)):
        start = max(0, i - half)
        end = min(len(arr), i + half + 1)
        window = arr[start:end]
        med = float(np.median(window))
        mad = float(np.median(np.abs(window - med)))
        threshold = float(n_sigma) * scale * mad
        if threshold <= 1e-12:
            continue
        if abs(arr[i] - med) > threshold:
            out[i] = med
            replaced += 1
    return out, replaced


def moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    arr = _fill_nan_linear(values)
    n = max(1, int(window_samples))
    if n <= 1 or len(arr) == 0:
        return arr.copy()
    pad_left = n // 2
    pad_right = n - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    kernel = np.ones(n, dtype=float) / float(n)
    return np.convolve(padded, kernel, mode="valid")


def optimize_monitor_data(data: MonitorData, params: OptimizationParams) -> OptimizedData:
    params = params.normalized()
    fs_hz = resolve_fs_hz(data.t_s, params)
    high = np.asarray(data.t_high_N, dtype=float).copy()
    low = np.asarray(data.t_low_N, dtype=float).copy()
    high_spikes = 0
    low_spikes = 0

    if params.enable_hampel and params.hampel_window_s > 0.0:
        win = _window_samples(params.hampel_window_s, fs_hz, minimum=3, odd=True)
        high, high_spikes = hampel_filter(high, win, params.hampel_sigma)
        low, low_spikes = hampel_filter(low, win, params.hampel_sigma)

    if params.enable_smoothing and params.smooth_window_s > 0.0:
        win = _window_samples(params.smooth_window_s, fs_hz, minimum=1, odd=True)
        high = moving_average(high, win)
        low = moving_average(low, win)

    t_avg = (high + low) / 2.0
    f_fric = high - low
    mu = np.full_like(high, np.nan, dtype=float)
    valid = np.isfinite(high) & np.isfinite(low) & (high > 0.0) & (low > 0.0)
    mu[valid] = np.log(high[valid] / low[valid]) / float(params.wrap_angle_rad)

    return OptimizedData(
        t_high_N=high,
        t_low_N=low,
        t_avg_N=t_avg,
        f_fric_N=f_fric,
        mu=mu,
        high_spike_count=int(high_spikes),
        low_spike_count=int(low_spikes),
    )


def optimized_to_monitor_data(source: MonitorData, optimized: OptimizedData, db_path: Optional[str] = None) -> MonitorData:
    return MonitorData(
        db_path=source.db_path if db_path is None else db_path,
        table_name=source.table_name,
        row_count=source.row_count,
        t_s=np.asarray(source.t_s, dtype=float).copy(),
        t_high_N=np.asarray(optimized.t_high_N, dtype=float).copy(),
        t_low_N=np.asarray(optimized.t_low_N, dtype=float).copy(),
        t_avg_N=np.asarray(optimized.t_avg_N, dtype=float).copy(),
        f_fric_N=np.asarray(optimized.f_fric_N, dtype=float).copy(),
        mu=np.asarray(optimized.mu, dtype=float).copy(),
        quality_flag=np.asarray(source.quality_flag, dtype=int).copy(),
    )
