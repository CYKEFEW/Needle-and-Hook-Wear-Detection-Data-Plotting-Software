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


def _replace_mask_linear(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    arr = np.asarray(values, dtype=float).copy()
    mask = np.asarray(mask, dtype=bool)
    if len(arr) == 0 or not bool(mask.any()):
        return arr, 0

    valid = np.isfinite(arr) & (~mask)
    if not bool(valid.any()):
        return arr, 0

    x = np.arange(len(arr))
    out = arr.copy()
    out[mask] = np.interp(x[mask], x[valid], arr[valid])
    return out, int(mask.sum())


def remove_extreme_outliers_linear(
    values: np.ndarray,
    window_samples: int,
    n_sigma: float,
    force_mask: Optional[np.ndarray] = None,
    iterations: int = 3,
) -> tuple[np.ndarray, int]:
    arr = _fill_nan_linear(values)
    if len(arr) < 5:
        return arr, 0

    outlier = np.zeros(len(arr), dtype=bool)
    if force_mask is not None:
        outlier |= np.asarray(force_mask, dtype=bool)

    half = max(3, int(window_samples) // 2)
    sigma = max(2.5, float(n_sigma))
    scale = 1.4826

    for _ in range(max(1, int(iterations))):
        work, _ = _replace_mask_linear(arr, outlier)
        residuals: list[float] = []
        local_median = np.zeros(len(arr), dtype=float)

        for i in range(len(arr)):
            start = max(0, i - half)
            end = min(len(arr), i + half + 1)
            window = np.concatenate((work[start:i], work[i + 1 : end]))
            window = window[np.isfinite(window)]
            if len(window) < 4:
                local_median[i] = work[i]
                continue
            med = float(np.median(window))
            local_median[i] = med
            residuals.append(abs(float(work[i]) - med))

        residual_arr = np.asarray(residuals, dtype=float)
        residual_arr = residual_arr[np.isfinite(residual_arr)]
        if residual_arr.size == 0:
            break
        residual_median = float(np.median(residual_arr))
        residual_mad = float(np.median(np.abs(residual_arr - residual_median)))
        global_scale = max(scale * residual_mad, 1e-9)
        threshold = max(0.12, sigma * global_scale)
        new_outlier = np.abs(work - local_median) > threshold
        before = int(outlier.sum())
        outlier |= new_outlier
        if int(outlier.sum()) == before:
            break

    return _replace_mask_linear(arr, outlier)


def remove_quantile_outliers_linear(values: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> tuple[np.ndarray, int]:
    arr = _fill_nan_linear(values)
    finite = arr[np.isfinite(arr)]
    if finite.size < 20:
        return arr, 0

    low = float(np.percentile(finite, low_pct))
    high = float(np.percentile(finite, high_pct))
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_sigma = max(1.4826 * mad, 1e-9)
    low = max(low, median - 4.0 * robust_sigma)
    high = min(high, median + 4.0 * robust_sigma)
    if high <= low:
        return arr, 0
    return _replace_mask_linear(arr, (arr < low) | (arr > high))


def _strength_fraction(strength_pct: float) -> float:
    return min(1.0, max(0.0, float(strength_pct) / 100.0))


def _despike_sigma(base_sigma: float, strength_pct: float) -> float:
    strength = _strength_fraction(strength_pct)
    multiplier = 1.75 - (1.50 * strength)
    return max(0.5, float(base_sigma) * multiplier)


def _despike_quantile_edge(strength_pct: float, base_edge: float) -> float:
    strength = _strength_fraction(strength_pct)
    return max(0.01, float(base_edge) * (0.10 + 1.80 * strength))


def _smooth_window_samples(base_window_s: float, fs_hz: float, strength_pct: float) -> int:
    strength = _strength_fraction(strength_pct)
    multiplier = 0.25 + (1.50 * strength)
    return _window_samples(float(base_window_s) * multiplier, fs_hz, minimum=1, odd=True)


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
    invalid_quality = np.asarray(data.quality_flag, dtype=int) == 0
    high_spikes = 0
    low_spikes = 0

    if params.enable_hampel and params.hampel_window_s > 0.0:
        win = _window_samples(params.hampel_window_s, fs_hz, minimum=3, odd=True)
        sigma = _despike_sigma(params.hampel_sigma, params.hampel_strength_pct)
        quantile_edge = _despike_quantile_edge(params.hampel_strength_pct, 0.5)
        high, high_extreme_spikes = remove_extreme_outliers_linear(high, win, sigma, force_mask=invalid_quality)
        low, low_extreme_spikes = remove_extreme_outliers_linear(low, win, sigma, force_mask=invalid_quality)
        high, high_hampel_spikes = hampel_filter(high, win, sigma)
        low, low_hampel_spikes = hampel_filter(low, win, sigma)
        high_spikes = high_extreme_spikes + high_hampel_spikes
        low_spikes = low_extreme_spikes + low_hampel_spikes
        high, high_quantile_spikes = remove_quantile_outliers_linear(high, quantile_edge, 100.0 - quantile_edge)
        low, low_quantile_spikes = remove_quantile_outliers_linear(low, quantile_edge, 100.0 - quantile_edge)
        high_spikes += high_quantile_spikes
        low_spikes += low_quantile_spikes

    if params.enable_smoothing and params.smooth_window_s > 0.0:
        win = _smooth_window_samples(params.smooth_window_s, fs_hz, params.smooth_strength_pct)
        high = moving_average(high, win)
        low = moving_average(low, win)
        if params.enable_hampel and params.hampel_window_s > 0.0:
            sigma = _despike_sigma(params.hampel_sigma, params.hampel_strength_pct)
            high, high_post_spikes = remove_extreme_outliers_linear(high, win, sigma, iterations=2)
            low, low_post_spikes = remove_extreme_outliers_linear(low, win, sigma, iterations=2)
            high_spikes += high_post_spikes
            low_spikes += low_post_spikes
            post_edge = _despike_quantile_edge(params.hampel_strength_pct, 0.2)
            high, high_post_quantile_spikes = remove_quantile_outliers_linear(high, post_edge, 100.0 - post_edge)
            low, low_post_quantile_spikes = remove_quantile_outliers_linear(low, post_edge, 100.0 - post_edge)
            high_spikes += high_post_quantile_spikes
            low_spikes += low_post_quantile_spikes

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
