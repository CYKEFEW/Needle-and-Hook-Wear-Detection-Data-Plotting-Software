from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from models import AnalysisParams, AnalysisResult, MonitorData


def infer_fs_hz(t_s: np.ndarray) -> float:
    if len(t_s) < 2:
        return 1.0
    dt = np.diff(np.asarray(t_s, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 1.0
    return float(1.0 / np.median(dt))


def resolve_fs_hz(t_s: np.ndarray, params: Optional[object] = None) -> float:
    if params is not None:
        try:
            fs_cfg = float(getattr(params, "sample_rate_hz", 0.0) or 0.0)
        except Exception:
            fs_cfg = 0.0
        if fs_cfg > 0.0:
            return fs_cfg
    return infer_fs_hz(t_s)


def interpolate_invalid_samples(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    if len(arr) == 0 or bool(valid.all()):
        return arr
    if not bool(valid.any()):
        return arr
    x = np.arange(len(arr))
    out = arr.copy()
    out[~valid] = np.interp(x[~valid], x[valid], arr[valid])
    return out


def _find_invalid_run_over_limit(
    valid_flag: np.ndarray,
    t_s: np.ndarray,
    fs_hz: float,
    max_break_s: float,
) -> Optional[Tuple[int, int, float]]:
    if max_break_s <= 0.0 or len(valid_flag) <= 0:
        return None

    run_start: Optional[int] = None
    valid = np.asarray(valid_flag, dtype=bool)
    for i, flag in enumerate(valid):
        if flag:
            if run_start is not None:
                dur_s = float(i - run_start) / max(1e-9, fs_hz)
                if dur_s > max_break_s + 1e-9:
                    return run_start, i - 1, dur_s
                run_start = None
            continue
        if run_start is None:
            run_start = i

    if run_start is None:
        return None

    dur_s = float(len(valid) - run_start) / max(1e-9, fs_hz)
    if dur_s > max_break_s + 1e-9:
        return run_start, len(valid) - 1, dur_s
    return None


def clip_stable_segments(
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

    fs_hz = resolve_fs_hz(t, params)
    win = int(round(params.stable_win_s * fs_hz))
    win = max(win, 50)
    step = max(1, win // 10)
    stable_mask = np.zeros(n_all, dtype=bool)

    for k0 in range(0, max(1, n - win), step):
        k1 = k0 + win
        if k1 > n:
            break
        q_window = np.asarray(q_valid[k0:k1], dtype=bool)
        if float(np.mean(q_window)) < params.stable_valid_min:
            continue

        yy = mu_f[k0:k1]
        valid = np.isfinite(yy) & q_window
        if int(valid.sum()) < 10:
            continue

        vals = yy[valid]
        if float(np.std(vals)) > params.stable_sigma_max:
            continue

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
    fs_hz = resolve_fs_hz(t, params)
    dt = 1.0 / max(1e-9, fs_hz)
    hold_s = float(max(0.0, params.fail_hold_s))
    break_s = float(max(0.0, params.fail_break_s))
    start_idx = int(max(0, min(len(mu_f) - 1, start_idx))) if len(mu_f) else 0

    run_start: Optional[int] = None
    break_count = 0
    for i in range(start_idx, len(mu_f)):
        if not bool(q_valid[i]):
            if run_start is not None:
                break_count += 1
                if (float(break_count) / max(1e-9, fs_hz)) > break_s + 1e-9:
                    run_start = None
                    break_count = 0
            continue
        above = bool(np.isfinite(mu_f[i]) and (mu_f[i] > mu_th))
        if above:
            if run_start is None:
                run_start = i
            break_count = 0
            if float(t[i] - t[run_start] + dt) >= hold_s:
                return float(t[run_start]), int(run_start), mu_th
        else:
            run_start = None
            break_count = 0
    return None, None, mu_th


def analyze_monitor_data(data: MonitorData, params: AnalysisParams) -> AnalysisResult:
    params = params.normalized()
    fs_hz = resolve_fs_hz(data.t_s, params)

    q_from_db = np.asarray(data.quality_flag, dtype=int) != 0
    invalid_run = _find_invalid_run_over_limit(q_from_db, data.t_s, fs_hz, params.fail_break_s)
    if invalid_run is not None:
        k0, k1, dur_s = invalid_run
        t0 = float(data.t_s[k0])
        t1 = float(data.t_s[k1])
        raise ValueError(
            "连续 q=0 数据超过最大容忍中断时间 "
            f"{params.fail_break_s:.3f}s："
            f"[{t0:.3f}s, {t1:.3f}s] 持续 {dur_s:.3f}s。已停止绘图。"
        )

    mu_eval = np.asarray(data.mu, dtype=float).copy()
    q_valid = q_from_db & np.isfinite(mu_eval)

    stable_idx0, mu_ss0, stable_segs0 = _find_stable_baseline(
        mu_eval, data.t_s, q_valid.astype(int), params, end_idx=None
    )
    start_fail_idx0 = int(stable_idx0[1]) if stable_idx0 is not None else 0
    tlife0_s, tlife0_idx, _ = _find_failure_time(
        mu_eval, data.t_s, q_valid.astype(int), mu_ss0, params, start_idx=start_fail_idx0
    )

    if tlife0_idx is not None:
        stable_idx, mu_ss, stable_segs = _find_stable_baseline(
            mu_eval, data.t_s, q_valid.astype(int), params, end_idx=tlife0_idx + 1
        )
        if mu_ss is None or (not np.isfinite(mu_ss)):
            stable_idx, mu_ss, stable_segs = stable_idx0, mu_ss0, stable_segs0
    else:
        stable_idx, mu_ss, stable_segs = stable_idx0, mu_ss0, stable_segs0

    start_fail_idx = int(stable_idx[1]) if stable_idx is not None else 0
    tlife_s, tlife_idx, mu_th = _find_failure_time(
        mu_eval, data.t_s, q_valid.astype(int), mu_ss, params, start_idx=start_fail_idx
    )

    display_t_high = interpolate_invalid_samples(data.t_high_N, q_from_db)
    display_t_low = interpolate_invalid_samples(data.t_low_N, q_from_db)

    return AnalysisResult(
        fs_hz=fs_hz,
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
        display_t_avg=(display_t_high + display_t_low) / 2.0,
        display_mu_eval=interpolate_invalid_samples(mu_eval, q_from_db),
    )


def summary_dict(data: MonitorData, result: AnalysisResult) -> Dict[str, object]:
    return {
        "db_path": data.db_path,
        "mu_ss": result.mu_ss,
        "mu_th": result.mu_th,
        "tlife_s": result.tlife_s,
    }


def summary_value_text(value: Optional[float], digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "None"
    return f"{float(value):.{digits}f}"


def summary_lines(payload: Dict[str, object]) -> List[str]:
    return [
        f"数据库路径: {payload['db_path']}",
        f"μss: {summary_value_text(payload['mu_ss'])}",
        f"μth: {summary_value_text(payload['mu_th'])}",
        f"tlife: {summary_value_text(payload['tlife_s'], 3)}",
    ]
