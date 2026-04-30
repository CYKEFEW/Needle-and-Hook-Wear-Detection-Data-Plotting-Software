from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AnalysisParams:
    stable_win_s: float = 1200.0
    stable_hold_s: float = 3600.0
    stable_sigma_max: float = 0.03
    stable_slope_max: float = 0.015
    stable_valid_min: float = 0.9
    fail_delta: float = 0.25
    fail_hold_s: float = 300.0
    fail_break_s: float = 1.0
    sample_rate_hz: float = 0.0
    max_plot_points: int = 40000

    def normalized(self) -> "AnalysisParams":
        slope = float(self.stable_slope_max)
        if slope < 1e-3:
            slope *= max(1.0, float(self.stable_win_s))
        return AnalysisParams(
            stable_win_s=max(1.0, float(self.stable_win_s)),
            stable_hold_s=max(0.0, float(self.stable_hold_s)),
            stable_sigma_max=max(0.0, float(self.stable_sigma_max)),
            stable_slope_max=max(0.0, slope),
            stable_valid_min=min(1.0, max(0.0, float(self.stable_valid_min))),
            fail_delta=max(0.0, float(self.fail_delta)),
            fail_hold_s=max(0.0, float(self.fail_hold_s)),
            fail_break_s=max(0.0, float(self.fail_break_s)),
            sample_rate_hz=max(0.0, float(self.sample_rate_hz)),
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
    mu: np.ndarray
    quality_flag: np.ndarray


@dataclass
class AnalysisResult:
    fs_hz: float
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
    display_mu_eval: np.ndarray


@dataclass
class AxisRange:
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None


@dataclass
class PlotOptions:
    axis_ranges: Dict[str, AxisRange] = field(default_factory=dict)
    tension_y_min: Optional[float] = None
    tension_y_max: Optional[float] = None
    high_tension_y_min: Optional[float] = None
    high_tension_y_max: Optional[float] = None
    low_tension_y_min: Optional[float] = None
    low_tension_y_max: Optional[float] = None
    mu_y_min: Optional[float] = None
    mu_y_max: Optional[float] = None
    show_mu: bool = True
    show_stable_segments: bool = True
    show_mu_ss: bool = True
    show_tlife: bool = True
    show_mu_th: bool = True


@dataclass
class OptimizationParams:
    wrap_angle_rad: float
    enable_hampel: bool = True
    enable_smoothing: bool = True
    hampel_window_s: float = 1.0
    hampel_sigma: float = 3.0
    smooth_window_s: float = 0.5
    sample_rate_hz: float = 0.0
    hampel_strength_pct: float = 50.0
    smooth_strength_pct: float = 50.0
    quantile_max_run_s: float = 10.0
    quantile_segment_s: float = 600.0

    def normalized(self) -> "OptimizationParams":
        return OptimizationParams(
            wrap_angle_rad=max(1e-12, float(self.wrap_angle_rad)),
            enable_hampel=bool(self.enable_hampel),
            enable_smoothing=bool(self.enable_smoothing),
            hampel_window_s=max(0.0, float(self.hampel_window_s)),
            hampel_sigma=max(0.1, float(self.hampel_sigma)),
            smooth_window_s=max(0.0, float(self.smooth_window_s)),
            sample_rate_hz=max(0.0, float(self.sample_rate_hz)),
            hampel_strength_pct=min(100.0, max(0.0, float(self.hampel_strength_pct))),
            smooth_strength_pct=min(100.0, max(0.0, float(self.smooth_strength_pct))),
            quantile_max_run_s=max(0.0, float(self.quantile_max_run_s)),
            quantile_segment_s=max(1.0, float(self.quantile_segment_s)),
        )


@dataclass
class OptimizedData:
    t_high_N: np.ndarray
    t_low_N: np.ndarray
    t_avg_N: np.ndarray
    f_fric_N: np.ndarray
    mu: np.ndarray
    high_spike_count: int
    low_spike_count: int


@dataclass
class OptimizationExportResult:
    output_path: str
    row_count: int
    wrap_angle_rad: float
    high_spike_count: int
    low_spike_count: int


@dataclass
class WrapAngleExportResult:
    output_path: str
    row_count: int
    wrap_angle_rad: float
    nan_count: int
