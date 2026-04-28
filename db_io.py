import os
import shutil
import sqlite3
from typing import Dict, List, Optional, Sequence

import numpy as np

from models import MonitorData, OptimizedData, OptimizationExportResult, OptimizationParams


DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "needle_hook_wear.db")


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def find_column(actual_columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lookup = {name.lower(): name for name in actual_columns}
    for alias in aliases:
        hit = lookup.get(alias.lower())
        if hit is not None:
            return hit
    return None


def resolve_data_columns(conn: sqlite3.Connection, table_name: str) -> Dict[str, Optional[str]]:
    pragma_rows = conn.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()
    if not pragma_rows:
        raise ValueError(f'表 "{table_name}" 不存在或不可读')
    actual_columns = [str(row[1]) for row in pragma_rows]
    return {
        "time": find_column(actual_columns, ["t_s", "time", "ts"]),
        "high": find_column(actual_columns, ["t_high_n", "t_high", "high", "high_n"]),
        "low": find_column(actual_columns, ["t_low_n", "t_low", "low", "low_n"]),
        "avg": find_column(actual_columns, ["t_avg_n", "t_avg", "avg", "avg_n"]),
        "fric": find_column(actual_columns, ["f_fric_n", "f_fric", "fric", "friction"]),
        "mu": find_column(actual_columns, ["mu", "mu_filt", "mu_raw"]),
        "qf": find_column(actual_columns, ["quality_flag", "q_valid", "质量标志"]),
    }


def load_monitor_db(db_path: str, table_name: str = "Data") -> MonitorData:
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cols = resolve_data_columns(conn, table_name)
        missing = [
            name
            for name, col in [
                ("t_s", cols["time"]),
                ("t_high_N", cols["high"]),
                ("t_low_N", cols["low"]),
                ("mu", cols["mu"]),
                ("quality_flag", cols["qf"]),
            ]
            if col is None
        ]
        if missing:
            raise ValueError("缺少必要列: " + ", ".join(missing))

        ordered = [cols["time"], cols["high"], cols["low"]]
        optional = [cols["avg"], cols["fric"], cols["mu"], cols["qf"]]
        select_cols = ordered + [col for col in optional if col is not None]
        sql = (
            "SELECT "
            + ", ".join(quote_ident(c) for c in select_cols)
            + f" FROM {quote_ident(table_name)} ORDER BY rowid"
        )

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
                t_s.append(float(row[idx_map[cols["time"]]]))
                t_high.append(float(row[idx_map[cols["high"]]]))
                t_low.append(float(row[idx_map[cols["low"]]]))
                if cols["avg"] is not None:
                    v = row[idx_map[cols["avg"]]]
                    t_avg.append(None if v is None else float(v))
                if cols["fric"] is not None:
                    v = row[idx_map[cols["fric"]]]
                    f_fric.append(None if v is None else float(v))
                if cols["mu"] is not None:
                    v = row[idx_map[cols["mu"]]]
                    mu_vals.append(None if v is None else float(v))
                if cols["qf"] is not None:
                    v = row[idx_map[cols["qf"]]]
                    qf_vals.append(int(v) if v is not None else 0)

        t_s_arr = np.asarray(t_s, dtype=float)
        t_high_arr = np.asarray(t_high, dtype=float)
        t_low_arr = np.asarray(t_low, dtype=float)

        if cols["avg"] is not None:
            t_avg_arr = np.asarray([np.nan if v is None else v for v in t_avg], dtype=float)
        else:
            t_avg_arr = (t_high_arr + t_low_arr) / 2.0

        if cols["fric"] is not None:
            f_fric_arr = np.asarray([np.nan if v is None else v for v in f_fric], dtype=float)
        else:
            f_fric_arr = t_high_arr - t_low_arr

        mu_arr = np.asarray([np.nan if v is None else v for v in mu_vals], dtype=float)
        qf_arr = np.asarray(qf_vals, dtype=int)

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


def export_optimized_db(
    source_db_path: str,
    output_db_path: str,
    table_name: str,
    optimized: OptimizedData,
    params: OptimizationParams,
) -> OptimizationExportResult:
    source_abs = os.path.abspath(source_db_path)
    output_abs = os.path.abspath(output_db_path)
    if source_abs == output_abs:
        raise ValueError("导出路径不能覆盖原数据库，请另存为新的 .db 文件。")
    if not os.path.exists(source_abs):
        raise FileNotFoundError(source_abs)

    out_dir = os.path.dirname(output_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(source_abs, output_abs)

    conn = sqlite3.connect(output_abs)
    try:
        cols = resolve_data_columns(conn, table_name)
        missing = [
            name
            for name, col in [
                ("t_high_N", cols["high"]),
                ("t_low_N", cols["low"]),
                ("t_avg_N", cols["avg"]),
                ("f_fric_N", cols["fric"]),
                ("mu", cols["mu"]),
            ]
            if col is None
        ]
        if missing:
            raise ValueError("导出数据库缺少待更新列: " + ", ".join(missing))

        rowids = [row[0] for row in conn.execute(f"SELECT rowid FROM {quote_ident(table_name)} ORDER BY rowid")]
        if len(rowids) != len(optimized.mu):
            raise ValueError(f"优化结果行数 {len(optimized.mu)} 与数据库行数 {len(rowids)} 不一致")

        update_sql = (
            f"UPDATE {quote_ident(table_name)} SET "
            f"{quote_ident(cols['high'])}=?, "
            f"{quote_ident(cols['low'])}=?, "
            f"{quote_ident(cols['avg'])}=?, "
            f"{quote_ident(cols['fric'])}=?, "
            f"{quote_ident(cols['mu'])}=? "
            "WHERE rowid=?"
        )
        payload = zip(
            map(float, optimized.t_high_N),
            map(float, optimized.t_low_N),
            map(float, optimized.t_avg_N),
            map(float, optimized.f_fric_N),
            [None if not np.isfinite(v) else float(v) for v in optimized.mu],
            rowids,
        )
        conn.executemany(update_sql, payload)
        conn.commit()
    except Exception:
        conn.close()
        raise
    else:
        conn.close()

    return OptimizationExportResult(
        output_path=output_abs,
        row_count=int(len(optimized.mu)),
        wrap_angle_rad=float(params.wrap_angle_rad),
        high_spike_count=int(optimized.high_spike_count),
        low_spike_count=int(optimized.low_spike_count),
    )
