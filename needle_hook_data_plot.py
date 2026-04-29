import argparse
import json
import tkinter as tk

from app import MonitorAnalyzerApp, run_headless
from db_io import DEFAULT_DB_PATH
from models import AnalysisParams


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
        app.opt_db_path_var.set(args.db)
        app.wrap_db_path_var.set(args.db)
    if args.table:
        app.table_name_var.set(args.table)
        app.opt_table_name_var.set(args.table)
        app.wrap_table_name_var.set(args.table)
    root.mainloop()


if __name__ == "__main__":
    main()
