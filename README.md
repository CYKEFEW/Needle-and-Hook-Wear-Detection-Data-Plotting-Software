# Draw

`needle_hook_monitor_gui.py` 是一个独立的监测数据 GUI 分析程序。

功能：

- 读取 sqlite 监测库，默认参考 `data/needle_hook_wear_sim.db`
- 绘制张力-时间图
- 绘制 μ-时间图
- 按“稳态窗口 + 有效比例 + 波动度 + 漂移量 + 持续超阈值”逻辑计算 `μss / μth / tlife`

运行：

```bash
python Draw\needle_hook_monitor_gui.py
```

无界面分析：

```bash
python Draw\needle_hook_monitor_gui.py --headless --db Draw\data\needle_hook_wear_sim.db --table Data
```
