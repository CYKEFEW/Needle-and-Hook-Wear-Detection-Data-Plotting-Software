# 针钩磨损监测数据绘图

这是一个基于 Tkinter + Matplotlib 的针钩磨损监测数据分析工具，用于读取 SQLite 数据库、绘制张力/摩擦系数曲线、计算稳定段和寿命指标，并支持数据优化、包角重算和导出新的 db 数据库。

## 功能

- 读取 SQLite 监测数据库，默认表名为 `Data`。
- 绘制 `T_high`、`T_low`、`T_avg` 张力曲线。
- 绘制 `μ (filtered)` 曲线，并显示 `Stable segments`、`Baseline μss`、`Threshold μth`、`tlife`。
- 绘图区预览支持切换显示：`张力`、`高张力`、`低张力`、`平均张力`、`μ`。
- 绘图区预览语言默认中文，可切换英文；该设置不影响导出图片语言。
- Y 轴上下限支持输入数字或 `Auto` 自适应。
- `张力`、`高张力`、`低张力`、`μ` 支持独立 Y 轴范围控制；`平均张力` 沿用 `张力` 范围。
- `μ`、稳定段、基线、阈值、`tlife` 支持独立开关显示。
- `q=0` 数据点不参与稳定段和 `tlife` 计算；绘图显示时用线性插值替换。
- 连续 `q=0` 超过 `最大容忍中断 Wbreak` 时停止绘图，避免使用不可信数据。
- 图片导出可选择内容：`张力`、`高张力`、`低张力`、`平均张力`、`摩擦系数`；默认导出 `张力` 和 `摩擦系数`。
- 数据优化区支持去毛刺、平滑、高/低张力优化前后预览、平均张力预览、μ 预览。
- 去毛刺和平滑均有启用开关和力度滑动条，默认力度为 `50%`。
- 去毛刺会先对明显离群点做线性插值替换，再执行 Hampel 滤波。
- 优化后按 capstan 公式重算：
  - `T_avg = (T_high + T_low) / 2`
  - `F_fric = T_high - T_low`
  - `μ = ln(T_high / T_low) / θ`
- 导出优化后的 db 时会弹出保存路径窗口，默认位于源数据库目录，且不会覆盖原库。
- 包角重算区支持自动反推当前包角、输入新包角、预览新旧 `μ-时间` 曲线，并导出只更新 `Data.mu` 的新数据库。
- 包角重算不按 `quality_flag` 过滤，不预先过滤高/低张力；无法计算的 `μ` 会写入 SQLite `NULL`。

## 数据优化算法

数据优化区只直接优化 `t_high_N` 和 `t_low_N` 两列。优化完成后，再根据优化后的高/低张力重算 `t_avg_N`、`f_fric_N` 和 `mu`。

### 1. 自动反推包角

自动反推包角使用当前数据库中的高/低张力和原始 `mu`：

```text
theta_i = ln(t_high_N / t_low_N) / mu
```

参与反推的行需要满足：

- `quality_flag != 0`
- `t_high_N` 有限且大于 0
- `t_low_N` 有限且大于 0
- `mu` 有限且不接近 0

最终包角取所有有效 `theta_i` 的中位数：

```text
theta_auto = median(theta_i)
```

### 2. 自动去毛刺

去毛刺受“自动去除毛刺”复选框控制，未启用时跳过。算法分别处理 `t_high_N` 和 `t_low_N`。

处理顺序：

1. 明显离群点线性插值：根据局部窗口中除当前点外的邻域中位数识别突刺，超过阈值的点用周围有效点线性插值替换。`quality_flag == 0` 的点也会在这一步作为无效点插值替换。
2. Hampel 滤波：对局部窗口计算中位数和 MAD，若当前点偏离超过阈值则替换为局部中位数。
3. 分位数离群点修正：按整体分布上下分位边界识别极端值，并用线性插值替换。

Hampel 阈值为：

```text
threshold = sigma * 1.4826 * MAD
```

“去毛刺力度”滑动条会影响 Hampel 阈值和分位数边界。力度越高，算法越激进，更容易替换尖峰。

### 3. 自动平滑

平滑受“自动平滑曲线”复选框控制，未启用时跳过。

平滑对去毛刺后的 `t_high_N` 和 `t_low_N` 分别执行居中移动平均：

```text
T_smooth[i] = mean(window around i)
```

窗口大小由基础平滑窗口、采样率和“平滑力度”共同决定。采样率可自动推断，也可手动输入。力度越高，实际窗口越大，曲线越平滑。

边界使用端点延拓，保证输出长度与原始数据一致。如果同时启用了去毛刺和平滑，平滑后会再执行一轮较轻的离群点修正，减少残留尖峰。

### 4. 重算派生列

优化完成后，根据优化后的高/低张力重算：

```text
t_avg_N = (t_high_N + t_low_N) / 2
f_fric_N = t_high_N - t_low_N
mu = ln(t_high_N / t_low_N) / theta
```

其中 `theta` 是用户输入或自动反推得到的包角。无法计算的 `mu` 会在导出数据库时写入 SQLite `NULL`。

### 5. 导出优化数据库

导出优化后的数据库时不会覆盖原数据库。程序会先复制原数据库到新路径，然后只更新 `Data` 表中的：

- `t_high_N`
- `t_low_N`
- `t_avg_N`
- `f_fric_N`
- `mu`

保持不变：

- `t_s`
- `quality_flag`
- 其他非优化列

导出后数据库行数不变，表结构保持兼容。

## 文件结构

- `needle_hook_data_plot.py`：程序入口，负责命令行参数和启动 GUI。
- `app.py`：Tkinter 主界面、绘图区、数据优化区、包角重算区和交互逻辑。
- `models.py`：数据模型和参数对象。
- `db_io.py`：SQLite 数据读取、优化数据库导出和包角重算数据库导出。
- `analysis.py`：稳定段、基线、阈值和 `tlife` 计算。
- `plotting.py`：Matplotlib 绘图和图片导出。
- `optimization.py`：包角反推、去毛刺、平滑、优化数据生成和包角重算。
- `database/test`：测试数据库样例。

## 运行

```bash
python needle_hook_data_plot.py
```

指定数据库和表名：

```bash
python needle_hook_data_plot.py --db "database/test/30min data_log_20260428_162624.db" --table Data
```

无界面分析：

```bash
python needle_hook_data_plot.py --headless --db "database/test/30min data_log_20260428_162624.db" --table Data
```

## 打包

使用 PyInstaller：

```bash
pyinstaller main.spec
```

`main.spec` 会显式包含本地模块、Tkinter/Matplotlib 后端依赖，并把 `database` 目录和 `README.md` 作为数据文件打包。

## 数据库字段要求

`Data` 表至少需要以下列：

- `t_s`
- `t_high_N`
- `t_low_N`
- `mu`
- `quality_flag`

可选列：

- `t_avg_N`
- `f_fric_N`

如果可选列缺失，程序会在读取或优化时根据高/低张力计算。
