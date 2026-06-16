# Plux-test

PLUX biosignalsplux 三通道(呼吸 RIP / 心电 ECG / 皮电 EDA)生理信号采集 + 特征工程 + 三分类压力状态 + 5 通道艺术输入。

> **目的**:为艺术项目提供"生理状态 → 5 个 0-100 数值"的完整数据管线。

## 项目结构

```
acquisition/    PLUX 蓝牙设备采集脚本(5 min @ 1000 Hz)
webui/          采集仪表盘 + 实时分类(端口 8000)
analysis/       7 个离线分析脚本(质量对比/压力梯度/分类训练/深度心电/5 状态)
art_viz/        艺术可视化前端(端口 8001)
models/         训练好的随机森林分类器 + 个体化阈值
tracks/         5 通道艺术输入数据(每条件 301 行)
data/           原始 5 min × 3 条件 CSV(jie + ziqi)
output/         分析脚本生成产物(gitignored)
PROJECT_SUMMARY.md   完整流水线 / 公式 / 文献参考
```

## 快速开始

### 1. 环境

```powershell
# Python 3.10(PLUX 不发布 3.11/3.12)
winget install Python.Python.3.10

# venv + 依赖
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### 2. 取得 PLUX Python API

PLUX 的 `.pyd` 二进制不包含在本仓库(版权 + 平台依赖)。从官方下载:

```powershell
git clone https://github.com/pluxbiosignals/python-samples.git
# 把 PLUX-API-Python3/ 整个目录复制到本仓库根目录,或:
$env:PLUX_API_DIR = "C:\path\to\python-samples\PLUX-API-Python3\Win64_310"
```

### 3. 配对蓝牙 + 改设备地址

在 Windows BT 设置里配对你的 biosignalsplux,记录 MAC:

```powershell
$env:PLUX_ADDRESS = "BTH00:07:80:XX:XX:XX"
```

或直接改 `acquisition/acquire_save_plot.py` 第 35 行的 `ADDRESS` 默认值。

### 4. 跑

```powershell
# 采集 5 分钟(实时绘图 + CSV)
.\.venv\Scripts\python acquisition\acquire_save_plot.py

# Web 仪表盘(http://127.0.0.1:8000)
.\.venv\Scripts\python webui\server.py

# 艺术可视化(http://127.0.0.1:8001)
.\.venv\Scripts\python art_viz\server_art.py

# 离线分析(用 data/jie/ 已有的 CSV)
.\.venv\Scripts\python analysis\_jie_report.py
.\.venv\Scripts\python analysis\_jie_classifier.py
.\.venv\Scripts\python analysis\_5tracks.py
```

## 5 个生理状态指标

`analysis/_5tracks.py` 把每秒的 60 秒滑动窗口转成 5 个数字:

| 变量 | 0 → 100 含义 | 主要驱动 |
|---|---|---|
| `fatigue` | 警觉 → 困倦 | RMSSD↑ + HR↓ + 慢呼吸 |
| `focus` | 涣散 → 深度专注 | DFA α1↑ + LF/HF↑ |
| `heart_age` | 心脏年龄(岁) | log(130/SDNN)/0.018 |
| `coherence` | 混乱 → 心肺完美同步 | scipy `coherence` 峰值 |
| `resilience` | 慢恢复 → 快反弹 | CVI(Cardiac Vagal Index) |

输出格式(`tracks/tracks_*.csv`):
```
t_sec, fatigue, focus, heart_age, coherence, resilience
0,     60.6,    32.8,  38.8,      75.7,      52.4
...
```

## 数据集

`data/jie/` 和 `data/ziqi/` 各 3 份 5 分钟 1000 Hz 采集,对应自我报告的 3 种情绪:

| 条件 | 含义 | 文件 |
|---|---|---|
| stable | 平静 | `stable.csv` |
| middle | 心事 / 微紧张 | `中.csv` (jie) / `middle.csv` (ziqi) |
| mess | 焦虑 | `乱.csv` (jie) / `mess.csv` (ziqi) |

每个 CSV 5 列:`nSeq, t_sec, RIP, ECG, EDA`(ADC 整数)。

**质量对比**:jie 数据质量明显优于 ziqi(电极接触好,ECG SNR 高 6 dB,EDA 噪声占比 16% vs 65%),后续深度分析全部基于 jie。

## 参考资料

完整的硬件配置、电极接法、信号处理流水线、特征公式、分类器评估、避坑日志见 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)。

引用资料:
- PLUX EDA Sensor Datasheet (EMG 03092020 REV B, © 2020 PLUX)
- PLUX biosignalsplux ECG User Manual
- [pluxbiosignals/python-samples](https://github.com/pluxbiosignals/python-samples)
- [pluxbiosignals/biosignalsnotebooks](https://github.com/pluxbiosignals/biosignalsnotebooks)
- HRV 经典文献(Task Force 1996;Voss 2015 年龄常模;Brennan 2001 Poincaré;Peng 1995 DFA;Toichi 1997 CVI/CSI)

## 许可

MIT — see [LICENSE](LICENSE).

PLUX 的 SDK 与示例(若额外下载到本仓库)受 PLUX 自己的协议约束,本仓库不再分发。

## 鸣谢

- 受试者 **jie** 与 **ziqi** 同意公开她们的匿名生理数据
- [PLUX Wireless Biosignals, S.A.](https://biosignalsplux.com/) 提供的开源 API
