# PLUX 生理信号采集与分析项目总结

> 用 PLUX biosignalsplux 设备采集呼吸、心电、皮电三通道生理信号,从原始 ADC 到压力分类、深度心电特征挖掘和 5 状态生理推断的全流程。

---

## 1. 项目目标

- 用 PLUX biosignalsplux 设备同步采集 **RIP / ECG / EDA** 三通道生理信号
- 在两个被试(jie / ziqi)三种情绪条件(stable / middle / mess)下各采集 5 分钟 1000 Hz 数据
- 建立从设备→CSV→特征→分类→生理状态推断的完整离线分析链路
- 训练 **3 类压力状态分类器**(平静 / 心事 / 焦虑)
- 从心电中挖掘 HR 之外的**深度特征**(非线性 HRV、波形形态、ECG 衍生呼吸、自主神经指数等)
- 推断 **5 个生理状态数值**(疲劳、专注、心脏年龄、心肺协调、弹性),作为下游艺术可视化的驱动信号

---

## 2. 硬件

### 2.1 设备

**PLUX biosignalsplux** 4-channel hub,蓝牙(BTH)连接。
- 设备 BT MAC:`BTH00:07:80:8C:AD:B3`
- 采样率:1000 Hz
- ADC 分辨率:16 bit
- VCC:3.0 V

### 2.2 传感器与端口配置

| 端口 | class 码 | 传感器类型 | 中文 |
|---|---|---|---|
| 1 | 6 | Inductive Respiration (RIP) | 电感式呼吸带(绑胸腔) |
| 2 | 2 | ECG | 心电(三电极,胸前) |
| 3 | 4 | EDA | 皮肤电导(指尖) |

### 2.3 通道位掩码

`device.start(SAMPLE_RATE, code=0x07, resolution=16)` — `0x07 = 0b0111` 启用 port 1+2+3。

### 2.4 电极接法

- **EDA**:依据手册 Fig. 3,中节指骨 + 相邻两手指(红色 = +,黑色 = −)
- **ECG**:三电极 Lead II 等效(红 = 左下肋骨 / 黑 = 右锁骨下 / 白 = 参考)
- **RIP**:胸腔围绕

---

## 3. 开发环境

### 3.1 Python

- **Python 3.10.11**(64-bit Windows)
  - PLUX Python API 在 Windows 上只提供 3.7/3.8/3.9/3.10/3.13 的预编译 `.pyd`,**没有 3.12**
  - 工作机原本是 3.12,通过 `winget install Python.Python.3.10` 并存装了 3.10
  - 用 `py -3.10 -m venv .venv` 建立虚拟环境

### 3.2 venv 路径

`E:\Performance\python-samples\.venv\`

### 3.3 安装的 Python 包

```
plux                 1.11 (PLUX API, 预编译 .pyd)
numpy                2.2.6
scipy                1.15.3
pandas               2.3.3
matplotlib           3.10.9
scikit-learn         1.7.2
biosppy              2.1.2
h5py                 3.16.0
seaborn              0.13.2
bokeh                3.9.0
statsmodels          0.14.6
PyWavelets           1.8.0
nbconvert            7.17.1
jupyter              (full meta-package)
jupyter_contrib_nbextensions
peakutils            (biosppy 的传递依赖)
fastapi              (Web 后端)
uvicorn[standard]    (ASGI 服务器)
joblib               (模型序列化)
```

### 3.4 PLUX 库的 DLL 依赖

- `Win64_310/plux.pyd` — Python 扩展模块
- `LibFT4222-64.dll` + `LibFT4222AB-64.dll` — USB-FTDI 桥接库(BT 连接时不强制需要,USB 线连接时需要;预备好放在 plux.pyd 同目录)

---

## 4. 引用的代码仓库

全部克隆自 GitHub 组织 [`pluxbiosignals`](https://github.com/pluxbiosignals):

| 仓库 | 用途 | 关键文件 |
|---|---|---|
| [python-samples](https://github.com/pluxbiosignals/python-samples) | PLUX Python API + 采集示例 | `OneDeviceAcquisitionExample.py`,`PLUX-API-Python3/Win64_310/plux.pyd` |
| [biosignalsnotebooks](https://github.com/pluxbiosignals/biosignalsnotebooks) | 信号处理 Python 包 + 教学 Notebook | 已裁剪到只留代码(约 1.1 GB) |
| [opensignals-samples](https://github.com/pluxbiosignals/opensignals-samples) | OpenSignals 集成示例 (LSL / TCP_IP / EPrime) | 未使用,备用 |
| [cpp-samples](https://github.com/pluxbiosignals/cpp-samples) | C++ API 示例 | `LibFT4222-64.dll`,`LibFT4222AB-64.dll` 从这里复制 |
| [android-sample](https://github.com/pluxbiosignals/android-sample) | Android API | 未使用,备用 |
| [unity-sample](https://github.com/pluxbiosignals/unity-sample) | Unity 集成 | 未使用,备用 |

---

## 5. 数据采集

### 5.1 采集脚本

`E:\Performance\python-samples\acquire_save_plot.py`

核心循环:
```python
class AcquisitionDevice(plux.SignalsDev):
    def __init__(self, address):
        plux.MemoryDev.__init__(address)
    
    def setup(self, csv_path):
        # CSV writer + matplotlib 实时绘图初始化
        ...
    
    def onRawFrame(self, nSeq, data):
        # 每个采样点回调
        # - 写入 CSV
        # - 追加到滚动 deque buffer
        # - 每 50 帧更新 matplotlib(20 fps)
        # - 每 1000 帧 flush CSV
        return nSeq >= self.target_samples
```

### 5.2 协议参数

| 参数 | 值 |
|---|---|
| 采样率 | 1000 Hz |
| 采集时长 | 300 s(5 分钟) |
| 通道 code | 0x07 (port 1+2+3) |
| 分辨率 | 16 bit |
| 蓝牙地址 | `BTH00:07:80:8C:AD:B3` |

### 5.3 调试过程中遇到的问题

| 问题 | 原因 | 解决 |
|---|---|---|
| `RuntimeError: The device could not be found` | 设备掉电休眠 | 充电 + 按电源键唤醒 |
| 之后还是连不上 | 用户报错的 MAC 末字节 `4F` 实际是 `B3` | 用 `plux.BaseDev.findDevices()` 拿到真实地址 |
| `findDevices()` 无参返回空 | 默认扫描 USB | 改用 `findDevices('BTH')` |
| `TypeError: ProactorEventLoop object is not callable` | uvicorn[standard] 的 WebSocket 在 Windows ProactorEventLoop 上不工作 | 顶层加 `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())` |
| 第二个 `TypeError: ProactorEventLoop object is not callable` | 自己把 `self.loop = asyncio_loop` 命名冲突覆盖了基类的 `plux.SignalsDev.loop()` 方法 | 改成 `self.event_loop` |
| sklearn 分类器输出"反向" | `predict_proba` 按 `clf.classes_` **字母序**返回,不是训练时的传入顺序 | 必须用 `clf.classes_` 映射概率 |

---

## 6. 数据集

### 6.1 实验设计

每个被试在 3 种自我报告情绪状态下各录 5 分钟,**不间断、不打标、不刺激**(静息态)。

| 条件 | 自我报告状态 | 持续 |
|---|---|---|
| `stable` | 平静 — 心情平稳 | 5 分钟 |
| `middle`(原文件名 `中.csv`) | 心事 — 有些心事/微紧张 | 5 分钟 |
| `mess`(原文件名 `乱.csv`) | 焦虑 — 明显焦虑 | 5 分钟 |

### 6.2 文件清单

```
E:\Performance\z_data\
├── jie\
│   ├── stable.csv         # 5 min 平静
│   ├── 中.csv             # 5 min 心事
│   └── 乱.csv             # 5 min 焦虑
└── ziqi\                  # 同结构,但数据质量差,本项目未深入分析
    ├── stable.csv
    ├── middle.csv
    └── mess.csv
```

每个 CSV:

| 列 | 描述 |
|---|---|
| `nSeq` | 设备帧号 0..299999 |
| `t_sec` | 秒数 0.0000..299.9990 |
| `RIP` | 端口 1 原始 ADC 值 (uint16) |
| `ECG` | 端口 2 原始 ADC 值 |
| `EDA` | 端口 3 原始 ADC 值 |

每文件 300,001 行 × 5 列,约 10 MB。

### 6.3 被试数据质量对比

通过对比分析(`_compare.py`)发现 jie 数据明显优于 ziqi:

| 指标 | jie 均值 | ziqi 均值 |
|---|---|---|
| ECG SNR (dB) | 32.6 | 26.6 |
| EDA 噪声占比 | 16.2% | 64.8% |
| RMSSD (ms) | 68 | 149(异常高,提示假心拍) |
| 复合质量分 | +0.35 | −0.35 |

**结论:本项目深度分析全部基于 jie 的数据。** ziqi 数据保留作为对比参考。

---

## 7. 信号预处理与基础特征

### 7.1 ECG 处理流水线

```python
from scipy.signal import butter, sosfiltfilt, find_peaks

# 1. 带通滤波 5-40 Hz(隔离 QRS 频带)
sos = butter(4, [5, 40], btype="band", fs=1000, output="sos")
ecg_bp = sosfiltfilt(sos, ecg - ecg.mean())

# 2. R 峰检测
rpeaks, _ = find_peaks(
    ecg_bp,
    distance=int(1000 * 0.4),       # 最小间距 400 ms(HR < 150)
    prominence=ecg_bp.std() * 2.0,  # 突出度阈值
)

# 3. RR 间期计算(ms),剔除生理外
rr_ms = np.diff(rpeaks) / 1000 * 1000
rr_clean = rr_ms[(rr_ms > 400) & (rr_ms < 1500)]
```

### 7.2 EDA 处理(根据手册公式)

**传递函数**(PLUX EDA Sensor Datasheet EMG 03092020, REV B © 2020 PLUX,第 2 页):

```
EDA(µS) = (ADC / 2^n) × VCC / 0.12
EDA(S)  = EDA(µS) × 10⁻⁶

VCC = 3 V, n = 16 (默认 ADC 分辨率)
量程: 0 - 25 µS
带宽: 0 - 3 Hz
```

```python
# 1. ADC -> µS
eda_us = (eda_adc / 2**16) * 3.0 / 0.12

# 2. 低通滤波到 3 Hz(手册带宽内,去工频/接触噪声)
sos_lp = butter(4, 3.0, btype="low", fs=1000, output="sos")
smooth = sosfiltfilt(sos_lp, eda_us)

# 3. Tonic(SCL)= 极低通(0.05 Hz)
sos_t = butter(2, 0.05, btype="low", fs=1000, output="sos")
tonic = sosfiltfilt(sos_t, smooth)

# 4. Phasic = smooth - tonic
phasic = smooth - tonic

# 5. SCR 检测
scr_peaks, props = find_peaks(
    phasic, distance=1000*1.0, prominence=0.02, height=0.03
)
```

### 7.3 RIP 处理

```python
# 1. 去 DC + 低通 1 Hz
dc = rip - rip.mean()
sos = butter(4, 1.0, btype="low", fs=1000, output="sos")
lp = sosfiltfilt(sos, dc)

# 2. 呼吸峰检测
peaks, _ = find_peaks(
    lp, distance=int(1000 * 1.5),  # 最快 40/min
    prominence=lp.std() * 0.4,
)

# 3. 呼吸率与规律性
intervals = np.diff(peaks) / 1000   # s
rate = 60 / intervals.mean()
cv = intervals.std() / intervals.mean()
```

---

## 8. 特征工程总览

按类别梳理所有用到的特征:

### 8.1 时域 HRV(传统)

| 名称 | 公式 | 物理意义 |
|---|---|---|
| HR (bpm) | 60000 / mean(RR ms) | 心率 |
| SDNN (ms) | std(RR) | 总变异 |
| RMSSD (ms) | √mean(diff(RR)²) | 短时变异(副交感) |
| pNN50 (%) | 相邻 RR 差 > 50ms 的比例 | 副交感强度 |

### 8.2 频域 HRV

RR 间期立方插值重采样到 4 Hz,Welch 法 PSD:

| 名称 | 频带 | 物理意义 |
|---|---|---|
| VLF | 0.003–0.04 Hz | 极低频(温度、代谢) |
| LF | 0.04–0.15 Hz | 交感为主 |
| HF | 0.15–0.40 Hz | 副交感(呼吸性) |
| LF/HF | — | 交感/副交感比 |
| LFnu, HFnu | LF/(LF+HF), HF/(LF+HF) | 归一化单位 |

### 8.3 非线性 HRV

| 名称 | 算法 | 物理意义 |
|---|---|---|
| SD1 (ms) | std(diff(RR))/√2 | Poincaré 短轴 ≈ 副交感 |
| SD2 (ms) | √(2·var(RR) − SD1²) | Poincaré 长轴 ≈ 总变异 |
| SD1/SD2 | — | 自主神经平衡 |
| SampEn | m=2, r=0.2·std(RR) | 心律复杂度 |
| DFA α1 | 4–16 scale, log-log slope | 短时尺度长程相关 |

**DFA α1 实现要点**(自实现,无外部包):
```python
def dfa_alpha1(rr, scales=range(4, 17)):
    y = np.cumsum(rr - rr.mean())
    F = []
    for n in scales:
        seg = len(y) // n
        local = []
        for k in range(seg):
            s = y[k*n:(k+1)*n]
            t = np.arange(n)
            trend = np.polyval(np.polyfit(t, s, 1), t)
            local.append(np.mean((s - trend) ** 2))
        F.append(np.sqrt(np.mean(local)))
    slope, _ = np.polyfit(np.log(list(scales)), np.log(F), 1)
    return slope
```

### 8.4 自主神经指数

| 名称 | 公式 | 物理意义 |
|---|---|---|
| **CVI** | log₁₀(16 · SD1 · SD2) | Cardiac Vagal Index(副交感强度) |
| **CSI** | SD2 / SD1 | Cardiac Sympathetic Index(交感优势) |

### 8.5 心电波形形态

| 名称 | 算法 |
|---|---|
| R 幅均值/std/CV | `ecg[rpeaks]` 统计 |
| QRS 宽度均值/std | 半幅宽估算,以 R 峰为中心 |
| 异位搏数 | RR 在 [400, 1500] 之外的拒绝数 |
| 大幅 RR 跳变 | abs(diff(RR)) > 20% · median(RR) 的数量 |

### 8.6 ECG 衍生呼吸(EDR)

物理基础:深呼吸使心脏在胸腔内位移,改变电极到心脏几何关系,**调制 R 波幅值**。

```python
# 1. 取 R 峰幅值序列
r_amps = ecg[rpeaks] - ecg.mean()

# 2. 立方插值到 4 Hz 均匀网格
t_r = rpeaks / 1000
t_uni = np.arange(t_r[0], t_r[-1], 0.25)
edr_uni = interp1d(t_r, r_amps, kind="cubic")(t_uni)

# 3. 低通到 1 Hz(呼吸频带)
sos = butter(4, 1.0, btype="low", fs=4.0, output="sos")
edr_smooth = sosfiltfilt(sos, edr_uni - edr_uni.mean())

# 4. 与真实 RIP 相关性 = EDR 质量
corr = np.corrcoef(rip_resampled, edr_smooth)[0, 1]
```

### 8.7 EDA 特征

| 名称 | 物理意义 |
|---|---|
| SCL_mean (µS) | 紧张基线 |
| SCL_slope (µS/min) | 紧张状态趋势 |
| SCR_rate (/min) | 唤醒事件频率 |
| SCR_amp_mean (µS) | 每次唤醒振幅 |
| EDA noise % | >3 Hz 噪声功率占比(数据质量) |

### 8.8 RIP 特征

| 名称 | 物理意义 |
|---|---|
| Resp rate (bpm) | 呼吸频率 |
| Resp CV | 呼吸规律性(变异系数) |
| Tidal amplitude | 潮气幅度 |

### 8.9 心肺耦合(Coherence)

```python
from scipy.signal import coherence

# RR 和 RIP 同步重采样到 4 Hz,scipy.signal.coherence 计算
f, Cxy = coherence(rr_uni, rip_resampled, fs=4.0, nperseg=256)
# 取呼吸带峰值
band = (f >= 0.1) & (f <= 0.45)
peak_coh = Cxy[band].max()
```

---

## 9. 三分类压力状态分类器

### 9.1 数据集构建(滑动窗口)

```
窗口长度:   60 秒
步长:       10 秒
特征数:     11
窗口数:     25 / 条件 × 3 条件 = 75 总样本
```

**11 维特征向量**:
```
['mean_HR', 'RMSSD', 'SDNN', 'LF_HF',
 'SCL_mean', 'SCL_slope', 'SCR_rate', 'SCR_amp_mean',
 'resp_rate', 'resp_CV', 'resp_amp_std']
```

### 9.2 模型与训练

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=8,
    random_state=42, class_weight="balanced"
)
```

### 9.3 评估

| 评估方式 | RF | Logistic Reg |
|---|---|---|
| Stratified 5-fold CV(乐观,窗口重叠泄漏) | **100.0%** | 98.7% |
| 时序拆分(前 70% 训 / 后 30% 测,诚实) | **83.3%** | — |

**时序拆分混淆矩阵**(8 个窗口/类):
```
              Predicted
              stable  middle  mess
True  stable     4       4      0
      middle     0       8      0
      mess       0       0      8
```

middle 和 mess 完全可分;stable 后段 50% 漂移到 middle 边界(后期 stable 状态逐渐松弛/进入 middle 基线)。

### 9.4 特征重要性(RF Gini)

```
SCL_mean        28.1%   ← 皮电基线
SCR_rate        17.9%   ← SCR 频率
resp_CV         16.5%   ← 呼吸规律性
SCR_amp_mean    10.5%   ← SCR 振幅
mean_HR          9.0%   ← 心率
resp_amp_std     8.3%   ← 潮气量变异
resp_rate        3.0%
SDNN             2.2%
RMSSD            2.2%
SCL_slope        1.2%
LF_HF            1.1%   ← HRV 频域贡献最低
```

**EDA 贡献了 56.5% 的判别力**。HRV 排末位(因 jie 的 HRV 在 mess 反弹,非单调,分类器学会忽略)。

### 9.5 模型持久化

```
E:\Performance\z_data\jie_report\jie_classifier.joblib
  ├── model       : RandomForestClassifier
  ├── scaler      : StandardScaler
  ├── feature_cols: 11 个特征名(顺序)
  └── classes     : ['stable','middle','mess']  (此处保存的是 CONDS 顺序)
```

**⚠ 关键 bug 教训**:`predict_proba` 按 `clf.classes_` **字母序**返回,与 `bundle["classes"]` 的字典传入顺序不一致。下游推断必须用 `clf.classes_`(实际为 `['mess','middle','stable']`)而非 bundle 里的列表。

### 9.6 个体化阈值表

基于 75 个训练窗口的 p10-p90 分位:

| 状态 | SCL (µS) | SCR /min | HR (bpm) | 呼吸 CV |
|---|---|---|---|---|
| stable | 2.53 – 4.21 | 0 – 3 | 64 – 71 | 0.15 – 0.29 |
| middle | 4.31 – 5.33 | 2 – 8 | 65 – 75 | 0.31 – 0.49 |
| mess | 6.37 – 6.65 | 12 – 19 | 74 – 80 | 0.32 – 0.55 |

**速判规则**:只看 SCL 一项 — `<4.2 平静,4.3-5.3 心事,>6 焦虑`(仅对 jie 校准)。

---

## 10. 压力梯度生理报告(三条件)

| 指标 | stable 平静 | middle 心事 | mess 焦虑 | 趋势 |
|---|---|---|---|---|
| HR (bpm) | 67.8 | 71.1 | 76.6 | ↑↑↑ 单调 ✓ |
| SDNN (ms) | 77.5 | 84.0 | 98.6 | ↑(深呼吸驱动 RSA 放大) |
| RMSSD (ms) | 53.8 | 47.3 | 76.9 | V 形 ⚠ |
| LF/HF | 1.20 | **2.67** | 1.06 | ⌒ 峰在 middle |
| SCL 基线 (µS) | 3.26 | 4.93 | **6.48** | ↑↑↑ 翻倍 |
| SCL 趋势 (µS/min) | −0.42 | −0.26 | **+0.20** | "降温" → "升温" |
| SCRs/min | 1.6 | 5.0 | **14.8** | 近 10 倍 |
| 呼吸 CV | 0.216 | 0.397 | 0.444 | ↑ |
| 潮气量 (RIP p2p) | 4477 | 6384 | **12556** | ~3 倍 |

**复合压力指数(z 综合)**:
```
stable: −3.15 ← 非常放松
middle: +2.09 ← 压力峰(LF/HF 主导)
mess:   +1.06 ← 高压力但 HRV 反弹
```

**核心发现**:
- middle 是经典"持续紧张"模式(LF/HF 峰、RMSSD 谷)
- mess 是"双激活"模式(交感 + 副交感同时拉满,深喘气驱动 RSA 反弹)
- middle 和 mess 是**两种不同质感的压力**,不是简单的程度差异

---

## 11. 深度心电特征挖掘

从单一 ECG 通道额外挖出 5 类信号(详见第 8 节相关算法):

| 类别 | 在三条件上的关键发现 |
|---|---|
| 非线性 HRV | DFA α1: stable 1.08 / middle **1.25**(认知占用) / mess 1.13;SampEn: mess 最低(深呼吸把心律锁到 RSA 节律) |
| 波形形态 | mess 的 R 波幅 std 是 stable 的 **3 倍**(物理证明深呼吸调制心电幅值) |
| EDR(衍生呼吸) | 与 RIP 相关性 0.05–0.15,弱但非零(简单 R 幅包络方法的局限) |
| 自主指数 | CVI: 4.80/4.79/5.05;CSI: 2.70/**3.41**/2.36 — 解开 V 形 HRV(mess 是"双激活"而非更紧张) |
| 心律质量 | 0 异位搏(三条件完全干净),mess 大幅 RR 跳变 13 次(均为生理性 RSA) |

---

## 12. 5 状态生理推断(艺术输入)

为下游艺术可视化提供 5 个 0-100 的数值通道。

### 12.1 状态定义

| # | 名字 | 0 → 100 含义 | 主要驱动特征 |
|---|---|---|---|
| 1 | **fatigue 疲劳度** | 警觉 → 困倦 | RMSSD↑ + HR↓ + 慢呼吸 + CVI↑ |
| 2 | **focus 专注度** | 涣散 → 深度专注 | DFA α1↑ + CSI↑ + 浅呼吸 |
| 3 | **heart_age 心脏年龄** | 直接是岁数 | log(130/SDNN)/0.018 |
| 4 | **coherence 协调度** | 混乱 → 心肺完美同步 | scipy.signal.coherence(RR, RIP) 在 0.1-0.45 Hz 峰值 |
| 5 | **resilience 弹性** | 慢恢复 → 快反弹 | CVI 归一化 |

### 12.2 公式(简化版,易于艺术 mapping)

```python
fatigue   = clip(50 + 0.6·(RMSSD-35) − 0.8·(HR-70) − 2.0·(breath_rate-14), 0, 100)
focus     = clip(20 + 50·(DFA_α1-0.9) + 8·ln(LF/HF+1), 0, 100)
heart_age = clip(ln(130/SDNN)/0.018, 0, 100)        # years
coherence = clip(100·peak_coh, 0, 100)
resilience= clip((CVI - 3.8)·60, 0, 100)
```

### 12.3 心脏年龄公式参考

基于公开 HRV 年龄常模(粗略对数拟合):
- 健康成人 SDNN 每年约下降 1%
- 拟合:`SDNN(age) ≈ 130 · exp(-0.018 · age)`
- 反解:`age ≈ ln(130/SDNN) / 0.018`

> 文献参考类型:Umetani et al. 1998, Voss et al. 2015 等短时 HRV 年龄常模研究(此处用简化对数模型,误差 ±5 年量级)。

### 12.4 输出格式

3 个 CSV(每条件一个),每个 301 行 × 6 列:

```
t_sec, fatigue, focus, heart_age, coherence, resilience
0,     60.6,    32.8,  38.8,      75.7,      52.4
1,     ...
...
300,   ...
```

- **滑动窗口**:60 秒,步长 1 秒
- 前 60 秒为"窗口预热"(用 t=60 的值重复填充)
- 路径:`E:\Performance\z_data\jie_report\tracks_{stable,middle,mess}.csv`

### 12.5 三条件均值

| | 疲劳 | 专注 | 心脏年龄 | 协调 | 弹性 |
|---|---|---|---|---|---|
| stable | 60.6 | 32.8 | 38.8 岁 | 75.7 | 52.4 |
| middle | 56.1 | 42.4 | 34.2 岁 | 78.2 | 53.9 |
| mess | 67.6 | 43.4 | **14.9 岁** ⚠ | 45.7 | **73.9** |

**有意思的 quirk**:mess 状态心脏年龄反而"最年轻"(15 岁) — 因焦虑深呼吸把 HRV 放大,数学算法直接读出"年轻心脏",这是真实数据里的物理悖论。

---

## 13. 文件清单

### 13.1 采集脚本

```
E:\Performance\python-samples\
├── acquire_save_plot.py     # 5 min 采集 + 实时绘图 + CSV
├── _analyze.py              # 单次采集后期分析 + 通道分类
├── _discover.py             # plux.BaseDev.findDevices() 设备发现
└── PLUX-API-Python3\Win64_310\
    ├── plux.pyd             # PLUX Python 绑定
    ├── LibFT4222-64.dll     # FTDI 桥接(USB 备用)
    └── LibFT4222AB-64.dll
```

### 13.2 分析脚本(`E:\Performance\z_data\`)

```
_compare.py             # jie vs ziqi 质量对比
_jie_report.py          # jie 三条件压力梯度报告
_jie_classifier.py      # 三分类训练 + 模型保存
_stress_timeline.py     # 离线压力轨迹(用训练好的分类器跑全 5 分钟)
_cardiac_deep.py        # 深度心电特征(非线性 HRV / 形态 / EDR / 自主指数)
_jie_5states.py         # 5 状态打包估算(单值/条件)
_5tracks.py             # 5 状态逐秒时间序列(艺术输入)
```

### 13.3 输出数据(`E:\Performance\z_data\jie_report\`)

```
jie_classifier.joblib       # 训练好的 RF + scaler + 特征名 + 类别
jie_classifier.png          # 分类性能可视化
jie_features.csv            # 75 窗口 × 11 特征 + 标签
jie_thresholds.json         # p10-p90 个体阈值

jie_metrics.csv             # 三条件汇总指标
jie_stress_report.png       # 压力梯度报告图
jie_report.json             # 结构化指标

jie_stress_timeline.csv     # 离线压力分轨迹
jie_stress_timeline.png

jie_cardiac_deep.csv        # 深度心电特征
jie_cardiac_deep.png        # Poincaré + 各项对比

jie_5states.csv             # 5 状态(每条件单值)
jie_5states.png

tracks_stable.csv           # 5 通道艺术输入(301 行/条件)
tracks_middle.csv
tracks_mess.csv
tracks_combined.csv         # 三条件拼接(900+ 行,带 condition 列)
tracks_overview.png
```

### 13.4 原始 5 分钟采集 CSV

```
E:\Performance\z_data\
├── jie\
│   ├── stable.csv          # 300001 × 5 列
│   ├── 中.csv
│   └── 乱.csv
└── ziqi\                   # (本项目未深入)
    ├── stable.csv
    ├── middle.csv
    └── mess.csv
```

---

## 14. 关键参考资料

### 14.1 PLUX 官方资料

- **EDA Sensor Datasheet** — *Electrodermal Activity (EDA) Sensor Datasheet*,EMG 03092020,REV B © 2020 PLUX Wireless Biosignals, S.A.
  - 引用页:量程 / 带宽 / 传递函数 / Fig.3 电极接法
  - 工作地址:Av. 5 de Outubro, n. 70 – 2, 1050-059 Lisbon, Portugal
- **ECG User Manual** — *biosignalsplux Electrocardiography (ECG) User Manual*(原 PDF 加密未读出,使用通用三电极接法)
- **OpenSignals 软件文档** — https://opensignals.net
- **官网** — https://biosignalsplux.com/

### 14.2 GitHub 仓库(组织 `pluxbiosignals`)

| 仓库 | URL |
|---|---|
| python-samples | https://github.com/pluxbiosignals/python-samples |
| biosignalsnotebooks | https://github.com/pluxbiosignals/biosignalsnotebooks |
| opensignals-samples | https://github.com/pluxbiosignals/opensignals-samples |
| cpp-samples | https://github.com/pluxbiosignals/cpp-samples |
| android-sample | https://github.com/pluxbiosignals/android-sample |
| unity-sample | https://github.com/pluxbiosignals/unity-sample |

### 14.3 主要 Python 包文档

- numpy: https://numpy.org/
- scipy: https://scipy.org/ —— `scipy.signal.butter`, `sosfiltfilt`, `find_peaks`, `welch`, `coherence`
- scikit-learn: https://scikit-learn.org/ —— `RandomForestClassifier`, `StratifiedKFold`, `StandardScaler`, `PCA`
- biosppy: https://github.com/scientisst/BioSPPy —— 生物信号通用工具(本项目主要直接用 scipy)
- pandas: https://pandas.pydata.org/

### 14.4 算法理论引用方向

| 主题 | 经典文献方向 |
|---|---|
| HRV 时频域指标 | Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology, 1996, *Circulation* |
| 短时 HRV 常模与年龄回归 | Umetani et al. 1998, *JACC*;Voss et al. 2015, *PLoS ONE* |
| Poincaré SD1/SD2 | Brennan et al. 2001, *IEEE TBME* |
| Sample Entropy | Richman & Moorman 2000, *Am J Physiol* |
| Detrended Fluctuation Analysis | Peng et al. 1995, *Chaos* |
| CVI / CSI(Toichi) | Toichi et al. 1997, *J Auton Nerv Syst* |
| ECG-Derived Respiration | Moody et al. 1985(R 波幅度法) |
| EDA 信号处理 | Boucsein 2012, *Electrodermal Activity*(Springer) |

---

## 15. 复现指南

完整复现这个项目需要的最小步骤:

```powershell
# 1. 安装 Python 3.10
winget install Python.Python.3.10

# 2. 克隆 PLUX 仓库
cd E:\Performance
git clone https://github.com/pluxbiosignals/python-samples.git
git clone https://github.com/pluxbiosignals/biosignalsnotebooks.git

# 3. 建立 venv 并安装依赖
cd python-samples
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install numpy scipy pandas matplotlib scikit-learn biosppy h5py seaborn bokeh statsmodels PyWavelets nbconvert jupyter fastapi "uvicorn[standard]" joblib

# 4. 蓝牙配对 PLUX 设备,记录 MAC,改写脚本里的 ADDRESS
# 5. 跑采集
.\.venv\Scripts\python acquire_save_plot.py

# 6. 后续分析按顺序
.\.venv\Scripts\python ..\z_data\_jie_report.py
.\.venv\Scripts\python ..\z_data\_jie_classifier.py
.\.venv\Scripts\python ..\z_data\_cardiac_deep.py
.\.venv\Scripts\python ..\z_data\_5tracks.py
```

---

## 16. 重要"踩坑日志"(避坑参考)

1. **Python 版本必须匹配 PLUX .pyd**:Windows 64-bit 只有 3.7/3.8/3.9/3.10/3.13。**没有 3.11/3.12**。装 3.10 并存。

2. **PLUX BT 设备休眠极快**:断开后几秒到几十秒就睡眠。重连前可能要按电源键唤醒。

3. **`findDevices()` 默认不扫描蓝牙**:必须显式 `findDevices('BTH')`。

4. **MAC 别人工抄**:用代码 `findDevices('BTH')` 拿,避免人眼读错。

5. **PLUX `getSensors()` 返回的 class 是真相**:用户口头说的"接的是什么传感器"可能错(本项目里 port 2 用户说是 EDA,设备说是 ECG,数据印证是 ECG)。

6. **EDA 电极位置照手册 Fig. 3**:中节指骨,不是指尖。常见网络资料说的"指尖"在 PLUX 手册里并不是首选,且实测中节抗运动伪迹更好。

7. **sklearn `predict_proba` 按字母序返回**:必须用 `clf.classes_` 索引,绝不能信任自定义的类别顺序变量。这个 bug 可以让模型输出"完全反向"且没有任何报错。

8. **Python 子类方法名别和基类冲突**:`self.loop = asyncio_loop` 会覆盖 `plux.SignalsDev.loop()`,调用 `dev.loop()` 时会试图把事件循环当函数调用,报 `TypeError: 'ProactorEventLoop' object is not callable`。

9. **Windows uvicorn + WebSocket**:Default `ProactorEventLoop` 不兼容,顶层强制 `WindowsSelectorEventLoopPolicy()`。

10. **CSV 文件名汉化要兼容**:被试中途把 `middle.csv` 改成 `中.csv`,下游脚本要做别名映射。

---

*Project compiled: 2026-06 · jie 数据(2026-06-04 采集)· 项目工作树位于 `E:\Performance\`*
