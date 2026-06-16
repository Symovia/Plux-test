# data/

每个 CSV 是一次 5 分钟、1000 Hz、3 通道的采集。

## 列格式

```
nSeq,t_sec,RIP,ECG,EDA
0,0.0000,38612,33284,5692
1,0.0010,38620,33260,5754
...
299999,299.9990,...
```

- `nSeq` — 设备帧号 0..299999
- `t_sec` — 秒数 0.0000..299.9990
- `RIP` — 呼吸带 ADC(整数,任意尺度)
- `ECG` — 心电 ADC(整数,任意尺度)
- `EDA` — 皮电 ADC(16-bit;µS 换算:`(ADC/65536) × 3.0 / 0.12`)

## 文件夹

```
jie/         被试 jie 三条件
├── stable.csv    平静
├── 中.csv         心事(脚本里 alias 到 "middle")
└── 乱.csv         焦虑(脚本里 alias 到 "mess")

ziqi/        被试 ziqi 三条件(数据质量较差,仅供对比参考)
├── stable.csv
├── middle.csv
└── mess.csv
```

## 用自己的数据

把你自己的 CSV 按上面的格式放到 `data/yourname/{stable,middle,mess}.csv`,然后改 analysis 脚本里的 subject 列表(`SUBJECTS = [...]`)即可。
