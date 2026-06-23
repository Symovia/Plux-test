# data/

Each CSV is one five-minute, 1000 Hz, three-channel acquisition.

## Column Format

```text
nSeq,t_sec,RIP,ECG,EDA
0,0.0000,38612,33284,5692
1,0.0010,38620,33260,5754
...
299999,299.9990,...
```

- `nSeq`: device frame number, usually 0..299999
- `t_sec`: time in seconds, usually 0.0000..299.9990
- `RIP`: respiration belt ADC value, raw integer scale
- `ECG`: electrocardiogram ADC value, raw integer scale
- `EDA`: electrodermal activity ADC value, 16-bit; approximate microsiemens conversion is `(ADC / 65536) * 3.0 / 0.12`

## Folders

```text
jie/         three conditions for subject jie
|-- stable.csv    calm baseline
|-- middle.csv    concern / mild tension
`-- mess.csv      anxious / messy state

ziqi/        same three-condition structure for subject ziqi; lower signal quality, kept for comparison only
|-- stable.csv
|-- middle.csv
`-- mess.csv
```

## Using Your Own Data

Place your own CSV files in `data/yourname/{stable,middle,mess}.csv` using the same column format, then update the subject list in the analysis scripts, for example `SUBJECTS = [...]`.
