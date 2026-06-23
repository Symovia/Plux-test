# Plux-test

PLUX biosignalsplux three-channel physiological signal project for RIP respiration, ECG heart signal, and EDA skin conductance. The repository covers acquisition, feature engineering, a three-class stress-state classifier, and five normalized outputs for art visualization.

> Purpose: provide a complete data pipeline from physiological signals to five 0-100 creative-control values for an art project.

## Project Structure

```text
acquisition/    PLUX Bluetooth acquisition scripts (5 min @ 1000 Hz)
webui/          Live acquisition dashboard and real-time classifier (port 8000)
analysis/       Offline analysis scripts for quality checks, stress gradients, classifier training, deep ECG analysis, and five-state outputs
art_viz/        Browser art-visualization frontend (port 8001)
models/         Trained Random Forest classifier and personalized thresholds
tracks/         Five-channel art-input CSV files, about 301 rows per condition
data/           Raw 5 min x 3 condition CSV recordings for jie and ziqi
output/         Generated analysis artifacts, ignored by Git
PROJECT_SUMMARY.md   Full pipeline notes, formulas, references, and troubleshooting
```

## Quick Start

### 1. Environment

```powershell
# Python 3.10 is recommended because the PLUX API does not provide wheels for every Python version.
winget install Python.Python.3.10

# Create venv and install dependencies.
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### 2. Get the PLUX Python API

The PLUX `.pyd` binary is not included in this repository because it is platform-specific and distributed under PLUX's own terms. Download it from the official samples repository:

```powershell
git clone https://github.com/pluxbiosignals/python-samples.git
# Copy PLUX-API-Python3/ into this repository root, or point to it with:
$env:PLUX_API_DIR = "C:\path\to\python-samples\PLUX-API-Python3\Win64_310"
```

### 3. Pair Bluetooth and Set the Device Address

Pair the biosignalsplux device in Windows Bluetooth settings, then record the MAC-style BTH address:

```powershell
$env:PLUX_ADDRESS = "BTH00:07:80:XX:XX:XX"
```

You can also edit the default `ADDRESS` value in `acquisition/acquire_save_plot.py`.

### 4. Run

```powershell
# Five-minute acquisition with live plotting and CSV output.
.\.venv\Scripts\python acquisition\acquire_save_plot.py

# Live web dashboard: http://127.0.0.1:8000
.\.venv\Scripts\python webui\server.py

# Art visualization: http://127.0.0.1:8001
.\.venv\Scripts\python art_viz\server_art.py

# Offline analysis using existing CSV files in data/jie/.
.\.venv\Scripts\python analysis\_jie_report.py
.\.venv\Scripts\python analysis\_jie_classifier.py
.\.venv\Scripts\python analysis\_5tracks.py
```

## Five Physiological Output Tracks

`analysis/_5tracks.py` converts each one-second step of a 60-second rolling window into five numeric tracks:

| Variable | 0 to 100 Meaning | Main Drivers |
|---|---|---|
| `fatigue` | Alert to drowsy | Higher RMSSD, lower HR, slower breathing |
| `focus` | Scattered to deeply focused | Higher DFA alpha1 and higher LF/HF |
| `heart_age` | Estimated cardiac age in years | `log(130 / SDNN) / 0.018` |
| `coherence` | Disordered to synchronized cardio-respiratory rhythm | Peak SciPy coherence |
| `resilience` | Slow recovery to fast rebound | CVI, Cardiac Vagal Index |

Output format in `tracks/tracks_*.csv`:

```text
t_sec, fatigue, focus, heart_age, coherence, resilience
0,     60.6,    32.8,  38.8,      75.7,      52.4
...
```

## Dataset

`data/jie/` and `data/ziqi/` each contain three five-minute recordings sampled at 1000 Hz. The recordings correspond to three self-reported affective conditions:

| Condition | Meaning | File |
|---|---|---|
| stable | Calm baseline | `stable.csv` |
| middle | Concern / mild tension | `middle.csv` |
| mess | Anxious / messy state | `mess.csv` |

Each CSV has five columns: `nSeq`, `t_sec`, `RIP`, `ECG`, and `EDA`, stored as raw ADC integers.

Quality comparison showed that jie's data is cleaner than ziqi's data, with better electrode contact, about 6 dB higher ECG SNR, and lower EDA noise occupancy. The deeper analysis therefore uses jie as the main subject and keeps ziqi as comparison data.

## References

See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for hardware configuration, electrode placement, signal-processing steps, feature formulas, classifier evaluation, and troubleshooting notes.

Primary references:

- PLUX EDA Sensor Datasheet (EMG 03092020 REV B, 2020 PLUX)
- PLUX biosignalsplux ECG User Manual
- [pluxbiosignals/python-samples](https://github.com/pluxbiosignals/python-samples)
- [pluxbiosignals/biosignalsnotebooks](https://github.com/pluxbiosignals/biosignalsnotebooks)
- Classic HRV references: Task Force 1996, Voss 2015 age norms, Brennan 2001 Poincare, Peng 1995 DFA, Toichi 1997 CVI/CSI

## License

MIT. See [LICENSE](LICENSE).

PLUX SDK files and examples, if downloaded separately into this repository, remain governed by PLUX's own license terms and are not redistributed here.

## Acknowledgements

- Participants **ziqi** agreed to publish her anonymized physiological recording.
- [PLUX Wireless Biosignals, S.A.](https://biosignalsplux.com/) provides the open-source API and examples used by this project.
