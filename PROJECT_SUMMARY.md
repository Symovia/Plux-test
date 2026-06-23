# PLUX Physiological Signal Acquisition and Analysis Summary

This project uses a PLUX biosignalsplux device to acquire three synchronized physiological signals: RIP respiration, ECG heart signal, and EDA skin conductance. The repository documents the full workflow from raw ADC samples to CSV files, signal features, stress-state classification, deeper ECG-derived features, and five normalized values for downstream art visualization.

## 1. Project Goals

- Acquire synchronized RIP, ECG, and EDA data with the PLUX biosignalsplux device.
- Record five-minute, 1000 Hz sessions for two participants, jie and ziqi, under three self-reported conditions: stable, middle, and mess.
- Build a complete offline pipeline from device output to CSV, features, classification, and physiological-state inference.
- Train a three-class stress-state classifier for stable, middle, and mess conditions.
- Extract deeper ECG features beyond heart rate, including nonlinear HRV, beat morphology, ECG-derived respiration, autonomic indices, and rhythm quality.
- Generate five physiological state tracks for art visualization: fatigue, focus, heart age, cardio-respiratory coherence, and resilience.

## 2. Hardware

The system uses a PLUX biosignalsplux four-channel hub connected through Bluetooth.

| Port | Class Code | Sensor | Placement |
|---|---:|---|---|
| 1 | 6 | Inductive Respiration (RIP) | Respiration belt around the chest |
| 2 | 2 | ECG | Three electrodes in an approximate Lead II layout |
| 3 | 4 | EDA | Two adjacent fingers, placed according to the PLUX EDA manual |

Important acquisition parameters:

| Parameter | Value |
|---|---|
| Sample rate | 1000 Hz |
| Duration | 300 s, or 5 min |
| Channel code | `0x07`, enabling ports 1, 2, and 3 |
| Resolution | 16 bit |
| Device address | Set with `PLUX_ADDRESS` or the script default |

## 3. Development Environment

The PLUX Python API requires a matching precompiled `plux.pyd`. Python 3.10 is used because it is available in the PLUX Windows binary set and is stable with the project dependencies.

Typical setup:

```powershell
winget install Python.Python.3.10
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The PLUX API can be copied into the repository as `PLUX-API-Python3/Win64_310`, or supplied through `PLUX_API_DIR`.

## 4. Reference Repositories

The project is informed by the official `pluxbiosignals` GitHub organization:

| Repository | Purpose |
|---|---|
| `python-samples` | PLUX Python API and acquisition examples |
| `biosignalsnotebooks` | Signal-processing examples and teaching notebooks |
| `opensignals-samples` | OpenSignals integration examples, kept as reference |
| `cpp-samples` | C++ API examples and DLL references |
| `android-sample` | Android API examples, not used directly |
| `unity-sample` | Unity integration examples, not used directly |

## 5. Data Acquisition

The acquisition scripts connect to the PLUX device, start a 1000 Hz stream, write each sample to CSV, and optionally render live traces. Each CSV row contains:

```text
nSeq,t_sec,RIP,ECG,EDA
```

The web dashboard in `webui/server.py` adds a browser interface, WebSocket streaming, rolling respiration assessment, and real-time classifier output when a trained model is available.

## 6. Dataset

The dataset contains three conditions per subject:

| Condition | Meaning | Duration |
|---|---|---|
| `stable` | Calm baseline | 5 min |
| `middle` | Concern / mild tension | 5 min |
| `mess` | Anxious or mentally messy state | 5 min |

Files are organized as:

```text
data/
|-- jie/
|   |-- stable.csv
|   |-- middle.csv
|   `-- mess.csv
`-- ziqi/
    |-- stable.csv
    |-- middle.csv
    `-- mess.csv
```

Quality comparison showed that jie's recordings are cleaner than ziqi's, especially for ECG SNR and EDA noise occupancy. The deeper analyses therefore use jie as the primary dataset and keep ziqi as a comparison subject.

## 7. Signal Processing

### ECG

The ECG pipeline band-pass filters the signal in the QRS range, detects R-peaks, converts peak intervals into RR intervals, removes physiologically impossible intervals, then computes HR and HRV metrics such as SDNN, RMSSD, pNN50, LF, HF, and LF/HF.

### EDA

EDA raw ADC values are converted to approximate microsiemens using the PLUX transfer formula:

```text
EDA_us = (ADC / 65536) * 3.0 / 0.12
```

The EDA pipeline smooths within the sensor bandwidth, estimates tonic SCL with a very low-pass filter, derives phasic activity by subtraction, then detects SCR events and amplitudes.

### RIP

The respiration pipeline removes DC offset, low-pass filters the belt signal, detects respiration peaks and troughs, then estimates respiration rate, breathing regularity, and amplitude variation.

## 8. Feature Engineering

The feature set includes:

- ECG and HRV: mean HR, SDNN, RMSSD, pNN50, LF/HF.
- EDA: SCL mean, SCL slope, SCR rate, SCR amplitude.
- RIP: respiration rate, respiration coefficient of variation, respiration amplitude variation.
- Deep ECG: Poincare SD1/SD2, sample entropy, DFA alpha1, beat morphology, ECG-derived respiration, CVI, CSI, ectopic rejection, and large RR jumps.

## 9. Classifier

`analysis/_jie_classifier.py` trains a Random Forest classifier on rolling 60-second windows. The labels are `stable`, `middle`, and `mess`. The final trained bundle is stored at:

```text
models/jie_classifier.joblib
```

The live dashboard loads this model, extracts matching rolling-window features, predicts class probabilities, and converts them into a continuous 0-100 stress score:

```text
stable = 0
middle = 50
mess = 100
stress_score = weighted probability average
```

The implementation uses `clf.classes_` to map `predict_proba` outputs because scikit-learn stores class probability columns in sorted class order.

## 10. Five Art Tracks

`analysis/_5tracks.py` produces five per-second tracks for the art interface:

| Track | Meaning |
|---|---|
| `fatigue` | Higher values suggest lower arousal and more fatigue-like physiology |
| `focus` | Higher values suggest stronger sustained attention or cognitive load |
| `heart_age` | HRV-derived estimated cardiac/autonomic age |
| `coherence` | Cardio-respiratory synchrony based on coherence |
| `resilience` | Recovery capacity based mainly on vagal/autonomic indices |

The generated files are placed in `tracks/` and served by `art_viz/server_art.py`.

## 11. Art Visualization

The art visualization maps the five tracks to animated visual properties:

| Physiological Track | Visual Mapping |
|---|---|
| Fatigue | Background intensity |
| Focus | Shape edge sharpness |
| Heart age | Hue |
| Coherence | Arc smoothness |
| Resilience | Bounce / rebound behavior |

The art visualization server runs on port 8001.

## 12. Important Limitations

This project is not a medical device and does not diagnose health, anxiety, fatigue, attention, or heart disease. The labels are self-reported experimental states for a creative project. The classifier is individualized to the available jie dataset and should not be treated as a general stress detector.

## 13. How to Reproduce

```powershell
# Install Python 3.10 and dependencies.
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# Set PLUX API path if needed.
$env:PLUX_API_DIR="C:\path\to\PLUX-API-Python3\Win64_310"

# Set device address if needed.
$env:PLUX_ADDRESS="BTH00:07:80:XX:XX:XX"

# Run acquisition.
.\.venv\Scripts\python acquisition\acquire_save_plot.py

# Run offline analysis.
.\.venv\Scripts\python analysis\_jie_report.py
.\.venv\Scripts\python analysis\_jie_classifier.py
.\.venv\Scripts\python analysis\_stress_timeline.py
.\.venv\Scripts\python analysis\_cardiac_deep.py
.\.venv\Scripts\python analysis\_jie_5states.py
.\.venv\Scripts\python analysis\_5tracks.py

# Run browser interfaces.
.\.venv\Scripts\python webui\server.py
.\.venv\Scripts\python art_viz\server_art.py
```

## 14. Troubleshooting Notes

- The Python version must match the PLUX `.pyd` binary.
- Bluetooth discovery must explicitly scan BTH devices with `findDevices('BTH')`.
- The PLUX device may sleep quickly after disconnecting and may need to be awakened before reconnecting.
- Device addresses should be discovered programmatically when possible to avoid transcription errors.
- `getSensors()` is the source of truth for connected sensor classes.
- The Windows uvicorn WebSocket stack works more reliably with `WindowsSelectorEventLoopPolicy()`.
- Avoid naming subclass attributes `loop` because that can shadow `plux.SignalsDev.loop()`.
- Use `clf.classes_` when mapping scikit-learn classifier probabilities.

## 15. References

- PLUX EDA Sensor Datasheet, EMG 03092020 REV B, 2020 PLUX.
- PLUX biosignalsplux ECG User Manual.
- PLUX official `python-samples` and `biosignalsnotebooks` repositories.
- Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology, 1996.
- Voss et al., 2015, HRV age norms.
- Brennan et al., 2001, Poincare HRV descriptors.
- Peng et al., 1995, DFA.
- Toichi et al., 1997, CVI/CSI.

Project compiled in June 2026 using jie data collected on 2026-06-04.
