"""PLUX 5-minute acquisition + live stress classifier dashboard.

Adds to base acquisition:
  - Loads trained 3-class stress classifier (jie_classifier.joblib)
  - Maintains rolling 60-s buffers for RIP/ECG/EDA
  - Every 5 s of acquisition, extracts 11 features -> classifier -> WS message
  - Emits both class prediction and a 0-100 continuous stress score
"""
import asyncio
import sys

# Windows: force SelectorEventLoop - ProactorEventLoop breaks uvicorn's WS impl.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import csv
import json
import platform
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from scipy.interpolate import interp1d
from joblib import load

# ----- Plux API -----
import os
REPO_ROOT = Path(__file__).resolve().parent.parent
_default_plux = REPO_ROOT / "PLUX-API-Python3" / f"Win64_{''.join(platform.python_version_tuple()[:2])}"
PLUX_API_DIR = Path(os.environ.get("PLUX_API_DIR", str(_default_plux)))
if not PLUX_API_DIR.is_dir():
    raise RuntimeError(
        f"PLUX API not found at {PLUX_API_DIR}. "
        f"Set PLUX_API_DIR env var to your plux.pyd directory."
    )
sys.path.append(str(PLUX_API_DIR))
import plux

# ----- Config -----
# Override with: $env:PLUX_ADDRESS = "BTH00:07:80:XX:XX:XX"
ADDRESS = os.environ.get("PLUX_ADDRESS", "BTH00:07:80:8C:AD:B3")
DURATION_SEC = 300
SAMPLE_RATE = 1000
RES_BITS = 16
CODE = 0x07
CHANNELS = ["RIP", "ECG", "EDA"]

UI_DECIMATE = 10
BATCH_SIZE = 20
RESP_WIN_SEC = 30
RESP_CHECK_EVERY_SEC = 1
CV_REGULAR = 0.25

# Classifier
CLF_WIN_SEC = 60                   # must match training
CLF_CHECK_EVERY_SEC = 5            # run classifier every 5 s

OUTPUT_DIR = REPO_ROOT / "captures"
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC = Path(__file__).parent / "static"

# ----- Load classifier + thresholds -----
CLF_PATH = REPO_ROOT / "models" / "jie_classifier.joblib"
THR_PATH = REPO_ROOT / "models" / "jie_thresholds.json"
if CLF_PATH.exists():
    BUNDLE = load(CLF_PATH)
    CLF_MODEL = BUNDLE["model"]
    CLF_SCALER = BUNDLE["scaler"]
    CLF_FEATURES = BUNDLE["feature_cols"]
    # Use clf.classes_ (alphabetical), NOT bundle["classes"] - sklearn reorders
    CLF_CLASSES = list(CLF_MODEL.classes_)
    print(f"[server] loaded classifier ({len(CLF_FEATURES)} feats, classes={CLF_CLASSES})")
else:
    BUNDLE = None
    print("[server] no classifier - running without stress prediction")

if THR_PATH.exists():
    with open(THR_PATH, "r", encoding="utf-8") as f:
        THRESHOLDS = json.load(f)
else:
    THRESHOLDS = {}

state = {
    "running": False,
    "csv_path": None,
    "samples": 0,
    "loop": None,
    "queue": None,
    "resp_text": "Waiting to start",
    "resp_color": "gray",
    "stress_text": "Waiting for 60 seconds before classification",
    "stress_score": 0.0,
    "stress_class": "warming",
}


# ============ Feature extraction (must match training pipeline) ============
def features_window(rip, ecg, eda_us, fs=SAMPLE_RATE):
    out = {}
    sos = butter(4, [5, 40], btype="band", fs=fs, output="sos")
    ecg_bp = sosfiltfilt(sos, ecg - ecg.mean())
    rpeaks, _ = find_peaks(ecg_bp, distance=int(fs * 0.4),
                           prominence=ecg_bp.std() * 2.0)
    rr = np.diff(rpeaks) / fs * 1000.0
    rr = rr[(rr > 400) & (rr < 1500)]
    if len(rr) > 3:
        out["mean_HR"] = float(60000.0 / rr.mean())
        out["RMSSD"] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        out["SDNN"] = float(rr.std())
        t_cum = np.cumsum(rr) / 1000.0
        if t_cum[-1] > 25:
            t_uni = np.arange(t_cum[0], t_cum[-1], 0.25)
            rr_uni = interp1d(t_cum, rr, kind="linear",
                              fill_value="extrapolate")(t_uni)
            nperseg = min(64, len(rr_uni))
            f, P = welch(rr_uni - rr_uni.mean(), fs=4.0, nperseg=nperseg)
            lf = float(P[(f >= 0.04) & (f < 0.15)].sum())
            hf = float(P[(f >= 0.15) & (f < 0.4)].sum())
            out["LF_HF"] = lf / (hf + 1e-9)
        else:
            out["LF_HF"] = 0.0
    else:
        out["mean_HR"] = out["RMSSD"] = out["SDNN"] = out["LF_HF"] = 0.0

    sos_lp = butter(4, 3.0, btype="low", fs=fs, output="sos")
    smooth = sosfiltfilt(sos_lp, eda_us)
    sos_t = butter(2, 0.05, btype="low", fs=fs, output="sos")
    tonic = sosfiltfilt(sos_t, smooth)
    phasic = smooth - tonic
    dur_min = len(smooth) / fs / 60.0
    out["SCL_mean"] = float(tonic.mean())
    out["SCL_slope"] = float((tonic[-1] - tonic[0]) / dur_min)
    scr_peaks, props = find_peaks(phasic, distance=int(fs * 1.0),
                                  prominence=0.02, height=0.03)
    out["SCR_rate"] = float(len(scr_peaks) / dur_min)
    out["SCR_amp_mean"] = float(np.mean(props.get("peak_heights",
                                                  [0]))) if len(scr_peaks) else 0.0

    dc = rip - rip.mean()
    sos_r = butter(4, 1.0, btype="low", fs=fs, output="sos")
    lp = sosfiltfilt(sos_r, dc)
    rp, _ = find_peaks(lp, distance=int(fs * 1.5), prominence=lp.std() * 0.4)
    rt, _ = find_peaks(-lp, distance=int(fs * 1.5), prominence=lp.std() * 0.4)
    if len(rp) > 1:
        iv = np.diff(rp) / fs
        out["resp_rate"] = float(60.0 / iv.mean())
        out["resp_CV"] = float(iv.std() / iv.mean())
    else:
        out["resp_rate"] = out["resp_CV"] = 0.0
    if len(rp) >= 2 and len(rt) >= 2:
        n = min(len(rp), len(rt))
        amps = lp[rp[:n]] - lp[rt[:n]]
        out["resp_amp_std"] = float(amps.std())
    else:
        out["resp_amp_std"] = 0.0
    return out


def classify(feats):
    """Return dict with predicted class, probs, and 0-100 stress score."""
    if BUNDLE is None:
        return None
    x = np.array([[feats[c] for c in CLF_FEATURES]])
    xs = CLF_SCALER.transform(x)
    probs = CLF_MODEL.predict_proba(xs)[0]
    # CLF_CLASSES is alphabetical order from clf.classes_
    prob_map = dict(zip(CLF_CLASSES, [float(p) for p in probs]))
    pred = max(prob_map, key=prob_map.get)
    weights = {"stable": 0.0, "middle": 50.0, "mess": 100.0}
    score = float(sum(prob_map.get(c, 0.0) * weights[c] for c in weights))
    label_text = {"stable": "Calm", "middle": "Concern", "mess": "Anxious"}[pred]
    color = {"stable": "green", "middle": "amber", "mess": "red"}[pred]
    return {
        "class": pred,
        "label": label_text,
        "color": color,
        "score": round(score, 1),
        "probs": {c: round(prob_map.get(c, 0.0), 3) for c in ["stable", "middle", "mess"]},
        "features": {k: round(v, 3) for k, v in feats.items()},
    }


# ============ Device wrapper ============
class Device(plux.SignalsDev):
    def __init__(self, address):
        plux.MemoryDev.__init__(address)

    def setup(self, csv_file, q, event_loop, target):
        self.writer = csv.writer(csv_file)
        self.writer.writerow(["nSeq", "t_sec", *CHANNELS])
        self.csv_file = csv_file
        self.q = q
        self.event_loop = event_loop
        self.target = target
        self.ui_batch = []
        # buffers
        self.rip_resp = deque(maxlen=RESP_WIN_SEC * SAMPLE_RATE)
        self.rip_clf = deque(maxlen=CLF_WIN_SEC * SAMPLE_RATE)
        self.ecg_clf = deque(maxlen=CLF_WIN_SEC * SAMPLE_RATE)
        self.eda_clf = deque(maxlen=CLF_WIN_SEC * SAMPLE_RATE)
        self.last_resp = -SAMPLE_RATE
        self.last_clf = -SAMPLE_RATE

    def onRawFrame(self, nSeq, data):
        t = nSeq / SAMPLE_RATE
        self.writer.writerow([nSeq, f"{t:.4f}", *data])
        rip_v, ecg_v, eda_v = data[0], data[1], data[2]
        self.rip_resp.append(rip_v)
        self.rip_clf.append(rip_v)
        self.ecg_clf.append(ecg_v)
        self.eda_clf.append(eda_v)

        if nSeq % UI_DECIMATE == 0:
            self.ui_batch.append([nSeq, list(data)])
            if len(self.ui_batch) >= BATCH_SIZE:
                self.event_loop.call_soon_threadsafe(
                    self.q.put_nowait, {"type": "data", "samples": self.ui_batch}
                )
                self.ui_batch = []

        # Respiration regularity
        if nSeq - self.last_resp >= RESP_CHECK_EVERY_SEC * SAMPLE_RATE:
            self.last_resp = nSeq
            v = compute_resp(self.rip_resp)
            state["resp_text"] = v["text"]
            state["resp_color"] = v["color"]
            self.event_loop.call_soon_threadsafe(
                self.q.put_nowait, {"type": "resp", **v}
            )

        # Classifier every CLF_CHECK_EVERY_SEC after we have full window
        if (BUNDLE is not None
                and len(self.rip_clf) >= CLF_WIN_SEC * SAMPLE_RATE
                and nSeq - self.last_clf >= CLF_CHECK_EVERY_SEC * SAMPLE_RATE):
            self.last_clf = nSeq
            rip_arr = np.array(self.rip_clf, dtype=float)
            ecg_arr = np.array(self.ecg_clf, dtype=float)
            eda_arr = np.array(self.eda_clf, dtype=float)
            eda_us = (eda_arr / 65536.0) * 3.0 / 0.12
            try:
                feats = features_window(rip_arr, ecg_arr, eda_us)
                res = classify(feats)
                if res:
                    state["stress_text"] = f"{res['label']}  ({res['score']:.0f}/100)"
                    state["stress_score"] = res["score"]
                    state["stress_class"] = res["class"]
                    self.event_loop.call_soon_threadsafe(
                        self.q.put_nowait, {"type": "stress", **res}
                    )
            except Exception as exc:
                print(f"[classify] {type(exc).__name__}: {exc}", flush=True)

        if nSeq % SAMPLE_RATE == 0:
            self.csv_file.flush()
        state["samples"] = nSeq
        return nSeq >= self.target


def compute_resp(rip_samples):
    n = len(rip_samples)
    if n < 10 * SAMPLE_RATE:
        return {"text": f"Collecting... ({n // SAMPLE_RATE}/10s of data)", "color": "gray"}
    x = np.fromiter(rip_samples, dtype=float)
    x = x - x.mean()
    sos = butter(3, 1.0, btype="low", fs=SAMPLE_RATE, output="sos")
    x = sosfiltfilt(sos, x)
    peaks, _ = find_peaks(x, distance=int(SAMPLE_RATE * 1.2),
                          prominence=x.std() * 0.4)
    if len(peaks) < 4:
        return {"text": "Not enough breathing cycles; keep collecting...", "color": "gray"}
    intervals = np.diff(peaks) / SAMPLE_RATE
    cv = float(intervals.std() / intervals.mean())
    rate = float(60.0 / intervals.mean())
    if cv < CV_REGULAR:
        return {"text": f"Regular breathing ({rate:.0f} breaths/min,CV={cv:.2f})", "color": "green"}
    return {"text": f"Irregular breathing ({rate:.0f} breaths/min,CV={cv:.2f})", "color": "red"}


def acquire(q, loop):
    import traceback
    csv_path = OUTPUT_DIR / f"plux5min_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    state["csv_path"] = str(csv_path)
    target = DURATION_SEC * SAMPLE_RATE
    print(f"[acquire] starting -> {csv_path}", flush=True)
    try:
        with open(csv_path, "w", newline="") as f:
            dev = Device(ADDRESS)
            dev.setup(f, q, loop, target)
            dev.start(SAMPLE_RATE, CODE, RES_BITS)
            dev.loop()
            dev.stop()
            dev.close()
        print(f"[acquire] complete -> {csv_path}", flush=True)
        loop.call_soon_threadsafe(
            q.put_nowait, {"type": "done", "csv_path": str(csv_path)}
        )
    except Exception as e:
        print(f"[acquire] EXCEPTION:\n{traceback.format_exc()}", flush=True)
        loop.call_soon_threadsafe(
            q.put_nowait, {"type": "error", "msg": f"{type(e).__name__}: {e}"}
        )
    finally:
        state["running"] = False


# ----- FastAPI -----
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/thresholds")
async def thresholds():
    return THRESHOLDS


@app.post("/start")
async def start():
    if state["running"]:
        return {"ok": False, "msg": "Already running"}
    state["running"] = True
    state["samples"] = 0
    state["csv_path"] = None
    state["stress_text"] = "Waiting for 60 seconds before classification"
    state["stress_score"] = 0.0
    state["stress_class"] = "warming"
    state["loop"] = asyncio.get_running_loop()
    state["queue"] = asyncio.Queue()
    threading.Thread(
        target=acquire, args=(state["queue"], state["loop"]), daemon=True
    ).start()
    return {"ok": True}


@app.get("/status")
async def status():
    return {
        "running": state["running"],
        "samples": state["samples"],
        "target": DURATION_SEC * SAMPLE_RATE,
        "duration_sec": DURATION_SEC,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "csv_path": state["csv_path"],
        "resp_text": state["resp_text"],
        "resp_color": state["resp_color"],
        "stress_text": state["stress_text"],
        "stress_score": state["stress_score"],
        "stress_class": state["stress_class"],
        "classifier_loaded": BUNDLE is not None,
    }


@app.get("/download")
async def download():
    p = state["csv_path"]
    if p and Path(p).exists():
        return FileResponse(p, filename=Path(p).name, media_type="text/csv")
    return {"error": "no csv yet"}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            q = state["queue"]
            if q is None:
                await asyncio.sleep(0.1)
                continue
            try:
                msg = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping", "samples": state["samples"]})
                continue
            await ws.send_json(msg)
            if msg.get("type") in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    print("===================================================")
    print(" PLUX 5-min WebUI + Stress Classifier")
    print(" Open: http://127.0.0.1:8000")
    print("===================================================")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
