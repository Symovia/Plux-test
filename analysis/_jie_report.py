"""Comprehensive stress-gradient physiological report for jie.

Analyzes 3 conditions: stable, middle, and mess.
  - ECG  -> R-peaks -> HR, HRV time-domain (SDNN/RMSSD/pNN50), freq-domain (LF/HF)
  - EDA  -> tonic (SCL) + phasic (SCR) decomposition, SCR detection
  - RIP  -> breath detection, rate, regularity, amplitude
  - Composite stress index across conditions
"""
import json
import sys
from pathlib import Path

# Windows console defaults to GBK; force UTF-8 so µ etc. print correctly
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from scipy.interpolate import interp1d

# Use a Chinese-capable font (Windows)
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "data" / "jie"
OUT = REPO_ROOT / "output"
OUT.mkdir(exist_ok=True)

FS = 1000
CONDS = ["stable", "middle", "mess"]
CN = {"stable": "Calm", "middle": "Concern", "mess": "Anxious"}
COLORS = {"stable": "#10b981", "middle": "#f59e0b", "mess": "#ef4444"}


# ---------- ECG ----------
def analyze_ecg(ecg, fs=FS):
    # Bandpass 5-40 Hz to isolate QRS
    sos = butter(4, [5, 40], btype="band", fs=fs, output="sos")
    ecg_bp = sosfiltfilt(sos, ecg - ecg.mean())
    # Peak detection
    rpeaks, _ = find_peaks(
        ecg_bp,
        distance=int(fs * 0.4),  # min 400ms (HR < 150)
        prominence=ecg_bp.std() * 2.0,
    )
    rr_all = np.diff(rpeaks) / fs * 1000.0  # ms
    # Clean: physiological range + reject sudden jumps
    rr = rr_all[(rr_all > 400) & (rr_all < 1500)]
    # Time-domain HRV
    sdnn = float(np.std(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    pnn50 = float(np.sum(np.abs(np.diff(rr)) > 50) / max(1, len(rr) - 1) * 100)
    mean_hr = float(60000.0 / rr.mean())
    # Frequency-domain HRV
    t_cum = np.cumsum(rr) / 1000.0
    fs_re = 4.0
    t_uni = np.arange(t_cum[0], t_cum[-1], 1.0 / fs_re)
    rr_uni = interp1d(t_cum, rr, kind="cubic", fill_value="extrapolate")(t_uni)
    nperseg = min(256, len(rr_uni))
    f, P = welch(rr_uni - rr_uni.mean(), fs=fs_re, nperseg=nperseg)
    lf = float(P[(f >= 0.04) & (f < 0.15)].sum())
    hf = float(P[(f >= 0.15) & (f < 0.4)].sum())
    lf_hf = lf / (hf + 1e-9)
    # Instantaneous HR series
    t_peaks = rpeaks[1:] / fs
    inst_hr = 60000.0 / rr_all
    return {
        "rpeaks": rpeaks,
        "ecg_bp": ecg_bp,
        "rr": rr,
        "mean_hr": mean_hr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "lf": lf,
        "hf": hf,
        "lf_hf": lf_hf,
        "t_peaks": t_peaks,
        "inst_hr": inst_hr,
    }


# ---------- EDA ----------
def analyze_eda(eda_adc, fs=FS):
    eda_us = (eda_adc.astype(float) / 65536.0) * 3.0 / 0.12
    # Smooth to 3 Hz (datasheet bandwidth)
    sos = butter(4, 3.0, btype="low", fs=fs, output="sos")
    smooth = sosfiltfilt(sos, eda_us)
    # Tonic = very slow baseline (0.05 Hz)
    sos2 = butter(2, 0.05, btype="low", fs=fs, output="sos")
    tonic = sosfiltfilt(sos2, smooth)
    phasic = smooth - tonic
    # SCR detection on phasic
    scr_peaks, props = find_peaks(
        phasic,
        distance=int(fs * 1.0),
        prominence=0.02,
        height=0.03,
    )
    amps = props.get("peak_heights", np.array([]))
    duration_min = len(smooth) / fs / 60.0
    return {
        "us": smooth,
        "tonic": tonic,
        "phasic": phasic,
        "scl_mean": float(smooth.mean()),
        "scl_slope_per_min": float((tonic[-1] - tonic[0]) / duration_min),
        "range_us": float(smooth.max() - smooth.min()),
        "scr_count": int(len(scr_peaks)),
        "scr_rate_per_min": float(len(scr_peaks) / duration_min),
        "scr_mean_amp": float(amps.mean()) if len(amps) else 0.0,
        "scr_peaks": scr_peaks,
    }


# ---------- RIP ----------
def analyze_rip(rip, fs=FS):
    x = rip - rip.mean()
    sos = butter(4, 1.0, btype="low", fs=fs, output="sos")
    x_lp = sosfiltfilt(sos, x)
    peaks, _ = find_peaks(
        x_lp, distance=int(fs * 1.5), prominence=x_lp.std() * 0.4
    )
    troughs, _ = find_peaks(
        -x_lp, distance=int(fs * 1.5), prominence=x_lp.std() * 0.4
    )
    intervals = np.diff(peaks) / fs
    rate = 60.0 / intervals.mean()
    cv = float(intervals.std() / intervals.mean())
    # Tidal amplitude (peak - trough)
    amp = float(x_lp[peaks].mean() - x_lp[troughs].mean()) if len(troughs) else 0
    return {
        "signal_lp": x_lp,
        "peaks": peaks,
        "troughs": troughs,
        "rate_bpm": rate,
        "cv": cv,
        "n_breaths": len(peaks),
        "amplitude": amp,
    }


# ---------- Main pipeline ----------
def run():
    results = {}
    for c in CONDS:
        print(f"  analyzing jie/{c} ...", flush=True)
        df = pd.read_csv(ROOT / f"{c}.csv")
        results[c] = {
            "ecg": analyze_ecg(df["ECG"].values.astype(float)),
            "eda": analyze_eda(df["EDA"].values),
            "rip": analyze_rip(df["RIP"].values.astype(float)),
            "t": df["t_sec"].values,
        }

    # ----- Numeric summary -----
    rows = []
    for c in CONDS:
        e = results[c]["ecg"]
        a = results[c]["eda"]
        r = results[c]["rip"]
        rows.append({
            "condition": f"{c} ({CN[c]})",
            "HR_bpm": round(e["mean_hr"], 1),
            "SDNN_ms": round(e["sdnn"], 1),
            "RMSSD_ms": round(e["rmssd"], 1),
            "pNN50_%": round(e["pnn50"], 1),
            "LF/HF": round(e["lf_hf"], 2),
            "SCL_µS": round(a["scl_mean"], 2),
            "SCL_slope_µS/min": round(a["scl_slope_per_min"], 3),
            "SCR/min": round(a["scr_rate_per_min"], 1),
            "SCR_amp_µS": round(a["scr_mean_amp"], 3),
            "EDA_range_µS": round(a["range_us"], 2),
            "Resp_bpm": round(r["rate_bpm"], 1),
            "Resp_CV": round(r["cv"], 3),
            "Tidal_amp": round(r["amplitude"], 0),
        })
    summary = pd.DataFrame(rows)
    print("\n===== PER-CONDITION METRICS =====")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False))
    summary.to_csv(OUT / "jie_metrics.csv", index=False)

    # ----- Composite stress index -----
    # Direction: higher z = more stress
    #   + HR, EDA, RIP_CV, LF/HF, SCR rate
    #   - RMSSD, SDNN
    metrics_for_score = {
        "HR_bpm": +1,
        "RMSSD_ms": -1,
        "SDNN_ms": -1,
        "LF/HF": +1,
        "SCL_µS": +1,
        "SCR/min": +1,
        "Resp_CV": +1,
    }
    arr = summary.set_index("condition")
    score = pd.DataFrame(index=arr.index)
    for col, sign in metrics_for_score.items():
        z = (arr[col] - arr[col].mean()) / (arr[col].std() + 1e-9)
        score[col] = sign * z
    score["stress_index"] = score.sum(axis=1).round(2)
    print("\n===== COMPOSITE STRESS INDEX (higher = more stressed) =====")
    print(score[["stress_index"]].to_string())

    # ----- VISUALIZATION -----
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(5, 3, height_ratios=[1.3, 1.3, 1.3, 1.0, 1.0],
                          hspace=0.45, wspace=0.25)

    # Top 3 rows: RIP, ECG with R-peaks, EDA tonic+phasic for each condition
    for i, c in enumerate(CONDS):
        col = COLORS[c]
        e = results[c]["ecg"]
        a = results[c]["eda"]
        r = results[c]["rip"]
        t = results[c]["t"]

        ax = fig.add_subplot(gs[0, i])
        ax.plot(t, r["signal_lp"], lw=0.6, color=col)
        ax.plot(r["peaks"] / FS, r["signal_lp"][r["peaks"]], "v", ms=4, mfc="white", mec=col)
        ax.set_title(f"{c} - {CN[c]}", fontsize=13, fontweight="bold", color=col)
        ax.set_ylabel("RIP (centered)")
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, i])
        # Plot one 6-sec window of ECG to see QRS shape, not full 5 min
        win = slice(60 * FS, 66 * FS)
        ax.plot(t[win], e["ecg_bp"][win], lw=0.5, color=col)
        r_in_win = e["rpeaks"][(e["rpeaks"] >= win.start) & (e["rpeaks"] < win.stop)]
        ax.plot(r_in_win / FS, e["ecg_bp"][r_in_win], "o", ms=4, mfc=col, mec="black")
        ax.set_ylabel("ECG (5-40 Hz)")
        ax.set_title(f"6s window @ 60s  |  HR={e['mean_hr']:.0f} bpm", fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[2, i])
        ax.plot(t, a["us"], lw=0.4, color="#999", alpha=0.5, label="raw (µS)")
        ax.plot(t, a["tonic"], lw=1.8, color=col, label="tonic (SCL)")
        ax.plot(a["scr_peaks"] / FS, a["us"][a["scr_peaks"]], "v",
                ms=5, mfc="white", mec=col, label=f"SCR ({a['scr_count']})")
        ax.set_ylabel("EDA µS")
        ax.set_title(f"SCL={a['scl_mean']:.2f} µS  |  SCRs={a['scr_count']}", fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

    # Row 4: bar metrics
    bar_axes = [fig.add_subplot(gs[3, j]) for j in range(3)]
    conditions_labels = [f"{c}\n{CN[c]}" for c in CONDS]
    colors_list = [COLORS[c] for c in CONDS]

    bar_axes[0].bar(conditions_labels, [results[c]["ecg"]["mean_hr"] for c in CONDS],
                    color=colors_list)
    bar_axes[0].set_title("HR (bpm)")
    bar_axes[0].grid(True, axis="y", alpha=0.3)

    bar_axes[1].bar(conditions_labels, [results[c]["ecg"]["rmssd"] for c in CONDS],
                    color=colors_list)
    bar_axes[1].set_title("RMSSD (ms) - higher = more parasympathetic")
    bar_axes[1].grid(True, axis="y", alpha=0.3)

    bar_axes[2].bar(conditions_labels, [results[c]["ecg"]["lf_hf"] for c in CONDS],
                    color=colors_list)
    bar_axes[2].set_title("LF/HF ratio - higher = more sympathetic")
    bar_axes[2].grid(True, axis="y", alpha=0.3)

    # Row 5: more bars + stress index
    bar_axes2 = [fig.add_subplot(gs[4, j]) for j in range(3)]

    bar_axes2[0].bar(conditions_labels, [results[c]["eda"]["scl_mean"] for c in CONDS],
                     color=colors_list)
    bar_axes2[0].set_title("EDA tonic level SCL (µS)")
    bar_axes2[0].grid(True, axis="y", alpha=0.3)

    bar_axes2[1].bar(conditions_labels, [results[c]["eda"]["scr_rate_per_min"] for c in CONDS],
                     color=colors_list)
    bar_axes2[1].set_title("SCR rate (/min)")
    bar_axes2[1].grid(True, axis="y", alpha=0.3)

    bar_axes2[2].bar(conditions_labels,
                     [score["stress_index"][f"{c} ({CN[c]})"] for c in CONDS],
                     color=colors_list)
    bar_axes2[2].set_title("🔥 Composite Stress Index (higher = more stressed)",
                           fontweight="bold")
    bar_axes2[2].grid(True, axis="y", alpha=0.3)
    bar_axes2[2].axhline(0, color="#666", lw=0.8)

    fig.suptitle("jie - Stress Gradient Physiological Report  (5min x 3 conditions)",
                 fontsize=15, fontweight="bold", y=0.995)
    out_png = OUT / "jie_stress_report.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"\nPNG saved: {out_png}")

    # JSON dump (without arrays) for completeness
    json_summary = {
        "metrics": rows,
        "composite_score": score["stress_index"].to_dict(),
    }
    with open(OUT / "jie_report.json", "w", encoding="utf-8") as f:
        json.dump(json_summary, f, ensure_ascii=False, indent=2)
    print(f"JSON saved: {OUT / 'jie_report.json'}")


if __name__ == "__main__":
    run()
