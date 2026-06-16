"""Compare data quality across 2 subjects x 3 conditions (jie, ziqi x stable/middle/mess)."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks, welch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
OUT = REPO_ROOT / "output"
OUT.mkdir(exist_ok=True)
FS = 1000
SUBJECTS = ["jie", "ziqi"]
CONDITIONS = ["stable", "middle", "mess"]


def lpf(x, cutoff, order=4):
    sos = butter(order, cutoff, btype="low", fs=FS, output="sos")
    return sosfiltfilt(sos, x)


def bpf(x, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, x)


def metrics(df):
    rip = df["RIP"].values.astype(float)
    ecg = df["ECG"].values.astype(float)
    eda = df["EDA"].values.astype(float)
    n = len(rip)
    dur_min = n / FS / 60

    # ----- RIP -----
    rip_dc = rip - rip.mean()
    rip_lp = lpf(rip_dc, 1.0)
    rip_peaks, _ = find_peaks(rip_lp, distance=int(FS * 1.5),
                              prominence=rip_lp.std() * 0.4)
    rip_rate = len(rip_peaks) / dur_min
    if len(rip_peaks) > 3:
        rr_iv = np.diff(rip_peaks) / FS
        rip_cv = float(rr_iv.std() / rr_iv.mean())
    else:
        rip_cv = float("nan")

    f, Pxx = welch(rip_dc, fs=FS, nperseg=8192)
    p_resp = Pxx[(f >= 0.1) & (f <= 0.5)].sum()
    p_hf = Pxx[f >= 5].sum() + 1e-12
    rip_snr_db = 10 * np.log10(p_resp / p_hf)

    # ----- ECG -----
    ecg_dc = ecg - ecg.mean()
    ecg_bp = bpf(ecg_dc, 5, 40)
    ecg_peaks, _ = find_peaks(ecg_bp, distance=int(FS * 0.4),
                              prominence=ecg_bp.std() * 1.5)
    hr = len(ecg_peaks) / dur_min
    if len(ecg_peaks) > 5:
        rr = np.diff(ecg_peaks) / FS * 1000  # ms
        # Drop physiologically impossible RR (<300ms or >2000ms) for HRV
        rr_ok = rr[(rr > 300) & (rr < 2000)]
        if len(rr_ok) > 3:
            rmssd = float(np.sqrt(np.mean(np.diff(rr_ok) ** 2)))
            sdnn = float(rr_ok.std())
        else:
            rmssd = sdnn = float("nan")
    else:
        rmssd = sdnn = float("nan")

    # ECG motion artifacts: count windows with extreme z-scored amplitude
    z = np.abs((ecg - ecg.mean()) / ecg.std())
    artifacts = int((z > 6).sum())

    f, Pxx = welch(ecg_dc, fs=FS, nperseg=8192)
    p_qrs = Pxx[(f >= 5) & (f <= 40)].sum()
    p_hf_ecg = Pxx[f >= 100].sum() + 1e-12
    ecg_snr_db = 10 * np.log10(p_qrs / p_hf_ecg)

    # ----- EDA -----
    eda_us = (eda / 65536.0) * 3.0 / 0.12
    eda_lp = lpf(eda_us, 3.0)
    eda_mean = float(eda_lp.mean())
    eda_range = float(eda_lp.max() - eda_lp.min())
    eda_std = float(eda_lp.std())

    f, Pxx = welch(eda_us - eda_us.mean(), fs=FS, nperseg=8192)
    p_sig = Pxx[f <= 3].sum()
    p_noise = Pxx[f > 3].sum()
    eda_noise_pct = 100 * p_noise / (p_sig + p_noise + 1e-12)

    return {
        "duration_min": round(dur_min, 2),
        "rip_breaths": len(rip_peaks),
        "rip_rate_bpm": round(rip_rate, 1),
        "rip_cv": round(rip_cv, 3) if not np.isnan(rip_cv) else None,
        "rip_snr_db": round(rip_snr_db, 1),
        "ecg_beats": len(ecg_peaks),
        "hr_bpm": round(hr, 1),
        "rmssd_ms": round(rmssd, 1) if not np.isnan(rmssd) else None,
        "sdnn_ms": round(sdnn, 1) if not np.isnan(sdnn) else None,
        "ecg_artifacts": artifacts,
        "ecg_snr_db": round(ecg_snr_db, 1),
        "eda_mean_us": round(eda_mean, 2),
        "eda_range_us": round(eda_range, 2),
        "eda_std_us": round(eda_std, 3),
        "eda_noise_pct": round(eda_noise_pct, 1),
    }


def main():
    rows = []
    print("Loading + analyzing 6 files...")
    for s in SUBJECTS:
        for c in CONDITIONS:
            path = DATA / s / f"{c}.csv"
            print(f"  {s}/{c}.csv ...", flush=True)
            df = pd.read_csv(path)
            m = metrics(df)
            rows.append({"subject": s, "condition": c, **m})

    res = pd.DataFrame(rows)
    print("\n===== FULL METRICS =====")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False))

    # Aggregate per-subject quality score
    # Higher is better: ecg_snr_db, rip_snr_db
    # Lower is better: ecg_artifacts, eda_noise_pct
    print("\n===== PER-SUBJECT AGGREGATE =====")
    agg_cols = ["ecg_snr_db", "rip_snr_db", "ecg_artifacts", "eda_noise_pct",
                "eda_std_us", "rmssd_ms"]
    agg = res.groupby("subject")[agg_cols].mean().round(2)
    print(agg.to_string())

    # Composite quality score (z-normalize each metric, sign by "higher is better")
    weights = {
        "ecg_snr_db": +1.0,    # higher = cleaner ECG
        "rip_snr_db": +1.0,    # higher = cleaner RIP
        "ecg_artifacts": -1.0, # lower = better (fewer motion events)
        "eda_noise_pct": -0.5, # lower = better (less high-freq noise)
    }
    score = pd.DataFrame(index=agg.index)
    for col, w in weights.items():
        vals = agg[col]
        z = (vals - vals.mean()) / (vals.std() + 1e-9)
        score[col] = w * z
    score["total"] = score.sum(axis=1).round(2)
    print("\n===== COMPOSITE QUALITY SCORE (higher = better) =====")
    print(score.to_string())

    # ----- Plot side-by-side -----
    fig, axes = plt.subplots(3, 6, figsize=(20, 9), sharex=True)
    for col_idx, (s, c) in enumerate([(s, c) for s in SUBJECTS for c in CONDITIONS]):
        df = pd.read_csv(DATA / s / f"{c}.csv")
        t = df["t_sec"].values
        rip = df["RIP"].values - df["RIP"].mean()
        ecg = df["ECG"].values - df["ECG"].mean()
        eda_us = (df["EDA"].values / 65536.0) * 3.0 / 0.12
        eda_lp = lpf(eda_us, 3.0)
        axes[0, col_idx].plot(t, rip, lw=0.4, color="#60a5fa")
        axes[0, col_idx].set_title(f"{s} / {c}", fontsize=10)
        axes[1, col_idx].plot(t, ecg, lw=0.3, color="#f87171")
        axes[2, col_idx].plot(t, eda_us, lw=0.3, color="#999", alpha=0.4)
        axes[2, col_idx].plot(t, eda_lp, lw=0.8, color="#10b981")
    axes[0, 0].set_ylabel("RIP (centered)")
    axes[1, 0].set_ylabel("ECG (centered)")
    axes[2, 0].set_ylabel("EDA µS")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("t (s)")
    fig.suptitle("jie vs ziqi  —  3 conditions side-by-side", fontsize=12)
    fig.tight_layout()
    out = OUT / "_comparison.png"
    fig.savefig(out, dpi=110)
    print(f"\nPNG saved: {out}")

    # CSV of metrics
    out_csv = OUT / "_metrics.csv"
    res.to_csv(out_csv, index=False)
    print(f"Metrics CSV: {out_csv}")


if __name__ == "__main__":
    main()
