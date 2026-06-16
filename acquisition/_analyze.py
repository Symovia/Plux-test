"""Classify each captured channel by waveform features."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks, butter, sosfiltfilt


VCC = 3.0
ADC_BITS = 16


def adc_to_eda_us(adc):
    """Datasheet: EDA(µS) = (ADC / 2^n) * VCC / 0.12"""
    return (np.asarray(adc, dtype=float) / (2 ** ADC_BITS)) * VCC / 0.12


def lowpass(x, fs, cutoff=3.0, order=4):
    sos = butter(order, cutoff, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def classify_channel(x, fs):
    x = np.asarray(x, dtype=float)
    n = len(x)
    duration = n / fs
    mean = float(x.mean())
    std = float(x.std())

    nperseg = min(8192, n // 4)
    f, Pxx = welch(x - mean, fs=fs, nperseg=nperseg)
    mask = (f >= 0.05) & (f <= 50)
    if mask.any():
        dom_freq = float(f[mask][np.argmax(Pxx[mask])])
    else:
        dom_freq = 0.0

    band_ecg = (f >= 0.7) & (f <= 5.0)
    band_resp = (f >= 0.1) & (f <= 0.5)
    band_eda = (f >= 0.005) & (f < 0.1)
    p_ecg = float(Pxx[band_ecg].sum())
    p_resp = float(Pxx[band_resp].sum())
    p_eda = float(Pxx[band_eda].sum())

    peaks, _ = find_peaks(x, prominence=std * 0.5, distance=int(fs * 0.3))
    peak_rate = len(peaks) / duration

    if peak_rate >= 0.7 and p_ecg >= max(p_resp, p_eda):
        verdict = "ECG"
    elif 0.15 <= dom_freq <= 0.5 and p_resp >= max(p_ecg, p_eda) * 0.5:
        verdict = "RESP/RIP"
    elif dom_freq < 0.15 and p_eda >= p_ecg:
        verdict = "EDA"
    else:
        verdict = "UNCLEAR"

    return {
        "verdict": verdict,
        "mean": mean,
        "std": std,
        "p2p": float(x.max() - x.min()),
        "dom_freq_hz": dom_freq,
        "peak_rate_per_sec": peak_rate,
        "p_ecg_0.7_5Hz": p_ecg,
        "p_resp_0.1_0.5Hz": p_resp,
        "p_eda_0.005_0.1Hz": p_eda,
    }


def main():
    captures = Path(__file__).resolve().parent.parent / "captures"
    csvs = sorted(captures.glob("plux_*.csv"))
    if not csvs:
        print("No captures found.")
        sys.exit(1)
    csv_path = csvs[-1]
    print(f"Loading: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}  Columns: {list(df.columns)}")
    fs = int(round(1.0 / (df["t_sec"].iloc[1] - df["t_sec"].iloc[0])))
    print(f"Sample rate: {fs} Hz\n")

    ch_cols = [c for c in df.columns if c not in ("nSeq", "t_sec")]
    fig, axes = plt.subplots(len(ch_cols), 1, figsize=(13, 8), sharex=True)
    if not hasattr(axes, "__iter__"):
        axes = [axes]

    results = []
    for ax, col in zip(axes, ch_cols):
        raw = df[col].values
        stats = classify_channel(raw, fs)
        results.append((col, stats))

        if "EDA" in col.upper():
            us_raw = adc_to_eda_us(raw)
            us_filt = lowpass(us_raw, fs, cutoff=3.0)
            ax.plot(df["t_sec"], us_raw, lw=0.5, alpha=0.3, color="C0", label="raw")
            ax.plot(df["t_sec"], us_filt, lw=1.2, color="C3", label="3 Hz low-pass")
            ax.set_ylabel(f"{col}\n[µS]")
            ax.legend(loc="upper right", fontsize=8)
            stats["mean_us"] = float(us_filt.mean())
            stats["p2p_us"] = float(us_filt.max() - us_filt.min())
        else:
            ax.plot(df["t_sec"], raw, lw=0.6)
            ax.set_ylabel(col)

        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"verdict={stats['verdict']}   "
            f"peak_rate={stats['peak_rate_per_sec']:.2f}/s   "
            f"dom_freq={stats['dom_freq_hz']:.2f} Hz   "
            f"std={stats['std']:.0f}"
        )
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Channel analysis — {csv_path.name}")
    fig.tight_layout()

    png_path = csv_path.with_suffix(".png")
    fig.savefig(png_path, dpi=110)
    print(f"PNG saved: {png_path}\n")

    for col, stats in results:
        print(f"=== {col} ===")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k:<22} {v:>14.4f}")
            else:
                print(f"  {k:<22} {v}")
        print()


if __name__ == "__main__":
    main()
