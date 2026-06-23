"""5 physiological state estimators for jie's 3 CSVs:

  1. Fatigue index           - combo of HRV/HR/EDA suggesting low arousal + parasympathetic
  2. Cognitive load index    - mental engagement signature (LF/HF up, breath shallow)
  3. Autonomic / heart age   - HRV-derived biological age (log-fit norms)
  4. Cardiopulmonary coherence - HR <-> breath synchronization (RSA strength)
  5. Stress recovery time    - half-life of EDA SCR events (smaller = faster recovery)
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks, welch, coherence
from scipy.interpolate import interp1d

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
OUT = REPO_ROOT / "output"
OUT.mkdir(exist_ok=True)
FS = 1000
CONDS = ["stable", "middle", "mess"]
CN = {"stable": "Calm", "middle": "Concern", "mess": "Anxious"}
COLORS = {"stable": "#10b981", "middle": "#f59e0b", "mess": "#ef4444"}
# Try English filename first, fall back to Chinese rename
FILE_ALIASES = {
    "stable": ["stable.csv", "stable.csv", "Calm.csv"],
    "middle": ["middle.csv", "middle.csv", "Concern.csv"],
    "mess":   ["mess.csv", "mess.csv", "Anxious.csv"],
}


def resolve_csv(cond):
    for name in FILE_ALIASES[cond]:
        p = DATA / "jie" / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No CSV found for '{cond}' (tried {FILE_ALIASES[cond]})")


# ---------- shared signal prep ----------
def get_rr(ecg, fs=FS):
    sos = butter(4, [5, 40], btype="band", fs=fs, output="sos")
    ecg_bp = sosfiltfilt(sos, ecg - ecg.mean())
    rpeaks, _ = find_peaks(ecg_bp, distance=int(fs * 0.4),
                           prominence=ecg_bp.std() * 2.0)
    rr_ms = np.diff(rpeaks) / fs * 1000.0
    rr_ms = rr_ms[(rr_ms > 400) & (rr_ms < 1500)]
    return rpeaks, rr_ms


def hrv_stats(rr_ms):
    if len(rr_ms) < 5:
        return dict(HR=0, SDNN=0, RMSSD=0, LF_HF=0)
    sdnn = float(np.std(rr_ms))
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    hr = float(60000 / rr_ms.mean())
    t = np.cumsum(rr_ms) / 1000
    fs_re = 4.0
    t_uni = np.arange(t[0], t[-1], 1.0 / fs_re)
    rr_uni = interp1d(t, rr_ms, kind="cubic",
                      fill_value="extrapolate")(t_uni)
    f, P = welch(rr_uni - rr_uni.mean(), fs=fs_re,
                 nperseg=min(256, len(rr_uni)))
    lf = float(P[(f >= 0.04) & (f < 0.15)].sum())
    hf = float(P[(f >= 0.15) & (f < 0.4)].sum())
    return dict(HR=hr, SDNN=sdnn, RMSSD=rmssd, LF_HF=lf / (hf + 1e-9))


def eda_split(eda_us, fs=FS):
    sos = butter(4, 3.0, btype="low", fs=fs, output="sos")
    smooth = sosfiltfilt(sos, eda_us)
    sos2 = butter(2, 0.05, btype="low", fs=fs, output="sos")
    tonic = sosfiltfilt(sos2, smooth)
    return smooth, tonic, smooth - tonic


def rip_stats(rip, fs=FS):
    dc = rip - rip.mean()
    sos = butter(4, 1.0, btype="low", fs=fs, output="sos")
    lp = sosfiltfilt(sos, dc)
    rp, _ = find_peaks(lp, distance=int(fs * 1.5),
                       prominence=lp.std() * 0.4)
    rt, _ = find_peaks(-lp, distance=int(fs * 1.5),
                       prominence=lp.std() * 0.4)
    if len(rp) < 2:
        return dict(rate=0, cv=0, amp_mean=0, lp=lp, peaks=rp, troughs=rt)
    iv = np.diff(rp) / fs
    rate = 60.0 / iv.mean()
    cv = float(iv.std() / iv.mean())
    n = min(len(rp), len(rt))
    amps = lp[rp[:n]] - lp[rt[:n]] if n else np.array([0])
    return dict(rate=rate, cv=cv, amp_mean=float(amps.mean()),
                lp=lp, peaks=rp, troughs=rt)


# ---------- 5 state estimators ----------
def state1_fatigue(hrv, eda_tonic, rip):
    """Higher = more fatigued. Parasympathetic dominance + low arousal."""
    # Rough z-anchors from typical resting adult population
    z_RMSSD = (hrv["RMSSD"] - 35) / 20      # high RMSSD -> fatigue-like
    z_HR = (hrv["HR"] - 72) / 10            # low HR -> fatigue
    z_SCL = (eda_tonic.mean() - 4) / 2      # low SCL -> fatigue
    z_resp = (rip["rate"] - 14) / 4         # slow breath -> fatigue
    score = z_RMSSD - z_HR - z_SCL - z_resp
    return float(score)


def state2_cognitive(hrv, eda_phasic, rip, fs=FS):
    """Higher = more cognitive engagement (focus without strong emotion)."""
    # SCRs from phasic - but use small ones (focus has tiny SCRs)
    scr_peaks, props = find_peaks(eda_phasic, distance=int(fs * 1.0),
                                  prominence=0.02, height=0.03)
    scr_amp = float(np.mean(props.get("peak_heights",
                                       [0]))) if len(scr_peaks) else 0
    z_LFHF = (hrv["LF_HF"] - 1.5) / 1.0     # high LF/HF -> engagement
    z_RMSSD = (hrv["RMSSD"] - 35) / 20      # lower RMSSD -> engagement
    z_resp_amp = (rip["amp_mean"] - 5000) / 3000  # shallow -> engagement
    # large SCRs penalize (suggests emotion not focus)
    z_scr = (scr_amp - 0.05) / 0.05
    score = z_LFHF - z_RMSSD - z_resp_amp - z_scr
    return float(score)


def state3_heart_age(sdnn, rmssd):
    """Estimate biological age from HRV (log-fit population norms)."""
    # Population norms (rough log fits to literature):
    #   SDNN(age) approx. 130 * exp(-0.018 * age)   -> ln(130/SDNN)/0.018
    #   RMSSD(age) approx. 80 * exp(-0.020 * age)
    age_sdnn = float(np.log(130 / max(sdnn, 1)) / 0.018)
    age_rmssd = float(np.log(80 / max(rmssd, 1)) / 0.020)
    return {
        "age_from_SDNN": round(age_sdnn, 1),
        "age_from_RMSSD": round(age_rmssd, 1),
        "age_combined": round((age_sdnn + age_rmssd) / 2, 1),
    }


def state4_coherence(rip, ecg, fs=FS):
    """Cardiopulmonary coherence - peak coherence in respiratory band (0.1-0.4 Hz).
    Higher = breath & HR strongly synchronized (RSA / coherence)."""
    rpeaks, rr_ms = get_rr(ecg, fs)
    if len(rr_ms) < 8:
        return {"peak_coh": 0, "peak_freq": 0, "f": np.array([]), "Cxy": np.array([])}
    t = np.cumsum(rr_ms) / 1000
    fs_re = 4.0
    t_uni = np.arange(t[0], t[-1], 1.0 / fs_re)
    rr_uni = interp1d(t, rr_ms, kind="cubic",
                      fill_value="extrapolate")(t_uni)
    # Resample RIP to same grid
    rip_dc = rip - rip.mean()
    sos = butter(4, 1.0, btype="low", fs=fs, output="sos")
    lp = sosfiltfilt(sos, rip_dc)
    t_full = np.arange(len(lp)) / fs
    rip_re = interp1d(t_full, lp, fill_value="extrapolate")(t_uni)
    # Coherence
    f, Cxy = coherence(rr_uni, rip_re, fs=fs_re,
                       nperseg=min(256, len(rr_uni)))
    band = (f >= 0.1) & (f <= 0.45)
    if band.sum() == 0:
        return {"peak_coh": 0, "peak_freq": 0, "f": f, "Cxy": Cxy}
    idx_in_band = np.where(band)[0]
    peak_idx = idx_in_band[int(np.argmax(Cxy[band]))]
    return {
        "peak_coh": float(Cxy[peak_idx]),
        "peak_freq": float(f[peak_idx]),
        "f": f,
        "Cxy": Cxy,
    }


def state5_recovery(rip, ecg, eda_us, fs=FS):
    """For each SCR event, measure time to fall to 50% of peak.
    Mean = recovery half-time (smaller = faster vagal rebound)."""
    smooth, tonic, phasic = eda_split(eda_us, fs)
    scr_peaks, props = find_peaks(phasic, distance=int(fs * 2),
                                  prominence=0.05, height=0.05)
    times = []
    for p in scr_peaks:
        peak_val = phasic[p]
        end = min(p + 30 * fs, len(phasic))
        below = np.where(phasic[p:end] < peak_val * 0.5)[0]
        if len(below) > 0:
            times.append(below[0] / fs)
    return {
        "n_scr": int(len(scr_peaks)),
        "mean_recovery_s": float(np.mean(times)) if times else float("nan"),
        "median_recovery_s": float(np.median(times)) if times else float("nan"),
        "recovery_times": times,
    }


# ---------- main ----------
def main():
    rows_summary = []
    cache = {}
    for c in CONDS:
        path = resolve_csv(c)
        print(f"  {c} ({path.name}) ...", flush=True)
        df = pd.read_csv(path)
        rip = df["RIP"].values.astype(float)
        ecg = df["ECG"].values.astype(float)
        eda_us = (df["EDA"].values.astype(float) / 65536) * 3.0 / 0.12

        rpeaks, rr_ms = get_rr(ecg, FS)
        hrv = hrv_stats(rr_ms)
        smooth, tonic, phasic = eda_split(eda_us, FS)
        rip_s = rip_stats(rip, FS)

        s1 = state1_fatigue(hrv, tonic, rip_s)
        s2 = state2_cognitive(hrv, phasic, rip_s, FS)
        s3 = state3_heart_age(hrv["SDNN"], hrv["RMSSD"])
        s4 = state4_coherence(rip, ecg, FS)
        s5 = state5_recovery(rip, ecg, eda_us, FS)

        cache[c] = dict(hrv=hrv, tonic_mean=tonic.mean(),
                        rip=rip_s, s4=s4, s5=s5)

        rows_summary.append({
            "condition": f"{c} ({CN[c]})",
            "fatigue_idx": round(s1, 2),
            "cog_load_idx": round(s2, 2),
            "heart_age_SDNN": s3["age_from_SDNN"],
            "heart_age_RMSSD": s3["age_from_RMSSD"],
            "heart_age": s3["age_combined"],
            "coh_peak": round(s4["peak_coh"], 3),
            "coh_freq_Hz": round(s4["peak_freq"], 3),
            "n_SCR": s5["n_scr"],
            "recovery_s_mean": round(s5["mean_recovery_s"], 2) if not np.isnan(s5["mean_recovery_s"]) else None,
            "recovery_s_median": round(s5["median_recovery_s"], 2) if not np.isnan(s5["median_recovery_s"]) else None,
        })

    summary = pd.DataFrame(rows_summary)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\n===== 5-state summary =====")
    print(summary.to_string(index=False))
    summary.to_csv(OUT / "jie_5states.csv", index=False)

    # ============ Visualization ============
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.30,
                          height_ratios=[1.1, 1.1, 1.3])

    cond_lbl = [f"{c}\n{CN[c]}" for c in CONDS]
    color_list = [COLORS[c] for c in CONDS]

    # Panel 1: Fatigue
    ax = fig.add_subplot(gs[0, 0])
    vals = [r["fatigue_idx"] for r in rows_summary]
    bars = ax.bar(cond_lbl, vals, color=color_list)
    ax.axhline(0, color="#666", lw=0.5)
    ax.set_title("1.  Fatigue index(higher = more fatigue-like)", fontweight="bold")
    ax.set_ylabel("z composite")
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.05 if v >= 0 else -0.15),
                f"{v:+.2f}", ha="center", fontsize=9, fontweight="bold")

    # Panel 2: Cognitive load
    ax = fig.add_subplot(gs[0, 1])
    vals = [r["cog_load_idx"] for r in rows_summary]
    bars = ax.bar(cond_lbl, vals, color=color_list)
    ax.axhline(0, color="#666", lw=0.5)
    ax.set_title("2.  Cognitive-load index(higher = stronger cognitive load)", fontweight="bold")
    ax.set_ylabel("z composite")
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.05 if v >= 0 else -0.15),
                f"{v:+.2f}", ha="center", fontsize=9, fontweight="bold")

    # Panel 3: Heart age (use stable as resting baseline)
    ax = fig.add_subplot(gs[0, 2])
    ages = [r["heart_age"] for r in rows_summary]
    bars = ax.bar(cond_lbl, ages, color=color_list)
    ax.set_title("3.  Autonomic age(based on HRV)", fontweight="bold")
    ax.set_ylabel("Estimated age(y)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(np.mean(ages), color="#444", ls="--", lw=0.7,
               label=f"mean {np.mean(ages):.1f} y")
    ax.legend(fontsize=8)
    for b, v in zip(bars, ages):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5,
                f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")

    # Panel 4: Coherence bars + spectrum overlay
    ax = fig.add_subplot(gs[1, 0])
    vals = [r["coh_peak"] for r in rows_summary]
    bars = ax.bar(cond_lbl, vals, color=color_list)
    ax.set_ylim(0, 1.05)
    ax.set_title("4.  Cardio-respiratory coupling / RSA, higher means HR and respiration are more synchronized", fontweight="bold")
    ax.set_ylabel("Peak coherence")
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    for c in CONDS:
        s4 = cache[c]["s4"]
        ax.plot(s4["f"], s4["Cxy"], color=COLORS[c],
                lw=1.7, label=f"{c} ({CN[c]})")
    ax.axvspan(0.1, 0.45, color="#aaa", alpha=0.15,
               label="RIP belt 0.1-0.45 Hz")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.set_title("RR-respiration frequency-domain coupling spectrum", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 5 (right of row 1): note
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    note = (
        "Interpretation quick guide:\n\n"
        "1. Fatigue: Concern and Anxious are not fatigue-like;\n"
        "  Calm is closest to fatigue-like physiology, but still not fatigue\n\n"
        "2. Cognition: Concern > Anxious > Calm\n"
        "  'Concern' is the strongest cognitive load\n\n"
        "3. Heart age: all three states look young by this metric\n"
        "  Anxious HRV rebound makes age look 'younger'\n"
        "  -> stable is the most reliable reference\n\n"
        "4. Coherence: look at the peak position\n"
        "  Alignment with respiration frequency indicates real coupling\n\n"
        "5. Recovery time: SCR half-life\n"
        "  The Anxious state has more SCR events and can be summarized")
    ax.text(0, 1, note, va="top", fontsize=10,
            family="ui-monospace, Consolas, monospace")

    # Panel 6: Recovery scatter
    ax = fig.add_subplot(gs[2, :])
    x_offset = 0
    for c in CONDS:
        s5 = cache[c]["s5"]
        if s5["n_scr"] == 0:
            continue
        x = np.arange(s5["n_scr"]) + x_offset
        y = s5["recovery_times"] + [np.nan] * (s5["n_scr"] - len(s5["recovery_times"]))
        # actually we keep only those with recoveries; lengths can differ
        y = s5["recovery_times"]
        x = np.arange(len(y)) + x_offset
        ax.scatter(x, y, c=COLORS[c], s=50, alpha=0.7,
                   edgecolor="white", lw=0.5, zorder=3,
                   label=f"{c} ({CN[c]}): n={len(y)},  "
                         f"median={np.median(y):.1f}s,  "
                         f"mean={np.mean(y):.1f}s")
        if y:
            ax.axhline(np.median(y), color=COLORS[c], ls="--", lw=0.7,
                       xmin=(x_offset) / 100, xmax=(x_offset + len(y)) / 100,
                       alpha=0.6)
        x_offset += max(len(y), 1) + 3
    ax.set_xlabel("SCR event index, grouped by condition")
    ax.set_ylabel("half-life (s)")
    ax.set_title("5. Stress recovery speed - time for each SCR event to decay to 50%; lower is faster",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.suptitle("jie five-state physiological inference report based on three existing CSV files, no new acquisition needed",
                 fontsize=15, fontweight="bold", y=0.995)

    out_png = OUT / "jie_5states.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"\nPNG saved: {out_png}")


if __name__ == "__main__":
    main()
