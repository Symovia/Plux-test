"""Output 5 per-second tracks per condition for art visualization.

For each of jie's 3 CSVs, slide a 60-s window every second and emit:
  fatigue, focus, heart_age, coherence, resilience  (each 0-100, age in years)

Outputs:
  jie_report/tracks_stable.csv   (300 rows x 6 cols)
  jie_report/tracks_middle.csv
  jie_report/tracks_mess.csv
  jie_report/tracks_combined.csv (900 rows, with condition col)
  jie_report/tracks_overview.png (5 metrics x 3 conditions overlay)
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
TRACKS = REPO_ROOT / "tracks"
OUT.mkdir(exist_ok=True)
TRACKS.mkdir(exist_ok=True)
FS = 1000
WIN_SEC = 60
DURATION_SEC = 300
CONDS = ["stable", "middle", "mess"]
CN = {"stable": "Calm", "middle": "Concern", "mess": "Anxious"}
COLORS = {"stable": "#10b981", "middle": "#f59e0b", "mess": "#ef4444"}
FILE_ALIASES = {
    "stable": ["stable.csv"],
    "middle": ["middle.csv"],
    "mess":   ["mess.csv"],
}


def resolve_csv(c):
    for n in FILE_ALIASES[c]:
        p = DATA / "jie" / n
        if p.exists():
            return p
    raise FileNotFoundError(c)


def detect_rpeaks(ecg, fs=FS):
    sos = butter(4, [5, 40], btype="band", fs=fs, output="sos")
    bp = sosfiltfilt(sos, ecg - ecg.mean())
    rpeaks, _ = find_peaks(bp, distance=int(fs * 0.4),
                           prominence=bp.std() * 2.0)
    return rpeaks


def detect_breath_peaks(rip, fs=FS):
    sos = butter(4, 1.0, btype="low", fs=fs, output="sos")
    lp = sosfiltfilt(sos, rip - rip.mean())
    rp, _ = find_peaks(lp, distance=int(fs * 1.5),
                       prominence=lp.std() * 0.4)
    return lp, rp


def dfa_alpha1(rr, scales=range(4, 17)):
    if len(rr) < 30:
        return 1.0
    y = np.cumsum(rr - np.mean(rr))
    F, used = [], []
    for n in scales:
        seg = len(y) // n
        if seg < 1:
            continue
        local = []
        for k in range(seg):
            s = y[k * n:(k + 1) * n]
            t = np.arange(n)
            c = np.polyfit(t, s, 1)
            trend = np.polyval(c, t)
            local.append(np.mean((s - trend) ** 2))
        F.append(np.sqrt(np.mean(local)))
        used.append(n)
    if len(F) < 3:
        return 1.0
    return float(np.polyfit(np.log(used), np.log(F), 1)[0])


def metrics_5(rip_win, ecg_win, fs=FS):
    """Compute the 5 art-friendly values from a single window."""
    rpeaks = detect_rpeaks(ecg_win, fs)
    rr = np.diff(rpeaks) / fs * 1000.0
    rr = rr[(rr > 400) & (rr < 1500)]
    if len(rr) < 8:
        return dict(fatigue=50, focus=50, heart_age=30,
                    coherence=50, resilience=50)

    HR = 60000.0 / rr.mean()
    RMSSD = np.sqrt(np.mean(np.diff(rr) ** 2))
    SDNN = rr.std()
    SD1 = np.std(np.diff(rr)) / np.sqrt(2)
    SD2 = np.sqrt(max(2 * rr.var() - SD1 ** 2, 1e-6))

    # Freq + coherence (RR resampled to 4 Hz)
    t = np.cumsum(rr) / 1000.0
    fs_re = 4.0
    if t[-1] > 20:
        t_uni = np.arange(t[0], t[-1], 1.0 / fs_re)
        rr_uni = interp1d(t, rr, kind="cubic",
                          fill_value="extrapolate")(t_uni)
        nperseg = min(64, len(rr_uni))
        f, P = welch(rr_uni - rr_uni.mean(), fs=fs_re, nperseg=nperseg)
        LF = P[(f >= 0.04) & (f < 0.15)].sum()
        HF = P[(f >= 0.15) & (f < 0.4)].sum()
        LF_HF = LF / (HF + 1e-9)

        # Coherence with breath
        lp, _ = detect_breath_peaks(rip_win, fs)
        t_full = np.arange(len(lp)) / fs
        rip_re = interp1d(t_full, lp, fill_value="extrapolate")(t_uni)
        fc, Cxy = coherence(rr_uni, rip_re, fs=fs_re, nperseg=nperseg)
        band = (fc >= 0.1) & (fc <= 0.45)
        coh = float(Cxy[band].max()) if band.any() else 0.0
    else:
        LF_HF = 1.0
        coh = 0.5

    DFA_a1 = dfa_alpha1(rr)
    CVI = np.log10(16 * SD1 * SD2 + 1e-9)

    # Breath rate
    _, rp = detect_breath_peaks(rip_win, fs)
    breath_rate = 60.0 / (np.mean(np.diff(rp)) / fs) if len(rp) > 1 else 14

    # 5 mapped values
    fatigue = float(np.clip(
        50 + 0.6 * (RMSSD - 35) - 0.8 * (HR - 70) - 2.0 * (breath_rate - 14),
        0, 100))
    focus = float(np.clip(
        20 + 50 * (DFA_a1 - 0.9) + 8 * np.log(LF_HF + 1),
        0, 100))
    heart_age = float(np.clip(
        np.log(130.0 / max(SDNN, 1)) / 0.018, 0, 100))
    coh_score = float(np.clip(100 * coh, 0, 100))
    resilience = float(np.clip((CVI - 3.8) * 60, 0, 100))

    return dict(fatigue=fatigue, focus=focus, heart_age=heart_age,
                coherence=coh_score, resilience=resilience)


def process(csv_path):
    df = pd.read_csv(csv_path)
    rip = df["RIP"].values.astype(float)
    ecg = df["ECG"].values.astype(float)
    win = WIN_SEC * FS

    rows = []
    # Compute from t_end=60 to t_end=300 (1-sec step)
    for t_end in range(WIN_SEC, DURATION_SEC + 1):
        start = (t_end - WIN_SEC) * FS
        end = t_end * FS
        m = metrics_5(rip[start:end], ecg[start:end])
        rows.append({"t_sec": t_end, **m})

    df_out = pd.DataFrame(rows)
    # Pad t=0..59 by repeating the first computed row
    pad = pd.DataFrame([{"t_sec": t, **df_out.iloc[0].drop("t_sec").to_dict()}
                        for t in range(WIN_SEC)])
    full = pd.concat([pad, df_out], ignore_index=True)
    # Round
    for c in ["fatigue", "focus", "heart_age", "coherence", "resilience"]:
        full[c] = full[c].round(1)
    return full


def main():
    all_tracks = {}
    print("Computing 5 tracks per condition (60s window, 1s stride)...")
    for cond in CONDS:
        path = resolve_csv(cond)
        print(f"  {cond} ({path.name}) ...", flush=True)
        tracks = process(path)
        all_tracks[cond] = tracks
        out_path = TRACKS / f"tracks_{cond}.csv"
        tracks.to_csv(out_path, index=False)
        print(f"    -> {out_path}  ({len(tracks)} rows)")

    # Combined
    combined = []
    for cond in CONDS:
        d = all_tracks[cond].copy()
        d.insert(0, "condition", cond)
        combined.append(d)
    pd.concat(combined, ignore_index=True).to_csv(TRACKS / "tracks_combined.csv", index=False)
    print(f"  Combined -> {TRACKS / 'tracks_combined.csv'}")

    # Quick stats per condition
    print("\n===== Per-condition means =====")
    for cond in CONDS:
        d = all_tracks[cond]
        print(f"  {cond} ({CN[cond]}): "
              f"fatigue={d['fatigue'].mean():.1f}, "
              f"focus={d['focus'].mean():.1f}, "
              f"age={d['heart_age'].mean():.1f}, "
              f"coherence={d['coherence'].mean():.1f}, "
              f"resilience={d['resilience'].mean():.1f}")

    # ---------- Visualization: 5 metrics x 3 conditions overlay ----------
    metrics = ["fatigue", "focus", "heart_age", "coherence", "resilience"]
    labels = {
        "fatigue": "1. Fatigue (0-100)",
        "focus": "2. Focus (0-100)",
        "heart_age": "3. Heart age (y)",
        "coherence": "4. Coherence (0-100)",
        "resilience": "5. Resilience (0-100)",
    }
    fig, axes = plt.subplots(5, 1, figsize=(15, 11), sharex=True)
    for ax, m in zip(axes, metrics):
        for cond in CONDS:
            d = all_tracks[cond]
            ax.plot(d["t_sec"], d[m], color=COLORS[cond], lw=1.4,
                    label=f"{cond} ({CN[cond]})", alpha=0.85)
        ax.set_ylabel(labels[m])
        ax.grid(True, alpha=0.3)
        ax.axvline(WIN_SEC, color="#999", ls="--", lw=0.6, alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=9, ncol=3)
    axes[-1].set_xlabel("Time (s) - first 60 s are warm-up windows")
    axes[0].set_title("jie: three conditions x five art-visualization tracks, 1-s resolution",
                      fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_png = OUT / "tracks_overview.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"\nPNG saved: {out_png}")


if __name__ == "__main__":
    main()
