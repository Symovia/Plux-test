"""Deep cardiac feature extraction beyond plain HR/HRV-time-domain.

Categories extracted from each 5-min ECG recording:
  A. Time-domain HRV          (HR, SDNN, RMSSD, pNN50)         — reference
  B. Frequency-domain HRV     (LF, HF, VLF, LF/HF, HFnu)       — reference
  C. Non-linear HRV           (SD1, SD2, SD1/SD2, SampEn, DFA-α1)
  D. Beat morphology          (R-amplitude mean/std/CV, QRS-width est.)
  E. ECG-derived respiration  (EDR via R-amplitude envelope)
  F. Autonomic indices        (CVI, CSI from Poincaré)
  G. Rhythm quality           (ectopic count, max RR drift)
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
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from scipy.interpolate import interp1d

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
OUT = REPO_ROOT / "output"
OUT.mkdir(exist_ok=True)
FS = 1000
CONDS = ["stable", "middle", "mess"]
CN = {"stable": "平静", "middle": "心事", "mess": "焦虑"}
COLORS = {"stable": "#10b981", "middle": "#f59e0b", "mess": "#ef4444"}
FILE_ALIASES = {
    "stable": ["stable.csv", "稳.csv"],
    "middle": ["middle.csv", "中.csv"],
    "mess":   ["mess.csv", "乱.csv"],
}


def resolve_csv(c):
    for n in FILE_ALIASES[c]:
        p = DATA / "jie" / n
        if p.exists():
            return p
    raise FileNotFoundError(c)


# --------- ECG preprocessing & R-peak amplitude ---------
def ecg_pipeline(ecg, fs=FS):
    sos = butter(4, [5, 40], btype="band", fs=fs, output="sos")
    bp = sosfiltfilt(sos, ecg - ecg.mean())
    rpeaks, _ = find_peaks(bp, distance=int(fs * 0.4),
                           prominence=bp.std() * 2.0)
    rr_ms = np.diff(rpeaks) / fs * 1000.0
    keep = (rr_ms > 400) & (rr_ms < 1500)
    rr_ms_clean = rr_ms[keep]
    # R amplitudes from original (centered) ECG, at peaks
    r_amps = ecg[rpeaks] - ecg.mean()
    return bp, rpeaks, rr_ms, rr_ms_clean, r_amps


# --------- A. Time-domain HRV ---------
def hrv_time(rr):
    if len(rr) < 5:
        return {}
    diff = np.diff(rr)
    return {
        "HR_bpm": float(60000 / rr.mean()),
        "SDNN_ms": float(rr.std()),
        "RMSSD_ms": float(np.sqrt(np.mean(diff ** 2))),
        "pNN50_pct": float(np.sum(np.abs(diff) > 50) / max(1, len(diff)) * 100),
    }


# --------- B. Frequency-domain HRV ---------
def hrv_freq(rr):
    if len(rr) < 10:
        return {}
    t = np.cumsum(rr) / 1000
    fs_re = 4.0
    t_uni = np.arange(t[0], t[-1], 1.0 / fs_re)
    rr_uni = interp1d(t, rr, kind="cubic", fill_value="extrapolate")(t_uni)
    f, P = welch(rr_uni - rr_uni.mean(), fs=fs_re,
                 nperseg=min(256, len(rr_uni)))
    vlf = float(P[(f >= 0.003) & (f < 0.04)].sum())
    lf = float(P[(f >= 0.04) & (f < 0.15)].sum())
    hf = float(P[(f >= 0.15) & (f < 0.4)].sum())
    total = vlf + lf + hf
    return {
        "VLF": vlf,
        "LF": lf,
        "HF": hf,
        "LF_HF": lf / (hf + 1e-9),
        "HFnu": hf / (lf + hf + 1e-9) * 100,
        "LFnu": lf / (lf + hf + 1e-9) * 100,
    }


# --------- C. Non-linear HRV ---------
def poincare(rr):
    """SD1 (short-term variability) and SD2 (long-term)."""
    if len(rr) < 5:
        return {}
    x1 = rr[:-1]
    x2 = rr[1:]
    diff = np.diff(rr)
    sd1 = float(np.std(diff) / np.sqrt(2))
    sd2 = float(np.sqrt(2 * rr.var() - sd1 ** 2))
    return {
        "SD1_ms": sd1,
        "SD2_ms": sd2,
        "SD1_SD2_ratio": sd1 / (sd2 + 1e-9),
        "x1": x1,
        "x2": x2,
    }


def sample_entropy(x, m=2, r=None):
    """Sample entropy — autonomic complexity. Lower = more regular."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < m + 2:
        return float("nan")
    if r is None:
        r = 0.2 * x.std()

    def _count(m_):
        templates = np.array([x[i:i + m_] for i in range(N - m_ + 1)])
        count = 0
        for i in range(len(templates) - 1):
            diff = np.abs(templates[i + 1:] - templates[i]).max(axis=1)
            count += int((diff <= r).sum())
        return count

    B = _count(m)
    A = _count(m + 1)
    if B == 0 or A == 0:
        return float("nan")
    return float(-np.log(A / B))


def dfa_alpha1(x, scales=range(4, 17)):
    """Detrended Fluctuation Analysis short-term scaling exponent."""
    x = np.asarray(x, dtype=float)
    if len(x) < 30:
        return float("nan")
    y = np.cumsum(x - x.mean())
    F = []
    used = []
    for n in scales:
        segments = len(y) // n
        if segments < 1:
            continue
        local = []
        for k in range(segments):
            seg = y[k * n:(k + 1) * n]
            t = np.arange(n)
            coefs = np.polyfit(t, seg, 1)
            trend = np.polyval(coefs, t)
            local.append(np.mean((seg - trend) ** 2))
        F.append(np.sqrt(np.mean(local)))
        used.append(n)
    if len(F) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(used), np.log(F), 1)
    return float(slope)


# --------- D. Beat morphology ---------
def beat_morphology(ecg, rpeaks, fs=FS):
    """R-wave amplitude statistics + QRS-width estimate."""
    if len(rpeaks) < 5:
        return {}
    amps = ecg[rpeaks] - ecg.mean()
    # QRS width: half-prominence width around each R
    bp = ecg - ecg.mean()
    widths = []
    for p in rpeaks:
        a = bp[p]
        half = a / 2
        lo = p
        while lo > max(0, p - int(0.08 * fs)) and bp[lo] > half:
            lo -= 1
        hi = p
        while hi < min(len(bp) - 1, p + int(0.08 * fs)) and bp[hi] > half:
            hi += 1
        widths.append((hi - lo) / fs * 1000)  # ms
    widths = np.array(widths)
    return {
        "R_amp_mean": float(amps.mean()),
        "R_amp_std": float(amps.std()),
        "R_amp_CV": float(amps.std() / (abs(amps.mean()) + 1e-9)),
        "QRS_width_mean_ms": float(widths.mean()),
        "QRS_width_std_ms": float(widths.std()),
    }


# --------- E. ECG-derived respiration ---------
def edr(rpeaks, r_amps, fs=FS, total_n=None):
    """Resample R-amplitude envelope to 4 Hz, low-pass to <=1 Hz."""
    if len(rpeaks) < 8:
        return None
    t_r = rpeaks / fs
    fs_re = 4.0
    t_uni = np.arange(t_r[0], t_r[-1], 1.0 / fs_re)
    edr_uni = interp1d(t_r, r_amps, kind="cubic",
                       fill_value="extrapolate")(t_uni)
    sos = butter(4, 1.0, btype="low", fs=fs_re, output="sos")
    edr_smooth = sosfiltfilt(sos, edr_uni - edr_uni.mean())
    return t_uni, edr_smooth


def edr_correlation(t_uni, edr_signal, rip, fs=FS):
    """Correlate EDR with the real RIP signal (same time grid)."""
    if t_uni is None:
        return float("nan")
    sos = butter(4, 1.0, btype="low", fs=fs, output="sos")
    rip_lp = sosfiltfilt(sos, rip - rip.mean())
    t_full = np.arange(len(rip_lp)) / fs
    rip_at = interp1d(t_full, rip_lp,
                      fill_value="extrapolate")(t_uni)
    if len(rip_at) < 5 or rip_at.std() == 0 or edr_signal.std() == 0:
        return float("nan")
    return float(np.corrcoef(rip_at, edr_signal)[0, 1])


# --------- F. Autonomic indices ---------
def autonomic_indices(sd1, sd2):
    """Cardiac Vagal Index (vagal tone), Cardiac Sympathetic Index (stress)."""
    return {
        "CVI": float(np.log10(16 * sd1 * sd2 + 1e-9)),
        "CSI": float(sd2 / (sd1 + 1e-9)),
    }


# --------- G. Rhythm quality ---------
def rhythm_quality(rr_all, rr_clean):
    """Count ectopic-like beats (extreme RR) and HRV outlier rate."""
    if len(rr_all) < 5:
        return {}
    rejected = len(rr_all) - len(rr_clean)
    # RR jumps > 20% of median (proxy for ectopic/PVC)
    med = np.median(rr_clean)
    jumps = np.sum(np.abs(np.diff(rr_clean)) > 0.20 * med)
    return {
        "n_beats": int(len(rr_all) + 1),
        "ectopic_rejected": int(rejected),
        "ectopic_pct": float(rejected / max(1, len(rr_all)) * 100),
        "large_jumps": int(jumps),
    }


# ============ Main ============
def main():
    print("Extracting deep cardiac features (5 min × 3 conditions)...")
    results = {}
    rows = []
    for c in CONDS:
        path = resolve_csv(c)
        print(f"  {c} ({path.name})", flush=True)
        df = pd.read_csv(path)
        rip = df["RIP"].values.astype(float)
        ecg = df["ECG"].values.astype(float)

        bp, rpeaks, rr_all, rr_clean, r_amps = ecg_pipeline(ecg)
        A = hrv_time(rr_clean)
        B = hrv_freq(rr_clean)
        poin = poincare(rr_clean)
        C = {k: poin[k] for k in ("SD1_ms", "SD2_ms", "SD1_SD2_ratio")}
        C["SampEn"] = sample_entropy(rr_clean)
        C["DFA_a1"] = dfa_alpha1(rr_clean)
        D = beat_morphology(ecg, rpeaks)
        F = autonomic_indices(poin["SD1_ms"], poin["SD2_ms"])
        G = rhythm_quality(rr_all, rr_clean)
        edr_out = edr(rpeaks, r_amps)
        if edr_out is not None:
            t_edr, edr_sig = edr_out
            E_corr = edr_correlation(t_edr, edr_sig, rip)
        else:
            t_edr, edr_sig, E_corr = None, None, float("nan")

        results[c] = dict(
            bp=bp, rpeaks=rpeaks, rr=rr_clean, r_amps=r_amps,
            poin=poin, edr_t=t_edr, edr_sig=edr_sig,
        )
        rows.append({
            "condition": f"{c} ({CN[c]})",
            **A, **B,
            **C,
            **D,
            "EDR_corr_with_RIP": round(E_corr, 3) if not np.isnan(E_corr) else None,
            **F,
            **G,
        })

    summary = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)
    print("\n===== Deep cardiac summary =====\n")
    # Print in groups
    groups = {
        "A 时间域 HRV": ["condition", "HR_bpm", "SDNN_ms", "RMSSD_ms", "pNN50_pct"],
        "B 频域 HRV": ["condition", "VLF", "LF", "HF", "LF_HF", "LFnu", "HFnu"],
        "C 非线性 HRV": ["condition", "SD1_ms", "SD2_ms", "SD1_SD2_ratio", "SampEn", "DFA_a1"],
        "D 波形形态": ["condition", "R_amp_mean", "R_amp_std", "R_amp_CV", "QRS_width_mean_ms", "QRS_width_std_ms"],
        "E 衍生呼吸 vs F 自主指数 G 心律": ["condition", "EDR_corr_with_RIP", "CVI", "CSI", "ectopic_rejected", "ectopic_pct", "large_jumps"],
    }
    for name, cols in groups.items():
        print(f"--- {name} ---")
        print(summary[cols].to_string(index=False))
        print()

    summary.to_csv(OUT / "jie_cardiac_deep.csv", index=False)
    print(f"CSV saved: {OUT / 'jie_cardiac_deep.csv'}")

    # ============ Visualization ============
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 4, hspace=0.50, wspace=0.35,
                          height_ratios=[1.4, 1.0, 1.2])

    # Row 1: Poincaré plots
    for i, c in enumerate(CONDS):
        ax = fig.add_subplot(gs[0, i])
        poin = results[c]["poin"]
        ax.scatter(poin["x1"], poin["x2"], s=10, alpha=0.5,
                   color=COLORS[c])
        # Ellipse around centroid
        cx, cy = poin["x1"].mean(), poin["x2"].mean()
        from matplotlib.patches import Ellipse
        e = Ellipse((cx, cy), width=2 * poin["SD2_ms"] * np.sqrt(2),
                    height=2 * poin["SD1_ms"] * np.sqrt(2),
                    angle=-45, fill=False, edgecolor="#333", lw=1.5)
        ax.add_patch(e)
        ax.set_xlabel("RRᵢ (ms)")
        ax.set_ylabel("RRᵢ₊₁ (ms)")
        ax.set_title(f"Poincaré: {c} ({CN[c]})\n"
                     f"SD1={poin['SD1_ms']:.0f}  SD2={poin['SD2_ms']:.0f}  "
                     f"SD1/SD2={poin['SD1_SD2_ratio']:.2f}",
                     fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    # Row 1 last cell: SampEn + DFA bars
    ax = fig.add_subplot(gs[0, 3])
    metrics = ["SampEn", "DFA_a1"]
    x = np.arange(len(metrics))
    w = 0.25
    for i, c in enumerate(CONDS):
        vals = [summary.loc[i, "SampEn"], summary.loc[i, "DFA_a1"]]
        ax.bar(x + (i - 1) * w, vals, w, color=COLORS[c],
               label=f"{c} ({CN[c]})")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("非线性 HRV(复杂度 + 长程相关)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Row 2: Bars for SD1, SD2, CVI, CSI
    bar_metrics = [
        ("SD1_ms", "SD1 (短时变异 ms)"),
        ("SD2_ms", "SD2 (长时变异 ms)"),
        ("CVI", "CVI 副交感(↑好)"),
        ("CSI", "CSI 交感比(↑紧张)"),
    ]
    for col, (m, title) in enumerate(zip(range(4), bar_metrics)):
        ax = fig.add_subplot(gs[1, col])
        metric, ttl = title
        vals = [summary.loc[i, metric] for i in range(len(CONDS))]
        bars = ax.bar([f"{c}\n{CN[c]}" for c in CONDS],
                      vals, color=[COLORS[c] for c in CONDS])
        ax.set_title(ttl, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                    f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}",
                    ha="center", fontsize=9, fontweight="bold")

    # Row 3: EDR (ECG-derived respiration) vs real RIP — overlay for one condition
    show_c = "stable"
    ax = fig.add_subplot(gs[2, :3])
    res = results[show_c]
    if res["edr_t"] is not None:
        # Plot real RIP first (after centering)
        rip = pd.read_csv(resolve_csv(show_c))["RIP"].values.astype(float)
        rip_lp_sos = butter(4, 1.0, btype="low", fs=FS, output="sos")
        rip_lp = sosfiltfilt(rip_lp_sos, rip - rip.mean())
        t_rip = np.arange(len(rip_lp)) / FS
        # Normalize each for overlay comparison
        rip_norm = rip_lp / (rip_lp.std() + 1e-9)
        edr_norm = res["edr_sig"] / (res["edr_sig"].std() + 1e-9)
        ax.plot(t_rip, rip_norm, color="#60a5fa", lw=1.2,
                label="真实 RIP (z-scored)", alpha=0.85)
        ax.plot(res["edr_t"], edr_norm, color="#f87171", lw=1.2,
                label="ECG-derived 呼吸 (z-scored)", alpha=0.85)
        ax.set_xlim(0, 60)
        ax.set_xlabel("时间 (s) — 显示前 60 s")
        ax.set_ylabel("归一化幅度")
        corr = summary.loc[CONDS.index(show_c), "EDR_corr_with_RIP"]
        ax.set_title(
            f"E. ECG 衍生呼吸 vs 真实 RIP — {show_c} ({CN[show_c]})  |  "
            f"correlation = {corr:.2f}\n"
            f"(意思:从心跳就能反推出呼吸波形,不靠 RIP 带也行)",
            fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Row 3 last cell: EDR correlations across 3 conditions
    ax = fig.add_subplot(gs[2, 3])
    corrs = [summary.loc[i, "EDR_corr_with_RIP"] for i in range(len(CONDS))]
    bars = ax.bar([f"{c}\n{CN[c]}" for c in CONDS],
                  corrs, color=[COLORS[c] for c in CONDS])
    ax.set_ylim(-0.2, 1)
    ax.axhline(0, color="#666", lw=0.5)
    ax.set_title("EDR ↔ RIP 相关性", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, corrs):
        if v is None:
            continue
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04,
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle(
        "心跳信号深度挖掘 — 5 类特征(非线性 HRV / 波形 / EDR / 自主指数 / 心律质量)",
        fontsize=14, fontweight="bold")
    out_png = OUT / "jie_cardiac_deep.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"\nPNG saved: {out_png}")


if __name__ == "__main__":
    main()
