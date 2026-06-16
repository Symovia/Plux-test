"""Compute stress-score trajectories from jie's 3 CSV files (offline, no device).

Uses the pre-trained 60-s classifier with small stride (5 s) to produce
continuous 0-100 stress curves for each condition.
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
from joblib import load

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
OUT = REPO_ROOT / "output"
MODELS = REPO_ROOT / "models"
OUT.mkdir(exist_ok=True)
FS = 1000
WIN_SEC = 60
STRIDE_SEC = 5
CONDS = ["stable", "middle", "mess"]
CN = {"stable": "平静", "middle": "心事", "mess": "焦虑"}
COLORS = {"stable": "#10b981", "middle": "#f59e0b", "mess": "#ef4444"}

bundle = load(MODELS / "jie_classifier.joblib")
CLF = bundle["model"]
SCALER = bundle["scaler"]
FEATURES = bundle["feature_cols"]
# IMPORTANT: use clf.classes_ (alphabetical), NOT the bundle's "classes" order!
CLF_ORDER = list(CLF.classes_)
print(f"Loaded classifier: clf.classes_ = {CLF_ORDER}, {len(FEATURES)} features")


def features_window(rip, ecg, eda_us, fs=FS):
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


def trajectory(csv_path):
    df = pd.read_csv(csv_path)
    rip = df["RIP"].values.astype(float)
    ecg = df["ECG"].values.astype(float)
    eda_us = (df["EDA"].values.astype(float) / 65536) * 3.0 / 0.12

    win, stride = WIN_SEC * FS, STRIDE_SEC * FS
    rows = []
    for start in range(0, len(rip) - win + 1, stride):
        end = start + win
        feats = features_window(rip[start:end], ecg[start:end], eda_us[start:end])
        x = np.array([[feats[c] for c in FEATURES]])
        xs = SCALER.transform(x)
        probs = CLF.predict_proba(xs)[0]
        prob_map = dict(zip(CLF_ORDER, probs))  # map by ACTUAL classifier order
        pred = CLF_ORDER[int(np.argmax(probs))]
        score = float(prob_map["middle"] * 50 + prob_map["mess"] * 100)
        t_center = (start + win / 2) / FS
        rows.append({
            "t_sec": t_center,
            "score": score,
            "pred": pred,
            "p_stable": float(prob_map["stable"]),
            "p_middle": float(prob_map["middle"]),
            "p_mess": float(prob_map["mess"]),
            "SCL": feats["SCL_mean"],
            "SCR_rate": feats["SCR_rate"],
            "HR": feats["mean_HR"],
            "resp_CV": feats["resp_CV"],
        })
    return pd.DataFrame(rows)


def main():
    trajs = {}
    print("Computing trajectories...")
    for c in CONDS:
        print(f"  {c} ...", flush=True)
        trajs[c] = trajectory(DATA / "jie" / f"{c}.csv")

    # Concat with global time offset for "full journey" plot
    all_rows = []
    offset = 0
    boundaries = [0]
    for c in CONDS:
        d = trajs[c].copy()
        d["t_global"] = d["t_sec"] + offset
        d["cond"] = c
        all_rows.append(d)
        offset += 300  # 5 min per condition
        boundaries.append(offset)
    full = pd.concat(all_rows, ignore_index=True)

    # Print summary
    print("\n===== Stress trajectory summary =====")
    summary = []
    for c in CONDS:
        d = trajs[c]
        summary.append({
            "condition": f"{c} ({CN[c]})",
            "n_windows": len(d),
            "score_mean": round(d["score"].mean(), 1),
            "score_std": round(d["score"].std(), 1),
            "score_min": round(d["score"].min(), 1),
            "score_max": round(d["score"].max(), 1),
            "%_green (<33)": round((d["score"] < 33).mean() * 100, 1),
            "%_amber (33-67)": round(((d["score"] >= 33) & (d["score"] < 67)).mean() * 100, 1),
            "%_red (>=67)": round((d["score"] >= 67).mean() * 100, 1),
        })
    print(pd.DataFrame(summary).to_string(index=False))

    # === Plot ===
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(5, 1, hspace=0.55, height_ratios=[2.0, 1.0, 1.0, 1.0, 1.2])

    # (1) Full stress score across the 3-condition "journey"
    ax = fig.add_subplot(gs[0])
    # Background zone bands
    ax.axhspan(0, 33, color="#10b981", alpha=0.08)
    ax.axhspan(33, 67, color="#f59e0b", alpha=0.08)
    ax.axhspan(67, 100, color="#ef4444", alpha=0.08)
    # Condition vertical bands
    for i, c in enumerate(CONDS):
        ax.axvspan(boundaries[i], boundaries[i + 1],
                   color=COLORS[c], alpha=0.05)
        ax.text(boundaries[i] + 150, 96, f"{c}\n({CN[c]})",
                ha="center", va="top", fontweight="bold",
                color=COLORS[c], fontsize=11)
        ax.axvline(boundaries[i + 1], color="#444", ls="--", lw=0.7)
    ax.plot(full["t_global"], full["score"], color="#3b82f6", lw=1.8)
    # Color-coded dots for predicted class
    for c in CONDS:
        m = full["pred"] == c
        ax.scatter(full.loc[m, "t_global"], full.loc[m, "score"],
                   c=COLORS[c], s=18, zorder=3, alpha=0.9)
    ax.set_xlim(0, boundaries[-1])
    ax.set_ylim(-2, 102)
    ax.set_ylabel("压力分 (0-100)")
    ax.set_xlabel("时间(三条件拼接,共 15 分钟)")
    ax.set_title("整体压力轨迹 — 蓝线 = 连续压力分,圆点颜色 = 离散预测类别",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # (2-4) Per-condition probability ribbons
    for i, c in enumerate(CONDS):
        ax = fig.add_subplot(gs[i + 1])
        d = trajs[c]
        t = d["t_sec"].values
        ps = d["p_stable"].values
        pm = d["p_middle"].values
        pme = d["p_mess"].values
        ax.fill_between(t, 0, ps, color=COLORS["stable"], label="P(平静)", alpha=0.85)
        ax.fill_between(t, ps, ps + pm, color=COLORS["middle"], label="P(心事)", alpha=0.85)
        ax.fill_between(t, ps + pm, ps + pm + pme, color=COLORS["mess"], label="P(焦虑)", alpha=0.85)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0, 1)
        ax.set_ylabel("概率")
        ax.set_title(f"{c} ({CN[c]}) — 类别概率随时间", fontsize=10)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc="center right", fontsize=8, framealpha=0.9)

    # (5) Bottom: SCL trajectory across all 3 (overlay)
    ax = fig.add_subplot(gs[4])
    for c in CONDS:
        d = trajs[c]
        ax.plot(d["t_sec"], d["SCL"], color=COLORS[c], lw=1.7,
                label=f"{c} ({CN[c]})")
    ax.set_xlabel("时间 within 5-min (s)")
    ax.set_ylabel("SCL (µS)")
    ax.set_title("EDA 基线 SCL 三条件叠加(看是否单调上升)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "jie 离线压力分析 — 60s 窗口 / 5s 步长 / 已训分类器",
        fontsize=14, fontweight="bold")

    out_png = OUT / "jie_stress_timeline.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"\nPNG saved: {out_png}")

    # Save the trajectory CSV
    out_csv = OUT / "jie_stress_timeline.csv"
    full.to_csv(out_csv, index=False)
    print(f"Trajectory CSV: {out_csv}")


if __name__ == "__main__":
    main()
