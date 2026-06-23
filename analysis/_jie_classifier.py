"""3-class stress state classifier for jie (stable/middle/mess).

Strategy:
  - Sliding 60-s windows, 10-s stride over each 5-min recording -> ~25 windows/condition
  - Per window extract 11 physiological features (HR/HRV/EDA/Resp)
  - Evaluate with BOTH stratified-k-fold (optimistic, samples overlap) and
    temporal split (honest, first 70% train / last 30% test)
  - Try Random Forest + Logistic Regression
  - Visualize: PCA scatter, confusion matrix, feature importance, per-feature boxplots
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "data" / "jie"
OUT = REPO_ROOT / "output"
MODELS = REPO_ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

FS = 1000
WIN_SEC = 60
STRIDE_SEC = 10
CONDS = ["stable", "middle", "mess"]
CN = {"stable": "Calm", "middle": "Concern", "mess": "Anxious"}
COLORS = {"stable": "#10b981", "middle": "#f59e0b", "mess": "#ef4444"}
STATE_NUM = {c: i for i, c in enumerate(CONDS)}


def features_window(rip, ecg, eda_us, fs=FS):
    out = {}
    # --- ECG / HRV ---
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

    # --- EDA ---
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

    # --- RIP ---
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


def build_dataset():
    samples = []
    for c in CONDS:
        print(f"  {c} ...", flush=True)
        df = pd.read_csv(ROOT / f"{c}.csv")
        rip = df["RIP"].values.astype(float)
        ecg = df["ECG"].values.astype(float)
        eda_us = (df["EDA"].values.astype(float) / 65536) * 3.0 / 0.12
        win, stride = WIN_SEC * FS, STRIDE_SEC * FS
        for start in range(0, len(rip) - win + 1, stride):
            end = start + win
            f = features_window(rip[start:end], ecg[start:end],
                                eda_us[start:end])
            f["label"] = c
            f["start_sec"] = start / FS
            samples.append(f)
    return pd.DataFrame(samples)


def temporal_split(df, train_frac=0.7):
    """Per condition, first train_frac of windows -> train, rest -> test."""
    train_idx, test_idx = [], []
    for c in CONDS:
        sub = df[df["label"] == c].sort_values("start_sec")
        n = len(sub)
        cut = int(n * train_frac)
        train_idx.extend(sub.index[:cut].tolist())
        test_idx.extend(sub.index[cut:].tolist())
    return train_idx, test_idx


def main():
    print("Building sliding-window dataset...")
    df = build_dataset()
    print(f"\nTotal windows: {len(df)}")
    print(df.groupby("label").size())

    feature_cols = ["mean_HR", "RMSSD", "SDNN", "LF_HF",
                    "SCL_mean", "SCL_slope", "SCR_rate", "SCR_amp_mean",
                    "resp_rate", "resp_CV", "resp_amp_std"]
    X = df[feature_cols].values
    y = df["label"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # === Stratified k-fold CV (optimistic - adjacent windows overlap) ===
    rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                random_state=42, class_weight="balanced")
    lr = LogisticRegression(max_iter=2000, C=1.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_cv = cross_val_score(rf, Xs, y, cv=skf, scoring="accuracy")
    lr_cv = cross_val_score(lr, Xs, y, cv=skf, scoring="accuracy")
    print("\n===== Stratified 5-fold CV (optimistic) =====")
    print(f"  Random Forest : acc = {rf_cv.mean():.3f}  ± {rf_cv.std():.3f}")
    print(f"  Logistic Reg  : acc = {lr_cv.mean():.3f}  ± {lr_cv.std():.3f}")

    y_pred_cv = cross_val_predict(rf, Xs, y, cv=skf)
    print("\nConfusion (RF, k-fold CV):")
    cm_cv = confusion_matrix(y, y_pred_cv, labels=CONDS)
    print(pd.DataFrame(cm_cv, index=CONDS, columns=CONDS))
    print("\n", classification_report(y, y_pred_cv, target_names=CONDS,
                                       digits=3))

    # === Temporal split (honest) ===
    train_idx, test_idx = temporal_split(df, train_frac=0.7)
    X_tr, y_tr = Xs[train_idx], y[train_idx]
    X_te, y_te = Xs[test_idx], y[test_idx]

    rf2 = RandomForestClassifier(n_estimators=300, max_depth=8,
                                 random_state=42, class_weight="balanced")
    rf2.fit(X_tr, y_tr)
    y_pred_te = rf2.predict(X_te)
    acc_temp = accuracy_score(y_te, y_pred_te)
    print(f"\n===== Temporal split (70% early / 30% late, honest) =====")
    print(f"  Random Forest test acc = {acc_temp:.3f}")
    cm_temp = confusion_matrix(y_te, y_pred_te, labels=CONDS)
    print(pd.DataFrame(cm_temp, index=CONDS, columns=CONDS))

    # === Feature importance from final model ===
    rf_final = RandomForestClassifier(n_estimators=300, max_depth=8,
                                      random_state=42)
    rf_final.fit(Xs, y)
    fi = pd.Series(rf_final.feature_importances_,
                   index=feature_cols).sort_values(ascending=False)
    print("\n===== Feature importance (RF, trained on all) =====")
    print(fi)

    # === PCA ===
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)

    # ============ VISUALIZATION ============
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # PCA scatter
    ax = fig.add_subplot(gs[0, 0])
    for c in CONDS:
        m = y == c
        ax.scatter(Xp[m, 0], Xp[m, 1], c=COLORS[c],
                   label=f"{c} ({CN[c]})", s=55, alpha=0.75,
                   edgecolor="white", lw=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
    ax.set_title("PCA: 60-s windows projected from the 11-feature space")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Confusion matrix (k-fold)
    ax = fig.add_subplot(gs[0, 1])
    cm_norm = cm_cv.astype(float) / cm_cv.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"{c}\n{CN[c]}" for c in CONDS])
    ax.set_yticklabels([f"{c}\n{CN[c]}" for c in CONDS])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"k-fold CV confusion matrix\nacc = {rf_cv.mean():.1%}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_cv[i,j]}\n{cm_norm[i,j]:.0%}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=11, fontweight="bold")

    # Confusion matrix (temporal)
    ax = fig.add_subplot(gs[0, 2])
    cm_t_norm = cm_temp.astype(float) / cm_temp.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_t_norm, cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"{c}\n{CN[c]}" for c in CONDS])
    ax.set_yticklabels([f"{c}\n{CN[c]}" for c in CONDS])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Temporal split confusion matrix: first 70% train, last 30% test\nacc = {acc_temp:.1%}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_temp[i,j]}\n{cm_t_norm[i,j]:.0%}",
                    ha="center", va="center",
                    color="white" if cm_t_norm[i, j] > 0.5 else "black",
                    fontsize=11, fontweight="bold")

    # Feature importance
    ax = fig.add_subplot(gs[1, :])
    pos = range(len(fi))
    bars = ax.barh(pos, fi.values, color="#3b82f6")
    ax.set_yticks(pos)
    ax.set_yticklabels(fi.index)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Feature importance: Random Forest")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, v in zip(bars, fi.values):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)

    # Top-3 feature boxplots
    top3 = fi.index[:3].tolist()
    for col_i, feat in enumerate(top3):
        ax = fig.add_subplot(gs[2, col_i])
        data = [df[df["label"] == c][feat].values for c in CONDS]
        bp = ax.boxplot(data, patch_artist=True, labels=[f"{c}\n{CN[c]}" for c in CONDS])
        for patch, c in zip(bp["boxes"], CONDS):
            patch.set_facecolor(COLORS[c])
            patch.set_alpha(0.6)
        ax.set_title(f"{feat}  (imp={fi[feat]:.3f})")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"jie three-class stress-state classifier - "
        f"{len(df)} 60-s windows,RF k-fold={rf_cv.mean():.1%}, temporal={acc_temp:.1%}",
        fontsize=14, fontweight="bold")
    out_png = OUT / "jie_classifier.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"\nPNG saved: {out_png}")

    # Save features CSV for any downstream use
    df.to_csv(OUT / "jie_features.csv", index=False)
    print(f"Features CSV saved: {OUT / 'jie_features.csv'}")

    # ---------- Save trained model + scaler for live dashboard ----------
    from joblib import dump
    bundle = {
        "model": rf_final,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "classes": list(CONDS),
    }
    dump(bundle, MODELS / "jie_classifier.joblib")
    print(f"Model saved: {MODELS / 'jie_classifier.joblib'}")

    # ---------- Personal thresholds (per-condition percentiles) ----------
    thresholds = {}
    for c in CONDS:
        sub = df[df["label"] == c]
        thresholds[c] = {
            col: {
                "p10": float(np.percentile(sub[col], 10)),
                "p50": float(np.percentile(sub[col], 50)),
                "p90": float(np.percentile(sub[col], 90)),
            }
            for col in feature_cols
        }
    import json
    with open(MODELS / "jie_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    print(f"Thresholds saved: {MODELS / 'jie_thresholds.json'}")

    # Print compact threshold table for headline features
    print("\n===== Personal thresholds (p10 - p90 per condition) =====")
    headline = ["SCL_mean", "SCR_rate", "mean_HR", "resp_CV"]
    rows_t = []
    for c in CONDS:
        row = {"condition": f"{c} ({CN[c]})"}
        for col in headline:
            t = thresholds[c][col]
            row[col] = f"{t['p10']:.2f} ~ {t['p90']:.2f}"
        rows_t.append(row)
    print(pd.DataFrame(rows_t).to_string(index=False))


if __name__ == "__main__":
    main()
