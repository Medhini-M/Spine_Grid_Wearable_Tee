"""Train and evaluate the Spine-Grid Tee posture models on the (synthetic) dataset.

    python train_models.py [--csv spine_grid_posture_synthetic.csv]

Pipeline (context-aware, hierarchical):
  Model A  activity classifier      2 s windows (50 % overlap) -> sitting / standing / walking / running / lying_down
  Model B  posture-quality model    same windows + activity context -> good / poor   (only sitting, standing, walking)
  Model C  warning logic            temporal state machine on Model A context + Model B P(poor) + deviation severity
                                    -> none / advisory / critical, compared against the unmodified firmware rules

Split is participant-independent: train P01-P05, validation P06, test P07-P08. No adjacent windows of one
participant ever appear in two splits. Outputs: models/, reports/REPORT.md, reports/metrics.json.

NOTE: the dataset is synthetic. The numbers below test the pipeline; they are not evidence of real-world accuracy.
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from posture_features import DERIVED_COLS, FS, compute_derived, make_windows, window_feature_names

TRAIN, VAL, TEST = ["P01", "P02", "P03", "P04", "P05"], ["P06"], ["P07", "P08"]
ACTIVITIES = ["sitting", "standing", "walking", "running", "lying_down"]
APPLICABLE = ["sitting", "standing", "walking"]
HOLD_S = 5.0                 # continuous poor-posture time before alerting (same as firmware)
CRIT_SUSTAIN_S = 20.0
CRIT_SEVERITY = 1.5
SEVERITY_REF = {"upper_pitch_dev_deg__mean": 12.0, "upper_roll_dev_deg__mean": 10.0,
                "shoulder_asymmetry_dev_deg__mean": 10.0, "shoulder_rounding_dev_deg__mean": 12.0}
SEED = 7
REPORT = []


def log(s=""):
    print(s)
    REPORT.append(s)


def md_table(df, floatfmt="{:.3f}"):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"]
    for idx, row in df.iterrows():
        vals = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        out.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join(out)


def report_block(title, y_true, y_pred, labels):
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    t = pd.DataFrame({k: rep[k] for k in labels}).T[["precision", "recall", "f1-score", "support"]]
    t["support"] = t["support"].astype(int)
    t.index.name = "class"
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    cm.index.name = "true \\ pred"
    log(f"\n**{title}** - accuracy {rep['accuracy']:.3f}, macro-F1 {rep['macro avg']['f1-score']:.3f}\n")
    log(md_table(t))
    log("\nConfusion matrix (rows = true, columns = predicted):\n")
    log(md_table(cm, "{}"))
    return rep, cm


# ---------------------------------------------------------------- ROCKET (time-series model, numpy only)
class Rocket:
    """ROCKET (Dempster et al., 2020): random dilated convolution kernels -> PPV + max features -> ridge."""

    def __init__(self, n_kernels=600, seed=SEED):
        self.n_kernels, self.rng = n_kernels, np.random.default_rng(seed)

    def _make(self, C, T):
        ks = []
        for _ in range(self.n_kernels):
            ln = int(self.rng.choice([7, 9, 11]))
            nch = int(self.rng.integers(1, min(C, 3) + 1))
            ch = self.rng.choice(C, nch, replace=False)
            w = self.rng.normal(0, 1, (nch, ln))
            w -= w.mean()
            dil = int(2 ** self.rng.uniform(0, np.log2((T - 1) / (ln - 1))))
            pad = ((ln - 1) * dil) // 2 if self.rng.random() < 0.5 else 0
            ks.append((ch, w, self.rng.uniform(-1, 1), dil, pad, ln))
        return ks

    def transform(self, X):           # X: (n, C, T)
        n, C, T = X.shape
        F = np.empty((n, 2 * len(self.kernels)), dtype=np.float32)
        for k, (ch, w, b, dil, pad, ln) in enumerate(self.kernels):
            Xp = np.pad(X[:, ch, :], ((0, 0), (0, 0), (pad, pad))) if pad else X[:, ch, :]
            Tout = Xp.shape[2] - (ln - 1) * dil
            out = np.full((n, Tout), b, dtype=np.float64)
            for j in range(ln):
                out += np.einsum("nct,c->nt", Xp[:, :, j * dil: j * dil + Tout], w[:, j])
            F[:, 2 * k] = (out > 0).mean(1)
            F[:, 2 * k + 1] = out.max(1)
        return F

    def fit(self, X, y):
        self.mu = np.nanmean(X, axis=(0, 2), keepdims=True)
        self.sd = np.nanstd(X, axis=(0, 2), keepdims=True) + 1e-6
        Xn = np.nan_to_num((X - self.mu) / self.sd)
        self.kernels = self._make(X.shape[1], X.shape[2])
        self.clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
        self.clf.fit(self.transform(Xn), y)
        return self

    def predict(self, X):
        return self.clf.predict(self.transform(np.nan_to_num((X - self.mu) / self.sd)))


SEQ_CHANNELS = [f"imu_{p}_{c}" for p in ("upper", "left", "right") for c in ("ax", "ay", "az", "gx", "gy", "gz")] \
    + ["flex_lr_dev", "flex_rl_dev"]


def sequences(df, meta):
    out = np.empty((len(meta), len(SEQ_CHANNELS), 2 * FS), dtype=np.float32)
    by_sess = {sid: g[SEQ_CHANNELS].to_numpy(float).T for sid, g in df.groupby("session_id", sort=False)}
    for i, r in enumerate(meta.itertuples()):
        out[i] = by_sess[r.session_id][:, r.start_row:r.start_row + 2 * FS]
    return out


# ---------------------------------------------------------------- warning logic (Model C)
def severity(Xw):
    s = np.zeros(len(Xw))
    for col, ref in SEVERITY_REF.items():
        s = np.maximum(s, np.nan_to_num(np.abs(Xw[col].to_numpy())) / ref)
    return s


def run_warning_logic(meta_s, act_pred, p_poor, sev, tau, crit_sev=CRIT_SEVERITY):
    """meta_s: windows of ONE session in time order. Decision time = window end. Returns level per window."""
    levels, timer = [], 0.0
    hop = 1.0
    for a, p, sv in zip(act_pred, p_poor, sev):
        if a in APPLICABLE and p >= tau:
            timer += hop
        else:
            timer = 0.0
        if timer >= HOLD_S:
            levels.append("critical" if (timer >= CRIT_SUSTAIN_S or sv >= crit_sev) else "advisory")
        else:
            levels.append("none")
    return np.array(levels)


def runs(mask, t):
    """Contiguous True runs -> list of (start_time, end_time)."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return [(t[a], t[b]) for a, b in zip(s, e)]


def event_metrics(events, alert_times, alert_runs, hours, gt_rows):
    """events: [(start, end)] unsafe; alerts: decision times; tolerance: alert may come from poor-onset
    (event start - 5 s) up to 3 s after the event ends."""
    detected, latencies = 0, []
    for s, e in events:
        hits = alert_times[(alert_times >= s - HOLD_S * 1000) & (alert_times <= e + 3000)]
        if len(hits):
            detected += 1
            latencies.append(max(0.0, (hits.min() - s) / 1000))
    false = short = border = 0
    t_all, pq_all = gt_rows
    for a0, a1 in alert_runs:
        if any((a1 >= s - HOLD_S * 1000) and (a0 <= e + 3000) for s, e in events):
            continue
        sel = pq_all[(t_all >= a0 - HOLD_S * 1000) & (t_all <= a1)]
        if (sel == "poor").any():
            short += 1          # real poor posture, but shorter than the 5 s needed to be "unsafe"
        elif (sel == "uncertain").any():
            border += 1         # borderline magnitude: annotators marked it uncertain
        else:
            false += 1          # alert while the posture was good / not applicable
    return dict(events=len(events), detected=detected, missed=len(events) - detected,
                recall=detected / max(len(events), 1),
                alerts_on_short_poor_episodes=short, alerts_on_borderline_episodes=border,
                false_alerts_good_posture=false, false_alerts_per_hour=false / hours,
                median_latency_s=float(np.median(latencies)) if latencies else None,
                p90_latency_s=float(np.percentile(latencies, 90)) if latencies else None)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="spine_grid_posture_synthetic.csv")
    ap.add_argument("--rocket-kernels", type=int, default=600)
    args = ap.parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    t0 = time.time()

    df = pd.read_csv(args.csv, low_memory=False)
    log("# Training report - Spine-Grid Tee posture models\n")
    log(f"Dataset: `{args.csv}`, {len(df):,} rows, data_source = {', '.join(df.data_source.unique())}. "
        "**Synthetic data: these results test the pipeline, not real-world accuracy.**\n")

    # 1. reproducibility check: re-derive every derived column from the raw columns
    der = compute_derived(df)
    diffs = {c: float(np.nanmax(np.abs(der[c].to_numpy() - df[c].to_numpy()))) for c in DERIVED_COLS}
    nan_mismatch = {c: int((der[c].isna() != df[c].isna()).sum()) for c in DERIVED_COLS}
    log("## 1. Feature reproducibility\n")
    log(f"Derived columns recomputed from raw data: max abs difference {max(diffs.values()):.4f} "
        f"(rounding only); NaN-pattern mismatches: {sum(nan_mismatch.values())}.\n")
    df[DERIVED_COLS] = der[DERIVED_COLS]

    # 2. windows
    X, meta = make_windows(df)
    split = np.where(meta.subject_id.isin(TRAIN), "train", np.where(meta.subject_id.isin(VAL), "val", "test"))
    meta["split"] = split
    log("## 2. Windows and participant-independent split\n")
    log(f"2 s windows, 50 % overlap (hop 1 s), never crossing sessions: {len(X):,} windows, "
        f"{X.shape[1]} features (mean, std, min, max, range of {X.shape[1] // 5} channels).\n")
    tab = pd.crosstab(meta.activity, meta.split)[["train", "val", "test"]]
    tab.index.name = "activity window"
    log(md_table(tab, "{}"))
    log(f"\nTrain = {', '.join(TRAIN)} | validation = {', '.join(VAL)} | test = {', '.join(TEST)}. "
        "Windows labelled `transition` (activity share < 60 % or transition rows) are excluded from Model A.\n")

    # ------------------------------------------------ Model A
    log("## 3. Model A - activity classification\n")
    a_mask = meta.activity.isin(ACTIVITIES)
    tr, va, te = (a_mask & (meta.split == s) for s in ("train", "val", "test"))
    ytr, yva, yte = meta.activity[tr], meta.activity[va], meta.activity[te]

    def rf(depth, leaf):
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300, max_depth=depth, min_samples_leaf=leaf,
                                                    class_weight="balanced", n_jobs=-1, random_state=SEED))
    cand = {}
    for depth in (None, 16):
        for leaf in (1, 3):
            m = rf(depth, leaf).fit(X[tr], ytr)
            cand[f"RF depth={depth} leaf={leaf}"] = (m, f1_score(yva, m.predict(X[va]), average="macro"))
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED).fit(X[tr], ytr)
    cand["HistGradientBoosting"] = (hgb, f1_score(yva, hgb.predict(X[va]), average="macro"))
    seq = sequences(df, meta)
    rk = Rocket(args.rocket_kernels).fit(seq[tr.to_numpy()], ytr.to_numpy())
    cand["ROCKET (time-series)"] = (rk, f1_score(yva, rk.predict(seq[va.to_numpy()]), average="macro"))
    vt = pd.DataFrame({"validation macro-F1": {k: v[1] for k, v in cand.items()}})
    vt.index.name = "candidate"
    log("Validation (P06) comparison:\n")
    log(md_table(vt))
    rf_names = [k for k in cand if k.startswith("RF")]
    best_rf = max(rf_names, key=lambda k: cand[k][1])
    log(f"\nSelected for deployment: **{best_rf}** (Random Forest: small-data friendly, interpretable, "
        "exportable to the ESP32). Refit on train + validation, then scored once on the test participants.\n")
    model_a_trainonly = cand[best_rf][0]
    depth = None if "None" in best_rf else 16
    leaf = int(best_rf.split("leaf=")[1])
    trva = tr | va
    model_a = rf(depth, leaf).fit(X[trva], meta.activity[trva])
    repA, _ = report_block("Model A (Random Forest) on test participants P07-P08", yte, model_a.predict(X[te]), ACTIVITIES)
    # comparison models on test
    hgb2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED).fit(X[trva], meta.activity[trva])
    rk2 = Rocket(args.rocket_kernels).fit(seq[trva.to_numpy()], meta.activity[trva].to_numpy())
    comp = pd.DataFrame({
        "test accuracy": {"Random Forest (selected)": repA["accuracy"],
                          "HistGradientBoosting": float((hgb2.predict(X[te]) == yte).mean()),
                          "ROCKET (time-series)": float((rk2.predict(seq[te.to_numpy()]) == yte.to_numpy()).mean())},
        "test macro-F1": {"Random Forest (selected)": repA["macro avg"]["f1-score"],
                          "HistGradientBoosting": f1_score(yte, hgb2.predict(X[te]), average="macro"),
                          "ROCKET (time-series)": f1_score(yte, rk2.predict(seq[te.to_numpy()]), average="macro")}})
    comp.index.name = "model"
    log("\nConventional vs time-series model on the test participants:\n")
    log(md_table(comp))
    per_subj = {s: float((model_a.predict(X[te & (meta.subject_id == s)]) ==
                          meta.activity[te & (meta.subject_id == s)]).mean()) for s in TEST}
    log(f"\nPer-participant test accuracy: " + ", ".join(f"{k} {v:.3f}" for k, v in per_subj.items()) + "\n")
    imp = pd.Series(model_a[-1].feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    log("Top features: " + ", ".join(f"`{k}` ({v:.3f})" for k, v in imp.items()) + "\n")

    # ------------------------------------------------ Model B
    log("## 4. Model B - posture quality (good vs poor, activity-aware)\n")
    act_onehot = lambda acts: pd.DataFrame({f"ctx_{a}": (np.asarray(acts) == a).astype(float) for a in APPLICABLE})
    b_mask = meta.activity.isin(APPLICABLE) & meta.posture.isin(["good", "poor"])
    btr, bva, bte = (b_mask & (meta.split == s) for s in ("train", "val", "test"))
    XB = lambda m, acts: pd.concat([X[m].reset_index(drop=True), act_onehot(acts)], axis=1)
    model_b = make_pipeline(SimpleImputer(strategy="median"),
                            RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                                   n_jobs=-1, random_state=SEED))
    model_b.fit(XB(btr, meta.activity[btr]), meta.posture[btr])
    poor_idx = list(model_b.classes_).index("poor")
    pva = model_b.predict_proba(XB(bva, meta.activity[bva]))[:, poor_idx]
    taus = np.round(np.arange(0.3, 0.81, 0.05), 2)
    f1s = [f1_score(meta.posture[bva] == "poor", pva >= t) for t in taus]
    tau = float(taus[int(np.argmax(f1s))])
    # tune the critical-severity factor of Model C on P06 with train-only models (no leakage)
    va_all = meta.split.eq("val")
    act_va = model_a_trainonly.predict(X[va_all])
    p_va = model_b.predict_proba(pd.concat([X[va_all].reset_index(drop=True), act_onehot(act_va)], axis=1))[:, poor_idx]
    sev_va, mva = severity(X[va_all]), meta[va_all]
    best_cs, best_f = CRIT_SEVERITY, -1
    for cs in np.round(np.arange(1.3, 2.61, 0.1), 1):
        lv = []
        for sid in mva.session_id.unique():
            m = (mva.session_id == sid).to_numpy()
            lv += list(run_warning_logic(None, act_va[m], p_va[m], sev_va[m], tau, cs))
        f = f1_score(mva.warning, lv, labels=["none", "advisory", "critical"], average="macro")
        if f > best_f:
            best_cs, best_f = float(cs), f
    log(f"Decision threshold on P(poor) tuned on validation P06: **tau = {tau:.2f}** (F1 {max(f1s):.3f}). "
        f"Critical-severity factor for Model C tuned on P06: **{best_cs:.1f}x** the protocol criterion "
        f"(warning-level macro-F1 {best_f:.3f}). Refit on train + validation.\n")
    btrva = btr | bva
    model_b.fit(XB(btrva, meta.activity[btrva]), meta.posture[btrva])
    poor_idx = list(model_b.classes_).index("poor")
    pb_true = model_b.predict_proba(XB(bte, meta.activity[bte]))[:, poor_idx]
    report_block("Model B on test, true activity context", meta.posture[bte],
                 np.where(pb_true >= tau, "poor", "good"), ["good", "poor"])
    actA_te = model_a.predict(X[bte])
    pb_pipe = model_b.predict_proba(XB(bte, actA_te))[:, poor_idx]
    report_block("Model B on test, activity context predicted by Model A (full pipeline)", meta.posture[bte],
                 np.where(pb_pipe >= tau, "poor", "good"), ["good", "poor"])
    unc = meta.split.eq("test") & meta.activity.isin(APPLICABLE) & meta.posture.eq("uncertain")
    if unc.any():
        pu = model_b.predict_proba(XB(unc, meta.activity[unc]))[:, poor_idx]
        log(f"\n{int(unc.sum())} test windows are labelled `uncertain` (borderline magnitude or posture "
            f"ramps); they are not scored. The model calls {np.mean(pu >= tau):.0%} of them poor.\n")
    impB = pd.Series(model_b[-1].feature_importances_, index=list(X.columns) + [f"ctx_{a}" for a in APPLICABLE]) \
        .sort_values(ascending=False).head(10)
    log("Top features: " + ", ".join(f"`{k}` ({v:.3f})" for k, v in impB.items()) + "\n")

    # ------------------------------------------------ Model C
    log("## 5. Model C - warning logic vs. firmware rules (test participants)\n")
    log(f"Rule: alert when Model A says sitting/standing/walking AND P(poor) >= {tau:.2f} for {HOLD_S:.0f} s "
        f"in a row (1 s decision steps). Level = critical if sustained >= {CRIT_SUSTAIN_S:.0f} s or deviation "
        f"severity >= {best_cs:.1f}x the protocol criterion, else advisory. Running, lying and transitions never alert.\n")
    te_all = meta.split.eq("test")
    Xte_all, mte = X[te_all], meta[te_all].copy()
    mte["act_pred"] = model_a.predict(Xte_all)
    mte["p_poor"] = model_b.predict_proba(pd.concat([Xte_all.reset_index(drop=True), act_onehot(mte.act_pred)],
                                                    axis=1))[:, poor_idx]
    mte["sev"] = severity(Xte_all)
    ml, fw = {"alert_times": [], "alert_runs": []}, {"alert_times": [], "alert_runs": []}
    gt_t, gt_pq = [], []
    events, levels_pred, levels_true = [], [], []
    hours = 0.0
    for sid, g in mte.groupby("session_id", sort=False):
        g = g.sort_values("t_start_ms")
        lv = run_warning_logic(g, g.act_pred.to_numpy(), g.p_poor.to_numpy(), g.sev.to_numpy(), tau, best_cs)
        tdec = g.t_end_ms.to_numpy()
        levels_pred += list(lv)
        levels_true += list(g.warning.to_numpy())
        ml["alert_times"] += list(tdec[lv != "none"])
        ml["alert_runs"] += runs(lv != "none", tdec)
        rows = df[df.session_id == sid]
        t = rows.timestamp_ms.to_numpy()
        events += runs((rows.safety_label == "unsafe").to_numpy(), t)
        gt_t.append(t); gt_pq.append(rows.posture_quality.to_numpy())
        fa = rows.alert_active.to_numpy().astype(bool)
        fw["alert_times"] += list(t[fa])
        fw["alert_runs"] += runs(fa, t)
        hours += len(rows) / FS / 3600
    gt = (np.concatenate(gt_t), np.concatenate(gt_pq))
    mC = event_metrics(events, np.array(ml["alert_times"]), ml["alert_runs"], hours, gt)
    fC = event_metrics(events, np.array(fw["alert_times"]), fw["alert_runs"], hours, gt)
    ct = pd.DataFrame({"ML pipeline (A+B+C)": mC, "Firmware rules (unmodified)": fC})
    ct.index.name = "metric (test participants)"
    log(md_table(ct.map(lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else v)), "{}"))
    log(f"\nTest recording time: {hours * 60:.1f} min. An unsafe event counts as detected if an alert fires "
        "between poor-posture onset (5 s before the event is labelled unsafe) and 3 s after it ends; latency is "
        "measured from the unsafe onset (0 = the alert was already on when the posture became unsafe). Alert episodes "
        "that overlap no unsafe event are split into: real poor posture that ended before 5 s, borderline "
        "episodes the annotation marks `uncertain`, and false alerts during good posture.\n")
    report_block("Warning level per 1 s decision step (ML pipeline)", np.array(levels_true), np.array(levels_pred),
                 ["none", "advisory", "critical"])

    # ------------------------------------------------ save
    joblib.dump({"model": model_a, "features": list(X.columns), "classes": ACTIVITIES}, "models/activity_model.joblib")
    joblib.dump({"model": model_b, "features": list(X.columns) + [f"ctx_{a}" for a in APPLICABLE],
                 "tau": tau}, "models/posture_model.joblib")
    json.dump({"tau_poor": tau, "hold_s": HOLD_S, "critical_sustain_s": CRIT_SUSTAIN_S,
               "critical_severity": best_cs, "severity_reference_deg": SEVERITY_REF,
               "applicable_activities": APPLICABLE, "window_s": 2, "hop_s": 1}, open("models/warning_config.json", "w"),
              indent=2)
    json.dump({"model_A_test": {"accuracy": repA["accuracy"], "macro_f1": repA["macro avg"]["f1-score"],
                                "per_participant_accuracy": per_subj},
               "model_C_ml": mC, "model_C_firmware": fC, "tau": tau,
               "data_source": "synthetic"}, open("reports/metrics.json", "w"), indent=2, default=float)
    log(f"\nSaved models/activity_model.joblib, models/posture_model.joblib, models/warning_config.json. "
        f"Run time {time.time() - t0:.0f} s.\n")
    open("reports/REPORT.md", "w").write("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
