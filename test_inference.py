"""Test C: run the saved models (A: activity, B: posture, C: warning logic) on the held-out
participants P07/P08 and save window-level predictions.

    python test_inference.py      -> test_inference_predictions.csv

Fixes compared with the first version:
  * posture uses the tuned threshold tau (p_poor >= tau), not model.predict()'s 0.5 cut-off
  * posture accuracy is reported for ALL good/poor windows (pipeline view) as well as eligible-only
  * alert_active and is_calibration are kept, so the firmware comparison (fw_alert_frac) is filled
  * windows with a missing feature are flagged (features_imputed) and handled by the pipeline's
    median imputer, instead of crashing the run
  * Model C warning logic is applied (warning_prediction) and scored against the ground truth
These results come from synthetic data and do not establish real-world accuracy.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from posture_features import compute_derived, make_windows

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "spine_grid_posture_synthetic.csv"
ACTIVITY_PATH = ROOT / "models" / "activity_model.joblib"
POSTURE_PATH = ROOT / "models" / "posture_model.joblib"
CONFIG_PATH = ROOT / "models" / "warning_config.json"
OUTPUT_PATH = ROOT / "test_inference_predictions.csv"

TEST_SUBJECTS = {"P07", "P08"}
ACTIVITIES = ["sitting", "standing", "walking", "running", "lying_down"]


def load_artifact(path, expected_features):
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError(f"{path.name} is not a dictionary.")
    for key in ("model", "features"):
        if key not in artifact:
            raise KeyError(f"{path.name} is missing key: {key}")
    model, features = artifact["model"], artifact["features"]
    if len(features) != expected_features:
        raise ValueError(f"{path.name}: expected {expected_features} features, found {len(features)}")
    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(features):
        raise ValueError(f"{path.name}: pipeline expects {model.n_features_in_} features "
                         f"but artifact lists {len(features)}")
    print(f"Loaded {path.name}: {type(model).__name__}, {len(features)} features, "
          f"classes {list(getattr(model, 'classes_', []))}")
    return artifact


def severity(X, refs):
    """Largest deviation relative to the posture criteria (same definition as train_models.py)."""
    s = np.zeros(len(X))
    for col, ref in refs.items():
        s = np.maximum(s, np.nan_to_num(np.abs(X[col].to_numpy())) / ref)
    return s


def warning_logic(activity_pred, p_poor, sev, cfg):
    """Model C: alert after hold_s of continuous P(poor) >= tau in an applicable activity (1 s steps)."""
    levels, timer = [], 0.0
    for act, p, sv in zip(activity_pred, p_poor, sev):
        if act in cfg["applicable_activities"] and np.isfinite(p) and p >= cfg["tau_poor"]:
            timer += cfg["hop_s"]
        else:
            timer = 0.0
        if timer >= cfg["hold_s"]:
            critical = timer >= cfg["critical_sustain_s"] or sv >= cfg["critical_severity"]
            levels.append("critical" if critical else "advisory")
        else:
            levels.append("none")
    return levels


def runs(mask, t_start, t_end):
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return [(t_start[a], t_end[b]) for a, b in zip(s, e)]


def main():
    # 1. Files
    for path in (DATASET, ACTIVITY_PATH, POSTURE_PATH, CONFIG_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path} (run generate_dataset.py / train_models.py first)")

    # 2. Models and warning configuration
    activity_artifact = load_artifact(ACTIVITY_PATH, 165)
    posture_artifact = load_artifact(POSTURE_PATH, 168)
    cfg = json.loads(CONFIG_PATH.read_text())
    tau = float(posture_artifact.get("tau", cfg["tau_poor"]))
    if not np.isclose(tau, cfg["tau_poor"]):
        raise ValueError(f"tau mismatch: posture model {tau} vs warning_config {cfg['tau_poor']}")

    # 3. Dataset
    df = pd.read_csv(DATASET, low_memory=False)
    sources = set(df["data_source"].dropna().unique())
    print("\nDataset source values:", sources)
    if sources != {"synthetic"}:
        raise ValueError("Unexpected data_source values; verify provenance.")

    # 4. Held-out participants only
    df = df[df["subject_id"].isin(TEST_SUBJECTS)].copy()
    if df.empty:
        raise ValueError("No rows found for P07/P08.")
    print("Held-out participant rows:", len(df), "| sessions:", df["session_id"].nunique())

    # 5. Recompute the derived features from the raw columns (reproducibility),
    #    keeping raw channels, labels and the firmware output column alongside them.
    derived = compute_derived(df)
    if len(derived) != len(df):
        raise RuntimeError("compute_derived() changed the row count.")
    df[derived.columns] = derived

    # 6. Windows with the training feature builder (2 s, 50 % overlap)
    X, meta = make_windows(df)
    activity_features = activity_artifact["features"]
    posture_features = posture_artifact["features"]
    missing = [c for c in activity_features if c not in X.columns]
    if missing:
        raise ValueError(f"Missing activity features: {missing[:15]}")
    X_activity = X.loc[:, activity_features]

    # Missing values: flag them; the saved pipelines impute with training medians.
    imputed = X_activity.isna().any(axis=1).to_numpy()
    print(f"\nWindows: {len(X)} | windows with at least one missing feature (imputed): {int(imputed.sum())}")

    # 7. Model A: activity
    activity_pred = activity_artifact["model"].predict(X_activity)

    # 8. Model B: posture (only sitting / standing / walking; tau threshold)
    X_posture = X_activity.copy()
    for act in cfg["applicable_activities"]:
        X_posture[f"ctx_{act}"] = (activity_pred == act).astype(float)
    missing_posture = [c for c in posture_features if c not in X_posture.columns]
    if missing_posture:
        raise ValueError(f"Missing posture features: {missing_posture}")
    X_posture = X_posture.loc[:, posture_features]

    eligible = np.isin(activity_pred, cfg["applicable_activities"])
    posture_pred = np.full(len(X), "not_applicable", dtype=object)
    p_poor = np.full(len(X), np.nan)
    posture_model = posture_artifact["model"]
    if eligible.any():
        poor_idx = list(posture_model.classes_).index("poor")
        p_poor[eligible] = posture_model.predict_proba(X_posture.loc[eligible])[:, poor_idx]
        posture_pred[eligible] = np.where(p_poor[eligible] >= tau, "poor", "good")

    # 9. Model C: warning logic, per session in time order
    sev = severity(X, cfg["severity_reference_deg"])
    warning_pred = np.empty(len(X), dtype=object)
    for sid in meta["session_id"].unique():
        idx = np.where(meta["session_id"].to_numpy() == sid)[0]
        idx = idx[np.argsort(meta["t_start_ms"].to_numpy()[idx])]
        warning_pred[idx] = warning_logic(activity_pred[idx], p_poor[idx], sev[idx], cfg)

    print("\nACTIVITY PREDICTIONS\n" + pd.Series(activity_pred).value_counts().to_string())
    print("\nPOSTURE PREDICTIONS (tau = %.2f)\n" % tau + pd.Series(posture_pred).value_counts().to_string())
    print("\nWARNING PREDICTIONS\n" + pd.Series(warning_pred).value_counts().to_string())

    # 10. Ground-truth evaluation
    print("\nGROUND-TRUTH EVALUATION")
    y_act = meta["activity"].astype(str).to_numpy()
    va = np.isin(y_act, ACTIVITIES)
    print(f"Activity accuracy: {np.mean(activity_pred[va] == y_act[va]):.4f}  ({int(va.sum())} windows)")

    y_pos = meta["posture"].astype(str).to_numpy()
    gp = np.isin(y_act, cfg["applicable_activities"]) & np.isin(y_pos, ["good", "poor"])
    acc_all = np.mean(posture_pred[gp] == y_pos[gp])
    skipped = int(np.sum(gp & ~eligible))
    acc_elig = np.mean(posture_pred[gp & eligible] == y_pos[gp & eligible])
    poor = gp & (y_pos == "poor")
    print(f"Posture accuracy, all good/poor windows (pipeline): {acc_all:.4f}  ({int(gp.sum())} windows)")
    print(f"  of which skipped because Model A predicted running/lying: {skipped}")
    print(f"Posture accuracy, eligible windows only:            {acc_elig:.4f}  ({int((gp & eligible).sum())} windows)")
    print(f"Poor-posture recall (pipeline): {np.mean(posture_pred[poor] == 'poor'):.4f}")

    y_warn = meta["warning"].astype(str).to_numpy()
    print(f"Warning-level agreement per window: {np.mean(warning_pred == y_warn):.4f}")

    # Event view: an unsafe event is detected if an alert fires from 5 s before it until 3 s after it.
    events = det = false_good = other = 0
    hold_ms = cfg["hold_s"] * 1000
    for sid in meta["session_id"].unique():
        m = (meta["session_id"] == sid).to_numpy()
        g = meta[m].assign(w=warning_pred[m]).sort_values("t_start_ms")
        ts, te = g["t_start_ms"].to_numpy(), g["t_end_ms"].to_numpy()
        ev = runs((g["unsafe_frac"] >= 0.5).to_numpy(), ts, te)
        on = (g["w"] != "none").to_numpy()
        at = te[on]
        events += len(ev)
        det += sum(bool(((at >= s - hold_ms) & (at <= e + 3000)).any()) for s, e in ev)
        for a0, a1 in runs(on, te, te):
            if not any(a1 >= s - hold_ms and a0 <= e + 3000 for s, e in ev):
                near = g[(te >= a0 - hold_ms) & (te <= a1)]["posture"]
                if near.isin(["poor", "uncertain"]).any():
                    other += 1
                else:
                    false_good += 1
    print(f"Unsafe events detected: {det}/{events} | alerts on short/borderline poor posture: {other} "
          f"| false alerts during good posture: {false_good}")

    # 11. Save predictions with the window metadata
    out = meta.copy()
    out["activity_prediction"] = activity_pred
    out["posture_prediction"] = posture_pred
    out["p_poor"] = p_poor
    out["warning_prediction"] = warning_pred
    out["features_imputed"] = imputed.astype(int)
    out["warning_threshold"] = tau
    out.to_csv(OUTPUT_PATH, index=False)

    print("\nTEST C PASSED: all three model stages produced predictions.")
    print("Saved predictions:", OUTPUT_PATH)
    print("These results are from synthetic data and do not establish real-world accuracy "
          "or validate warning timing on the device.")


if __name__ == "__main__":
    main()
