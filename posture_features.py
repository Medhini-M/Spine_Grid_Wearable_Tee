"""Reproducible feature extraction for the Spine-Grid Tee posture dataset.

Used by BOTH generate_dataset.py (to write the derived columns) and train_models.py
(to re-derive them from the raw columns and build 2 s windows), so every derived
value in the CSV can be recomputed from the raw sensor columns.

Coordinate system (all three MPU6050 boards, see COLLECTION_PROTOCOL.md):
  board flat on the back, chip side facing OUT, printed Y arrow pointing UP the spine
    +Y : up along the spine (toward the head)
    +Z : out of the back (posterior)
    +X : toward the wearer's right
  Upright & still  ->  accelerometer ~ (0, +1, 0) g

Angles (degrees)
  pitch = atan2(az, ay)                     forward (+) / backward (-) bend in the sagittal plane
  roll  = atan2(-ax, sqrt(ay^2 + az^2))     side lean, + = toward the wearer's right
Deviations are relative to the session's calibration pose (see compute_baseline).
"""
import numpy as np
import pandas as pd

IMUS = {"upper": "imu_upper", "left": "imu_left", "right": "imu_right"}
AX = ["ax", "ay", "az"]
GX = ["gx", "gy", "gz"]
RAW_IMU_COLS = [f"{p}_{c}" for p in IMUS.values() for c in AX + GX]
FLEX_COLS = ["flex_lr_raw", "flex_rl_raw"]
RAW_COLS = RAW_IMU_COLS + FLEX_COLS

# Firmware configuration (from the Spine-Grid firmware, used as metadata / baseline logic)
WARN_DEG, CRITICAL_DEG = 8.0, 16.0
FLEX_WARN, FLEX_CRITICAL = 400, 900
BAD_POSTURE_HOLD_MS = 5000
FIRMWARE_TICK_ROWS = 5            # firmware evaluates its rules every 200 ms = every 5 rows at 25 Hz
CALIB_SKIP_S = 2.0                # dataset baseline ignores the first 2 s of the calibration trial
FS = 25

DERIVED_COLS = [
    "upper_pitch_deg", "upper_roll_deg", "left_shoulder_pitch_deg", "left_shoulder_roll_deg",
    "right_shoulder_pitch_deg", "right_shoulder_roll_deg",
    "upper_pitch_dev_deg", "upper_roll_dev_deg", "shoulder_asymmetry_dev_deg", "shoulder_rounding_dev_deg",
    "flex_lr_dev", "flex_rl_dev", "flex_dev_total", "accel_magnitude", "gyro_magnitude",
    "thoracic_dev_deg",
]
FIRMWARE_COLS = ["warning_condition", "critical_condition", "critical_duration_ms",
                 "alert_active", "led_state", "buzzer_state"]


def pitch_roll(ax, ay, az):
    pitch = np.degrees(np.arctan2(az, ay))
    roll = np.degrees(np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2)))
    return pitch, roll


def flex_valid(v):
    """A flex reading of exactly 0 or 4095 is treated as invalid (open/short connector, ADC rail)."""
    v = np.asarray(v, dtype=float)
    return np.isfinite(v) & (v > 0) & (v < 4095)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def absolute_angles(df):
    out = {}
    for name, p in IMUS.items():
        pit, rol = pitch_roll(df[f"{p}_ax"].to_numpy(float), df[f"{p}_ay"].to_numpy(float),
                              df[f"{p}_az"].to_numpy(float))
        key = "upper" if name == "upper" else f"{name}_shoulder"
        out[f"{key}_pitch_deg"] = pit
        out[f"{key}_roll_deg"] = rol
    return out


def compute_baseline(sess):
    """Dataset calibration baseline for one session: mean of the calibration trial (after the first 2 s)."""
    cal = sess[sess["is_calibration"] == 1]
    t0 = cal["timestamp_ms"].min()
    cal = cal[cal["timestamp_ms"] >= t0 + CALIB_SKIP_S * 1000]
    ang = absolute_angles(cal)
    base = {k: float(np.nanmean(v)) for k, v in ang.items()}
    for c in FLEX_COLS:
        v = cal[c].to_numpy(float)
        base[c] = float(np.nanmean(np.where(flex_valid(v), v, np.nan)))
    return base


def compute_derived(df):
    """Per-sample derived columns for a dataframe holding one or more complete sessions."""
    parts = []
    for _, sess in df.groupby("session_id", sort=False):
        base = compute_baseline(sess)
        ang = absolute_angles(sess)
        d = pd.DataFrame(ang, index=sess.index)
        d["upper_pitch_dev_deg"] = wrap180(d["upper_pitch_deg"] - base["upper_pitch_deg"])
        d["upper_roll_dev_deg"] = wrap180(d["upper_roll_deg"] - base["upper_roll_deg"])
        lr_dev = wrap180(d["left_shoulder_roll_deg"] - base["left_shoulder_roll_deg"])
        rr_dev = wrap180(d["right_shoulder_roll_deg"] - base["right_shoulder_roll_deg"])
        lp_dev = wrap180(d["left_shoulder_pitch_deg"] - base["left_shoulder_pitch_deg"])
        rp_dev = wrap180(d["right_shoulder_pitch_deg"] - base["right_shoulder_pitch_deg"])
        d["shoulder_asymmetry_dev_deg"] = np.abs(lr_dev - rr_dev)
        d["shoulder_rounding_dev_deg"] = (lp_dev + rp_dev) / 2 - d["upper_pitch_dev_deg"]
        for c, name in (("flex_lr_raw", "flex_lr_dev"), ("flex_rl_raw", "flex_rl_dev")):
            v = sess[c].to_numpy(float)
            d[name] = np.where(flex_valid(v), v - base[c], np.nan)
        d["flex_dev_total"] = np.abs(d["flex_lr_dev"]) + np.abs(d["flex_rl_dev"])
        a = sess[[f"imu_upper_{c}" for c in AX]].to_numpy(float)
        g = sess[[f"imu_upper_{c}" for c in GX]].to_numpy(float)
        d["accel_magnitude"] = np.sqrt((a ** 2).sum(axis=1))
        d["gyro_magnitude"] = np.sqrt((g ** 2).sum(axis=1))
        d["thoracic_dev_deg"] = np.abs(d["upper_pitch_dev_deg"])
        parts.append(d[DERIVED_COLS])
    return pd.concat(parts).loc[df.index]


# ---------------------------------------------------------------- firmware simulation
def simulate_firmware(sess):
    """Re-implements the on-shirt fallback rules of firmware v3 (spine_grid_tee_spine.ino, localMonitor):
       * evaluated every 200 ms (every 5th row); outputs are held between evaluations
       * firmware tilt formulas: fwd = atan2(az, ay), side = atan2(-ax, ay)
       * firmware baseline = mean of the last 2 s (50 rows) of the calibration trial
       * a failed IMU read keeps the previous raw value (hold-last); flex spikes are NOT filtered
       * crit -> start timer; alert when the continuous critical time >= 5000 ms; timer resets when crit clears
       * LED: red = alert, yellow = warn or crit (not yet sustained), green = otherwise
       * buzzer_state = vibration motor output: on for 400 ms of every 1500 ms while the alert is active
    Unmodified thresholds; NO activity-specific changes (see report for why that matters)."""
    n = len(sess)
    t = (sess["timestamp_ms"].to_numpy() - sess["timestamp_ms"].iloc[0]).astype(np.int64)
    raw = {}
    for p in IMUS.values():
        for c in AX:
            raw[f"{p}_{c}"] = pd.Series(sess[f"{p}_{c}"].to_numpy(float)).ffill().bfill().to_numpy()
    f1 = sess["flex_lr_raw"].to_numpy(float)
    f2 = sess["flex_rl_raw"].to_numpy(float)
    f1 = pd.Series(f1).ffill().bfill().to_numpy()
    f2 = pd.Series(f2).ffill().bfill().to_numpy()

    def tilt(p):
        ax, ay, az = raw[f"{p}_ax"], raw[f"{p}_ay"], raw[f"{p}_az"]
        return np.degrees(np.arctan2(az, ay)), np.degrees(np.arctan2(-ax, ay))

    uf, us = tilt("imu_upper"); lf, ls = tilt("imu_left"); rf, rs = tilt("imu_right")
    cal_idx = np.where(sess["is_calibration"].to_numpy() == 1)[0][-50:]
    b = {k: np.mean(v[cal_idx]) for k, v in dict(uf=uf, us=us, lf=lf, ls=ls, rf=rf, rs=rs, f1=f1, f2=f2).items()}

    warn_c = np.zeros(n, np.int8); crit_c = np.zeros(n, np.int8); dur = np.zeros(n, np.int64)
    alert = np.zeros(n, np.int8); led = np.empty(n, dtype=object); buzz = np.zeros(n, np.int8)
    crit_since = None
    cur = (0, 0, 0, 0, "green")
    for i in range(n):
        if i % FIRMWARE_TICK_ROWS == 0:
            dUf, dUs = uf[i] - b["uf"], us[i] - b["us"]
            dLf, dRf = lf[i] - b["lf"], rf[i] - b["rf"]
            dLs, dRs = ls[i] - b["ls"], rs[i] - b["rs"]
            vals = [abs(dUf), abs(dUs), abs((dLf + dRf) / 2 - dUf), abs(dLs - dRs)]
            lumbar = abs(f1[i] - b["f1"]) + abs(f2[i] - b["f2"])
            crit = any(v > CRITICAL_DEG for v in vals) or lumbar > FLEX_CRITICAL
            warn = any(v > WARN_DEG for v in vals) or lumbar > FLEX_WARN
            if crit:
                if crit_since is None:
                    crit_since = t[i]
                d = int(t[i] - crit_since)
                al = d >= BAD_POSTURE_HOLD_MS
            else:
                crit_since, d, al = None, 0, False
            colour = "red" if al else ("yellow" if (crit or warn) else "green")
            cur = (int(warn), int(crit), d, int(al), colour)
        warn_c[i], crit_c[i], dur[i], alert[i], led[i] = cur
        buzz[i] = 1 if (cur[3] and (t[i] % 1500) < 400) else 0
    return pd.DataFrame({"warning_condition": warn_c, "critical_condition": crit_c,
                         "critical_duration_ms": dur, "alert_active": alert,
                         "led_state": led, "buzzer_state": buzz}, index=sess.index)


# ---------------------------------------------------------------- windowing (Models A/B/C)
WINDOW_ROWS = 2 * FS          # 2 s
HOP_ROWS = FS                 # 50 % overlap
WINDOW_CHANNELS = (
    [f"{p}_{c}" for p in IMUS.values() for c in AX + GX]
    + ["upper_pitch_deg", "upper_roll_deg", "left_shoulder_pitch_deg", "left_shoulder_roll_deg",
       "right_shoulder_pitch_deg", "right_shoulder_roll_deg",
       "upper_pitch_dev_deg", "upper_roll_dev_deg", "shoulder_asymmetry_dev_deg", "shoulder_rounding_dev_deg",
       "flex_lr_dev", "flex_rl_dev", "flex_dev_total", "accel_magnitude", "gyro_magnitude"]
)
STATS = ["mean", "std", "min", "max", "range"]


def window_feature_names():
    return [f"{c}__{s}" for c in WINDOW_CHANNELS for s in STATS]


def _mode(values):
    vals, counts = np.unique(values, return_counts=True)
    i = counts.argmax()
    return vals[i], counts[i] / len(values)


def make_windows(df):
    """2 s windows, 50 % overlap, never crossing a session. Returns (X features DataFrame, meta DataFrame)."""
    feats, meta = [], []
    for sid, sess in df.groupby("session_id", sort=False):
        arr = sess[WINDOW_CHANNELS].to_numpy(float)
        n = len(sess)
        for s in range(0, n - WINDOW_ROWS + 1, HOP_ROWS):
            w = arr[s:s + WINDOW_ROWS]
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mn, sd = np.nanmean(w, 0), np.nanstd(w, 0)
                    lo, hi = np.nanmin(w, 0), np.nanmax(w, 0)
            feats.append(np.column_stack([mn, sd, lo, hi, hi - lo]).reshape(-1))
            ws = sess.iloc[s:s + WINDOW_ROWS]
            act, act_share = _mode(ws["activity_label"].to_numpy())
            pq, pq_share = _mode(ws["posture_quality"].to_numpy())
            sl, _ = _mode(ws["safety_label"].to_numpy())
            wl, _ = _mode(ws["warning_level"].to_numpy())
            meta.append(dict(
                session_id=sid, subject_id=ws["subject_id"].iloc[0], start_row=s,
                t_start_ms=int(ws["timestamp_ms"].iloc[0]), t_end_ms=int(ws["timestamp_ms"].iloc[-1]),
                activity=act if act_share >= 0.6 else "transition", activity_share=act_share,
                posture=pq if pq_share >= 0.6 else "uncertain", safety=sl, warning=wl,
                unsafe_frac=float((ws["safety_label"] == "unsafe").mean()),
                fw_alert_frac=float(ws["alert_active"].mean()) if "alert_active" in ws else np.nan,
                missing_frac=float(np.isnan(w).any(axis=1).mean()),
            ))
    X = pd.DataFrame(np.vstack(feats), columns=window_feature_names())
    return X, pd.DataFrame(meta)
