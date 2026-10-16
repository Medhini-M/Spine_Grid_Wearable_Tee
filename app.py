import json
import threading
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd
import serial
from serial.tools import list_ports
from flask import Flask, jsonify, render_template_string

from posture_features import (
    IMUS, AX, GX, FLEX_COLS, WINDOW_CHANNELS, STATS, FS,
    WINDOW_ROWS, HOP_ROWS, absolute_angles, wrap180, flex_valid,
)

BAUD_RATE = 115200
SERIAL_PORT = None  # Set to "/dev/cu.usbmodemXXXX" if auto-detection fails.
CALIBRATION_SECONDS = 4.0
CALIBRATION_SKIP_SECONDS = 2.0
SAMPLE_INTERVAL_MS = 1000 / FS
HISTORY_LIMIT = 1800  # Keep at most ~72 seconds at 25 Hz.

BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent
ACTIVITY_PATH = BASE_DIR / "models" / "activity_model.joblib"
POSTURE_PATH = BASE_DIR / "models" / "posture_model.joblib"
CONFIG_PATH = BASE_DIR / "models" / "warning_config.json"

activity_artifact = joblib.load(ACTIVITY_PATH)
posture_artifact = joblib.load(POSTURE_PATH)
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

activity_model = activity_artifact["model"]
posture_model = posture_artifact["model"]
activity_features = list(activity_artifact["features"])
posture_features = list(posture_artifact["features"])
tau = float(posture_artifact.get("tau", cfg.get("tau_poor", 0.5)))

app = Flask(__name__)
lock = threading.Lock()
latest_reading = None
last_received = 0.0
serial_status = "Searching for ESP32..."
history = deque(maxlen=HISTORY_LIMIT)
session_start = None
calibration_complete = False
calibration_baseline = None
calibration_message = "Keep still in your normal upright posture for calibration."
latest_prediction = {
    "activity": "Calibrating",
    "posture": "Waiting",
    "poor_probability": None,
    "warning": "Waiting",
    "model_note": "Experimental — trained on synthetic data",
    "window_ready": False,
    "sample_count": 0,
}
last_prediction_count = 0
poor_hold_seconds = 0.0
sensor_rule_hold_seconds = 0.0
warning_state = "none"


def find_esp32_port():
    ports = list(list_ports.comports())
    for port in ports:
        device = port.device.lower()
        description = (port.description or "").lower()
        if (
            "usbmodem" in device or "usbserial" in device
            or "esp32" in description or "cp210" in description
            or "ch340" in description
        ):
            return port.device
    print("Could not automatically identify the ESP32 port. Available ports:")
    for port in ports:
        print(f"  {port.device}: {port.description}")
    return None


def row_from_reading(data, timestamp_ms, is_calibration):
    """Convert the ESP32 JSON payload to the repository's expected raw columns."""
    row = {
        "session_id": "live_session",
        "timestamp_ms": int(timestamp_ms),
        "is_calibration": int(is_calibration),
        "flex_lr_raw": float(data.get("flex1", np.nan)),
        "flex_rl_raw": float(data.get("flex2", np.nan)),
    }
    for name in ("upper", "left", "right"):
        vals = data.get(name)
        if not isinstance(vals, list) or len(vals) < 6:
            raise ValueError(f"{name} must contain ax, ay, az, gx, gy, gz")
        prefix = f"imu_{name}"
        for i, axis in enumerate(AX):
            row[f"{prefix}_{axis}"] = float(vals[i])
        for i, axis in enumerate(GX):
            row[f"{prefix}_{axis}"] = float(vals[i + 3])
    return row


def make_baseline(df):
    cal = df[df["is_calibration"] == 1].copy()
    if cal.empty:
        return None
    first_t = int(cal["timestamp_ms"].min())
    cal = cal[cal["timestamp_ms"] >= first_t + CALIBRATION_SKIP_SECONDS * 1000]
    if len(cal) < max(5, FS // 2):
        return None
    angles = absolute_angles(cal)
    base = {key: float(np.nanmean(values)) for key, values in angles.items()}
    for col in FLEX_COLS:
        values = cal[col].to_numpy(float)
        valid_values = np.where(flex_valid(values), values, np.nan)
        base[col] = float(np.nanmean(valid_values)) if np.isfinite(valid_values).any() else np.nan
    if not all(np.isfinite(v) for v in base.values()):
        return None
    return base


def derive_live_features(df, baseline):
    """Same derived-channel definitions as posture_features.py, using the live calibration."""
    angles = absolute_angles(df)
    d = pd.DataFrame(angles, index=df.index)
    d["upper_pitch_dev_deg"] = wrap180(d["upper_pitch_deg"] - baseline["upper_pitch_deg"])
    d["upper_roll_dev_deg"] = wrap180(d["upper_roll_deg"] - baseline["upper_roll_deg"])
    left_roll_dev = wrap180(d["left_shoulder_roll_deg"] - baseline["left_shoulder_roll_deg"])
    right_roll_dev = wrap180(d["right_shoulder_roll_deg"] - baseline["right_shoulder_roll_deg"])
    left_pitch_dev = wrap180(d["left_shoulder_pitch_deg"] - baseline["left_shoulder_pitch_deg"])
    right_pitch_dev = wrap180(d["right_shoulder_pitch_deg"] - baseline["right_shoulder_pitch_deg"])
    d["shoulder_asymmetry_dev_deg"] = np.abs(left_roll_dev - right_roll_dev)
    d["shoulder_rounding_dev_deg"] = (left_pitch_dev + right_pitch_dev) / 2 - d["upper_pitch_dev_deg"]

    for raw_col, dev_col in (("flex_lr_raw", "flex_lr_dev"), ("flex_rl_raw", "flex_rl_dev")):
        values = df[raw_col].to_numpy(float)
        d[dev_col] = np.where(
            flex_valid(values) & np.isfinite(baseline[raw_col]),
            values - baseline[raw_col],
            np.nan,
        )
    d["flex_dev_total"] = np.abs(d["flex_lr_dev"]) + np.abs(d["flex_rl_dev"])
    accel = df[[f"imu_upper_{axis}" for axis in AX]].to_numpy(float)
    gyro = df[[f"imu_upper_{axis}" for axis in GX]].to_numpy(float)
    d["accel_magnitude"] = np.sqrt(np.sum(accel ** 2, axis=1))
    d["gyro_magnitude"] = np.sqrt(np.sum(gyro ** 2, axis=1))
    d["thoracic_dev_deg"] = np.abs(d["upper_pitch_dev_deg"])
    return d


def window_vector(window_df, derived_df, artifact_features):
    combined = pd.concat([window_df.reset_index(drop=True), derived_df.reset_index(drop=True)], axis=1)
    arr = combined[WINDOW_CHANNELS].to_numpy(float)
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        low = np.nanmin(arr, axis=0)
        high = np.nanmax(arr, axis=0)
    values = np.column_stack([mean, std, low, high, high - low]).reshape(-1)
    names = [f"{channel}__{stat}" for channel in WINDOW_CHANNELS for stat in STATS]
    vector = pd.DataFrame([values], columns=names)
    missing = [name for name in artifact_features if name not in vector.columns]
    if missing:
        raise ValueError(f"Feature mismatch; missing: {missing[:8]}")
    return vector.loc[:, artifact_features]


def severity_score(X, config):
    refs = config.get("severity_reference_deg", {})
    scores = []
    for col, ref in refs.items():
        if col in X.columns:
            try:
                denom = float(ref)
                if denom > 0:
                    scores.append(abs(float(X.iloc[0][col])) / denom)
            except (TypeError, ValueError):
                pass
    return max(scores, default=0.0)


def update_prediction():
    global latest_prediction, last_prediction_count, poor_hold_seconds, sensor_rule_hold_seconds, warning_state
    if not calibration_complete or calibration_baseline is None:
        latest_prediction.update({
            "activity": "Calibrating",
            "posture": "Waiting",
            "poor_probability": None,
            "warning": "Waiting",
            "window_ready": False,
            "sample_count": len(history),
        })
        return
    if len(history) < WINDOW_ROWS:
        latest_prediction.update({
            "activity": "Collecting window",
            "posture": "Waiting",
            "poor_probability": None,
            "warning": "Waiting",
            "window_ready": False,
            "sample_count": len(history),
        })
        return
    # Predict once per hop (25 new samples = approximately 1 second).
    if len(history) - last_prediction_count < HOP_ROWS:
        return
    frame = pd.DataFrame(list(history))
    raw_window = frame.tail(WINDOW_ROWS).reset_index(drop=True)
    derived_all = derive_live_features(frame, calibration_baseline)
    derived_window = derived_all.tail(WINDOW_ROWS).reset_index(drop=True)
    try:
        X_activity = window_vector(raw_window, derived_window, activity_features)
        activity = str(activity_model.predict(X_activity)[0])
        X_posture = X_activity.copy()
        for act in cfg.get("applicable_activities", ["sitting", "standing", "walking"]):
            X_posture[f"ctx_{act}"] = float(activity == act)
        missing = [col for col in posture_features if col not in X_posture.columns]
        if missing:
            raise ValueError(f"Missing posture context/features: {missing}")
        X_posture = X_posture.loc[:, posture_features]
        p_poor = None
        posture = "not applicable"
        applicable = cfg.get("applicable_activities", ["sitting", "standing", "walking"])
        if activity in applicable:
            classes = list(posture_model.classes_)
            poor_idx = classes.index("poor")
            p_poor = float(posture_model.predict_proba(X_posture)[0, poor_idx])
            posture = "poor" if p_poor >= tau else "good"

        # Approximate persistence logic from the repository configuration.
        if activity in applicable and p_poor is not None and p_poor >= tau:
            poor_hold_seconds += HOP_ROWS / FS
        else:
            poor_hold_seconds = 0.0
        hold_s = float(cfg.get("hold_s", 5.0))
        critical_s = float(cfg.get("critical_sustain_s", hold_s))
        sev = severity_score(X_activity, cfg)

        # Independent, transparent sensor-rule checks in addition to the model.
        # IMU angle deviations are measured relative to the live calibration pose.
        # Flex thresholds are intentionally set for a demonstrator and must be
        # validated with real shirt-mounted recordings before making accuracy claims.
        rule_frame = derived_window
        angle_cols = [
            "upper_pitch_dev_deg",
            "upper_roll_dev_deg",
            "shoulder_asymmetry_dev_deg",
            "shoulder_rounding_dev_deg",
        ]
        angle_deviations = []
        for col in angle_cols:
            if col in rule_frame.columns:
                vals = pd.to_numeric(rule_frame[col], errors="coerce").abs().dropna()
                if not vals.empty:
                    angle_deviations.append(float(vals.mean()))
        imu_deviation_deg = max(angle_deviations, default=0.0)

        flex_vals = pd.to_numeric(rule_frame.get("flex_dev_total", pd.Series(dtype=float)), errors="coerce").dropna()
        flex_deviation = float(flex_vals.mean()) if not flex_vals.empty else 0.0

        # Demo thresholds tuned using the user's flex-bent logger sample.
        # The bent sample's flex means (~2361 and ~2059 ADC) differed by about
        # 90 and 80 counts from the earlier unbent reference readings. Since
        # flex_dev_total sums absolute deviations from the live calibration,
        # the old 80/180 thresholds were too high for reliable flex alerts.
        # These remain experimental thresholds, not clinically validated limits.
        FLEX_WARNING_COUNTS = 25.0
        FLEX_CRITICAL_COUNTS = 55.0
        rule_warning = imu_deviation_deg >= 8.0 or flex_deviation >= FLEX_WARNING_COUNTS
        rule_critical_now = imu_deviation_deg >= 20.0 or flex_deviation >= FLEX_CRITICAL_COUNTS
        if rule_critical_now:
            sensor_rule_hold_seconds += HOP_ROWS / FS
        else:
            sensor_rule_hold_seconds = 0.0
        rule_warning_state = (
            "critical" if rule_critical_now and sensor_rule_hold_seconds >= 5.0
            else "advisory" if (rule_warning or rule_critical_now)
            else "none"
        )

        # Three-state demo logic is driven by deviations from the live calibration.
        # The posture model was trained on synthetic data, so it is shown for
        # comparison but does not override the transparent sensor-rule state.
        if rule_critical_now and sensor_rule_hold_seconds >= 5.0:
            posture = "BAD POSTURE"
            warning_state = "critical"
        elif rule_warning or rule_critical_now:
            posture = "NEUTRAL"
            warning_state = "none"  # neutral is informative, not an alarm
        else:
            posture = "GOOD"
            warning_state = "none"

        if rule_critical_now:
            rule_reason = "Critical sensor deviation (IMU ≥20° or combined flex deviation ≥55 ADC)"
        elif rule_warning:
            rule_reason = "Sensor deviation warning (IMU ≥8° or combined flex deviation ≥25 ADC)"
        else:
            rule_reason = "No sensor-rule threshold exceeded"

        latest_prediction = {
            "activity": activity,
            "posture": posture,
            "poor_probability": p_poor,
            "warning": warning_state,
            "model_note": "Experimental — trained on synthetic data; 3-state label uses live sensor deviations",
            "window_ready": True,
            "sample_count": len(history),
            "threshold": tau,
            "severity_score": round(sev, 3),
            "sensor_rule_warning": bool(rule_warning),
            "sensor_rule_critical": bool(rule_critical_now),
            "sensor_rule_state": rule_warning_state,
            "sensor_rule_reason": rule_reason,
            "imu_deviation_deg": round(imu_deviation_deg, 2),
            "flex_deviation_total": round(flex_deviation, 1),
        }
        last_prediction_count = len(history)
    except Exception as exc:
        latest_prediction.update({
            "activity": "Inference error",
            "posture": str(exc),
            "poor_probability": None,
            "warning": "Unavailable",
            "window_ready": False,
            "sample_count": len(history),
        })
        print("Inference error:", repr(exc))


def serial_reader():
    global latest_reading, last_received, serial_status
    global session_start, calibration_complete, calibration_baseline, calibration_message
    while True:
        port = SERIAL_PORT or find_esp32_port()
        if not port:
            with lock:
                serial_status = "ESP32 port not found"
            time.sleep(3)
            continue
        try:
            print(f"Opening ESP32 serial port: {port}")
            with serial.Serial(port, BAUD_RATE, timeout=1) as connection:
                time.sleep(2)
                connection.reset_input_buffer()
                with lock:
                    serial_status = f"Connected: {port}"
                print("Waiting for JSON sensor readings...")
                while True:
                    line = connection.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    required = {"upper", "left", "right", "flex1", "flex2"}
                    if not required.issubset(data):
                        continue
                    try:
                        now = time.time()
                        with lock:
                            if session_start is None:
                                session_start = now
                            elapsed = now - session_start
                            is_cal = int(elapsed < CALIBRATION_SECONDS)
                            row = row_from_reading(data, int(elapsed * 1000), is_cal)
                            latest_reading = data
                            last_received = now
                            history.append(row)
                            if not calibration_complete and elapsed >= CALIBRATION_SECONDS:
                                baseline = make_baseline(pd.DataFrame(list(history)))
                                if baseline is not None:
                                    calibration_baseline = baseline
                                    calibration_complete = True
                                    calibration_message = "Calibration complete"
                                    print("Calibration complete; live inference enabled.")
                                else:
                                    calibration_message = "Calibration readings invalid; check flex sensors and restart."
                            update_prediction()
                    except Exception as exc:
                        print("Sensor processing error:", repr(exc))
        except (serial.SerialException, OSError) as error:
            with lock:
                serial_status = f"Serial disconnected: {error}"
            print(f"Serial error: {error}")
            time.sleep(3)


HTML = r"""
<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spine-Grid | Live Posture Monitor</title>
<style>
:root{--ink:#17243a;--muted:#66758b;--line:#e3e9f2;--bg:#f3f6fb;--blue:#2867c7;--red:#b42318;--amber:#9a6700;--green:#137547}
*{box-sizing:border-box}body{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:22px}
main{max-width:1250px;margin:auto}h1{margin:0 0 5px;font-size:clamp(27px,4vw,36px)}h2{font-size:1.08rem;margin:0 0 12px}h3{font-size:1rem;margin:0 0 7px}
.muted{color:var(--muted);font-size:.9rem}.small{font-size:.78rem}
.status,.card,.architecture,.layer,.alert-panel{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 3px 12px #20305008}
.status{margin:16px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap;justify-content:space-between}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:13px;margin:14px 0}
.big{font-size:1.5rem;font-weight:800;margin:7px 0;overflow-wrap:anywhere}.axis{display:flex;justify-content:space-between;gap:10px;padding:6px 0;border-bottom:1px solid #edf0f5;font-variant-numeric:tabular-nums}.axis span{font-weight:650}
.pill{display:inline-block;border-radius:999px;padding:5px 10px;background:#e9eef6;font-weight:750;font-size:.76rem}
button{border:0;border-radius:9px;padding:10px 13px;background:var(--blue);color:white;font-weight:750;cursor:pointer}
#alertBanner{display:none;border-radius:13px;padding:15px 17px;margin:14px 0;font-weight:750}
#alertBanner.advisory{display:block;background:#fff4d6;color:#754b00;border:1px solid #f1d38a}
#alertBanner.critical{display:block;background:#fee4e2;color:#912018;border:1px solid #f5aaa4;animation:pulse 1.1s infinite}
#alertBanner.none{display:none}
@keyframes pulse{50%{box-shadow:0 0 0 4px #f5aaa455}}
.section{margin:25px 0 12px}.arch-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
.layer{position:relative;box-shadow:none}.layer .num{font-size:.72rem;color:var(--blue);font-weight:850;letter-spacing:.08em}.layer p{margin:7px 0 0;font-size:.86rem;color:var(--muted)}
.flow{font-size:1.3rem;text-align:center;color:#9aabc1;align-self:center}
.arch-note{margin-top:12px;background:#eef5ff;border:1px solid #d4e4ff;border-radius:10px;padding:12px;font-size:.85rem;color:#29476f}
pre{background:#17243a;color:#e7f1ff;padding:14px;border-radius:10px;overflow-wrap:anywhere;white-space:pre-wrap;max-height:220px;overflow:auto}
@media(max-width:850px){.arch-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.flow{display:none}}
@media(max-width:520px){body{padding:12px}.arch-grid{grid-template-columns:1fr}.grid{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Spine-Grid</h1>
<p class="muted">Live wearable sensor monitor · real ESP32 readings · experimental posture inference</p>
<div class="status">
 <div>Connection: <strong id="status">Connecting...</strong><br><span class="muted" id="updated">Waiting for sensor data</span><br><span class="muted" id="calibration">Calibration pending</span></div>
 <button id="soundButton" onclick="enableAlerts()">Enable sound alerts</button>
</div>
<div id="alertBanner" class="none" role="alert" aria-live="assertive">Waiting for posture status…</div>
<div class="grid">
 <div class="card"><h2>Activity</h2><div class="big" id="activity">Waiting</div><span class="pill">Live model output</span></div>
 <div class="card"><h2>Posture status</h2><div class="big" id="posture">Waiting</div><div class="muted" id="probability">Poor-posture probability: --</div><span class="pill">Experimental model</span></div>
 <div class="card"><h2>Warning / alert</h2><div class="big" id="warning">Waiting</div><div class="muted" id="severity">Severity: --</div><div class="muted small" id="alertDetail">Alerts appear when the model flags sustained poor posture.</div></div>
</div>

<div class="section"><h2>SPINE architecture</h2><p class="muted">Architecture mapping for the prototype demo</p></div>
<div class="architecture">
 <div class="arch-grid">
  <div class="layer"><div class="num">LAYER 01</div><h3>Sensor nodes</h3><b>3 × MPU6050 + 2 flex</b><p>ESP32-S3 reads upper-back, left/right shoulder motion and lower-back bend.</p></div>
  <div class="layer"><div class="num">LAYER 02</div><h3>Communication</h3><b>USB serial · JSON</b><p>Live sensor packet is sent from the ESP32 to the Python dashboard process.</p></div>
  <div class="layer"><div class="num">LAYER 03</div><h3>Processing / coordinator</h3><b>Calibration + features</b><p>Python builds windows, derives posture features and runs activity/posture models.</p></div>
  <div class="layer"><div class="num">LAYER 04</div><h3>Application / alert</h3><b>Flask dashboard</b><p>Displays live axes, model output, poor-posture warning and browser alert.</p></div>
 </div>
 <div class="arch-note"><b>Implementation note:</b> this live dashboard currently uses the direct USB-serial data path. The diagram maps the project's SPINE-style sensor/coordinator/application layers; it does not claim that the Java SPINE coordinator middleware is carrying these particular dashboard packets.</div>
</div>

<div class="section"><h2>Live IMU readings</h2><p class="muted">Acceleration in g · gyroscope in degrees/second</p></div>
<div class="grid">
 <div class="card"><h2>Upper-back IMU</h2>
 <div class="axis">AX (g)<span id="u0">--</span></div><div class="axis">AY (g)<span id="u1">--</span></div><div class="axis">AZ (g)<span id="u2">--</span></div>
 <div class="axis">GX (°/s)<span id="u3">--</span></div><div class="axis">GY (°/s)<span id="u4">--</span></div><div class="axis">GZ (°/s)<span id="u5">--</span></div></div>
 <div class="card"><h2>Left-shoulder IMU</h2>
 <div class="axis">AX (g)<span id="l0">--</span></div><div class="axis">AY (g)<span id="l1">--</span></div><div class="axis">AZ (g)<span id="l2">--</span></div>
 <div class="axis">GX (°/s)<span id="l3">--</span></div><div class="axis">GY (°/s)<span id="l4">--</span></div><div class="axis">GZ (°/s)<span id="l5">--</span></div></div>
 <div class="card"><h2>Right-shoulder IMU</h2>
 <div class="axis">AX (g)<span id="r0">--</span></div><div class="axis">AY (g)<span id="r1">--</span></div><div class="axis">AZ (g)<span id="r2">--</span></div>
 <div class="axis">GX (°/s)<span id="r3">--</span></div><div class="axis">GY (°/s)<span id="r4">--</span></div><div class="axis">GZ (°/s)<span id="r5">--</span></div></div>
 <div class="card"><h2>Flex sensor 1</h2><div class="big" id="f1">--</div><div class="muted">ADC counts</div></div>
 <div class="card"><h2>Flex sensor 2</h2><div class="big" id="f2">--</div><div class="muted">ADC counts</div></div>
</div>
<p class="muted">The models currently load from the project's trained artifacts, which were trained on synthetic data. Live inputs are real hardware readings; posture predictions remain experimental and are not validated for real-world accuracy.</p>
<h2>Latest ESP32 JSON</h2><pre id="raw">Waiting for data...</pre>
</main>
<script>
let audioContext=null, soundEnabled=false, lastAlertKey="none";
function enableAlerts(){
 try{
  const AC=window.AudioContext||window.webkitAudioContext;
  if(!AC){document.getElementById('soundButton').textContent='Sound unavailable';return;}
  audioContext=new AC(); if(audioContext.state==='suspended') audioContext.resume();
  soundEnabled=true; document.getElementById('soundButton').textContent='Sound alerts enabled';
  beep(660,0.09);
 }catch(e){document.getElementById('soundButton').textContent='Sound unavailable';}
}
function beep(freq,duration){
 if(!soundEnabled||!audioContext)return;
 const osc=audioContext.createOscillator(), gain=audioContext.createGain();
 osc.type='sine';osc.frequency.value=freq;gain.gain.value=0.08;
 osc.connect(gain);gain.connect(audioContext.destination);osc.start();
 osc.stop(audioContext.currentTime+duration);
}
function handleAlert(p){
 const banner=document.getElementById('alertBanner');
 const warning=String(p.warning||'Waiting').toLowerCase();
 const poor=String(p.posture||'').toLowerCase()==='poor';
 const active=warning==='advisory'||warning==='critical';
 if(active){
  banner.className=warning;
  banner.textContent=warning==='critical'
   ? '🚨 POSTURE ALERT: Poor posture has persisted. Adjust your posture and reset to a comfortable neutral position.'
   : '⚠️ POSTURE NOTICE: The current model flags possible poor posture. Sit comfortably upright and check your position.';
  if(!p.sensor_rule_reason) document.getElementById('alertDetail').textContent='Alert status is generated from the live model output and configured persistence threshold.';
 }else if(p.window_ready){
  banner.className='none';banner.textContent='';
  if(!p.sensor_rule_reason) document.getElementById('alertDetail').textContent=poor?'Model currently flags poor posture; warning persistence threshold has not been reached.':'No posture warning from the current model window.';
 }else{
  banner.className='none';banner.textContent='';
 }
 const key=active?warning:'none';
 if(key!==lastAlertKey && active){
  beep(warning==='critical'?880:620,0.18);
  if(warning==='critical'){setTimeout(()=>beep(880,0.18),240);}
 }
 lastAlertKey=key;
}
async function updateDashboard(){
 try{
  const response=await fetch('/api/latest',{cache:'no-store'});const data=await response.json();
  document.getElementById('status').textContent=data.status||'Unknown';
  document.getElementById('calibration').textContent=data.calibration_message||'';
  const p=data.prediction||{};
  document.getElementById('activity').textContent=p.activity||'Waiting';
  document.getElementById('posture').textContent=p.posture||'Waiting';
  document.getElementById('warning').textContent=p.warning||'Waiting';
  document.getElementById('probability').textContent='Poor-posture probability: '+(p.poor_probability==null?'--':(100*p.poor_probability).toFixed(1)+'%');
  document.getElementById('severity').textContent='Model severity score: '+(p.severity_score==null?'--':p.severity_score);
  if(p.sensor_rule_reason){
    document.getElementById('alertDetail').textContent =
      p.sensor_rule_reason + ' · IMU deviation: ' + (p.imu_deviation_deg ?? '--') +
      '° · combined flex deviation: ' + (p.flex_deviation_total ?? '--') + ' ADC';
  }
  handleAlert(p);
  if(!data.reading)return;
  const d=data.reading;
  [['upper','u'],['left','l'],['right','r']].forEach(([s,prefix])=>{
   const v=d[s]||[];for(let i=0;i<6;i++)document.getElementById(prefix+i).textContent=(v[i]===undefined?'--':Number(v[i]).toFixed(3));
  });
  document.getElementById('f1').textContent=d.flex1??'--';
  document.getElementById('f2').textContent=d.flex2??'--';
  document.getElementById('raw').textContent=JSON.stringify(d,null,2);
  document.getElementById('updated').textContent='Last reading: '+new Date(data.received_at*1000).toLocaleTimeString()+' · samples: '+(p.sample_count??0);
 }catch(e){document.getElementById('status').textContent='Cannot reach Flask';}
}
updateDashboard();setInterval(updateDashboard,500);
</script></body></html>
"""


@app.route("/")
def dashboard():
    return render_template_string(HTML)


@app.route("/api/latest")
def latest():
    with lock:
        age = time.time() - last_received if last_received else None
        status = "No recent sensor data" if age is not None and age > 3 else serial_status
        return jsonify(
            status=status,
            reading=latest_reading,
            received_at=last_received,
            prediction=dict(latest_prediction),
            calibration_message=calibration_message,
        )


if __name__ == "__main__":
    print("Models loaded.")
    print(f"Activity features: {len(activity_features)}; posture features: {len(posture_features)}")
    print("Keep the Arduino Serial Monitor CLOSED while this dashboard reads the ESP32.")
    threading.Thread(target=serial_reader, daemon=True).start()
    app.run(host="127.0.0.1", port=5001, debug=False)
