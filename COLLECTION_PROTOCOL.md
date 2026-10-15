# Collection and annotation protocol

This protocol is written for **real** recordings with the Spine-Grid Tee. The synthetic dataset follows the same structure so that the same scripts work unchanged once real data exists. Section 8 explains how the synthetic data was produced.

## 1. Hardware and mounting

- ESP32-S3 at the upper-back hub. Three MPU6050 boards: upper back (between the shoulder blades), left shoulder blade, right shoulder blade. Two flex sensors forming an X over the lumbar region (`flex_lr`: left shoulder side → right hip; `flex_rl`: right shoulder side → left hip). No lumbar IMU.
- Every MPU6050 lies flat on the back, chip side facing out, printed **Y arrow pointing up toward the neck**. Sensor frame: +Y up the spine, +Z out of the back, +X to the wearer's right.
- MPU6050 configured to ±2 g and ±250 °/s, digital low-pass filter ~21 Hz. Flex dividers read by 12-bit ADC1 pins at 11 dB attenuation (0–4095).

## 2. Sampling and synchronization

- All channels are read in one loop at 25 Hz (40 ms). One CSV row holds one read of every sensor plus one timestamp.
- For real data: log the ESP32 `millis()` per row, convert to epoch ms using a time sync at session start, and resample to an exact 40 ms grid by linear interpolation *only* if the jitter exceeds 5 ms. Document any resampling per session.
- A failed I2C read leaves that IMU's six cells blank and adds `imu_<name>_missing` to `quality_flags`. A flex reading of exactly 0 or 4095 is flagged invalid (open or shorted connector).

## 3. Session structure

Each participant records **two sessions on different days**. Each session (about 5 minutes) consists of:

1. **Calibration (10 s):** stand upright and still in the posture the participant considers good. The baseline is the mean over seconds 2–10.
2. **Activity trials** in a fixed order (two orders alternate between sessions): sitting (60–80 s), standing (35–45 s), walking (40–50 s), running (25–30 s), walking, standing, lying down (supine → side-lying → supine, 30–40 s), sitting.
3. **Transitions** between trials (sit↔stand, stand↔walk, walk↔run, to/from lying; 1.5–3.5 s) get their own label, `transition`.

Within sitting, standing and walking trials, the participant performs scripted posture episodes on cue:

| Episode | When | Durations |
| --- | --- | --- |
| Slouching (thoracic + lumbar flexion) | sitting, standing, walking (phone use) | short 2–4.5 s, medium 5.5–9 s, long 12–35 s |
| Mild slouch (below criterion) | sitting, walking | 5.5–9 s |
| Excessive forward bending (picking up an object) | standing | 2.5–4.5 s and 6–10 s |
| Excessive backward bending (leaning back) | standing | 5.5–9 s |
| Leaning left / right | sitting, standing | 5.5–14 s |
| Shoulder asymmetry (bag on one shoulder, leaning on an armrest) | sitting, standing, walking | 8–20 s |
| Rounded shoulders (reason `other`) | sitting | 5.5–9 s |
| Borderline versions (magnitude close to the criterion) | sitting, standing | 4–9 s |

Natural movement (fidgeting, breathing, turning while walking) is kept and is never labelled poor by itself.

## 4. Ground truth for real recordings

Record a synchronized side-view and back-view video (or a marker-based system). Annotators mark episode start, plateau and end, then measure the deviation from the calibration pose on video. **Ground-truth labels must never be derived from the firmware thresholds or outputs.**

## 5. Annotation rules

Criteria below are **initial posture-monitoring categories, not medical thresholds**. They should be reviewed by a physiotherapist or ergonomics professional before real collection.

| Reason | Criterion (deviation from calibrated neutral) |
| --- | --- |
| slouching | thoracic flexion ≥ 12° |
| excessive_forward_bending | trunk forward bend ≥ 30° |
| excessive_backward_bending | trunk extension ≥ 15° |
| leaning_left / leaning_right | side lean ≥ 10° |
| shoulder_asymmetry | difference between left and right shoulder tilt ≥ 10° |
| other (rounded shoulders) | shoulder protraction ≥ 12° beyond the spine |

- `posture_quality`
  - `good`: sitting, standing or walking with no deviation at or above a criterion, including mild episodes and normal movement.
  - `poor`: during the plateau of an episode whose magnitude is clearly above the criterion (more than 1.5° beyond it).
  - `uncertain`: during the ramp into or out of an episode, or for borderline magnitudes within ±1.5° of the criterion.
  - `not_applicable`: running, lying_down and transitions. Trunk lean while running is normal and is not judged.
- `safety_label`
  - `unsafe`: poor posture that has continued for **≥ 5 s**, from second 5 onward.
  - `uncertain`: where posture quality is uncertain.
  - `safe`: everything else, including short poor episodes under 5 s.
- `warning_level`
  - `critical`: unsafe and either the magnitude is ≥ 1.5 × the criterion or the poor posture has lasted ≥ 20 s.
  - `advisory`: unsafe otherwise.
  - `none`: all other rows.

## 6. Firmware configuration (metadata only)

WARN_DEG = 8, CRITICAL_DEG = 16, FLEX_WARN = 400, FLEX_CRITICAL = 900, BAD_POSTURE_HOLD_MS = 5000. The firmware columns reproduce the unmodified on-shirt rules, with no activity-specific changes. Any activity-specific thresholds belong in the ML warning logic (`models/warning_config.json`), not silently in the firmware.

## 7. Participant-independent split plan

| Split | Participants | Use |
| --- | --- | --- |
| Train | P01–P05 | Fit models |
| Validation | P06 | Choose hyper-parameters, the P(poor) threshold and the critical-severity factor |
| Test | P07–P08 | Scored once, at the end |

Windows never cross sessions, and no participant appears in two splits. With real data, use leave-one-participant-out cross-validation as well, because 8 participants is small.

## 8. How the synthetic data was produced (`generate_dataset.py`, seed 20261006)

- **Participants:** 8, each with randomized height, sensor-mounting offsets (0–14°), habitual sitting and standing posture, gait and running cadence (1.72–1.98 / 2.55–2.95 Hz), flex baseline (2500–2950 counts), flex sensitivity (24–36 counts per degree of lumbar flexion), gyro bias (σ 1.5 °/s), accelerometer bias (σ 12 mg) and noise level.
- **Sessions:** 16, two per day from 6 to 13 Oct 2026 (IST), each 4.7–5.2 minutes.
- **Physics:**
  - Accelerometer = gravity projected onto the simulated tilt of each board, (−sin φ, cos θ cos φ, sin θ cos φ), plus gait acceleration (walking ±0.2 g vertical; running impact peaks up to ~1.7 g, clipped at the ±2 g sensor limit), bias and noise (~4 mg).
  - Gyroscope = rate of change of the simulated angles, plus yaw from walking turns and fidgets, bias and noise.
  - Flex = baseline − sensitivity × lumbar flexion ∓ side-bend term, plus slow drift (σ 15 counts) and noise (σ 10 counts).
  - Values are quantized to MPU6050 / ADC resolution.
- **Imperfections:**
  - 1–5-sample IMU dropouts (~0.06% chance per sample per IMU), plus an occasional 0.5–1.5 s connector dropout.
  - Single-sample flex spikes to 0 or 4095.
  - Accelerometer saturation while running.
  - Slow OU-process drift in posture, sensor drift, breathing at 0.2–0.3 Hz.
- **Labels:** come from the scripted episode magnitudes and timings using the rules in §5, never from the simulated sensor values or firmware.
