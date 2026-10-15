# Data dictionary — `spine_grid_posture.csv`

One row = one synchronized reading of all sensors at 25 Hz (40 ms grid). 55 columns. Blank cell = missing value (never zero-filled).

Sensor frame for every MPU6050 (board flat on the back, chip side out, printed Y arrow up the spine):
**+Y** up the spine, **+Z** out of the back (posterior), **+X** toward the wearer's right. Upright and still reads about (0, +1, 0) g.

## Metadata and ground-truth labels

| Column | Type / unit | Values | Purpose |
| --- | --- | --- | --- |
| `data_source` | text | `-` | Marks every row as generated data |
| `timestamp_ms` | int, ms | Unix epoch milliseconds | Sample time; strictly increasing by 40 within a session |
| `timestamp` | ISO 8601, IST (+05:30) | 2026-10-06 … 2026-10-13 | Human-readable form of `timestamp_ms` |
| `subject_id` | text | P01 … P08 | Anonymous participant |
| `session_id` | text | e.g. `P03_S2` | Recording session (2 per participant) |
| `trial_id` | text | `<session>_CAL`, `<session>_Tnn`, `<session>_TRnn` | Calibration trial, activity trial, or transition |
| `is_calibration` | 0/1 | | 1 = the 10 s standing calibration trial at the start of each session |
| `activity_label` | text | sitting, standing, walking, running, lying_down, transition | Ground-truth activity. Transitions (sit↔stand, stand↔walk, walk↔run, to/from lying) have their own label |
| `posture_quality` | text | good, poor, uncertain, not_applicable | Ground truth from the annotation protocol (see COLLECTION_PROTOCOL.md §5). `not_applicable` for running, lying_down and transitions |
| `safety_label` | text | safe, unsafe, uncertain | `unsafe` = poor posture sustained ≥ 5 s |
| `unsafe_reason` | text | none, slouching, excessive_forward_bending, excessive_backward_bending, leaning_left, leaning_right, shoulder_asymmetry, other | Why the row is unsafe (`other` = rounded/protracted shoulders). `none` when not unsafe |
| `warning_level` | text | none, advisory, critical | Ground-truth warning the system *should* give |

## Raw sensor columns

| Column | Unit | Range / resolution | Sensor |
| --- | --- | --- | --- |
| `imu_upper_ax`, `_ay`, `_az` | g | ±2 g, 1/16384 g steps | Upper-back MPU6050 (hexagon hub) accelerometer |
| `imu_upper_gx`, `_gy`, `_gz` | °/s | ±250 °/s, 1/131 °/s steps | Upper-back gyroscope (gx: forward/back bend rate, gy: twist/yaw, gz: side-lean rate) |
| `imu_left_ax` … `imu_left_gz` | g, °/s | as above | Left-shoulder MPU6050 (left triangle) |
| `imu_right_ax` … `imu_right_gz` | g, °/s | as above | Right-shoulder MPU6050 (right triangle) |
| `flex_lr_raw` | ADC counts | 0–4095 (12-bit, 11 dB attenuation) | Lumbar-X flex strip, left shoulder side → right hip. Bending the lower back forward lowers the reading |
| `flex_rl_raw` | ADC counts | 0–4095 | Lumbar-X flex strip, right shoulder side → left hip |

No lumbar IMU is fitted, so the optional `imu_lumbar_*` columns and `lumbar_pitch_dev_deg` / `lumbar_dev_deg` are **not included**. Lumbar information comes only from the two flex sensors.

## Derived per-sample features (reproducible: `posture_features.compute_derived`)

| Column | Unit | Formula |
| --- | --- | --- |
| `upper_pitch_deg` | ° | atan2(az, ay) of the upper IMU: + = bent forward, − = bent backward |
| `upper_roll_deg` | ° | atan2(−ax, √(ay² + az²)): + = leaning to the wearer's right |
| `left_shoulder_pitch_deg`, `left_shoulder_roll_deg` | ° | Same formulas on the left IMU |
| `right_shoulder_pitch_deg`, `right_shoulder_roll_deg` | ° | Same formulas on the right IMU |
| `upper_pitch_dev_deg` | ° | upper_pitch − calibration mean, wrapped to ±180 |
| `upper_roll_dev_deg` | ° | upper_roll − calibration mean |
| `shoulder_asymmetry_dev_deg` | ° | \|(left_roll − its calibration) − (right_roll − its calibration)\| |
| `shoulder_rounding_dev_deg` | ° | mean of the two shoulder pitch deviations − upper_pitch_dev (shoulders tipping forward relative to the spine) |
| `flex_lr_dev`, `flex_rl_dev` | ADC counts | raw − that sensor's calibration mean; blank if the reading is invalid (0 or 4095). **Not degrees**: no validated counts-to-angle calibration exists |
| `flex_dev_total` | ADC counts | \|flex_lr_dev\| + \|flex_rl_dev\| (same as the firmware) |
| `accel_magnitude` | g | √(ax² + ay² + az²) of the upper IMU |
| `gyro_magnitude` | °/s | √(gx² + gy² + gz²) of the upper IMU |
| `thoracic_dev_deg` | ° | \|upper_pitch_dev_deg\| (warning-system feature) |

Calibration baseline = mean over the calibration trial after its first 2 s. Any derived value whose inputs are missing or invalid is blank.

## Firmware-output columns (simulation of the unmodified on-shirt rules)

These columns show what the shirt's own rules would do. They are **not** ground truth and were generated independently of the labels.

| Column | Values | Meaning |
| --- | --- | --- |
| `warning_condition` | 0/1 | Any of: \|forward bend\|, \|side lean\|, \|shoulder rounding\|, \|shoulder asymmetry\| > 8°, or flex total > 400 |
| `critical_condition` | 0/1 | Any angle > 16° or flex total > 900 |
| `critical_duration_ms` | ms | Continuous time critical_condition has been true; resets to 0 when it clears |
| `alert_active` | 0/1 | critical_duration_ms ≥ 5000 |
| `led_state` | green / yellow / red | red = alert, yellow = warning or critical, green otherwise |
| `buzzer_state` | 0/1 | Vibration-motor output: on 400 ms of every 1500 ms while the alert is active |

The firmware evaluates every 200 ms (every 5th row) and holds its outputs in between. It uses its own tilt formulas (side = atan2(−ax, ay)) and baseline (last 2 s of calibration), and holds the last value on a failed IMU read.

## Data quality

| Column | Values |
| --- | --- |
| `quality_flags` | `ok`, or a `;`-separated list of: `imu_<upper/left/right>_missing`, `flex_lr_invalid`, `flex_rl_invalid`, `accel_saturated_<imu>` (some accelerometer axis at the ±2 g limit, mostly during running) |

Companion file `participants_synthetic.csv`: subject_id and height (cm) of each synthetic participant.
