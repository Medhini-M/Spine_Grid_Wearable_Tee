# Spine-Grid Tee — posture dataset (SYNTHETIC) and models

**Every row in this dataset is synthetic** (`data_source = synthetic`). It is for building and testing the ML pipeline before real recordings exist. See LIMITATIONS.md.

## Files

| File | What it is |
| --- | --- |
| `spine_grid_posture_synthetic.csv` | 119,246 rows × 55 columns. 25 Hz, 3 MPU6050 IMUs + 2 flex sensors, 8 participants, 16 sessions, 6–13 Oct 2026 |
| `participants_synthetic.csv` | Participant ids and heights |
| `DATA_DICTIONARY.md` | Every column: unit, range, formula, purpose |
| `COLLECTION_PROTOCOL.md` | Mounting, coordinate system, sampling, sessions, annotation rules, firmware thresholds, split plan, how the synthetic data was made |
| `DATASET_SUMMARY.md` | Sample counts per activity / posture / safety / warning class, coverage per participant, sessions, missing and invalid readings, firmware behaviour per activity |
| `LIMITATIONS.md` | Dataset limitations and the real data needed before deployment |
| `posture_features.py` | Reproducible feature extraction (angles, deviations, flex deviations, magnitudes), firmware-rule simulation, 2 s windowing |
| `generate_dataset.py` | Regenerates the dataset exactly (seed 20261006) |
| `train_models.py` | Trains and evaluates Models A, B and C with a participant-independent split |
| `test_inference.py` | Test C: runs the saved models on the held-out participants and scores them |
| `test_e_validation.py` | Test E: consistency checks on models, configuration, metrics and predictions |
| `models/` | Trained activity model, posture model, warning configuration |
| `reports/REPORT.md`, `reports/metrics.json` | Training results |

## Run it

```
pip install -r requirements.txt
python generate_dataset.py      # rebuilds the CSV + DATASET_SUMMARY.md (~10 s)
python train_models.py          # trains and evaluates; writes models/ and reports/ (~40 s)
python test_inference.py        # Test C: models A+B+C on P07/P08 -> test_inference_predictions.csv
python test_e_validation.py     # Test E: checks models, config, metrics and predictions agree
```

The dataset CSV and `models/*.joblib` are not stored in git; the first two commands rebuild them.
SHA-256 of the generated CSV: `987d80279d669b06c23ddab3424fc3763362c5819402e0d8ccfefe6cf9fb3639`

## Recommended model: a context-aware, three-stage pipeline

1. **Model A: activity (Random Forest, 300 trees).** Input: 2 s windows with 50% overlap, using the mean, std, min, max and range of 33 channels (165 features). It tells the rest of the pipeline whether posture rules apply at all.
2. **Model B: posture quality (Random Forest, balanced classes).** Same window features plus the activity from Model A. Trained only on sitting, standing and walking; running, lying and transitions are `not_applicable`. The decision threshold on P(poor) is tuned on the validation participant.
3. **Model C: warning logic (transparent state machine, not a black box).** It alerts when the activity is sitting, standing or walking and P(poor) stays above the threshold for 5 s. The alert becomes critical after 20 s, or when the deviation is far beyond the posture criterion.

**Why Random Forest:**
- It works well with small datasets.
- It needs no feature scaling and tolerates noisy sensor features.
- It shows which sensors matter.
- It can be exported to C for the ESP32 (for example with `micromlgen`; keep about 20–50 trees and max depth about 10 on the board).

ROCKET, a time-series model built from random convolution kernels, and gradient boosting were compared. Neither beat the Random Forest on the held-out participants. A 1D-CNN or TCN is worth revisiting only once real data reaches several hours per class.

## Results on the synthetic test participants (P07, P08)

| | Result |
| --- | --- |
| Model A activity | accuracy 0.990, macro-F1 0.989 (RF); ROCKET 0.978; gradient boosting 0.969 |
| Model B posture | accuracy 0.971 (good F1 0.979, poor F1 0.957), same with Model A's predicted activity |
| Model C vs firmware | unsafe events caught 18/20 vs 11/20; false alerts during good posture 1 vs 4 (3.0 vs 11.9 per hour) |
| Warning level | advisory/critical separation is weak (macro-F1 0.64) — see LIMITATIONS.md |

These numbers validate the pipeline on synthetic data. **They are not evidence of real-world accuracy.**
