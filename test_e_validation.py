"""Test E: end-to-end artifact and configuration validation.

Run AFTER test_inference.py (it checks that script's output):
    python test_e_validation.py        (or:  pytest test_e_validation.py)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
PREDICTIONS = ROOT / "test_inference_predictions.csv"

ACTIVITIES = {"sitting", "standing", "walking", "running", "lying_down"}
APPLICABLE = {"sitting", "standing", "walking"}


def test_end_to_end():
    print("Test E: End-to-end artifact and configuration validation\n")

    # 1. Required files
    required = [MODELS / "activity_model.joblib", MODELS / "posture_model.joblib",
                MODELS / "warning_config.json", REPORTS / "metrics.json", REPORTS / "REPORT.md"]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, f"Missing files: {missing} - run train_models.py first"
    assert PREDICTIONS.exists(), f"Missing {PREDICTIONS.name} - run test_inference.py first"
    print("PASS: All required files exist")

    # 2. Models, configuration, metrics
    activity = joblib.load(MODELS / "activity_model.joblib")
    posture = joblib.load(MODELS / "posture_model.joblib")
    config = json.loads((MODELS / "warning_config.json").read_text())
    metrics = json.loads((REPORTS / "metrics.json").read_text())

    # 3. Feature counts
    assert len(activity["features"]) == 165 and activity["model"].n_features_in_ == 165
    assert len(posture["features"]) == 168 and posture["model"].n_features_in_ == 168
    print("PASS: Model feature counts match saved pipelines")

    # 4. Classes
    assert set(activity["model"].classes_) == ACTIVITIES
    assert set(posture["model"].classes_) == {"good", "poor"}
    print("PASS: Activity and posture classes are correct")

    # 5. Context features
    assert {"ctx_sitting", "ctx_standing", "ctx_walking"}.issubset(posture["features"])
    assert set(activity["features"]).issubset(posture["features"])
    print("PASS: Posture model includes activity context features")

    # 6. Warning configuration
    assert 0 < config["tau_poor"] < 1
    assert config["hold_s"] == 5.0
    assert config["critical_sustain_s"] == 20.0
    assert config["hop_s"] == 1
    assert config["critical_severity"] > 1
    assert set(config["applicable_activities"]) == APPLICABLE
    assert np.isclose(config["tau_poor"], posture.get("tau", float("nan")))
    print("PASS: Warning configuration matches posture model")

    # 7. Recorded metrics are internally consistent
    for key in ("model_C_ml", "model_C_firmware"):
        m = metrics[key]
        assert m["events"] == m["detected"] + m["missed"]
        assert np.isclose(m["recall"], m["detected"] / max(m["events"], 1))
    assert 0 <= metrics["model_A_test"]["accuracy"] <= 1
    assert 0 <= metrics["model_A_test"]["macro_f1"] <= 1
    print("PASS: Warning metrics are internally consistent")

    # 8. Inference predictions
    pred = pd.read_csv(PREDICTIONS)
    needed = {"activity_prediction", "posture_prediction", "p_poor", "warning_prediction",
              "warning_threshold", "fw_alert_frac"}
    assert needed.issubset(pred.columns), f"Missing columns: {needed - set(pred.columns)}"
    assert len(pred) > 0
    assert set(pred["activity_prediction"]).issubset(ACTIVITIES)
    assert set(pred["posture_prediction"]).issubset({"good", "poor", "not_applicable"})
    assert set(pred["warning_prediction"]).issubset({"none", "advisory", "critical"})
    assert pred["p_poor"].dropna().between(0, 1).all()
    print("PASS: Inference predictions contain valid labels")

    # 9. Posture labels follow the tuned threshold (catches use of predict()'s 0.5 cut-off)
    tau = config["tau_poor"]
    assert np.allclose(pred["warning_threshold"], tau)
    gp = pred["posture_prediction"].isin(["good", "poor"])
    agree = (pred.loc[gp, "p_poor"] >= tau) == (pred.loc[gp, "posture_prediction"] == "poor")
    assert agree.all(), f"{int((~agree).sum())} posture labels disagree with p_poor >= tau"
    print(f"PASS: Posture labels follow tau = {tau:.2f}")

    # 10. Activity gating: no posture score or alert outside sitting / standing / walking
    na = ~pred["activity_prediction"].isin(APPLICABLE)
    assert (pred.loc[na, "posture_prediction"] == "not_applicable").all()
    assert pred.loc[na, "p_poor"].isna().all()
    assert (pred.loc[na, "warning_prediction"] == "none").all()
    print("PASS: Running / lying windows are never scored or alerted")

    # 11. Firmware comparison column carried through
    assert pred["fw_alert_frac"].notna().all()
    print("PASS: Firmware alert fraction present for every window")

    print(f"PASS: Validated {len(pred)} prediction windows")
    print("\nRESULT: TEST E PASSED")
    print("Note: consistency checks do not establish real-world accuracy or validate "
          "physical-device warning timing.")


if __name__ == "__main__":
    test_end_to_end()
