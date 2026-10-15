"""Generate the SYNTHETIC Spine-Grid Tee posture dataset (pipeline-testing data, based on real recordings).

    python generate_dataset.py            # -> spine_grid_posture_synthetic.csv + DATASET_SUMMARY.md

The physics model (gravity projection onto each IMU,
gait-induced acceleration, gyro = angular rate of the simulated trunk/shoulder angles, flex
ADC = baseline - sensitivity x lumbar bend) is documented in COLLECTION_PROTOCOL.md.
Ground-truth labels come from the scripted episodes and the annotation protocol, never from
the firmware thresholds.
"""
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from posture_features import (FIRMWARE_COLS, FLEX_COLS, FS, RAW_IMU_COLS, compute_derived,
                              simulate_firmware)

SEED = 20261006
DT = 1.0 / FS
IST = timezone(timedelta(hours=5, minutes=30))
IMU_KEYS = ["upper", "left", "right"]

# ---------------------------------------------------------------- annotation protocol criteria
# (true deviation from the calibrated neutral pose; magnitudes in degrees)
CRITERIA = {
    "slouching": 12.0,                   # thoracic flexion
    "excessive_forward_bending": 30.0,   # trunk forward bend
    "excessive_backward_bending": 15.0,  # trunk extension
    "leaning_left": 10.0, "leaning_right": 10.0,
    "shoulder_asymmetry": 10.0,
    "other": 12.0,                       # rounded shoulders (protraction) - reason "other"
}
BORDER = 1.5          # |magnitude - criterion| < 1.5 deg  -> "uncertain"
SUSTAIN_S = 5.0       # poor posture becomes "unsafe" after 5 s continuous
CRITICAL_SEVERITY = 1.5
CRITICAL_SUSTAIN_S = 20.0


# ---------------------------------------------------------------- helpers
def ou(rng, n, sigma, tau_s):
    """Ornstein-Uhlenbeck (slow drift) process, stationary std = sigma."""
    a = np.exp(-DT / tau_s)
    w = rng.normal(0, sigma * np.sqrt(1 - a * a), n)
    x = lfilter([1.0], [1.0, -a], w)
    return x


def smooth_step(u):
    u = np.clip(u, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def make_subjects(rng, n=8):
    subs = []
    for i in range(1, n + 1):
        subs.append(dict(
            id=f"P{i:02d}",
            height_cm=round(float(rng.uniform(152, 186)), 1),
            mount={"upper": (rng.uniform(-5, 8), rng.uniform(-4, 4)),
                   "left": (rng.uniform(3, 14), rng.uniform(-6, 2)),
                   "right": (rng.uniform(3, 14), rng.uniform(-2, 6))},
            stand_theta=rng.uniform(-1, 3), sit_theta=rng.uniform(2, 7), sit_L=rng.uniform(4, 9),
            walk_f=rng.uniform(1.72, 1.98), run_f=rng.uniform(2.55, 2.95),
            gait_amp=rng.uniform(0.85, 1.2), run_lean=rng.uniform(7, 13),
            breath_f=rng.uniform(0.2, 0.3),
            flex_base=(rng.uniform(2550, 2950), rng.uniform(2500, 2900)),
            flex_k=(rng.uniform(24, 36), rng.uniform(24, 36)),
            flex_s=(rng.uniform(16, 28), rng.uniform(16, 28)),
            gyro_bias={k: rng.normal(0, 1.5, 3) for k in IMU_KEYS},
            acc_bias={k: rng.normal(0, 0.012, 3) for k in IMU_KEYS},
            noise=rng.uniform(0.8, 1.3),
        ))
    return subs


def session_schedule(rng, subjects):
    """16 sessions, 2 per participant, spread over 6-13 Oct 2026 (two sessions per day)."""
    out = []
    for i, s in enumerate(subjects):
        for k, day in ((1, 6 + i // 2), (2, 10 + i // 2)):
            hour = 9 + (i % 2) * 5 + int(rng.integers(0, 4))       # mornings / afternoons
            minute = int(rng.integers(0, 60))
            start = datetime(2026, 10, day, hour, minute, int(rng.integers(0, 60)), tzinfo=IST)
            out.append(dict(subject=s, session_no=k, session_id=f"{s['id']}_S{k}", start=start))
    return out


def plan_blocks(rng, variant):
    j = lambda d: d * rng.uniform(0.85, 1.15)
    if variant == 1:
        acts = [("standing", 10, True), ("sitting", j(70), False), ("standing", j(40), False),
                ("walking", j(45), False), ("running", j(25), False), ("walking", j(15), False),
                ("standing", j(10), False), ("lying_down", j(35), False), ("sitting", j(25), False)]
    else:
        acts = [("standing", 10, True), ("walking", j(40), False), ("running", j(28), False),
                ("walking", j(20), False), ("standing", j(40), False), ("sitting", j(75), False),
                ("lying_down", j(35), False), ("sitting", j(20), False), ("standing", j(10), False)]
    blocks = []
    for k, (a, d, cal) in enumerate(acts):
        if k > 0:
            prev = acts[k - 1][0]
            tdur = 3.0 if "lying_down" in (prev, a) else (2.5 if "sitting" in (prev, a) else
                                                         (2.0 if "running" in (prev, a) else 1.5))
            blocks.append(dict(kind="transition", frm=prev, to=a, dur=tdur * rng.uniform(0.85, 1.15)))
        blocks.append(dict(kind="activity", act=a, dur=d, cal=cal))
    return blocks


# ---------------------------------------------------------------- episodes
def episode_menu(rng, act):
    sh = lambda: rng.uniform(2.0, 4.5)
    med = lambda: rng.uniform(5.5, 9.0)
    lng = lambda: rng.uniform(12, 35)
    side = lambda: ("leaning_left", -1) if rng.random() < 0.5 else ("leaning_right", 1)
    if act == "sitting":
        L = side()
        return [("slouching", "poor", lng()), ("slouching", "poor", sh()), ("slouching", "poor", med()),
                ("slouching", "mild", med()), (L[0], "poor", med()), ("shoulder_asymmetry", "poor", rng.uniform(8, 20)),
                ("other", "poor", med()), ("slouching", "border", med()), ("shoulder_asymmetry", "mild", med())]
    if act == "standing":
        L = side()
        return [("excessive_forward_bending", "poor", rng.uniform(2.5, 4.5)),
                ("excessive_forward_bending", "poor", rng.uniform(6, 10)),
                ("excessive_backward_bending", "poor", med()), ("slouching", "poor", rng.uniform(6, 15)),
                (L[0], "poor", rng.uniform(8, 14)), ("shoulder_asymmetry", "poor", rng.uniform(10, 20)),
                (side()[0], "mild", med()), ("excessive_forward_bending", "border", rng.uniform(4, 6))]
    if act == "walking":
        return [("slouching", "poor", rng.uniform(8, 20)), ("shoulder_asymmetry", "poor", rng.uniform(10, 20)),
                ("slouching", "mild", med())]
    return []


def magnitude(rng, reason, kind):
    c = CRITERIA[reason]
    if kind == "border":
        return rng.uniform(c - BORDER + 0.1, c + BORDER - 0.1)
    if kind == "mild":
        return rng.uniform(0.35 * c, 0.7 * c)
    hi = {"excessive_forward_bending": 65, "slouching": 26}.get(reason, c * 2.0)
    return rng.uniform(c + BORDER + 0.5, hi)


def apply_episode(st, rng, a0, a1, reason, kind, act, s):
    """Writes the episode into the state arrays (st) and the label arrays for rows [a0, a1)."""
    n = a1 - a0
    R = int(round(rng.uniform(0.6, 1.0) * FS))
    P = n - 2 * R
    M = magnitude(rng, reason, kind)
    env = np.ones(n)
    env[:R] = smooth_step(np.arange(R) / R)
    env[-R:] = smooth_step(1 - np.arange(1, R + 1) / R)
    m = M * env * (1 + 0.04 * ou(rng, n, 1.0, 2.0))
    sl = slice(a0, a1)
    if reason == "slouching":
        st["theta"][sl] += m
        st["shround"][sl] += 0.35 * m
        st["L"][sl] += (0.95 if act == "sitting" else 0.7) * m
    elif reason == "excessive_forward_bending":
        st["theta"][sl] += m
        st["L"][sl] += 0.55 * m
        st["shround"][sl] += 0.1 * m
    elif reason == "excessive_backward_bending":
        st["theta"][sl] -= m
        st["L"][sl] -= 0.6 * m
    elif reason in ("leaning_left", "leaning_right"):
        sg = -1 if reason == "leaning_left" else 1
        st["phi"][sl] += sg * m
        st["psi"][sl] += sg * 0.7 * m
    elif reason == "shoulder_asymmetry":
        st["asym"][sl] += (1 if rng.random() < 0.5 else -1) * m
    elif reason == "other":
        st["shround"][sl] += m
        st["theta"][sl] += 0.25 * m

    if kind == "mild":
        return
    lab = st["labels"]
    ramp_rows = list(range(a0, a0 + R)) + list(range(a0 + R + P, a1))
    for i in ramp_rows:
        lab["pq"][i], lab["safety"][i] = "uncertain", "uncertain"
    plateau = range(a0 + R, a0 + R + P)
    if kind == "border":
        for i in plateau:
            lab["pq"][i], lab["safety"][i] = "uncertain", "uncertain"
        return
    ratio = M / CRITERIA[reason]
    for k, i in enumerate(plateau):
        el = k / FS
        lab["pq"][i] = "poor"
        if el >= SUSTAIN_S:
            lab["safety"][i], lab["reason"][i] = "unsafe", reason
            lab["warn"][i] = "critical" if (ratio >= CRITICAL_SEVERITY or el >= CRITICAL_SUSTAIN_S) else "advisory"


# ---------------------------------------------------------------- one session
def generate_session(rng, sess):
    s = sess["subject"]
    blocks = plan_blocks(rng, sess["session_no"])
    for b in blocks:
        b["n"] = int(round(b["dur"] * FS))
    N = sum(b["n"] for b in blocks)
    st = {k: np.zeros(N) for k in ("theta", "phi", "shround", "asym", "L", "psi", "walk_w", "run_w",
                                   "yaw", "lin_v")}
    act = np.empty(N, dtype=object)
    trial = np.empty(N, dtype=object)
    is_cal = np.zeros(N, np.int8)
    labels = {"pq": np.empty(N, dtype=object), "safety": np.empty(N, dtype=object),
              "reason": np.full(N, "none", dtype=object), "warn": np.full(N, "none", dtype=object)}
    st["labels"] = labels

    # ---- pass 1: activity blocks
    pos, k_act, k_tr = 0, 0, 0
    for b in blocks:
        b["a0"], b["a1"] = pos, pos + b["n"]
        pos += b["n"]
    for b in blocks:
        a0, a1, n = b["a0"], b["a1"], b["n"]
        sl = slice(a0, a1)
        if b["kind"] == "transition":
            k_tr += 1
            act[sl] = "transition"
            trial[sl] = f"{sess['session_id']}_TR{k_tr:02d}"
            labels["pq"][sl], labels["safety"][sl] = "not_applicable", "safe"
            continue
        a = b["act"]
        trial[sl] = f"{sess['session_id']}_T{k_act:02d}" if not b["cal"] else f"{sess['session_id']}_CAL"
        k_act += 1
        act[sl] = a
        is_cal[sl] = 1 if b["cal"] else 0
        applicable = a in ("sitting", "standing", "walking")
        labels["pq"][sl] = "good" if applicable else "not_applicable"
        labels["safety"][sl] = "safe"
        if a == "standing":
            sig = 0.25 if b["cal"] else 0.5
            st["theta"][sl] = s["stand_theta"] + ou(rng, n, sig, 4)
            st["phi"][sl] = ou(rng, n, 0.4 if not b["cal"] else 0.2, 4)
            st["L"][sl] = ou(rng, n, 0.8, 6)
        elif a == "sitting":
            st["theta"][sl] = s["sit_theta"] + ou(rng, n, 1.0, 15)
            st["phi"][sl] = ou(rng, n, 0.8, 10)
            st["L"][sl] = s["sit_L"] + ou(rng, n, 1.2, 15)
            st["shround"][sl] = 1.0 + ou(rng, n, 1.0, 12)
        elif a == "walking":
            st["theta"][sl] = s["stand_theta"] + 2 + ou(rng, n, 0.8, 6)
            st["phi"][sl] = ou(rng, n, 0.6, 6)
            st["L"][sl] = 2 + ou(rng, n, 0.8, 6)
            st["walk_w"][sl] = 1
        elif a == "running":
            st["theta"][sl] = s["run_lean"] + ou(rng, n, 1.2, 4)
            st["phi"][sl] = ou(rng, n, 0.8, 4)
            st["L"][sl] = 4 + ou(rng, n, 1.0, 4)
            st["run_w"][sl] = 1
        elif a == "lying_down":
            # supine -> roll onto one side -> supine (all labelled lying_down)
            u = np.arange(n) / n
            side_sign = -1 if rng.random() < 0.5 else 1
            w = smooth_step((u - 0.5) / 0.06) - smooth_step((u - 0.82) / 0.06)
            st["theta"][sl] = (-90 + ou(rng, n, 1.0, 10)) * (1 - w) + (-8 + ou(rng, n, 1.5, 10)) * w
            st["phi"][sl] = side_sign * 86 * w + ou(rng, n, 0.8, 10)
            st["L"][sl] = 2 + ou(rng, n, 0.8, 10)
        if b["cal"]:
            continue
        st["asym"][sl] += ou(rng, n, 0.6, 8)
        st["shround"][sl] += ou(rng, n, 0.6, 8)
        # fidgets (normal movement, always "good")
        if a in ("sitting", "standing"):
            for _ in range(rng.poisson(n / FS / 20)):
                c = int(rng.integers(a0, a1))
                w = int(rng.uniform(0.6, 1.2) * FS)
                e = np.sin(np.pi * np.arange(w) / w)
                e = e[: max(0, min(w, a1 - c))]
                which = "theta" if rng.random() < 0.6 else "phi"
                st[which][c:c + len(e)] += rng.uniform(-5, 5) * e
                st["yaw"][c:c + len(e)] += rng.uniform(-25, 25) * e
        # scripted posture episodes
        t = int(rng.uniform(3, 6) * FS) + a0
        menu = episode_menu(rng, a)
        rng.shuffle(menu)
        for reason, kind, dur in menu:
            nrows = int(dur * FS) + 2 * FS
            if t + nrows > a1 - 3 * FS:
                continue
            apply_episode(st, rng, t, t + nrows, reason, kind, a, s)
            t += nrows + int(rng.uniform(3, 8) * FS)

    # ---- pass 2: transitions (interpolate between neighbours, add the movement itself)
    for bi, b in enumerate(blocks):
        if b["kind"] != "transition":
            continue
        a0, a1, n = b["a0"], b["a1"], b["n"]
        u = smooth_step(np.arange(n) / max(n - 1, 1))
        for key in ("theta", "phi", "shround", "asym", "L", "psi", "walk_w", "run_w"):
            v0, v1 = st[key][a0 - 1], st[key][a1]
            st[key][a0:a1] = v0 + (v1 - v0) * u
        pair = {b["frm"], b["to"]}
        bump = np.sin(np.pi * np.arange(n) / n)
        if pair == {"sitting", "standing"}:
            st["theta"][a0:a1] += rng.uniform(22, 32) * bump        # trunk flexes forward to rise / sit
            st["L"][a0:a1] += rng.uniform(10, 18) * bump
            st["lin_v"][a0:a1] += (0.25 if b["to"] == "standing" else -0.2) * np.sin(2 * np.pi * np.arange(n) / n)
        if "lying_down" in pair:
            st["theta"][a0:a1] += rng.uniform(5, 15) * bump
            st["yaw"][a0:a1] += rng.uniform(-15, 15) * bump

    # ---- dynamics: gait, breathing, turns
    t_s = np.arange(N) * DT
    f = s["walk_f"] * st["walk_w"] + s["run_f"] * st["run_w"]
    f = np.where(f > 0, f, s["walk_f"]) * (1 + 0.015 * ou(rng, N, 1.0, 5))
    ph = np.cumsum(2 * np.pi * f * DT)
    aw, ar = s["gait_amp"] * st["walk_w"], s["gait_amp"] * st["run_w"]
    s3 = np.maximum(0, np.sin(ph)) ** 3
    run_vert = 2.2 * s3 - 0.467
    breath = 0.4 * np.sin(2 * np.pi * s["breath_f"] * t_s)
    theta = st["theta"] + breath + aw * 1.5 * np.sin(ph + 0.3) + ar * 3.0 * np.sin(ph + 0.5)
    phi = st["phi"] + aw * 2.0 * np.sin(ph / 2) + ar * 3.5 * np.sin(ph / 2)
    yaw = st["yaw"] + aw * 12 * np.cos(ph / 2) + ar * 28 * np.cos(ph / 2)
    # walking turns (180 deg in 1.5-2.5 s, every 12-20 s)
    walk_rows = np.where(st["walk_w"] > 0.9)[0]
    if len(walk_rows):
        i = walk_rows[0] + int(rng.uniform(8, 14) * FS)
        while i < N and st["walk_w"][i] > 0.9:
            w = int(rng.uniform(1.5, 2.5) * FS)
            e = np.sin(np.pi * np.arange(w) / w)[: max(0, min(w, N - i))]
            sgn = 1 if rng.random() < 0.5 else -1
            yaw[i:i + len(e)] += sgn * (180 / (w * DT)) * (np.pi / 2) * e
            phi[i:i + len(e)] += sgn * 3 * e
            i += w + int(rng.uniform(12, 20) * FS)
    lin = {"x": aw * 0.06 * np.sin(ph / 2) + ar * 0.18 * np.sin(ph / 2),
           "y": aw * (0.22 * np.sin(ph) + 0.06 * np.sin(2 * ph)) + ar * run_vert + st["lin_v"],
           "z": aw * 0.12 * np.sin(ph + 0.8) + ar * 0.35 * np.sin(ph + 0.9)}

    # per-IMU orientation
    swing = aw * 2.5 * np.sin(ph / 2 + 0.4) + ar * 7 * np.sin(ph / 2 + 0.4)
    ori = {}
    mu = s["mount"]
    ori["upper"] = (mu["upper"][0] + theta, mu["upper"][1] + phi, 1.0)
    ori["left"] = (mu["left"][0] + theta + st["shround"] + ou(rng, N, 0.5, 6) + 0.6 * swing,
                   mu["left"][1] + phi - st["asym"] / 2 + swing, 1.05)
    ori["right"] = (mu["right"][0] + theta + st["shround"] + ou(rng, N, 0.5, 6) - 0.6 * swing,
                    mu["right"][1] + phi + st["asym"] / 2 - swing, 1.05)

    data = {}
    sat_flags = {}
    for k in IMU_KEYS:
        th, fi, lscale = ori[k]
        thr, fir = np.radians(th), np.radians(fi)
        g = np.stack([-np.sin(fir), np.cos(thr) * np.cos(fir), np.sin(thr) * np.cos(fir)], axis=1)
        la = np.stack([lin["x"], lin["y"], lin["z"]], axis=1) * lscale
        nz = 0.004 * s["noise"] + 0.02 * ar[:, None]           # sensor noise + garment vibration
        acc = g + la + s["acc_bias"][k] + rng.normal(0, 1, (N, 3)) * nz
        sat_flags[k] = (np.abs(acc) >= 2.0).any(axis=1)
        acc = np.clip(acc, -2.0, 2.0 - 1 / 16384)
        acc = np.round(np.round(acc * 16384) / 16384, 5)
        gx = np.gradient(th, DT)
        gz = np.gradient(fi, DT)
        gy = yaw * (1.1 if k != "upper" else 1.0)
        gyr = np.stack([gx, gy, gz], axis=1) + s["gyro_bias"][k] \
            + rng.normal(0, 1, (N, 3)) * (0.15 * s["noise"] + 2.0 * ar[:, None])
        gyr = np.clip(gyr, -250, 250)
        gyr = np.round(np.round(gyr * 131) / 131, 3)
        for j, c in enumerate(["ax", "ay", "az"]):
            data[f"imu_{k}_{c}"] = acc[:, j]
        for j, c in enumerate(["gx", "gy", "gz"]):
            data[f"imu_{k}_{c}"] = gyr[:, j]

    # flex sensors (lumbar X)
    Lt = st["L"] + aw * 3 * np.sin(ph) + ar * 6 * np.sin(ph)
    psi = st["psi"] + aw * 4 * np.sin(ph / 2) + ar * 6 * np.sin(ph / 2)
    (b1, b2), (k1, k2), (s1, s2) = s["flex_base"], s["flex_k"], s["flex_s"]
    lr = b1 - k1 * Lt - s1 * psi + ou(rng, N, 15, 60) + rng.normal(0, 10, N)
    rl = b2 - k2 * Lt + s2 * psi + ou(rng, N, 15, 60) + rng.normal(0, 10, N)
    data["flex_lr_raw"] = np.round(np.clip(lr, 40, 4055)).astype(float)
    data["flex_rl_raw"] = np.round(np.clip(rl, 40, 4055)).astype(float)

    # ---- missing / invalid readings (documented, never silently filled)
    flags = [[] for _ in range(N)]
    for k in IMU_KEYS:
        for i in np.where(sat_flags[k])[0]:
            flags[i].append(f"accel_saturated_{k}")
        starts = rng.choice(N, rng.poisson(N * 0.0006), replace=False)
        drops = [(st0, int(rng.integers(1, 6))) for st0 in starts]
        if rng.random() < 0.25:                         # loose connector: one longer gap
            drops.append((int(rng.integers(FS * 15, N - FS * 2)), int(rng.uniform(0.5, 1.5) * FS)))
        for st0, ln in drops:
            for i in range(st0, min(N, st0 + ln)):
                for c in ["ax", "ay", "az", "gx", "gy", "gz"]:
                    data[f"imu_{k}_{c}"][i] = np.nan
                flags[i].append(f"imu_{k}_missing")
    for c, nm in (("flex_lr_raw", "flex_lr"), ("flex_rl_raw", "flex_rl")):
        for i in rng.choice(N, rng.poisson(N * 0.0002), replace=False):
            data[c][i] = 4095.0 if rng.random() < 0.6 else 0.0
            flags[i].append(f"{nm}_invalid")
    # never corrupt the calibration trial's last 2 s (firmware baseline) - real users would recalibrate
    cal_rows = np.where(is_cal == 1)[0]
    for i in cal_rows[-60:]:
        if flags[i]:
            for key in list(data):
                if np.isnan(data[key][i]) or (key in FLEX_COLS and data[key][i] in (0.0, 4095.0)):
                    data[key][i] = data[key][i - 1] if i > 0 else data[key][i + 1]
            flags[i] = []

    start_ms = int(sess["start"].timestamp() * 1000)
    ts = start_ms + np.arange(N) * 40
    iso = [(sess["start"] + timedelta(milliseconds=40 * i)).isoformat(timespec="milliseconds") for i in range(N)]
    df = pd.DataFrame({
        "data_source": "synthetic", "timestamp_ms": ts, "timestamp": iso,
        "subject_id": s["id"], "session_id": sess["session_id"], "trial_id": trial,
        "is_calibration": is_cal, "activity_label": act,
        "posture_quality": labels["pq"], "safety_label": labels["safety"],
        "unsafe_reason": labels["reason"], "warning_level": labels["warn"],
    })
    for c in RAW_IMU_COLS + FLEX_COLS:
        df[c] = data[c]
    df["quality_flags"] = [";".join(f) if f else "ok" for f in flags]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="spine_grid_posture_synthetic.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    subjects = make_subjects(rng)
    sessions = session_schedule(rng, subjects)
    frames = []
    for sess in sessions:
        df = generate_session(rng, sess)
        der = compute_derived(df)
        fw = simulate_firmware(df)
        frames.append(pd.concat([df, der, fw], axis=1))
        print(f"{sess['session_id']}  {sess['start']:%Y-%m-%d %H:%M}  rows={len(df)}")
    full = pd.concat(frames, ignore_index=True)

    order = (["data_source", "timestamp_ms", "timestamp", "subject_id", "session_id", "trial_id",
              "is_calibration", "activity_label", "posture_quality", "safety_label", "unsafe_reason",
              "warning_level"] + RAW_IMU_COLS + FLEX_COLS +
             [c for c in der.columns] + FIRMWARE_COLS + ["quality_flags"])
    full = full[order]
    round_map = {c: 3 for c in der.columns}
    round_map.update({"flex_lr_dev": 1, "flex_rl_dev": 1, "flex_dev_total": 1, "accel_magnitude": 4,
                      "gyro_magnitude": 3})
    full = full.round(round_map)
    full.to_csv(args.out, index=False, na_rep="")
    pd.DataFrame([{k: v for k, v in s.items() if k in ("id", "height_cm")} for s in subjects]) \
        .rename(columns={"id": "subject_id"}).to_csv("participants_synthetic.csv", index=False)
    print(f"\nWrote {args.out}: {len(full)} rows x {full.shape[1]} columns")
    write_summary(full)


def write_summary(df):
    L = []
    L.append("# Dataset summary (SYNTHETIC)\n")
    L.append(f"Rows: {len(df):,} at 25 Hz ({len(df)/25/60:.1f} min) | participants: {df.subject_id.nunique()} | "
             f"sessions: {df.session_id.nunique()} | dates: {df.timestamp.str[:10].min()} to {df.timestamp.str[:10].max()}\n")
    L.append("Every row has data_source = synthetic. Generated by generate_dataset.py (seed 20261006).\n")

    def table(series, title):
        c = series.value_counts()
        L.append(f"\n## {title}\n\n| Value | Rows | Minutes | Share |\n| --- | ---: | ---: | ---: |")
        for k, v in c.items():
            L.append(f"| {k} | {v:,} | {v/25/60:.1f} | {v/len(series):.1%} |")

    table(df.activity_label, "Activity")
    table(df.posture_quality, "Posture quality")
    table(df.safety_label, "Safety label")
    table(df.unsafe_reason[df.safety_label == "unsafe"], "Unsafe reason (unsafe rows only)")
    table(df.warning_level, "Ground-truth warning level")

    L.append("\n## Activity x posture quality (rows)\n")
    ct = pd.crosstab(df.activity_label, df.posture_quality)
    L.append("| activity | " + " | ".join(ct.columns) + " |")
    L.append("| --- | " + " | ".join("---:" for _ in ct.columns) + " |")
    for a, r in ct.iterrows():
        L.append(f"| {a} | " + " | ".join(f"{v:,}" for v in r) + " |")

    L.append("\n## Participant coverage (minutes per activity)\n")
    ct = pd.crosstab(df.subject_id, df.activity_label) / 25 / 60
    L.append("| participant | " + " | ".join(ct.columns) + " |")
    L.append("| --- | " + " | ".join("---:" for _ in ct.columns) + " |")
    for a, r in ct.iterrows():
        L.append(f"| {a} | " + " | ".join(f"{v:.1f}" for v in r) + " |")

    L.append("\n## Sessions\n\n| session | participant | start (IST) | rows |\n| --- | --- | --- | ---: |")
    for sid, g in df.groupby("session_id"):
        L.append(f"| {sid} | {g.subject_id.iloc[0]} | {g.timestamp.iloc[0][:19].replace('T', ' ')} | {len(g):,} |")

    # unsafe events
    ev = 0
    for _, g in df.groupby("session_id"):
        u = (g.safety_label == "unsafe").to_numpy().astype(int)
        ev += int(((np.diff(np.r_[0, u]) == 1)).sum())
    L.append(f"\nGround-truth unsafe events (continuous unsafe runs): {ev}\n")

    L.append("\n## Missing and invalid readings\n")
    L.append("| Column group | Rows affected | Share |\n| --- | ---: | ---: |")
    for k in IMU_KEYS:
        n = int(df[f"imu_{k}_ax"].isna().sum())
        L.append(f"| imu_{k}_* missing (all 6 channels blank) | {n:,} | {n/len(df):.3%} |")
    for c in FLEX_COLS:
        n = int(df[c].isin([0, 4095]).sum())
        L.append(f"| {c} invalid (0 or 4095) | {n:,} | {n/len(df):.3%} |")
    for k in IMU_KEYS:
        n = int(df.quality_flags.str.contains(f"accel_saturated_{k}").sum())
        L.append(f"| imu_{k} accelerometer at the +-2 g limit | {n:,} | {n/len(df):.3%} |")

    L.append("\n## Firmware rule output vs. activity (alert_active share of rows)\n")
    L.append("Shows how the unmodified firmware thresholds behave in each activity context.\n")
    L.append("| activity | alert_active | critical_condition | warning_condition |\n| --- | ---: | ---: | ---: |")
    for a, g in df.groupby("activity_label"):
        L.append(f"| {a} | {g.alert_active.mean():.1%} | {g.critical_condition.mean():.1%} | {g.warning_condition.mean():.1%} |")
    open("DATASET_SUMMARY.md", "w").write("\n".join(L) + "\n")
    print("Wrote DATASET_SUMMARY.md")


if __name__ == "__main__":
    main()
