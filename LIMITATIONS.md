# Limitations and the real-world data still needed

## What this dataset is

A **synthetic** dataset built to exercise the complete pipeline for the Spine-Grid Tee: raw sensors, derived features, 2 s windows, Models A/B/C and the firmware comparison. Its structure, units, sampling, labels and split match the specification, so real recordings can drop in without changing any code.

## What it is not

- **Not a measured dataset.** A genuinely measured dataset cannot be created from a specification alone. No person wore the shirt, and the dates 6–13 Oct 2026 are simulated session dates. Do not report these numbers, or present this data, as real recordings in papers, the IDF-B disclosure or a demo.
- **Model scores are optimistic.** The generator and the labels share the same scripted episodes, so classes are cleaner than in real life. On real data, expect lower accuracy and more confusion between slouching and forward bending, sitting and standing, and good and poor posture near the criteria.
- **The physics is simplified.** The generator does not model:
  - fabric sliding or rotating the sensors during movement
  - soft-tissue wobble
  - shoulder-blade motion during arm use
  - temperature drift of the flex sensors
  - the flex sensors' hysteresis and non-linearity
  - magnetic or electrical interference
  - realistic timing jitter (the data sits on an exact 40 ms grid)
- **Behaviour is scripted.** Real people do not slouch in clean plateaus, and their mixed postures (slouching while leaning) are rare here.
- **Coverage is narrow.** There are 8 synthetic adults, 2 sessions each and about 80 minutes in total. The dataset has no:
  - children, older adults, or different body types beyond height and mounting variation
  - stairs or cycling
  - desk work with arm movement
  - loose shirts or different fabrics
  - long free-living recordings
- **Posture criteria are provisional.** The angle criteria (12°, 30°, 15°, 10°) are posture-monitoring categories chosen for this prototype. They are not clinical thresholds and have not been reviewed by a physiotherapist.

## What the experiments show (pipeline-level only)

- **Activity context matters.** The unmodified firmware rules alert during 92% of lying-down time, and are "critical" during 62% of running and 34% of ordinary sitting. Sitting bends the lumbar spine relative to the standing calibration, so a single standing calibration plus fixed thresholds over-alerts. Model A, used as context, removes these.
- **Telling advisory from critical is the weakest part** (macro-F1 about 0.64). Deviation severity is measured against the standing calibration, while ground truth is relative to each activity's own neutral posture. With real data, add a per-activity neutral baseline (for example a rolling median of good-posture sitting) or learn the level directly.
- **Borderline episodes trigger alerts.** Most "extra" alerts happen during borderline postures or poor postures shorter than 5 s. Decide with the posture expert whether those alerts are wanted.

## Real data needed before deployment

1. **Pilot:** 3–5 people × 2 sessions with synchronized video, using COLLECTION_PROTOCOL.md. Check sensor placement, label quality and whether the pipeline runs. Look at learning curves.
2. **Main collection:** at least 15–20 participants of different heights, body types, sexes and ages, at least 3 sessions on different days, and both seated desk work and free movement. Have an expert define and check posture criteria. Record at least 2 annotators on a subset to measure label agreement.
3. **Free-living:** at least 1–2 hours per participant of unscripted wear, with a few self-reported or video-checked labels, to measure the false alerts per hour that users would actually experience.
4. **Robustness:** loose versus snug shirts, re-donning the shirt between sessions, sweat and washing cycles (flex-sensor drift), and battery-powered timing jitter.
5. **Evaluation:** leave-one-participant-out results, per-class precision, recall, F1 and confusion matrices, missed unsafe events, false alerts per hour, and warning latency. Repeat the firmware-versus-ML comparison on the real data.
