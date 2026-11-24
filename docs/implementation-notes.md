# Implementation and verification notes

## Scope of the portfolio snapshot

This repository contains the supplied 2025 academic submission, organized for inspection. The source was kept substantially intact. Asset and model paths now resolve relative to the script, an absent sign image falls back to text, misleading comments and a map-loader description were corrected, and a minimal dependency list and offline smoke test were added. The original technical report's institutional email was removed and its PDF title metadata corrected. No historical experimental results have been changed.

## Differences between the report and supplied code

- The report identifies a 7,500-iteration checkpoint as best, but the supplied file is named `yolov3-tiny-obj_5000.weights`. The archive does not establish that they are identical. The reported mAP, F1 and latency should not be attributed to the bundled checkpoint without new evaluation.
- The client explicitly selects OpenCV's CPU backend/target. GPU timings in the report refer to the historical experiment; automatic CUDA selection is not present in this snapshot.
- The weak-sign monitor is implemented in the driving loop, but its `detection_ran` flag is calculated once at startup rather than refreshed for each inference. This can cause processing of stale observations or failure to update. A reliable implementation should synchronize new inference observations with the state machine and test episode boundaries. The portfolio cleanup intentionally did not change this safety-related logic without simulator validation.
- The original camera channel-conversion sequence deserves verification against the training preprocessing; the cleanup did not alter it.
- The report describes an episode in terms of a sign class; the submitted monitor tracks the highest-confidence observation globally. The original report's passing scenario verdicts do not prove that this snapshot implements the intended per-class logic.
- Autopilot detection depends on API availability and has not been validated; the acoustic warning logic is in the manual-driving branch. Reverse gear handling is level-triggered in the original keyboard loop, rather than debounced.
- Post-pass activation and thermal throttling are documented as failed test scenarios. Physical-world, safety, security and thermal compliance were not established.
- The original training data, annotation files, training pipeline, other checkpoints and simulator evaluation scripts were not in the supplied archive.

## Verification during repository preparation

The two Python sources were parsed successfully and the included model was loaded with OpenCV DNN 4.13.0. A synthetic zero input of shape `(1, 3, 608, 608)` produced three finite output arrays: `(1083, 8)`, `(4332, 8)` and `(17328, 8)`. The eight columns correspond to the standard YOLO proposal layout with three class-score columns. The included smoke test reproduces these basic checks; results may vary with library versions.

The original Python 3.7 / OpenCV 4.8.1 environment and CARLA driving integration were not available for a new end-to-end test. Before claiming that the reorganized demo reproduces the historical results, run it with the matching simulator and retest the scenarios in the report.

Before making the repository public, verify permission to redistribute the course submission, model weights, icons, audio and any adapted code. The archive did not provide explicit licensing or asset provenance. No new license has been asserted.
