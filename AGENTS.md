# BehaveGuard AI Agent Developer Guide

Welcome, AI Coding Assistant! This document contains the critical technical specifications, architecture decisions, and development guidelines for the BehaveGuard codebase. Read this before making modifications to feature extraction or model training pipelines.

---

## 1. Core Architecture

BehaveGuard uses a hybrid approach for continuous behavioral biometrics authentication:

### Keystroke Dynamics (Keyboard)
*   **Pipeline**: Keyboard scoring fuses a statistical aggregates model (One-Class SVM) with deep sequence models (LSTM and TCN Autoencoders).
*   **Fusion Weights**: Keyboard score is calculated as:
    $$\text{kb\_score} = 0.30 \times \text{SVM} + 0.35 \times \text{LSTM} + 0.35 \times \text{TCN}$$
*   **Decision Boundaries**: Verdicts are classified per-window based on individual user calibration thresholds (`t_anomaly`). Overall session verdicts (`legitimate`, `uncertain`, `impostor`) are aggregated based on the anomaly rates of the windows.

### Mouse Dynamics (Mouse)
*   **Passive Mouse**: Tracks kinematic windows (speed, acceleration, curvature, drag duration) using a One-Class SVM.
*   **Active Mouse**: Tracks click accuracy, drag durations, and reaction times in target-hitting games, computing deviations (Z-scores) relative to the user's enrollment baseline.
*   **Fusion**: Fused mouse score is a 50/50 combination of the passive and active task anomaly scores.

---

## 2. Feature Engineering & Normalization

### Keyboard Aggregates (23-Dimensional Feature Vector)
The features extracted in `extract_keystroke_aggregates` in [features.py](file:///Users/akshitmehta/Development/behave-guard/behaveguard/features.py) are:
1.  `dwell_mean`, `dwell_std` (Indices 00-01)
2.  `flight_mean`, `flight_std` (Indices 02-03)
3.  `digraph_mean`, `digraph_std` (Indices 04-05)
4.  `alphanum_dwell_mean`, `alphanum_dwell_std` (Indices 06-07)
5.  `symbol_dwell_mean`, `symbol_dwell_std` (Indices 08-09)
6.  `special_dwell_mean`, `special_dwell_std` (Indices 10-11)
7.  `iki_mean`, `iki_std` (Indices 12-13)
8.  `fd_ratio_mean`, `fd_ratio_std` (Indices 14-15)
9.  `t_sin`, `t_cos` (Indices 16-17) - Cyclical time encoding (currently disabled, returns `0.0`).
10. `alphanum_ratio`, `symbol_ratio`, `special_ratio` (Indices 18-20)
11. `wpm` (Index 21) - Normalized words-per-minute (`wpm / 100.0`).
12. `digraph_coverage` (Index 22) - Fraction of common English digraphs typed in the window.

### Timing Normalization Rules (CRITICAL)
To maintain speed-scale invariance (so typing fast doesn't trigger false impostor alerts), timings are normalized as follows:
*   `dwell_mean` floor: Enforced at `50.0ms` to prevent division by near-zero.
*   `digraph_mean` floor: Enforced at `100.0ms` to prevent division by near-zero.
*   **Flight Normalization**: Flight times are normalized by dividing by the **dwell mean** (`mean_d`) rather than the flight mean. Because flight times frequently approach zero or become negative (key overlaps) during fast typing, dividing by the flight mean is mathematically unstable. Normalizing by the dwell mean avoids this issue and generalizes perfectly to high typing speeds.

---

## 3. Model Hyperparameters & Calibration

### One-Class SVM (Keystroke Baseline)
*   **Hyperparameters**: `nu=0.02`, `kernel='rbf'`, `gamma=0.002` (in [svm.py](file:///Users/akshitmehta/Development/behave-guard/behaveguard/models/svm.py)).
*   **StandardScaler Scale Floor**: Set to `0.20` for keyboard features to prevent overfitting to low-variance enrollment sessions. This allows normal variations in typing speed and ratios to occur without blowing up Z-scores.
*   **Calibration Floor**: Calibrated threshold `t_anomaly` is floored at `0.15` to align discrete verdicts with global calibration bounds.

### Deep Learning Sequence Autoencoders
*   **LSTM**: 2-layer LSTM autoencoder (sequence length = 50, batch size = 16, epochs = 120).
*   **TCN**: Temporal Convolutional Network autoencoder (epochs = 120).
*   **Data Padding**: All keystroke sequence windows shorter than 50 events are padded with constant zeros up to 50 for PyTorch compatibility.

### Global Calibration Thresholds
To compare anomaly scores across candidate models during identification runs, raw decision scores are mapped to a `[0, 1]` range using fixed global thresholds:
*   Keyboard SVM: `0.30`
*   Mouse SVM: `0.20`
*   LSTM Autoencoder: `1.20`
*   TCN Autoencoder: `15.0`

---

## 4. Development & Retraining Guidelines

1.  **Retraining Models**: Any change to feature shapes, normalization steps, or SVM hyperparameters requires retraining all profile weights. Run:
    ```bash
    .venv/bin/python -m behaveguard.data.clean_mouse_passive_db
    ```
2.  **Code Synchronicity**: Keep the feature extraction in [features.py](file:///Users/akshitmehta/Development/behave-guard/behaveguard/features.py) synchronized with any offline diagnostic/simulation scripts (located under `scratch/` or similar directories).
3.  **Mouse SVM Dimensions**: The passive mouse SVM runs on a 7-dimensional feature space (where the 7th dimension is `avg_drag_duration`). Do not apply the keyboard standard deviation floor (`0.20`) to the mouse model; it must remain at `0.05` to preserve its high discriminative power.
