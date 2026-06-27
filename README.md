# BehaveGuard — Continuous Behavioral Biometrics Authentication

BehaveGuard is a state-of-the-art continuous authentication system that utilizes behavioral biometrics (keystroke dynamics and mouse dynamics) to continuously verify a user's identity. 

The system fuses statistical models (One-Class SVM) with deep learning sequence models (LSTM and TCN Autoencoders) to perform highly robust, scale-invariant identification and anomaly detection.

---

## Project Structure

The repository is structured as a full-stack application:

*   **`behaveguard/` (Backend)**: FastAPI server containing feature extraction, data storage, and machine learning models.
    *   `main.py`: API endpoints for enrollment, verification, and identification.
    *   `features.py`: Feature extraction logic for keystrokes (23-dimensional aggregates & timing sequences) and mouse movements (kinematic windows).
    *   `pipeline.py`: Orchestrates training, scoring, and multi-model fusion.
    *   `storage.py`: Handles SQLite database initialization and data loading.
    *   `models/`: Class wrappers and model weights (`.pkl` pickles) for `svm`, `lstm`, and `tcn` profiles.
    *   `data/`: CSV-based SQLite tables (`key_events.csv`, `mouse_passive.csv`, `sessions.csv`) and backup logs.
*   **`behaveguard-client-master/` (Frontend)**: Next.js application providing sandbox playgrounds for enrollment, verification, and live identification testing.

---

## Getting Started

### Prerequisites
*   Python 3.10+
*   Node.js 18+ & npm

### 1. Backend Setup
Navigate to the root directory and set up the Python virtual environment:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r behaveguard/requirements.txt
```
*(If a requirements file is not present, the core dependencies are `fastapi`, `uvicorn`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, and `matplotlib`.)*

### 2. Frontend Setup
Navigate to the frontend directory and install npm packages:
```bash
cd behaveguard-client-master
npm install
cd ..
```

---

## Running the Applications

To run the local sandbox environment, start both servers in separate terminal windows:

### Start Backend FastAPI Server (Port 8000)
From the root directory:
```bash
source .venv/bin/activate
python -m uvicorn behaveguard.main:app --port 8000
```

### Start Frontend Next.js Server (Port 3000)
From the root directory:
```bash
cd behaveguard-client-master
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your web browser.

---

## Biometric System Architecture

### 1. Keystroke Dynamics
Keystroke authentication uses a multi-model fusion pipeline:
*   **Feature Space (23-Dimensions)**: Dwell times, flight times, and inter-key intervals (IKIs) are extracted. Timings are normalized by the chunk's **dwell mean** to achieve speed-scale invariance. Negative flight times (from key overlaps during fast typing) are fully supported.
*   **One-Class SVM (Statistical Baseline)**: Modeled using custom hyperparameters (`nu=0.02`, `gamma=0.002`, and standard deviation scale floor of `0.20`) to tolerate normal speed variations while strictly rejecting impostors.
*   **LSTM & TCN Autoencoders (Sequence Models)**: Capture sequence transition dependencies and key typing rhythms over 50-key windows.
*   **Score Fusion**: Fuses model anomaly scores: **30% SVM + 35% LSTM + 35% TCN**.

### 2. Mouse Dynamics
Mouse authentication fuses passive track kinematics with active game metrics:
*   **Passive Mouse Dynamics (SVM)**: Extracts 7-dimensional kinematic features (speed, acceleration, curvature, and drag duration) from passive cursor movements over 100-point windows.
*   **Active Mouse Dynamics (Z-Score)**: Measures reaction times, accuracy, and drag task durations against user baseline averages.
*   **Fused Mouse Score**: Computes a 50/50 weighted combination of passive and active task anomaly scores.

### 3. Fused Biometric Match Rate
The final authentication decision is a fused combination of the keyboard score (50%) and the mouse score (50%). If no mouse data is available, it falls back to the keyboard score alone.

---

## Database Cleaning & Retraining

If you need to filter out dwell time outliers (e.g. enter-key holds) and rebuild all user profiles from scratch:

1.  Make sure the backend is stopped or idle.
2.  Run the cleaning and retraining script:
    ```bash
    source .venv/bin/activate
    python -m behaveguard.data.clean_mouse_passive_db
    ```
    *(This runs IQR-based timing outlier removal and fits new SVM, LSTM, and TCN weights for all enrolled profiles.)*
