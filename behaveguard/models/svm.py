import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

class BehaveGuardSVM:
    """
    One-class SVM wrapper for BehaveGuard:
    - Trains on genuine aggregate features (shape: [N, 43])
    - Calibrates threshold at 95th percentile of enrollment decision values
    - Scores new windows -> {anomaly_score, verdict}
    """
    def __init__(self, nu: float = 0.02, kernel: str = 'rbf', gamma: float | str = 0.002):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.svm = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        
        self.t_anomaly = 0.0
        self.enrollment_mean = None
        self.enrollment_std = None
        self.is_trained = False
 
    def fit(self, windows: List[np.ndarray]) -> Dict[str, Any]:
        """
        Train the model on genuine user windows.
        windows: list of 43-dim aggregate feature vectors
        """
        X = np.stack(windows)  # Shape: (N, 43) or (N, 7)
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply standard deviation floor (0.20 for keyboard 23-dim, 0.05 for mouse)
        scale_floor = 0.20 if X.shape[1] == 23 else 0.05
        self.scaler.scale_ = np.maximum(self.scaler.scale_, scale_floor)
        
        if X.shape[1] == 7:
            self.scaler.scale_[6] = 200.0
            
        # Re-scale using the updated scale_
        X_scaled = (X - self.scaler.mean_) / self.scaler.scale_
        self.svm.fit(X_scaled)

        # Higher decision function output = more normal.
        # We negate the decision function so higher = more anomalous.
        raw_scores = -self.svm.decision_function(X_scaled)
        floor_t = 0.15 if X.shape[1] == 23 else 0.05
        self.t_anomaly = max(float(np.percentile(raw_scores, 95)), floor_t)
        
        self.enrollment_mean = np.mean(X, axis=0)
        self.enrollment_std = np.std(X, axis=0) + 1e-8
        self.is_trained = True

        return {
            "t_anomaly": self.t_anomaly,
            "n_windows": len(windows)
        }

    def score_window(self, window: np.ndarray) -> Dict[str, Any]:
        """
        Score a single 43-dim aggregate feature vector.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
            
        X = window.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        raw_decision = float(-self.svm.decision_function(X_scaled)[0])
        
        # Map to anomaly_score in [0, 1] using global calibration thresholds
        # to ensure scores are comparable across candidate models in identification tasks.
        calib_thresh = 0.20 if X.shape[1] == 7 else 0.30
        norm_score = raw_decision / (calib_thresh * 1.5)
        anomaly_score = float(np.clip(norm_score, 0.0, 1.0))
        
        # Verdict logic
        lo = self.t_anomaly * 0.6
        hi = self.t_anomaly * 1.4
        if raw_decision <= lo:
            verdict = "legitimate"
        elif raw_decision <= hi:
            verdict = "uncertain"
        else:
            verdict = "anomaly"

        return {
            "anomaly_score": anomaly_score,
            "raw_decision": raw_decision,
            "verdict": verdict,
            "threshold": self.t_anomaly
        }

    def score_session(self, window_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate window scores into a session verdict."""
        if not window_scores:
            return {"session_verdict": "unknown", "anomaly_rate": 0.0, "mean_score": 0.0}

        anomaly_rate = sum(1 for w in window_scores if w["verdict"] == "anomaly") / len(window_scores)
        mean_score = float(np.mean([w["anomaly_score"] for w in window_scores]))
        
        if anomaly_rate < 0.25:
            verdict = "legitimate"
        elif anomaly_rate > 0.60:
            verdict = "impostor"
        else:
            verdict = "uncertain"

        return {
            "session_verdict": verdict,
            "anomaly_rate": anomaly_rate,
            "mean_score": mean_score,
            "n_windows": len(window_scores)
        }

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "scaler": self.scaler,
            "svm": self.svm,
            "t_anomaly": self.t_anomaly,
            "enrollment_mean": self.enrollment_mean,
            "enrollment_std": self.enrollment_std,
            "is_trained": self.is_trained,
            "nu": self.nu,
            "kernel": self.kernel,
            "gamma": self.gamma
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str | Path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.scaler = state["scaler"]
        self.svm = state["svm"]
        self.t_anomaly = state["t_anomaly"]
        self.enrollment_mean = state["enrollment_mean"]
        self.enrollment_std = state["enrollment_std"]
        self.is_trained = state["is_trained"]
        self.nu = state.get("nu", 0.05)
        self.kernel = state.get("kernel", 'rbf')
        self.gamma = state.get("gamma", 'scale')
